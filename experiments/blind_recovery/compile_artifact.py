"""Compile the LAC core-12 ISA to explicit weight tensors, run programs to verify,
then emit an ANONYMIZED artifact (permuted d_model basis, no names) for blind recovery.

Dim layout and parabolic key scheme are taken verbatim from oaustegard/llm-as-computer
(isa.py / executor.py). Deviation from upstream, flagged: ADD/SUB/SWAP/ROT are expressed
as linear write-routing matrices instead of being special-cased in forward(), so that the
whole dispatch lives in tensors. i32 wrap masking is dropped (test programs stay in range).
"""
import numpy as np

D = 51
EPS = 1e-6

# --- upstream dim indices (isa.py) ---
IS_PROG, IS_STACK, IS_STATE = 0, 1, 2
PROG_K0, PROG_K1 = 3, 4
STACK_K0, STACK_K1 = 5, 6
OPCODE, VALUE, IP, SP, ONE = 7, 8, 9, 10, 11
# one-hot opcode dims 12..23 for the core 12
OPDIM = dict(PUSH=12, POP=13, ADD=14, DUP=15, HALT=16, SUB=17,
             JZ=18, JNZ=19, NOP=20, SWAP=21, OVER=22, ROT=23)
OPS = ['PUSH', 'POP', 'ADD', 'DUP', 'HALT', 'SUB', 'JZ', 'JNZ', 'NOP', 'SWAP', 'OVER', 'ROT']
OPNUM = {n: i + 1 for i, n in enumerate(OPS)}   # upstream opcode integers 1..12
IDX = {n: i for i, n in enumerate(OPS)}          # row index in dispatch matrices

# ---------------- weights ----------------
def onehot(rows):
    W = np.zeros((len(rows), D))
    for r, (d, v) in enumerate(rows):
        W[r, d] = v
    return W

HEADS = {}
def head(name, q_rows, k_rows, v_rows, bq=None):
    HEADS[name] = dict(W_Q=onehot(q_rows), W_K=onehot(k_rows), W_V=onehot(v_rows),
                       b_Q=np.array(bq if bq is not None else [0.0] * len(q_rows)))

head('prog_op',  [(IP, 1), (ONE, 1)], [(PROG_K0, 1), (PROG_K1, 1)], [(OPCODE, 1)])
head('prog_arg', [(IP, 1), (ONE, 1)], [(PROG_K0, 1), (PROG_K1, 1)], [(VALUE, 1)])
head('stack_a',  [(SP, 1), (ONE, 1)], [(STACK_K0, 1), (STACK_K1, 1)], [(VALUE, 1)])
head('stack_b',  [(SP, 1), (ONE, 1)], [(STACK_K0, 1), (STACK_K1, 1)], [(VALUE, 1)], bq=[-1.0, 0.0])
head('stack_c',  [(SP, 1), (ONE, 1)], [(STACK_K0, 1), (STACK_K1, 1)], [(VALUE, 1)], bq=[-2.0, 0.0])
# address-verify head: recovers addr = key0/2 of the winning stack row
head('stack_addr', [(SP, 1), (ONE, 1)], [(STACK_K0, 1), (STACK_K1, 1)], [(STACK_K0, 0.5)])

HEAD_ORDER = ['prog_op', 'prog_arg', 'stack_a', 'stack_b', 'stack_c', 'stack_addr']

# dispatch: inputs u = [arg, va, vb, vc]  (va=stack[SP], vb=stack[SP-1], vc=stack[SP-2])
NU = 4
W_write = np.zeros((12, 3, NU))    # [op, cell_offset_below_new_top, input]
n_write = np.zeros(12)             # how many cells the op writes
sp_delta = np.zeros(12)
ctrl = np.zeros((12, 3))           # (jump_if_zero_va, jump_if_nonzero_va, halt)

def dispatch(op, delta, writes, jz=0, jnz=0, halt=0):
    i = IDX[op]
    sp_delta[i] = delta
    n_write[i] = len(writes)
    for c, coeffs in enumerate(writes):
        for k, v in coeffs.items():
            W_write[i, c, k] = v
    ctrl[i] = (jz, jnz, halt)

ARG, VA, VB, VC = 0, 1, 2, 3
dispatch('PUSH', +1, [{ARG: 1}])
dispatch('POP',  -1, [{VB: 1}])
dispatch('ADD',  -1, [{VA: 1, VB: 1}])
dispatch('DUP',  +1, [{VA: 1}])
dispatch('HALT',  0, [{VA: 1}], halt=1)
dispatch('SUB',  -1, [{VA: -1, VB: 1}])
dispatch('JZ',   -1, [{VB: 1}], jz=1)
dispatch('JNZ',  -1, [{VB: 1}], jnz=1)
dispatch('NOP',   0, [{VA: 1}])
dispatch('SWAP',  0, [{VB: 1}, {VA: 1}])
dispatch('OVER', +1, [{VB: 1}])
dispatch('ROT',   0, [{VC: 1}, {VA: 1}, {VB: 1}])

# ---------------- embeddings ----------------
def embed_prog(pos, opname, arg):
    e = np.zeros(D)
    e[IS_PROG] = 1.0
    e[PROG_K0] = 2.0 * pos
    e[PROG_K1] = -float(pos * pos)
    e[OPCODE] = float(OPNUM[opname])
    e[VALUE] = float(arg)
    e[ONE] = 1.0
    e[OPDIM[opname]] = 1.0
    return e

def embed_stack(addr, value, wo):
    e = np.zeros(D)
    e[IS_STACK] = 1.0
    e[STACK_K0] = 2.0 * addr
    e[STACK_K1] = -float(addr * addr) + EPS * wo
    e[VALUE] = float(value)
    e[ONE] = 1.0
    return e

def embed_state(ip, sp):
    e = np.zeros(D)
    e[IS_STATE] = 1.0
    e[IP] = float(ip)
    e[SP] = float(sp)
    e[ONE] = 1.0
    return e

# ---------------- executor (weights only) ----------------
def attend(h, q_emb, mem):
    W = HEADS[h]
    if mem.shape[0] == 0:
        return 0.0, -1
    q = W['W_Q'] @ q_emb + W['b_Q']
    K = mem @ W['W_K'].T
    s = K @ q
    i = int(np.argmax(s))
    return float((W['W_V'] @ mem[i])[0]), i

def run(prog, max_steps=4000, capture=None):
    prog_embs = np.stack([embed_prog(p, o, a) for p, (o, a) in enumerate(prog)])
    stack_rows, wo, sp, ip = [], 0, 0, 0
    for step in range(max_steps):
        q = embed_state(ip, sp)
        mem = np.stack(stack_rows) if stack_rows else np.zeros((0, D))
        opv, _ = attend('prog_op', q, prog_embs)
        arg, _ = attend('prog_arg', q, prog_embs)
        op = int(round(opv))
        vals = []
        for h, off in (('stack_a', 0), ('stack_b', -1), ('stack_c', -2)):
            v, i = attend(h, q, mem)
            if i < 0:
                vals.append(0.0); continue
            got = round(mem[i, STACK_K0] / 2.0)
            vals.append(v if got == sp + off else 0.0)
        va, vb, vc = vals
        u = np.array([arg, va, vb, vc])
        r = op - 1
        if capture is not None and step in capture:
            capture[step] = dict(q=q, stack=mem.copy())
        new_sp = sp + sp_delta[r]
        for c in range(int(n_write[r])):
            stack_rows.append(embed_stack(int(new_sp) - c, float(W_write[r, c] @ u), wo)); wo += 1
        jz, jnz, halt = ctrl[r]
        if halt:
            return va, step + 1, prog_embs
        if (jz and va == 0) or (jnz and va != 0):
            ip = int(round(arg))
        else:
            ip += 1
        sp = int(new_sp)
    raise RuntimeError('no halt')

# ---------------- test programs ----------------
def sum_1_to_n(n):
    return [('PUSH', n), ('PUSH', 0),
            ('OVER', 0), ('ADD', 0), ('SWAP', 0), ('PUSH', 1), ('SUB', 0),
            ('SWAP', 0), ('OVER', 0), ('JNZ', 2), ('HALT', 0)], n * (n + 1) // 2

def countdown(n):
    return [('PUSH', n), ('PUSH', 1), ('SUB', 0), ('DUP', 0), ('JNZ', 1), ('HALT', 0)], 0

def rot_jz_nop():
    return [('PUSH', 1), ('PUSH', 2), ('PUSH', 3), ('ROT', 0), ('NOP', 0),
            ('PUSH', 7), ('POP', 0), ('PUSH', 0), ('JZ', 10), ('PUSH', 99),
            ('HALT', 0)], 1

if __name__ == '__main__':
    cases = [('sum_1_to_15', *sum_1_to_n(15)), ('countdown_5', *countdown(5)),
             ('rot_jz_nop', *rot_jz_nop()), ('sum_1_to_100', *sum_1_to_n(100))]
    ok = True
    for name, prog, expect in cases:
        got, steps, _ = run(prog)
        good = abs(got - expect) < 1e-9
        ok &= good
        print(f'{name:14s} got={got:<10g} expect={expect:<10g} steps={steps} {"OK" if good else "FAIL"}')
    assert ok

    # ---- anonymize: random permutation of the d_model axis ----
    rng = np.random.default_rng(20260811)
    perm = rng.permutation(D)
    P = lambda M: M[..., perm]

    out = {}
    horder = rng.permutation(len(HEAD_ORDER))
    for new_i, old_i in enumerate(horder):
        h = HEADS[HEAD_ORDER[old_i]]
        out[f'head{new_i:02d}_Wq'] = P(h['W_Q'])
        out[f'head{new_i:02d}_Wk'] = P(h['W_K'])
        out[f'head{new_i:02d}_Wv'] = P(h['W_V'])
        out[f'head{new_i:02d}_bq'] = h['b_Q']
    out['ffn_A'] = W_write.reshape(12, 3 * NU)
    out['ffn_B'] = np.stack([n_write, sp_delta]).T
    out['ffn_C'] = ctrl

    progs = [sum_1_to_n(15)[0], countdown(5)[0], rot_jz_nop()[0]]
    for pi, prog in enumerate(progs):
        cap = {s_: None for s_ in (2, 9, 17)}
        run(prog, capture=cap)
        out[f'act_rom_{pi}'] = P(np.stack([embed_prog(p, o, a) for p, (o, a) in enumerate(prog)]))
        for k, s_ in enumerate(sorted(cap)):
            if cap[s_] is None:
                continue
            out[f'act_query_{pi}_{k}'] = P(cap[s_]['q'])
            out[f'act_mem_{pi}_{k}'] = P(cap[s_]['stack'])
    np.savez('/home/claude/exp/artifact.npz', **out)
    np.savez('/home/claude/exp/ground_truth.npz', perm=perm,
             head_order=np.array(horder), ops=np.array(OPS),
             sp_delta=sp_delta, n_write=n_write, W_write=W_write, ctrl=ctrl,
             opdims=np.array([OPDIM[o] for o in OPS]),
             head_names=np.array(HEAD_ORDER))
    print('artifact written:', len(out), 'tensors, d_model axis permuted')
