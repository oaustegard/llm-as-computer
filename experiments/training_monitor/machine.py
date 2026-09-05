"""The compiled LAC core-12 machine as a parameter dict, plus a hard-argmax executor
that runs from ANY parameter dict of the same shape -- compiled or trained.

Semantics are ``experiments/blind_recovery/compile_artifact.py`` verbatim (same dim
layout, same parabolic keys, same append-only stack with the 1e-6 recency tiebreak,
ADD/SUB/SWAP/ROT as linear write-routing rows, i32 wrap dropped). That file is the
oracle; ``test_machine.py`` checks this module against it on the four programs.

Two things the trained machine needs that the compiled one did not:

  * ``sp_delta`` and ``ctrl`` are floats after training. The executor rounds ``sp``
    to the nearest integer and thresholds ``ctrl`` at 0.5, which is the identity on
    the compiled values.
  * ``n_write`` (how many cells an opcode writes) is an integer count and is NOT a
    trained parameter. It stays at the compiled value in every arm.
"""
import numpy as np

D = 51
EPS = 1e-6

IS_PROG, IS_STACK, IS_STATE = 0, 1, 2
PROG_K0, PROG_K1 = 3, 4
STACK_K0, STACK_K1 = 5, 6
OPCODE, VALUE, IP, SP, ONE = 7, 8, 9, 10, 11
OPDIM = dict(PUSH=12, POP=13, ADD=14, DUP=15, HALT=16, SUB=17,
             JZ=18, JNZ=19, NOP=20, SWAP=21, OVER=22, ROT=23)
OPS = ['PUSH', 'POP', 'ADD', 'DUP', 'HALT', 'SUB', 'JZ', 'JNZ', 'NOP', 'SWAP', 'OVER', 'ROT']
OPNUM = {n: i + 1 for i, n in enumerate(OPS)}
IDX = {n: i for i, n in enumerate(OPS)}
N_OPS = 12
NU = 4                      # dispatch inputs u = [arg, va, vb, vc]
ARG, VA, VB, VC = 0, 1, 2, 3

HEAD_ORDER = ['prog_op', 'prog_arg', 'stack_a', 'stack_b', 'stack_c', 'stack_addr']
HEAD_REGION = dict(prog_op='rom', prog_arg='rom', stack_a='mem', stack_b='mem',
                   stack_c='mem', stack_addr='mem')
READ_HEADS = (('stack_a', 0), ('stack_b', -1), ('stack_c', -2))


def _onehot(rows):
    W = np.zeros((len(rows), D))
    for r, (d, v) in enumerate(rows):
        W[r, d] = v
    return W


def compile_params():
    """The compiled point in weight space. Every entry is a float64 array."""
    p = {}

    def head(name, q_rows, k_rows, v_rows, bq=None):
        p[f'{name}.W_Q'] = _onehot(q_rows)
        p[f'{name}.W_K'] = _onehot(k_rows)
        p[f'{name}.W_V'] = _onehot(v_rows)
        p[f'{name}.b_Q'] = np.array(bq if bq is not None else [0.0] * len(q_rows))

    head('prog_op', [(IP, 1), (ONE, 1)], [(PROG_K0, 1), (PROG_K1, 1)], [(OPCODE, 1)])
    head('prog_arg', [(IP, 1), (ONE, 1)], [(PROG_K0, 1), (PROG_K1, 1)], [(VALUE, 1)])
    head('stack_a', [(SP, 1), (ONE, 1)], [(STACK_K0, 1), (STACK_K1, 1)], [(VALUE, 1)])
    head('stack_b', [(SP, 1), (ONE, 1)], [(STACK_K0, 1), (STACK_K1, 1)], [(VALUE, 1)],
         bq=[-1.0, 0.0])
    head('stack_c', [(SP, 1), (ONE, 1)], [(STACK_K0, 1), (STACK_K1, 1)], [(VALUE, 1)],
         bq=[-2.0, 0.0])
    head('stack_addr', [(SP, 1), (ONE, 1)], [(STACK_K0, 1), (STACK_K1, 1)],
         [(STACK_K0, 0.5)])

    W_write = np.zeros((N_OPS, 3, NU))
    n_write = np.zeros(N_OPS)
    sp_delta = np.zeros(N_OPS)
    ctrl = np.zeros((N_OPS, 3))

    def dispatch(op, delta, writes, jz=0, jnz=0, halt=0):
        i = IDX[op]
        sp_delta[i] = delta
        n_write[i] = len(writes)
        for c, coeffs in enumerate(writes):
            for k, v in coeffs.items():
                W_write[i, c, k] = v
        ctrl[i] = (jz, jnz, halt)

    dispatch('PUSH', +1, [{ARG: 1}])
    dispatch('POP', -1, [{VB: 1}])
    dispatch('ADD', -1, [{VA: 1, VB: 1}])
    dispatch('DUP', +1, [{VA: 1}])
    dispatch('HALT', 0, [{VA: 1}], halt=1)
    dispatch('SUB', -1, [{VA: -1, VB: 1}])
    dispatch('JZ', -1, [{VB: 1}], jz=1)
    dispatch('JNZ', -1, [{VB: 1}], jnz=1)
    dispatch('NOP', 0, [{VA: 1}])
    dispatch('SWAP', 0, [{VB: 1}, {VA: 1}])
    dispatch('OVER', +1, [{VB: 1}])
    dispatch('ROT', 0, [{VC: 1}, {VA: 1}, {VB: 1}])

    p['W_write'] = W_write
    p['n_write'] = n_write
    p['sp_delta'] = sp_delta
    p['ctrl'] = ctrl
    return p


TRAINABLE = [f'{h}.{m}' for h in HEAD_ORDER for m in ('W_Q', 'W_K', 'W_V', 'b_Q')] + \
            ['W_write', 'sp_delta', 'ctrl']
HEAD_KEYS = [k for k in TRAINABLE if '.' in k]
ADDR_KEYS = [k for k in HEAD_KEYS if not k.endswith('W_V')]     # W_Q, W_K, b_Q
VALUE_KEYS = [k for k in HEAD_KEYS if k.endswith('W_V')]        # value readouts
DISPATCH_KEYS = ['W_write', 'sp_delta', 'ctrl']


# ---------------- embeddings (fixed functions, not parameters) ----------------
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


def rom_of(prog):
    return np.stack([embed_prog(i, o, a) for i, (o, a) in enumerate(prog)])


# ---------------- hard-argmax executor over a parameter dict ----------------
def attend(p, h, q_emb, mem):
    if mem.shape[0] == 0:
        return 0.0, -1
    q = p[f'{h}.W_Q'] @ q_emb + p[f'{h}.b_Q']
    s = (mem @ p[f'{h}.W_K'].T) @ q
    i = int(np.argmax(s))
    return float((p[f'{h}.W_V'] @ mem[i])[0]), i


def run(p, prog, max_steps=4000, trace=None, overwrite=False, quantize=False):
    """Execute ``prog`` on the machine ``p``. Returns (result, steps) or
    (None, steps) when the opcode read is out of range, the stack pointer goes
    negative, or the machine never halts -- a failure is None, never a number.

    ``trace``, if a list, receives one dict per step with everything a
    teacher-forced trainer needs: the query, the memory rows at that step, the
    integer opcode, and the reference values of every read and write.

    ``quantize=True`` rounds every scalar read out of the residual stream (the
    immediate and the three stack reads) to the nearest integer before dispatch,
    as the machine already does for the opcode and the address check. The
    compiled machine is an integer machine, and its conditional jumps test
    ``va == 0`` exactly, so without this any perturbation of a value readout is
    fatal and there is no curve to measure. Same decision as the packed
    executor of ``experiments/superposition``.

    ``overwrite=True`` keeps one row per address (overwrite-in-place) instead of
    the append-only log with the 1e-6 recency tiebreak. Same results on every
    oracle program; the training arms use it because a softmax cannot resolve a
    1e-6 tiebreak (see ``experiments/superposition/RESULTS-A.md``).
    """
    rom = rom_of(prog)
    rows, wo, sp, ip = [], 0, 0, 0
    for step in range(max_steps):
        q = embed_state(ip, sp)
        mem = np.stack(rows) if rows else np.zeros((0, D))
        opv, _ = attend(p, 'prog_op', q, rom)
        arg, _ = attend(p, 'prog_arg', q, rom)
        if not (np.isfinite(opv) and np.isfinite(arg)):
            return None, step
        op = int(round(opv))
        if not 1 <= op <= N_OPS or not 0 <= ip < len(prog):
            return None, step
        vals, hits = [], []
        for h, off in READ_HEADS:
            v, i = attend(p, h, q, mem)
            if i < 0:
                vals.append(0.0)
                hits.append(0.0)
                continue
            got = round(mem[i, STACK_K0] / 2.0)
            hit = got == sp + off
            vals.append(v if hit else 0.0)
            hits.append(float(hit))
        if not np.all(np.isfinite(vals)):
            return None, step
        if quantize:
            arg = float(round(arg))
            vals = [float(round(v)) for v in vals]
        va, vb, vc = vals
        u = np.array([arg, va, vb, vc])
        r = op - 1
        if not np.isfinite(p['sp_delta'][r]):
            return None, step
        new_sp = int(round(sp + p['sp_delta'][r]))
        if new_sp < 0:
            return None, step
        writes = [float(p['W_write'][r, c] @ u) for c in range(int(p['n_write'][r]))]
        if not np.all(np.isfinite(writes)) or max(abs(w) for w in writes) > 1e12:
            return None, step
        jz, jnz, halt = (p['ctrl'][r] > 0.5)
        if trace is not None:
            trace.append(dict(q=q, mem=mem.copy(), op=r, arg=arg, vals=np.array(vals),
                              hits=np.array(hits), writes=np.array(writes),
                              new_sp=new_sp, ip=ip, sp=sp))
        for c, w in enumerate(writes):
            if overwrite:
                addr = new_sp - c
                for k, row in enumerate(rows):
                    if round(row[STACK_K0] / 2.0) == addr:
                        rows[k] = embed_stack(addr, w, 0)
                        break
                else:
                    rows.append(embed_stack(addr, w, 0))
            else:
                rows.append(embed_stack(new_sp - c, w, wo))
                wo += 1
        if halt:
            return va, step + 1
        if (jz and va == 0) or (jnz and va != 0):
            ip = int(round(arg))
        else:
            ip += 1
        sp = new_sp
    return None, max_steps


# ---------------- the four oracle programs (unchanged from blind_recovery) ----
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


ORACLE = [('sum_1_to_15', *sum_1_to_n(15)), ('countdown_5', *countdown(5)),
          ('rot_jz_nop', *rot_jz_nop()), ('sum_1_to_100', *sum_1_to_n(100))]


def oracle_score(p, overwrite=False, quantize=False, max_steps=1000):
    """Fraction of the four oracle programs returning their exact expected value,
    plus the per-program verdicts. ``max_steps`` is above the longest oracle run
    (sum_1_to_100 halts at 803) and caps the cost of a broken machine."""
    verdicts = {}
    for name, prog, expect in ORACLE:
        got, _ = run(p, prog, max_steps=max_steps, overwrite=overwrite, quantize=quantize)
        verdicts[name] = bool(got is not None and abs(got - expect) < 1e-9)
    return sum(verdicts.values()) / len(verdicts), verdicts


if __name__ == '__main__':
    p = compile_params()
    for ow in (False, True):
        for qz in (False, True):
            frac, v = oracle_score(p, overwrite=ow, quantize=qz)
            for k, ok in v.items():
                print(f'{"overwrite" if ow else "append":9s} {"quant" if qz else "raw":5s} '
                      f'{k:14s} {"OK" if ok else "FAIL"}')
            assert frac == 1.0
