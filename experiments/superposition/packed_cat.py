"""Categorical-address LAC core-12: the same machine, addressed by one-hot equality.

``packed.py`` selects a memory row by a numeric dot product over parabolic keys --
row ``j`` carries ``(2j, -j^2)``, a query for address ``x`` carries ``(x, 1)``, and the
score ``2jx - j^2 = -(j - x)^2 + x^2`` peaks at ``j = x`` with a winner-to-runner-up
gap of exactly 1. ALTA's compiled programs instead select by one-hot equality, and the
addressing is the last untested candidate for why LAC's fitted code gains one dimension
where ALTA's gain 20-45% (``RESULTS-M.md``, "Remaining candidate").

This module transplants ALTA's select into LAC and changes nothing else. Same twelve
opcodes, same dispatch tensors, same four oracle programs, same overwrite-in-place
stack, same value/opcode/indicator features. Only the keys and queries change:

  ROM keys      ``pos_0 .. pos_{A_ROM-1}``      one-hot on the instruction address
  stack keys    ``addr_0 .. addr_{A_ST-1}``     one-hot on the stack address
  ip query      ``ipq_0 .. ipq_{A_ROM-1}``      one-hot on ip
  sp queries    ``spq0_* / spq1_* / spq2_*``    one-hot on sp, sp-1, sp-2

The three sp query groups replace the three ``b_Q`` offsets (0, -1, -2) of the
parabolic machine: an offset is a shift of a scalar, and there is no scalar left to
shift. An address below 0 lights no bit, so every score in the pool is 0 and the read
is discarded by the address check -- the same outcome the parabolic machine reaches by
attending to the nearest live row and failing the check.

The score is a one-hot dot product, so the winner scores 1 and every other row scores
0: the winner-to-runner-up gap is 1, exactly as in ``packed.py``. What differs is
where that gap comes from -- 5 one-hot bits rather than 2 numeric ones.

With the identity codebook this file executes the same traces as ``packed.py``, step
for step; ``test_addressing.py`` is that equivalence.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'blind_recovery'))

from compile_artifact import (  # noqa: E402
    EPS,
    IDX,
    OPNUM,
    OPS,
    W_write,
    countdown,
    ctrl,
    n_write,
    rot_jz_nop,
    sp_delta,
    sum_1_to_n,
)

# ---------------- address space ----------------
# A_ROM: the longest ROM among the oracle programs (sum_1_to_n and rot_jz_nop are 11).
# A_ST : one past the largest stack address any of the four programs reaches (4).
# Both are measured from the reference traces, not guessed -- ``test_addressing.py``
# re-measures them and fails if a program outgrows either.
A_ROM = 11
A_ST = 5

# ---------------- semantic features ----------------
POS = [f'pos_{j}' for j in range(A_ROM)]
ADDR = [f'addr_{a}' for a in range(A_ST)]
IPQ = [f'ipq_{j}' for j in range(A_ROM)]
SPQ = [[f'spq{g}_{a}' for a in range(A_ST)] for g in range(3)]

FEATURES = (['is_prog', 'is_stack', 'is_state'] + POS + ADDR + IPQ + SPQ[0] + SPQ[1]
            + SPQ[2] + ['opcode', 'value', 'one'] + [f'op_{o}' for o in OPS])
F = {name: i for i, name in enumerate(FEATURES)}
NF = len(FEATURES)

# every feature the machine actually uses: everything but the twelve opcode indicators,
# which nothing in the compiled machine reads (RESULTS-A.md, A2).
DENSE = [f for f in FEATURES if not f.startswith('op_')]

HEAD_ORDER = ['prog_op', 'prog_arg', 'stack_a', 'stack_b', 'stack_c', 'stack_addr']
# which pool each argmax head selects from; consumed by learned_generic.harvest
ARGMAX_HEADS = [('prog_op', 'rom'), ('prog_arg', 'rom'),
                ('stack_a', 'mem'), ('stack_b', 'mem'), ('stack_c', 'mem')]


def _onehot_rows(names):
    return [[(n, 1.0)] for n in names]


# (rows of W_Q, rows of W_K, rows of W_V, b_Q). One Q/K row pair per one-hot bit, so
# score = sum_k q_k k_k = 1 on the matching row and 0 elsewhere.
_ROM_Q, _ROM_K = _onehot_rows(IPQ), _onehot_rows(POS)
_ADDR_V = [[(ADDR[a], float(a)) for a in range(A_ST)]]   # addr = sum_a a * addr_a

HEAD_SPEC = {
    'prog_op':    (_ROM_Q, _ROM_K, [[('opcode', 1)]], [0.0] * A_ROM),
    'prog_arg':   (_ROM_Q, _ROM_K, [[('value', 1)]], [0.0] * A_ROM),
    'stack_a':    (_onehot_rows(SPQ[0]), _onehot_rows(ADDR), [[('value', 1)]], [0.0] * A_ST),
    'stack_b':    (_onehot_rows(SPQ[1]), _onehot_rows(ADDR), [[('value', 1)]], [0.0] * A_ST),
    'stack_c':    (_onehot_rows(SPQ[2]), _onehot_rows(ADDR), [[('value', 1)]], [0.0] * A_ST),
    'stack_addr': (_onehot_rows(SPQ[0]), _onehot_rows(ADDR), _ADDR_V, [0.0] * A_ST),
}


def ADDR_SCALARS(addr):
    """Tolerance constraints for the address-verification read, as
    (feature, weight, target) triples.

    The parabolic machine checks one scalar (``stack_k0 / 2``); this one checks a
    one-hot group, so every bit of it is constrained to its true value at the same
    absolute tolerance. Consumed by ``learned_generic.harvest``.
    """
    return [(ADDR[a], 1.0, 1.0 if a == addr else 0.0) for a in range(A_ST)]


# ---------------- codebook + readout ----------------
def codebook(d, seed):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((NF, d))
    return U / np.linalg.norm(U, axis=1, keepdims=True)


def identity_codebook():
    return np.eye(NF)


def readout(U, arm):
    if arm == 'dot':
        return U.copy()
    if arm == 'pinv':
        return np.linalg.pinv(U.T)
    raise ValueError(f'unknown arm {arm!r}')


def build_heads(R):
    """Weight tensors over a readout matrix R (NF, d)."""
    def rows(spec):
        return np.stack([sum(c * R[F[f]] for f, c in row) for row in spec])

    H = {}
    for name, (q, k, v, bq) in HEAD_SPEC.items():
        H[name] = dict(W_Q=rows(q), W_K=rows(k), W_V=rows(v), b_Q=np.array(bq))
    return H


# ---------------- embeddings ----------------
def embed(U, **vals):
    y = np.zeros(NF)
    for k, v in vals.items():
        y[F[k]] = v
    return y @ U


def embed_prog(U, pos, opname, arg):
    if not 0 <= pos < A_ROM:
        raise ValueError(f'ROM address {pos} outside the one-hot span A_ROM={A_ROM}')
    return embed(U, is_prog=1.0, opcode=float(OPNUM[opname]), value=float(arg),
                 one=1.0, **{POS[pos]: 1.0, f'op_{opname}': 1.0})


def embed_stack(U, addr, value, wo, eps=EPS):
    """``wo``/``eps`` are accepted for interface parity and ignored: the write-order
    tiebreak is a perturbation of a numeric key, and there is no numeric key here.
    Only the overwrite-in-place stack -- one row per address -- is meaningful."""
    if not 0 <= addr < A_ST:
        raise ValueError(f'stack address {addr} outside the one-hot span A_ST={A_ST}')
    return embed(U, is_stack=1.0, value=float(value), one=1.0, **{ADDR[addr]: 1.0})


def embed_state(U, ip, sp):
    """One-hot ip, plus one-hot sp / sp-1 / sp-2. An address outside the span lights
    no bit, so the head scores every row 0 and the read fails its address check."""
    bits = {}
    if 0 <= ip < A_ROM:
        bits[IPQ[int(ip)]] = 1.0
    for g, off in enumerate((0, -1, -2)):
        a = int(sp) + off
        if 0 <= a < A_ST:
            bits[SPQ[g][a]] = 1.0
    return embed(U, is_state=1.0, one=1.0, **bits)


# ---------------- executor ----------------
def attend(W, q_emb, mem):
    """Argmax with recency tie-breaking.

    One-hot scores tie exactly whenever two rows share an address, where the
    parabolic machine's 1e-6 write-order term made the newest row win. Breaking ties
    toward the last row reproduces that, so the append-only stack traces match too.
    Under ``overwrite=True`` -- the mode everything here is trained and measured in --
    addresses are unique and no tie arises.
    """
    if mem.shape[0] == 0:
        return 0.0, -1
    q = W['W_Q'] @ q_emb + W['b_Q']
    s = (mem @ W['W_K'].T) @ q
    i = int(len(s) - 1 - np.argmax(s[::-1]))
    return float((W['W_V'] @ mem[i])[0]), i


def run(prog, U, R, H, max_steps=4000, capture=None, trace=None, quantize=True,
        overwrite=False):
    """Execute a program in the packed basis. Same contract as ``packed.run``."""
    dig = (lambda x: float(round(x))) if quantize else (lambda x: x)
    r_addr = sum(float(a) * R[F[ADDR[a]]] for a in range(A_ST))
    prog_embs = np.stack([embed_prog(U, p, o, a) for p, (o, a) in enumerate(prog)])
    stack_rows, addrs, sym_rows, wo, sp, ip = [], [], [], 0, 0, 0
    for step in range(max_steps):
        q = embed_state(U, ip, sp)
        mem = np.stack(stack_rows) if stack_rows else np.zeros((0, U.shape[1]))
        opv, i_op = attend(H['prog_op'], q, prog_embs)
        arg_raw, i_arg = attend(H['prog_arg'], q, prog_embs)
        arg = dig(arg_raw)
        op = int(round(opv))
        vals, idxs = [], []
        for h, off in (('stack_a', 0), ('stack_b', -1), ('stack_c', -2)):
            v, i = attend(H[h], q, mem)
            idxs.append(i)
            if i < 0:
                vals.append(0.0)
                continue
            got = round(float(r_addr @ mem[i]))
            vals.append(dig(v) if got == sp + off else 0.0)
        if trace is not None:
            rec = dict(ip=ip, sp=sp, n_rows=len(stack_rows), op=op, arg=arg,
                       idx=(i_op, i_arg, *idxs), vals=tuple(vals), addrs=list(addrs))
            trace.append(rec)
        if capture is not None and step in capture:
            capture[step] = dict(q=q, stack=mem.copy(), sym_q=(ip, sp),
                                 sym_stack=list(sym_rows))
        if not 1 <= op <= 12:
            return None, step + 1
        va, vb, vc = vals
        u = np.array([arg, va, vb, vc])
        r = op - 1
        new_sp = sp + sp_delta[r]
        for c in range(int(n_write[r])):
            val = float(W_write[r, c] @ u)
            a = int(new_sp) - c
            if not 0 <= a < A_ST:
                return None, step + 1
            row = embed_stack(U, a, val, wo)
            if overwrite and a in addrs:
                i = addrs.index(a)
                stack_rows[i], sym_rows[i] = row, (a, val, wo)
            else:
                stack_rows.append(row)
                addrs.append(a)
                sym_rows.append((a, val, wo))
            if trace is not None:
                rec.setdefault('wrote', []).append((a, val, wo))
            wo += 1
        jz, jnz, halt = ctrl[r]
        if halt:
            return va, step + 1
        nip = int(round(arg)) if ((jz and va == 0) or (jnz and va != 0)) else ip + 1
        if not 0 <= nip < A_ROM:
            return None, step + 1
        ip, sp = nip, int(new_sp)
    return None, max_steps


# ---------------- oracle programs ----------------
def programs():
    return [('sum_1_to_15', *sum_1_to_n(15)), ('countdown_5', *countdown(5)),
            ('rot_jz_nop', *rot_jz_nop()), ('sum_1_to_100', *sum_1_to_n(100))]


HEAD_OF_IDX = ['prog_op', 'prog_arg', 'stack_a', 'stack_b', 'stack_c']


def reference_traces(overwrite=False):
    U = identity_codebook()
    H = build_heads(U)
    out = {}
    for name, prog, expect in programs():
        tr = []
        got, _ = run(prog, U, U, H, trace=tr, overwrite=overwrite)
        assert got is not None and abs(got - expect) < 1e-9, f'{name} reference broken'
        out[name] = tr
    return out


def first_divergence(trace, ref):
    """Same contract as ``packed.first_divergence``. There is no ``tiebreak`` kind
    under one-hot equality with a unique address, so a flipped argmax is always
    ``argmax:<head>``."""
    for s, (t, r) in enumerate(zip(trace, ref)):
        if (t['ip'], t['sp'], t['n_rows']) != (r['ip'], r['sp'], r['n_rows']):
            return s, 'state'
        for k, h in enumerate(HEAD_OF_IDX):
            if t['idx'][k] != r['idx'][k]:
                same = _same_address(t['idx'][k], r['idx'][k], r)
                return s, f'{"tiebreak" if same else "argmax"}:{h}'
        if t['op'] != r['op']:
            return s, 'opcode_decode'
        if any(abs(a - b) > 1e-9 for a, b in zip(t['vals'], r['vals'])) or \
                abs(t['arg'] - r['arg']) > 1e-9:
            return s, 'value_drift'
    return None


def _same_address(i_got, i_ref, ref_step):
    addrs = ref_step['addrs']
    if i_got < 0 or i_ref < 0 or max(i_got, i_ref) >= len(addrs):
        return False
    return addrs[i_got] == addrs[i_ref]


TRUE_ROW_OF = {OPNUM[o]: IDX[o] for o in OPS}
