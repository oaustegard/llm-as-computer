"""Packed (superposed) LAC core-12.

Same ISA, same dispatch tensors, same four oracle programs as
``experiments/blind_recovery/compile_artifact.py``. The one change: the residual
stream stops being axis-aligned. Each of the 24 semantic features gets a random
unit vector u_f in R^d and a row embeds as sum_f value_f * u_f, so weight
matrices become readouts instead of selectors.

Two readout arms:
  dot   w_f = u_f              (Elhage toy-models convention)
  pinv  W  = pinv(U.T)         (least-squares optimal; exact for d >= n_features)

With the identity codebook (d = 24, U = I) this file reproduces the axis-aligned
machine exactly -- that equivalence is the self-test in ``test_packed.py``.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'blind_recovery'))

from compile_artifact import (
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

# ---------------- semantic features ----------------
# Order matches the axis-aligned dim layout of compile_artifact (dims 0..23), so the
# identity codebook is that machine.
FEATURES = ['is_prog', 'is_stack', 'is_state', 'prog_k0', 'prog_k1', 'stack_k0', 'stack_k1',
            'opcode', 'value', 'ip', 'sp', 'one'] + [f'op_{o}' for o in OPS]
F = {name: i for i, name in enumerate(FEATURES)}
NF = len(FEATURES)

HEAD_ORDER = ['prog_op', 'prog_arg', 'stack_a', 'stack_b', 'stack_c', 'stack_addr']
# which pool each argmax head selects from, and, below, the address-verification
# tolerance. Both are read by learned_generic so that the trainer needs nothing about
# this machine beyond its module interface (packed_cat.py exposes the same two).
ARGMAX_HEADS = [('prog_op', 'rom'), ('prog_arg', 'rom'),
                ('stack_a', 'mem'), ('stack_b', 'mem'), ('stack_c', 'mem')]


def ADDR_SCALARS(addr):
    """Tolerance constraints for the address-verification read, as
    (feature, weight, target) triples. ``run`` recovers the address as
    ``stack_k0 / 2``, so the single scalar is constrained at weight 0.5."""
    return [('stack_k0', 0.5, 2.0 * addr)]


DENSE = ['is_prog', 'is_stack', 'is_state', 'prog_k0', 'prog_k1', 'stack_k0',
         'stack_k1', 'opcode', 'value', 'ip', 'sp', 'one']

# (rows of W_Q, rows of W_K, rows of W_V, b_Q) -- each row is a list of (feature, coeff)
HEAD_SPEC = {
    'prog_op':    ([[('ip', 1)], [('one', 1)]], [[('prog_k0', 1)], [('prog_k1', 1)]],
                   [[('opcode', 1)]], [0.0, 0.0]),
    'prog_arg':   ([[('ip', 1)], [('one', 1)]], [[('prog_k0', 1)], [('prog_k1', 1)]],
                   [[('value', 1)]], [0.0, 0.0]),
    'stack_a':    ([[('sp', 1)], [('one', 1)]], [[('stack_k0', 1)], [('stack_k1', 1)]],
                   [[('value', 1)]], [0.0, 0.0]),
    'stack_b':    ([[('sp', 1)], [('one', 1)]], [[('stack_k0', 1)], [('stack_k1', 1)]],
                   [[('value', 1)]], [-1.0, 0.0]),
    'stack_c':    ([[('sp', 1)], [('one', 1)]], [[('stack_k0', 1)], [('stack_k1', 1)]],
                   [[('value', 1)]], [-2.0, 0.0]),
    'stack_addr': ([[('sp', 1)], [('one', 1)]], [[('stack_k0', 1)], [('stack_k1', 1)]],
                   [[('stack_k0', 0.5)]], [0.0, 0.0]),
}


# ---------------- codebook + readout ----------------
def codebook(d, seed):
    """n_features random unit vectors in R^d, as an (NF, d) matrix."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((NF, d))
    return U / np.linalg.norm(U, axis=1, keepdims=True)


def identity_codebook():
    return np.eye(NF)


def workload_scales():
    """Per-feature typical magnitude over everything the exact machine encodes.

    RMS conditioned on the feature being ACTIVE, not RMS over all rows. The twelve
    opcode indicators are one-hot -- each fires on about a twelfth of the ROM and
    never anywhere else -- so their unconditional RMS is ~0.02, and dividing by that
    would stretch their directions ~50x and pour interference into every other
    readout. A sparse feature's typical magnitude is its magnitude when it is there.

    One constant per feature, calibrated once over all four oracle programs.
    """
    U = identity_codebook()
    H = build_heads(U)
    acc = np.zeros(NF)
    n = 0
    for name, prog, _ in programs():
        ys = [np.abs(embed_prog(U, p, o, a)) for p, (o, a) in enumerate(prog)]
        tr = []
        run(prog, U, U, H, trace=tr)
        for rec in tr:
            ys.append(np.abs(embed_state(U, rec['ip'], rec['sp'])))
            for addr, val, wo in rec.get('wrote', ()):
                ys.append(np.abs(embed_stack(U, addr, val, wo)))
        Y = np.array(ys)
        acc += (Y ** 2).sum(0)
        n = n + (np.abs(Y) > 0).sum(0)
    return np.maximum(np.sqrt(acc / np.maximum(n, 1)), 1e-9)


def readout(U, arm):
    if arm in ('dot', 'scaled'):
        # (s_f u_f) . (u_f / s_f) = 1, so the scaled code inverts the same way the
        # plain one does; only the interference changes.
        return U.copy() if arm == 'dot' else \
            U * (np.linalg.norm(U, axis=1, keepdims=True) ** -2)
    if arm == 'pinv':
        return np.linalg.pinv(U.T)
    raise ValueError(f'unknown arm {arm!r}')


def codebook_for(d, seed, arm, scales=None):
    """The code an arm uses. 'scaled' divides each direction by that feature's
    typical magnitude, so every feature contributes comparably to the residual norm
    -- feature count untouched, dynamic range flattened."""
    U = codebook(d, seed)
    if arm == 'scaled':
        U = U / (scales if scales is not None else workload_scales())[:, None]
    return U


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
    return embed(U, is_prog=1.0, prog_k0=2.0 * pos, prog_k1=-float(pos * pos),
                 opcode=float(OPNUM[opname]), value=float(arg), one=1.0,
                 **{f'op_{opname}': 1.0})


def embed_stack(U, addr, value, wo, eps=EPS):
    return embed(U, is_stack=1.0, stack_k0=2.0 * addr,
                 stack_k1=-float(addr * addr) + eps * wo, value=float(value), one=1.0)


def embed_state(U, ip, sp):
    return embed(U, is_state=1.0, ip=float(ip), sp=float(sp), one=1.0)


# ---------------- executor ----------------
def attend(W, q_emb, mem):
    if mem.shape[0] == 0:
        return 0.0, -1
    q = W['W_Q'] @ q_emb + W['b_Q']
    i = int(np.argmax((mem @ W['W_K'].T) @ q))
    return float((W['W_V'] @ mem[i])[0]), i


def run(prog, U, R, H, max_steps=4000, capture=None, trace=None, quantize=True,
        overwrite=False):
    """Execute a program in the packed basis.

    quantize: re-digitize every scalar read out of the residual stream (opcode,
    immediate, the three stack values) by rounding to the nearest integer. The
    axis-aligned machine gets exact integers for free and leans on that -- ``JZ``
    tests ``va == 0`` exactly. Under any superposition that test fails at machine
    epsilon, at every d, so without a quantizer there is no curve to measure. The
    quantizer is the analog-register-digital-machine assumption made explicit;
    ``quantize=False`` measures what it buys.

    trace: if a list, per-step dicts of (ip, sp, argmax per head, decoded op) are
    appended -- used both to build the reference trace and to compare against it.
    Returns (result, steps) or (None, steps) if it never halts.
    """
    dig = (lambda x: float(round(x))) if quantize else (lambda x: x)
    # overwrite: keep one row per address, so every score gap is >= 1 and the machine
    # never leans on the 1e-6 write-order tiebreak. PREDICTIONS.md registered this as
    # the fallback if the tiebreak turned out to be the binding constraint.
    eps = 0.0 if overwrite else EPS
    r_k0 = R[F['stack_k0']]
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
            got = round(float(r_k0 @ mem[i]) / 2.0)
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
            row = embed_stack(U, a, val, wo, eps=eps)
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
        ip = int(round(arg)) if ((jz and va == 0) or (jnz and va != 0)) else ip + 1
        sp = int(new_sp)
    return None, max_steps


# ---------------- oracle programs ----------------
def programs():
    """The four verified programs from compile_artifact, unchanged."""
    return [('sum_1_to_15', *sum_1_to_n(15)), ('countdown_5', *countdown(5)),
            ('rot_jz_nop', *rot_jz_nop()), ('sum_1_to_100', *sum_1_to_n(100))]


HEAD_OF_IDX = ['prog_op', 'prog_arg', 'stack_a', 'stack_b', 'stack_c']


def reference_traces():
    """Per-program reference traces from the exact (identity-codebook) machine."""
    U = identity_codebook()
    H = build_heads(U)
    out = {}
    for name, prog, expect in programs():
        tr = []
        got, steps = run(prog, U, U, H, trace=tr)
        assert abs(got - expect) < 1e-9, f'{name} reference broken'
        out[name] = tr
    return out


def first_divergence(trace, ref):
    """Where the packed run first parted company with the exact machine.

    Returns (step, kind) or None. Kinds:
      tiebreak:<head>  argmax flipped to another row at the SAME address -- the
                       EPS recency tiebreak lost (P3)
      argmax:<head>    argmax flipped to a different address (addressing broke)
      opcode_decode    right row, wrong integer after rounding the opcode readout
      value_drift      everything above intact, stored values wrong
    """
    for s, (t, r) in enumerate(zip(trace, ref)):
        if (t['ip'], t['sp'], t['n_rows']) != (r['ip'], r['sp'], r['n_rows']):
            return s, 'state'
        for k, h in enumerate(HEAD_OF_IDX):
            if t['idx'][k] != r['idx'][k]:
                same_addr = _same_address(h, t['idx'][k], r['idx'][k], r)
                return s, f'{"tiebreak" if same_addr else "argmax"}:{h}'
        if t['op'] != r['op']:
            return s, 'opcode_decode'
        if any(abs(a - b) > 1e-9 for a, b in zip(t['vals'], r['vals'])) or \
                abs(t['arg'] - r['arg']) > 1e-9:
            return s, 'value_drift'
    return None


def _same_address(head, i_got, i_ref, ref_step):
    """Both rows sit at the same address, so only the recency tiebreak separated them."""
    addrs = ref_step['addrs']
    if i_got < 0 or i_ref < 0 or max(i_got, i_ref) >= len(addrs):
        return False
    return addrs[i_got] == addrs[i_ref]


# ---------------- artifact for the blind analyst ----------------
def make_artifact(U, R, H, seed=0, ideal=False):
    """Anonymized tensors + activations, head order shuffled. No names, no source.

    ideal=False -- activations are what THIS packed machine actually produced. At
        low d that machine is malfunctioning, and its traces are traces of the
        malfunction, so recovery cannot outrun computation by construction.
    ideal=True -- activations are the reference machine's trajectory re-encoded in
        the packed basis: the states a working machine would hold. This is the
        artifact P1 is a claim about -- whether the ISA is still *readable* out of a
        compressed representation, separate from whether the machine still runs.
    """
    rng = np.random.default_rng(seed)
    out = {}
    horder = rng.permutation(len(HEAD_ORDER))
    for new_i, old_i in enumerate(horder):
        h = H[HEAD_ORDER[old_i]]
        out[f'head{new_i:02d}_Wq'] = h['W_Q']
        out[f'head{new_i:02d}_Wk'] = h['W_K']
        out[f'head{new_i:02d}_Wv'] = h['W_V']
        out[f'head{new_i:02d}_bq'] = h['b_Q']
    out['ffn_A'] = W_write.reshape(12, 3 * 4)
    out['ffn_B'] = np.stack([n_write, sp_delta]).T
    out['ffn_C'] = ctrl

    Uref = identity_codebook()
    Href = build_heads(Uref)
    progs = [sum_1_to_n(15)[0], countdown(5)[0], rot_jz_nop()[0]]
    for pi, prog in enumerate(progs):
        cap = {s_: None for s_ in (2, 9, 17)}
        # the captures sit at steps 2/9/17, so there is no reason to keep running a
        # machine that may never halt -- at low d most of them do not
        if ideal:
            run(prog, Uref, Uref, Href, capture=cap, max_steps=18)
        else:
            run(prog, U, R, H, capture=cap, max_steps=18)
        out[f'act_rom_{pi}'] = np.stack([embed_prog(U, p, o, a) for p, (o, a) in enumerate(prog)])
        for k, s_ in enumerate(sorted(cap)):
            if cap[s_] is None:
                continue
            if ideal:
                ip, sp = cap[s_]['sym_q']
                rows = cap[s_]['sym_stack']
                out[f'act_query_{pi}_{k}'] = embed_state(U, ip, sp)
                out[f'act_mem_{pi}_{k}'] = np.stack(
                    [embed_stack(U, a, v, w) for a, v, w in rows]) if rows \
                    else np.zeros((0, U.shape[1]))
            else:
                out[f'act_query_{pi}_{k}'] = cap[s_]['q']
                out[f'act_mem_{pi}_{k}'] = cap[s_]['stack']
    return out


TRUE_ROW_OF = {OPNUM[o]: IDX[o] for o in OPS}
