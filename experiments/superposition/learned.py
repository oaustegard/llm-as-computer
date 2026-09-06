"""Fit the residual-stream code by gradient descent instead of drawing it at random.

Follows Tracr §5's setup: every weight in the machine stays frozen and only the
embedding matrix trains, with the readout tied to it (``R = U``, Tracr's shared-W
convention -- the ``dot`` arm with a learned ``U``).

The objective is teacher-forced, so nothing is unrolled. The reference machine's own
trajectory says which decisions it depends on, and the code is asked to keep making
them:

  margin      every head picks its winner by hard argmax, so for each (step, head)
              hinge on  score(winner) - score(other) >= MARGIN  against the hardest
              competitors
  tolerance   every scalar the machine rounds must land within TOL of its true value

Both are hinges at a threshold rather than losses driven to zero, and that is the
whole design. An MSE on the readouts would pour capacity into ``value``, whose
magnitudes reach 5050, at the expense of features whose tolerance is equally absolute
but whose magnitudes are 1 -- which is exactly the failure the post-hoc ``scaled`` arm
demonstrated in RESULTS.md.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import packed as P

MARGIN = 0.5      # half the reference parabolic gap of 1
TOL = 0.25        # half the quantizer's rounding tolerance
N_NEG = 4         # hardest competitors kept per (step, head)

# heads whose argmax the machine's control flow depends on, with the query offset
ARGMAX_HEADS = [('prog_op', 'rom', 0.0), ('prog_arg', 'rom', 0.0),
                ('stack_a', 'mem', 0.0), ('stack_b', 'mem', -1.0), ('stack_c', 'mem', -2.0)]


def _y_prog(pos, opname, arg):
    y = np.zeros(P.NF)
    y[P.F['is_prog']] = 1.0
    y[P.F['prog_k0']] = 2.0 * pos
    y[P.F['prog_k1']] = -float(pos * pos)
    y[P.F['opcode']] = float(P.OPNUM[opname])
    y[P.F['value']] = float(arg)
    y[P.F['one']] = 1.0
    y[P.F[f'op_{opname}']] = 1.0
    return y


def _y_stack(addr, value, wo, eps=0.0):
    y = np.zeros(P.NF)
    y[P.F['is_stack']] = 1.0
    y[P.F['stack_k0']] = 2.0 * addr
    y[P.F['stack_k1']] = -float(addr * addr) + eps * wo
    y[P.F['value']] = float(value)
    y[P.F['one']] = 1.0
    return y


def _y_state(ip, sp):
    y = np.zeros(P.NF)
    y[P.F['is_state']] = 1.0
    y[P.F['ip']] = float(ip)
    y[P.F['sp']] = float(sp)
    y[P.F['one']] = 1.0
    return y


def harvest(programs=None, overwrite=True):
    """Every decision the reference machine makes, as feature-value vectors.

    Runs the overwrite-in-place stack by default. On the append-only stack, 11,118 of
    the 18,995 margin constraints are same-address pairs separated only by the 1e-6
    write-order tiebreak, and no weighting of them works: at their true target they
    contribute no gradient and the optimizer inverts 97% of them, while normalized to
    matter they carry a 1e6 weight and own the objective. That is P3 showing up as an
    optimization fact -- the tiebreak is not compressible -- and PREDICTIONS.md
    registered switching the stack as the response.

    Returns (queries, winners, competitors, head indices, scalars, target gaps).
    Nothing here is an embedding; the code being trained has not been chosen yet.
    """
    U = P.identity_codebook()
    H = P.build_heads(U)
    progs = programs if programs is not None else P.programs()

    q_rows, win_rows, neg_rows, head_ix, gaps = [], [], [], [], []
    scalars = []          # (feature-value vector, feature name, true value)
    for name, prog, _ in progs:
        rom = [_y_prog(p, o, a) for p, (o, a) in enumerate(prog)]
        for y in rom:
            scalars.append((y, 'opcode', y[P.F['opcode']]))
            scalars.append((y, 'value', y[P.F['value']]))
        tr = []
        P.run(prog, U, U, H, trace=tr, overwrite=overwrite)
        stack, saddr = [], []
        for rec in tr:
            q = _y_state(rec['ip'], rec['sp'])
            for hi, (h, region, off) in enumerate(ARGMAX_HEADS):
                pool = rom if region == 'rom' else stack
                if len(pool) < 2:
                    continue
                target = rec['ip'] if region == 'rom' else rec['sp'] + off
                k0, k1 = ('prog_k0', 'prog_k1') if region == 'rom' else ('stack_k0', 'stack_k1')
                s = np.array([y[P.F[k0]] * target + y[P.F[k1]] for y in pool])
                order = np.argsort(-s)
                w = order[0]
                for c in order[1:1 + N_NEG]:
                    q_rows.append(q)
                    win_rows.append(pool[w])
                    neg_rows.append(pool[c])
                    head_ix.append(hi)
                    # ask the code to preserve the gap the reference machine had,
                    # capped at MARGIN. Two rows at the same address are separated
                    # only by the 1e-6 write-order tiebreak, so demanding 0.5 there
                    # would be demanding something the uncompressed machine never had.
                    gaps.append(min(MARGIN, float(s[w] - s[c])))
            for addr, val, wo in rec.get('wrote', ()):
                y = _y_stack(addr, val, wo, eps=0.0 if overwrite else P.EPS)
                if overwrite and addr in saddr:
                    stack[saddr.index(addr)] = y
                else:
                    stack.append(y)
                    saddr.append(addr)
                scalars.append((y, 'value', float(val)))
                scalars.append((y, 'stack_k0', y[P.F['stack_k0']]))
    return (np.array(q_rows), np.array(win_rows), np.array(neg_rows),
            np.array(head_ix), scalars, np.array(gaps))


def train(d, seed, data=None, iters=4000, lr=0.02, verbose=False, init=None,
          tied=True):
    """Fit the (NF, d) code. Returns it as a numpy array.

    The objective spans five orders of magnitude -- ``value`` constraints carry true
    magnitudes up to 5050 while indicator constraints carry 1 -- so plain Adam at a
    fixed step oscillates instead of converging. The objective is not reweighted to
    fix that (the absolute tolerance IS the thing being measured); the optimizer is:
    gradient clipping, cosine-decayed steps, and an L-BFGS polish, over 24*d
    parameters. The convergence anchor is d = NF, where the identity code is an exact
    solution and the loss must reach ~0.
    """
    if data is None:
        data = harvest()
    q, win, neg, hix, scalars, gaps = data
    torch.manual_seed(seed)

    U0 = P.codebook(d, seed) if init is None else np.asarray(init, dtype=float)
    U = torch.tensor(U0, dtype=torch.float64, requires_grad=True)
    # tied=True is Tracr's shared-W convention (readout = code). Untying asks whether
    # that tie, rather than the machine, is what forbids superposition.
    Rm = U if tied else torch.tensor(U0.copy(), dtype=torch.float64, requires_grad=True)
    params = [U] if tied else [U, Rm]

    tq = torch.tensor(q)
    tw = torch.tensor(win)
    tn = torch.tensor(neg)
    thix = torch.tensor(hix)
    # query offsets b_Q enter the score through the 'one' coordinate of the query
    offs = torch.tensor([h[2] for h in ARGMAX_HEADS], dtype=torch.float64)[thix]
    q_is_rom = torch.tensor([h[1] == 'rom' for h in ARGMAX_HEADS])[thix]

    sy = torch.tensor(np.array([s[0] for s in scalars]))
    sf = torch.tensor([P.F[s[1]] for s in scalars])
    sv = torch.tensor(np.array([s[2] for s in scalars]))
    s_coef = torch.tensor([0.5 if s[1] == 'stack_k0' else 1.0 for s in scalars],
                          dtype=torch.float64)

    ipf, spf, onef = P.F['ip'], P.F['sp'], P.F['one']
    rk = (P.F['prog_k0'], P.F['prog_k1'])
    mk = (P.F['stack_k0'], P.F['stack_k1'])

    def parts():
        G = Rm @ U.T                     # readout f of a row with values y is (G y)_f
        # score(row) = readout_k0(row) * (readout_addr(query) + offset)
        #            + readout_k1(row) * readout_one(query)
        x = torch.where(q_is_rom, tq @ G[ipf], tq @ G[spf]) + offs
        one_q = tq @ G[onef]

        def score(rows):
            k0 = torch.where(q_is_rom, rows @ G[rk[0]], rows @ G[mk[0]])
            k1 = torch.where(q_is_rom, rows @ G[rk[1]], rows @ G[mk[1]])
            return k0 * x + k1 * one_q

        sep = score(tw) - score(tn)
        # Scale-free surrogates. What the machine needs from each constraint is
        # binary -- the ordering holds, or the scalar rounds correctly -- while the
        # raw quantities span five orders of magnitude. An unnormalized hinge is
        # blind to the tiebreak pairs (target 1e-6, so violating one costs 1e-6) and
        # swamped by the value pairs (magnitude 5050). Dividing each by its own
        # threshold puts every constraint on the same footing; the clamp keeps a
        # single wildly-violated value constraint from owning the gradient.
        margin = torch.relu(1.0 - sep / tgap).mean()
        pred = (sy * G[sf]).sum(1) * s_coef
        err = (pred - sv * s_coef).abs()
        tol = torch.relu(err / TOL - 1.0).clamp(max=100.0).mean()
        return margin, tol

    tgap = torch.tensor(gaps)
    structural = torch.tensor(gaps) >= MARGIN

    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr / 500)
    for it in range(iters):
        margin, tol = parts()
        loss = margin + tol
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if verbose and it % max(iters // 6, 1) == 0:
            print(f'  it{it:5d} loss={loss.item():.5f} margin={margin.item():.5f} '
                  f'tol={tol.item():.5f}')

    lbfgs = torch.optim.LBFGS(params, max_iter=300, line_search_fn='strong_wolfe',
                              tolerance_grad=1e-12, tolerance_change=1e-14)

    def closure():
        lbfgs.zero_grad()
        m, t = parts()
        (m + t).backward()
        return m + t

    lbfgs.step(closure)
    with torch.no_grad():
        margin, tol = parts()
        G = Rm @ U.T
        x = torch.where(q_is_rom, tq @ G[ipf], tq @ G[spf]) + offs
        one_q = tq @ G[onef]
        k0w = torch.where(q_is_rom, tw @ G[rk[0]], tw @ G[mk[0]])
        k1w = torch.where(q_is_rom, tw @ G[rk[1]], tw @ G[mk[1]])
        k0n = torch.where(q_is_rom, tn @ G[rk[0]], tn @ G[mk[0]])
        k1n = torch.where(q_is_rom, tn @ G[rk[1]], tn @ G[mk[1]])
        sep = (k0w * x + k1w * one_q) - (k0n * x + k1n * one_q)
        viol_struct = float((sep[structural] < tgap[structural]).double().mean())
        viol_tie = float((sep[~structural] < tgap[~structural]).double().mean()) \
            if (~structural).any() else 0.0
    if verbose:
        print(f'  final loss={(margin + tol).item():.6f} margin={margin.item():.6f} '
              f'tol={tol.item():.6f}')
    return (U.detach().numpy(), Rm.detach().numpy()), dict(margin=float(margin), tol=float(tol),
                                    viol_structural=viol_struct, viol_tiebreak=viol_tie,
                                    n_structural=int(structural.sum()),
                                    n_tiebreak=int((~structural).sum()))


_CACHE = {}


def learned_codebook(d, seed):
    """Cached so a sweep does not refit the same code once per configuration."""
    key = (d, seed)
    if key not in _CACHE:
        if 'data' not in _CACHE:
            _CACHE['data'] = harvest()
        _CACHE[key] = train(d, seed, data=_CACHE["data"])[0][0]
    return _CACHE[key]


if __name__ == '__main__':
    data = harvest()
    n_tie = int((data[5] < MARGIN).sum())
    print(f'harvested {len(data[0])} margin constraints ({n_tie} of them same-address '
          f'tiebreaks), {len(data[4])} tolerance constraints')
    for d in (4, 6, 8, 12, 16, 24):
        (U, _), fin = train(d, 0, data=data)
        R = P.readout(U, 'dot')
        H = P.build_heads(R)
        got = []
        for name, prog, expect in P.programs():
            g, _ = P.run(prog, U, R, H, max_steps=2000)
            got.append('.' if (g is not None and abs(g - expect) < 1e-9) else 'x')
        norms = np.linalg.norm(U, axis=1)
        ind = norms[[P.F[f'op_{o}'] for o in P.OPS]]
        dense = norms[[P.F[f] for f in ('prog_k0', 'prog_k1', 'stack_k0', 'stack_k1',
                                        'opcode', 'value', 'ip', 'sp', 'one')]]
        print(f'd={d:3d} computes={"".join(got)} '
              f'viol_struct={fin["viol_structural"]:.3f} '
              f'viol_tie={fin["viol_tiebreak"]:.3f} tol={fin["tol"]:.4f} '
              f'indicator_norm={np.median(ind):.3f} dense_norm={np.median(dense):.3f}')


def visited_states(data):
    """Every feature-value vector the reference machine visits, as rows (N, NF)."""
    q, win, neg, hix, scalars, gaps = data
    rows = [np.asarray(q, dtype=float).reshape(-1, P.NF),
            np.asarray(win, dtype=float).reshape(-1, P.NF),
            np.asarray(neg, dtype=float).reshape(-1, P.NF)]
    if len(scalars):
        rows.append(np.stack([np.asarray(v, dtype=float) for v, _, _ in scalars]))
    return np.unique(np.vstack(rows), axis=0)


def train_continuation(d_min=4, d_max=None, data=None, iters=4000, verbose=True,
                       tied=True, projection='code'):
    """Compress one dimension at a time, starting from the exact solution.

    Training from a random init does not reach the optimum even where one provably
    exists: at d = NF the identity code scores exactly 0, and random-init Adam+L-BFGS
    plateaus around 20% violated margins with five times the iteration budget. Any
    threshold read off that would be a fact about the optimizer.

    So walk down instead. Start at d = NF with the identity (loss 0 by construction),
    and to go from d to d-1 project the current code onto its own top-(d-1) right
    singular subspace -- dropping the direction it is using least -- then retrain.
    Each step starts from a solved problem one dimension away.

    ``projection`` picks what the SVD is taken over when a dimension is dropped:
    'code' is the SVD of U itself (the rule used in RESULTS-A.md); 'data' is the SVD
    of S @ U, the code's image of the visited states, which drops the direction the
    trajectory uses least. The ALTA replication (experiments/alta-superposition)
    found the two rules give different geometry on every program it tried.

    Returns {d: (U, diagnostics)}.
    """
    if data is None:
        data = harvest()
    S = visited_states(data) if projection == 'data' else None
    d_max = d_max or P.NF
    out = {}
    U = np.eye(P.NF)[:, :d_max] if d_max <= P.NF else None
    if U is None:
        raise ValueError('d_max above n_features')
    for d in range(d_max, d_min - 1, -1):
        if d < U.shape[1]:
            # drop the least-used direction, keeping the code's own geometry
            basis = U if S is None else S @ U
            _, _, Vt = np.linalg.svd(basis, full_matrices=False)
            U = U @ Vt[:d].T
        (U, Rm), fin = train(d, 0, data=data, iters=iters, init=U, tied=tied)
        out[d] = (U.copy(), Rm.copy(), fin)
        if verbose:
            print(f'  d={d:3d} margin={fin["margin"]:.5f} tol={fin["tol"]:.5f} '
                  f'viol={fin["viol_structural"]:.4f}', flush=True)
    return out
