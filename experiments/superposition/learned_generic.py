"""``learned.py``, with the attention score read off HEAD_SPEC instead of hardcoded.

``learned.py`` computes a head's score as ``k0*x + k1*one_q`` with ``rk``/``mk`` the
parabolic key feature indices and ``x`` the scalar address query plus the head's b_Q
offset. That is the parabolic machine written into the trainer. A categorical-address
machine has 11 or 5 one-hot key bits per head and no scalar to offset, so the same
objective has to be expressed over whatever Q/K row pairs the machine declares:

    score(query, key) = sum_r ( sum_(f,c) in Q_r  c * readout_f(query) + b_r )
                            * ( sum_(f,c) in K_r  c * readout_f(key) )

with ``readout_f(y) = (R U^T y)_f``, exactly what ``learned.parts()`` does for the two
parabolic rows. On ``packed`` this reduces to that expression term for term.

Everything else is held fixed on purpose, because the point of the experiment is that
only the addressing changes: margin hinge ``relu(1 - sep/gap)`` at MARGIN = 0.5,
absolute tolerance hinge at TOL = 0.25, N_NEG = 4 hardest competitors, continuation
from the identity code, the trajectory-SVD ('data') projection rule, the tied 'dot'
readout, and the same Adam + L-BFGS schedule.

The machine is a parameter: any module exposing FEATURES/F/NF, HEAD_SPEC,
ARGMAX_HEADS, ADDR_SCALARS, embed_* and run will do.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import packed as _default_machine  # noqa: E402

MARGIN = 0.5      # half the reference winner-to-runner-up gap of 1
TOL = 0.25        # half the quantizer's rounding tolerance
N_NEG = 4         # hardest competitors kept per (step, head)


# ---------------- generic score over HEAD_SPEC ----------------
def _coef_matrix(P, rows):
    """(n_rows, NF) of the (feature, coeff) pairs in one side of a head spec."""
    M = np.zeros((len(rows), P.NF))
    for r, row in enumerate(rows):
        for f, c in row:
            M[r, P.F[f]] += float(c)
    return M


def head_matrices(P, head):
    q, k, _v, bq = P.HEAD_SPEC[head]
    return _coef_matrix(P, q), _coef_matrix(P, k), np.asarray(bq, dtype=float)


def _ref_scores(P, head, q_y, pool):
    """Scores the exact (identity-code) machine assigns, for picking winner and
    competitors. Same generic expression, with the readout being the identity."""
    Qc, Kc, bq = head_matrices(P, head)
    qv = Qc @ q_y + bq                      # (n_rows,)
    K = np.asarray(pool) @ Kc.T             # (n_pool, n_rows)
    return K @ qv


# ---------------- constraint harvest ----------------
def harvest(P=_default_machine, programs=None, overwrite=True):
    """Every decision the reference machine makes, as feature-value vectors.

    Returns (queries, winners, competitors, head indices, scalars, target gaps),
    sorted by head index so ``parts`` can slice contiguously. ``scalars`` entries are
    (feature-value vector, feature name, target value, weight).

    Pairs whose reference gap is <= 0 are dropped. Under one-hot addressing a query
    for an address no row holds scores every row 0, so the argmax is arbitrary and
    the read is discarded by the address check -- there is no decision there to
    preserve, and a zero target gap would divide the margin hinge by zero. On the
    parabolic machine no pair is ever dropped.
    """
    eye = np.eye(P.NF)
    H = P.build_heads(eye)
    progs = programs if programs is not None else P.programs()
    eps = 0.0 if overwrite else getattr(P, 'EPS', 0.0)

    q_rows, win_rows, neg_rows, head_ix, gaps = [], [], [], [], []
    scalars = []
    n_dropped = 0
    for _name, prog, _expect in progs:
        rom = [P.embed_prog(eye, p, o, a) for p, (o, a) in enumerate(prog)]
        for y in rom:
            scalars.append((y, 'opcode', y[P.F['opcode']], 1.0))
            scalars.append((y, 'value', y[P.F['value']], 1.0))
        tr = []
        P.run(prog, eye, eye, H, trace=tr, overwrite=overwrite)
        stack, saddr = [], []
        for rec in tr:
            q = P.embed_state(eye, rec['ip'], rec['sp'])
            for hi, (h, region) in enumerate(P.ARGMAX_HEADS):
                pool = rom if region == 'rom' else stack
                if len(pool) < 2:
                    continue
                s = _ref_scores(P, h, q, pool)
                order = np.argsort(-s)
                w = order[0]
                for c in order[1:1 + N_NEG]:
                    gap = float(s[w] - s[c])
                    if gap <= 0.0:
                        n_dropped += 1
                        continue
                    q_rows.append(q)
                    win_rows.append(pool[w])
                    neg_rows.append(pool[c])
                    head_ix.append(hi)
                    # preserve the gap the reference machine had, capped at MARGIN
                    gaps.append(min(MARGIN, gap))
            for addr, val, wo in rec.get('wrote', ()):
                y = P.embed_stack(eye, addr, val, wo, eps=eps)
                if overwrite and addr in saddr:
                    stack[saddr.index(addr)] = y
                else:
                    stack.append(y)
                    saddr.append(addr)
                scalars.append((y, 'value', float(val), 1.0))
                for f, wgt, tgt in P.ADDR_SCALARS(addr):
                    scalars.append((y, f, tgt, wgt))
    order = np.argsort(np.array(head_ix), kind='stable')
    return (np.array(q_rows)[order], np.array(win_rows)[order],
            np.array(neg_rows)[order], np.array(head_ix)[order], scalars,
            np.array(gaps)[order], n_dropped)


# ---------------- training ----------------
def train(P, d, seed, data, iters=4000, lr=0.02, verbose=False, init=None, tied=True):
    """Fit the (NF, d) code. Same objective and optimizer schedule as ``learned.train``."""
    q, win, neg, hix, scalars, gaps, _drop = data
    torch.manual_seed(seed)

    U0 = P.codebook(d, seed) if init is None else np.asarray(init, dtype=float)
    U = torch.tensor(U0, dtype=torch.float64, requires_grad=True)
    Rm = U if tied else torch.tensor(U0.copy(), dtype=torch.float64, requires_grad=True)
    params = [U] if tied else [U, Rm]

    tq, tw, tn = torch.tensor(q), torch.tensor(win), torch.tensor(neg)
    tgap = torch.tensor(gaps)
    structural = tgap >= MARGIN

    # contiguous slice per head (harvest sorted by head index)
    blocks = []
    for hi, (h, _region) in enumerate(P.ARGMAX_HEADS):
        idx = np.flatnonzero(hix == hi)
        if not len(idx):
            continue
        Qc, Kc, bq = head_matrices(P, h)
        blocks.append((int(idx[0]), int(idx[-1]) + 1, torch.tensor(Qc),
                       torch.tensor(Kc), torch.tensor(bq)))

    sy = torch.tensor(np.array([s[0] for s in scalars]))
    sf = torch.tensor([P.F[s[1]] for s in scalars])
    sv = torch.tensor(np.array([float(s[2]) for s in scalars]))
    s_coef = torch.tensor([float(s[3]) for s in scalars], dtype=torch.float64)

    def separations(G):
        out = []
        for lo, hi_, Qc, Kc, bq in blocks:
            qv = tq[lo:hi_] @ (Qc @ G).T + bq          # (n, n_rows)
            kw = tw[lo:hi_] @ (Kc @ G).T
            kn = tn[lo:hi_] @ (Kc @ G).T
            out.append(((qv * kw).sum(1) - (qv * kn).sum(1)))
        return torch.cat(out)

    def parts():
        G = Rm @ U.T                       # readout f of a row with values y is (G y)_f
        sep = separations(G)
        margin = torch.relu(1.0 - sep / tgap).mean()
        pred = (sy * G[sf]).sum(1) * s_coef
        err = (pred - sv * s_coef).abs()
        tol = torch.relu(err / TOL - 1.0).clamp(max=100.0).mean()
        return margin, tol

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
                  f'tol={tol.item():.5f}', flush=True)

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
        sep = separations(Rm @ U.T)
        viol_struct = float((sep[structural] < tgap[structural]).double().mean())
        viol_tie = float((sep[~structural] < tgap[~structural]).double().mean()) \
            if bool((~structural).any()) else 0.0
    if verbose:
        print(f'  final loss={(margin + tol).item():.6f} margin={margin.item():.6f} '
              f'tol={tol.item():.6f}', flush=True)
    return (U.detach().numpy(), Rm.detach().numpy()), dict(
        margin=float(margin), tol=float(tol), viol_structural=viol_struct,
        viol_tiebreak=viol_tie, n_structural=int(structural.sum()),
        n_tiebreak=int((~structural).sum()))


def visited_states(P, data):
    """Every feature-value vector the reference machine visits, as rows (N, NF)."""
    q, win, neg, _hix, scalars, _gaps, _drop = data
    rows = [np.asarray(q, dtype=float).reshape(-1, P.NF),
            np.asarray(win, dtype=float).reshape(-1, P.NF),
            np.asarray(neg, dtype=float).reshape(-1, P.NF)]
    if len(scalars):
        rows.append(np.stack([np.asarray(s[0], dtype=float) for s in scalars]))
    return np.unique(np.vstack(rows), axis=0)


def train_continuation(P=_default_machine, d_min=4, d_max=None, data=None, iters=4000,
                       verbose=True, tied=True, projection='data'):
    """Compress one dimension at a time, starting from the exact identity solution.

    ``projection='data'`` is the trajectory-SVD rule: drop the direction the visited
    states use least. See ``learned.train_continuation``; the only change is that the
    machine is a parameter.
    """
    if data is None:
        data = harvest(P)
    S = visited_states(P, data) if projection == 'data' else None
    d_max = d_max or P.NF
    if d_max > P.NF:
        raise ValueError('d_max above n_features')
    out = {}
    U = np.eye(P.NF)[:, :d_max]
    for d in range(d_max, d_min - 1, -1):
        if d < U.shape[1]:
            basis = U if S is None else S @ U
            _, _, Vt = np.linalg.svd(basis, full_matrices=False)
            U = U @ Vt[:d].T
        (U, Rm), fin = train(P, d, 0, data, iters=iters, init=U, tied=tied)
        out[d] = (U.copy(), Rm.copy(), fin)
        if verbose:
            print(f'  d={d:3d} margin={fin["margin"]:.5f} tol={fin["tol"]:.5f} '
                  f'viol={fin["viol_structural"]:.4f}', flush=True)
    return out
