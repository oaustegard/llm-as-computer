"""Addressing sweep: is one-hot selection what lets ALTA's codes compress?

LAC's fitted code gains one dimension (12 dense features -> d_min 11 under the
trajectory-SVD rule) where ALTA's compiled programs gain 20-45%. Iteration fell to the
ALTA replication, orthogonality to the projection rule, and magnitude to
``magnitude_sweep.py``. The remaining difference is how a memory row is selected:
LAC by a numeric dot product over parabolic keys, ALTA by one-hot equality.

This sweep fits the same objective, with the same continuation and the same
projection rule, to two machines that differ only in that -- ``packed`` (parabolic)
and ``packed_cat`` (categorical) -- and compares ``d_min / dense feature count``.
Predictions in PREDICTIONS-C.md.

Usage:  python3 addressing_sweep.py [--machines parabolic,categorical]
                                    [--dmax N] [--dmin N] [--iters N]
                                    [--out addressing_results.json]
Resumable per machine: a finished machine is skipped when its key is in the output.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import learned_generic as LG  # noqa: E402
import packed as P_PARA  # noqa: E402
import packed_cat as P_CAT  # noqa: E402

OVERWRITE = True
MACHINES = {'parabolic': P_PARA, 'categorical': P_CAT}


def used_features(P):
    """Features any head actually reads -- the Q, K and V rows of HEAD_SPEC."""
    used = set()
    for q, k, v, _bq in P.HEAD_SPEC.values():
        for rows in (q, k, v):
            for row in rows:
                for f, _c in row:
                    used.add(f)
    return sorted(used)


def evaluate(P, U, refs, dense_idx):
    R = P.readout(U, 'dot')
    H = P.build_heads(R)
    row = {'programs': {}}
    for name, prog, expect in P.programs():
        ref = refs[name]
        tr = []
        got, _ = P.run(prog, U, R, H, max_steps=2 * len(ref) + 50, trace=tr,
                       overwrite=OVERWRITE)
        dv = P.first_divergence(tr, ref)
        row['programs'][name] = dict(
            ok=bool(got is not None and abs(got - expect) < 1e-9),
            got=None if got is None else float(got),
            div_step=None if dv is None else dv[0],
            div_kind=None if dv is None else dv[1])
    rows = U[dense_idx]
    unit = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
    G = unit @ unit.T
    T = rows @ R[dense_idx].T
    row['gram_off_dense'] = float(np.abs(G - np.diag(np.diag(G))).max())
    row['transfer_off_dense'] = float(np.abs(T - np.diag(np.diag(T))).max())
    norms = np.linalg.norm(U, axis=1)
    row['n_live_dense'] = int((norms[dense_idx] > 1e-6).sum())
    ind = [P.F[f] for f in P.FEATURES if f.startswith('op_')]
    row['indicator_norm_median'] = float(np.median(norms[ind])) if ind else 0.0
    row['dense_norm_median'] = float(np.median(norms[dense_idx]))
    return row


def run_machine(key, dmax, dmin, iters):
    P = MACHINES[key]
    eye = P.identity_codebook()
    H0 = P.build_heads(eye)
    refs = {}
    for name, prog, expect in P.programs():
        tr = []
        got, _ = P.run(prog, eye, eye, H0, trace=tr, overwrite=OVERWRITE)
        assert got is not None and abs(got - expect) < 1e-9, f'{name} reference broken'
        refs[name] = tr
    data = LG.harvest(P, overwrite=OVERWRITE)
    dense = list(P.DENSE)
    dense_idx = [P.F[f] for f in dense]
    used = used_features(P)
    top = dmax or P.NF
    print(f'[{key}] NF={P.NF} dense={len(dense)} used={len(used)} '
          f'{len(data[0])} margin constraints ({data[6]} degenerate pairs dropped), '
          f'{len(data[4])} tolerance constraints; d {top} -> {dmin}', flush=True)

    rows = []
    S = LG.visited_states(P, data)
    U = np.eye(P.NF)[:, :top]
    # Per-width checkpoint: a killed run resumes from the last finished width
    # (the continuation is stateful, each width starts from the previous code).
    ckpt = HERE / f'partial_{key}.npz'
    widths = list(range(top, dmin - 1, -1))
    if ckpt.exists():
        z = np.load(ckpt, allow_pickle=True)
        rows = list(z['rows'])
        U = z['U']
        done_d = int(z['d'])
        widths = [w for w in widths if w < done_d]
        print(f'  {key} resumed after d={done_d}, {len(rows)} rows', flush=True)
    for d in widths:
        t0 = time.time()
        if d < U.shape[1]:
            _, _, Vt = np.linalg.svd(S @ U, full_matrices=False)
            U = U @ Vt[:d].T
        (U, _Rm), fin = LG.train(P, d, 0, data, iters=iters, init=U)
        row = evaluate(P, U, refs, dense_idx)
        row.update(d=d, viol=fin['viol_structural'], loss_margin=fin['margin'],
                   loss_tol=fin['tol'], secs=round(time.time() - t0, 1))
        rows.append(row)
        np.savez(ckpt, rows=np.array(rows, dtype=object), U=U, d=d)
        comp = ''.join('.' if p['ok'] else 'x' for p in row['programs'].values())
        print(f'  {key} d={d:3d} computes={comp} viol={row["viol"]:.3f} '
              f'live={row["n_live_dense"]:3d}/{len(dense)} '
              f'gram_off={row["gram_off_dense"]:.3f} '
              f'xfer_off={row["transfer_off_dense"]:.3f} '
              f'ind_norm={row["indicator_norm_median"]:.3f} '
              f'{row["secs"]:.1f}s', flush=True)

    working = [r['d'] for r in rows if all(p['ok'] for p in r['programs'].values())]
    d_min = min(working) if working else None
    at = next((r for r in rows if r['d'] == d_min), None)
    return dict(machine=key, n_features=P.NF, n_dense=len(dense), n_used=len(used),
                dense=dense, d_max=top, d_floor=dmin, iters=iters, d_min=d_min,
                # d_min equal to the floor is the sweep running out of widths, not a
                # measured threshold; only a run that reaches a failing width has one
                d_min_is_floor=bool(d_min is not None and d_min == dmin),
                ratio=None if d_min is None else d_min / len(dense),
                transfer_at_dmin=None if at is None else at['transfer_off_dense'],
                gram_at_dmin=None if at is None else at['gram_off_dense'],
                n_margin=len(data[0]), n_tol=len(data[4]), n_dropped=data[6],
                rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--machines', default='parabolic,categorical')
    ap.add_argument('--dmax', type=int, default=None, help='default: each machine NF')
    ap.add_argument('--dmin', type=int, default=4)
    ap.add_argument('--iters', type=int, default=4000)
    ap.add_argument('--out', default='addressing_results.json')
    args = ap.parse_args()

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for key in args.machines.split(','):
        if key in results:
            print(f'[{key}] done, skipping', flush=True)
            continue
        results[key] = run_machine(key, args.dmax, args.dmin, args.iters)
        json.dump(results, open(args.out, 'w'), indent=1)
        r = results[key]
        print(f'[{key}] d_min={r["d_min"]} dense={r["n_dense"]} ratio={r["ratio"]} '
              f'transfer_at_dmin={r["transfer_at_dmin"]}', flush=True)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
