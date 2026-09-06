"""Magnitude sweep: does the largest value the machine holds set how far the fitted
code can compress? Program shape, trace length and opcode mix are held fixed; only the
magnitude of the values changes. Predictions in PREDICTIONS-M.md.

Usage:  python3 magnitude_sweep.py [--cs 1,10,100] [--out magnitude_results.json]
Resumable per c: a finished c is skipped when its key is already in the output file.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyst_sp as AN  # noqa: E402
import learned as L  # noqa: E402
import packed as P  # noqa: E402
from compile_artifact import countdown, rot_jz_nop  # noqa: E402

OVERWRITE = True
DENSE = ('is_prog', 'is_stack', 'is_state', 'prog_k0', 'prog_k1', 'stack_k0', 'stack_k1',
         'opcode', 'value', 'ip', 'sp', 'one')


def ladder(c, k=50):
    """PUSH c, then k times (PUSH c; ADD), HALT. Returns (k+1)*c. Same trace length and
    control flow for every c; only the values differ."""
    prog = [('PUSH', c)]
    for _ in range(k):
        prog += [('PUSH', c), ('ADD', 0)]
    prog.append(('HALT', 0))
    return prog, (k + 1) * c


def program_set(c):
    return [(f'ladder_{c}', *ladder(c)), ('countdown_5', *countdown(5)),
            ('rot_jz_nop', *rot_jz_nop())]


def evaluate(U, progs, refs):
    R = P.readout(U, 'dot')
    H = P.build_heads(R)
    row = {'programs': {}}
    for name, prog, expect in progs:
        ref = refs[name]
        tr = []
        got, _ = P.run(prog, U, R, H, max_steps=2 * len(ref) + 50, trace=tr,
                       overwrite=OVERWRITE)
        row['programs'][name] = dict(ok=bool(got is not None and abs(got - expect) < 1e-9),
                                     got=None if got is None else float(got))
    out = AN.recover(P.make_artifact(U, R, H, seed=11, ideal=True))
    row['recover_ideal'] = (sum(1 for v in out['row_of'] if out['row_of'][v] == P.TRUE_ROW_OF[v])
                            if out['ok'] else 0)
    idx = [P.F[f] for f in DENSE]
    rows = U[idx]
    unit = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
    G = unit @ unit.T
    T = rows @ R[idx].T
    row['gram_off_dense'] = float(np.abs(G - np.diag(np.diag(G))).max())
    row['transfer_off_dense'] = float(np.abs(T - np.diag(np.diag(T))).max())
    ind = np.linalg.norm(U, axis=1)[[P.F[f'op_{o}'] for o in P.OPS]]
    row['indicator_norm_median'] = float(np.median(ind))
    return row


def run_c(c, dmin, iters):
    progs = program_set(c)
    U0 = P.identity_codebook()
    H0 = P.build_heads(U0)
    refs = {}
    for name, prog, expect in progs:
        tr = []
        got, _ = P.run(prog, U0, U0, H0, trace=tr, overwrite=OVERWRITE)
        assert got is not None and abs(got - expect) < 1e-9, f'{name} reference broken'
        refs[name] = tr
    data = L.harvest(programs=progs, overwrite=OVERWRITE)
    print(f'[c={c}] max value {max(e for _, _, e in progs)}; trace steps '
          f'{ {k: len(v) for k, v in refs.items()} }; {len(data[0])} margin constraints, '
          f'{len(data[4])} tolerance constraints', flush=True)
    codes = L.train_continuation(d_min=dmin, data=data, iters=iters, projection='data',
                                 verbose=False)
    rows = []
    for d, (U, Rm, fin) in sorted(codes.items()):
        row = evaluate(U, progs, refs)
        row.update(d=d, viol=fin['viol_structural'])
        rows.append(row)
        comp = ''.join('.' if p['ok'] else 'x' for p in row['programs'].values())
        print(f'  c={c:4d} d={d:3d} computes={comp} recover={row["recover_ideal"]:2d}/12 '
              f'gram_off={row["gram_off_dense"]:.2f} xfer_off={row["transfer_off_dense"]:.2f} '
              f'ind_norm={row["indicator_norm_median"]:.3f}', flush=True)
    working = [r['d'] for r in rows if all(p['ok'] for p in r['programs'].values())]
    d_min = min(working) if working else None
    at = next((r for r in rows if r['d'] == d_min), None)
    return dict(c=c, max_value=max(e for _, _, e in progs), d_min=d_min,
                transfer_at_dmin=None if at is None else at['transfer_off_dense'],
                gram_at_dmin=None if at is None else at['gram_off_dense'],
                n_margin=len(data[0]), n_tol=len(data[4]),
                steps={k: len(v) for k, v in refs.items()}, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cs', default='1,10,100')
    ap.add_argument('--dmin', type=int, default=4)
    ap.add_argument('--iters', type=int, default=4000)
    ap.add_argument('--out', default='magnitude_results.json')
    args = ap.parse_args()
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for c in [int(x) for x in args.cs.split(',')]:
        if str(c) in results:
            print(f'[c={c}] done, skipping', flush=True)
            continue
        results[str(c)] = run_c(c, args.dmin, args.iters)
        json.dump(results, open(args.out, 'w'), indent=1)
        print(f'[c={c}] d_min={results[str(c)]["d_min"]} '
              f'transfer_at_dmin={results[str(c)]["transfer_at_dmin"]}', flush=True)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
