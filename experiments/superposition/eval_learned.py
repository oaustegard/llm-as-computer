"""Evaluate continuation-trained codes with the same machine and the same blind
analyst the random-code sweep used. Only the code changes.

Usage:  python3 eval_learned.py [--out learned_results.json]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyst_sp as AN
import learned as L
import packed as P

OVERWRITE = True   # the codes are trained against the overwrite-in-place stack


def evaluate(U, refs):
    R = P.readout(U, 'dot')          # Tracr's shared-W convention: readout tied to code
    H = P.build_heads(R)
    row = {'programs': {}}
    for name, prog, expect in P.programs():
        ref = refs[name]
        tr = []
        got, steps = P.run(prog, U, R, H, max_steps=2 * len(ref) + 50, trace=tr,
                           overwrite=OVERWRITE)
        dv = P.first_divergence(tr, ref)
        row['programs'][name] = dict(
            ok=bool(got is not None and abs(got - expect) < 1e-9),
            got=None if got is None else float(got),
            div_step=None if dv is None else dv[0],
            div_kind=None if dv is None else dv[1])
    for mode, tag in ((True, 'ideal'), (False, 'self')):
        out = AN.recover(P.make_artifact(U, R, H, seed=11, ideal=mode))
        row[f'recover_{tag}'] = (
            sum(1 for v in out['row_of'] if out['row_of'][v] == P.TRUE_ROW_OF[v])
            if out['ok'] else 0)
        if not out['ok']:
            row[f'why_{tag}'] = out['why']
    ind = np.linalg.norm(U, axis=1)[[P.F[f'op_{o}'] for o in P.OPS]]
    dense = np.linalg.norm(U, axis=1)[
        [P.F[f] for f in ('is_prog', 'is_stack', 'is_state', 'prog_k0', 'prog_k1',
                          'stack_k0', 'stack_k1', 'opcode', 'value', 'ip', 'sp', 'one')]]
    row['indicator_norm_median'] = float(np.median(ind))
    row['dense_norm_median'] = float(np.median(dense))
    row['indicator_norm_max'] = float(np.max(ind))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='learned_results.json')
    ap.add_argument('--dmin', type=int, default=4)
    args = ap.parse_args()

    refs = {}
    U0 = P.identity_codebook()
    H0 = P.build_heads(U0)
    for name, prog, expect in P.programs():
        tr = []
        got, _ = P.run(prog, U0, U0, H0, trace=tr, overwrite=OVERWRITE)
        assert abs(got - expect) < 1e-9, f'{name} reference broken under overwrite'
        refs[name] = tr

    data = L.harvest(overwrite=OVERWRITE)
    print(f'{len(data[0])} margin constraints, {len(data[4])} tolerance constraints',
          flush=True)
    codes = L.train_continuation(d_min=args.dmin, data=data)

    rows = []
    for d, (U, Rm, fin) in sorted(codes.items()):
        row = evaluate(U, refs)
        row.update(d=d, loss_margin=fin['margin'], loss_tol=fin['tol'],
                   viol=fin['viol_structural'])
        rows.append(row)
        comp = ''.join('.' if p['ok'] else 'x' for p in row['programs'].values())
        print(f'd={d:3d} computes={comp} recover={row["recover_ideal"]:2d}/12 '
              f'viol={row["viol"]:.3f} ind_norm={row["indicator_norm_median"]:.4f} '
              f'dense_norm={row["dense_norm_median"]:.3f}', flush=True)

    json.dump(dict(meta=dict(overwrite=OVERWRITE, n_features=P.NF,
                             margin=L.MARGIN, tol=L.TOL), rows=rows),
              open(args.out, 'w'), indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
