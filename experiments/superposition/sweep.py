"""Sweep: does the machine still compute, and is the ISA still recoverable, as the
residual stream is compressed?

Per (arm, d, seed):
  computes   -- each of the four oracle programs run in the packed basis, scored on
                returning the exact expected value, plus where it first parted
                company with the reference machine
  recovers   -- the blind analyst's exact-ISA score out of 12, on two artifacts:
                'ideal'  activations are the reference trajectory re-encoded in the
                         packed basis (is the ISA still readable?)
                'self'   activations are what this packed machine actually produced
                         (is the ISA still readable off a machine that is itself
                         malfunctioning?)

Usage:  python3 sweep.py [--quick] [--out results.json] [--seeds N] [--procs N]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyst_sp as AN
import packed as P

# pinv is exact at d >= n_features and falls off a cliff below, so it wants a fine
# grid there; the dot arms degrade like 1/sqrt(d) and want decades.
GRIDS = {
    'pinv': [4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 26, 32, 48],
    'dot': [8, 16, 24, 32, 64, 256, 1024, 2048, 4096, 8192, 16384],
    'scaled': [8, 16, 24, 32, 64, 256, 1024, 2048, 4096, 8192, 16384],
}
QUICK = {'pinv': [16, 23, 24], 'dot': [64, 1024], 'scaled': [64, 1024]}

_REFS = None
_SCALES = None


def _init():
    global _REFS, _SCALES
    _REFS = P.reference_traces()
    _SCALES = P.workload_scales()


def one(job):
    arm, d, seed = job
    t0 = time.time()
    U = P.codebook_for(d, seed, arm, _SCALES)
    R = P.readout(U, arm)
    H = P.build_heads(R)

    row = dict(arm=arm, d=d, seed=seed, programs={})
    for name, prog, expect in P.programs():
        ref = _REFS[name]
        tr = []
        got, steps = P.run(prog, U, R, H, max_steps=2 * len(ref) + 50, trace=tr)
        dv = P.first_divergence(tr, ref)
        row['programs'][name] = dict(
            ok=bool(got is not None and abs(got - expect) < 1e-9),
            got=None if got is None else float(got), steps=int(steps),
            div_step=None if dv is None else dv[0], div_kind=None if dv is None else dv[1])

    for mode, tag in ((True, 'ideal'), (False, 'self')):
        A = P.make_artifact(U, R, H, seed=1000 + seed, ideal=mode)
        out = AN.recover(A)
        if out['ok']:
            score = sum(1 for v in out['row_of'] if out['row_of'][v] == P.TRUE_ROW_OF[v])
            row[f'recover_{tag}'] = dict(score=score, why=None,
                                         n_cand=out['diag'].get('n_candidates'),
                                         replay_err=out['diag'].get('replay_err'))
        else:
            row[f'recover_{tag}'] = dict(score=0, why=out['why'],
                                         matched=out['diag'].get('best_partial', {}).get('matched'))
        row[f'recover_{tag}']['law_rom'] = out['diag'].get('law_rom', {}).get('resid')
        row[f'recover_{tag}']['law_mem'] = out['diag'].get('law_mem', {}).get('resid')
    row['secs'] = round(time.time() - t0, 2)
    return row


def audit_divergence(seeds=3):
    """Every run whose trace parted company with the reference machine should also
    have failed. Aborting the run at that point was tried as an optimization and
    rejected: it flipped 4 of 192 verdicts, all countdown_5, all in the same
    direction -- a machine can leave the reference trajectory and still land on the
    right answer. The runs below are therefore never truncated early."""
    _init()
    bad = []
    for arm in ('pinv', 'dot'):
        for d in (16, 23, 24, 64):
            for seed in range(seeds):
                U = P.codebook_for(d, seed, arm, _SCALES)
                R = P.readout(U, arm)
                H = P.build_heads(R)
                for name, prog, expect in P.programs():
                    ref = _REFS[name]
                    tr = []
                    got = P.run(prog, U, R, H, max_steps=2 * len(ref) + 50, trace=tr)[0]
                    ok = got is not None and abs(got - expect) < 1e-9
                    if ok and P.first_divergence(tr, ref) is not None:
                        bad.append((arm, d, seed, name))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='results.json')
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--procs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--audit', action='store_true')
    args = ap.parse_args()

    if args.audit:
        bad = audit_divergence()
        print('runs that diverged yet still returned the right answer:',
              bad if bad else 'none')
        return

    grids = QUICK if args.quick else GRIDS
    jobs = [(arm, d, s) for arm, ds in grids.items() for d in ds for s in range(args.seeds)]
    # the large-d configs run for minutes, so finished rows land in a jsonl as they
    # arrive and a re-run picks up where the last one stopped
    part = args.out + '.partial'
    done, rows = set(), []
    if os.path.exists(part):
        for line in open(part):
            r = json.loads(line)
            rows.append(r)
            done.add((r['arm'], r['d'], r['seed']))
        print(f'resuming: {len(done)} configs already done', flush=True)
    jobs = [j for j in jobs if j not in done]
    print(f'{len(jobs)} configs over {args.procs} processes', flush=True)
    t0 = time.time()
    with open(part, 'a', buffering=1) as fh, Pool(args.procs, initializer=_init) as pool:
        for i, row in enumerate(pool.imap_unordered(one, jobs, chunksize=1)):
            rows.append(row)
            fh.write(json.dumps(row) + '\n')
            if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                print(f'  {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s', flush=True)
    meta = dict(n_features=P.NF, features=P.FEATURES, seeds=args.seeds,
                grids={k: v for k, v in grids.items()},
                scales={n: float(v) for n, v in zip(P.FEATURES, P.workload_scales())},
                elapsed=round(time.time() - t0, 1))
    json.dump(dict(meta=meta, rows=rows), open(args.out, 'w'), indent=1)
    print('wrote', args.out, f'({len(rows)} rows, {meta["elapsed"]}s)')


def thresholds(path='results.json'):
    """Smallest d at which a majority of seeds still compute / still recover."""
    data = json.load(open(path))
    rows = data['rows']
    arms = sorted({r['arm'] for r in rows})
    progs = list(rows[0]['programs'])
    out = {}
    for arm in arms:
        rs = [r for r in rows if r['arm'] == arm]
        ds = sorted({r['d'] for r in rs})
        col = {}
        for name in progs:
            frac = {d: np.mean([r['programs'][name]['ok'] for r in rs if r['d'] == d])
                    for d in ds}
            good = [d for d in ds if frac[d] > 0.5]
            col[name] = min(good) if good else None
        for tag in ('ideal', 'self'):
            frac = {d: np.mean([r[f'recover_{tag}']['score'] == 12 for r in rs if r['d'] == d])
                    for d in ds}
            good = [d for d in ds if frac[d] > 0.5]
            col[f'recover_{tag}'] = min(good) if good else None
        out[arm] = col
    return out


if __name__ == '__main__':
    main()
