"""Two checks the result demands.

1. Is the d=12 code actually superposing, or has it just dropped features and kept the
   survivors orthogonal? The Gram matrix answers that directly.
2. PREDICTIONS-A.md fixed the rule in advance: the learned arm is trained on the same
   programs it is scored on, so if it succeeds, re-run with a program held out before
   claiming the number.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import learned as L
import packed as P

DENSE = ['is_prog', 'is_stack', 'is_state', 'prog_k0', 'prog_k1', 'stack_k0',
         'stack_k1', 'opcode', 'value', 'ip', 'sp', 'one']


def gram_report(U, d):
    G = U @ U.T
    di = [P.F[f] for f in DENSE]
    ii = [P.F[f'op_{o}'] for o in P.OPS]
    sub = G[np.ix_(di, di)]
    off = sub - np.diag(np.diag(sub))
    print(f'd={d}: dense block  max|diag-1|={np.abs(np.diag(sub) - 1).max():.2e}  '
          f'max|offdiag|={np.abs(off).max():.2e}')
    print(f'       indicator norms: max={np.abs(np.diag(G[np.ix_(ii, ii)])).max():.2e}')
    worst = np.unravel_index(np.abs(off).argmax(), off.shape)
    print(f'       largest dense overlap: {DENSE[worst[0]]} . {DENSE[worst[1]]} '
          f'= {off[worst]:+.2e}')


def main():
    refs = {}
    U0 = P.identity_codebook()
    H0 = P.build_heads(U0)
    for name, prog, expect in P.programs():
        tr = []
        P.run(prog, U0, U0, H0, trace=tr, overwrite=True)
        refs[name] = tr

    print('=== 1. is the d=12 code superposing? ===')
    data = L.harvest(overwrite=True)
    codes = L.train_continuation(d_min=11, data=data, verbose=False)
    for d in (12, 11):
        gram_report(codes[d][0], d)

    print('\n=== 2. held-out program (pre-registered leakage rule) ===')
    allp = P.programs()
    for held in range(len(allp)):
        train_progs = [p for i, p in enumerate(allp) if i != held]
        hdata = L.harvest(programs=train_progs, overwrite=True)
        hcodes = L.train_continuation(d_min=12, d_max=P.NF, data=hdata, verbose=False)
        U = hcodes[12][0]
        R = P.readout(U, 'dot')
        H = P.build_heads(R)
        line = []
        for name, prog, expect in allp:
            got, _ = P.run(prog, U, R, H, max_steps=2 * len(refs[name]) + 50,
                           overwrite=True)
            ok = got is not None and abs(got - expect) < 1e-9
            line.append(f'{"." if ok else "x"}{"*" if name == allp[held][0] else ""}')
        print(f'  held out {allp[held][0]:14s} -> d=12 computes {" ".join(line)}   '
              f'(* = the held-out one)')


if __name__ == '__main__':
    main()
