"""Emit the results tables that RESULTS.md quotes, so the numbers in the writeup
are generated rather than transcribed.

Usage:  python3 tables.py [results.json] > results_tables.md
"""
import collections
import json
import sys

import numpy as np

PROGS = [('countdown_5', 5), ('rot_jz_nop', 99), ('sum_1_to_15', 120),
         ('sum_1_to_100', 5050)]


def frac(rows, arm, d, key):
    g = [r for r in rows if r['arm'] == arm and r['d'] == d]
    return float(np.mean([key(r) for r in g])) if g else float('nan')


def main(path='results.json'):
    data = json.load(open(path))
    meta, rows = data['meta'], data['rows']
    arms = [a for a in ('pinv', 'dot', 'scaled') if any(r['arm'] == a for r in rows)]
    n = meta['seeds']

    print(f'# Results tables\n\n{len(rows)} configurations, {n} seeds each, '
          f'{meta["n_features"]} semantic features.\n')

    print('## Success fraction by residual width\n')
    print('`computes` = the program returned its exact expected value. '
          '`recovers` = the blind analyst scored 12/12 on the ISA.\n')
    for arm in arms:
        ds = sorted({r['d'] for r in rows if r['arm'] == arm})
        print(f'### {arm} readout' + ('  (post-hoc arm)' if arm == 'scaled' else '') + '\n')
        print('| d | ' + ' | '.join(f'{p} (max {m})' for p, m in PROGS) +
              ' | recovers (ideal) | recovers (self) |')
        print('|--:|' + '--:|' * (len(PROGS) + 2))
        for d in ds:
            cells = [f'{frac(rows, arm, d, lambda r, p=p: r["programs"][p]["ok"]):.2f}'
                     for p, _ in PROGS]
            for tag in ('ideal', 'self'):
                cells.append(f'{frac(rows, arm, d, lambda r, t=tag: r[f"recover_{t}"]["score"] == 12):.2f}')
            print(f'| {d} | ' + ' | '.join(cells) + ' |')
        print()

    print('## Where the machine first parted company with the reference\n')
    print('Counted over failed runs. `tiebreak:<head>` means the argmax moved to '
          'another row at the *same* address, so only the `1e-6` write-order '
          'tiebreak separated them; `argmax:<head>` means it moved to a different '
          'address.\n')
    print('| arm | ' + ' | '.join(k for k in ('argmax:prog_op', 'opcode_decode',
                                              'tiebreak:stack_*', 'value_drift',
                                              'argmax:stack_*')) + ' |')
    print('|---|' + '--:|' * 5)
    for arm in arms:
        c = collections.Counter()
        for r in rows:
            if r['arm'] != arm:
                continue
            for p in r['programs'].values():
                if not p['ok'] and p['div_kind']:
                    k = p['div_kind']
                    k = 'tiebreak:stack_*' if k.startswith('tiebreak') else \
                        'argmax:stack_*' if k.startswith('argmax:stack') else k
                    c[k] += 1
        tot = max(sum(c.values()), 1)
        print(f'| {arm} | ' + ' | '.join(
            f'{100 * c[k] / tot:.0f}%' for k in ('argmax:prog_op', 'opcode_decode',
                                                 'tiebreak:stack_*', 'value_drift',
                                                 'argmax:stack_*')) + ' |')
    print()

    print('## Failure mode by program, dot arm at d >= 1024\n')
    print('The regime where interference is small enough that addressing mostly '
          'survives.\n')
    print('| program | max value | dominant first divergence | median step |')
    print('|---|--:|---|--:|')
    for p, mx in PROGS:
        sub = [r['programs'][p] for r in rows
               if r['arm'] == 'dot' and r['d'] >= 1024 and not r['programs'][p]['ok']]
        c = collections.Counter(x['div_kind'] for x in sub if x['div_kind'])
        steps = [x['div_step'] for x in sub if x['div_step'] is not None]
        top = c.most_common(1)
        print(f'| {p} | {mx} | ' +
              (f'{top[0][0]} ({100 * top[0][1] / max(sum(c.values()), 1):.0f}%)' if top else '-') +
              f' | {int(np.median(steps)) if steps else "-"} |')
    print()

    print('## Per-feature scales used by the post-hoc scaled arm\n')
    sc = meta.get('scales', {})
    print('| feature | typical magnitude |')
    print('|---|--:|')
    for k, v in sc.items():
        print(f'| `{k}` | {v:.2f} |')


if __name__ == '__main__':
    main(*(sys.argv[1:] or []))
