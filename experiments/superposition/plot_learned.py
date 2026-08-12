"""Plot the learned-code result against the random codes, on the same machine.

Usage:  python3 plot_learned.py [learned_results.json] [learned_curves.png]
"""
import json
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROGS = ['sum_1_to_15', 'countdown_5', 'rot_jz_nop', 'sum_1_to_100']


def main(path='learned_results.json', out='learned_curves.png'):
    rows = json.load(open(path))['rows']
    ds = [r['d'] for r in rows]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2))

    allfour = [float(all(p['ok'] for p in r['programs'].values())) for r in rows]
    rec = [r['recover_ideal'] / 12 for r in rows]
    ind = [r['indicator_norm_median'] for r in rows]
    ax.plot(ds, allfour, 'o-', lw=2, label='computes all four programs')
    ax.plot(ds, rec, 's--', lw=2, label='blind analyst recovers 12/12')
    ax.axvline(12, color='crimson', lw=1)
    ax.text(12.4, 1.03, 'd = 12 = dense features', color='crimson', fontsize=8)
    ax.set_xlabel('residual width d')
    ax.set_ylabel('fraction succeeding')
    ax.set_ylim(-.05, 1.12)
    ax.set_title('Learned code: a cliff, not a slope')
    ax.grid(alpha=.25)
    ax2 = ax.twinx()
    ax2.plot(ds, ind, '^:', color='gray', lw=1.4,
             label='median opcode-indicator norm (right axis)')
    ax2.set_ylabel('indicator norm', color='gray')
    ax2.tick_params(axis='y', colors='gray')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc='center left')

    try:
        rnd = json.load(open('random_overwrite.json'))
    except FileNotFoundError:
        rnd = []
    for arm, style in (('dot', 'o-'), ('pinv', 's-')):
        pts = [(r['d'], float(all(f > 0.5 for f in r['frac']))) for r in rnd
               if r['arm'] == arm]
        if pts:
            bx.plot([p[0] for p in pts], [p[1] for p in pts], style, lw=1.8,
                    label=f'random {arm}')
    bx.plot([min(ds), max(ds)], [1, 1], color='none')
    bx.axvline(12, color='crimson', lw=2, label='learned code works from d = 12')
    bx.set_xscale('log', base=2)
    bx.set_xlabel('residual width d')
    bx.set_ylabel('all four programs compute')
    bx.set_ylim(-.05, 1.15)
    bx.set_title('Same machine, code chosen three ways')
    bx.grid(alpha=.25)
    bx.legend(fontsize=7.5, loc='center right')

    fig.suptitle('Learned residual code for the LAC core-12 (overwrite-in-place stack)',
                 y=.99)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print('wrote', out)


if __name__ == '__main__':
    main(*(sys.argv[1:] or []))
