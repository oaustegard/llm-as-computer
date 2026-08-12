"""Plot the two curves the experiment exists to compare: at each residual width,
the fraction of seeds whose machine still computes each program, and the fraction
whose ISA the blind analyst still recovers exactly.

Usage:  python3 plot_curves.py [results.json] [curves.png]
"""
import json
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROG_STYLE = {'countdown_5': ('o', 'values <= 5'), 'rot_jz_nop': ('s', 'values <= 99'),
              'sum_1_to_15': ('^', 'values <= 120'), 'sum_1_to_100': ('v', 'values <= 5050')}


def load(path):
    data = json.load(open(path))
    return data['meta'], data['rows']


def fractions(rows, arm, key):
    rs = [r for r in rows if r['arm'] == arm]
    ds = sorted({r['d'] for r in rs})
    return ds, [float(np.mean([key(r) for r in rs if r['d'] == d])) for d in ds]


def main(path='results.json', out='curves.png'):
    meta, rows = load(path)
    arms = [a for a in ('pinv', 'dot', 'scaled') if any(r['arm'] == a for r in rows)]
    fig, axes = plt.subplots(1, len(arms), figsize=(5 * len(arms), 4.2), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, arm in zip(axes, arms):
        for name, (mark, lab) in PROG_STYLE.items():
            ds, fr = fractions(rows, arm, lambda r, n=name: r['programs'][n]['ok'])
            ax.plot(ds, fr, marker=mark, ms=4, lw=1.2, alpha=.85, label=f'computes {name} ({lab})')
        for tag, ls in (('ideal', '--'), ('self', ':')):
            ds, fr = fractions(rows, arm, lambda r, t=tag: r[f'recover_{t}']['score'] == 12)
            ax.plot(ds, fr, ls, color='k', lw=2, alpha=.8 if tag == 'ideal' else .5,
                    label=f'recovers 12/12 ({tag} artifact)')
        ax.set_xscale('log', base=2)
        ax.set_xlabel('residual width d')
        ax.set_title(f'{arm} readout' + ('  [post-hoc arm]' if arm == 'scaled' else ''))
        ax.grid(alpha=.25)
        ax.axvline(meta['n_features'], color='gray', lw=.8)
        ax.text(meta['n_features'], 1.02, f' d = n_features = {meta["n_features"]}',
                fontsize=7, color='gray')
    axes[0].set_ylabel(f'fraction of {meta["seeds"]} seeds')
    axes[0].set_ylim(-.04, 1.08)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=3, loc='lower center',
               bbox_to_anchor=(.5, -.01), frameon=False)
    fig.suptitle('Compiled superposition: computing vs recovering the ISA', y=.99)
    fig.tight_layout(rect=(0, .13, 1, 1))
    fig.savefig(out, dpi=160)
    print('wrote', out)


if __name__ == '__main__':
    main(*(sys.argv[1:] or []))
