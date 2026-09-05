"""One plot: training step on x, program correctness and ISA recoverability on y,
one panel per arm. Reads results.json."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results.json')))
arms = ['neutral_sgd', 'neutral_adam', 'rival', 'rival_slow']
fig, axes = plt.subplots(1, len(arms), figsize=(18, 4.2), sharey=True)
for ax, arm in zip(axes, arms):
    c = R['arms'][arm]
    x = [max(m['step'], 0.5) for m in c]
    ax.plot(x, [m['oracle'] for m in c], 'o-', color='C3',
            label='computes, append-only stack (1e-6 recency tiebreak)')
    ax.plot(x, [m['oracle_overwrite'] for m in c], 'o-', color='C3', alpha=0.45,
            label='computes, overwrite-in-place stack')
    ax.plot(x, [m['oracle_only']['addr'] for m in c], '^--', color='C2',
            label='computes with only addressing (Q,K,b) trained')
    ax.plot(x, [m['oracle_only']['values'] for m in c], 'v--', color='C4',
            label='computes with only value readouts (V) trained')
    ax.plot(x, [m['oracle_only']['dispatch'] for m in c], 's--', color='C1',
            label='computes with only dispatch trained')
    for tau, ls in ((0.05, ':'), (0.2, '-.'), (0.45, '-')):
        ax.plot(x, [next(r['isa'] for r in m['sweep'] if r['tau'] == tau) for m in c],
                ls, color='C0', label=f'ISA recovered, tau={tau}')
    ax.set_xscale('log')
    ax.set_xlabel('training step (0 plotted at 0.5)')
    ax.set_title(f'{arm} arm')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
axes[0].set_ylabel('fraction')
axes[-1].legend(fontsize=7, loc='lower left')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'decay.png'), dpi=130)
print('wrote decay.png')
