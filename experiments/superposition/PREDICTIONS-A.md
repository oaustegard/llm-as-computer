# Pre-registered predictions — learned code (Version A)

Written 2026-08-12, **before** the training code was written. Graded in `RESULTS-A.md`.

## What changes from the first sweep

The first sweep drew the code at random and froze it, which `RESULTS.md` flags as its
main weakness: a random code is a strictly weaker compressor than Tracr's learned one,
so every measured `d` is an upper bound rather than a capacity limit. This run replaces
the random code with one fit by gradient descent, following Tracr §5's setup — freeze
every weight in the machine, train only the embedding matrix, keep the readout tied to
it (`R = U`, Tracr's shared-`W` convention, which is the `dot` arm with a learned `U`).

The objective is teacher-forced and needs no unrolling: harvest, from the reference
machine's own trajectory, every decision the machine depends on, and require the
compressed code to keep making them.

- **Margin constraints.** At each step, every attention head picks a winner by hard
  argmax. For each (step, head) keep the winner and its hardest competitors, and hinge
  on `score(winner) - score(other) >= 0.5` — half the reference gap of 1.
- **Tolerance constraints.** Every scalar the machine rounds — opcode, immediate, the
  three stack reads, the address check — must land within 0.25 of its true value, half
  the quantizer's 0.5. Hinged at the tolerance rather than minimized, deliberately: an
  MSE would pour capacity into `value`, whose magnitudes reach 5050, and reproduce the
  failure the post-hoc `scaled` arm already demonstrated.

Evaluation is unchanged — the same four oracle programs, the same blind analyst, the
same scoring. Only the code changes.

## Predictions

**A1. The learned code works far below `d = 24`, and the reason is disjointness, not
cleverness.** Two structures the random code got no credit for. First, the twelve
opcode indicators are one-hot: each fires on about a twelfth of one ROM and never
co-occurs with another, which is exactly the sparsity Elhage et al. say superposition
exploits. Second, and stronger, the row *types* are disjoint — a ROM row never carries
`stack_k0/k1`, a stack row never carries `prog_k0/k1`, a query carries neither. Counting
dense features that are ever simultaneously non-zero in one vector gives 6 for a ROM
row, 5 for a stack row, 4 for a query. So predict the machine still computes the
small-value programs somewhere around **`d` = 6 to 8**, against 24 features.

**A2. The learned code discards the opcode indicators entirely.** No compiled head reads
`op_*`; they exist in the artifact only because the axis-aligned analyst detected the
opcode via a one-hot block. Neither the machine nor the current analyst consumes them.
A code fit to the machine's decisions should therefore zero them out — the same
behaviour Tracr reports for `tokens:a/b/c`, which its `W` discarded as unnecessary.

**A3. `value` is the wall, and learning does not move it.** `sum_1_to_100` should fail
at every `d` below the point where `5050 / sqrt(d)` clears the 0.25 tolerance, because
the tolerance is absolute and no allocation of directions changes that. If A1 and A3
both hold, the finding is that a learned code buys back the whole *sparse* half of the
problem and none of the *dynamic-range* half.

**A4. The P1 gap narrows.** Under a random code the analyst kept recovering the ISA at
widths where the machine had stopped computing. A code trained to preserve exactly the
margins the machine needs also preserves the structure the analyst reads — the parabolic
law and the scalar readouts are the same quantities. So expect recovery and computation
to rise together here, rather than recovery running ahead.

## Grading rules (fixed now)

- **A1 confirmed** if the smallest `d` at which a majority of seeds compute
  `countdown_5` is at most 8, and if that `d` is below the count of dense features (12).
- **A2 confirmed** if the learned embedding norms of the twelve `op_*` features are at
  least an order of magnitude below the median across the other twelve.
- **A3 confirmed** if `sum_1_to_100`'s compute threshold under the learned code is no
  better than the random code's within the tested grid.
- **A4 confirmed** if the gap between the recovery threshold and the compute threshold
  is smaller under the learned code than under `dot`.

## Leakage, declared in advance

The code is trained on the same four programs it is evaluated on. That is the most
generous setting for the learned arm and matches Tracr, which also fits `W` to the task
it then measures. The rule, fixed now so it is not chosen after seeing results: **if the
learned arm succeeds at some `d`, re-run it with one program held out of training before
claiming the number.** If it fails even with the leakage, the failure stands without a
held-out run.
