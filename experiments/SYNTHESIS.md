# LAC under four perturbations

Trained transformers superpose: they hold more features than dimensions, and
features that rarely co-occur share a direction. A stack machine compiled by
hand into transformer weights did not, under the fitting procedure used. Given
a free choice of code and twelve spare dimensions, gradient descent deleted the
features nothing read and kept every survivor exactly orthogonal, with a largest
Gram off-diagonal of 0.00e+00. Remove one more dimension, so that sharing is
forced, and every program stops computing.

That is one of four results over LAC since August, and a fifth run on a second
compiler (below) qualifies it: the orthogonal outcome reproduces under the same
continuation rule and disappears under a different one. What the four results
say together is what this kind of computation needs: exact values, no gradient,
and, on the evidence so far, more room than a trained model gives its features.

## The machine

LAC (`oaustegard/llm-as-computer`) is a stack machine compiled into transformer
weights rather than trained into them: twelve opcodes, six attention heads, a
51-dimensional residual stream carrying 24 semantic features on their own axes,
and a memory addressed by content. Each memory row carries the key `(2j, -j²)`
for its address `j`, a construction taken from Percepta's transformer-vm (Tzamos
et al., March 2026). The score for query `(x, 1)` is `2xj - j²`, which is
`x² - (j-x)²`, so the hard argmax lands on `j = x` and the gap between winner and
runner-up is exactly 1 for integer addresses. Every weight is
known and every instruction has an exact meaning. The four oracle programs
return 5050, 120, 0 and 1; the largest integer any of them holds is 5050, read
back with an absolute tolerance of 0.5.

Oskar asked the four questions below. Muninn ran the first experiment and wrote
the specs for the other three on claude.ai; Claude Code on the Web ran
experiments two through four.

## Anonymized weights

The first experiment (2026-08-11) permuted the compiled machine's weight
matrices and a handful of captured activations, stripped every name, and handed
the result to an analyst script with no access to the source and no opcode
table. The answer key was the compiler's own instruction set.

The analyst recovered all twelve opcodes exactly, both conditional jumps and
the three-cell rotate included. It also recovered the addressing scheme and the
alignment between opcode ids and dispatch rows, then replayed the recovered
machine against the captured memory snapshots and reproduced every one. The
limit it hit was identifiability. 27 of the 51 dimensions are never written and
never read, so no analyst can tell them apart, and no recovered opcode carries a
name, only its effect on the stack and the instruction pointer.

## Random directions

The second experiment (2026-08-12) removed the axis alignment that made the
first one easy. Each of the 24 features got a random unit vector in `d`
dimensions, a row embedded as the sum of value times direction, read back by
dot product against the same vector. Sweeping `d` from 8 to 16384 over 760
configurations at 20 seeds each, two questions at every width: does the machine
compute, and can the blind analyst recover the ISA.

The machine stops computing on the two summation programs at every width
tested, while at `d = 16384` the analyst still returns 12 of 12 opcodes in 55%
of seeds. All four programs use the same 24 features, so what separates them is
the size of the numbers they hold. `countdown_5` (values ≤ 5) computes in 55%
of seeds at best, `rot_jz_nop` (99) in 10%, `sum_1_to_15` (120) and
`sum_1_to_100` (5050) in none, the last diverging at step 0. Readout error
under a random code runs about `‖y‖ / √d` against a fixed tolerance of 0.5, so
the width required scales as the square of the largest magnitude, which puts
5050 at roughly 10⁸ dimensions.

The analyst reads the ISA out of activations a *working* machine produced, which
narrows the claim. Fed the packed machine's own traces, recovery peaks
at 0.20 and falls to zero at large `d`, because those traces record a
malfunction and no correct ISA reproduces them.

## A fitted code

The third experiment (2026-08-12) replaced the random code with one fit by
gradient descent, following Tracr §5: freeze every weight in the machine and
train only the embedding, by continuation from the identity code at `d = 24`
downward one dimension at a time.

The smallest working width is 12. At and above it the twelve dense features are
exactly orthogonal, and the twelve opcode indicators, which nothing in the
machine reads, sit at exactly zero norm. The compression is deletion throughout.
At `d = 11` every program fails, the analyst recovers nothing, and the Gram
off-diagonals jump to 13.8. A code trained on three programs still runs the
fourth, so this is feature selection rather than fitting to values.

This corrects the previous experiment. `sum_1_to_100` computes at `d = 12`
under a learned code, having computed at no width up to 16384 under a random
one. Dynamic range walls off random codes specifically: an optimizer can hand
`value` its own axis, which rescaling random directions cannot do.

The no-sharing half of this result did not survive a second compiler
(`oaustegard/experiments`, `alta-superposition/`, 2026-09-06). The same frozen
weights, hinge objective and continuation were run on three ALTA programs
(Shaw et al. 2024): SUBLEQ, a looped one-instruction computer; a looped running
parity; and a feed-forward parity. Under the continuation rule LAC used, which
projects onto the code's own top singular subspace, all three cliff at exactly
their live dimension count with orthogonal survivors, the LAC pattern. Under a
rule that projects onto the code's image of the visited states, all three
compress below live with shared directions: SUBLEQ to 95 of 121 (interference
0.47), sequential parity to 17 of 31 (0.46), feed-forward parity to 5 of 7
(0.38). Looped and feed-forward programs behave alike in both arms, so
iteration is not what forbade sharing. The LAC run has not yet been repeated
with the second rule. The remaining candidate specific to LAC is its dynamic
range, values to 5050 read against a tolerance of 0.5, which ALTA's SUBLEQ,
holding values in [-16, 16] as one-hot buckets, does not test.

## Continued training

The fourth experiment (2026-09-05) started a torch model at the compiled
weights and trained it with AdamW at 1e-3 on targets from a rival ISA, with no
circuit-preservation term. Checkpoints out to 3000 steps, both questions at
each: does it compute, is the ISA readable.

Over three seeds, correctness dies below an L2 distance of 0.1 per component
from the compiled point, every oracle program stopped within 20 to 200 steps.
Full 12-of-12 recovery holds to L2 about 1.0 (step 200 or 300), and the first
opcode is lost at L2 about 1.5 to 2.4 (step 300 or 500). At lr 1e-4 the same
transitions land at steps 2000 and 3000 at the same L2 values, so the threshold
is a distance in weight space rather than a step count.

Dispatch structure degrades before addressing structure: at 3000 steps the
addressing score is 0.83 against a 0.17 floor, while dispatch sits at its 0.27
floor. As behaviour the ordering does not hold. A 1e-4 leak of the `value`
column into a key adds 0.5 to a score when the value is 5050, which flips the
argmax, so addressing drift kills three of four programs by step 5 as well.

At the compiled point the loss is 4.6e-11, and a single AdamW step at 1e-3 takes it to 8.2e6 with one
program already failing: Adam normalizes the gradient, so a near-zero loss still
buys a full step on every parameter carrying any. Plain SGD at 1e-8 moves
nothing in 3000 steps. A drift control on a compiled machine has to name its
optimizer.

The same step size decides whether a preservation loss can hold the machine in
place. With the machine's own loss added to a competing task, AdamW at 1e-3
fails exactly as without it, and the preservation term never re-converges. At
1e-5 all four programs compute for 3000 steps on every seed while the competing
task is still learned; at 1e-4 three of four. Adam settles in a ball of radius
about the learning rate around any optimum, and an error of that size on a
coefficient multiplying 5050 has to stay under the read tolerance of 0.5.
InterpBench and Tracr-Injection hold circuits over small-magnitude features and
never meet that constraint; an integer machine does.

## Prior art

Tracr (Lindner et al., arXiv 2301.05062) compiles RASP programs to transformers
and studies residual-stream compression in §5, including learned projections
that discard unneeded features; the learned-code experiment is that setup
applied to a stateful machine. Thurnherr and Riesen (arXiv 2410.00061) trained
a seq2seq decompiler on Tracr weight/RASP pairs, so recovering a program from
compiled weights is established; the blind-recovery run differs in being
analytic and exact over a permuted basis. InterpBench/SIIT (arXiv 2407.14494)
and Tracr-Injection (Vergara-Browne and Soto, arXiv 2505.10719) both train
around a compiled circuit with a loss holding it in place and report the
endpoint; the training monitor drops that term and measures the path between
endpoints. Giannou et al. (arXiv 2301.13196) build looped transformers as
programmable computers, and NALU (Trask et al. 2018) and its successors give
arithmetic units that are exact once the weights are right, leaving open which
inputs feed the operation.

## Four perturbations

| experiment | perturbed | survived | died |
|---|---|---|---|
| blind recovery (2026-08-11) | basis permuted, names stripped | all 12 opcodes, addressing, replay | 27 dead dimensions, all names |
| random code (2026-08-12) | 24 features on random unit vectors | ISA recovery to `d` = 16384 | computation at every width |
| learned code (2026-08-12) | code fit by gradient descent | computation and recovery to `d` = 12 | everything at `d` = 11 |
| training monitor (2026-09-05) | AdamW from the compiled point | ISA readability to L2 ≈ 1.0; correctness under a preservation loss at lr ≤ 1e-5 | correctness below L2 0.1 at lr 1e-3 |
| ALTA replication (2026-09-06) | same code fit on a second compiler, two projection rules | shared directions on every program under trajectory SVD (95/121, 17/31, 5/7) | the no-sharing claim as a property of iteration; it holds only under code SVD |

Across all four, the quantity that ordered the outcomes was the same one: 5050,
the largest integer the machine holds, against a read tolerance of 0.5.
