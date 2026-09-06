# Magnitude sweep over the fitted code

Three program sets identical in shape, trace length and opcode mix, differing only in
the largest value the machine holds (51, 510, 5100), produce the same fitted code to
thirteen decimal places at every width: the same smallest working width of 19, the same
interference of 0.270, the same failure below it. Predictions M1 and M2 in
`PREDICTIONS-M.md` are refuted, and the dynamic-range explanation for LAC's small
compression gain goes with them.

## Design

`ladder(c, 50)` pushes `c` and adds it fifty more times, a straight-line program of 102
steps whose values are `c, 2c, ..., 51c`. Each set is `{ladder(c), countdown_5,
rot_jz_nop}` for `c` in {1, 10, 100}, so the loops, jumps and stack opcodes come from the
same two small-valued programs in every set and only `ladder`'s values change. Each set
yields 1489 margin constraints and 510 tolerance constraints, the value constraints among
them reaching 99, 510 and 5100. The code is fit by continuation from the identity under
the trajectory-SVD rule, 24 down to 4, and at each width all three programs must compute
exactly.

## Numbers

| `c` | largest value | `d_min` | transfer off-diagonal at `d_min` | Gram off-diagonal at `d_min` |
|--:|--:|--:|--:|--:|
| 1 | 51 | 19 | 0.270 | 0.369 |
| 10 | 510 | 19 | 0.270 | 0.369 |
| 100 | 5100 | 19 | 0.270 | 0.369 |

The three transfer values agree to 1e-13. Below `d_min` the runs diverge, since the
optimizer is then trading violated constraints whose penalties do scale with the values,
and none of those widths computes in any set.

## Grading

**M1 refuted.** `d_min` is 19 for every `c`; the grading rule called a flat result a
refutation.

**M2 refuted.** Interference at `d_min` is flat to thirteen decimals.

## Mechanism

The tolerance hinge is absolute, `relu(|err| / 0.25 - 1)`, and the value constraints
carry the true magnitudes, so the objective could have seen them. It did not, because at
every working width the optimizer keeps the `value` feature's readout exact: its
tolerance terms are zero at `c` = 1 and at `c` = 100 alike. Every other constraint, the
attention margins over the parabolic keys and the opcode and address readouts, involves
values that do not change with `c`. `RESULTS-A.md` reported the same mechanism from the
other side, where `sum_1_to_100` computed under a learned code at a width where no random
code could hold it: an optimizer hands `value` its own axis. The consequence not drawn
there is that once it has, the magnitude on that axis costs nothing, and the width the
machine needs is set entirely by the other eleven dense features.

## Remaining candidate

Dynamic range explained why random codes fail (`RESULTS.md`, capacity proportional to
the square of the largest magnitude, confirmed with the post-hoc scaled arm). It does not
explain why the learned code gains one dimension on the four-program set where ALTA's
SUBLEQ gains 26 of 121. Of the three explanations this line has offered for that gap,
iteration fell to the ALTA replication, orthogonality to the projection rule, and
magnitude to this sweep. The gap stands unexplained. The untested candidate is the
addressing itself: LAC selects memory rows by a numeric dot product with a fixed
winner-to-runner-up gap of 1 on every read, where ALTA selects by one-hot equality. A
sweep that varies the key gap or swaps the parabolic head for a categorical one would
test it.

## Limits

One seed; three magnitudes; one program family for the varying arm. The `d_min` of 19
here is not comparable to the 11 of the four-program set, since the constraint set is
different. The result is a statement about this trainer's objective, and any trainer with
an absolute tolerance and a private axis available for `value` should reproduce it.
