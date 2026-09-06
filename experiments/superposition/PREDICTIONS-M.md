# Pre-registered prediction — magnitude sweep

Written 2026-09-06, before `magnitude_sweep.py` was written. Graded in `RESULTS-M.md`.

## The claim under test

After the projection-rule rerun (`RESULTS-A.md`, "Projection rule"), one LAC-specific
claim remains: the machine barely compresses (12 dense features to 11 dimensions, where
ALTA's programs gave up 20-45% of theirs) because it holds integers up to 5050 and reads
them against an absolute tolerance of 0.5, so any shared direction costs absolute error
proportional to the values it carries. That has been asserted in four writeups and tested
in none.

## Design

Hold the program shape, the trace length, and the opcode mix fixed; vary only the
magnitude of the values. `ladder(c, k)` is a straight-line program: `PUSH c`, then `k`
repetitions of `PUSH c; ADD`, then `HALT`, returning `(k+1)·c`. Its trace has the same
length and the same control flow for every `c`. Each program set is
`{ladder(c, 50), countdown_5, rot_jz_nop}`, the last two supplying loops, jumps and the
stack opcodes at small values, so the only thing that changes across sets is the largest
value the machine holds: 51, 510, 5100 for `c` in {1, 10, 100}.

For each set: harvest, fit the code by continuation from the identity with the
**trajectory-SVD** rule (the rule under which LAC and ALTA both share), walk `d` from 24
down to 4, and record at every width whether all three programs compute exactly, the
blind analyst's recovery, and the Gram and transfer off-diagonals of the twelve dense
features. `d_min` is the smallest width at which all three compute.

## M1

`d_min` falls as `c` falls. **Confirmed** if `d_min(c=1) < d_min(c=100)` by at least two
dimensions. **Refuted** if `d_min` is the same for all three `c`, or if it is ordered the
other way. A one-dimension difference is inconclusive and is reported as such.

## M2

At each set's own `d_min`, interference among the dense features is larger for smaller
`c`. **Confirmed** if the transfer off-diagonal at `d_min` is ordered `c=1 > c=10 > c=100`.
Refuted if it is flat (within 0.05) or ordered the other way.

## What refutation would mean

If `d_min` does not move with magnitude, the dynamic-range explanation joins iteration and
orthogonality as things that were asserted about LAC and did not survive a direct test,
and the gap to ALTA is unexplained. The step count and opcode mix are held fixed by
construction, so a refutation could not be attributed to those.
