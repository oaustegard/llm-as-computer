# Pre-registered prediction — addressing sweep

Written 2026-09-06, after `packed_cat.py`, `learned_generic.py` and
`addressing_sweep.py` passed their equivalence gates and before any width below the
gates was fitted. Graded in `RESULTS-C.md`.

## The claim under test

LAC's fitted residual code gains one dimension: 12 dense features compress to
`d_min` = 11 under the trajectory-SVD rule (`learned_results_data.json`), a ratio of
11/12 = 0.92. ALTA's compiled programs give up 20-45% of theirs — SUBLEQ 95 of 121 =
0.79, parity 17 of 31 and 5 of 7. Three explanations for that gap have been offered
and tested: iteration fell to the ALTA replication, orthogonality to the projection
rule (`RESULTS-A.md`, "Projection rule"), and value magnitude to the magnitude sweep
(`RESULTS-M.md`). The remaining structural difference is how a memory row is selected.

LAC selects by a numeric dot product over parabolic keys: row `j` carries
`(2j, -j^2)`, a query for address `x` carries `(x, 1)`, the score is
`-(j - x)^2 + x^2`, and the winner beats the runner-up by exactly 1. ALTA selects by
one-hot equality. A numeric key is one scalar that has to be read accurately at every
address the machine visits; a one-hot key is a bit that only has to beat zero.

## The delta

`packed_cat.py` is ALTA's select transplanted into LAC with everything else held
fixed: same twelve opcodes, same four oracle programs, same overwrite-in-place stack,
same dispatch tensors, same value and opcode features, same objective (margin hinge at
MARGIN = 0.5, absolute tolerance at TOL = 0.25), same continuation from the identity,
same trajectory-SVD projection rule, same tied `dot` readout, same optimizer and
iteration budget. `learned_generic.py` reads the attention score off `HEAD_SPEC`
instead of hardcoding the two parabolic key rows, so both machines are fitted by the
same trainer.

Only the addressing differs. The parabolic `prog_k0/prog_k1` become 11 one-hot
`pos_*` bits, `stack_k0/stack_k1` become 5 one-hot `addr_*` bits, the scalar `ip`
query becomes 11 `ipq_*` bits, and the scalar `sp` query with its three `b_Q` offsets
(0, -1, -2) becomes three one-hot groups `spq0_*`, `spq1_*`, `spq2_*`. The
winner-to-runner-up gap is still exactly 1. The dense feature count rises from 12 to
48, which is why the comparison is the *ratio* `d_min / dense`, not `d_min`.

## C1 — the compression ratio falls

The categorical machine's `d_min / dense` is at least 0.15 below the parabolic
machine's 11/12 = 0.92.

**Confirmed** if the categorical ratio is at most 0.767 (with 48 dense features,
`d_min` at most 36).
**Refuted** if it is 0.867 or higher, that is within 0.05 of the parabolic ratio, or
higher than it.
A ratio strictly between 0.767 and 0.867 is inconclusive and is reported as such.

## C2 — the survivors share directions

At the categorical machine's own `d_min`, the transfer off-diagonal among the dense
features exceeds 0.3 — genuine superposition, not deletion.

**Confirmed** if `transfer_off_dense` at `d_min` is above 0.3.
**Refuted** if it is at or below 0.3.

C2 is not implied by C1. A code could shed dimensions purely by deleting one-hot bits
the trajectory never lights, which would compress without superposing, and the live
count reported per width is what separates the two readings.

## What each outcome would mean

If C1 confirms, the addressing is the thing: LAC's poor compression is a consequence of
selecting memory by a numeric key read against an absolute tolerance, and ALTA's gains
come from selecting by equality. The LAC-versus-ALTA gap would then be explained by a
design choice inside LAC's ISA rather than by anything about sequential computation.

If C1 refutes, the fourth and last candidate this line has offered is gone with the
other three, and the gap stands unexplained by iteration, orthogonality, magnitude or
addressing.
