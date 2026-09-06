# Addressing sweep

The addressing is the cause. With one-hot equality select in place of the parabolic
numeric key, and everything else about the machine held fixed, the fitted code compresses
to 35 of 48 dense features (ratio 0.73) with survivors in shared directions (transfer
off-diagonal 0.40). The parabolic machine on the same four programs, same objective and
same continuation rule stops at 11 of 12 (ratio 0.92). C1 and C2 in `PREDICTIONS-C.md`
are both confirmed under their registered rules, and the gap between LAC and ALTA's
compiled programs that three earlier experiments failed to explain is a design choice
inside LAC's instruction set.

## Design

`packed_cat.py` is the LAC core-12 machine with its addressing replaced. The parabolic
machine gives memory row `j` the key `(2j, -j²)` and reads address `x` with query `(x, 1)`,
so the dot product is `x² - (j-x)²` and hard argmax lands on `j = x` with a
winner-to-runner-up gap of 1. The categorical machine gives row `j` a one-hot position
vector and the query a one-hot address, so the dot product is 1 on the matching row and 0
elsewhere, the same gap of 1 by a different geometry. The stack reads at `sp`, `sp-1` and
`sp-2` become three one-hot query groups; the address-verification head reads the matched
row's one-hot address back. Opcode, value and indicator features, dispatch, the
overwrite-in-place stack and the four programs are unchanged, and under the identity code
the categorical machine's traces match the parabolic machine's step for step.

`learned_generic.py` is `learned.py` with the attention score computed from each
machine's head specification instead of hardcoded parabolic feature indices, so one
trainer fits both. Its gate on the parabolic machine reproduces the trajectory-rule
result of `RESULTS-A.md`, `d_min` = 11, with the same Gram and transfer values at every
width. Both machines are then fit by continuation from the identity under the
trajectory-SVD rule, 24 down to 4 for the parabolic machine and 60 down to 4 for the
categorical one, with the margin hinge at half the compiled gap and the absolute
tolerance hinge at 0.25 on every rounded scalar.

## Numbers

| machine | NF | dense | used | `d_min` | `d_min` / dense | transfer at `d_min` | Gram at `d_min` | margin / tolerance constraints |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| parabolic | 24 | 12 | 9 | 11 | 0.917 | 0.300 | 0.443 | 13307 / 2458 |
| categorical | 60 | 48 | 44 | 35 | 0.729 | 0.400 | 1.000 | 12124 / 7218 |

The categorical machine computes all four programs at every width from 60 to 35 and
fails at 34 (`x.xx`, the two summation programs and the rotate program lost together).
From width 40 down, 44 of the 48 dense features are live: four one-hot bits the
trajectory never lights are deleted, and the remaining 44 fit in 35 dimensions. The
transfer off-diagonal rises from 0.33 at width 48 to 0.40 at 40 and stays there to the
cliff; the Gram off-diagonal reaches 1.0 by width 40, meaning at least one pair of
survivor rows is parallel. The parabolic machine gains one width and keeps its transfer
at 0.30.

## Grading

**C1 confirmed.** The categorical ratio is 0.729, below the registered confirmation line
of 0.767 and 0.19 below the parabolic 0.917.

**C2 confirmed.** Transfer at `d_min` is 0.400, above 0.3. The compression is not deletion:
four bits are dropped and the other 44 share 35 dimensions.

## The open question in SYNTHESIS.md

`SYNTHESIS.md` asked why LAC's fitted code gains one dimension where ALTA's SUBLEQ gains
26 of 121, after iteration, orthogonality and value magnitude had each been tested and
refuted as the cause. This sweep answers it. A numeric key read by dot product forces the
optimizer to keep the two key features and the two query features on exactly reproduced
axes, because any leakage between them shifts every address score by an amount
proportional to the address, and the margin of 1 does not survive it. One-hot equality
has no such coupling: a bit that reads back at 0.7 instead of 1 still wins by 0.7 against
zeros, and the optimizer can spend that slack on sharing. ALTA's gains came from equality
select, and LAC's absence of gains came from the parabola. Nothing about sequential
computation is implicated.

Two earlier findings read differently in this light. The random-code result
(`RESULTS.md`), where capacity scaled with the square of the largest value, was also an
addressing result: the interference a random code adds to a key scales with the values
sharing its directions. And the magnitude sweep (`RESULTS-M.md`), where the value feature
got its own axis at any magnitude, was the optimizer protecting the one feature the
parabola does not constrain.

## Limits

One seed per machine. The categorical machine drops 1183 of 13307 margin pairs whose
reference gap is zero, a one-hot query for an address no row holds scoring every row 0,
where the read is discarded by the address check anyway. It breaks ties toward the
newest row, which is the parabolic machine's recency term in one-hot form. The blind analyst was not
run on it, since `analyst_sp` fits the parabolic law; recovery is measured by exact
computation only. The two machines' dense counts differ by four times, so the comparison
is by ratio, and a ratio is the quantity `PREDICTIONS-C.md` registered. The categorical
machine's `d_min` of 35 against 44 live features leaves it well short of ALTA's SUBLEQ at
0.79; the parabola explains the difference between 0.92 and 0.73, and what separates
0.73 from 0.79 is not measured here.
