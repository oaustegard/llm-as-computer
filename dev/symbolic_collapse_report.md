# LAC program catalog — symbolic collapse report

Snapshot of `python symbolic_programs_catalog.py` run with eml-sr on
`PYTHONPATH`. Regenerate with:

```bash
PYTHONPATH=../eml-sr python symbolic_programs_catalog.py \
    > dev/symbolic_collapse_report.md
```

This is the issue #65 demo: every branchless program in `programs.py`
collapses to a single polynomial (the `poly` column), which then compiles
to a pure-EML tree (`eml size`/`eml depth`). Non-branchless programs are
classified by their first blocker so the report splits cleanly between
what the symbolic executor handles natively and what it doesn't.

Invariants pinned by `test_symbolic_programs_catalog.py`:

- `dup_add_chain_x4`: **9 heads → 1 monomial**, top = `16*x0` — matches
  the issue's headline collapse claim.
- `add_dup_add`: **5 heads → 2 monomials**, top = `2*x0 + 2*x1` — second
  PoC from the issue.
- Every collapsed row satisfies `NumPy top == Poly.eval_at == eval_eml`.
- Multiplication's EML cost (35 nodes, depth 8) matches Table 4 of
  Odrzywolek 2026, so a drift here would flag an upstream change.

---

_15 collapsed, 4 blocked-by-branch, 7 blocked-by-opcode (total 26)._

## Collapsed (branchless, polynomial-closed)

| Program | k heads | # mono | poly | eml size | eml depth | match |
|---|---:|---:|---|---:|---:|:-:|
| `basic_add` | 3 | 2 | `x0 + x1` | 21 | 6 | ✓ |
| `push_halt` | 1 | 1 | `x0` | 1 | 0 | ✓ |
| `push_pop` | 3 | 1 | `x0` | 1 | 0 | ✓ |
| `dup_add` | 3 | 1 | `2*x0` | 35 | 8 | ✓ |
| `multi_add` | 5 | 3 | `x0 + x1 + x2` | 41 | 10 | ✓ |
| `stack_depth` | 5 | 1 | `x0` | 1 | 0 | ✓ |
| `overwrite` | 3 | 1 | `x1` | 1 | 0 | ✓ |
| `complex` | 6 | 2 | `2*x1 + 2*x2` | 89 | 12 | ✓ |
| `many_pushes` | 19 | 10 | `x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9` | 181 | 38 | ✓ |
| `alternating` | 7 | 4 | `x0 + x1 + x2 + x3` | 61 | 14 | ✓ |
| `native_multiply(3,7)` | 3 | 1 | `x0*x1` | 35 | 8 | ✓ |
| `square_via_dupmul(9)` | 3 | 1 | `x0^2` | 43 | 12 | ✓ |
| `sum_of_squares(3,4)` | 7 | 2 | `x0^2 + x1^2` | 105 | 16 | ✓ |
| `dup_add_chain_x4` | 9 | 1 | `16*x0` | 35 | 8 | ✓ |
| `add_dup_add` | 5 | 2 | `2*x0 + 2*x1` | 89 | 12 | ✓ |

## Blocked (out of symbolic-executor scope)

| Program | reason | blocker |
|---|---|---|
| `native_divmod(2,7)` | non-polynomial op | `DIV_S` |
| `native_clz(16)` | non-polynomial op | `CLZ` |
| `native_abs_unary(-3)` | non-polynomial op | `ABS` |
| `native_neg(5)` | non-polynomial op | `NEG` |
| `compare_lt_s(3,5)` | non-polynomial op | `LT_S` |
| `bitwise_and(12,10)` | non-polynomial op | `AND` |
| `fibonacci(5)` | control flow | `JNZ` |
| `factorial(4)` | control flow | `JZ` |
| `is_even(6)` | control flow | `JZ` |
| `power_of_2(4)` | control flow | `JZ` |
| `native_max(3,5)` | non-polynomial op | `GT_S` |

