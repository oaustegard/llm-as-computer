# LAC program catalog — symbolic collapse report

_28 collapsed | 4 guarded | 5 unrolled | 4 closed-form | 0 loop-symbolic | 2 blocked-by-opcode (total 43)._

**Reading the status columns.** _Collapsed_ rows are straight-line
programs that reduce to a single polynomial (the issue-#65 claim).
_Guarded_ rows contain finite conditionals (JZ/JNZ on symbolic
inputs) and reduce to a `GuardedPoly` — a partitioned case table
whose cases together cover the domain. _Unrolled_ rows contain
bounded loops with concrete trip counts: the executor runs them
in `input_mode="concrete"` (every PUSH is specialised to its
literal arg) so the loop unrolls by execution rather than by
invariant inference. "Unrolled at n=5" is therefore a claim
about a specific input, **not** a symbolic proof over all n.

_eml-sr not on PYTHONPATH — eml tree columns show `–`._

## Collapsed (branchless, polynomial-closed)

| Program | k heads | # mono | poly | eml size | eml depth | match |
|---|---:|---:|---|---:|---:|:-:|
| `basic_add` | 3 | 2 | `x0 + x1` | – | – | ✓ |
| `push_halt` | 1 | 1 | `x0` | – | – | ✓ |
| `push_pop` | 3 | 1 | `x0` | – | – | ✓ |
| `dup_add` | 3 | 1 | `2*x0` | – | – | ✓ |
| `multi_add` | 5 | 3 | `x0 + x1 + x2` | – | – | ✓ |
| `stack_depth` | 5 | 1 | `x0` | – | – | ✓ |
| `overwrite` | 3 | 1 | `x2` | – | – | ✓ |
| `complex` | 6 | 2 | `2*x1 + 2*x2` | – | – | ✓ |
| `many_pushes` | 19 | 10 | `x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9` | – | – | ✓ |
| `alternating` | 7 | 4 | `x0 + x1 + x3 + x5` | – | – | ✓ |
| `native_multiply(3,7)` | 3 | 1 | `x0*x1` | – | – | ✓ |
| `square_via_dupmul(9)` | 3 | 1 | `x0^2` | – | – | ✓ |
| `sum_of_squares(3,4)` | 7 | 2 | `x0^2 + x3^2` | – | – | ✓ |
| `dup_add_chain_x4` | 9 | 1 | `16*x0` | – | – | ✓ |
| `add_dup_add` | 5 | 2 | `2*x0 + 2*x1` | – | – | ✓ |
| `native_divmod(2,7)` | 3 | 2 | `(x0) /ₜ (x1)` | – | – | ✓ |
| `native_remainder(2,7)` | 3 | 2 | `(x0) %ₜ (x1)` | – | – | ✓ |
| `compare_lt_s(3,5)` | 3 | 2 | `[x0 - x1 < 0]` | – | – | ✓ |
| `compare_eqz(0)` | 2 | 1 | `[x0 == 0]` | – | – | ✓ |
| `bitwise_and(12,10)` | 3 | 3 | `(x0 & x1)` | – | – | ✓ |
| `bitwise_or(12,10)` | 3 | 3 | `(x0 | x1)` | – | – | ✓ |
| `bitwise_xor(12,10)` | 3 | 3 | `(x0 ^ x1)` | – | – | ✓ |
| `native_clz(16)` | 2 | 2 | `clz(x0)` | – | – | ✓ |
| `native_ctz(8)` | 2 | 2 | `ctz(x0)` | – | – | ✓ |
| `native_popcnt(13)` | 2 | 2 | `popcnt(x0)` | – | – | ✓ |
| `bit_extract(5,0)` | 5 | 5 | `((x0 >>ᵤ x1) & x3)` | – | – | ✓ |
| `log2_floor(8)` | 5 | 4 | `(x2 - clz(x0))` | – | – | ✓ |
| `is_power_of_2(8)` | 4 | 4 | `[(popcnt(x0) - x2) == 0]` | – | – | ✓ |

## Collapsed (guarded — finite conditionals)

Each case is `guards ⇒ value_poly`. Guarded dispatch has two
EML costs: the **value** trees (one per case's `value_poly`)
and the **guard** trees (one per `Guard` in every case's
conjunction). Both are reported separately rather than rolled
together — so the "what does one execution cost?" number and
the "what does it take to realise the whole case table?"
number stay distinguishable. _value Σ size_ / _value max depth_
sum and max across cases' value trees; _guard Σ size_ / _guard
max depth_ do the same across every guard tree.

| Program | k heads | # cases | cases | value Σ size | value max depth | guard Σ size | guard max depth | match |
|---|---:|---:|---|---:|---:|---:|---:|:-:|
| `select_by_sign(7)` | 5 | 2 | `{x0 != 0} → x4`<br>`{x0 == 0} → x7` | – | – | – | – | ✓ |
| `clamp_zero(5)` | 5 | 2 | `{x0 != 0} → x0`<br>`{x0 == 0} → x5` | – | – | – | – | ✓ |
| `either_or(3,7,1)` | 6 | 2 | `{x2 != 0} → x1`<br>`{x2 == 0} → x0` | – | – | – | – | ✓ |
| `native_max(3,5)` | 8 | 2 | `{x0 - x1 <= 0} → x1`<br>`{x0 - x1 > 0} → x0` | – | – | – | – | ✓ |

## Collapsed (unrolled at the catalog's concrete inputs)

| Program | k heads | # mono | poly | eml size | eml depth | match |
|---|---:|---:|---|---:|---:|:-:|
| `fibonacci(5)` | 50 | 1 | `5` | – | – | ✓ |
| `factorial(4)` | 45 | 1 | `24` | – | – | ✓ |
| `is_even(6)` | 35 | 1 | `1` | – | – | ✓ |
| `power_of_2(4)` | 45 | 1 | `16` | – | – | ✓ |
| `popcount_loop(5)` | 41 | 18 | `((((5 >>ᵤ 1) >>ᵤ 1) & 1) + (((5 >>ᵤ 1) & 1) + ((5 & 1) + 0)))` | – | – | ✓ |

## Collapsed (closed form from symbolic loop — issue #89)

Rows whose loop body is an affine / linear / multiplicative
recurrence on the loop-carried stack slice. The recurrence
solver emits a `Poly` (Tier 1, via Faulhaber), `ClosedForm`
(Tier 2, constant integer matrix), or `ProductForm` (Tier 3,
bounded product of a Poly factor). Unlike the _unrolled_ rows
above, this is a **symbolic proof** that holds at every `n`,
not a single-input execution trace. eml-sr columns stay `–`
— matrix power and bounded products aren't expressible in the
single-operator EML family.

| Program | k heads | size | closed form | match |
|---|---:|---:|---|:-:|
| `sum_1_to_n_sym(n)` | 22 | 3 | `x0 + 1/2*x1 + 1/2*x1^2` | ✓ |
| `power_of_2_sym(n)` | 22 | 1 | `ClosedForm(A=((2,),), b=(0,), s_0=('x0',), trip=x1, proj=0)` | ✓ |
| `fibonacci_sym(n)` | 29 | 2 | `ClosedForm(A=((0, 1), (1, 1)), b=(0, 0), s_0=('x0', 'x1'), trip=x2, proj=1)` | ✓ |
| `factorial_sym(n)` | 22 | 1 | `ProductForm(p=x1000001, k=x1000001, lower=1, upper=x1, init=1)` | ✓ |

## Blocked (out of symbolic-executor scope)

| Program | reason | blocker |
|---|---|---|
| `native_abs_unary(-3)` | non-polynomial op | `ABS` |
| `native_neg(5)` | non-polynomial op | `NEG` |

