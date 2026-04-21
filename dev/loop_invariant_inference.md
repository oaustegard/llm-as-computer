# Loop-invariant inference for symbolic loops

_Issue #79 design doc. Stretch bite X1 from the #68 roadmap. Defines the scope of a follow-up implementation issue; not itself an implementation spec._

## The claim, in one sentence

For the subfamily of loops whose body is an **affine update** on the loop-carried stack slots and whose trip count is a **symbolic linear Poly**, the forking executor can emit a closed-form top in a new `ClosedForm` sibling type instead of halting with `loop_symbolic` — extending the `Poly` / `RationalPoly` / `IndicatorPoly` / `BitVec` pattern to cover `fibonacci(n)`, `sum_1_to_n(n)`, `power_of_2(n)`, and `factorial(n)` at symbolic `n`.

## Why this is hard (and where the polynomial ring stops helping)

`run_forking(input_mode="symbolic")` already handles bounded loops with concrete counters by unrolling, and finite conditionals by forking with guards. What it cannot do: follow a back-edge whose guard is a Poly in the PUSH variables. The pre-flight for symbolic back-edge revisit (symbolic_executor.py:1737–1741) exists precisely to stop the worklist from blowing up. This is the right default — but it throws away the one thing we need for the motivating cases: the loop body itself is a polynomial, applied `n−1` times, starting from a polynomial state.

The polynomial ring `ℤ[x₀, x₁, …]` closes under the four motivating bodies:

| Program            | Update (post-body)                          | Closed form at trip count `n`                    | Ring fit                               |
| ------------------ | ------------------------------------------- | ------------------------------------------------ | -------------------------------------- |
| `sum_1_to_n(n)`    | `(acc, k) ↦ (acc + k, k − 1)`               | `acc = n(n+1)/2`                                 | `Poly` (needs `Fraction` coefficients — already supported) |
| `factorial(n)`     | `(acc, k) ↦ (acc · k, k − 1)`               | `acc = n!`                                       | **NOT** polynomial in `n`              |
| `power_of_2(n)`    | `v ↦ 2 · v` (trip `n`)                      | `v = 2ⁿ`                                         | **NOT** polynomial in `n`              |
| `fibonacci(n)`     | `(a, b) ↦ (b, a + b)` (trip `n−1`)          | `b = F(n)` via `[[0,1],[1,1]]ⁿ⁻¹`                | **NOT** polynomial in `n`              |

Only the arithmetic-sum family stays inside `Poly`. Geometric and linear-recurrence families escape the ring — same shape of tension #75 (rationals), #76 (indicators), and #77 (bit-vectors) encountered. The fix is the same: **keep the polynomial ring closed at the expression level, push the non-polynomial step to a first-class sibling type**.

## The three tiers we cover

### Tier 1 — Affine-polynomial recurrences (stays in `Poly`)

Body is affine in the loop-carried variables with Poly coefficients. Trip count is a linear Poly in the input variables. After the body runs `n` times, the closed form is a polynomial of degree ≤ (1 + body-degree) in `n`.

Covers: `sum_1_to_n(n)`, `sum_of_squares_1_to_n(n)` and any `Σ p(k)` for `p` a fixed Poly (Faulhaber's formula), **but also** polynomial accumulators of arbitrary degree over a linear-decremented counter.

Implementation: symbolic summation over `Fraction`-coefficient Poly. `Poly.eval_at` already returns `Fraction` when needed.

### Tier 2 — Linear recurrences over `Poly` (introduces `ClosedForm` sibling)

Body is a linear map on the loop-carried stack slice: `s_{k+1} = A · s_k + b` where `A` is a constant integer matrix, `b` a constant integer vector, `s_k` a vector of Polys. Trip count is a linear Poly.

Covers: `fibonacci(n)` (A = `[[0,1],[1,1]]`, b = 0), `power_of_2(n)` (A = `[[2]]`, b = 0), generalised Lucas sequences, any linear homogeneous recurrence with constant integer coefficients.

Implementation: a new `ClosedForm` frozen dataclass holding `(A, b, s_0, trip_count_poly, projection)`. `eval_at(bindings)` resolves `trip_count_poly` to a concrete integer `n`, computes `Aⁿ · s_0 + (Aⁿ − I)(A − I)⁻¹ · b` via integer matrix-power (`numpy.linalg.matrix_power` on `object` dtype, or a recursive squaring implementation to stay pure-Python), and extracts the projected slot.

We **deliberately do not** produce a symbolic Binet-formula expression. The ring doesn't close over eigenvalues (`φ = (1 + √5)/2` is irrational), and introducing algebraic-number support is a larger scope than this bite. `ClosedForm` keeps the recurrence structure symbolic and evaluates numerically at binding time — same as `BitVec` evaluates its bit operations at binding time rather than simplifying algebraically.

### Tier 3 — Multiplicative recurrences (introduces `ProductForm` sibling)

Body is `acc ← acc · p(k)` for `p` a fixed Poly; trip count is linear.

Covers: `factorial(n)` (p(k) = k), `double_factorial(n)`, any `∏ p(k)`.

Implementation: a `ProductForm` dataclass holding `(p_poly, lower, upper, init)`. `eval_at` resolves bounds and multiplies. We treat this as a **separate** sibling from `ClosedForm` because `∏` is neither a linear recurrence in Poly (the state grows unboundedly in degree each step) nor a polynomial in the trip count.

This tier is optional for the first ship — it's listed so the design doesn't paint itself into a Tier-2-only corner.

## Pipeline

```
run_forking symbolic mode → hits symbolic back-edge → instead of halting
with loop_symbolic:
  1. Identify loop body (pc range between back-edge target and JZ/JNZ).
  2. Symbolically execute the body ONCE over the loop-carried slice
     of the stack to derive the transition.
  3. Try to classify the transition in order (Tier 1 → 2 → 3):
       - Tier 1: is every output Poly an affine function of the inputs
                 plus a polynomial in the counter? → close via Faulhaber.
       - Tier 2: is the transition a linear map with constant coefficients
                 and a linear trip count? → emit ClosedForm.
       - Tier 3: is the accumulator update a single MUL with a Poly in
                 the counter? → emit ProductForm.
  4. On success: replace the loop body with a single symbolic step that
     assigns the closed-form top; continue the worklist from the exit
     pc. Classification → collapsed_closed_form.
  5. On failure: fall through to the existing loop_symbolic path
     unchanged.
```

Steps 2–3 are the real work. Step 2 reuses `run_forking` itself on the body fragment, with a specialised `_Path` that treats the loop-carried slots as fresh symbolic inputs. Step 3 is a small classifier that inspects the resulting Poly expressions — linear in inputs? constant matrix? pure multiplication with a Poly factor?

## Status code + catalog row

New classification status:

```python
STATUS_COLLAPSED_CLOSED_FORM = "collapsed_closed_form"
```

`ClassificationResult` gains one optional field:

```python
closed_form: Optional[Union[ClosedForm, ProductForm]] = None
```

Populated alongside `poly` (for Tier 1 — the closed form stays a `Poly`, the status just records that we got there via recurrence solving rather than branchless evaluation).

## Catalog impact

Four new parametric-symbolic rows land as `collapsed_closed_form`:

| Row                         | Tier | Closed form                    | Verified against `NumPyExecutor` at |
| --------------------------- | ---- | ------------------------------ | ----------------------------------- |
| `sum_1_to_n_sym(n)`         | 1    | `Poly(n(n+1)/2)`               | `n ∈ {1, 2, 5, 10, 20}`             |
| `power_of_2_sym(n)`         | 2    | `ClosedForm(A=[[2]], ...)`     | `n ∈ {0, 1, 4, 8, 10}`              |
| `fibonacci_sym(n)`          | 2    | `ClosedForm(A=[[0,1],[1,1]], ...)` | `n ∈ {1, 2, 5, 10, 15}`          |
| `factorial_sym(n)`          | 3    | `ProductForm(p=x₀, 1..n, init=1)` | `n ∈ {1, 2, 5, 7, 10}`           |

These are *new* entries, parallel to the existing concrete-counter rows (`fibonacci(5)`, `factorial(4)`, `power_of_2(4)`). The concrete rows stay as `collapsed_unrolled` — unchanged. Symbolic rows use a new `make_*_symbolic` generator family in `programs.py` that does not bake `n` into a `PUSH` arg but leaves the PUSH to the forking executor's variable-allocation path.

Equivalence tests: `ClosedForm.eval_at({0: k}) == NumPyExecutor.execute(make_*(k)).top` for every `k` in the validation set above.

## New sibling types — minimal API

```python
@dataclass(frozen=True)
class ClosedForm:
    A: Tuple[Tuple[int, ...], ...]        # constant integer matrix
    b: Tuple[int, ...]                    # constant integer additive vector
    s_0: Tuple[Poly, ...]                 # initial state (each entry a Poly)
    trip_count: Poly                      # symbolic number of iterations
    projection: int                       # which slot is emitted as top
    def eval_at(self, bindings: Mapping[int, int]) -> int: ...

@dataclass(frozen=True)
class ProductForm:
    p: Poly                               # factor polynomial (in counter var)
    counter_var: int                      # which variable is the counter
    lower: Poly                           # lower bound (symbolic)
    upper: Poly                           # upper bound (symbolic)
    init: int                             # product identity (usually 1)
    def eval_at(self, bindings: Mapping[int, int]) -> int: ...
```

Both are frozen, value-equal, and produce an `int` at `eval_at` — same contract as `BitVec`.

## Acceptance criteria (for the implementation issue that follows this doc)

1. `symbolic_executor.py` exports `ClosedForm` and `ProductForm`; `run_forking` grows a `solve_recurrences: bool = True` flag.
2. Four new symbolic-`n` programs in `programs.py`; four new catalog entries in `symbolic_programs_catalog.py`.
3. `classify_program` returns `STATUS_COLLAPSED_CLOSED_FORM` for all four, with the expected sibling type populated.
4. A new test file `test_closed_form.py` pins structural equality of the emitted closed forms and numeric agreement with `NumPyExecutor` at ≥ 5 values per row.
5. No regressions in `test_symbolic_executor.py`, `test_symbolic_programs_catalog.py`, or `test_ff_symbolic.py`. The `solve_recurrences=False` path must reproduce today's `blocked_loop_symbolic` behaviour exactly.
6. `run_catalog` summary printout gains a `closed_form` column alongside `collapsed | guarded | unrolled | loop_symbolic | …`.

## Non-goals (explicit follow-ups)

- **General recurrence solving.** Non-linear recurrences (e.g. `s_{k+1} = s_k²`), higher-order recurrences that don't reduce to constant-matrix form, and recurrences whose trip count is itself a polynomial of degree > 1 are out of scope. We cover the subfamily that matches the ISA's loop idioms in this codebase, not the general case.
- **Algebraic-number closed forms.** Binet's formula involves `√5`. We deliberately keep `ClosedForm` *structural* — `A, b, s_0, trip_count, projection` — and let `eval_at` do integer matrix exponentiation. A symbolic Binet-style representation would require an extension to the coefficient ring (algebraic integers or their polynomial-over-`ℤ[√d]` closure), which is a much larger scope.
- **Nested loops.** The motivating cases are all single-loop. Nested loops require either recursive recurrence inference (tractable when the outer counter is concrete) or a more general fixed-point iteration — the latter is explicitly out of scope per the issue text.
- **Symbolic fixed-point computation.** Matching the issue's non-goals.
- **FF-layer equivalence for `ClosedForm` / `ProductForm`.** The bilinear-form story (#69, #75, #76, #77) needs a separate treatment for closed-form tops — matrix exponentiation is not a bilinear form in the PUSH variables. This is a real gap but belongs in its own issue (an analogue of S3 for the closed-form fragment), not here.

## Dependencies

Per the issue: probably after M2 (sign indicators for exit conditions). In practice M2 already landed as #76 (`IndicatorPoly` is a sign indicator for the comparison family), so the dependency is satisfied — the loop-invariant pass can assume guards are expressed as `IndicatorPoly` over a linear `Poly` and use the sign to infer monotonic counter decrement.

## What this design doc is worth, in one line

Loop-invariant inference fits the same spec shape that carried the rational, indicator, and bit-vector extensions: **a first-class sibling type for the non-polynomial closed form, structural equality for the symbolic comparison, boundary evaluation at binding time.** Shipping Tiers 1+2 — even without Tier 3 — turns the four motivating parametric programs from `blocked_loop_symbolic` into `collapsed_closed_form`, closing the last major class of catalog blockers that isn't a missing opcode.

## Cross-references

- Roadmap: #68 (post-PR-67 follow-ups), bite X1.
- Prior sibling types: #75 (`RationalPoly`), #76 (`IndicatorPoly`), #77 (`BitVec`).
- Dependent on: #76 sign-indicator machinery (already landed).
- Paper (external, contextual): Kincaid et al., *"Closed Forms for Numerical Loops"*, POPL 2019 — the canonical reference for Tier 2; our construction is a scoped-down integer-matrix version of theirs.
