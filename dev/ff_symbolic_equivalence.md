# FF symbolic equivalence — the weights ARE the polynomial

_Issue #69 writeup. Follow-up to #65 (PR #66 symbolic executor, PR #67 catalog runner). Extended by #75 to cover DIV_S / REM_S via rational-pair algebra, and by #76 to cover comparisons (EQZ / EQ / NE / LT_S / GT_S / LE_S / GE_S) via gated bilinear forms with an `IndicatorPoly` sibling type._

## The claim, in one sentence

For the arithmetic fragment `{ADD, SUB, MUL}` of the ISA, the FF layer
of `CompiledModel` is a bilinear form, the symbolic executor is the
same bilinear form over `Poly` inputs, and the two agree structurally
on every collapsed catalog program.

## What the previous story proved

PR #66 and PR #67 established that the **ISA semantics** compose into a
single polynomial for branchless programs. `symbolic_executor.run_symbolic`
walks the program with a `Poly`-valued stack and emits the top-of-stack
polynomial at HALT. `symbolic_programs_catalog.run_catalog` cross-checks
that polynomial against `NumPyExecutor` for every branchless catalog entry
and the numbers match.

That's a genuine claim — but it's narrow. It says that if you compose
`Poly.__add__`, `Poly.__mul__`, etc. in the order the ISA says to, you get
a polynomial that evaluates to the right number. It does **not** say that
the compiled transformer's weights realise that polynomial. Before this
issue, `executor.CompiledModel.forward` had two dispatch paths:

- **Linear** (`executor.py:670-703, 800-811`): `M_top` routes one of
  `[arg, val_a, val_b, val_c, local_val, heap_val]` to the output per
  opcode. Ops like PUSH/POP/DUP/HALT/SWAP/OVER/ROT are pure linear
  routing — no arithmetic; `M_top` does the whole job.
- **Nonlinear** (`executor.py:813-870`, pre-#69): for ADD/SUB/MUL and
  every other arithmetic or bitwise op, the forward pass fell through
  to CPython:

  ```python
  nonlinear[OPCODE_IDX[OP_ADD]] = float((va + vb) & MASK32)
  nonlinear[OPCODE_IDX[OP_SUB]] = float((vb - va) & MASK32)
  nonlinear[OPCODE_IDX[OP_MUL]] = float((va * vb) & MASK32)
  ```

  The `M_top` row for ADD is literally zero. The transformer routed
  results that CPython computed. The thesis slogan — "weights are a
  compiler target; the forward pass is a CPU" — was only fully supported
  by the linear path.

## What this issue does

Replace the Python-arithmetic calls for `ADD`, `SUB`, `MUL` with three
analytically-set weight matrices in `ff_symbolic.py`, and prove (by
construction + test) that:

1. The matrices implement the right arithmetic on float tensors.
2. The same matrices, re-interpreted as operations on `Poly`, produce
   exactly the polynomial `symbolic_executor.run_symbolic` emits.
3. On every currently-collapsed catalog program, the two interpreters
   agree structurally, not just numerically.

## Embedding choice

The existing value embedding (`isa.embed_stack_entry`) is already scalar:
`embed[DIM_VALUE] = v`. So `E(v) = v · e_{DIM_VALUE}` where `e_i` is the
standard basis. That choice makes the three constructions direct:

| Op  | Form    | Construction                                   | Claim                                     |
| --- | ------- | ---------------------------------------------- | ----------------------------------------- |
| ADD | Linear  | `M_ADD[DIM_VALUE, DIM_VALUE] = 1`, `M_ADD[DIM_VALUE, d+DIM_VALUE] = 1` | `M_ADD @ [E(a); E(b)] = E(a+b)` |
| SUB | Linear  | `M_SUB[DIM_VALUE, DIM_VALUE] = -1`, `M_SUB[DIM_VALUE, d+DIM_VALUE] = 1` | `M_SUB @ [E(a); E(b)] = E(b-a)` |
| MUL | Bilinear | `B_MUL[DIM_VALUE, DIM_VALUE] = 1` (rank-1 outer product) | `E(a)^T B_MUL E(b) = a·b`              |

Every other entry in those matrices is zero. The total weight budget
added by #69 is five non-zero values
(`M_ADD: 2, M_SUB: 2, B_MUL: 1`); the blog post's "964 compiled
parameters" becomes 969. Issue #75 adds four more (`M_DIV_S: 2,
M_REM_S: 2`) for a total of nine — see the rational-extension
section below.

Under the scalar embedding, ADD and SUB are linear (because `E` is a
linear map on `ℤ`) and MUL is bilinear (a degree-2 polynomial, which is
exactly what `B_MUL` encodes as a rank-1 outer product). A higher-degree
embedding — e.g. `E_mul(a) = (1, a, a^2, ..., a^K)` — would let a single
bilinear form realise polynomials of degree up to K without composition.
For this issue we don't need that; degree-1 `E` plus composition is
enough to reach every polynomial the collapsed catalog exhibits (up to
the degree induced by the `MUL`/`DUP` chain in the program).

## The two interpreters

The same operator tree has two interpretations:

- **Numeric** (`forward_add`, `forward_sub`, `forward_mul`): inputs are
  `torch.Tensor` of shape `(d_model,)`, outputs are the same shape. The
  computation is a matmul (`M_ADD @ stacked`) or a bilinear contraction
  (`ea @ B_MUL @ eb`). The integer value is recovered with `E_inv`.

- **Symbolic** (`symbolic_add`, `symbolic_sub`, `symbolic_mul`): inputs
  are `Poly`, outputs are `Poly`, and the bodies are literally
  `pa + pb`, `pb - pa`, `pa * pb`. The polynomial-algebra interpretation
  of `M_ADD`, `M_SUB`, `B_MUL` *is* `Poly` arithmetic.

The equivalence is that these two live at the same named spec:
`ff_symbolic.forward_mul(ea, eb)` and `ff_symbolic.symbolic_mul(pa, pb)`
are the tensor and polynomial interpretations of the same bilinear form
`B_MUL`. The compiler for the arithmetic fragment is therefore *one*
formula (the bilinear form), with *two* interpreters (floats and polys).

## Worked example: `dup_add_chain_x4`

Program: `PUSH 5; (DUP; ADD) × 4; HALT` — nine instructions, nine heads.

```
After PUSH 5:          stack = [x0]
After DUP:             stack = [x0, x0]
After ADD:             stack = [E(x0) + E(x0) = E(2·x0)]          i.e. [2·x0]
After DUP:             stack = [2·x0, 2·x0]
After ADD:             stack = [E(2·x0) + E(2·x0) = E(4·x0)]       i.e. [4·x0]
After DUP; ADD:        [8·x0]
After DUP; ADD:        [16·x0]
HALT                   top = 16·x0
```

Every ADD step is `ea + eb` under `M_ADD`; every DUP is a stack copy
(already linear in the pre-#69 model). The whole chain is a composition
of linear maps, which is itself linear — its analytic form is
`16 · e_{DIM_VALUE}`. So the FF layer's nine compositions of `M_ADD`
produce a single monomial `16·x0`, which is exactly what
`symbolic_executor.run_symbolic(PROG).top` returns.

The check at `test_ff_symbolic.py::test_dup_add_chain_pin`
pins this equality; the parametrised `test_equivalence_structural`
generalises it to all 15 currently-collapsed catalog programs.

## Rational extension (issue #75)

Integer division and remainder aren't polynomial operations over ℤ, so
the `{ADD, SUB, MUL}` story above — "the weight matrix _is_ the
operator" — doesn't extend cleanly. Issue #75 keeps that spirit going
by pushing the arithmetic into the rational field ℚ up to the final
truncation step:

- `symbolic_executor.Poly` coefficients widen from `int` to
  `int | Fraction`. `_normalise` still collapses ratios with denominator
  1 back to `int`, so every structural Poly equality test from #69
  continues to compare pure-`int` coefficients. Fraction only appears
  inside a `RationalPoly` / `SymbolicRemainder` wrapper.
- Two minimal wrappers carry ``(num, denom)`` pairs through the
  symbolic stack:

  ```python
  @dataclass(frozen=True)
  class RationalPoly:       # DIV_S result
      num: Poly
      denom: Poly
      def eval_at(self, bindings): return _trunc_div(num_val, denom_val)

  @dataclass(frozen=True)
  class SymbolicRemainder:  # REM_S result
      num: Poly
      denom: Poly
      def eval_at(self, bindings): return _trunc_rem(num_val, denom_val)
  ```

  `eval_at` is where the non-polynomial truncation lives: we collapse
  to ℤ only at the boundary, matching WASM `i32.div_s` / `i32.rem_s`
  semantics.
- `ff_symbolic.M_DIV_S` / `M_REM_S` are 2 × 2·d_model **pair-selector**
  matrices — they pluck `(va, vb)` from the stacked `[E(a); E(b)]`
  input and feed them to the boundary `_trunc_div(vb, va)` /
  `_trunc_rem(vb, va)`. Two non-zero entries each; the weight budget
  (non-zero entries across all five matrices) is
  `M_ADD(2) + M_SUB(2) + B_MUL(1) + M_DIV_S(2) + M_REM_S(2) = 9`.
- `symbolic_div_s` / `symbolic_rem_s` are the polynomial-side
  interpretation: construct `RationalPoly(num=pb, denom=pa)` /
  `SymbolicRemainder(num=pb, denom=pa)` with `(pa, pb) = (top, SP-1)`
  matching the numeric argument order.

The equivalence theorem for the rational fragment is weaker than the
bilinear-form story of #69 because integer-truncating division is a
**non-linear boundary step**. The honest statement is:

> For every DIV_S / REM_S catalog program P and its concrete bindings,
> `run_symbolic(P).top` and `forward_symbolic(P).top` are structurally
> equal (same wrapper, same num, same denom), and each evaluates — via
> `eval_at` — to the same integer that `NumPyExecutor(P).top` reports,
> provided the dividend and divisor are both in the positive i32 range
> (so NumPy's `& MASK32` doesn't perturb the equality).

### Composition non-goal

A second DIV_S / REM_S / ADD / SUB / MUL applied to a `RationalPoly`
or `SymbolicRemainder` raises `SymbolicOpNotSupported`
(`BlockedOpcodeForSymbolic` on the FF side). Supporting composition
would require either (a) a proper rational-function algebra with GCD
cancellation (rational coefficients are insufficient because the
numerator / denominator can share symbolic factors) or (b) delaying
truncation through every subsequent op, which is unsound. Neither is
small. Left to a follow-up.

### Catalog impact

Two programs move from `blocked_opcode` to `collapsed`:
`native_divmod(2, 7)` (a `RationalPoly` of `7 / 2` → 3) and
`native_remainder(2, 7)` (a `SymbolicRemainder` of `7 mod 2` → 1). The
`CatalogRow.is_rational` flag marks them; `poly_expr` renders as
`"(num) /ₜ (denom)"` / `"(num) %ₜ (denom)"`; eml-sr columns stay `–`
because the single `eml(x, y) = exp(x) − ln(y)` operator has no
division primitive.

## Gated bilinear extension (issue #76)

Comparisons (`EQZ`, `EQ`, `NE`, `LT_S`, `GT_S`, `LE_S`, `GE_S`) aren't
polynomial operations over ℤ either — their outputs (0 or 1) depend on
the **sign** of the input, not its algebraic structure. Issue #76
carries the same idea that #75 used for rational ops into the
comparison fragment:

> Keep the polynomial fragment linear. Push the non-polynomial gate to
> the boundary. Make the gate first-class rather than pretending it
> isn't there.

### The three moves

1. **Symbolic executor: a new sibling type `IndicatorPoly`.**

   ```python
   @dataclass(frozen=True)
   class IndicatorPoly:
       poly: Poly        # the linear diff (pb − pa for binary cmps, pa for EQZ)
       relation: str     # one of REL_EQ / REL_NE / REL_LT / REL_LE / REL_GT / REL_GE
       def eval_at(self, bindings) -> int:
           return int(_relation_holds(self.relation, self.poly.eval_at(bindings)))
   ```

   The polynomial stays pure (a linear diff). The gate lives in
   `eval_at`. This is the **exact analogue of `RationalPoly`**: the
   underlying `Poly` arithmetic is closed, the non-polynomial boundary
   step (`_trunc_div` for rationals, `_relation_holds` for indicators)
   fires only when a concrete integer is demanded.

2. **`Guard` broadens from `eq_zero: bool` to `relation: str`.**

   Pre-#76, a `Guard` meant "this polynomial equals zero" or "this
   polynomial is nonzero" — the only relations JZ/JNZ branching
   produced. Post-#76, a `Guard` carries any of the six relations.
   `eq_zero` survives as a `@property` shim so existing code continues
   to read the old semantics, but the canonical field is `relation`.

   This lets JZ/JNZ consume an `IndicatorPoly` directly. Rather than
   wrapping the indicator in an extra case-split, the executor **hoists**
   the indicator's relation into the Guard pair:

   ```python
   # JZ on IndicatorPoly(poly=p, relation=LT) produces:
   #   take_guard  (condition was 0 ⇒ ¬(p < 0))  : Guard(poly=p, relation=GE)
   #   skip_guard  (condition was 1 ⇒ p < 0)     : Guard(poly=p, relation=LT)
   ```

   The `native_max` program — `GT_S; JZ skip; POP; HALT; skip: SWAP POP HALT` —
   now collapses to a clean two-case `GuardedPoly`:

   ```
   Guarded[
     {(x0 - x1 <= 0)} → x1,   # GT_S returned 0 ⇒ vb ≤ va ⇒ max is va = x1
     {(x0 - x1 >  0)} → x0,   # GT_S returned 1 ⇒ vb > va ⇒ max is vb = x0
   ]
   ```

   Note the guards carry `LE` / `GT` directly — not `EQ` / `NE` with
   the indicator wrapped inside. The sign test the comparison opcode
   implies is now the Guard's own relation.

3. **`ff_symbolic`: two new linear matrices plus a relation gate.**

   Comparisons decompose into **linear diff extraction** followed by a
   **non-polynomial relation gate**:

   | Op       | Weight tensor                                   | Shape        | Non-zero entries | Gate                  |
   | -------- | ----------------------------------------------- | ------------ | ---------------: | --------------------- |
   | Binary cmp | `M_CMP[0, DIM_VALUE] = -1`, `M_CMP[0, d+DIM_VALUE] = 1` | `(1, 2d)` | 2                | `1 if diff <rel> 0 else 0` |
   | `EQZ`      | `M_EQZ[0, DIM_VALUE] = 1`                             | `(1, d)`   | 1                | `1 if va == 0 else 0` |

   `forward_cmp(ea, eb, op)` computes `diff = M_CMP @ stack(ea, eb)`
   (a scalar `vb − va`) then returns `E(_relation_holds(op_rel, diff))`.
   `forward_eqz(ea)` is the `d`-wide variant for the unary case.

   On the symbolic side, `symbolic_cmp(pa, pb, op)` returns
   `IndicatorPoly(poly=pb - pa, relation=OP_RELATION[op])`; `symbolic_eqz(pa)`
   returns `IndicatorPoly(poly=pa, relation=REL_EQ)`. The weight tensor's
   polynomial interpretation is exactly this diff-extraction.

   Total new non-zero weight budget: **3** (two for `M_CMP`, one for
   `M_EQZ`). Running total after #69 + #75 + #76 is **12** non-zero
   entries across seven matrices — the comparison fragment is cheaper
   per-op than ADD/SUB because it only extracts a scalar, not a vector.

### Equivalence theorem for the comparison fragment

> For every comparison opcode `op ∈ {EQZ, EQ, NE, LT_S, GT_S, LE_S, GE_S}`
> and every input embedding `ea, eb`:
>
> 1. **Structural.** `run_symbolic(P).top == forward_symbolic(P).top` on
>    `IndicatorPoly` value-equality (same underlying `Poly`, same
>    `relation` tag) for every program `P` ending on `op`.
> 2. **Numeric (three-way).** `run_symbolic(P).top.eval_at(bindings) ==
>    NumPyExecutor(P).top == TorchExecutor(P).top` for every `P` and
>    every concrete bindings — the same agreement the `{ADD, SUB, MUL}`
>    and `{DIV_S, REM_S}` fragments enjoy, now extended over the six
>    signed comparison opcodes plus `EQZ`.
> 3. **Dispatch.** If `op` is immediately consumed by `JZ` / `JNZ`, the
>    resulting `GuardedPoly` has guards whose `relation` reflects the
>    taken / skipped interpretation of the comparison, not a two-level
>    `indicator == 0` / `indicator != 0` wrapping.

Point (3) is the cosmetic-but-important payoff: the `GuardedPoly` for
`native_max(a, b)` reads as *"if a ≤ b then b else a"* at the case-table
level, without the reader having to unfold an intermediate indicator.

### Catalog impact

Two straight-line rows move from `blocked_opcode` to `collapsed`
(`IndicatorPoly` tops):
- `compare_lt_s(3, 5)` → `[x0 - x1 < 0]`, numeric = 1.
- `compare_eqz(0)`    → `[x0 == 0]`, numeric = 1.

One row moves from `blocked_opcode` to `collapsed_guarded`:
- `native_max(3, 5)` → `Guarded[{x0 - x1 ≤ 0} → x1; {x0 - x1 > 0} → x0]`,
  numeric = 5.

eml-sr columns stay `–` for indicator rows — the single-operator family
`eml(x, y) = exp(x) − ln(y)` has no sign primitive. The gate lives at
the FF-dispatch boundary, not inside the polynomial the eml tree would
compile. That's the same reason `is_rational` rows skip eml: the
polynomial part of the tree is perfectly expressible, but the wrapping
semantics aren't.

### Composition non-goal

Arithmetic composed on top of an `IndicatorPoly` (e.g. `LT_S` followed by
`ADD`) raises `SymbolicOpNotSupported` / `BlockedOpcodeForSymbolic`,
mirroring the rational-composition non-goal from #75. Supporting it
would require either (a) promoting `IndicatorPoly` to a first-class
piecewise polynomial algebra, or (b) materialising the 0/1 value
eagerly — either of which defeats the "keep the polynomial fragment
linear" principle. Out of scope for this issue.

### Refactor plan: migrating S1 guards to sign indicators

S1 (PR #71 / issue #70) introduced the forking executor with
`Guard(poly, eq_zero=bool)`. Post-#76, `Guard.eq_zero` is a `@property`
shim over the canonical `relation` field, but the *shape* of what
guards can express is now strictly larger than it was in S1:

- S1 guards only ever carried `REL_EQ` or `REL_NE` (the only relations
  JZ/JNZ against a plain `Poly` could produce).
- Post-#76 guards can carry any of six relations, because JZ/JNZ
  consuming an `IndicatorPoly` *hoists* that indicator's relation
  directly into the Guard.

The S1 refactor to lean on sign indicators is already mostly done —
this PR changed `Guard.relation` in place rather than keeping a parallel
field. What still has pre-#76 phrasing and would be cosmetically nicer
as signs:

1. **`_as_concrete_int` collapses concrete `IndicatorPoly`** values to
   0/1 at JZ/JNZ time. If a program happens to supply concrete pushes
   all the way into a comparison, the stack entry is already a plain
   `int` by the time `JZ` runs — the IndicatorPoly gets evaluated
   eagerly. This keeps concrete-mode traces matching pre-#76 behavior.
   No change needed; noted for readers.

2. **S1 catalog rows (`select_by_sign`, `clamp_zero`, `either_or`)**
   still render with `EQ/NE` guards in their `case_exprs`. That's
   correct — these programs use plain JZ on a raw `Poly` (not an
   `IndicatorPoly`), so the guard relations really *are* equality vs
   inequality. The broader relation set only kicks in when a comparison
   op feeds JZ/JNZ.

3. **Follow-up opportunity.** The sign-indicator framework makes it
   possible to *canonicalise* guards during `GuardedPoly` merges: two
   cases whose guards are `{p < 0}` and `{p > 0}` could be proven
   disjoint-but-not-covering (missing `p == 0`). Today the executor
   doesn't do this reasoning. A future issue could add a guard
   simplifier / coverage checker that uses the six-relation algebra
   to detect redundant or missing cases. Not in this PR.

The net effect: the S1 guard layer is now strictly richer, with no
behavioral drift on pre-#76 rows (verified: `test_classify_guarded_on_symbolic_branch`
and the `select_by_sign` / `clamp_zero` / `either_or` catalog pins
still pass exactly, including the `{g.eq_zero for ...} == {True, False}`
assertion that reads guards via the backward-compat property).

## Honest limits

### Range / i32-wrap

The bilinear form computes over `ℤ` — no `& MASK32`. The issue framed
this as Option (a): "produce a polynomial over `ℤ`; add a `range_check`
that asserts no wrap would have occurred on the catalog inputs." That's
the route taken in #69 and it still holds.

`CompiledModel.forward` still applies the mask *after* the bilinear form
so that `NumPyExecutor` parity holds bit-for-bit (`test_consolidated.py`
stays green — verified: all 39 programs pass). The equivalence theorem,
however, is stated pre-mask:

> For every collapsed catalog program P and the bindings it defines,
> `forward_symbolic(P).top.eval_at(bindings) == NumPyExecutor(P).top`
> **and** that evaluation fits inside `[I32_MIN, I32_MAX)` (verified by
> `ff_symbolic.range_check` at test time).

Every catalog program satisfies the range check; the largest unmasked
value is `factorial(10) = 3,628,800`, well inside i32.

Issue #78 landed Option (b) as a sibling artefact — see the next
section. Option (a) remains the "clean catalog" interpretation; Option
(b) is the honest wrap theorem that also covers overflow-probing
inputs.

## Wrap-aware extension (issue #78 option (b))

Issue #69 decided that carrying `mod 2³²` through the algebra was out
of scope and pinned the range via :func:`ff_symbolic.range_check`
instead (Option (a)). Issue #78's motivating concern is that the
resulting theorem is *only* tested in the range where both sides agree,
so "a bug in either direction would pass silently". Option (b) closes
that gap — not by replacing Option (a), but by adding a sibling
polynomial ring whose structural equality mirrors the FF layer's
`& MASK32` step in the algebra itself.

### `ModPoly`: polynomials over ℤ / 2³²

`symbolic_executor.ModPoly` is a sibling of :class:`Poly`. Same canonical
form (monomial dict, zero-coefficient terms dropped), same arithmetic
surface (`+`, `-`, `*`, `__neg__`) — the one difference is that every
operation reduces coefficients modulo 2³² via `__post_init__`. That
reduction *is* the `& MASK32` step, now visible in one place instead of
as a silent boundary fact.

Key operations:

```python
ModPoly.from_poly(p)              # lift a pure-integer Poly
mp1 + mp2, mp1 - mp2, mp1 * mp2   # closed under the ring ops
mp.eval_at(bindings)              # unsigned u32 int in [0, 2³²)
mp.eval_at_signed(bindings)       # signed i32 int in [-2³¹, 2³¹)
```

Non-goal: division. ℤ/2³² is a ring with zero divisors (`2 · 2³¹ = 0`),
so no `a / b` is well-defined. That's fine — DIV_S / REM_S already live
in their own wrapper (`RationalPoly` / `SymbolicRemainder`) whose
boundary evaluator handles truncation and wrap together.

### The mod-2³² driver

`ff_symbolic.evaluate_program_mod(prog)` mirrors
:func:`evaluate_program`, swapping `Poly` for `ModPoly` at every stack
site. The arithmetic primitives are `symbolic_add_mod` / `_sub_mod` /
`_mul_mod` — the ModPoly interpretation of the same operator tree
`M_ADD` / `M_SUB` / `B_MUL` realise over floats.

Scope is strictly narrower than :func:`evaluate_program`:
`{ADD, SUB, MUL, PUSH, POP, DUP, NOP, SWAP, OVER, ROT, HALT}`.
Comparisons, rationals, and bitwise ops raise
`BlockedOpcodeForSymbolic` — those fragments have their own boundary
wrap (`IndicatorPoly` / `RationalPoly` / `BitVec`) that already
corresponds to the correct semantics; lifting them into ℤ/2³² would
duplicate that logic.

### Equivalence theorem (Option (b))

> **Structural over ℤ/2³²:** For every collapsed catalog program P
> whose `run_symbolic(P).top` is a plain `Poly`,
>
> ```python
> ModPoly.from_poly(run_symbolic(P).top) == evaluate_program_mod(P).top
> ```
>
> as dict-equality on canonical coefficient maps in [0, 2³²). 15/15
> pure-Poly collapsed rows satisfy this
> (`test_modpoly_catalog_structural`).

The proof is a one-liner: the ring homomorphism ℤ → ℤ/2³² commutes
with every primitive operation the two drivers apply, so lifting the
inputs (each PUSH produces a ModPoly variable instead of a Poly
variable) and doing the same operations produces the lifted result.
`test_modpoly_homomorphism_on_catalog` asserts the commutation directly
— `ModPoly.from_poly(evaluate_program.top) == evaluate_program_mod.top`.

> **Numeric bit-for-bit:** For every collapsed row,
>
> ```python
> evaluate_program_mod(P).top.eval_at(bindings) \
>   == NumPyExecutor(P).top & 0xFFFFFFFF
> ```
>
> No range check needed — the two sides agree in ℤ/2³² **by
> construction**, not just on in-range inputs
> (`test_modpoly_catalog_numeric`).

### Overflow-stress coverage

`test_modpoly_overflow_stress` exercises four programs where the ℤ
result either doesn't fit signed i32 or exceeds `2³²` entirely:

| Program                         | ℤ value          | ℤ/2³² value    | Native (NumPy) | Notes                                            |
| ------------------------------- | ---------------: | -------------: | -------------: | ------------------------------------------------ |
| `PUSH 46341; DUP; MUL`          |    2,147,488,281 |  2,147,488,281 |  2,147,488,281 | overflows signed i32 (I32_MAX = 2,147,483,647) |
| `PUSH 100_000; PUSH 100_000; MUL` |  10,000,000,000 |  1,410,065,408 |  1,410,065,408 | exceeds u32; ℤ disagrees with native, ℤ/2³² matches |
| `PUSH 65536; DUP; MUL`          |   4,294,967,296 |              0 |              0 | zero-divisor witness: `2¹⁶ · 2¹⁶ = 0 (mod 2³²)`  |
| `PUSH I32_MAX; PUSH 2; ADD`     |    2,147,483,649 |  2,147,483,649 |  2,147,483,649 | signed overflow (reads as `I32_MIN + 1` signed)  |

The 100k × 100k case is the one Option (a) could not cover: its
`range_check` rejects the value (so the equivalence-under-range claim
doesn't apply), but the program is still a valid catalog-style
arithmetic trace. ModPoly makes the native-match provable for *any*
i32-representable inputs, not just the ones that happen to stay below
`I32_MAX`.

### What Option (b) deliberately doesn't change

- **Option (a) survives.** `range_check`, the equivalence theorem over
  ℤ, the writeup above — all still accurate for the catalog. They're
  the cleanest statement when inputs are known to be in range.
- **`forward_symbolic` stays over ℤ.** The mod-2³² path is an
  independent driver (`evaluate_program_mod`); the existing Poly-based
  API is unchanged. No existing test regresses.
- **Non-polynomial fragments stay in their wrappers.** Rationals,
  indicators, bit-vectors already wrap correctly at their boundary
  evaluators; lifting them into a ℤ/2³² polynomial would either
  duplicate or contradict their semantics. Out of scope.

### What this issue does **not** prove

Listing these explicitly so the PR description stays honest:

- **DIV_S / REM_S beyond a single op at the top of the stack** —
  covered for terminal DIV_S / REM_S by issue #75 (pair-selector +
  boundary trunc), but composition *past* a rational stack entry
  still raises. Full rational-function algebra (with GCD cancellation)
  is a follow-up.
- **Comparisons** (EQ, LT_S, GT_S, …) — handled by issue #76 via
  gated bilinear forms (linear diff extraction + relation gate at the
  dispatch boundary). See the "Gated bilinear extension" section above.
  Composition **past** a comparison (arithmetic on an `IndicatorPoly`)
  is still out of scope; `IndicatorPoly` is a terminal wrapper like
  `RationalPoly`.
- **Bitwise** (AND, OR, XOR, shifts, rotates) — different algebra (mod-2
  bilinear forms over bit decompositions); out of scope.
- **Unary ops** (CLZ, CTZ, POPCNT, ABS, NEG) — some are polynomial
  (NEG), some aren't (CLZ/CTZ). Uniform treatment would require a
  follow-up issue.
- **Control flow** (JZ/JNZ, CALL/RETURN) — covered for the *symbolic
  executor* by issue #70 (PR #71) via a forking model; that's a
  program-level construction, not a weight-level one. The FF-layer
  counterpart remains a follow-up.
- **Attention heads / stack reads** — unchanged by this issue. The
  bilinear form consumes `(val_a, val_b)` already extracted by the
  existing heads.

## Test harness

`test_ff_symbolic.py` has three layers:

1. **Primitive checks.** Sanity on `E`, `E_inv`, the three matrices'
   shapes, and spot-check `forward_add/sub/mul` on eight sign
   combinations including integers near i32 boundaries.
2. **Structural equivalence.** For every row with `STATUS_COLLAPSED` in
   `symbolic_programs_catalog._default_catalog()`, assert
   `run_symbolic(P).top == forward_symbolic(P).top` on canonical `Poly`
   equality. 15/15 pass.
3. **Numerical cross-check.** For the same entries, evaluate the
   `forward_symbolic` output at the catalog's bindings, run
   `range_check`, and compare with `NumPyExecutor`'s integer top.
4. **Blocked-opcode rejection.** `DIV_S`, `JZ`, `AND` each raise
   `BlockedOpcodeForSymbolic` rather than silently returning a
   plausible-but-wrong `Poly`.

All 70+ individual checks pass.

## Why this is the right next bite

Before #69 the LAC repo's claim was "we compiled the ISA semantics into
a PyTorch module that happens to wrap Python arithmetic". Post-#69 the
claim is "for the arithmetic fragment, the weights *are* the polynomial:
the FF layer is a bilinear form, the symbolic executor is the same
bilinear form over `Poly` inputs, and the two agree structurally."

That's the sentence the bridging blog post wanted to write. It also
sets up the cross-grammar comparison with eml-sr cleanly: the eml-sr
tree isn't just "a tree that computes the same function as the
transformer" — it's "a tree that computes the same polynomial that
`B_MUL` and `M_ADD` compose to evaluate." Three grammars, one object,
provably.

## Follow-ups this unlocks

Each of these is a separate issue:

- **Rational-function algebra with composition.** #75 handled the
  terminal DIV_S / REM_S case; supporting ADD / SUB / MUL / DIV_S /
  REM_S applied to an already-rational stack entry is the next bite.
  Needs GCD-based cancellation in the `Poly` ring (or a `RationalPoly`
  type with an explicit normal form).
- ~~**Piecewise bilinear forms** for comparisons.~~ **Done in #76.**
  `IndicatorPoly` + gated bilinear forms (`M_CMP` / `M_EQZ` + relation
  gate) realise the six signed comparisons + `EQZ`, and JZ/JNZ
  consuming an indicator hoists its relation into the Guard pair.
- **Mod-2 bilinear forms** for bitwise ops via bit decomposition. Same
  FF-layer machinery, different base ring.

None of these are in this issue. They are listed so the roadmap is
visible.

## References

- Parent issue: #65 · follow-ups: #69, #75, #76
- Symbolic executor: PR #66
- Catalog runner + eml bridge: PR #67
- Forking / guarded execution: PR #71 (issue #70)
- ADD/SUB/MUL bilinear forms: this module (issue #69).
- DIV_S / REM_S rational extension: issue #75 — adds
  `RationalPoly` / `SymbolicRemainder` to `symbolic_executor.py`,
  `M_DIV_S` / `M_REM_S` + `forward_div_s` / `forward_rem_s` +
  `symbolic_div_s` / `symbolic_rem_s` to `ff_symbolic.py`, rational
  rows to `symbolic_programs_catalog.py`, and the rational section
  above to this writeup.
- Comparison extension (gated bilinear forms): issue #76 — adds
  `IndicatorPoly` sibling type and the six-relation `Guard` to
  `symbolic_executor.py`, `M_CMP` / `M_EQZ` + `forward_cmp` /
  `forward_eqz` + `symbolic_cmp` / `symbolic_eqz` to `ff_symbolic.py`,
  indicator rows to `symbolic_programs_catalog.py`, and the "Gated
  bilinear extension" + "Refactor plan" sections above to this
  writeup.
