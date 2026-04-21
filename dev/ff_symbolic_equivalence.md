# FF symbolic equivalence — the weights ARE the polynomial

_Issue #69 writeup. Follow-up to #65 (PR #66 symbolic executor, PR #67 catalog runner). Extended by #75 to cover DIV_S / REM_S via rational-pair algebra._

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

## Honest limits

### Range / i32-wrap

The bilinear form computes over `ℤ` — no `& MASK32`. The issue framed
this as Option (a): "produce a polynomial over `ℤ`; add a `range_check`
that asserts no wrap would have occurred on the catalog inputs." That's
the route taken here.

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

Option (b) — carry `mod 2^32` through the polynomial algebra — is
strictly heavier (polynomials over `ℤ/2^32ℤ` have gcd factoring issues
and no division) and is deliberately out of scope.

### What this issue does **not** prove

Listing these explicitly so the PR description stays honest:

- **DIV_S / REM_S beyond a single op at the top of the stack** —
  covered for terminal DIV_S / REM_S by issue #75 (pair-selector +
  boundary trunc), but composition *past* a rational stack entry
  still raises. Full rational-function algebra (with GCD cancellation)
  is a follow-up.
- **Comparisons** (EQ, LT_S, GT_S, …) — piecewise, need sign indicators
  and/or Heaviside gating; out of scope.
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
- **Piecewise bilinear forms** for comparisons. A sign-indicator
  attention pattern gates into one of two output branches — the FF
  counterpart of the symbolic executor's forking model (#70).
- **Mod-2 bilinear forms** for bitwise ops via bit decomposition. Same
  FF-layer machinery, different base ring.

None of these are in this issue. They are listed so the roadmap is
visible.

## References

- Parent issue: #65 · follow-ups: #69, #75
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
