# FF equivalence for closed-form loop tops

_Issue #90 design doc. Follow-up to #89 (loop-invariant inference —
``ClosedForm`` / ``ProductForm`` sibling types) and #69 / #75 / #76 /
#77 (the FF-layer equivalence story for arithmetic / rational /
comparison / bit-vector fragments). The S3 analogue for the
closed-form fragment asked for in #88's ``dev/loop_invariant_inference.md``
non-goals bullet._

## The claim, in one sentence

For every catalog row the #89 recurrence solver closes into a
``collapsed_closed_form`` top, the FF-driven executor
(``evaluate_program_forking``) emits a sibling that is structurally
equal to the native solver's output; forward-time numerical agreement
with ``NumPyExecutor`` is carried through ``eval_at`` at the boundary,
matching the same "polynomial ring inside, non-poly step at the edge"
pattern ``RationalPoly`` / ``IndicatorPoly`` / ``BitVec`` already use.

## Why the bilinear-form story needs a separate bite here

Issues #69 / #75 / #76 / #77 share one shape: find a weight-layer
representation for the opcode family (pair-selector matrix, gated
bilinear form, extraction matrix + boundary step) such that the
numeric FF forward pass and the symbolic-``Poly`` interpreter agree
*structurally* on the operator tree, not just numerically. That
works opcode-by-opcode because every opcode's result is a finite
polynomial (or extractor-plus-boundary) in the PUSH variables.

Closed-form loop tops break this in two places.

1. **``Aⁿ`` is not a bilinear form in the PUSH variables.** A
   ``ClosedForm`` evaluates ``Aⁿ · s_0 + (Aⁿ − I)(A − I)⁻¹ · b``
   projected to a slot. ``n`` is itself a Poly in the PUSH variables
   (the trip count), so ``Aⁿ`` is a polynomial in ``n`` of degree that
   grows with ``n``. No fixed bilinear form over a fixed-width
   embedding reproduces it — the degree is unbounded.
2. **``∏ p(k)`` from ``lower`` to ``upper`` has the same obstruction.**
   For ``ProductForm`` the accumulator's degree in the counter grows
   linearly with the trip count, so no single weight matrix realises
   it for symbolic ``n``.

Both obstructions are about *forward-time* evaluation at the weight
layer. They do **not** stop us from making the strong structural
claim at solver time — because the solver itself is bilinear-compatible.

## Tier 1 rides free on #69

The first tier of #89 (affine-polynomial recurrences via Faulhaber)
produces a plain ``Poly`` top — ``sum_1_to_n_sym(n)`` collapses to
``x0 + ½·x1 + ½·x1²`` (accumulator + ``n(n+1)/2``). There is nothing
new to say here: the existing ``M_ADD`` / ``B_MUL`` composition from
#69 realises this polynomial, the ``test_equivalence_structural`` test
family already pins it, and ``run_catalog`` reports it as
``ff_equiv=bilinear``. We call the Tier 1 case out explicitly so the
design doc doesn't paint all three tiers with the same brush.

## The move: solver-level structural equivalence

The recurrence solver lives in ``forking_executor`` (and is dispatched
by ``symbolic_executor.run_forking``). When it walks a loop body to
derive the transition, it uses whichever ``arithmetic_ops`` the caller
passed in. ``evaluate_program_forking`` passes
``FF_ARITHMETIC_OPS``, so the body's ADD/SUB/MUL operations run
through ``ff_symbolic.symbolic_add`` / ``_sub`` / ``_mul`` — the
Poly-level interpretation of ``M_ADD`` / ``M_SUB`` / ``B_MUL``. Those
are structurally identical to the native ``Poly.__add__`` / ``__sub__``
/ ``__mul__`` by #69's equivalence theorem.

Consequence: the solver's classification (Tier 1 / 2 / 3), its derived
transition (``A``, ``b``, ``s_0``, ``trip_count``, ``projection`` for
``ClosedForm``; ``p``, ``counter_var``, ``lower``, ``upper``, ``init``
for ``ProductForm``) is identical across the two drivers. This is a
real claim — and it is free.

Formally:

> **Solver-level structural equivalence.** For every program ``P`` on
> which ``classify_program(P, solve_recurrences=True)`` returns
> ``status = STATUS_COLLAPSED_CLOSED_FORM``,
>
> ```python
> run_forking(P, input_mode="symbolic", solve_recurrences=True).top \
>     == evaluate_program_forking(P, input_mode="symbolic").top
> ```
>
> on value-based ``==`` of the emitted sibling (``Poly`` for Tier 1,
> ``ClosedForm`` for Tier 2, ``ProductForm`` for Tier 3). Verified in
> ``test_ff_symbolic.test_equivalence_closed_form_structural`` over the
> four ``*_sym(n)`` catalog rows.

The proof is the same two-level argument #69 / #77 use:

1. **Unit level.** ``symbolic_add`` / ``_sub`` / ``_mul`` are
   definitionally Poly arithmetic. The recurrence classifier inspects
   Poly-level expressions (affine? constant matrix? single MUL with
   Poly factor?). Identical Polys produce identical classifications and
   identical ``(A, b, s_0, …)`` tuples.
2. **Compositional level.** The solver is indifferent to the concrete
   identity of the arithmetic primitives — ``FF_ARITHMETIC_OPS`` and
   ``DEFAULT_ARITHMETIC_OPS`` are wired through the same
   ``arithmetic_ops`` parameter. Plugging either produces the same
   ``ClosedForm`` / ``ProductForm`` because both reduce to the same
   Poly operations under the hood.

## The boundary: forward-time evaluation via ``eval_at``

Once the sibling is emitted, numeric agreement with ``NumPyExecutor``
is provided by the sibling's own ``eval_at``:

- ``ClosedForm.eval_at(bindings)`` resolves ``trip_count`` and ``s_0``
  to integers, iterates ``s_{k+1} = A · s_k + b`` for ``n`` steps,
  and returns the projected slot.
- ``ProductForm.eval_at(bindings)`` resolves the bounds and walks the
  counter from ``lower`` to ``upper`` inclusive, multiplying the
  accumulator.

That boundary step is the non-polynomial operation — it's not a
bilinear form in the PUSH variables and cannot be made one without
the obstructions spelled out above. We therefore *do not* state a
weight-layer structural-equality theorem for Tier 2 / Tier 3 rows.
What we state instead:

> **Boundary numeric agreement.** For every program ``P`` above and
> every concrete ``bindings`` in the row's validation set,
>
> ```python
> evaluate_program_forking(P, input_mode="symbolic").top.eval_at(bindings) \
>     == NumPyExecutor.execute(make_*(concrete_n)).steps[-1].top
> ```
>
> where ``concrete_n`` is the integer the row's bindings assign to the
> trip-count variable. Verified in
> ``test_ff_symbolic.test_equivalence_closed_form_numeric`` at ≥ 5
> bindings per row.

This is honestly weaker than the ``{ADD, SUB, MUL}`` / comparison /
bit-vector theorems, and the doc says so. The strength it retains —
solver-level structural equality plus boundary numeric agreement — is
the same contract the ``RationalPoly`` / ``IndicatorPoly`` / ``BitVec``
siblings satisfy (one spec, two interpreters at the Poly layer; the
non-polynomial step fires only at ``eval_at``). What's weaker here is
that the non-polynomial step (``Aⁿ`` or ``∏``) is not a single
boundary opcode but a bounded unrolling — so the "one spec, two
interpreters" view lives at the solver, not at the weight layer.

## Weight budget: unchanged

No new matrices are introduced. The non-zero weight count stays at
**15** after #69 + #75 + #76 + #77. Tier 2 / Tier 3 closed-form
evaluation is explicitly *not* realised in the weight layer; Tier 1 is
already realised by the existing ``M_ADD`` / ``B_MUL`` composition.

## Catalog reporting: new ``ff_equiv`` column

``CatalogRow`` gains an ``ff_equiv`` field populated by ``run_catalog``.
Three values:

| Value               | Meaning                                                                                                                                    | Rows |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---- |
| ``bilinear``        | The FF layer's matrices (``M_ADD`` / ``M_SUB`` / ``B_MUL`` / ``M_DIV_S`` / ``M_REM_S`` / ``M_CMP`` / ``M_EQZ`` / ``M_BITBIN`` / ``M_BITUN``) compose to emit this top structurally equal to the symbolic executor. Covers every currently-collapsed row except Tier 2 / Tier 3 closed-form. | all ``collapsed`` / ``collapsed_unrolled`` / ``collapsed_guarded`` rows plus Tier 1 ``collapsed_closed_form`` (Poly top) |
| ``solver_structural`` | The recurrence solver emits a sibling structurally equal across drivers, but forward-time ``eval_at`` is the non-polynomial boundary step — no weight-layer claim past the solver. Per this issue. | Tier 2 / Tier 3 ``collapsed_closed_form`` rows (``power_of_2_sym``, ``fibonacci_sym``, ``factorial_sym``) |
| ``n/a``             | Row didn't collapse (``blocked_opcode``, ``blocked_loop_sym``, path-explosion, …). No FF-equivalence claim to make.                                | all blocked / unreachable rows |

The column is orthogonal to ``status``: the latter says *what the
symbolic executor produced*, the former says *how strong an
FF-equivalence claim we make for it*.

## Equivalence theorem (full statement)

> Let ``P`` be any program in ``_default_catalog()`` with
> ``ff_equiv in {bilinear, solver_structural}``.
>
> 1. **Structural.** There exists a sibling type ``T ∈ {Poly,
>    GuardedPoly, RationalPoly, SymbolicRemainder, IndicatorPoly,
>    BitVec, ClosedForm, ProductForm}`` such that
>    ``run_symbolic(P).top`` (or ``run_forking(P, …).top`` for
>    guarded / unrolled / closed-form rows) is a ``T`` and
>    ``evaluate_program_forking(P, …).top`` is also a ``T`` with
>    ``==`` on value-based equality.
> 2. **Numeric (boundary).** For every concrete bindings in the row's
>    validation set,
>    ``evaluate_program_forking(P, …).top.eval_at(bindings) ==
>     NumPyExecutor.execute(make_*(k)).steps[-1].top``.
> 3. **Scope clause.** For rows with ``ff_equiv = solver_structural``,
>    the structural claim holds at the *solver's emitted sibling*, not
>    at a weight-layer realisation of ``Aⁿ`` / ``∏ p(k)``. The latter
>    is explicitly out of scope (see non-goals below).

Point 3 is the honest restriction this issue introduces. Everything
else was already claimed by the parent equivalence doc; we are just
making its scope explicit in the presence of closed-form tops.

## Worked example: ``fibonacci_sym(n)``

Program: ``PUSH 0; PUSH 1; PUSH n; body; HALT``. The ``body`` region
advances the pair ``(a, b) → (b, a+b)`` per iteration, with the trip
count in the third stack slot.

```
run_forking / evaluate_program_forking on symbolic inputs:
  initial stack = [x0, x1, x2]      (x0=0, x1=1, x2=n in bindings)
  solver walks body once:
    in:  [a: Poly, b: Poly]
    out: [b, a + b]                  (via symbolic_add / native Poly)
  classifier:
    linear in (a, b)? yes
    constant integer matrix?         A = ((0, 1), (1, 1))
    linear trip count? yes           trip = x2
    → Tier 2, ClosedForm emitted

Both drivers produce exactly:
  ClosedForm(A=((0, 1), (1, 1)), b=(0, 0),
             s_0=(Poly.variable(0), Poly.variable(1)),
             trip=Poly.variable(2),
             projection=1)
```

Structural equality is immediate (same dataclass, same field values).
``eval_at({0:0, 1:1, 2:10})`` on either side iterates the matrix
recurrence ten times starting from ``(0, 1)`` and reads slot 1 — the
integer 55, matching ``NumPyExecutor(make_fibonacci(10))``.

## Worked example: ``factorial_sym(n)``

Program: ``PUSH 1; PUSH n; body; HALT``. The ``body`` decrements the
counter and multiplies into the accumulator.

```
solver walks body once:
  accumulator update is a single MUL by a Poly in the counter (x_k).
  trip count is linear in the input.
  → Tier 3, ProductForm emitted

Both drivers produce exactly:
  ProductForm(p=Poly.variable(1000001),     # synthetic counter var
              counter_var=1000001,
              lower=Poly.constant(1),
              upper=Poly.variable(1),       # n
              init=1)
```

``eval_at({1: 5})`` multiplies ``1 · 2 · 3 · 4 · 5 = 120``, matching
``NumPyExecutor(make_factorial(5))``.

## Non-goals (explicit follow-ups)

- **Weight-layer realisation of ``Aⁿ``.** Would require either (a) a
  polynomial embedding ``E(n) = (1, n, n², …, nᴷ)`` that only works for
  bounded ``K`` (and the degree of ``Aⁿ`` is unbounded in ``n``), (b) a
  recurrent FF gadget that iterates the state per step (leaves the
  single-layer bilinear-form story entirely), or (c) algebraic-number
  coefficients for eigenvalue-based closed forms (explicitly rejected
  by #89's "no Binet" non-goal). None are small; each would be bigger
  than #69.
- **Weight-layer realisation of ``∏ p(k)``.** Same obstruction — the
  accumulator's degree in the counter grows linearly with the trip
  count.
- **Algebraic-number closed forms.** Inherited non-goal from #89.
- **Nested loops.** Inherited non-goal from #89.

Per the recommendation in #90, none of these are pursued here. The
issue #90 comment lists them under "Path B" — deferred indefinitely
unless a motivating catalog row demands forward-time closed-form
evaluation at weight granularity.

> **Update (issue #109):** Path B has since landed — see ``path_b.py``,
> ``ff_symbolic_poly_embedding.py`` (B.1), ``ff_symbolic_recurrent.py``
> (B.2), and ``algebraic_poly.py`` (B.3). The trigger was
> ``fibonacci_sym(n)`` flipping its CatalogEntry's
> ``requests_weight_layer=True`` flag — the motivator gate #107
> demanded. The ``ff_equiv`` column gains a fourth value
> ``bilinear_weight_layer`` for rows whose Path B realisation is
> proven; other Tier 2/3 rows stay at ``solver_structural`` until
> their own motivator appears. The bullets above are preserved as
> historical context for what the obstruction shape looked like
> before B.1/B.2/B.3 were each priced out.

## What this doc *is* worth, in one line

The closed-form fragment now has the same spec shape that carried
arithmetic, rational, comparison, and bit-vector extensions:
**structural equivalence at the bilinear-Poly layer where it fits,
and an honest boundary step at ``eval_at`` where it doesn't** — with
the caveat that for Tier 2 / Tier 3 tops the "bilinear-Poly layer" is
the recurrence solver itself, not a weight matrix, because the
non-polynomial step (matrix exponentiation, bounded product) isn't
realisable as a fixed bilinear form over the PUSH variables. The #69
equivalence theorem gains a one-line scope clause; the catalog gains
a ``ff_equiv`` column; no weights change.

## Cross-references

- Parent: #69 (ADD/SUB/MUL bilinear forms; the original
  ``dev/ff_symbolic_equivalence.md`` this doc sits next to) + #75 /
  #76 / #77 (rational / indicator / bit-vector extensions).
- Loop-invariant inference: #88 (``dev/loop_invariant_inference.md``
  design doc) + #89 (Tier 1–3 implementation; introduced
  ``ClosedForm`` / ``ProductForm``).
- This doc closes #90 — the "S3 analogue for closed-form fragment"
  bullet explicitly carved out in #88's non-goals.
