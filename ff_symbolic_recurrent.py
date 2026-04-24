"""B.2 — recurrent FF gadget sub-path of issue #109 / Path B.

One FF microstep ≡ one loop iteration. Iterating the gadget ``n`` times
reproduces ``Aⁿ · s_0 + …`` for Tier 2 (ClosedForm) and ``init · ∏ p(k)``
for Tier 3 (ProductForm), with **no degree bound** in n. This is the
sub-path that actually closes the catalog's collapsed_closed_form rows
at the weight layer for any n.

The framing trade — explicit and unavoidable, the design doc names it:

* The "single bilinear form" framing that #69 / #75 / #76 / #77 made
  clean is **abandoned** here. B.2 is a recurrence whose unrolling has
  ``n`` weight-layer applications, not 1. The constant
  :data:`SINGLE_LAYER_CLAIM_B2` is literally False so this honesty
  cannot be misread.

* B.2 requires a loop counter in the architecture (positional
  re-embedding or a dedicated head). The result carries an explicit
  ``iterations_run`` field so callers can audit that the counter is
  real, not a claim in prose.

The microstep itself is a tight loop in pure Python — the same recurrence
``ClosedForm.eval_at`` would walk, only here we expose the iteration
count and structure it as the unrolled FF call sequence the design doc
describes. Numerical correctness is by construction (we run the same
recurrence); the value Path B adds is *the explicit count*.
"""
from __future__ import annotations

from typing import Tuple

from closed_form import ClosedForm, ProductForm
from path_b import PathBOutOfScope, PathBResult
from poly import Poly


SINGLE_LAYER_CLAIM_B2: bool = False
"""B.2 abandons the single-layer bilinear claim of #69 / #75 / #76 / #77.

Set as a module-level constant so the framing change is queryable from
outside (catalog reporter, blog #82). The value MUST stay False — flip
to True only if a future variant collapses the n-step microstep loop
back into a single layer (which would require the polynomial-embedding
trick from B.1 and inherit B.1's degree bound, defeating the point).
"""


def _extract_n(fr) -> int:
    """Counter binding for the row's symbolic ``x0`` (matches B.1)."""
    return int(fr.bindings.get(0, 0))


def _iterate_closed_form(top: ClosedForm, bindings) -> Tuple[int, int]:
    """Run the linear recurrence ``s_{k+1} = A · s_k + b`` for ``n`` steps.

    Returns ``(weight_layer_top, iterations_run)``. The iterations are
    the FF microstep applications; ``weight_layer_top`` is the projected
    slot of ``s_n``. Mirrors :meth:`closed_form.ClosedForm.eval_at` but
    surfaces the iteration count.
    """
    n = int(top.trip_count.eval_at(bindings))
    if n < 0:
        raise PathBOutOfScope(
            f"B.2: ClosedForm trip_count < 0 ({n}) — recurrence undefined"
        )
    m = len(top.s_0)
    if not top.A or len(top.A) != m or any(len(row) != m for row in top.A):
        raise PathBOutOfScope(
            f"B.2: ClosedForm A must be {m}×{m}; got shape mismatch"
        )
    if len(top.b) != m:
        raise PathBOutOfScope(
            f"B.2: ClosedForm b must be length {m}; got {len(top.b)}"
        )
    state = [int(p.eval_at(bindings)) for p in top.s_0]
    # Initial-state load: one FF microstep per s_0 slot beyond the
    # first. A first-order recurrence (m=1) has the initial value as
    # part of the embedding — no prep call. A higher-order recurrence
    # (m≥2, e.g. fibonacci's (F(0), F(1))) needs (m-1) preliminary FF
    # calls to populate the additional state slots in the residual
    # stream before the recurrence body fires. This is what closes
    # the off-by-one between fibonacci's trip_count (n-1) and the
    # caller's n: iter = trip + (m - 1) = (n-1) + 1 = n.
    iterations = max(m - 1, 0)
    for _ in range(n):
        # One FF microstep: state' = A · state + b. In a real
        # transformer this is a single attention+FF call with the
        # recurrence's A, b baked into weights and the previous state
        # in residual stream. Here we just step the recurrence — the
        # numerical answer is identical, the iteration count is what
        # we expose.
        nxt = [
            sum(top.A[i][j] * state[j] for j in range(m)) + top.b[i]
            for i in range(m)
        ]
        state = nxt
        iterations += 1
    if not 0 <= top.projection < m:
        raise PathBOutOfScope(
            f"B.2: ClosedForm projection={top.projection} out of [0,{m})"
        )
    return int(state[top.projection]), iterations


def _iterate_product_form(top: ProductForm, bindings) -> Tuple[int, int]:
    """Run ``acc ← acc · p(k)`` from ``lower`` to ``upper`` inclusive.

    Returns ``(weight_layer_top, iterations_run)``. Each step of the
    product is one FF microstep; iterations_run is ``upper - lower + 1``
    when the range is non-empty, else 0 (Python's empty-product
    convention — the gadget is invoked zero times).
    """
    lo = int(top.lower.eval_at(bindings))
    hi = int(top.upper.eval_at(bindings))
    acc = int(top.init)
    iterations = 0
    for k in range(lo, hi + 1):
        b = dict(bindings)
        b[top.counter_var] = k
        acc *= int(top.p.eval_at(b))
        iterations += 1
    return acc, iterations


def evaluate_program_forking_recurrent(prog, *, input_mode: str = "symbolic") -> PathBResult:
    """B.2 entrypoint — iterates the FF microstep n times for any
    ClosedForm or ProductForm top. Distinct from the generic Path B
    dispatcher in :func:`path_b.evaluate_program_forking_weight_layer`
    only in that it always uses B.2 (no auto-dispatch).

    Tier 1 (Poly top) doesn't need the recurrent gadget — it's covered
    by the existing bilinear claim — so calling B.2 on a Poly top still
    works (degenerate: 0 iterations, value comes from eval_at) but
    callers who want the single-layer Tier 1 story should use B.1.
    """
    # Lazy import: ff_symbolic re-exports path_b which re-exports this.
    from ff_symbolic import evaluate_program_forking

    fr = evaluate_program_forking(prog, input_mode=input_mode)
    return b2_forward(fr, prog, row_name="<unknown>")


def b2_forward(fr, prog, *, row_name: str) -> PathBResult:
    """Run the B.2 gadget for an already-computed forking-executor result.

    The Path B dispatcher calls this directly, avoiding a second
    ``run_forking`` pass.
    """
    if isinstance(fr.top, ClosedForm):
        out, iters = _iterate_closed_form(fr.top, fr.bindings)
        return PathBResult(
            top=fr.top,
            weight_layer_top=out,
            path_used="b2",
            bindings=dict(fr.bindings),
            iterations_run=iters,
        )
    if isinstance(fr.top, ProductForm):
        out, iters = _iterate_product_form(fr.top, fr.bindings)
        return PathBResult(
            top=fr.top,
            weight_layer_top=out,
            path_used="b2",
            bindings=dict(fr.bindings),
            iterations_run=iters,
        )
    if isinstance(fr.top, Poly):
        # Degenerate: a Poly top has no recurrence to iterate. Return
        # eval_at with iterations_run reflecting the symbolic counter,
        # so callers can still read off "how many gadget applications
        # would the recurrence cost." For sum_1_to_n_sym this equals n.
        n = _extract_n(fr)
        out = int(fr.top.eval_at(fr.bindings))
        return PathBResult(
            top=fr.top,
            weight_layer_top=out,
            path_used="b2",
            bindings=dict(fr.bindings),
            iterations_run=max(n, 0),
        )
    raise PathBOutOfScope(
        f"B.2: row={row_name!r}: top type {type(fr.top).__name__} has no "
        f"recurrent realisation (Path B covers Poly / ClosedForm / ProductForm)"
    )


__all__ = [
    "SINGLE_LAYER_CLAIM_B2",
    "b2_forward",
    "evaluate_program_forking_recurrent",
]
