"""Path B — weight-layer realisation of ``ClosedForm`` / ``ProductForm``.

Tracking issue: #109. Design doc: ``dev/ff_closed_form_equivalence.md``.

#107 closed #90 with the "solver-structural + boundary ``eval_at``" claim
(Path A). For symbolic-loop programs, ``ClosedForm`` / ``ProductForm``
tops are evaluated *outside* the FF weight layer at bind time, via
``eval_at``'s iterated recurrence. Path B widens that — three sub-paths,
each with real costs:

* **B.1** polynomial embedding ``E(n) = (1, n, …, nᴷ)`` —
  :mod:`ff_symbolic_poly_embedding`. Exact for total-degree ≤ K Polys
  (Tier 1). Honest out-of-scope for Tier 2/3 past K.
* **B.2** recurrent FF gadget — :mod:`ff_symbolic_recurrent`. Iterates
  the FF microstep ``n`` times. Covers Tier 2/3 with no degree bound.
  Trades single-layer framing for recurrence.
* **B.3** algebraic-number coefficients — :mod:`algebraic_poly`. Binet-
  style ``(φⁿ − ψⁿ)/√5`` over ℚ(√5). Reopens #89's "no Binet" non-goal.

This module is the dispatcher: it owns the typed scope exception, the
``in-scope`` predicate, the ``PathBResult`` carrier, and the public
``evaluate_program_forking_weight_layer`` entrypoint that picks the
sub-path. The default entry point ``ff.evaluate_program_forking``
remains Path A — Path B is opt-in by calling THIS entrypoint or by
passing a sub-path label explicitly.

Per #107: Path B should not activate silently. The default catalog row
classification stays at ``solver_structural``; rows are only upgraded to
``bilinear_weight_layer`` when their CatalogEntry sets
``requests_weight_layer=True``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from closed_form import ClosedForm, ProductForm
from poly import Poly


class PathBOutOfScope(Exception):
    """Raised when a Path B sub-path cannot realise a (row, n) at the
    weight layer. The message must cite enough context (row, n, K when
    applicable) to be auditable from the catalog and the blog narrative.

    Honest failure beats silent wrap. Each sub-path that has a bounded
    scope (B.1's degree K, B.3's algebraic-eigenvalue requirement) raises
    this rather than returning a wrong integer.
    """


@dataclass
class PathBResult:
    """Carrier for a Path B forward-pass result.

    ``top`` is the same structural top Path A produces (Poly / ClosedForm
    / ProductForm). ``weight_layer_top`` is the integer the weight-layer
    forward pass returned — the whole point of Path B is that this comes
    from a weight-layer realisation rather than ``top.eval_at(bindings)``.

    ``iterations_run`` is the loop counter B.2 exposes (None for B.1 /
    B.3). ``path_used`` records which sub-path actually ran. ``bindings``
    mirrors :class:`ForkingResult.bindings`.
    """

    top: Any
    weight_layer_top: int
    path_used: str
    bindings: Dict[int, int] = field(default_factory=dict)
    iterations_run: Optional[int] = None


# Public alias under the canonical name the test contract uses.
PATH_B_OUT_OF_SCOPE_EXCEPTION = PathBOutOfScope


# ─── Scope predicate ──────────────────────────────────────────────
#
# Each path's scope is documented here in one place so the catalog
# report and the blog narrative can both query it without parsing
# error messages. The four collapsed_closed_form rows split:
#
#   sum_1_to_n_sym   — Tier 1 Poly        — covered by B.1, B.2 (any n)
#   power_of_2_sym   — Tier 2 ClosedForm  — covered by B.2 (any n);
#                                           B.1 only for n small enough
#                                           that 2ⁿ fits in degree K
#   fibonacci_sym    — Tier 2 ClosedForm  — covered by B.2 (any n);
#                                           B.1 bounded; B.3 (Binet) any n
#   factorial_sym    — Tier 3 ProductForm — covered by B.2 (any n);
#                                           B.1 bounded; B.3 not in scope
#                                           (factorial isn't algebraic)

_TIER1_ROWS = {"sum_1_to_n_sym"}
_TIER2_ROWS = {"power_of_2_sym", "fibonacci_sym"}
_TIER3_ROWS = {"factorial_sym"}
_KNOWN_ROWS = _TIER1_ROWS | _TIER2_ROWS | _TIER3_ROWS

# B.3 covers exactly the rows whose recurrence eigenvalues live in a
# named algebraic extension we ship with :mod:`algebraic_poly`. Today
# that is ℚ(√5) — Fibonacci's φ / ψ. Power-of-2 has integer eigenvalues
# and falls back to B.2; factorial isn't a linear recurrence at all.
_B3_ROWS = {"fibonacci_sym"}


def path_b_in_scope(row_name: str, n: int, *, path: Optional[str] = None) -> bool:
    """Return True iff a Path B weight-layer call for ``(row_name, n)``
    will succeed without raising :class:`PathBOutOfScope`.

    Pure function — no side effects, no executor invocation. Used by the
    catalog reporter and the blog narrative to advertise scope without
    having to swallow exceptions.

    When ``path`` is None, returns True iff the default auto-dispatch
    path (B.1 for Tier 1, B.2 for Tier 2/3) would succeed. When ``path``
    is one of "b1" / "b2" / "b3", returns True iff THAT specific path
    covers the row at this n.
    """
    if row_name not in _KNOWN_ROWS:
        # Unknown row → no Path B scope to advertise.
        return False

    if path is None:
        # Default dispatch: B.1 covers Tier 1 trivially; B.2 covers
        # Tier 2/3 universally. Both cover any n, so default is in scope
        # for every known row.
        return True

    if path == "b1":
        # B.1 is exact for Tier 1 (degree-fits trivially for sum_1_to_n).
        # For Tier 2/3, the closed-form has unbounded degree in n, so
        # B.1 only covers small n. The exact bound depends on
        # POLY_EMBEDDING_DEGREE; we read it lazily to avoid a circular
        # import.
        if row_name in _TIER1_ROWS:
            return True
        try:
            from ff_symbolic_poly_embedding import POLY_EMBEDDING_DEGREE
        except ImportError:
            return False
        return n <= POLY_EMBEDDING_DEGREE

    if path == "b2":
        # B.2 iterates the FF microstep n times — no scope limit on n
        # for any of the four collapsed_closed_form rows.
        return True

    if path == "b3":
        return row_name in _B3_ROWS

    return False


# ─── Dispatcher ───────────────────────────────────────────────────

def _default_path_for_top(top: Any) -> str:
    """Pick a sub-path that covers a top of this type.

    Tier 1 (Poly) → B.1 trivially. Tier 2/3 (ClosedForm / ProductForm) →
    B.2 (universal). B.3 is opt-in only.
    """
    if isinstance(top, Poly):
        return "b1"
    if isinstance(top, (ClosedForm, ProductForm)):
        return "b2"
    raise PathBOutOfScope(
        f"top type {type(top).__name__} has no Path B realisation; "
        "Path B is defined for Poly / ClosedForm / ProductForm tops only"
    )


def evaluate_program_forking_weight_layer(
    prog,
    *,
    input_mode: str = "symbolic",
    path: Optional[str] = None,
    row_name: Optional[str] = None,
) -> PathBResult:
    """Forward-pass entrypoint for Path B.

    Distinct from :func:`ff_symbolic.evaluate_program_forking` (Path A —
    solver-structural + boundary ``eval_at``). The structural top is
    still produced by Path A; this entrypoint adds a weight-layer
    forward pass that returns the numeric answer via one of B.1 / B.2 /
    B.3, dispatched on ``path`` (or auto-selected from the top type).

    ``row_name`` is optional context for the scope exception message —
    callers that know the catalog row name should pass it.

    Out-of-scope cases raise :class:`PathBOutOfScope` — never silent
    fallback to ``eval_at``.
    """
    # Lazy import to break the dependency cycle: ff_symbolic re-exports
    # this entrypoint.
    from ff_symbolic import evaluate_program_forking

    fr = evaluate_program_forking(prog, input_mode=input_mode)
    chosen = path or _default_path_for_top(fr.top)
    label = row_name or "<unknown>"

    if chosen == "b1":
        from ff_symbolic_poly_embedding import b1_forward
        return b1_forward(fr, prog, row_name=label)
    if chosen == "b2":
        from ff_symbolic_recurrent import b2_forward
        return b2_forward(fr, prog, row_name=label)
    if chosen == "b3":
        from algebraic_poly import b3_forward
        return b3_forward(fr, prog, row_name=label)
    raise ValueError(f"unknown Path B sub-path: {chosen!r}")


__all__ = [
    "PathBOutOfScope",
    "PATH_B_OUT_OF_SCOPE_EXCEPTION",
    "PathBResult",
    "path_b_in_scope",
    "evaluate_program_forking_weight_layer",
]
