"""Boundary symbolic types (issues #75, #76).

These types escape :class:`poly.Poly` for operations that leave the
polynomial ring:

  - :class:`IndicatorPoly` — comparison results (EQ/NE/LT_S/GT_S/LE_S/
    GE_S/EQZ), evaluated to {0, 1} only at the ``eval_at`` boundary
    (issue #76).
  - :class:`RationalPoly` — symbolic ``num / denom`` from DIV_S,
    truncation-divided only at ``eval_at`` (issue #75).
  - :class:`SymbolicRemainder` — symbolic ``num % denom`` from REM_S
    (issue #75).

The relation constants (``REL_EQ``, ``REL_NE``, ...) and the
:data:`_RELATIONS` / :data:`_REL_SYMBOL` / :data:`_NEGATE_REL` tables
are defined here because :class:`IndicatorPoly` and :class:`guarded.Guard`
both consume them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Mapping, Tuple

from isa import _trunc_div, _trunc_rem
from poly import Poly

# ─── Sign-indicator form (issue #76) ──────────────────────────────
#
# Comparisons (EQ, NE, LT_S, GT_S, LE_S, GE_S, EQZ) are not polynomial
# operations on integer values: they're piecewise — 1 when a relation
# holds, 0 otherwise. Symbolically we carry the comparison forward as a
# (poly, relation) pair via :class:`IndicatorPoly`. Truncation to {0, 1}
# happens only at :meth:`IndicatorPoly.eval_at`, the boundary step —
# matching the same "polynomial-ring inside, non-poly step at the edge"
# pattern :class:`RationalPoly` and :class:`SymbolicRemainder` already use
# for DIV_S / REM_S.
#
# Composition past one comparison (e.g. ADD on top of an IndicatorPoly)
# is out of scope — the consuming opcode raises
# :class:`SymbolicOpNotSupported`. The catalog rows this unblocks
# (``compare_lt_s``, ``compare_eqz``, ``native_max``, ...) all either
# halt on the indicator directly (collapse to a non-Poly top) or pass it
# straight to a JZ/JNZ that the forking executor turns into a guarded
# split. Either way, the indicator never has to compose with another
# arithmetic op.

# Relation codes — comparisons are uniformly ``poly REL 0``, where REL
# is one of these six. EQZ folds into ``IndicatorPoly(va, REL_EQ)`` (the
# unary case is a degenerate binary one). Binary comparisons reduce to
# the relation on the difference ``vb - va`` so a single shared diff
# matrix suffices on the FF side (see ``ff_symbolic.M_CMP``).
REL_EQ = "EQ"
REL_NE = "NE"
REL_LT = "LT"
REL_LE = "LE"
REL_GT = "GT"
REL_GE = "GE"
_RELATIONS = (REL_EQ, REL_NE, REL_LT, REL_LE, REL_GT, REL_GE)

# Pretty-print symbols for guard / indicator repr.
_REL_SYMBOL = {
    REL_EQ: "==", REL_NE: "!=",
    REL_LT: "<",  REL_LE: "<=",
    REL_GT: ">",  REL_GE: ">=",
}

# Negation table — used when JZ/JNZ pops an IndicatorPoly: the "not
# taken" branch carries the negated relation as its guard.
_NEGATE_REL = {
    REL_EQ: REL_NE, REL_NE: REL_EQ,
    REL_LT: REL_GE, REL_GE: REL_LT,
    REL_LE: REL_GT, REL_GT: REL_LE,
}


def _relation_holds(rel: str, value: Union[int, Fraction]) -> bool:
    """Evaluate ``value REL 0`` for the given relation code."""
    if rel == REL_EQ: return value == 0
    if rel == REL_NE: return value != 0
    if rel == REL_LT: return value <  0
    if rel == REL_LE: return value <= 0
    if rel == REL_GT: return value >  0
    if rel == REL_GE: return value >= 0
    raise ValueError(f"unknown relation {rel!r}")


@dataclass(frozen=True)
class IndicatorPoly:
    """Sign indicator on a polynomial / bit-vector: ``1 if poly REL 0 else 0``.

    Carries a comparison's symbolic result through the stack without
    leaving the polynomial ring at the *expression* level — the gate
    fires only at :meth:`eval_at`, mirroring the boundary-truncation
    pattern :class:`RationalPoly` uses for DIV_S. The wrapped ``poly``
    is the (vb − va) difference for binary comparisons or the unary
    operand directly for EQZ; ``relation`` is one of
    ``REL_EQ, REL_NE, REL_LT, REL_LE, REL_GT, REL_GE``.

    Issue #77 widened ``poly`` to the :class:`SymbolicIntAst` union so
    comparisons against a :class:`BitVec` (``is_power_of_2``'s
    ``POPCNT; PUSH 1; EQ``) land here rather than raising. The BitVec
    gets evaluated to a concrete int at :meth:`eval_at`; the relation
    is then applied identically.

    Composition past an :class:`IndicatorPoly` (e.g. another ADD) is
    out of scope for issue #76 — the consuming op raises
    :class:`SymbolicOpNotSupported`, matching the rational story.
    """
    poly: Union[Poly, "BitVec"]
    relation: str

    def __post_init__(self):
        if self.relation not in _RELATIONS:
            raise ValueError(
                f"IndicatorPoly.relation must be one of {_RELATIONS}, "
                f"got {self.relation!r}"
            )

    def variables(self) -> List[int]:
        return self.poly.variables()

    def eval_at(self, bindings: Mapping[int, int]) -> int:
        v = self.poly.eval_at(bindings)
        return 1 if _relation_holds(self.relation, v) else 0

    def __repr__(self) -> str:
        return f"[{self.poly} {_REL_SYMBOL[self.relation]} 0]"


# ─── Rational + remainder forms (issue #75) ───────────────────────
#
# DIV_S / REM_S break out of the polynomial ring: integer division is
# not polynomial, and even the underlying rational a/b lands outside
# :class:`Poly` without coefficient generalisation. Per the issue's
# design note ("Probably the latter, since WASM i32.div_s is truncating
# and we can model the rational inside the ring while the i32 rounding
# lives at the boundary"), we keep the symbolic form rational and apply
# ``trunc_div`` / ``trunc_rem`` only at ``eval_at`` — the same boundary
# pattern :func:`ff_symbolic.range_check` uses for i32 wrap on ADD/SUB/MUL.
#
# Two minimal types, one per op: :class:`RationalPoly` for DIV_S and
# :class:`SymbolicRemainder` for REM_S. Algebra past the op itself is
# deliberately not closed — the catalog rows this unblocks
# (``native_divmod``, ``native_remainder``) consist of ``PUSH b; PUSH a;
# DIV_S/REM_S; HALT``, so no composition with Poly arithmetic is
# required. Composing DIV_S with further ADD/SUB/MUL would need a full
# rational-function algebra, listed as a follow-up (the issue's
# non-goal list calls this out explicitly).


@dataclass(frozen=True)
class RationalPoly:
    """Symbolic quotient ``num / denom`` under WASM ``i32.div_s`` semantics.

    Stores the two operand polynomials verbatim. ``eval_at`` reduces
    them to integers under the given bindings and then applies
    truncating-toward-zero division (:func:`isa._trunc_div`) — the same
    semantic the compiled transformer's nonlinear path applies. Trapping
    on ``denom == 0`` is the caller's responsibility at the bindings
    site; ``eval_at`` raises :class:`ZeroDivisionError` in that case.

    Structural equality is value-based on ``(num, denom)``, so two
    symbolic executors that emit the same operand polys produce equal
    tops — the equivalence test the issue asks for.
    """
    num: Poly
    denom: Poly

    def variables(self) -> List[int]:
        return sorted(set(self.num.variables()) | set(self.denom.variables()))

    def eval_at(self, bindings: Mapping[int, int]) -> int:
        """Integer result: ``trunc(num(bindings) / denom(bindings))``.

        ``denom`` evaluating to 0 raises :class:`ZeroDivisionError` — the
        catalog's `native_divmod(0, b)` variants already produce a trap
        in :class:`executor.NumPyExecutor` rather than a value, so the
        symbolic side mirrors that failure mode.
        """
        n = self.num.eval_at(bindings)
        d = self.denom.eval_at(bindings)
        if d == 0:
            raise ZeroDivisionError(
                f"RationalPoly.eval_at: denom {self.denom!r} = 0 at bindings={dict(bindings)}"
            )
        return _trunc_div(int(n), int(d))

    def __repr__(self) -> str:
        return f"({self.num}) /ₜ ({self.denom})"


@dataclass(frozen=True)
class SymbolicRemainder:
    """Symbolic remainder ``num % denom`` under WASM ``i32.rem_s`` semantics.

    Stored as a ``(num, denom)`` pair rather than reduced to a closed
    polynomial form, because ``b mod a`` is not rational in ``(a, b)``
    — the truncation that defines it is piecewise, not algebraic.
    ``eval_at`` applies :func:`isa._trunc_rem` at the boundary, matching
    the compiled transformer.
    """
    num: Poly
    denom: Poly

    def variables(self) -> List[int]:
        return sorted(set(self.num.variables()) | set(self.denom.variables()))

    def eval_at(self, bindings: Mapping[int, int]) -> int:
        n = self.num.eval_at(bindings)
        d = self.denom.eval_at(bindings)
        if d == 0:
            raise ZeroDivisionError(
                f"SymbolicRemainder.eval_at: denom {self.denom!r} = 0 at bindings={dict(bindings)}"
            )
        return _trunc_rem(int(n), int(d))

    def __repr__(self) -> str:
        return f"({self.num}) %ₜ ({self.denom})"



__all__ = [
    "REL_EQ", "REL_NE", "REL_LT", "REL_LE", "REL_GT", "REL_GE",
    "IndicatorPoly", "RationalPoly", "SymbolicRemainder",
]
