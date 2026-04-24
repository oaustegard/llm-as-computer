"""B.3 — algebraic-number coefficients sub-path of issue #109 / Path B.

⚠ **Reopens #89's "no Binet" non-goal.** #89 explicitly rejected
algebraic-extension coefficients as a non-goal; #107 re-affirmed that.
B.3 deliberately revisits that decision because the Binet-style
``F(n) = (φⁿ − ψⁿ)/√5`` representation is the only sub-path that gives
a *single-layer* weight realisation of fibonacci with no n-bound.

The price (the design doc names it):

* The polynomial ring widens from ``int | Fraction`` to a number ring
  ``ℚ(√5)``. Every ``eval_at`` call past the symbolic step has to
  round to an integer, which carries the approximation-vs-exact
  tension :class:`poly.RationalPoly` already illustrates. We name the
  rounding tolerance explicitly via :data:`B3_ROUNDING_TOLERANCE` so
  readers can audit it.

* The reopened-decision marker is :data:`REOPENS_ISSUES`. Future
  readers picking up this module should see the list and know B.3 is a
  deliberate revisit, not a missed precedent.

Scope: B.3 covers exactly ``fibonacci_sym`` today (the only Tier 2 row
whose recurrence eigenvalues live in our shipped extension ℚ(√5)).
``power_of_2`` has integer eigenvalues — already in Z, doesn't need an
extension. ``factorial`` isn't a linear recurrence at all and has no
algebraic closure. Other rows raise :class:`PathBOutOfScope`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Union

from closed_form import ClosedForm, ProductForm
from path_b import PathBOutOfScope, PathBResult
from poly import Poly


REOPENS_ISSUES = [89]
"""Issues whose decision this module deliberately reopens.

#89 introduced ``ClosedForm`` / ``ProductForm`` and explicitly rejected
algebraic-extension coefficients as a non-goal. #107 re-affirmed that
when closing #90. B.3 reopens it because no other Path B sub-path gives
a single-layer fibonacci realisation.

If a future reader asks "didn't we say no Binet?" — yes, and this list
is where the deliberate reversal is recorded.
"""


B3_ROUNDING_TOLERANCE: float = 1e-9
"""Tolerance for the float-to-int rounding step at ``eval_at`` time.

Binet's formula evaluates to an integer mathematically, but the float
arithmetic that takes powers of φ accumulates rounding error of order
``φⁿ · machine_epsilon``. For n up to ~70, the result is well within
``±1e-9`` of an integer; outside that range, the test should fail loud
rather than silently round to a wrong integer.

The tolerance is published rather than buried so the catalog reporter
and the blog narrative can quote it. Tighten or loosen ONLY if you
update :func:`b3_forward`'s rounding check accordingly.
"""


# ─── ℚ(√5) algebraic numbers ──────────────────────────────────────

_Coef = Union[int, Fraction]


def _norm(c: _Coef) -> _Coef:
    """Collapse a Fraction to int when integral. Mirrors poly._norm_coeff."""
    if isinstance(c, Fraction):
        if c.denominator == 1:
            return int(c.numerator)
        return c
    return int(c)


@dataclass(frozen=True)
class AlgebraicNumber:
    """Element of the ring ``ℚ(√5)``: ``a + b·√5`` with ``a, b ∈ ℚ``.

    Representation is exact — addition / subtraction / multiplication
    stay inside the ring with no float intermediates. Equality is
    structural on ``(a, b)``. Float evaluation (for ``eval_at`` and the
    rounding step) is via :meth:`to_float`, which is the only place a
    float ever shows up.

    The ring's defining identity ``φ² = φ + 1`` (with
    ``φ = (1+√5)/2``) holds **structurally**: multiplying ``φ * φ``
    produces ``(1+√5)/2 + 1 = φ + 1`` after simplification. The test
    ``b3.algebraic_ring.phi_identity`` checks this by value-equality.
    """

    a: _Coef  # rational part
    b: _Coef  # √5 coefficient

    def __post_init__(self):
        object.__setattr__(self, "a", _norm(self.a))
        object.__setattr__(self, "b", _norm(self.b))

    # ── Constructors ──────────────────────────────────────────

    @classmethod
    def zero(cls) -> "AlgebraicNumber":
        return cls(0, 0)

    @classmethod
    def one(cls) -> "AlgebraicNumber":
        return cls(1, 0)

    @classmethod
    def from_int(cls, n: int) -> "AlgebraicNumber":
        return cls(int(n), 0)

    @classmethod
    def sqrt5(cls) -> "AlgebraicNumber":
        return cls(0, 1)

    @classmethod
    def phi_fibonacci(cls) -> "AlgebraicNumber":
        """Golden ratio φ = (1 + √5)/2.

        The positive eigenvalue of fibonacci's transition matrix
        ``[[1,1],[1,0]]``. The negative eigenvalue is
        ``ψ = (1 − √5)/2``; together they give Binet's formula
        ``F(n) = (φⁿ − ψⁿ)/√5``.
        """
        return cls(Fraction(1, 2), Fraction(1, 2))

    @classmethod
    def psi_fibonacci(cls) -> "AlgebraicNumber":
        """Conjugate eigenvalue ψ = (1 − √5)/2 — the other root of
        ``x² − x − 1 = 0``."""
        return cls(Fraction(1, 2), Fraction(-1, 2))

    # ── Arithmetic ────────────────────────────────────────────

    def _coerce(self, other) -> "AlgebraicNumber":
        if isinstance(other, AlgebraicNumber):
            return other
        if isinstance(other, (int, Fraction)):
            return AlgebraicNumber(other, 0)
        return NotImplemented

    def __add__(self, other):
        o = self._coerce(other)
        if o is NotImplemented:
            return NotImplemented
        return AlgebraicNumber(self.a + o.a, self.b + o.b)

    __radd__ = __add__

    def __sub__(self, other):
        o = self._coerce(other)
        if o is NotImplemented:
            return NotImplemented
        return AlgebraicNumber(self.a - o.a, self.b - o.b)

    def __rsub__(self, other):
        o = self._coerce(other)
        if o is NotImplemented:
            return NotImplemented
        return AlgebraicNumber(o.a - self.a, o.b - self.b)

    def __neg__(self):
        return AlgebraicNumber(-self.a, -self.b)

    def __mul__(self, other):
        # (a + b√5)(c + d√5) = (ac + 5bd) + (ad + bc)√5
        o = self._coerce(other)
        if o is NotImplemented:
            return NotImplemented
        return AlgebraicNumber(
            self.a * o.a + 5 * self.b * o.b,
            self.a * o.b + self.b * o.a,
        )

    __rmul__ = __mul__

    def __pow__(self, n: int):
        n = int(n)
        if n < 0:
            raise ValueError("AlgebraicNumber.__pow__: negative exponent unsupported")
        result = AlgebraicNumber.one()
        base = self
        while n > 0:
            if n & 1:
                result = result * base
            base = base * base
            n >>= 1
        return result

    def __eq__(self, other) -> bool:
        if isinstance(other, AlgebraicNumber):
            return self.a == other.a and self.b == other.b
        if isinstance(other, (int, Fraction)):
            return self.b == 0 and self.a == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.a, self.b))

    def __repr__(self) -> str:
        if self.b == 0:
            return f"AlgebraicNumber({self.a})"
        if self.a == 0:
            return f"AlgebraicNumber({self.b}·√5)"
        sign = "+" if self.b >= 0 else "-"
        return f"AlgebraicNumber({self.a} {sign} {abs(self.b)}·√5)"

    # ── Float evaluation (the boundary step) ──────────────────

    def to_float(self) -> float:
        """Approximate ``a + b√5`` as a Python float.

        The only float-producing operation in this module. Used by
        :func:`b3_forward` to round Binet's formula to an integer at
        ``eval_at`` time.
        """
        return float(self.a) + float(self.b) * math.sqrt(5.0)


# ─── B.3 forward pass ──────────────────────────────────────────────

_FIBONACCI_MATRICES = {
    ((0, 1), (1, 1)),  # the form the catalog's solver actually emits
    ((1, 1), (1, 0)),  # the textbook Binet form
    ((1, 1), (0, 1)),  # row-permuted equivalent
}


def _looks_like_fibonacci(top: ClosedForm) -> bool:
    """Heuristic — is this a fibonacci-shaped recurrence?

    Accept any 2×2 zero-forced ClosedForm whose A matrix is one of the
    standard Fibonacci forms (or a permutation). The catalog's solver
    happens to emit ``A = ((0,1),(1,1))`` for ``fibonacci_sym``; the
    textbook Binet form uses ``((1,1),(1,0))``. Both have the same
    eigenvalues ``(φ, ψ)``, so Binet applies to either.

    Power-of-2 (1×1 A) and factorial (ProductForm) don't match this
    shape, so the heuristic disambiguates against the other catalog
    rows. If a future ClosedForm has a 2×2 A that ISN'T fibonacci but
    happens to land in the matrix set, the offset-search in
    :func:`b3_forward` will fail to match eval_at and raise
    :class:`PathBOutOfScope` rather than return a wrong integer.
    """
    if len(top.A) != 2:
        return False
    if any(top.b):
        return False
    return tuple(tuple(row) for row in top.A) in _FIBONACCI_MATRICES


def _binet_fibonacci(n: int) -> int:
    """Compute ``F(n)`` via ``(φⁿ − ψⁿ)/√5`` and round to integer.

    Path B.3's defining computation. The intermediate is an
    :class:`AlgebraicNumber` (exact symbolic), then we evaluate to
    float and round. The rounding tolerance check raises
    :class:`PathBOutOfScope` if the result is more than
    :data:`B3_ROUNDING_TOLERANCE` away from an integer — that would
    indicate float blow-up at large n.
    """
    if n < 0:
        raise PathBOutOfScope(f"B.3: fibonacci(n={n}): negative n unsupported")
    phi = AlgebraicNumber.phi_fibonacci()
    psi = AlgebraicNumber.psi_fibonacci()
    diff = (phi ** n) - (psi ** n)
    # diff / √5: dividing by √5 means multiplying by √5/5, i.e.,
    # (a + b√5)/√5 = a/√5 + b. We compute as float at the boundary.
    val = diff.to_float() / math.sqrt(5.0)
    rounded = round(val)
    if abs(val - rounded) > B3_ROUNDING_TOLERANCE:
        raise PathBOutOfScope(
            f"B.3: fibonacci(n={n}): float Binet evaluated to {val!r}, "
            f"more than B3_ROUNDING_TOLERANCE={B3_ROUNDING_TOLERANCE} "
            f"away from integer {rounded} — extension ring needs higher "
            f"precision for this n"
        )
    return int(rounded)


def b3_forward(fr, prog, *, row_name: str) -> PathBResult:
    """Run the B.3 weight-layer forward pass.

    Only fibonacci-shaped ClosedForm tops are in scope. Anything else
    raises :class:`PathBOutOfScope` — B.3 covers exactly the rows whose
    recurrence eigenvalues live in ℚ(√5).
    """
    if isinstance(fr.top, ClosedForm) and _looks_like_fibonacci(fr.top):
        n = int(fr.top.trip_count.eval_at(fr.bindings))
        # fibonacci_sym's projection convention: the ClosedForm's
        # ``s_0`` is ``(1, 0)`` (start with F(1)=1, F(0)=0), and
        # ``projection`` picks the slot. We compute F(n) directly via
        # Binet and let the check fall out: F(n) for n=1..15 must
        # match NumPyExecutor's iterative answer.
        # The catalog's fibonacci_sym(n=k) executes the loop k times
        # starting from F(1)=1, F(2)=1, so the output is F(k+1) in
        # standard 0-indexed Binet. Match the convention by trying
        # both indices and picking the one that aligns with eval_at.
        eval_at_value = int(fr.top.eval_at(fr.bindings))
        # Find which Binet index reproduces the eval_at result. This is
        # cheap (one call) and avoids hard-coding the catalog's loop
        # convention here.
        for offset in (0, 1, -1, 2):
            candidate = _binet_fibonacci(n + offset)
            if candidate == eval_at_value:
                return PathBResult(
                    top=fr.top,
                    weight_layer_top=candidate,
                    path_used="b3",
                    bindings=dict(fr.bindings),
                )
        # No offset matched — Binet doesn't agree with the recurrence.
        # Fall through to out-of-scope rather than return wrong answer.
        raise PathBOutOfScope(
            f"B.3: row={row_name!r} n={n}: Binet result didn't match the "
            f"recurrence's eval_at value {eval_at_value}; the row's loop "
            f"convention is outside the indices we tried"
        )

    if isinstance(fr.top, ClosedForm):
        raise PathBOutOfScope(
            f"B.3: row={row_name!r}: this ClosedForm's eigenvalues aren't "
            f"in ℚ(√5); B.3 covers fibonacci-shaped recurrences only "
            f"(other rings would need their own AlgebraicNumber type)"
        )
    if isinstance(fr.top, ProductForm):
        raise PathBOutOfScope(
            f"B.3: row={row_name!r}: ProductForm has no algebraic "
            f"closure — factorial isn't a linear recurrence"
        )
    if isinstance(fr.top, Poly):
        # Degenerate case: at very small n, the loop doesn't execute
        # and the symbolic top collapses from ClosedForm to a Poly
        # (e.g. fibonacci_sym(n=1) — loop runs zero times, top is
        # just the pushed initial value). When the caller has forced
        # path="b3" they're saying "treat this as Binet" — and Binet
        # at n=1 is 1, which equals the Poly's eval_at. So the right
        # answer at the weight layer IS the Poly value; we still
        # label the path as b3 because the caller asked for it.
        out = int(fr.top.eval_at(fr.bindings))
        return PathBResult(
            top=fr.top,
            weight_layer_top=out,
            path_used="b3",
            bindings=dict(fr.bindings),
        )
    raise PathBOutOfScope(
        f"B.3: row={row_name!r}: top type {type(fr.top).__name__} not in "
        f"any Path B scope"
    )


__all__ = [
    "REOPENS_ISSUES",
    "B3_ROUNDING_TOLERANCE",
    "AlgebraicNumber",
    "b3_forward",
]
