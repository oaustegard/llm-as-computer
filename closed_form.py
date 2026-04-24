"""Loop closed forms (issue #89).

:class:`ClosedForm` and :class:`ProductForm` carry the recurrence
solver's output through to :class:`forking_executor.ForkingResult`.
The solver lives in :mod:`forking_executor`; this module just provides
the carrier types.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Mapping, Tuple

from poly import Poly

# ─── Loop closed forms (issue #89) ─────────────────────────────────
#
# Symbolic loops whose body is a linear recurrence on the loop-carried
# stack slice, and whose trip count is a linear Poly in the input
# variables, no longer have to halt with ``loop_symbolic``. Instead the
# solver in :func:`run_forking` walks the body, classifies the
# transition, and emits a :class:`ClosedForm` (Tier 2 — linear
# recurrences with constant integer matrix) or :class:`ProductForm`
# (Tier 3 — multiplicative recurrences over a Poly factor) sibling.
# Tier 1 (affine-polynomial recurrences) stays inside :class:`Poly`
# via Faulhaber's formula — no sibling needed.
#
# Same design as the other sibling types: the polynomial ring stays
# closed inside the executor; the non-polynomial step (integer matrix
# exponentiation for Tier 2, bounded product for Tier 3) fires only at
# ``eval_at`` time. Structural equality is value-based on the stored
# recurrence so two executors that emit the same loop shape produce
# equal tops.


@dataclass(frozen=True)
class ClosedForm:
    """Closed form for a linear recurrence with constant integer coefficients.

    Encodes ``s_{k+1} = A · s_k + b`` where ``A`` is an ``m×m`` integer
    matrix, ``b`` an integer vector of length ``m``, and ``s_0`` the
    initial state (each entry a :class:`Poly` in the input variables).
    ``trip_count`` is the symbolic number of iterations (a :class:`Poly`
    resolving to a non-negative integer at binding time), and
    ``projection`` picks the scalar slot that ``eval_at`` returns.

    ``eval_at`` evaluates numerically: resolve ``trip_count`` and ``s_0``
    to integers, compute ``Aⁿ`` via recursive squaring (pure Python —
    no numpy dependency), apply to ``s_0``, add the geometric-series
    correction for ``b``, and return the projected slot.

    Structural-equality contract matches :class:`BitVec`: two
    ClosedForms with the same ``(A, b, s_0, trip_count, projection)``
    are ``==``. No algebraic simplification (e.g. Binet's formula for
    Fibonacci) — the ring doesn't close over eigenvalues, so we keep
    the recurrence structure symbolic and evaluate numerically.
    """
    A: Tuple[Tuple[int, ...], ...]
    b: Tuple[int, ...]
    s_0: Tuple[Poly, ...]
    trip_count: Poly
    projection: int = 0

    def variables(self) -> List[int]:
        seen = set(self.trip_count.variables())
        for p in self.s_0:
            seen.update(p.variables())
        return sorted(seen)

    def eval_at(self, bindings: Mapping[int, int]) -> int:
        n = int(self.trip_count.eval_at(bindings))
        if n < 0:
            raise ValueError(
                f"ClosedForm.eval_at: trip_count < 0 "
                f"(trip_count={self.trip_count!r} at bindings={dict(bindings)})"
            )
        m = len(self.s_0)
        if not self.A or len(self.A) != m or any(len(row) != m for row in self.A):
            raise ValueError(
                f"ClosedForm: A must be {m}×{m}; got {self.A!r}"
            )
        if len(self.b) != m:
            raise ValueError(
                f"ClosedForm: b must be length {m}; got {self.b!r}"
            )
        s0 = [int(p.eval_at(bindings)) for p in self.s_0]
        # Resolve s_n = A^n · s_0 + sum_{k=0}^{n-1} A^k · b by iterative
        # evaluation. O(n · m²) is fine for the catalog's n ≤ 32 range;
        # matrix power squaring would buy nothing at these sizes and
        # complicates the geometric-series correction.
        state = list(s0)
        for _ in range(n):
            nxt = [
                sum(self.A[i][j] * state[j] for j in range(m)) + self.b[i]
                for i in range(m)
            ]
            state = nxt
        if not 0 <= self.projection < m:
            raise IndexError(
                f"ClosedForm.projection={self.projection} out of range [0, {m})"
            )
        return int(state[self.projection])

    def __repr__(self) -> str:
        return (
            f"ClosedForm(A={self.A}, b={self.b}, "
            f"s_0={tuple(repr(p) for p in self.s_0)}, "
            f"trip={self.trip_count!r}, proj={self.projection})"
        )


@dataclass(frozen=True)
class ProductForm:
    """Closed form for a multiplicative recurrence ``acc ← acc · p(k)``.

    Encodes ``acc_N = init · ∏_{k=lower}^{upper} p(k_var)`` where
    ``p`` is a :class:`Poly` (the per-step factor in the counter
    variable), ``counter_var`` identifies the variable in ``p`` that
    takes the counter's successive values, ``lower`` / ``upper`` are
    inclusive bounds as :class:`Poly` (symbolic in the input
    variables), and ``init`` is the product identity (usually 1).

    ``eval_at`` resolves ``lower``, ``upper``, and every non-counter
    variable in ``p`` to integers, then walks the counter from
    ``lower`` to ``upper`` inclusive, multiplying into the accumulator.
    Empty range (``lower > upper``) returns ``init`` — matches Python's
    empty-product convention.

    Treated as a separate sibling from :class:`ClosedForm` because ``∏``
    is neither a linear recurrence in Poly (the state grows in degree
    each step) nor a polynomial in the trip count.
    """
    p: Poly
    counter_var: int
    lower: Poly
    upper: Poly
    init: int = 1

    def variables(self) -> List[int]:
        seen = set(self.lower.variables()) | set(self.upper.variables())
        for v in self.p.variables():
            if v != self.counter_var:
                seen.add(v)
        return sorted(seen)

    def eval_at(self, bindings: Mapping[int, int]) -> int:
        lo = int(self.lower.eval_at(bindings))
        hi = int(self.upper.eval_at(bindings))
        acc = int(self.init)
        for k in range(lo, hi + 1):
            b = dict(bindings)
            b[self.counter_var] = k
            acc *= int(self.p.eval_at(b))
        return acc

    def __repr__(self) -> str:
        return (
            f"ProductForm(p={self.p!r}, k=x{self.counter_var}, "
            f"lower={self.lower!r}, upper={self.upper!r}, init={self.init})"
        )



__all__ = ["ClosedForm", "ProductForm"]
