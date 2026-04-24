"""Polynomial over ℤ / 2³² (issue #78).

:class:`ModPoly` is a width-tracking variant of :class:`poly.Poly` with
integer coefficients reduced mod 2³² after every arithmetic op. Used by
the bit-fragment catalog entries where the WASM i32 wraparound matters
semantically — e.g. overflow-driven identities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Mapping, Tuple

from isa import MASK32
from poly import Poly, Monomial, _norm_coeff, _mono_mul, _mono_str

# ─── ModPoly — polynomial over ℤ / 2³² (issue #78) ────────────────
#
# Option (b) of issue #78: carry i32 wrap semantics through the
# polynomial algebra rather than only at the boundary via
# :func:`ff_symbolic.range_check`. ``ModPoly`` mirrors :class:`Poly` but
# reduces every coefficient modulo 2³² after every operation, matching
# the ``& MASK32`` the compiled transformer's FF layer applies to the
# results of ADD / SUB / MUL.
#
# The motivating gap: ``Poly`` arithmetic computes over ℤ, so the
# equivalence theorem from issue #69 carries a range assumption
# (:func:`ff_symbolic.range_check`) rather than a proof. Every catalog
# input happens to fit well inside ``[I32_MIN, I32_MAX]`` — max is
# ``factorial(10) = 3,628,800`` — so a bug in either direction on the
# overflow boundary would pass silently. ``ModPoly`` closes that:
# evaluations agree with ``NumPyExecutor`` bit-for-bit under wrap, and
# the equivalence theorem becomes *structural over ℤ/2³²* rather than
# *numeric on in-range inputs over ℤ*.
#
# Structural note: ℤ/2³² is a ring but not a field (2 divides the
# modulus, so the ring has zero divisors). That matters for DIV_S /
# REM_S — division is not well-defined — but issue #78 scope is only
# ADD / SUB / MUL, which are the ring operations. Comparisons /
# bitwise / rationals stay in their existing wrappers (``IndicatorPoly``,
# ``BitVec``, ``RationalPoly``) whose boundary evaluators already apply
# the appropriate wrap / truncation.

_MOD32 = 1 << 32
_I31 = 1 << 31  # signed / unsigned split


def _reduce_u32(c) -> int:
    """Reduce a coefficient (possibly negative / Fraction-with-denom-1) to [0, 2³²)."""
    if isinstance(c, Fraction):
        if c.denominator != 1:
            raise ValueError(
                f"ModPoly cannot carry non-integer coefficients; got {c}"
            )
        c = int(c.numerator)
    return int(c) & MASK32


@dataclass(frozen=True)
class ModPoly:
    """Multivariate polynomial with coefficients in ℤ / 2³².

    Sibling of :class:`Poly`. Closed under ``+``, ``-``, ``*`` (the ring
    operations); every operation reduces coefficients modulo 2³² so the
    canonical form is unique per congruence class. Structural equality
    (``==``) is therefore well-defined per the ring.

    Lift a pure-integer :class:`Poly` with :meth:`from_poly`; the result
    is the same polynomial under the ring homomorphism ℤ → ℤ/2³².
    Evaluation with :meth:`eval_at` agrees bit-for-bit with i32-masked
    integer evaluation of the lifted :class:`Poly` (the homomorphism
    property).
    """

    terms: Mapping[Monomial, int]

    @staticmethod
    def _normalise(terms: Mapping[Monomial, Union[int, Fraction]]
                   ) -> Dict[Monomial, int]:
        out: Dict[Monomial, int] = {}
        for m, c in terms.items():
            r = _reduce_u32(c)
            if r != 0:
                out[m] = r
        return out

    def __post_init__(self):
        object.__setattr__(self, "terms", self._normalise(dict(self.terms)))

    # ── Constructors ──────────────────────────────────────────

    @classmethod
    def constant(cls, c: int) -> "ModPoly":
        if _reduce_u32(c) == 0:
            return cls({})
        return cls({(): _reduce_u32(c)})

    @classmethod
    def variable(cls, idx: int) -> "ModPoly":
        return cls({((int(idx), 1),): 1})

    @classmethod
    def from_poly(cls, p: "Poly") -> "ModPoly":
        """Lift a :class:`Poly` with integer coefficients via ℤ → ℤ/2³².

        Raises ``ValueError`` on rational (non-integer) coefficients —
        those belong to the rational-extension fragment (``RationalPoly``
        from issue #75) whose boundary truncator already handles wrap.
        """
        return cls({m: _reduce_u32(c) for m, c in p.terms.items()})

    # ── Arithmetic ────────────────────────────────────────────
    #
    # The addition / subtraction / multiplication formulas are
    # identical to ``Poly``; what changes is the post-op coefficient
    # reduction in ``__post_init__``, which lands every result back
    # inside [0, 2³²).

    def __add__(self, other: "ModPoly") -> "ModPoly":
        out: Dict[Monomial, int] = dict(self.terms)
        for m, c in other.terms.items():
            out[m] = out.get(m, 0) + c
        return ModPoly(out)

    def __sub__(self, other: "ModPoly") -> "ModPoly":
        out: Dict[Monomial, int] = dict(self.terms)
        for m, c in other.terms.items():
            out[m] = out.get(m, 0) - c
        return ModPoly(out)

    def __neg__(self) -> "ModPoly":
        return ModPoly({m: -c for m, c in self.terms.items()})

    def __mul__(self, other: "ModPoly") -> "ModPoly":
        out: Dict[Monomial, int] = {}
        for ma, ca in self.terms.items():
            for mb, cb in other.terms.items():
                m = _mono_mul(ma, mb)
                out[m] = out.get(m, 0) + ca * cb
        return ModPoly(out)

    # ── Inspection ────────────────────────────────────────────

    def n_monomials(self) -> int:
        return len(self.terms)

    def variables(self) -> List[int]:
        seen = set()
        for m in self.terms:
            for v, _ in m:
                seen.add(v)
        return sorted(seen)

    def eval_at(self, bindings: Mapping[int, int]) -> int:
        """Evaluate as an integer in ``[0, 2³²)`` — the i32-wrapped result.

        Uses the ring homomorphism: evaluate over ℤ (Python's arbitrary-
        precision ints) and reduce once at the end. Equivalent to
        reducing after every multiply / add thanks to the homomorphism;
        the bulk form is simpler.
        """
        total = 0
        for mono, coeff in self.terms.items():
            term = coeff
            for v, p in mono:
                term *= int(bindings[v]) ** p
            total += term
        return int(total) & MASK32

    def eval_at_signed(self, bindings: Mapping[int, int]) -> int:
        """Evaluate and reinterpret as signed i32 (``[-2³¹, 2³¹)``)."""
        u = self.eval_at(bindings)
        return u - _MOD32 if u >= _I31 else u

    # ── Equality / display ────────────────────────────────────

    def __eq__(self, other) -> bool:
        if not isinstance(other, ModPoly):
            return NotImplemented
        return self.terms == other.terms

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.terms.items())))

    def __repr__(self) -> str:
        if not self.terms:
            return "0 (mod 2³²)"

        def _key(item):
            m, _ = item
            return (sum(p for _, p in m), m)

        def _signed(c: int) -> int:
            # Display coefficients near the upper boundary as negative
            # — so ``-1 mod 2³² = 4_294_967_295`` prints as ``-1`` instead
            # of the (correct but unreadable) u32 form. Makes catalog-
            # sized polynomials read identically to their ``Poly`` twin.
            return c - _MOD32 if c >= _I31 else c

        pieces = []
        for mono, coeff in sorted(self.terms.items(), key=_key):
            ms = _mono_str(mono)
            disp = _signed(coeff)
            if ms == "1":
                pieces.append(str(disp))
                continue
            if disp == 1:
                pieces.append(ms)
            elif disp == -1:
                pieces.append(f"-{ms}")
            else:
                pieces.append(f"{disp}·{ms}")
        out = pieces[0]
        for p in pieces[1:]:
            if p.startswith("-"):
                out += f" - {p[1:]}"
            else:
                out += f" + {p}"
        return f"{out} (mod 2³²)"



__all__ = ["ModPoly"]
