"""Polynomial over integer-indexed symbolic variables (issue #65).

See :class:`Poly` for the canonical representation. This is the
foundational stack-value type for LAC's symbolic executor: every PUSH
allocates a fresh variable, and ADD/SUB/MUL compose :class:`Poly`
operands via the bilinear forms here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Mapping, Tuple

# ─── Poly ──────────────────────────────────────────────────────────
#
# Polynomial over integer-indexed symbolic variables with rational
# (``int`` or :class:`fractions.Fraction`) coefficients. Canonical form:
# ``terms`` is a dict keyed by a monomial (tuple of (var_idx, power)
# pairs, sorted by var_idx, powers > 0). The empty tuple ``()`` is the
# constant monomial. Zero-coefficient terms are dropped on construction
# so comparisons are value-equal when the polynomials are mathematically
# equal.
#
# Rational coefficients land via issue #75 (symbolic DIV_S / REM_S): the
# bilinear forms for ADD/SUB/MUL produce polynomials over ℤ, but DIV_S
# introduces rational polynomials (``a/b`` with integer ``a, b``). To keep
# one canonical type, coefficients accept ``int | Fraction`` and
# normalise to ``int`` whenever the denominator is 1 — so every Poly
# produced by ADD/SUB/MUL still has literal ``int`` coefficients and
# existing structural-equality tests remain green.

Monomial = Tuple[Tuple[int, int], ...]


def _norm_coeff(c):
    """Normalise a coefficient to ``int`` when integral, else ``Fraction``.

    Accepts ``int`` or ``Fraction`` input. The canonical form keeps
    integer coefficients as ``int`` so value-compare against the
    pre-#75 Polys still works and ``repr`` output stays unchanged for
    the ADD/SUB/MUL fragment.
    """
    if isinstance(c, Fraction):
        if c.denominator == 1:
            return int(c.numerator)
        return c
    return int(c)


def _mono_mul(a: Monomial, b: Monomial) -> Monomial:
    """Merge two monomials. Powers add; result is sorted by var index."""
    if not a:
        return b
    if not b:
        return a
    merged: Dict[int, int] = {}
    for v, p in a:
        merged[v] = merged.get(v, 0) + p
    for v, p in b:
        merged[v] = merged.get(v, 0) + p
    return tuple(sorted(merged.items()))


def _mono_str(mono: Monomial) -> str:
    if not mono:
        return "1"
    parts = []
    for v, p in mono:
        parts.append(f"x{v}" if p == 1 else f"x{v}^{p}")
    return "·".join(parts)


@dataclass(frozen=True)
class Poly:
    """Multivariate polynomial with rational coefficients.

    ``terms`` maps a monomial (canonical-form tuple) to its coefficient,
    which is ``int`` when the coefficient is integral and
    :class:`fractions.Fraction` when the denominator is >1. Zero-
    coefficient entries are never stored.
    """

    terms: Mapping[Monomial, Union[int, Fraction]]

    @staticmethod
    def _normalise(terms: Mapping[Monomial, Union[int, Fraction]]
                   ) -> Dict[Monomial, Union[int, Fraction]]:
        return {m: _norm_coeff(c) for m, c in terms.items() if c != 0}

    def __post_init__(self):
        # Freeze a normalised copy. Doing it this way so callers can pass
        # any mapping and still get the value-equality guarantee.
        object.__setattr__(self, "terms", self._normalise(dict(self.terms)))

    # ── Constructors ──────────────────────────────────────────

    @classmethod
    def constant(cls, c: Union[int, Fraction]) -> "Poly":
        if c == 0:
            return cls({})
        return cls({(): _norm_coeff(c)})

    @classmethod
    def variable(cls, idx: int) -> "Poly":
        return cls({((int(idx), 1),): 1})

    # ── Arithmetic ────────────────────────────────────────────

    def __add__(self, other: "Poly") -> "Poly":
        out: Dict[Monomial, int] = dict(self.terms)
        for m, c in other.terms.items():
            out[m] = out.get(m, 0) + c
        return Poly(out)

    def __sub__(self, other: "Poly") -> "Poly":
        out: Dict[Monomial, int] = dict(self.terms)
        for m, c in other.terms.items():
            out[m] = out.get(m, 0) - c
        return Poly(out)

    def __neg__(self) -> "Poly":
        return Poly({m: -c for m, c in self.terms.items()})

    def __mul__(self, other: "Poly") -> "Poly":
        out: Dict[Monomial, int] = {}
        for ma, ca in self.terms.items():
            for mb, cb in other.terms.items():
                m = _mono_mul(ma, mb)
                out[m] = out.get(m, 0) + ca * cb
        return Poly(out)

    # ── Inspection ────────────────────────────────────────────

    def n_monomials(self) -> int:
        return len(self.terms)

    def variables(self) -> List[int]:
        """Variable indices referenced by any monomial, sorted."""
        seen = set()
        for m in self.terms:
            for v, _ in m:
                seen.add(v)
        return sorted(seen)

    def eval_at(self, bindings: Mapping[int, int]) -> Union[int, Fraction]:
        """Substitute ``bindings[i]`` for each ``x_i`` and reduce.

        Returns ``int`` when the result is integral (the common case for
        ADD/SUB/MUL Polys), otherwise returns :class:`fractions.Fraction`
        (after a DIV_S introduces a rational coefficient). Missing
        variables raise ``KeyError`` — symbolic executors that emit a
        variable per PUSH should pass one binding per PUSH.
        """
        total: Union[int, Fraction] = 0
        for mono, coeff in self.terms.items():
            term: Union[int, Fraction] = coeff
            for v, p in mono:
                term *= bindings[v] ** p
            total += term
        return _norm_coeff(total)

    # ── Equality / display ────────────────────────────────────

    def __eq__(self, other) -> bool:
        if not isinstance(other, Poly):
            return NotImplemented
        return self.terms == other.terms

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.terms.items())))

    def __repr__(self) -> str:  # deterministic for tests
        if not self.terms:
            return "0"
        # sort by (total degree, monomial) for readable output
        def _key(item):
            m, _ = item
            return (sum(p for _, p in m), m)

        pieces = []
        for mono, coeff in sorted(self.terms.items(), key=_key):
            ms = _mono_str(mono)
            if ms == "1":
                pieces.append(str(coeff))
                continue
            if coeff == 1:
                pieces.append(ms)
            elif coeff == -1:
                pieces.append(f"-{ms}")
            else:
                pieces.append(f"{coeff}·{ms}")
        # join with explicit signs
        out = pieces[0]
        for p in pieces[1:]:
            if p.startswith("-"):
                out += f" - {p[1:]}"
            else:
                out += f" + {p}"
        return out



__all__ = ["Poly", "Monomial"]
