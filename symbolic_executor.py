"""Symbolic executor for LAC programs (issues #65 / #70).

Claim: for branchless straight-line programs over {PUSH, POP, DUP, SWAP,
OVER, ROT, ADD, SUB, MUL}, the k-instruction sequence executed by LAC's
k attention heads collapses to a single polynomial in the PUSH constants.
This module makes the collapse mechanical: walk the program, carry a
symbolic stack whose entries are `Poly` expressions, and emit whatever's
on top when HALT fires.

Each PUSH allocates a fresh symbolic variable (`x0`, `x1`, ...) so the
output generalises across the concrete PUSH constants. `eval_at` plugs
the real integers back in to verify against ``NumPyExecutor``.

Issue #70 extends the executor past straight-line code:

  - **Guarded traces.** On a JZ/JNZ whose condition polynomial still
    carries variables, the executor forks into two paths with
    complementary guards (``cond == 0`` vs ``cond != 0``) and carries
    them to HALT independently. Halted tops are combined into a
    ``GuardedPoly`` — a partitioned case table over the input domain.
  - **Bounded-loop unrolling.** Running in ``input_mode="concrete"``
    pushes the raw PUSH args onto the symbolic stack (no variables)
    so every branch collapses deterministically and loops unroll by
    normal execution. The polynomial has no variables in this mode;
    it's a single integer, honest for "unrolled at the catalog input".
  - **Loop-symbolic detection.** If a path revisits ``(pc, sp)`` on a
    back-edge while the controlling condition still has variables, the
    path halts with ``loop_symbolic`` — we don't attempt invariant
    inference.

Issue #75 lifts DIV_S / REM_S out of scope: they emit
:class:`RationalPoly` / :class:`SymbolicRemainder` boundary types so the
ring stays closed inside the executor and truncation lives at
``eval_at``. Issue #76 does the same for the comparison opcodes (EQ,
NE, LT_S, GT_S, LE_S, GE_S, EQZ): the result is an
:class:`IndicatorPoly` carrying ``poly`` + ``relation``, evaluated to
{0, 1} only at the boundary. JZ/JNZ on an :class:`IndicatorPoly` cond
hoists the relation directly into the resulting :class:`Guard` so a
``LT_S; JZ ...`` pair produces a ``GuardedPoly`` whose cases carry
``<`` / ``>=`` semantics — not just ``== 0`` / ``!= 0``.

Out of scope (file as follow-ups):
  - Bitwise opcodes (AND/OR/XOR/SHL/SHR_S/SHR_U/ROTL/ROTR).
  - Composition past one DIV_S/REM_S/comparison (e.g. ``LT_S; ADD``).
  - Loop-invariant inference for truly symbolic loops.
  - Locals, heap, memory — no symbolic address model yet.
  - Emitting W_Q/W_K/W_V themselves as expression trees (the issue's
    longer-horizon export story). The claim about composition collapse
    lives at the semantic level and doesn't need the tree export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union

import isa
from isa import (
    _clz32,
    _ctz32,
    _popcnt32,
    _rotl32,
    _rotr32,
    _shr_s,
    _shr_u,
    _to_i32,
    _trunc_div,
    _trunc_rem,
    MASK32,
)


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


# ─── Bit-vector AST (issue #77) ────────────────────────────────────
#
# Bitwise ops (AND/OR/XOR/SHL/SHR_S/SHR_U/CLZ/CTZ/POPCNT) are not
# polynomial over ℤ: AND/OR/XOR need (ℤ/2ℤ)[bits] to close, shifts need
# an exponent-lookup path for the shift amount, and CLZ/CTZ/POPCNT are
# piecewise. Instead of forcing them into the Poly ring, we carry a
# lightweight :class:`BitVec` AST through the stack — same boundary-step
# pattern :class:`RationalPoly` / :class:`IndicatorPoly` already use for
# DIV_S/REM_S (issue #75) and comparisons (issue #76).
#
# The AST is recursive: ``BitVec("AND", (poly_1, BitVec("SHR_U", (k, n))))``
# is the ``bit_extract(n, k)`` program's top, composing one bit op inside
# another. ``eval_at`` walks the tree bottom-up, applying the named op
# (``_apply_bitop``) at each node — the non-polynomial leaves live there,
# exactly on par with ``_trunc_div`` (DIV_S) and ``_relation_holds``
# (comparisons).
#
# ADD/SUB/MUL composed with a :class:`BitVec` operand (the ``log2_floor``
# case: ``SUB(31, CLZ(n))``) lift the arithmetic into the AST by encoding
# it as a ``BitVec("ADD"/"SUB"/"MUL", ...)`` node. This is the
# minimum-disruption way to cover the catalog's hybrid-arithmetic-and-bit
# programs without widening Poly itself.


_BIT_BINARY_OPS = {"AND", "OR", "XOR", "SHL", "SHR_S", "SHR_U", "ROTL", "ROTR"}
_BIT_UNARY_OPS = {"CLZ", "CTZ", "POPCNT"}
# Arithmetic lifted into the BitVec AST when one operand is already a BitVec
# (log2_floor's ``SUB(31, CLZ(n))`` case). Kept distinct from the bit ops
# so ``_apply_bitop`` can dispatch cleanly.
_BIT_ARITH_OPS = {"ADD", "SUB", "MUL"}
_BIT_OPS = _BIT_BINARY_OPS | _BIT_UNARY_OPS | _BIT_ARITH_OPS


def _apply_bitop(op: str, values: List[int]) -> int:
    """Apply a named bit / lifted-arithmetic op to concrete integer operands.

    The boundary nonlinearity for :class:`BitVec` nodes. ``values`` are
    the already-evaluated operand integers, in *natural* left-to-right
    reading order: for binary ops ``[left, right] = [SP-1, top]``, so the
    expression reads as ``left OP right`` (e.g. ``SUB`` is
    ``left − right`` = ``SP-1 − top`` = WASM ``i32.sub``'s ``vb − va``).
    Returns the i32-wrapped integer result.

    The named arithmetic ops (``ADD``, ``SUB``, ``MUL``) are here rather
    than in :class:`Poly` because they're used only when at least one
    operand is a :class:`BitVec` — the Poly-closed path never constructs
    them. Matches ``executor.py`` semantics.
    """
    if op == "AND":
        left, right = values
        return _to_i32(left) & _to_i32(right)
    if op == "OR":
        left, right = values
        return _to_i32(left) | _to_i32(right)
    if op == "XOR":
        left, right = values
        return _to_i32(left) ^ _to_i32(right)
    if op == "SHL":
        left, right = values
        return (_to_i32(left) << (int(right) & 31)) & MASK32
    if op == "SHR_S":
        left, right = values
        return _shr_s(left, right)
    if op == "SHR_U":
        left, right = values
        return _shr_u(left, right)
    if op == "ROTL":
        left, right = values
        return _rotl32(left, right)
    if op == "ROTR":
        left, right = values
        return _rotr32(left, right)
    if op == "CLZ":
        (v,) = values
        return _clz32(v)
    if op == "CTZ":
        (v,) = values
        return _ctz32(v)
    if op == "POPCNT":
        (v,) = values
        return _popcnt32(v)
    if op == "ADD":
        left, right = values
        return (int(left) + int(right)) & MASK32
    if op == "SUB":
        left, right = values
        return (int(left) - int(right)) & MASK32
    if op == "MUL":
        left, right = values
        return (int(left) * int(right)) & MASK32
    raise ValueError(f"unknown bit/arith op {op!r}")


# Display symbol for each op — used by :meth:`BitVec.__repr__`.
_BITVEC_DISPLAY = {
    "AND": "&", "OR": "|", "XOR": "^",
    "SHL": "<<", "SHR_S": ">>ₛ", "SHR_U": ">>ᵤ",
    "ROTL": "rotl", "ROTR": "rotr",
    "CLZ": "clz", "CTZ": "ctz", "POPCNT": "popcnt",
    "ADD": "+", "SUB": "-", "MUL": "·",
}


@dataclass(frozen=True)
class BitVec:
    """Symbolic i32 value from the bit-vector fragment (issue #77).

    Stored as an AST node with a named op and a tuple of operands. Each
    operand is a :class:`Poly` (a literal or variable) or another
    :class:`BitVec` (nested composition, like ``bit_extract``'s ``AND(1,
    SHR_U(k, n))`` top).

    The AST is never simplified — two ``BitVec("AND", (x, x))`` nodes compare
    equal, but they are not auto-rewritten to ``x``. A ring-level algebra
    over ``(ℤ/2ℤ)[bits]`` would let ``AND`` / ``OR`` / ``XOR`` cancel and
    absorb; that's a follow-up, not required to unblock the catalog's
    bitwise rows.

    :meth:`eval_at` is the boundary step: it evaluates each operand to a
    concrete int and then applies :func:`_apply_bitop`. The non-polynomial
    computation fires only at that boundary, matching the
    :class:`RationalPoly` / :class:`IndicatorPoly` design.

    :attr:`op` is one of the strings in :data:`_BIT_OPS` — the binary bit
    ops ``AND / OR / XOR / SHL / SHR_S / SHR_U / ROTL / ROTR``, the unary
    counters ``CLZ / CTZ / POPCNT``, or a *lifted* arithmetic op
    ``ADD / SUB / MUL`` (see module docstring for why arithmetic with a
    :class:`BitVec` operand lands here rather than widening :class:`Poly`).
    """

    op: str
    operands: Tuple[Union[Poly, "BitVec"], ...]

    def __post_init__(self):
        if self.op not in _BIT_OPS:
            raise ValueError(
                f"BitVec.op must be one of {sorted(_BIT_OPS)}, got {self.op!r}"
            )
        if self.op in _BIT_BINARY_OPS or self.op in _BIT_ARITH_OPS:
            expected = 2
        else:
            expected = 1
        if len(self.operands) != expected:
            raise ValueError(
                f"BitVec({self.op!r}) expects {expected} operand(s), "
                f"got {len(self.operands)}"
            )
        for o in self.operands:
            if not isinstance(o, (Poly, BitVec)):
                raise TypeError(
                    f"BitVec operand must be Poly or BitVec, got {type(o).__name__}"
                )

    def variables(self) -> List[int]:
        seen = set()
        for o in self.operands:
            seen.update(o.variables())
        return sorted(seen)

    def eval_at(self, bindings: Mapping[int, int]) -> int:
        """Evaluate the AST at concrete bindings. Recursively reduces
        each operand to an int and applies :func:`_apply_bitop`.
        """
        vals = [int(o.eval_at(bindings)) for o in self.operands]
        return _apply_bitop(self.op, vals)

    def __repr__(self) -> str:
        sym = _BITVEC_DISPLAY.get(self.op, self.op)
        if self.op in _BIT_UNARY_OPS:
            return f"{sym}({self.operands[0]})"
        a, b = self.operands
        return f"({a} {sym} {b})"


# Union covering every "top of symbolic stack" type run_symbolic /
# run_forking might emit for a branchless polynomial-plus-rational-plus-
# indicator program. Issue #76 added :class:`IndicatorPoly` to carry
# comparison results (EQ/NE/LT_S/GT_S/LE_S/GE_S/EQZ) through the stack.
# Issue #77 adds :class:`BitVec` for the bit-vector fragment (AND, OR,
# XOR, SHL, SHR_S, SHR_U, CLZ, CTZ, POPCNT).
RationalStackValue = Union[
    Poly, RationalPoly, SymbolicRemainder, IndicatorPoly, BitVec,
]

# "Int-valued AST" — any symbolic type that evaluates to an i32 integer
# at ``eval_at``. Used by the hybrid arithmetic hook (issue #77): ``ADD``
# / ``SUB`` / ``MUL`` on a :class:`BitVec` operand lifts the whole
# expression into the :class:`BitVec` AST rather than widening
# :class:`Poly`. The comparison primitives similarly accept any
# :class:`SymbolicIntAst` so ``is_power_of_2`` (``POPCNT; PUSH 1; EQ``)
# collapses cleanly.
SymbolicIntAst = Union[Poly, BitVec]


# ─── Guard + GuardedPoly ──────────────────────────────────────────
#
# A guard is a polynomial we assert satisfies one of six relations vs
# zero (``== / != / < / <= / > / >=``). A conjunction is a tuple of
# guards that must all hold simultaneously. GuardedPoly is a case
# table — one (conjunction, value_poly) entry per partition of the
# input domain.
#
# Guards are value-compared on (poly, relation), so two paths that
# derive the same guard chain in different orders merge cleanly after
# sorting. Issue #76 broadened the relation field from a binary
# ``eq_zero`` flag to the full six-relation set so that JZ/JNZ on an
# :class:`IndicatorPoly` cond produces guards that carry the comparison's
# semantics — not just an "== 0 / != 0" approximation. The
# :attr:`Guard.eq_zero` property survives as backwards-compat shorthand.


@dataclass(frozen=True)
class Guard:
    """Assertion ``poly REL 0`` for one of the six standard relations.

    ``relation`` is one of ``REL_EQ, REL_NE, REL_LT, REL_LE, REL_GT,
    REL_GE``. Pre-issue-#76 code only spoke of ``eq_zero=True/False``;
    that's preserved as a derived property for the dedup-by-(poly,
    eq_zero) call sites that have not been migrated. New code should
    construct guards via the relation directly.
    """
    poly: Poly
    relation: str

    def __post_init__(self):
        if self.relation not in _RELATIONS:
            raise ValueError(
                f"Guard.relation must be one of {_RELATIONS}, "
                f"got {self.relation!r}"
            )

    @property
    def eq_zero(self) -> bool:
        """Backwards-compat shim — True iff this guard asserts ``poly == 0``.

        Pre-#76 ``Guard`` only had ``eq_zero`` (True/False). The
        property keeps that read-side API working for the
        equality/inequality-only callers that haven't migrated to the
        full relation enum. Comparison-derived guards (``LT/LE/GT/GE``)
        return ``False`` here, since they don't assert equality with
        zero — callers that need to distinguish them should switch to
        ``g.relation``.
        """
        return self.relation == REL_EQ

    def holds_at(self, bindings: Mapping[int, int]) -> bool:
        """True iff ``poly REL 0`` holds at the given bindings."""
        return _relation_holds(self.relation, self.poly.eval_at(bindings))

    def __repr__(self) -> str:
        return f"({self.poly} {_REL_SYMBOL[self.relation]} 0)"


def _canonical_guards(guards: Tuple[Guard, ...]) -> Tuple[Guard, ...]:
    """Deduplicate + sort a guard conjunction for value-based equality."""
    # Use hash-based dedupe; Guard is frozen so hashable.
    return tuple(sorted(set(guards), key=lambda g: (repr(g.poly), g.relation)))


def _guards_complementary(a: Tuple[Guard, ...], b: Tuple[Guard, ...]) -> bool:
    """True iff a and b differ on exactly one guard by relation negation.

    Two relations are complementary if one is the other's :data:`_NEGATE_REL`
    image — i.e. ``EQ↔NE``, ``LT↔GE``, ``LE↔GT``. Used by callers that
    want to detect "these two cases together cover the full domain" merges.
    """
    if len(a) != len(b):
        return False
    diff = 0
    for ga, gb in zip(a, b):
        if ga == gb:
            continue
        if ga.poly == gb.poly and _NEGATE_REL[ga.relation] == gb.relation:
            diff += 1
        else:
            return False
    return diff == 1


@dataclass(frozen=True)
class GuardedPoly:
    """Partitioned case table: ``[(guards, value_poly), ...]``.

    Each case's ``guards`` tuple is a conjunction that must hold for
    that case's ``value_poly`` to apply. The set of cases is expected
    to partition the domain — i.e. for any concrete bindings, exactly
    one guard conjunction evaluates to True.
    """
    cases: Tuple[Tuple[Tuple[Guard, ...], Poly], ...]

    def __post_init__(self):
        canonical = tuple(
            (_canonical_guards(gs), v) for gs, v in self.cases
        )
        # Sort cases deterministically for equality.
        canonical = tuple(sorted(canonical,
                                 key=lambda c: (tuple(repr(g) for g in c[0]), repr(c[1]))))
        object.__setattr__(self, "cases", canonical)

    def n_cases(self) -> int:
        return len(self.cases)

    def variables(self) -> List[int]:
        seen = set()
        for gs, v in self.cases:
            for g in gs:
                seen.update(g.poly.variables())
            seen.update(v.variables())
        return sorted(seen)

    def eval_at(self, bindings: Mapping[int, int]) -> int:
        """Pick the unique case whose guards all hold at ``bindings``."""
        hits: List[int] = []
        for gs, v in self.cases:
            ok = True
            for g in gs:
                try:
                    val = g.poly.eval_at(bindings)
                except KeyError:
                    ok = False
                    break
                if not _relation_holds(g.relation, val):
                    ok = False
                    break
            if ok:
                hits.append(v.eval_at(bindings))
        if len(hits) != 1:
            raise ValueError(
                f"GuardedPoly.eval_at: {len(hits)} cases hit (expected 1) "
                f"at bindings={dict(bindings)}"
            )
        return hits[0]

    def __repr__(self) -> str:
        body = ", ".join(
            f"{{{' ∧ '.join(repr(g) for g in gs) or 'True'}}} → {v}"
            for gs, v in self.cases
        )
        return f"Guarded[{body}]"


# ─── SymbolicExecutor ──────────────────────────────────────────────

# Opcodes the *branchless* fragment understands — preserved for the
# legacy run_symbolic entry point that the original PoC tests exercise.
# Issue #75 adds DIV_S / REM_S: they break the polynomial-ring closure
# (outputs are :class:`RationalPoly` / :class:`SymbolicRemainder`) but
# the executor still accepts them as long as nothing downstream tries to
# compose a Poly op against a rational stack entry.
# Issue #76 adds the comparisons (EQ / NE / LT_S / GT_S / LE_S / GE_S /
# EQZ): they break the ring too, producing :class:`IndicatorPoly` tops.
# Same composition rule applies — the consuming op must be HALT or
# JZ/JNZ.
_CMP_BIN_OPS = {
    isa.OP_EQ, isa.OP_NE,
    isa.OP_LT_S, isa.OP_GT_S, isa.OP_LE_S, isa.OP_GE_S,
}
_CMP_UNARY_OPS = {isa.OP_EQZ}
_CMP_OPS = _CMP_BIN_OPS | _CMP_UNARY_OPS

# Per-op (binary) relation when wrapping ``IndicatorPoly(a - b, REL)``,
# where ``a = stack[SP-1]`` (the WASM ``vb``) and ``b = top`` (``va``).
# This matches :file:`executor.py:230-245` ("``1 if vb < va else 0``" etc.).
_BIN_OP_RELATION = {
    isa.OP_EQ: REL_EQ,
    isa.OP_NE: REL_NE,
    isa.OP_LT_S: REL_LT,
    isa.OP_GT_S: REL_GT,
    isa.OP_LE_S: REL_LE,
    isa.OP_GE_S: REL_GE,
}

_BIT_BIN_OPS = {
    isa.OP_AND, isa.OP_OR, isa.OP_XOR,
    isa.OP_SHL, isa.OP_SHR_S, isa.OP_SHR_U,
}
_BIT_UN_OPS = {isa.OP_CLZ, isa.OP_CTZ, isa.OP_POPCNT}
_BITVEC_OPCODES = _BIT_BIN_OPS | _BIT_UN_OPS

_POLY_OPS = {
    isa.OP_PUSH, isa.OP_POP, isa.OP_DUP, isa.OP_HALT,
    isa.OP_ADD, isa.OP_SUB, isa.OP_MUL,
    isa.OP_DIV_S, isa.OP_REM_S,
    isa.OP_SWAP, isa.OP_OVER, isa.OP_ROT, isa.OP_NOP,
} | _CMP_OPS | _BITVEC_OPCODES
# Branch ops the forking executor additionally handles.
_BRANCH_OPS = {isa.OP_JZ, isa.OP_JNZ}
# Union: what run_forking accepts.
_FORKING_OPS = _POLY_OPS | _BRANCH_OPS


@dataclass
class SymbolicResult:
    """Outcome of running a program symbolically.

    ``top`` is the top-of-stack value after HALT (or at the end of the
    trace if HALT is absent). For the ADD/SUB/MUL fragment it's a
    :class:`Poly`; for the DIV_S / REM_S rows added in issue #75 it can
    also be :class:`RationalPoly` or :class:`SymbolicRemainder`. ``stack``
    is the full final stack (bottom at index 0). ``n_heads`` is the
    number of instructions executed — the "k heads" the issue talks
    about. ``bindings`` maps the allocated variable indices back to the
    original PUSH constants.
    """

    top: RationalStackValue
    stack: List[RationalStackValue]
    n_heads: int
    bindings: Dict[int, int]

    def collapse_ratio(self) -> float:
        """k_heads ÷ n_monomials in the top expression, after simplification.

        Matches the issue's "9 heads → 1 monomial" style report. Returns
        ``inf`` when the top collapses to zero (no monomials) — flagged
        by callers who want a cleaner representation. For rational
        outputs (DIV_S / REM_S) the denominator counts as an additional
        monomial bundle; we sum the two sides' monomial counts to keep
        the "one number" shape of the ratio.
        """
        if isinstance(self.top, Poly):
            n = self.top.n_monomials()
        elif isinstance(self.top, (RationalPoly, SymbolicRemainder)):
            n = self.top.num.n_monomials() + self.top.denom.n_monomials()
        else:
            return float("inf")
        if n == 0:
            return float("inf")
        return self.n_heads / n


class SymbolicStackUnderflow(RuntimeError):
    pass


class SymbolicOpNotSupported(NotImplementedError):
    pass


class SymbolicLoopSymbolic(RuntimeError):
    """Raised when a path hits a back-edge whose cond is still symbolic."""
    pass


class SymbolicPathExplosion(RuntimeError):
    """Raised when the live path count exceeds the configured cap."""
    pass


def run_symbolic(prog: List[isa.Instruction]) -> SymbolicResult:
    """Execute ``prog`` symbolically. One variable allocated per PUSH.

    Polynomial composition happens eagerly so the final form is already
    simplified — no separate simplification pass needed.

    Branch ops raise :class:`SymbolicOpNotSupported` to preserve the
    issue-#65 "branchless only" contract. Use :func:`run_forking` for
    programs with JZ/JNZ.
    """
    stack: List[RationalStackValue] = []
    bindings: Dict[int, int] = {}
    next_var = 0
    n_heads = 0

    def _pop() -> RationalStackValue:
        if not stack:
            raise SymbolicStackUnderflow("pop from empty stack")
        return stack.pop()

    for instr in prog:
        op = instr.op
        arg = instr.arg
        if op not in _POLY_OPS:
            name = isa.OP_NAMES.get(op, f"?{op}")
            raise SymbolicOpNotSupported(
                f"op {name!r} is not polynomial-closed; issue #65 scope is "
                f"branchless straight-line programs"
            )
        if op == isa.OP_HALT:
            # HALT terminates the trace; the issue's head counts match
            # the number of instructions executed *before* HALT fires.
            break
        n_heads += 1
        if op == isa.OP_NOP:
            continue

        if op == isa.OP_PUSH:
            v = next_var
            next_var += 1
            bindings[v] = int(arg)
            stack.append(Poly.variable(v))
        elif op == isa.OP_POP:
            _pop()
        elif op == isa.OP_DUP:
            if not stack:
                raise SymbolicStackUnderflow("dup on empty stack")
            stack.append(stack[-1])
        elif op == isa.OP_ADD:
            b = _pop(); a = _pop()
            if isinstance(a, BitVec) or isinstance(b, BitVec):
                if not isinstance(a, (Poly, BitVec)) or not isinstance(b, (Poly, BitVec)):
                    raise SymbolicOpNotSupported(
                        "ADD mixing BitVec with rational/indicator entries is out of scope"
                    )
                stack.append(BitVec(op="ADD", operands=(a, b)))
            elif not isinstance(a, Poly) or not isinstance(b, Poly):
                raise SymbolicOpNotSupported(
                    "ADD on rational stack entries is out of scope "
                    "(composition past one DIV_S/REM_S is a follow-up)"
                )
            else:
                stack.append(a + b)
        elif op == isa.OP_SUB:
            b = _pop(); a = _pop()
            if isinstance(a, BitVec) or isinstance(b, BitVec):
                if not isinstance(a, (Poly, BitVec)) or not isinstance(b, (Poly, BitVec)):
                    raise SymbolicOpNotSupported(
                        "SUB mixing BitVec with rational/indicator entries is out of scope"
                    )
                stack.append(BitVec(op="SUB", operands=(a, b)))
            elif not isinstance(a, Poly) or not isinstance(b, Poly):
                raise SymbolicOpNotSupported(
                    "SUB on rational stack entries is out of scope "
                    "(composition past one DIV_S/REM_S is a follow-up)"
                )
            else:
                stack.append(a - b)
        elif op == isa.OP_MUL:
            b = _pop(); a = _pop()
            if isinstance(a, BitVec) or isinstance(b, BitVec):
                if not isinstance(a, (Poly, BitVec)) or not isinstance(b, (Poly, BitVec)):
                    raise SymbolicOpNotSupported(
                        "MUL mixing BitVec with rational/indicator entries is out of scope"
                    )
                stack.append(BitVec(op="MUL", operands=(a, b)))
            elif not isinstance(a, Poly) or not isinstance(b, Poly):
                raise SymbolicOpNotSupported(
                    "MUL on rational stack entries is out of scope "
                    "(composition past one DIV_S/REM_S is a follow-up)"
                )
            else:
                stack.append(a * b)
        elif op == isa.OP_DIV_S:
            b = _pop(); a = _pop()
            if not isinstance(a, Poly) or not isinstance(b, Poly):
                raise SymbolicOpNotSupported(
                    "DIV_S on rational stack entries is out of scope "
                    "(composition past one DIV_S/REM_S is a follow-up)"
                )
            stack.append(RationalPoly(num=a, denom=b))
        elif op == isa.OP_REM_S:
            b = _pop(); a = _pop()
            if not isinstance(a, Poly) or not isinstance(b, Poly):
                raise SymbolicOpNotSupported(
                    "REM_S on rational stack entries is out of scope "
                    "(composition past one DIV_S/REM_S is a follow-up)"
                )
            stack.append(SymbolicRemainder(num=a, denom=b))
        elif op in _CMP_BIN_OPS:
            b = _pop(); a = _pop()
            # WASM convention: ``a = stack[SP-1]`` (vb), ``b = top`` (va).
            # For Poly operands, wrap the difference as a Poly; for a
            # BitVec operand (``is_power_of_2``'s ``POPCNT; PUSH 1; EQ``)
            # lift the difference into the BitVec AST so the indicator can
            # still evaluate to {0, 1} at the boundary.
            if isinstance(a, Poly) and isinstance(b, Poly):
                stack.append(IndicatorPoly(poly=a - b,
                                           relation=_BIN_OP_RELATION[op]))
            elif isinstance(a, (Poly, BitVec)) and isinstance(b, (Poly, BitVec)):
                # Matches the Poly path's ``a - b`` = ``SP-1 - top``; the
                # relation table maps each opcode to the right comparison
                # against zero on that difference.
                stack.append(IndicatorPoly(
                    poly=BitVec(op="SUB", operands=(a, b)),
                    relation=_BIN_OP_RELATION[op],
                ))
            else:
                raise SymbolicOpNotSupported(
                    f"{isa.OP_NAMES[op]} on non-Poly stack entries is out "
                    "of scope (composition past one DIV_S/REM_S/comparison "
                    "is a follow-up)"
                )
        elif op == isa.OP_EQZ:
            a = _pop()
            if isinstance(a, (Poly, BitVec)):
                stack.append(IndicatorPoly(poly=a, relation=REL_EQ))
            else:
                raise SymbolicOpNotSupported(
                    "EQZ on non-Poly stack entries is out of scope "
                    "(composition past one DIV_S/REM_S/comparison is a follow-up)"
                )
        elif op == isa.OP_SWAP:
            if len(stack) < 2:
                raise SymbolicStackUnderflow("swap needs 2 entries")
            stack[-1], stack[-2] = stack[-2], stack[-1]
        elif op == isa.OP_OVER:
            if len(stack) < 2:
                raise SymbolicStackUnderflow("over needs 2 entries")
            stack.append(stack[-2])
        elif op == isa.OP_ROT:
            if len(stack) < 3:
                raise SymbolicStackUnderflow("rot needs 3 entries")
            # [a, b, c] -> [b, c, a] (matches test_algorithm semantics)
            a, b, c = stack[-3], stack[-2], stack[-1]
            stack[-3], stack[-2], stack[-1] = b, c, a
        elif op in _BIT_BIN_OPS:
            # Binary bit op: ``a = SP-1``, ``b = top``. BitVec wraps each
            # operand verbatim (no simplification) — see module docstring.
            b = _pop(); a = _pop()
            if not isinstance(a, (Poly, BitVec)) or not isinstance(b, (Poly, BitVec)):
                raise SymbolicOpNotSupported(
                    f"{isa.OP_NAMES[op]} on rational/indicator entries is out of scope"
                )
            stack.append(BitVec(op=isa.OP_NAMES[op], operands=(a, b)))
        elif op in _BIT_UN_OPS:
            a = _pop()
            if not isinstance(a, (Poly, BitVec)):
                raise SymbolicOpNotSupported(
                    f"{isa.OP_NAMES[op]} on rational/indicator entries is out of scope"
                )
            stack.append(BitVec(op=isa.OP_NAMES[op], operands=(a,)))
        else:  # pragma: no cover — guarded by _POLY_OPS
            raise SymbolicOpNotSupported(f"unreachable: op {op}")

    top = stack[-1] if stack else Poly.constant(0)
    return SymbolicResult(top=top, stack=list(stack),
                          n_heads=n_heads, bindings=bindings)


# ─── Arithmetic hook (issue #68 S3) ───────────────────────────────
#
# ``run_forking`` calls three arithmetic primitives in its inner loop —
# one each for ADD, SUB, MUL. The default primitives are ``Poly``'s
# native ``+ - *``; the forking driver is indifferent to which
# implementation is plugged in as long as the signature is ``(Poly,
# Poly) -> Poly`` and the algebra is the same.
#
# The hook exists so :mod:`ff_symbolic` can drive the same forking
# executor with its bilinear-FF interpretation of the primitives
# (``symbolic_add/sub/mul``) and demonstrate the equivalence claim
# from issue #69 extends across JZ/JNZ control flow — not just
# branchless straight-line programs.


ArithFn = Callable[["Poly", "Poly"], "Poly"]
DivFn = Callable[["Poly", "Poly"], "RationalPoly"]
RemFn = Callable[["Poly", "Poly"], "SymbolicRemainder"]
CmpBinFn = Callable[["Poly", "Poly"], "IndicatorPoly"]
CmpUnaryFn = Callable[["Poly"], "IndicatorPoly"]
# Issue #77: bit-vector primitives. Binary bit ops take
# ``(a, b) = (SP-1, top)`` (matching the arithmetic convention), each
# operand may already be a :class:`BitVec` (nested bit programs) or a
# plain :class:`Poly` (fresh values). Unary ops take a single
# :class:`SymbolicIntAst` operand. Hybrid-arithmetic ops
# (:data:`ArithmeticOps.bit_add` etc.) are invoked when at least one side
# is a :class:`BitVec` and take the same argument shape as ``ArithFn``,
# except they accept :class:`SymbolicIntAst` and return :class:`BitVec`.
BitBinFn = Callable[["SymbolicIntAst", "SymbolicIntAst"], "BitVec"]
BitUnaryFn = Callable[["SymbolicIntAst"], "BitVec"]
BitArithFn = Callable[["SymbolicIntAst", "SymbolicIntAst"], "BitVec"]


@dataclass(frozen=True)
class ArithmeticOps:
    """Operator spec the forking executor consumes for the arithmetic
    fragment of the ISA.

    ``sub(a, b)`` must return ``a - b`` (executor computes ``a - b``
    where ``a`` is the second-from-top, matching the existing Poly
    order). ``ff_symbolic.symbolic_sub`` matches this spec.

    ``div_s(a, b)`` / ``rem_s(a, b)`` (issue #75) must return the
    symbolic quotient / remainder of ``a / b`` under WASM
    ``i32.div_s`` / ``i32.rem_s`` semantics — ``a`` is stack[SP-1] (the
    dividend) and ``b`` is top (the divisor), matching the numeric
    path's ``_trunc_div(vb, va)`` convention (``executor.py:835-838``).

    ``cmp_eq / cmp_ne / cmp_lt_s / cmp_gt_s / cmp_le_s / cmp_ge_s``
    (issue #76) must return an :class:`IndicatorPoly` capturing
    "``vb REL va`` ⇔ ``a - b REL 0``" under the same ``a = SP-1, b =
    top`` convention. ``eqz(a)`` returns the unary "``a == 0``"
    indicator. The ``cmp(op)`` helper resolves an opcode to the right
    primitive — used by :func:`_apply_poly_op`.
    """
    add: ArithFn
    sub: ArithFn
    mul: ArithFn
    div_s: DivFn = None  # type: ignore[assignment]
    rem_s: RemFn = None  # type: ignore[assignment]
    cmp_eq: CmpBinFn = None  # type: ignore[assignment]
    cmp_ne: CmpBinFn = None  # type: ignore[assignment]
    cmp_lt_s: CmpBinFn = None  # type: ignore[assignment]
    cmp_gt_s: CmpBinFn = None  # type: ignore[assignment]
    cmp_le_s: CmpBinFn = None  # type: ignore[assignment]
    cmp_ge_s: CmpBinFn = None  # type: ignore[assignment]
    eqz: CmpUnaryFn = None  # type: ignore[assignment]
    # Issue #77: bit-vector primitives. The default (Poly-ring) path
    # builds :class:`BitVec` AST nodes; :mod:`ff_symbolic` overrides with
    # bilinear-FF versions that still produce :class:`BitVec` tops but
    # with values threaded through the residual stream.
    bit_and: BitBinFn = None  # type: ignore[assignment]
    bit_or: BitBinFn = None  # type: ignore[assignment]
    bit_xor: BitBinFn = None  # type: ignore[assignment]
    bit_shl: BitBinFn = None  # type: ignore[assignment]
    bit_shr_s: BitBinFn = None  # type: ignore[assignment]
    bit_shr_u: BitBinFn = None  # type: ignore[assignment]
    bit_clz: BitUnaryFn = None  # type: ignore[assignment]
    bit_ctz: BitUnaryFn = None  # type: ignore[assignment]
    bit_popcnt: BitUnaryFn = None  # type: ignore[assignment]
    # Lifted arithmetic for BitVec ⟷ Poly mixed operands (log2_floor case).
    bit_add: BitArithFn = None  # type: ignore[assignment]
    bit_sub: BitArithFn = None  # type: ignore[assignment]
    bit_mul: BitArithFn = None  # type: ignore[assignment]

    def cmp(self, op: int) -> Optional[CmpBinFn]:
        """Resolve an OP_* opcode to the matching binary-cmp primitive.

        Returns ``None`` if the relevant field is not wired — caller
        raises :class:`SymbolicOpNotSupported`.
        """
        return {
            isa.OP_EQ: self.cmp_eq,
            isa.OP_NE: self.cmp_ne,
            isa.OP_LT_S: self.cmp_lt_s,
            isa.OP_GT_S: self.cmp_gt_s,
            isa.OP_LE_S: self.cmp_le_s,
            isa.OP_GE_S: self.cmp_ge_s,
        }.get(op)

    def bit_binary(self, op: int) -> Optional[BitBinFn]:
        """Resolve an OP_* opcode to the matching binary bit primitive.

        Covers AND/OR/XOR/SHL/SHR_S/SHR_U. ROTL/ROTR aren't in the
        issue-#77 catalog scope but are representable via the same
        :class:`BitVec` AST if a future row needs them. Returns ``None``
        when the primitive isn't wired.
        """
        return {
            isa.OP_AND: self.bit_and,
            isa.OP_OR: self.bit_or,
            isa.OP_XOR: self.bit_xor,
            isa.OP_SHL: self.bit_shl,
            isa.OP_SHR_S: self.bit_shr_s,
            isa.OP_SHR_U: self.bit_shr_u,
        }.get(op)

    def bit_unary(self, op: int) -> Optional[BitUnaryFn]:
        """Resolve an OP_* opcode to the matching unary bit primitive.

        Covers CLZ / CTZ / POPCNT. Returns ``None`` when unwired.
        """
        return {
            isa.OP_CLZ: self.bit_clz,
            isa.OP_CTZ: self.bit_ctz,
            isa.OP_POPCNT: self.bit_popcnt,
        }.get(op)


DEFAULT_ARITHMETIC_OPS = ArithmeticOps(
    add=lambda a, b: a + b,
    sub=lambda a, b: a - b,
    mul=lambda a, b: a * b,
    div_s=lambda a, b: RationalPoly(num=a, denom=b),
    rem_s=lambda a, b: SymbolicRemainder(num=a, denom=b),
    cmp_eq=lambda a, b: IndicatorPoly(poly=a - b, relation=REL_EQ),
    cmp_ne=lambda a, b: IndicatorPoly(poly=a - b, relation=REL_NE),
    cmp_lt_s=lambda a, b: IndicatorPoly(poly=a - b, relation=REL_LT),
    cmp_gt_s=lambda a, b: IndicatorPoly(poly=a - b, relation=REL_GT),
    cmp_le_s=lambda a, b: IndicatorPoly(poly=a - b, relation=REL_LE),
    cmp_ge_s=lambda a, b: IndicatorPoly(poly=a - b, relation=REL_GE),
    eqz=lambda a: IndicatorPoly(poly=a, relation=REL_EQ),
    bit_and=lambda a, b: BitVec(op="AND", operands=(a, b)),
    bit_or=lambda a, b: BitVec(op="OR", operands=(a, b)),
    bit_xor=lambda a, b: BitVec(op="XOR", operands=(a, b)),
    bit_shl=lambda a, b: BitVec(op="SHL", operands=(a, b)),
    bit_shr_s=lambda a, b: BitVec(op="SHR_S", operands=(a, b)),
    bit_shr_u=lambda a, b: BitVec(op="SHR_U", operands=(a, b)),
    bit_clz=lambda a: BitVec(op="CLZ", operands=(a,)),
    bit_ctz=lambda a: BitVec(op="CTZ", operands=(a,)),
    bit_popcnt=lambda a: BitVec(op="POPCNT", operands=(a,)),
    bit_add=lambda a, b: BitVec(op="ADD", operands=(a, b)),
    bit_sub=lambda a, b: BitVec(op="SUB", operands=(a, b)),
    bit_mul=lambda a, b: BitVec(op="MUL", operands=(a, b)),
)


# ─── Forking executor (issue #70) ─────────────────────────────────

# Default caps. Catalog programs stay under these comfortably; the
# symbolic-loop exit trips long before we get close.
DEFAULT_MAX_PATHS = 64
DEFAULT_MAX_STEPS = 50_000


def _as_concrete_int(p: "RationalStackValue") -> Optional[int]:
    """Return integer value if ``p`` has no variables; else None.

    Rational stack values (``RationalPoly`` / ``SymbolicRemainder``) never
    collapse to a concrete int for branching purposes — DIV_S/REM_S past a
    subsequent JZ/JNZ is out of scope for issue #75, so return ``None``
    and let the caller fall into the symbolic-cond path (which will then
    raise when it tries to wrap the value in a Guard).

    :class:`IndicatorPoly` *can* collapse: when its inner ``poly`` is
    concrete, the indicator evaluates to 0 or 1 deterministically and
    JZ/JNZ should follow the branch without forking. Issue #76 needs this
    so ``compare_lt_s(3, 5)`` (concrete inputs) collapses straight to
    ``1`` even when consumed by a later branch.
    """
    if isinstance(p, IndicatorPoly):
        inner = _as_concrete_int(p.poly)
        if inner is None:
            return None
        return 1 if _relation_holds(p.relation, inner) else 0
    if isinstance(p, BitVec):
        # Bit-vector fragment (issue #77). A BitVec with no free
        # variables reduces to an i32 literal via :meth:`eval_at`, so
        # JZ/JNZ on it takes the concretely-decided branch rather than
        # forking. Matches the IndicatorPoly(Poly) path above and
        # unblocks ``popcount_loop(n)`` at concrete ``n``.
        if p.variables():
            return None
        return int(p.eval_at({}))
    if not isinstance(p, Poly):
        return None
    if not p.terms:
        return 0
    if len(p.terms) == 1 and () in p.terms:
        return int(p.terms[()])
    for mono in p.terms:
        if mono:
            return None
    # Only the constant monomial appears (possibly not present).
    return int(p.terms.get((), 0))


@dataclass
class _Path:
    """One symbolic execution thread.

    ``visited_branches`` records ``(pc, sp)`` pairs where this path took
    a symbolic branch; revisiting such a pair with a still-symbolic cond
    at the same site is the loop-symbolic signal.

    Variables are indexed by the PC of the PUSH instruction that
    allocated them, so forked paths sharing a prefix also share the
    variable ids of pre-fork PUSHes — and post-fork PUSHes at distinct
    static sites get distinct ids even across paths.
    """
    pc: int
    stack: Tuple["RationalStackValue", ...]
    guards: Tuple[Guard, ...]
    bindings: Dict[int, int]
    n_heads: int = 0
    visited_branches: frozenset = field(default_factory=frozenset)
    loop_unrolled: bool = False  # True if this path ever took a back-edge
    halted_top: Optional["RationalStackValue"] = None

    def with_(self, **kwargs) -> "_Path":
        """Return a copy with selected fields replaced."""
        base = dict(
            pc=self.pc, stack=self.stack, guards=self.guards,
            bindings=dict(self.bindings),
            n_heads=self.n_heads, visited_branches=self.visited_branches,
            loop_unrolled=self.loop_unrolled, halted_top=self.halted_top,
        )
        base.update(kwargs)
        return _Path(**base)


@dataclass
class ForkingResult:
    """Outcome of running a program via the forking executor.

    ``top`` collapses:
      - to a single :class:`Poly` if all halted paths agree;
      - to a :class:`GuardedPoly` when paths disagree;
      - to ``None`` if no path halted (loop_symbolic-only outcome).

    ``status`` is one of ``"straight" | "guarded" | "unrolled" |
    "loop_symbolic" | "path_explosion" | "blocked_underflow"``.
    """

    top: Optional[Union[Poly, GuardedPoly]]
    status: str
    n_heads: int                  # max k_heads across halted paths
    bindings: Dict[int, int]      # union of all paths' bindings
    n_halted: int = 0
    n_loop_symbolic: int = 0
    paths_explored: int = 0


def _eq_guard(p: Poly, eq_zero: bool) -> Guard:
    """Backwards-compat shim: build an EQ/NE guard from a bool flag.

    Pre-#76 callers built guards with ``eq_zero=True/False``. New code
    should construct :class:`Guard` with the relation directly.
    """
    return Guard(poly=p, relation=REL_EQ if eq_zero else REL_NE)


def _branch_guards(cond: "RationalStackValue", op: int) -> Tuple[Guard, Guard]:
    """Build (take_guard, skip_guard) for a JZ/JNZ on a symbolic ``cond``.

    For a plain :class:`Poly` cond, JZ takes when ``cond == 0`` (skip
    when ``cond != 0``); JNZ flips. For an :class:`IndicatorPoly` cond
    with ``(poly, R)``, "cond == 0" ⇔ "``poly`` does NOT satisfy R" ⇔
    "``poly (negate R) 0``", so we hoist the comparison's polynomial
    and relation into the guard rather than wrapping the indicator
    inside one — that way the resulting :class:`GuardedPoly` carries
    LT/LE/GT/GE guards directly and matches the semantics the catalog
    needs to render with the right ``<``/``<=``/``>``/``>=`` symbols.

    Raises :class:`SymbolicOpNotSupported` for cond types we can't gate
    on (RationalPoly / SymbolicRemainder — DIV_S/REM_S past JZ/JNZ
    isn't in scope yet).
    """
    if isinstance(cond, IndicatorPoly):
        # cond == 0  ⇔  not (poly R 0)  ⇔  poly (negate R) 0
        zero_relation = _NEGATE_REL[cond.relation]
        nonzero_relation = cond.relation
        zero_guard = Guard(poly=cond.poly, relation=zero_relation)
        nonzero_guard = Guard(poly=cond.poly, relation=nonzero_relation)
    elif isinstance(cond, Poly):
        zero_guard = Guard(poly=cond, relation=REL_EQ)
        nonzero_guard = Guard(poly=cond, relation=REL_NE)
    else:
        # BitVec / RationalPoly / SymbolicRemainder with free variables:
        # out of scope. Concrete-mode BitVec conds are handled earlier
        # by :func:`_as_concrete_int` (issue #77); this branch only
        # fires for truly symbolic non-Poly conds.
        raise SymbolicOpNotSupported(
            f"JZ/JNZ on a {type(cond).__name__} cond is out of scope; "
            "branching past DIV_S/REM_S is a follow-up"
        )
    # JZ: take when cond == 0. JNZ: take when cond != 0.
    if op == isa.OP_JZ:
        return zero_guard, nonzero_guard
    return nonzero_guard, zero_guard


def run_forking(prog: List[isa.Instruction], *,
                input_mode: str = "symbolic",
                max_paths: int = DEFAULT_MAX_PATHS,
                max_steps: int = DEFAULT_MAX_STEPS,
                arithmetic_ops: Optional[ArithmeticOps] = None) -> ForkingResult:
    """Forking symbolic executor with finite-conditional + bounded-loop support.

    ``input_mode``:
      - ``"symbolic"``: each PUSH allocates a fresh variable. Branches on
        symbolic conditions fork the path. Suitable for ``collapsed_guarded``.
      - ``"concrete"``: each PUSH pushes its literal arg (no variables).
        All branches collapse deterministically; loops unroll naturally.
        Suitable for ``collapsed_unrolled``.

    ``arithmetic_ops``: override the ADD/SUB/MUL primitives applied to
    Poly stack entries. Defaults to :data:`DEFAULT_ARITHMETIC_OPS` (plain
    Poly ``+ - *``). :mod:`ff_symbolic` passes its bilinear-FF
    interpretation here (issue #68 S3) to demonstrate equivalence across
    control flow.

    The executor uses a worklist. Each fork splits the path into two new
    paths carrying complementary guards. When a path's top polynomial is
    concrete at a branch, the branch is followed deterministically. A
    symbolic back-edge that revisits ``(pc, sp)`` halts the path with
    ``loop_symbolic``.
    """
    if input_mode not in ("symbolic", "concrete"):
        raise ValueError(f"unknown input_mode {input_mode!r}")
    ops = arithmetic_ops if arithmetic_ops is not None else DEFAULT_ARITHMETIC_OPS

    # Pre-flight: reject programs with non-polynomial, non-branch opcodes.
    for instr in prog:
        if instr.op not in _FORKING_OPS:
            name = isa.OP_NAMES.get(instr.op, f"?{instr.op}")
            raise SymbolicOpNotSupported(
                f"op {name!r} is out of scope for the forking executor "
                f"(polynomial + JZ/JNZ only)"
            )

    init = _Path(
        pc=0, stack=(), guards=(),
        bindings={}, n_heads=0,
        visited_branches=frozenset(), loop_unrolled=False,
    )
    worklist: List[_Path] = [init]
    halted: List[_Path] = []
    loop_symbolic_paths: List[_Path] = []
    paths_explored = 0
    total_steps = 0
    underflow_seen = False

    def _spawn(new: _Path):
        if len(worklist) + len(halted) + len(loop_symbolic_paths) + 1 > max_paths:
            raise SymbolicPathExplosion(
                f"path count exceeds max_paths={max_paths}"
            )
        worklist.append(new)

    try:
        while worklist:
            path = worklist.pop()
            paths_explored += 1
            # Step this path until it halts, forks, or loops symbolically.
            while True:
                total_steps += 1
                if total_steps > max_steps:
                    raise SymbolicPathExplosion(
                        f"total step count exceeds max_steps={max_steps}"
                    )
                if path.pc < 0 or path.pc >= len(prog):
                    # Implicit fall-off-end acts as HALT with current top.
                    path = path.with_(
                        halted_top=path.stack[-1] if path.stack
                        else Poly.constant(0),
                    )
                    halted.append(path)
                    break
                instr = prog[path.pc]
                op = instr.op
                if op == isa.OP_HALT:
                    path = path.with_(
                        halted_top=path.stack[-1] if path.stack
                        else Poly.constant(0),
                    )
                    halted.append(path)
                    break

                # Non-branch, non-halt → advance n_heads and apply op.
                if op != isa.OP_JZ and op != isa.OP_JNZ:
                    try:
                        stack = _apply_poly_op(path, instr, input_mode, ops)
                    except SymbolicStackUnderflow:
                        underflow_seen = True
                        # drop this path; don't propagate partial result
                        break
                    new_bindings = path.bindings
                    if op == isa.OP_PUSH and input_mode == "symbolic":
                        # Variable id = PUSH's pc (stable across forked paths).
                        new_bindings = dict(path.bindings)
                        new_bindings[path.pc] = int(instr.arg)
                    path = path.with_(
                        pc=path.pc + 1,
                        stack=stack,
                        n_heads=path.n_heads + (0 if op == isa.OP_NOP else 1),
                        bindings=new_bindings,
                    )
                    continue

                # JZ / JNZ: pop cond, decide branch.
                if not path.stack:
                    underflow_seen = True
                    break
                cond = path.stack[-1]
                popped_stack = path.stack[:-1]
                path = path.with_(
                    n_heads=path.n_heads + 1,
                    stack=popped_stack,
                )
                sp = len(popped_stack)
                target = int(instr.arg)
                fall_through = path.pc + 1
                is_back_edge = target <= path.pc

                concrete = _as_concrete_int(cond)
                if concrete is not None:
                    taken = (concrete == 0) if op == isa.OP_JZ else (concrete != 0)
                    new_pc = target if taken else fall_through
                    path = path.with_(
                        pc=new_pc,
                        loop_unrolled=path.loop_unrolled or (is_back_edge and taken),
                    )
                    continue

                # Symbolic condition → fork. Check for symbolic back-edge revisit.
                site = (path.pc, sp, op)
                if is_back_edge and site in path.visited_branches:
                    # This path already forked at this back-edge once with a
                    # symbolic cond — seeing it again means no progress.
                    loop_symbolic_paths.append(path)
                    break
                new_visited = path.visited_branches | {site}

                # _branch_guards understands both bare-Poly and
                # IndicatorPoly conds — for the latter it hoists the
                # comparison's relation directly into the guards.
                take_guard, skip_guard = _branch_guards(cond, op)

                take_path = path.with_(
                    pc=target,
                    guards=path.guards + (take_guard,),
                    visited_branches=new_visited,
                )
                skip_path = path.with_(
                    pc=fall_through,
                    guards=path.guards + (skip_guard,),
                    visited_branches=new_visited,
                )
                _spawn(skip_path)
                _spawn(take_path)
                break  # current thread is replaced by the two new ones
    except SymbolicPathExplosion:
        # Collect what we have and report.
        return ForkingResult(
            top=None, status="path_explosion",
            n_heads=max((p.n_heads for p in halted + loop_symbolic_paths), default=0),
            bindings={}, n_halted=len(halted),
            n_loop_symbolic=len(loop_symbolic_paths),
            paths_explored=paths_explored,
        )

    # Combine halted tops.
    if not halted:
        if loop_symbolic_paths:
            return ForkingResult(
                top=None, status="loop_symbolic", n_heads=0,
                bindings={}, n_halted=0,
                n_loop_symbolic=len(loop_symbolic_paths),
                paths_explored=paths_explored,
            )
        if underflow_seen:
            return ForkingResult(
                top=None, status="blocked_underflow", n_heads=0,
                bindings={}, n_halted=0, n_loop_symbolic=0,
                paths_explored=paths_explored,
            )
        return ForkingResult(
            top=None, status="blocked_underflow", n_heads=0,
            bindings={}, n_halted=0, n_loop_symbolic=0,
            paths_explored=paths_explored,
        )

    merged_bindings: Dict[int, int] = {}
    for p in halted:
        merged_bindings.update(p.bindings)

    tops = [(p.guards, p.halted_top) for p in halted]
    # Determine status.
    any_forked = any(p.guards for p in halted)
    any_looped = any(p.loop_unrolled for p in halted) or bool(loop_symbolic_paths)

    # Build the top: a single Poly if all agree and no guards, else GuardedPoly.
    unique_values = {v for _, v in tops}
    if not any_forked and len(unique_values) == 1:
        top_val: Union[Poly, GuardedPoly] = next(iter(unique_values))
    else:
        top_val = _build_guarded_poly(tops)

    if loop_symbolic_paths and not halted:
        status = "loop_symbolic"
    elif loop_symbolic_paths:
        # Partial collapse: some paths halted, others hit symbolic loops.
        status = "loop_symbolic"
    elif any_looped:
        status = "unrolled"
    elif any_forked:
        status = "guarded"
    else:
        status = "straight"

    n_heads = max((p.n_heads for p in halted), default=0)
    return ForkingResult(
        top=top_val, status=status, n_heads=n_heads,
        bindings=merged_bindings, n_halted=len(halted),
        n_loop_symbolic=len(loop_symbolic_paths),
        paths_explored=paths_explored,
    )


def _apply_poly_op(path: _Path, instr: isa.Instruction,
                   input_mode: str,
                   arithmetic_ops: ArithmeticOps = DEFAULT_ARITHMETIC_OPS) -> Tuple[RationalStackValue, ...]:
    """Apply a non-branch opcode to ``path.stack`` and return the new stack.

    ``arithmetic_ops`` picks the ADD/SUB/MUL/DIV_S/REM_S implementations.
    Defaults to Poly's native operators (plus RationalPoly/SymbolicRemainder
    wrappers for DIV_S/REM_S); :mod:`ff_symbolic` passes its bilinear-FF
    primitives.
    """
    op = instr.op
    stack = list(path.stack)

    def _pop() -> RationalStackValue:
        if not stack:
            raise SymbolicStackUnderflow(f"pop from empty stack at pc={path.pc}")
        return stack.pop()

    if op == isa.OP_PUSH:
        if input_mode == "symbolic":
            stack.append(Poly.variable(path.pc))
        else:
            stack.append(Poly.constant(int(instr.arg)))
    elif op == isa.OP_POP:
        _pop()
    elif op == isa.OP_DUP:
        if not stack:
            raise SymbolicStackUnderflow(f"dup on empty stack at pc={path.pc}")
        stack.append(stack[-1])
    elif op == isa.OP_ADD:
        b = _pop(); a = _pop()
        if isinstance(a, BitVec) or isinstance(b, BitVec):
            if not isinstance(a, (Poly, BitVec)) or not isinstance(b, (Poly, BitVec)):
                raise SymbolicOpNotSupported(
                    "ADD mixing BitVec with rational/indicator entries is out of scope"
                )
            if arithmetic_ops.bit_add is None:
                raise SymbolicOpNotSupported(
                    "arithmetic_ops.bit_add is not wired; pass a bit_add primitive"
                )
            stack.append(arithmetic_ops.bit_add(a, b))
        elif not isinstance(a, Poly) or not isinstance(b, Poly):
            raise SymbolicOpNotSupported(
                "ADD on rational stack entries is out of scope "
                "(composition past one DIV_S/REM_S is a follow-up)"
            )
        else:
            stack.append(arithmetic_ops.add(a, b))
    elif op == isa.OP_SUB:
        b = _pop(); a = _pop()
        if isinstance(a, BitVec) or isinstance(b, BitVec):
            if not isinstance(a, (Poly, BitVec)) or not isinstance(b, (Poly, BitVec)):
                raise SymbolicOpNotSupported(
                    "SUB mixing BitVec with rational/indicator entries is out of scope"
                )
            if arithmetic_ops.bit_sub is None:
                raise SymbolicOpNotSupported(
                    "arithmetic_ops.bit_sub is not wired; pass a bit_sub primitive"
                )
            stack.append(arithmetic_ops.bit_sub(a, b))
        elif not isinstance(a, Poly) or not isinstance(b, Poly):
            raise SymbolicOpNotSupported(
                "SUB on rational stack entries is out of scope "
                "(composition past one DIV_S/REM_S is a follow-up)"
            )
        else:
            stack.append(arithmetic_ops.sub(a, b))
    elif op == isa.OP_MUL:
        b = _pop(); a = _pop()
        if isinstance(a, BitVec) or isinstance(b, BitVec):
            if not isinstance(a, (Poly, BitVec)) or not isinstance(b, (Poly, BitVec)):
                raise SymbolicOpNotSupported(
                    "MUL mixing BitVec with rational/indicator entries is out of scope"
                )
            if arithmetic_ops.bit_mul is None:
                raise SymbolicOpNotSupported(
                    "arithmetic_ops.bit_mul is not wired; pass a bit_mul primitive"
                )
            stack.append(arithmetic_ops.bit_mul(a, b))
        elif not isinstance(a, Poly) or not isinstance(b, Poly):
            raise SymbolicOpNotSupported(
                "MUL on rational stack entries is out of scope "
                "(composition past one DIV_S/REM_S is a follow-up)"
            )
        else:
            stack.append(arithmetic_ops.mul(a, b))
    elif op == isa.OP_DIV_S:
        b = _pop(); a = _pop()
        if not isinstance(a, Poly) or not isinstance(b, Poly):
            raise SymbolicOpNotSupported(
                "DIV_S on rational stack entries is out of scope "
                "(composition past one DIV_S/REM_S is a follow-up)"
            )
        if arithmetic_ops.div_s is None:
            raise SymbolicOpNotSupported(
                "arithmetic_ops.div_s is not wired; pass a div_s primitive"
            )
        stack.append(arithmetic_ops.div_s(a, b))
    elif op == isa.OP_REM_S:
        b = _pop(); a = _pop()
        if not isinstance(a, Poly) or not isinstance(b, Poly):
            raise SymbolicOpNotSupported(
                "REM_S on rational stack entries is out of scope "
                "(composition past one DIV_S/REM_S is a follow-up)"
            )
        if arithmetic_ops.rem_s is None:
            raise SymbolicOpNotSupported(
                "arithmetic_ops.rem_s is not wired; pass a rem_s primitive"
            )
        stack.append(arithmetic_ops.rem_s(a, b))
    elif op in _CMP_BIN_OPS:
        b = _pop(); a = _pop()
        if isinstance(a, Poly) and isinstance(b, Poly):
            cmp_fn = arithmetic_ops.cmp(op)
            if cmp_fn is None:
                raise SymbolicOpNotSupported(
                    f"arithmetic_ops.cmp({isa.OP_NAMES[op]}) is not wired"
                )
            stack.append(cmp_fn(a, b))
        elif isinstance(a, (Poly, BitVec)) and isinstance(b, (Poly, BitVec)):
            # is_power_of_2 composes POPCNT (BitVec) with PUSH 1 (Poly).
            # Matches the Poly path's ``a - b`` = ``SP-1 - top`` difference.
            stack.append(IndicatorPoly(
                poly=BitVec(op="SUB", operands=(a, b)),
                relation=_BIN_OP_RELATION[op],
            ))
        else:
            raise SymbolicOpNotSupported(
                f"{isa.OP_NAMES[op]} on non-Poly stack entries is out "
                "of scope (composition past one DIV_S/REM_S/comparison "
                "is a follow-up)"
            )
    elif op == isa.OP_EQZ:
        a = _pop()
        if isinstance(a, Poly):
            if arithmetic_ops.eqz is None:
                raise SymbolicOpNotSupported(
                    "arithmetic_ops.eqz is not wired; pass an eqz primitive"
                )
            stack.append(arithmetic_ops.eqz(a))
        elif isinstance(a, BitVec):
            stack.append(IndicatorPoly(poly=a, relation=REL_EQ))
        else:
            raise SymbolicOpNotSupported(
                "EQZ on non-Poly stack entries is out of scope "
                "(composition past one DIV_S/REM_S/comparison is a follow-up)"
            )
    elif op == isa.OP_SWAP:
        if len(stack) < 2:
            raise SymbolicStackUnderflow(f"swap needs 2 entries at pc={path.pc}")
        stack[-1], stack[-2] = stack[-2], stack[-1]
    elif op == isa.OP_OVER:
        if len(stack) < 2:
            raise SymbolicStackUnderflow(f"over needs 2 entries at pc={path.pc}")
        stack.append(stack[-2])
    elif op == isa.OP_ROT:
        if len(stack) < 3:
            raise SymbolicStackUnderflow(f"rot needs 3 entries at pc={path.pc}")
        a, b, c = stack[-3], stack[-2], stack[-1]
        stack[-3], stack[-2], stack[-1] = b, c, a
    elif op in _BIT_BIN_OPS:
        b = _pop(); a = _pop()
        if not isinstance(a, (Poly, BitVec)) or not isinstance(b, (Poly, BitVec)):
            raise SymbolicOpNotSupported(
                f"{isa.OP_NAMES[op]} on rational/indicator entries is out of scope"
            )
        bit_fn = arithmetic_ops.bit_binary(op)
        if bit_fn is None:
            raise SymbolicOpNotSupported(
                f"arithmetic_ops.bit_binary({isa.OP_NAMES[op]}) is not wired"
            )
        stack.append(bit_fn(a, b))
    elif op in _BIT_UN_OPS:
        a = _pop()
        if not isinstance(a, (Poly, BitVec)):
            raise SymbolicOpNotSupported(
                f"{isa.OP_NAMES[op]} on rational/indicator entries is out of scope"
            )
        bit_fn = arithmetic_ops.bit_unary(op)
        if bit_fn is None:
            raise SymbolicOpNotSupported(
                f"arithmetic_ops.bit_unary({isa.OP_NAMES[op]}) is not wired"
            )
        stack.append(bit_fn(a))
    elif op == isa.OP_NOP:
        pass
    else:  # pragma: no cover
        raise SymbolicOpNotSupported(f"op {op} unexpected in _apply_poly_op")

    return tuple(stack)


def _build_guarded_poly(
    tops: List[Tuple[Tuple[Guard, ...], Poly]],
) -> Union[Poly, GuardedPoly]:
    """Merge per-path ``(guards, value)`` pairs into a single GuardedPoly.

    Paths with the same value polynomial are combined by merging their
    guard chains. If all paths produce the same value *and* their guards
    together span the full domain (the trivial case of a single path
    with empty guards), we return the bare Poly.
    """
    # Group by value poly.
    by_value: Dict[Poly, List[Tuple[Guard, ...]]] = {}
    for gs, v in tops:
        by_value.setdefault(v, []).append(_canonical_guards(gs))

    # If only one distinct value and at least one path has no guards, it's unconditional.
    if len(by_value) == 1:
        sole_value = next(iter(by_value))
        guard_chains = by_value[sole_value]
        if any(len(gs) == 0 for gs in guard_chains):
            return sole_value

    cases: List[Tuple[Tuple[Guard, ...], Poly]] = []
    for value, guard_chains in by_value.items():
        for gs in guard_chains:
            cases.append((gs, value))
    return GuardedPoly(cases=tuple(cases))


# ─── Reporting helper ─────────────────────────────────────────────

def collapse_report(prog: List[isa.Instruction], *,
                    name: str = "") -> str:
    """Run ``prog`` symbolically and return a one-line collapse summary.

    Example::

        PUSH 5; DUP;ADD;DUP;ADD;DUP;ADD;DUP;ADD  →  9 heads, 1 monomial, top = 16·x0
    """
    r = run_symbolic(prog)
    prefix = f"{name}: " if name else ""
    return (f"{prefix}{r.n_heads} heads → {r.top.n_monomials()} "
            f"monomials, top = {r.top}")


def guarded_to_mermaid(gp: "GuardedPoly") -> str:
    """Render a ``GuardedPoly`` case table as a Mermaid flowchart decision tree.

    Returns valid Mermaid ``flowchart TD`` source. The last case is rendered
    as an implicit ``else`` leaf — its guards are implied by the preceding
    decisions — so a 2-case / single-guard GuardedPoly produces exactly one
    decision diamond and two value leaves.

    Multi-guard cases chain their guards with ``True`` edges; the ``False``
    edge of each case's last guard connects to the next case.
    """
    def _label(g: "Guard") -> str:
        return f"{g.poly} {_REL_SYMBOL[g.relation]} 0"

    def _mq(s: str) -> str:
        return '"' + s.replace('"', "'") + '"'

    lines = ["flowchart TD"]
    ctr = [0]

    def _fresh(prefix: str) -> str:
        ctr[0] += 1
        return f"{prefix}{ctr[0]}"

    cases = list(gp.cases)
    n = len(cases)
    pending: Optional[Tuple[str, str]] = None  # (from_node_id, edge_label)

    for i, (guards, value) in enumerate(cases):
        guards_list = list(guards)

        if i == n - 1:
            # Last case: render as implied else leaf (guards follow by elimination).
            leaf = _fresh("L")
            lines.append(f"    {leaf}[{_mq(repr(value))}]")
            if pending:
                src, lbl = pending
                lines.append(f"    {src} -->|{lbl}| {leaf}")
            break

        if not guards_list:
            # Unconditional middle case (degenerate; shouldn't appear in a valid partition).
            leaf = _fresh("L")
            lines.append(f"    {leaf}[{_mq(repr(value))}]")
            if pending:
                src, lbl = pending
                lines.append(f"    {src} -->|{lbl}| {leaf}")
            pending = None
            continue

        # Chain each guard in the conjunction with True edges between them.
        prev_dec: Optional[str] = None
        for j, g in enumerate(guards_list):
            dec = _fresh("D")
            lines.append(f"    {dec}{{{_mq(_label(g))}}}")
            if j == 0 and pending:
                src, lbl = pending
                lines.append(f"    {src} -->|{lbl}| {dec}")
            elif j > 0 and prev_dec is not None:
                lines.append(f"    {prev_dec} -->|True| {dec}")
            prev_dec = dec

        # True branch of the last guard leads to this case's value leaf.
        leaf = _fresh("L")
        lines.append(f"    {leaf}[{_mq(repr(value))}]")
        lines.append(f"    {prev_dec} -->|True| {leaf}")

        # False branch of the last guard connects to the next case.
        pending = (prev_dec, "False")

    return "\n".join(lines)


__all__ = [
    "Poly",
    "ModPoly",
    "RationalPoly",
    "SymbolicRemainder",
    "IndicatorPoly",
    "BitVec",
    "SymbolicIntAst",
    "Guard",
    "GuardedPoly",
    "SymbolicResult",
    "ForkingResult",
    "SymbolicStackUnderflow",
    "SymbolicOpNotSupported",
    "SymbolicLoopSymbolic",
    "SymbolicPathExplosion",
    "ArithmeticOps",
    "DEFAULT_ARITHMETIC_OPS",
    "DEFAULT_MAX_PATHS",
    "DEFAULT_MAX_STEPS",
    "REL_EQ", "REL_NE", "REL_LT", "REL_LE", "REL_GT", "REL_GE",
    "run_symbolic",
    "run_forking",
    "collapse_report",
    "guarded_to_mermaid",
]
