"""Guard + GuardedPoly — partition-table symbolic values (issue #70).

A :class:`Guard` asserts ``poly REL 0`` for one of the six relations
from :mod:`symbolic_types`. :class:`GuardedPoly` is a list of
(guard-conjunction, value-polynomial) cases that partitions the input
domain.

This module also hosts the type unions :data:`RationalStackValue` and
:data:`SymbolicIntAst` — they span every symbolic-value module, so they
live here as the first aggregator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Tuple, Union

from bitvec import BitVec
from poly import Poly
from symbolic_types import (
    REL_EQ, REL_NE, REL_LT, REL_LE, REL_GT, REL_GE,
    _RELATIONS, _REL_SYMBOL, _NEGATE_REL,
    _relation_holds,
    IndicatorPoly, RationalPoly, SymbolicRemainder,
)


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




__all__ = [
    "RationalStackValue", "SymbolicIntAst",
    "Guard", "GuardedPoly",
]
