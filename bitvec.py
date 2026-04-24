"""Bit-vector AST for the symbolic executor (issue #77).

:class:`BitVec` is an un-simplified AST node with a named op and a tuple
of :class:`poly.Poly` or :class:`BitVec` operands. The non-polynomial
step happens at :meth:`BitVec.eval_at`, matching the
:class:`symbolic_types.RationalPoly` /
:class:`symbolic_types.IndicatorPoly` design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Tuple, Union

from isa import (
    _clz32,
    _ctz32,
    _popcnt32,
    _rotl32,
    _rotr32,
    _shr_s,
    _shr_u,
    _to_i32,
    MASK32,
)
from poly import Poly

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



__all__ = ["BitVec"]
