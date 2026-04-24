"""Arithmetic op hook for the forking executor (issue #68 S3).

:class:`ArithmeticOps` decouples the forking executor from the
concrete ADD/SUB/MUL/DIV/REM/CMP/BITWISE primitives. The default
(:data:`DEFAULT_ARITHMETIC_OPS`) uses :class:`poly.Poly`'s native
operators; :mod:`ff_symbolic` plugs in its bilinear-FF variants here to
demonstrate equivalence under issue #69.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import isa
from bitvec import BitVec
from guarded import SymbolicIntAst
from poly import Poly
from symbolic_types import (
    REL_EQ, REL_NE, REL_LT, REL_LE, REL_GT, REL_GE,
    IndicatorPoly, RationalPoly, SymbolicRemainder,
)

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




__all__ = [
    "ArithmeticOps", "DEFAULT_ARITHMETIC_OPS",
    "ArithFn", "DivFn", "RemFn",
    "CmpBinFn", "CmpUnaryFn",
    "BitBinFn", "BitUnaryFn", "BitArithFn",
]
