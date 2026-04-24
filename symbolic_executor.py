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

from dataclasses import dataclass
from typing import Dict, List

import isa

# Re-export the split-out types so existing callers
# (``from symbolic_executor import Poly, ...``) keep working without
# changes. This is the backward-compat surface for oaustegard/
# llm-as-computer#99.
from arithmetic_ops import (
    ArithFn, DivFn, RemFn,
    CmpBinFn, CmpUnaryFn,
    BitBinFn, BitUnaryFn, BitArithFn,
    ArithmeticOps, DEFAULT_ARITHMETIC_OPS,
)
from bitvec import BitVec, _apply_bitop
from closed_form import ClosedForm, ProductForm
from guarded import (
    Guard, GuardedPoly,
    RationalStackValue, SymbolicIntAst,
    _canonical_guards, _guards_complementary,
)
from modpoly import ModPoly
from poly import Monomial, Poly, _norm_coeff, _mono_mul, _mono_str
from symbolic_types import (
    REL_EQ, REL_NE, REL_LT, REL_LE, REL_GT, REL_GE,
    _RELATIONS, _REL_SYMBOL, _NEGATE_REL,
    _relation_holds,
    IndicatorPoly, RationalPoly, SymbolicRemainder,
)

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
    # Local-variable slots are polynomial-closed: a slot just stores and
    # retrieves whatever stack value was written to it. Adding them here
    # unlocks compilation strategies that would otherwise be blocked by
    # the top-3 reach limit of ROT/SWAP (see issue #100).
    isa.OP_LOCAL_GET, isa.OP_LOCAL_SET, isa.OP_LOCAL_TEE,
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
    locals_: Dict[int, RationalStackValue] = {}
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
        elif op == isa.OP_LOCAL_GET:
            if arg not in locals_:
                raise SymbolicStackUnderflow(
                    f"LOCAL_GET of uninitialized slot {arg}"
                )
            stack.append(locals_[arg])
        elif op == isa.OP_LOCAL_SET:
            if not stack:
                raise SymbolicStackUnderflow("LOCAL_SET on empty stack")
            locals_[arg] = _pop()
        elif op == isa.OP_LOCAL_TEE:
            if not stack:
                raise SymbolicStackUnderflow("LOCAL_TEE on empty stack")
            locals_[arg] = stack[-1]
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




# ─── Forking executor re-exports ──────────────────────────────────
#
# ``run_forking``, ``ForkingResult``, the recurrence solver, and the
# reporting helpers live in :mod:`forking_executor`. Re-exported here so
# ``from symbolic_executor import run_forking`` keeps working.
from forking_executor import (  # noqa: E402  (re-export after run_symbolic)
    DEFAULT_MAX_PATHS, DEFAULT_MAX_STEPS,
    ForkingResult,
    run_forking,
    collapse_report,
    guarded_to_mermaid,
)


__all__ = [
    "Poly",
    "ModPoly",
    "RationalPoly",
    "SymbolicRemainder",
    "ClosedForm",
    "ProductForm",
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
