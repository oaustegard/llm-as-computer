"""Compiler from Poly -> branchless LAC program (issue #94).

Round-trip invariant::

    run_symbolic(poly_to_program(p)).top == p

for any ``Poly`` with integer coefficients, zero constant term, and
contiguous variables {0, 1, ..., n-1}.

Emitted programs use the ``_POLY_OPS`` subset:
PUSH, POP, DUP, SWAP, OVER, ROT, ADD, SUB, MUL, LOCAL_GET, LOCAL_SET, HALT.

**Strategy (issue #100 fix):**
Each source variable ``x_i`` is PUSHed once and then immediately stashed
into local slot ``i`` via ``LOCAL_SET``. Every subsequent use fetches a
fresh copy via ``LOCAL_GET i``. This bypasses the top-3 reach limit of
ROT/SWAP/OVER that the prior ``copy_var`` dance ran into: locals are
O(1)-addressable regardless of stack depth, so there is no longer a
``depth > 2`` broken branch to fix.

**Constant-term restriction:** the output ring of branchless programs
cannot represent bare integer constants.  ``poly_to_program`` validates
the input and raises ``ValueError`` on non-zero constant terms.

**Negative coefficients** consume extra variable indices: the negation
trick PUSHes a dummy, DUPs it, SUBs to manufacture 0, then SUBs the
value to negate. The dummy and its duplicate cancel algebraically so
the output polynomial never mentions them.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List

import isa
from isa import Instruction
from symbolic_executor import Poly


# -- Public API -------------------------------------------------------


def poly_to_program(poly: Poly) -> List[Instruction]:
    """Compile a Poly into a branchless LAC program.

    Parameters
    ----------
    poly : Poly
        Must have integer coefficients and zero constant term.
        Variables must be contiguous from 0.

    Returns
    -------
    List[Instruction]
        A branchless program whose symbolic execution yields ``poly``.

    Raises
    ------
    ValueError
        Non-zero constant term, fractional coefficient, or
        non-contiguous variable indices.
    """
    # -- Validate -----------------------------------------------------
    for mono, coeff in poly.terms.items():
        if isinstance(coeff, Fraction):
            raise ValueError(
                f"poly_to_program: fractional coefficient {coeff} "
                f"not compilable to ADD/SUB/MUL fragment"
            )
    if () in poly.terms:
        raise ValueError(
            "poly_to_program: non-zero constant term is not representable "
            "in the branchless compiled-transformer ring"
        )

    # Zero polynomial -> PUSH, DUP, SUB -> x0 - x0 = 0
    if not poly.terms:
        return [
            Instruction(isa.OP_PUSH, 0),
            Instruction(isa.OP_DUP),
            Instruction(isa.OP_SUB),
            Instruction(isa.OP_HALT),
        ]

    vars_used = poly.variables()
    n_vars = max(vars_used) + 1 if vars_used else 0
    if vars_used != list(range(n_vars)):
        raise ValueError(
            f"poly_to_program: variables must be contiguous from 0; "
            f"got {vars_used}"
        )

    ctx = _CompilerContext()

    # -- Phase 1: bind each source variable to a local slot -----------
    # The symbolic executor assigns PUSH #k -> v_k, so PUSHing in source
    # order preserves the variable-index convention required by
    # ``result.top == p`` round-trip equality.
    for i in range(n_vars):
        ctx.emit(isa.OP_PUSH, 0)
        ctx.emit(isa.OP_LOCAL_SET, i)

    # -- Phase 2: compile monomials -----------------------------------
    monomials = list(poly.terms.items())

    for mono_idx, (mono, coeff) in enumerate(monomials):
        coeff = int(coeff)
        abs_coeff = abs(coeff)

        # Build the monomial's variable-product on top of stack.
        # Each (var, power) contributes one v^power factor; we then
        # MUL across factors to combine them.
        parts = 0
        for var_idx, power in mono:
            # LOCAL_GET the variable and raise to `power` via repeated
            # self-multiplication. v^1 is just one LOCAL_GET.
            ctx.emit(isa.OP_LOCAL_GET, var_idx)
            for _ in range(power - 1):
                ctx.emit(isa.OP_LOCAL_GET, var_idx)
                ctx.emit(isa.OP_MUL)
            parts += 1

        # MUL partial products together
        for _ in range(parts - 1):
            ctx.emit(isa.OP_MUL)

        # Scale by |coeff| via repeated addition: |c|*x = x + x + ... + x
        if abs_coeff > 1:
            for _ in range(abs_coeff - 1):
                ctx.emit(isa.OP_DUP)
            for _ in range(abs_coeff - 1):
                ctx.emit(isa.OP_ADD)

        # Negate if needed
        if coeff < 0:
            ctx.emit_negate()

        # Accumulate with prior monomials. The first monomial becomes
        # the accumulator; subsequent ones ADD onto it from the top.
        if mono_idx > 0:
            ctx.emit(isa.OP_ADD)

    ctx.emit(isa.OP_HALT)
    return ctx.instrs


# -- Compiler internals -----------------------------------------------


class _CompilerContext:
    """Minimal instruction emitter.

    The ``LOCAL_GET``/``LOCAL_SET``-based strategy means we no longer
    need the virtual-stack tag tracker, accumulator-position bookkeeping,
    or depth-aware ``copy_var`` that the pre-#100 compiler relied on.
    """

    def __init__(self):
        self.instrs: List[Instruction] = []

    def emit(self, op: int, arg: int = 0) -> None:
        """Append one instruction."""
        self.instrs.append(Instruction(op, arg))

    def emit_negate(self) -> None:
        """Negate top of stack: PUSH dummy, DUP, SUB (->0), SWAP, SUB.

        The dummy (a fresh symbolic variable in the polynomial ring)
        immediately cancels itself via ``dummy - dummy = 0``, so it
        leaves no trace in the final polynomial.
        """
        self.emit(isa.OP_PUSH, 0)
        self.emit(isa.OP_DUP)
        self.emit(isa.OP_SUB)
        self.emit(isa.OP_SWAP)
        self.emit(isa.OP_SUB)
