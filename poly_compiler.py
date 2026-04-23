"""Compiler from Poly -> branchless LAC program (issue #94).

Round-trip invariant::

    run_symbolic(poly_to_program(p)).top == p

for any ``Poly`` with integer coefficients, zero constant term, and
contiguous variables {0, 1, ..., n-1}.

Emitted programs use only the ``_POLY_OPS`` subset:
PUSH, POP, DUP, SWAP, OVER, ROT, ADD, SUB, MUL, HALT.

**Constant-term restriction:** the output ring of branchless programs
cannot represent bare integer constants.  ``poly_to_program`` validates
the input and raises ``ValueError`` on non-zero constant terms.

**Negative coefficients** consume extra variable indices: the negation
trick PUSHes a dummy, DUPs it, SUBs to manufacture 0, then SUBs the
value to negate.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Union

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

    # -- Phase 1: PUSH base variables ---------------------------------
    for _ in range(n_vars):
        ctx.emit(isa.OP_PUSH, 0)

    # -- Phase 2: compile monomials -----------------------------------
    monomials = list(poly.terms.items())

    for mono_idx, (mono, coeff) in enumerate(monomials):
        coeff = int(coeff)
        abs_coeff = abs(coeff)

        # Build the monomial's variable-product on top of stack.
        parts = 0
        for var_idx, power in mono:
            ctx.copy_var(var_idx)
            for _ in range(power - 1):
                ctx.emit(isa.OP_DUP)
            for _ in range(power - 1):
                ctx.emit(isa.OP_MUL)
            parts += 1

        # MUL partial products together
        for _ in range(parts - 1):
            ctx.emit(isa.OP_MUL)

        # Scale by |coeff|
        if abs_coeff > 1:
            for _ in range(abs_coeff - 1):
                ctx.emit(isa.OP_DUP)
            for _ in range(abs_coeff - 1):
                ctx.emit(isa.OP_ADD)

        # Negate if needed
        if coeff < 0:
            ctx.emit_negate()

        # Accumulate with prior monomials
        if mono_idx == 0:
            ctx.accum_vsidx = len(ctx.vstack) - 1
        else:
            # The accumulator may have been displaced by ROT during
            # copy_var.  Bring it adjacent to the monomial result
            # (currently on top) before ADD.
            mono_top = len(ctx.vstack) - 1
            accum_depth = mono_top - ctx.accum_vsidx

            if accum_depth == 1:
                ctx.emit(isa.OP_ADD)
            elif accum_depth == 2:
                # ROT brings accum to top; ADD pops accum + mono_result.
                ctx.emit(isa.OP_ROT)
                ctx.emit(isa.OP_ADD)
            elif accum_depth == 0:
                raise RuntimeError(
                    "accumulator collided with monomial result"
                )
            else:
                raise NotImplementedError(
                    f"accumulator at depth {accum_depth}; "
                    f"need deeper stack access (file a follow-up)"
                )
            ctx.accum_vsidx = len(ctx.vstack) - 1

    # -- Phase 3: discard base variables ------------------------------
    # Stack has base vars below the result.  Repeated SWAP+POP peels
    # them off regardless of their ordering (ROT may have permuted them).
    n_base = len(ctx.vstack) - 1  # everything except the top (result)
    for _ in range(n_base):
        ctx.emit(isa.OP_SWAP)
        ctx.emit(isa.OP_POP)

    ctx.emit(isa.OP_HALT)
    return ctx.instrs


# -- Compiler internals -----------------------------------------------


class _CompilerContext:
    """Virtual-stack tracker and instruction emitter.

    Tracks:
    - ``vstack``: tag per stack slot (``'v0'``, ``'v1'``, ... for base
      variables; ``'expr'`` for computed values).
    - ``accum_vsidx``: position of the monomial accumulator in vstack,
      updated through every emitted instruction so it stays valid even
      after ROT/SWAP reorderings.
    """

    def __init__(self):
        self.instrs: List[Instruction] = []
        self.vstack: List[str] = []
        self.next_var: int = 0
        self.accum_vsidx: Optional[int] = None

    def emit(self, op: int, arg: int = 0) -> None:
        """Append one instruction and update bookkeeping."""
        self.instrs.append(Instruction(op, arg))
        self._track_accum(op)
        self._mirror(op)

    # -- High-level helpers -------------------------------------------

    def copy_var(self, var_idx: int) -> None:
        """Copy base variable ``var_idx`` to top of stack."""
        tag = f"v{var_idx}"
        pos = self._find_tag(tag)
        depth = len(self.vstack) - 1 - pos

        if depth == 0:
            self.emit(isa.OP_DUP)
        elif depth == 1:
            self.emit(isa.OP_OVER)
        elif depth == 2:
            self.emit(isa.OP_ROT)
            self.emit(isa.OP_DUP)
        else:
            for _ in range(depth - 2):
                self.emit(isa.OP_ROT)
                self.emit(isa.OP_SWAP)
            self.emit(isa.OP_ROT)
            self.emit(isa.OP_DUP)

    def emit_negate(self) -> None:
        """Negate top of stack: PUSH dummy, DUP, SUB (->0), SWAP, SUB."""
        self.emit(isa.OP_PUSH, 0)
        self.emit(isa.OP_DUP)
        self.emit(isa.OP_SUB)
        self.emit(isa.OP_SWAP)
        self.emit(isa.OP_SUB)

    # -- Accumulator position tracking --------------------------------

    def _track_accum(self, op: int) -> None:
        """Update ``accum_vsidx`` to reflect the effect of ``op``.

        Called BEFORE ``_mirror`` (so ``len(self.vstack)`` is the
        pre-instruction stack size).
        """
        p = self.accum_vsidx
        if p is None:
            return
        n = len(self.vstack)
        if op == isa.OP_PUSH or op == isa.OP_DUP or op == isa.OP_OVER:
            # Stack grows by 1; items below unchanged.
            pass
        elif op == isa.OP_POP:
            if p == n - 1:
                self.accum_vsidx = None  # consumed!
            # else: unchanged
        elif op == isa.OP_SWAP:
            if p == n - 1:
                self.accum_vsidx = n - 2
            elif p == n - 2:
                self.accum_vsidx = n - 1
        elif op == isa.OP_ROT:
            if n >= 3:
                if p == n - 3:
                    self.accum_vsidx = n - 1
                elif p == n - 2:
                    self.accum_vsidx = n - 3
                elif p == n - 1:
                    self.accum_vsidx = n - 2
        elif op in (isa.OP_ADD, isa.OP_SUB, isa.OP_MUL):
            # Pops top 2, pushes 1.  Items at positions < n-2 unchanged.
            if p >= n - 2:
                # Consumed by binary op -- this is the accumulate ADD
                # itself; caller will reset accum_vsidx afterwards.
                self.accum_vsidx = None
            # else: unchanged

    # -- Virtual stack bookkeeping ------------------------------------

    def _find_tag(self, tag: str) -> int:
        """Find bottom-most position of ``tag`` in the virtual stack."""
        for i, t in enumerate(self.vstack):
            if t == tag:
                return i
        raise KeyError(f"variable tag {tag!r} not found in virtual stack")

    def _mirror(self, op: int) -> None:
        """Mirror one instruction's effect on the virtual stack."""
        vs = self.vstack
        if op == isa.OP_PUSH:
            vs.append(f"v{self.next_var}")
            self.next_var += 1
        elif op == isa.OP_POP:
            vs.pop()
        elif op == isa.OP_DUP:
            vs.append(vs[-1])
        elif op == isa.OP_SWAP:
            vs[-1], vs[-2] = vs[-2], vs[-1]
        elif op == isa.OP_OVER:
            vs.append(vs[-2])
        elif op == isa.OP_ROT:
            a, b, c = vs[-3], vs[-2], vs[-1]
            vs[-3], vs[-2], vs[-1] = b, c, a
        elif op in (isa.OP_ADD, isa.OP_SUB, isa.OP_MUL):
            vs.pop()
            vs[-1] = "expr"
        elif op == isa.OP_HALT:
            pass
        elif op == isa.OP_NOP:
            pass
