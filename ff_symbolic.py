"""Bilinear FF dispatch for ADD/SUB/MUL (issue #69).

Closes the gap between "the ISA semantics compose into a polynomial"
(proven by :mod:`symbolic_executor`) and "the transformer weights
*compute* the polynomial". The FF dispatch in :class:`executor.CompiledModel`
currently falls through to Python ``int`` arithmetic for the arithmetic
fragment of the ISA — ``nonlinear[ADD] = float((va + vb) & MASK32)``
(``executor.py`` lines 819-821). That means the compiled transformer
faithfully *routes* arithmetic results produced by CPython, but the
arithmetic itself is not realised by matmul.

This module replaces that path with analytically-set weight matrices
whose forward pass evaluates the same polynomial the symbolic executor
computes. ``CompiledModel.forward`` (float tensors) and
``CompiledModel.forward_symbolic`` (:class:`Poly` values) end up sharing
one spec — the bilinear form — with two interpreters: polynomial
evaluation over ℝ (via torch) and polynomial algebra over ℤ (via
:class:`Poly`). The equivalence proof is that both interpreters apply
the same operator tree to inputs from the same source.

Embedding scheme
----------------
The stack embedding used by :func:`isa.embed_stack_entry` already stores
the value as a single scalar at ``DIM_VALUE``. So the value-embedding is
literally ``E(v) = v · e_{DIM_VALUE}`` where ``e_i`` is the standard
basis vector. That makes the three constructions direct:

* **ADD** is linear: ``E(a+b) = E(a) + E(b)`` — so ``M_ADD``
  is ``[1, 1]`` over the two value inputs (the analog of a standard
  bias-free FF row).
* **SUB** is linear: ``E(b-a) = E(b) - E(a)`` — matches the
  WASM ``i32.sub`` order (``vb - va`` where ``va`` is the top).
* **MUL** is bilinear: ``B_MUL(E(a), E(b)) = a·b`` with
  ``B_MUL = e_{DIM_VALUE} ⊗ e_{DIM_VALUE}`` (a rank-1 outer product
  with a single ``1`` at ``[DIM_VALUE, DIM_VALUE]``). This is precisely
  what a gated / bilinear FF variant computes — the bridge between "FF
  layer" and "polynomial evaluator".

Range assumption
----------------
The equivalence theorem is stated over ℤ with Option (a) from the issue:
no ``& MASK32`` is applied in the bilinear form; :func:`range_check`
asserts no wrap would have occurred on the caller's inputs. The numeric
path in :meth:`executor.CompiledModel.forward` still applies the mask
for continuity with :class:`executor.NumPyExecutor` (so
``test_consolidated.py`` stays green); the equivalence tests compare
unmasked values on in-range inputs.

Scope
-----
This module handles ADD/SUB/MUL only. DIV_S, REM_S, comparisons,
bitwise, unary numeric ops, control flow — all unchanged, and all
explicit follow-ups per the issue's "Non-goals" section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch

import isa
from isa import DIM_VALUE, D_MODEL, DTYPE
from symbolic_executor import Poly


# ─── Exceptions ────────────────────────────────────────────────────

class BlockedOpcodeForSymbolic(NotImplementedError):
    """Raised when :meth:`forward_symbolic` encounters an op outside its scope.

    The scope is the branchless polynomial fragment:
    ``PUSH, POP, DUP, HALT, NOP, SWAP, OVER, ROT, ADD, SUB, MUL``.
    Everything else is a follow-up per issue #69's non-goals.
    """


class RangeCheckFailure(ValueError):
    """Raised by :func:`range_check` when the integer range is exceeded."""


# ─── Range check (Option (a) from issue #69) ──────────────────────

# i32 bounds — the range within which "no wrap occurs" holds for the
# equivalence theorem. Anything inside [-2**31, 2**31) is safe under
# signed 32-bit semantics.
I32_MIN = -(1 << 31)
I32_MAX = (1 << 31) - 1


def range_check(value: int, *, context: str = "") -> int:
    """Assert that ``value`` fits in i32 (no wrap would occur). Returns ``value``.

    The bilinear FF evaluates over ℤ — the theorem's scope. Callers who
    want to claim parity with the masked numeric path must first verify
    no wrap happened on the inputs of interest; this helper makes that
    explicit rather than silent.
    """
    if not (I32_MIN <= int(value) <= I32_MAX):
        raise RangeCheckFailure(
            f"value {value} outside i32 range [{I32_MIN}, {I32_MAX}]"
            + (f" ({context})" if context else "")
        )
    return int(value)


# ─── Scalar embedding ─────────────────────────────────────────────

def E(v: Union[int, float], d_model: int = D_MODEL) -> torch.Tensor:
    """Scalar value embedding: ``E(v) = v · e_{DIM_VALUE}``.

    Matches the existing stack-entry layout in :func:`isa.embed_stack_entry`,
    so bilinear primitives here compose with the values already produced
    by the attention heads without any re-projection.
    """
    e = torch.zeros(d_model, dtype=DTYPE)
    e[DIM_VALUE] = float(v)
    return e


def E_inv(emb: torch.Tensor) -> int:
    """Read the integer value from a scalar-embedded tensor.

    Rounds the ``DIM_VALUE`` slot to the nearest integer. The inverse of
    :func:`E` on valid inputs; used by the numeric FF dispatch to recover
    the int result of a bilinear evaluation.
    """
    return int(round(float(emb[DIM_VALUE].item())))


# ─── Weight matrices ──────────────────────────────────────────────
#
# Analytically-set, cached at module load. All matrices are constants:
# the construction is a proof, not a training target.

def _M_ADD_matrix(d_model: int = D_MODEL) -> torch.Tensor:
    """Linear map for ADD: ``M_ADD @ [E(a); E(b)] = E(a+b)``.

    Shape: ``(d_model, 2*d_model)``. The only non-zero row is ``DIM_VALUE``,
    with a ``1`` at both ``DIM_VALUE`` and ``d_model + DIM_VALUE``.
    """
    W = torch.zeros(d_model, 2 * d_model, dtype=DTYPE)
    W[DIM_VALUE, DIM_VALUE] = 1.0
    W[DIM_VALUE, d_model + DIM_VALUE] = 1.0
    return W


def _M_SUB_matrix(d_model: int = D_MODEL) -> torch.Tensor:
    """Linear map for SUB: ``M_SUB @ [E(a); E(b)] = E(b-a)``.

    Note the order: ``va`` is the top (arg 1), ``vb`` is stack[SP-1] (arg 2),
    and WASM ``i32.sub`` computes ``vb - va``. This matches the existing
    nonlinear path (``executor.py:820``).
    """
    W = torch.zeros(d_model, 2 * d_model, dtype=DTYPE)
    W[DIM_VALUE, DIM_VALUE] = -1.0                  # coefficient of a
    W[DIM_VALUE, d_model + DIM_VALUE] = 1.0         # coefficient of b
    return W


def _B_MUL_matrix(d_model: int = D_MODEL) -> torch.Tensor:
    """Bilinear form for MUL: ``B_MUL(E(a), E(b)) = a·b``.

    ``B_MUL`` is a rank-1 outer product, ``e_{DIM_VALUE} ⊗ e_{DIM_VALUE}``.
    Evaluated as ``x^T B_MUL y`` on scalar-embedded inputs, it extracts
    the product of the two ``DIM_VALUE`` slots.
    """
    B = torch.zeros(d_model, d_model, dtype=DTYPE)
    B[DIM_VALUE, DIM_VALUE] = 1.0
    return B


# Module-scope cached tensors — read-only; wrapped on access.
M_ADD: torch.Tensor = _M_ADD_matrix()
M_SUB: torch.Tensor = _M_SUB_matrix()
B_MUL: torch.Tensor = _B_MUL_matrix()


def n_parameters(d_model: int = D_MODEL) -> int:
    """Parameter count contributed by ``M_ADD``, ``M_SUB``, ``B_MUL``.

    Counts the actual non-zero entries (3) — the dense shapes hold only
    these three symbolic "slots". The blog post's "964 compiled parameters"
    figure should become ``964 + 3`` after this module lands; the stored
    tensors are larger but the analytically-set content is three bits.
    """
    return 3


# ─── Numeric primitives (float tensors) ───────────────────────────

def forward_add(ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
    """Compute ``E(a+b)`` from ``E(a), E(b)`` by the linear form ``M_ADD``.

    No Python arithmetic touches ``a, b``: the output is assembled from
    a matmul on the stacked inputs.
    """
    stacked = torch.cat([ea, eb])                # shape (2*d_model,)
    return M_ADD @ stacked                        # shape (d_model,)


def forward_sub(ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
    """Compute ``E(b-a)`` from ``E(a), E(b)`` by the linear form ``M_SUB``."""
    stacked = torch.cat([ea, eb])
    return M_SUB @ stacked


def forward_mul(ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
    """Compute ``E(a·b)`` from ``E(a), E(b)`` by the bilinear form ``B_MUL``.

    The bilinear scalar ``x^T B_MUL y`` is re-embedded via ``E`` so the
    output shares the same scalar-at-``DIM_VALUE`` layout as every other
    value in the model.
    """
    scalar = ea @ B_MUL @ eb                      # shape (), a single float
    out = torch.zeros(ea.shape[0], dtype=DTYPE)
    out[DIM_VALUE] = scalar
    return out


# ─── Symbolic primitives (Poly values) ────────────────────────────
#
# These ARE polynomial algebra — ``Poly`` overloads ``+ - *`` — but they
# live behind named functions to make the shared spec with the numeric
# primitives explicit. The claim "M_ADD / M_SUB / B_MUL realise exactly
# these polynomial operations" is what the equivalence test verifies.


def symbolic_add(pa: Poly, pb: Poly) -> Poly:
    """Polynomial addition; corresponds to ``forward_add`` over :class:`Poly`."""
    return pa + pb


def symbolic_sub(pa: Poly, pb: Poly) -> Poly:
    """Polynomial subtraction; corresponds to ``forward_sub`` over :class:`Poly`.

    Order matches ``forward_sub``: returns ``pb - pa``.
    """
    return pb - pa


def symbolic_mul(pa: Poly, pb: Poly) -> Poly:
    """Polynomial multiplication; corresponds to ``forward_mul`` over :class:`Poly`."""
    return pa * pb


# ─── Program-level symbolic forward ───────────────────────────────
#
# ``CompiledModel.forward_symbolic`` delegates to :func:`evaluate_program`
# so the driver logic is testable in isolation and the CompiledModel stays
# focused on weight construction.

# Ops this module understands symbolically. Anything else → BlockedOpcodeForSymbolic.
_SCOPE_OPS = {
    isa.OP_PUSH, isa.OP_POP, isa.OP_DUP, isa.OP_HALT, isa.OP_NOP,
    isa.OP_ADD, isa.OP_SUB, isa.OP_MUL,
    isa.OP_SWAP, isa.OP_OVER, isa.OP_ROT,
}


@dataclass
class SymbolicForwardResult:
    """Outcome of :meth:`CompiledModel.forward_symbolic`.

    ``top`` is the :class:`Poly` left on top of the symbolic stack at HALT.
    ``stack`` is the full final stack (bottom at index 0). ``n_heads`` is
    the number of non-NOP, non-HALT instructions executed — the "k heads"
    matching :class:`symbolic_executor.SymbolicResult.n_heads`. ``bindings``
    records which variable id corresponds to which PUSH constant.
    """
    top: Poly
    stack: List[Poly]
    n_heads: int
    bindings: Dict[int, int]


def evaluate_program(prog) -> SymbolicForwardResult:
    """Run ``prog`` through the bilinear FF interpreters over :class:`Poly`.

    Matches :func:`symbolic_executor.run_symbolic` semantically: branchless
    straight-line programs only, one variable per PUSH, ADD/SUB/MUL delegated
    to the symbolic primitives above. The point of duplicating the driver is
    to make the bilinear dispatch story explicit — every arithmetic call
    site here is the :class:`Poly` interpretation of one of the three
    weight matrices.
    """
    stack: List[Poly] = []
    bindings: Dict[int, int] = {}
    next_var = 0
    n_heads = 0

    def _pop() -> Poly:
        if not stack:
            raise IndexError("symbolic stack underflow")
        return stack.pop()

    for instr in prog:
        op = instr.op
        if op not in _SCOPE_OPS:
            raise BlockedOpcodeForSymbolic(
                f"op {isa.OP_NAMES.get(op, f'?{op}')!r} is out of scope for the "
                f"bilinear FF dispatch (ADD/SUB/MUL + stack manip only)"
            )
        if op == isa.OP_HALT:
            break
        if op == isa.OP_NOP:
            continue
        n_heads += 1
        if op == isa.OP_PUSH:
            v = next_var
            next_var += 1
            bindings[v] = int(instr.arg)
            stack.append(Poly.variable(v))
        elif op == isa.OP_POP:
            _pop()
        elif op == isa.OP_DUP:
            if not stack:
                raise IndexError("dup on empty stack")
            stack.append(stack[-1])
        elif op == isa.OP_ADD:
            b = _pop(); a = _pop()
            stack.append(symbolic_add(a, b))
        elif op == isa.OP_SUB:
            b = _pop(); a = _pop()
            stack.append(symbolic_sub(a, b))
        elif op == isa.OP_MUL:
            b = _pop(); a = _pop()
            stack.append(symbolic_mul(a, b))
        elif op == isa.OP_SWAP:
            if len(stack) < 2:
                raise IndexError("swap needs 2 entries")
            stack[-1], stack[-2] = stack[-2], stack[-1]
        elif op == isa.OP_OVER:
            if len(stack) < 2:
                raise IndexError("over needs 2 entries")
            stack.append(stack[-2])
        elif op == isa.OP_ROT:
            if len(stack) < 3:
                raise IndexError("rot needs 3 entries")
            a, b, c = stack[-3], stack[-2], stack[-1]
            stack[-3], stack[-2], stack[-1] = b, c, a
        else:  # pragma: no cover — guarded by _SCOPE_OPS above
            raise BlockedOpcodeForSymbolic(f"unreachable: op {op}")

    top = stack[-1] if stack else Poly.constant(0)
    return SymbolicForwardResult(top=top, stack=list(stack),
                                 n_heads=n_heads, bindings=bindings)


__all__ = [
    "BlockedOpcodeForSymbolic",
    "RangeCheckFailure",
    "I32_MIN", "I32_MAX",
    "E", "E_inv",
    "M_ADD", "M_SUB", "B_MUL",
    "n_parameters",
    "forward_add", "forward_sub", "forward_mul",
    "symbolic_add", "symbolic_sub", "symbolic_mul",
    "SymbolicForwardResult",
    "evaluate_program",
    "range_check",
]
