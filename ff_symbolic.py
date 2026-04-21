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

Rational extension (issue #75)
------------------------------
DIV_S / REM_S are *not* polynomial. The workaround — and the content of
issue #75 — is to keep the symbolic form rational (``RationalPoly`` /
``SymbolicRemainder`` track ``(num, denom)`` pairs through the stack)
and apply integer truncation only at the boundary, during
``eval_at`` / ``forward_div_s`` evaluation.

The "weight matrix" story for DIV_S / REM_S is therefore weaker than
for ADD/SUB/MUL: ``M_DIV_S`` / ``M_REM_S`` are 2×2 *pair-selector*
matrices that pluck the two scalar operands out of the stacked input,
and the truncating division ``_trunc_div(vb, va)`` is a non-linear
boundary step — not itself a matmul. That is the honest cost of
integer-rounded division in a polynomial framework; the value of this
extension is that the rational *algebra* (up to truncation) does
compose cleanly, and the catalog's ``native_divmod`` / ``native_remainder``
programs now have a closed-form symbolic top.

Composition past a single DIV_S / REM_S on the same stack slot is not
supported in this issue — a subsequent arithmetic op on a rational
stack entry raises ``SymbolicOpNotSupported``. That is a follow-up.

Scope
-----
ADD/SUB/MUL via true (bi)linear forms; DIV_S/REM_S via pair-selector
+ boundary trunc (issue #75). Comparisons, bitwise, unary numeric
ops, control flow — all unchanged, and all explicit follow-ups per
the issue's "Non-goals" section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch

import isa
from isa import DIM_VALUE, D_MODEL, DTYPE, _trunc_div, _trunc_rem
from symbolic_executor import (
    ArithmeticOps,
    ForkingResult,
    Poly,
    RationalPoly,
    RationalStackValue,
    SymbolicRemainder,
    run_forking,
)


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


def _M_DIV_S_matrix(d_model: int = D_MODEL) -> torch.Tensor:
    """Pair-selector for DIV_S: pluck ``(va, vb)`` from ``[E(a); E(b)]``.

    Shape: ``(2, 2*d_model)``. Row 0 picks ``va = ea[DIM_VALUE]`` (the top),
    row 1 picks ``vb = eb[DIM_VALUE]`` (stack[SP-1]). The subsequent
    ``_trunc_div(vb, va)`` is a *non-linear boundary step* — unlike
    ``M_ADD`` / ``M_SUB`` / ``B_MUL``, this matrix alone does not
    realise the op. Its role is to expose, at the weight level, that
    the FF receives exactly the two scalars and nothing else.
    """
    W = torch.zeros(2, 2 * d_model, dtype=DTYPE)
    W[0, DIM_VALUE] = 1.0                   # va
    W[1, d_model + DIM_VALUE] = 1.0         # vb
    return W


def _M_REM_S_matrix(d_model: int = D_MODEL) -> torch.Tensor:
    """Pair-selector for REM_S: identical shape/role to :func:`_M_DIV_S_matrix`.

    The boundary nonlinearity is ``_trunc_rem(vb, va)``.
    """
    W = torch.zeros(2, 2 * d_model, dtype=DTYPE)
    W[0, DIM_VALUE] = 1.0
    W[1, d_model + DIM_VALUE] = 1.0
    return W


# Module-scope cached tensors — read-only; wrapped on access.
M_ADD: torch.Tensor = _M_ADD_matrix()
M_SUB: torch.Tensor = _M_SUB_matrix()
B_MUL: torch.Tensor = _B_MUL_matrix()
M_DIV_S: torch.Tensor = _M_DIV_S_matrix()
M_REM_S: torch.Tensor = _M_REM_S_matrix()


def n_parameters(d_model: int = D_MODEL) -> int:
    """Parameter count contributed by the analytically-set FF weights.

    Counts non-zero entries across all five matrices:
    2 (M_ADD) + 2 (M_SUB) + 1 (B_MUL) + 2 (M_DIV_S) + 2 (M_REM_S) = 9.
    The stored tensors are larger but the analytically-set content is nine
    bits. Issue #69 shipped with the first three (for 5 non-zeros); issue
    #75 added M_DIV_S / M_REM_S.
    """
    return 9


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


def forward_div_s(ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
    """Compute ``E(trunc_div(vb, va))`` from ``E(a), E(b)``.

    Applies ``M_DIV_S`` as a pair-selector (``[va, vb] = M_DIV_S @ stacked``)
    and then the truncating integer division ``_trunc_div(vb, va)`` — the
    non-linear boundary step that polynomial algebra cannot express.
    Matches the existing numeric path in ``executor.py:836``:
    ``_trunc_div(vb, va)`` with ``va`` = top, ``vb`` = stack[SP-1].
    """
    stacked = torch.cat([ea, eb])                 # shape (2*d_model,)
    pair = M_DIV_S @ stacked                      # shape (2,): [va, vb]
    va = int(round(float(pair[0].item())))
    vb = int(round(float(pair[1].item())))
    q = _trunc_div(vb, va)                        # boundary nonlinearity
    out = torch.zeros(ea.shape[0], dtype=DTYPE)
    out[DIM_VALUE] = float(q)
    return out


def forward_rem_s(ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
    """Compute ``E(trunc_rem(vb, va))`` from ``E(a), E(b)``.

    Same pair-selector pattern as :func:`forward_div_s`; the boundary
    nonlinearity is ``_trunc_rem(vb, va)`` (sign-of-dividend remainder,
    matching WASM ``i32.rem_s``).
    """
    stacked = torch.cat([ea, eb])
    pair = M_REM_S @ stacked
    va = int(round(float(pair[0].item())))
    vb = int(round(float(pair[1].item())))
    r = _trunc_rem(vb, va)
    out = torch.zeros(ea.shape[0], dtype=DTYPE)
    out[DIM_VALUE] = float(r)
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


def symbolic_div_s(pa: Poly, pb: Poly) -> RationalPoly:
    """Rational form of DIV_S; corresponds to ``forward_div_s`` over :class:`Poly`.

    Order matches ``forward_div_s`` (``pa`` is top, ``pb`` is stack[SP-1]):
    returns ``RationalPoly(num=pb, denom=pa)``. Truncation to int happens
    at :meth:`RationalPoly.eval_at` time (the boundary step).
    """
    return RationalPoly(num=pb, denom=pa)


def symbolic_rem_s(pa: Poly, pb: Poly) -> SymbolicRemainder:
    """Symbolic remainder; corresponds to ``forward_rem_s`` over :class:`Poly`.

    Order matches ``forward_rem_s``. Truncation-style remainder is applied
    at :meth:`SymbolicRemainder.eval_at` time.
    """
    return SymbolicRemainder(num=pb, denom=pa)


# Arithmetic primitives packaged for :func:`symbolic_executor.run_forking`'s
# ``arithmetic_ops`` hook (issue #68 S3; extended for DIV_S/REM_S in #75).
# The wrappers flip arg order where needed: the forking executor's convention
# is ``op(a, b)`` with ``a`` = stack[SP-1] and ``b`` = top. ``symbolic_sub`` /
# ``symbolic_div_s`` / ``symbolic_rem_s`` are written in the FF ``(pa, pb)``
# order where ``pa`` = top, ``pb`` = SP-1, so we call them with ``(b, a)``.
FF_ARITHMETIC_OPS = ArithmeticOps(
    add=lambda a, b: symbolic_add(a, b),
    sub=lambda a, b: symbolic_sub(b, a),
    mul=lambda a, b: symbolic_mul(a, b),
    div_s=lambda a, b: symbolic_div_s(b, a),
    rem_s=lambda a, b: symbolic_rem_s(b, a),
)


# ─── Program-level symbolic forward ───────────────────────────────
#
# ``CompiledModel.forward_symbolic`` delegates to :func:`evaluate_program`
# so the driver logic is testable in isolation and the CompiledModel stays
# focused on weight construction.

# Ops this module understands symbolically. Anything else → BlockedOpcodeForSymbolic.
_SCOPE_OPS = {
    isa.OP_PUSH, isa.OP_POP, isa.OP_DUP, isa.OP_HALT, isa.OP_NOP,
    isa.OP_ADD, isa.OP_SUB, isa.OP_MUL,
    isa.OP_DIV_S, isa.OP_REM_S,
    isa.OP_SWAP, isa.OP_OVER, isa.OP_ROT,
}


@dataclass
class SymbolicForwardResult:
    """Outcome of :meth:`CompiledModel.forward_symbolic`.

    ``top`` is the value left on top of the symbolic stack at HALT — a
    :class:`Poly` for the arithmetic fragment, a :class:`RationalPoly`
    when the last op is DIV_S, or a :class:`SymbolicRemainder` when the
    last op is REM_S (issue #75). ``stack`` is the full final stack
    (bottom at index 0). ``n_heads`` is the number of non-NOP, non-HALT
    instructions executed — the "k heads" matching
    :class:`symbolic_executor.SymbolicResult.n_heads`. ``bindings``
    records which variable id corresponds to which PUSH constant.
    """
    top: RationalStackValue
    stack: List[RationalStackValue]
    n_heads: int
    bindings: Dict[int, int]


def evaluate_program(prog) -> SymbolicForwardResult:
    """Run ``prog`` through the bilinear FF interpreters over :class:`Poly`.

    Matches :func:`symbolic_executor.run_symbolic` semantically: branchless
    straight-line programs only, one variable per PUSH, arithmetic delegated
    to this module's symbolic primitives. The point of duplicating the
    driver is to make the weight-level dispatch story explicit — every
    arithmetic call site here is the :class:`Poly` interpretation of one
    of the FF weight matrices (or, for DIV_S / REM_S, the rational-pair
    form carried by :class:`RationalPoly` / :class:`SymbolicRemainder`).
    """
    stack: List[RationalStackValue] = []
    bindings: Dict[int, int] = {}
    next_var = 0
    n_heads = 0

    def _pop() -> RationalStackValue:
        if not stack:
            raise IndexError("symbolic stack underflow")
        return stack.pop()

    def _require_poly(v: RationalStackValue, op_name: str) -> Poly:
        if not isinstance(v, Poly):
            raise BlockedOpcodeForSymbolic(
                f"{op_name} on rational stack entries is out of scope "
                f"(composition past one DIV_S/REM_S is a follow-up)"
            )
        return v

    for instr in prog:
        op = instr.op
        if op not in _SCOPE_OPS:
            raise BlockedOpcodeForSymbolic(
                f"op {isa.OP_NAMES.get(op, f'?{op}')!r} is out of scope for the "
                f"bilinear FF dispatch (ADD/SUB/MUL/DIV_S/REM_S + stack manip only)"
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
            b = _require_poly(_pop(), "ADD"); a = _require_poly(_pop(), "ADD")
            stack.append(symbolic_add(a, b))
        elif op == isa.OP_SUB:
            b = _require_poly(_pop(), "SUB"); a = _require_poly(_pop(), "SUB")
            stack.append(symbolic_sub(a, b))
        elif op == isa.OP_MUL:
            b = _require_poly(_pop(), "MUL"); a = _require_poly(_pop(), "MUL")
            stack.append(symbolic_mul(a, b))
        elif op == isa.OP_DIV_S:
            b = _require_poly(_pop(), "DIV_S"); a = _require_poly(_pop(), "DIV_S")
            # forking executor convention: a = SP-1, b = top; result = a / b.
            # symbolic_div_s uses (pa=top, pb=SP-1) ordering.
            stack.append(symbolic_div_s(b, a))
        elif op == isa.OP_REM_S:
            b = _require_poly(_pop(), "REM_S"); a = _require_poly(_pop(), "REM_S")
            stack.append(symbolic_rem_s(b, a))
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

    top: RationalStackValue = stack[-1] if stack else Poly.constant(0)
    return SymbolicForwardResult(top=top, stack=list(stack),
                                 n_heads=n_heads, bindings=bindings)


def evaluate_program_forking(prog, *, input_mode: str = "symbolic") -> ForkingResult:
    """Run ``prog`` through :func:`symbolic_executor.run_forking` with the
    bilinear-FF ADD/SUB/MUL primitives (issue #68 S3).

    The control-flow driver (JZ/JNZ, worklist, path merging) is shared
    with the symbolic executor's native call; only the arithmetic
    primitives differ — they're routed through this module's
    :func:`symbolic_add` / :func:`symbolic_sub` / :func:`symbolic_mul`,
    the Poly-level interpretation of ``M_ADD`` / ``M_SUB`` / ``B_MUL``.

    Returns a :class:`symbolic_executor.ForkingResult`. For guarded
    programs the ``top`` is a :class:`symbolic_executor.GuardedPoly`;
    for unrolled or straight-line programs it's a :class:`Poly`.

    The equivalence claim (structural): for every catalog program
    accepted by the forking executor, this function's output is
    structurally equal to :func:`symbolic_executor.run_forking`'s.
    Verified by ``test_ff_symbolic.test_equivalence_guarded_*`` and
    ``_equivalence_unrolled_*``.
    """
    return run_forking(prog, input_mode=input_mode,
                       arithmetic_ops=FF_ARITHMETIC_OPS)


__all__ = [
    "BlockedOpcodeForSymbolic",
    "RangeCheckFailure",
    "I32_MIN", "I32_MAX",
    "E", "E_inv",
    "M_ADD", "M_SUB", "B_MUL", "M_DIV_S", "M_REM_S",
    "n_parameters",
    "forward_add", "forward_sub", "forward_mul",
    "forward_div_s", "forward_rem_s",
    "symbolic_add", "symbolic_sub", "symbolic_mul",
    "symbolic_div_s", "symbolic_rem_s",
    "FF_ARITHMETIC_OPS",
    "SymbolicForwardResult",
    "evaluate_program",
    "evaluate_program_forking",
    "range_check",
]
