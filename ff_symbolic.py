"""Bilinear FF dispatch for ADD/SUB/MUL + DIV_S/REM_S + comparisons.

Issue #69 introduced this module for the polynomial-closed core
(ADD/SUB/MUL); issue #75 extended it to DIV_S / REM_S via a pair-
selector matrix plus a boundary truncation; issue #76 extends it
further to the comparison opcodes (EQ / NE / LT_S / GT_S / LE_S /
GE_S / EQZ).

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

Comparison extension (issue #76)
--------------------------------
WASM ``i32.eq`` / ``i32.lt_s`` / ... return ``1 if (vb REL va) else 0``,
a piecewise function: not polynomial. We model them as **gated bilinear
forms**: a *linear* matrix ``M_CMP`` extracts the single scalar
``vb − va`` from the stacked input, and the relation gate
``1 if (diff REL 0) else 0`` is applied at the boundary — the same
"polynomial inside, non-polynomial step at the edge" pattern DIV_S /
REM_S already use. EQZ is the unary case: ``M_EQZ`` extracts ``va``
and the gate fires on ``va == 0``.

This is "gated" rather than truly bilinear because the gate fires
*after* the linear extraction — ``B = e_DIM_VALUE ⊗ e_DIM_VALUE``
would be the bilinear product (used by MUL); for comparisons the
output is a Boolean indicator on the difference, not the product.
Symbolically this becomes :class:`symbolic_executor.IndicatorPoly` —
a (poly, relation) pair whose ``eval_at`` applies the gate.

The cost is the same as DIV_S: comparisons leave the polynomial ring,
so composition past one comparison (e.g. ``LT_S; ADD``) raises
:class:`BlockedOpcodeForSymbolic`. The catalog rows this unblocks
(``compare_lt_s``, ``compare_eqz``, ``native_max``) all either halt
on the indicator directly or feed it to JZ/JNZ, where the forking
executor hoists the relation into a :class:`Guard` rather than
composing it.

Refactoring guards as sign indicators (S1 follow-up)
----------------------------------------------------
Pre-#76 :class:`Guard` only carried an ``eq_zero: bool``. M2 broadens
it to a six-relation enum so the same data shape that backs
``IndicatorPoly`` also backs guards — i.e. a guard *is* a sign
indicator we've asserted to hold along a path. PR #71's S1 worked
hard to express LT_S/GE_S/etc. via "synthetic ``cond − threshold``"
guards on the EQ/NE-only Guard; once everything goes through the
relation field, that synthesis collapses to a straight pass-through.
The catalog renderer (``symbolic_programs_catalog._guard_to_expr``)
is the natural follow-up site to migrate.

Bit-vector extension (issue #77)
--------------------------------
WASM ``i32.and`` / ``i32.or`` / ``i32.xor`` / ``i32.shl`` / ``i32.shr_s``
/ ``i32.shr_u`` / ``i32.clz`` / ``i32.ctz`` / ``i32.popcnt`` are not
polynomial over ℤ — AND/OR/XOR need (ℤ/2ℤ)[bits] to close, shifts are
exponential in the shift amount, and the counter ops are piecewise.
We model the family with the same "linear extractor + boundary
nonlinearity" pattern DIV_S uses: ``M_BITBIN`` is a ``(2, 2*d_model)``
pair-selector that plucks ``(va, vb)`` out of the stacked inputs,
``M_BITUN`` is a ``(1, d_model)`` single-value extractor, and
:func:`symbolic_executor._apply_bitop` is the boundary nonlinearity
that applies the named op to the concrete integers.

Symbolically the result carries as a :class:`BitVec` AST (issue #77)
— the same "polynomial-ring inside, non-poly step at the edge"
pattern :class:`RationalPoly` and :class:`IndicatorPoly` already use.
Hybrid arithmetic (``SUB(31, CLZ(n))`` for log2_floor) lifts into the
BitVec AST so the executor stays closed across one hybrid step; the
catalog rows don't need anything deeper.

Scope
-----
ADD/SUB/MUL via true (bi)linear forms; DIV_S/REM_S via pair-selector
+ boundary trunc (issue #75); comparisons via diff-extractor +
relation gate (issue #76); bitwise family via pair / value selector
+ ``_apply_bitop`` boundary (issue #77). ROTL / ROTR, i64 variants,
and ring-level simplification over (ℤ/2ℤ)[bits] are follow-ups per
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
    BitVec,
    ForkingResult,
    IndicatorPoly,
    ModPoly,
    Poly,
    RationalPoly,
    RationalStackValue,
    REL_EQ, REL_NE, REL_LT, REL_LE, REL_GT, REL_GE,
    SymbolicIntAst,
    SymbolicRemainder,
    _apply_bitop,
    _relation_holds,
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


def _M_CMP_matrix(d_model: int = D_MODEL) -> torch.Tensor:
    """Diff-extractor for binary comparisons (issue #76).

    Shape: ``(1, 2*d_model)``. Computes the single scalar
    ``diff = vb - va`` from the stacked input ``[E(a); E(b)]``. The
    relation gate (``1 if diff REL 0 else 0``) is applied at the
    boundary by :func:`forward_cmp`, mirroring the DIV_S boundary-trunc
    pattern.

    All six binary comparisons share this matrix — only the gate
    differs. Two non-zero entries: ``-1`` at ``DIM_VALUE`` (coefficient
    of ``va``, the top), ``+1`` at ``d_model + DIM_VALUE`` (coefficient
    of ``vb``, stack[SP-1]).
    """
    W = torch.zeros(1, 2 * d_model, dtype=DTYPE)
    W[0, DIM_VALUE] = -1.0                          # coefficient of a (top)
    W[0, d_model + DIM_VALUE] = 1.0                 # coefficient of b (SP-1)
    return W


def _M_EQZ_matrix(d_model: int = D_MODEL) -> torch.Tensor:
    """Single-value extractor for EQZ (issue #76).

    Shape: ``(1, d_model)``. Plucks ``va`` from ``E(a)`` so the gate
    can fire on ``va == 0`` at the boundary. EQZ is the unary
    degenerate case of the comparison family — the difference
    extraction collapses to "just read the value".
    """
    W = torch.zeros(1, d_model, dtype=DTYPE)
    W[0, DIM_VALUE] = 1.0
    return W


def _M_BITBIN_matrix(d_model: int = D_MODEL) -> torch.Tensor:
    """Pair-selector for binary bit ops (issue #77).

    Shape: ``(2, 2*d_model)``. Row 0 picks ``va = ea[DIM_VALUE]`` (the
    top — the shift amount for SHL/SHR, or one operand of AND/OR/XOR);
    row 1 picks ``vb = eb[DIM_VALUE]`` (stack[SP-1] — the value being
    shifted / the other operand). Shared across all six binary bit ops
    (AND / OR / XOR / SHL / SHR_S / SHR_U); the per-op nonlinearity
    fires at the boundary via :func:`symbolic_executor._apply_bitop`.

    Same "linear extractor + boundary nonlinearity" shape as
    ``M_DIV_S`` — the matrix alone doesn't realise the op, it exposes
    that the FF receives exactly the two scalars it needs.
    """
    W = torch.zeros(2, 2 * d_model, dtype=DTYPE)
    W[0, DIM_VALUE] = 1.0                       # va (top)
    W[1, d_model + DIM_VALUE] = 1.0             # vb (SP-1)
    return W


def _M_BITUN_matrix(d_model: int = D_MODEL) -> torch.Tensor:
    """Single-value extractor for unary bit ops (issue #77).

    Shape: ``(1, d_model)``. Plucks ``va`` from ``E(a)`` so the
    boundary nonlinearity (``_clz32`` / ``_ctz32`` / ``_popcnt32``) can
    fire. Shared by CLZ / CTZ / POPCNT; same role as ``M_EQZ`` in the
    comparison family.
    """
    W = torch.zeros(1, d_model, dtype=DTYPE)
    W[0, DIM_VALUE] = 1.0
    return W


# Module-scope cached tensors — read-only; wrapped on access.
M_ADD: torch.Tensor = _M_ADD_matrix()
M_SUB: torch.Tensor = _M_SUB_matrix()
B_MUL: torch.Tensor = _B_MUL_matrix()
M_DIV_S: torch.Tensor = _M_DIV_S_matrix()
M_REM_S: torch.Tensor = _M_REM_S_matrix()
M_CMP: torch.Tensor = _M_CMP_matrix()
M_EQZ: torch.Tensor = _M_EQZ_matrix()
M_BITBIN: torch.Tensor = _M_BITBIN_matrix()
M_BITUN: torch.Tensor = _M_BITUN_matrix()


def n_parameters(d_model: int = D_MODEL) -> int:
    """Parameter count contributed by the analytically-set FF weights.

    Counts non-zero entries across all matrices:

    * ``M_ADD``     2 (two ``+1`` coefficients)
    * ``M_SUB``     2 (one ``+1``, one ``-1``)
    * ``B_MUL``     1 (single rank-1 outer product)
    * ``M_DIV_S``   2 (pair-selector for ``(va, vb)``)
    * ``M_REM_S``   2 (pair-selector for ``(va, vb)``)
    * ``M_CMP``     2 (diff-extractor; shared by all six binary comparisons)
    * ``M_EQZ``     1 (single-value extractor)
    * ``M_BITBIN``  2 (pair-selector; shared by AND/OR/XOR/SHL/SHR_S/SHR_U)
    * ``M_BITUN``   1 (single-value extractor; shared by CLZ/CTZ/POPCNT)

    Total: 15. The stored tensors are larger but the analytically-set
    content is fifteen bits. Sharing matrices across families (six
    comparisons behind ``M_CMP``, six binary bit ops behind ``M_BITBIN``,
    three counter ops behind ``M_BITUN``) keeps the weight budget
    proportional to the *number of operator families* the transformer
    needs to route, not the number of opcodes.
    """
    return 15


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


# Map from comparison opcode (binary) to the relation code applied to
# ``vb − va``. Single source of truth shared by the symbolic and
# numeric forward paths so they cannot drift.
_OP_RELATION = {
    isa.OP_EQ: REL_EQ,
    isa.OP_NE: REL_NE,
    isa.OP_LT_S: REL_LT,
    isa.OP_GT_S: REL_GT,
    isa.OP_LE_S: REL_LE,
    isa.OP_GE_S: REL_GE,
}


def forward_cmp(ea: torch.Tensor, eb: torch.Tensor, op: int) -> torch.Tensor:
    """Compute ``E(1 if (vb REL va) else 0)`` for one of the six binary cmps.

    ``op`` is one of ``isa.OP_EQ / OP_NE / OP_LT_S / OP_GT_S / OP_LE_S
    / OP_GE_S``. Applies ``M_CMP`` as a diff-extractor (``diff = vb - va``
    via the ``(1, 2*d_model)`` linear form), then the relation gate at
    the boundary. The output is re-embedded via ``E`` so the result
    shares the same scalar-at-``DIM_VALUE`` layout as every other value
    in the model — a downstream FF can read it back without any
    re-projection.

    The gate is an ``int(_relation_holds(...))`` call: not a matmul.
    That's the honest cost of integer-valued comparisons in a polynomial
    framework — same shape as DIV_S's ``_trunc_div`` boundary step.
    """
    relation = _OP_RELATION.get(op)
    if relation is None:
        raise ValueError(
            f"forward_cmp: op {isa.OP_NAMES.get(op, op)!r} is not a "
            f"binary comparison opcode (expected one of {sorted(_OP_RELATION)})"
        )
    stacked = torch.cat([ea, eb])                    # shape (2*d_model,)
    diff = float((M_CMP @ stacked).item())           # shape (), then scalar
    bit = 1 if _relation_holds(relation, diff) else 0
    out = torch.zeros(ea.shape[0], dtype=DTYPE)
    out[DIM_VALUE] = float(bit)
    return out


def forward_eqz(ea: torch.Tensor) -> torch.Tensor:
    """Compute ``E(1 if va == 0 else 0)`` from ``E(a)``.

    Applies ``M_EQZ`` as a single-value extractor (``va = ea[DIM_VALUE]``
    via the ``(1, d_model)`` linear form), then the equality gate at
    the boundary. Unary degenerate case of :func:`forward_cmp`.
    """
    va = float((M_EQZ @ ea).item())
    bit = 1 if _relation_holds(REL_EQ, va) else 0
    out = torch.zeros(ea.shape[0], dtype=DTYPE)
    out[DIM_VALUE] = float(bit)
    return out


# Map from binary bit opcode to the named op understood by
# :func:`symbolic_executor._apply_bitop` and :class:`BitVec`.
_BITBIN_OP_NAME = {
    isa.OP_AND: "AND",
    isa.OP_OR: "OR",
    isa.OP_XOR: "XOR",
    isa.OP_SHL: "SHL",
    isa.OP_SHR_S: "SHR_S",
    isa.OP_SHR_U: "SHR_U",
}
_BITUN_OP_NAME = {
    isa.OP_CLZ: "CLZ",
    isa.OP_CTZ: "CTZ",
    isa.OP_POPCNT: "POPCNT",
}


def forward_bit_binary(ea: torch.Tensor, eb: torch.Tensor, op: int) -> torch.Tensor:
    """Compute ``E(bit_binary(vb, va))`` from ``E(a), E(b)`` via ``M_BITBIN``.

    ``op`` is one of ``OP_AND / OP_OR / OP_XOR / OP_SHL / OP_SHR_S / OP_SHR_U``.
    The pair-selector extracts ``[va, vb] = M_BITBIN @ stacked`` (``va`` = top,
    ``vb`` = SP-1) and then :func:`symbolic_executor._apply_bitop` applies the
    named op with left=``vb``, right=``va`` (the "natural reading" convention
    matching WASM: ``SHL(value, count) = value << count`` with value=SP-1,
    count=top).

    The output is re-embedded via ``E`` so the result shares the same
    scalar-at-``DIM_VALUE`` layout as every other value in the model.
    """
    name = _BITBIN_OP_NAME.get(op)
    if name is None:
        raise ValueError(
            f"forward_bit_binary: op {isa.OP_NAMES.get(op, op)!r} is not a "
            f"binary bit opcode (expected one of {sorted(_BITBIN_OP_NAME)})"
        )
    stacked = torch.cat([ea, eb])                       # shape (2*d_model,)
    pair = M_BITBIN @ stacked                           # shape (2,): [va, vb]
    va = int(round(float(pair[0].item())))
    vb = int(round(float(pair[1].item())))
    result = _apply_bitop(name, [vb, va])               # left=SP-1, right=top
    out = torch.zeros(ea.shape[0], dtype=DTYPE)
    out[DIM_VALUE] = float(result)
    return out


def forward_bit_unary(ea: torch.Tensor, op: int) -> torch.Tensor:
    """Compute ``E(bit_unary(va))`` from ``E(a)`` via ``M_BITUN``.

    ``op`` is one of ``OP_CLZ / OP_CTZ / OP_POPCNT``. The single-value
    extractor plucks ``va = M_BITUN @ ea`` and then
    :func:`symbolic_executor._apply_bitop` applies the named counter op.
    Unary degenerate case of :func:`forward_bit_binary`.
    """
    name = _BITUN_OP_NAME.get(op)
    if name is None:
        raise ValueError(
            f"forward_bit_unary: op {isa.OP_NAMES.get(op, op)!r} is not a "
            f"unary bit opcode (expected one of {sorted(_BITUN_OP_NAME)})"
        )
    va = int(round(float((M_BITUN @ ea).item())))
    result = _apply_bitop(name, [va])
    out = torch.zeros(ea.shape[0], dtype=DTYPE)
    out[DIM_VALUE] = float(result)
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


def symbolic_cmp(pa: Poly, pb: Poly, op: int) -> IndicatorPoly:
    """Symbolic comparison; corresponds to ``forward_cmp`` over :class:`Poly`.

    Order matches ``forward_cmp`` (``pa`` is top, ``pb`` is stack[SP-1]):
    returns ``IndicatorPoly(poly=pb - pa, relation=_OP_RELATION[op])`` so
    that ``vb REL va`` becomes ``(pb - pa) REL 0`` — the same identity
    ``M_CMP``'s linear extraction realises numerically.

    The gate fires only at :meth:`IndicatorPoly.eval_at`, mirroring
    :class:`RationalPoly`'s boundary truncation. Composition past this
    point (e.g. ``LT_S; ADD``) raises
    :class:`symbolic_executor.SymbolicOpNotSupported` upstream.
    """
    relation = _OP_RELATION.get(op)
    if relation is None:
        raise ValueError(
            f"symbolic_cmp: op {isa.OP_NAMES.get(op, op)!r} is not a "
            f"binary comparison opcode (expected one of {sorted(_OP_RELATION)})"
        )
    return IndicatorPoly(poly=pb - pa, relation=relation)


def symbolic_eqz(pa: Poly) -> IndicatorPoly:
    """Symbolic EQZ; corresponds to ``forward_eqz`` over :class:`Poly`.

    Returns ``IndicatorPoly(poly=pa, relation=REL_EQ)``; the gate
    ``pa == 0`` fires at :meth:`IndicatorPoly.eval_at`.
    """
    return IndicatorPoly(poly=pa, relation=REL_EQ)


def symbolic_bit_binary(pa: SymbolicIntAst, pb: SymbolicIntAst, op: int) -> BitVec:
    """Symbolic binary bit op; corresponds to ``forward_bit_binary``.

    Order matches ``forward_bit_binary`` (``pa`` is top, ``pb`` is
    stack[SP-1]): returns ``BitVec(name, (pb, pa))`` — operands are
    stored in the natural left-to-right reading order
    ``(left=SP-1, right=top)``. The gate applies ``_apply_bitop(name,
    [left, right])`` at :meth:`BitVec.eval_at`, same boundary-step
    pattern :class:`RationalPoly` uses for DIV_S.

    ``pa`` / ``pb`` may be :class:`Poly` or :class:`BitVec` — nested
    bit programs (e.g. ``AND`` composed with a prior ``SHR_U``) land
    here with a :class:`BitVec` on one or both sides.
    """
    name = _BITBIN_OP_NAME.get(op)
    if name is None:
        raise ValueError(
            f"symbolic_bit_binary: op {isa.OP_NAMES.get(op, op)!r} is not a "
            f"binary bit opcode (expected one of {sorted(_BITBIN_OP_NAME)})"
        )
    return BitVec(op=name, operands=(pb, pa))


def symbolic_bit_unary(pa: SymbolicIntAst, op: int) -> BitVec:
    """Symbolic unary bit op; corresponds to ``forward_bit_unary``.

    Returns ``BitVec(name, (pa,))``; the gate applies the named counter
    op (``_clz32`` / ``_ctz32`` / ``_popcnt32``) at the boundary.
    """
    name = _BITUN_OP_NAME.get(op)
    if name is None:
        raise ValueError(
            f"symbolic_bit_unary: op {isa.OP_NAMES.get(op, op)!r} is not a "
            f"unary bit opcode (expected one of {sorted(_BITUN_OP_NAME)})"
        )
    return BitVec(op=name, operands=(pa,))


def symbolic_bit_arith(pa: SymbolicIntAst, pb: SymbolicIntAst, op: int) -> BitVec:
    """Hybrid arithmetic lifted into :class:`BitVec` (issue #77).

    Fires when :func:`symbolic_executor._apply_poly_op` sees ADD / SUB /
    MUL with at least one :class:`BitVec` operand (log2_floor's
    ``SUB(31, CLZ(n))`` case). Operand order matches
    :func:`symbolic_bit_binary`: ``pa`` = top, ``pb`` = SP-1, stored
    left-to-right in the BitVec AST as ``(pb, pa)`` so the expression
    reads as "left OP right" = ``SP-1 OP top`` (matching WASM).
    """
    name = {isa.OP_ADD: "ADD", isa.OP_SUB: "SUB", isa.OP_MUL: "MUL"}.get(op)
    if name is None:
        raise ValueError(
            f"symbolic_bit_arith: op {isa.OP_NAMES.get(op, op)!r} is not "
            f"lifted-arithmetic (expected ADD/SUB/MUL)"
        )
    return BitVec(op=name, operands=(pb, pa))


# Arithmetic primitives packaged for :func:`symbolic_executor.run_forking`'s
# ``arithmetic_ops`` hook (issue #68 S3; extended for DIV_S/REM_S in #75;
# extended for the comparison family in #76). The wrappers flip arg order
# where needed: the forking executor's convention is ``op(a, b)`` with
# ``a`` = stack[SP-1] and ``b`` = top. ``symbolic_sub`` / ``symbolic_div_s``
# / ``symbolic_rem_s`` / ``symbolic_cmp`` are written in the FF ``(pa, pb)``
# order where ``pa`` = top, ``pb`` = SP-1, so we call them with ``(b, a)``.
FF_ARITHMETIC_OPS = ArithmeticOps(
    add=lambda a, b: symbolic_add(a, b),
    sub=lambda a, b: symbolic_sub(b, a),
    mul=lambda a, b: symbolic_mul(a, b),
    div_s=lambda a, b: symbolic_div_s(b, a),
    rem_s=lambda a, b: symbolic_rem_s(b, a),
    cmp_eq=lambda a, b: symbolic_cmp(b, a, isa.OP_EQ),
    cmp_ne=lambda a, b: symbolic_cmp(b, a, isa.OP_NE),
    cmp_lt_s=lambda a, b: symbolic_cmp(b, a, isa.OP_LT_S),
    cmp_gt_s=lambda a, b: symbolic_cmp(b, a, isa.OP_GT_S),
    cmp_le_s=lambda a, b: symbolic_cmp(b, a, isa.OP_LE_S),
    cmp_ge_s=lambda a, b: symbolic_cmp(b, a, isa.OP_GE_S),
    eqz=lambda a: symbolic_eqz(a),
    # Bit-vector primitives (issue #77). The forking executor's convention
    # is ``op(a, b)`` with ``a`` = SP-1 and ``b`` = top (same as the Poly
    # primitives above). ``symbolic_bit_binary`` / ``symbolic_bit_arith``
    # take ``(pa, pb, op)`` in FF order (pa=top, pb=SP-1) and store them
    # as ``(pb, pa) = (SP-1, top)`` = PUSH order, so we pass ``(b, a)``
    # — identical flip to the ``sub`` / ``div_s`` / ``cmp_*`` wrappers
    # above. Passing ``(a, b)`` unswapped (the pre-#108 bug) put operands
    # in ``(top, SP-1)`` order, silently breaking non-commutative ops
    # like SHL (`3 << 2` returned `16` instead of `12`).
    bit_and=lambda a, b: symbolic_bit_binary(b, a, isa.OP_AND),
    bit_or=lambda a, b: symbolic_bit_binary(b, a, isa.OP_OR),
    bit_xor=lambda a, b: symbolic_bit_binary(b, a, isa.OP_XOR),
    bit_shl=lambda a, b: symbolic_bit_binary(b, a, isa.OP_SHL),
    bit_shr_s=lambda a, b: symbolic_bit_binary(b, a, isa.OP_SHR_S),
    bit_shr_u=lambda a, b: symbolic_bit_binary(b, a, isa.OP_SHR_U),
    bit_clz=lambda a: symbolic_bit_unary(a, isa.OP_CLZ),
    bit_ctz=lambda a: symbolic_bit_unary(a, isa.OP_CTZ),
    bit_popcnt=lambda a: symbolic_bit_unary(a, isa.OP_POPCNT),
    # Hybrid arithmetic lifted into the BitVec AST — same flip as above,
    # for the same reason. Under the pre-#108 bug the commutative cases
    # (ADD / MUL) were cosmetically wrong (structural-equality failure
    # only); SUB was also silently numerically wrong.
    bit_add=lambda a, b: symbolic_bit_arith(b, a, isa.OP_ADD),
    bit_sub=lambda a, b: symbolic_bit_arith(b, a, isa.OP_SUB),
    bit_mul=lambda a, b: symbolic_bit_arith(b, a, isa.OP_MUL),
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
    isa.OP_EQ, isa.OP_NE,
    isa.OP_LT_S, isa.OP_GT_S, isa.OP_LE_S, isa.OP_GE_S,
    isa.OP_EQZ,
    isa.OP_SWAP, isa.OP_OVER, isa.OP_ROT,
    # Issue #77 bit-vector fragment.
    isa.OP_AND, isa.OP_OR, isa.OP_XOR,
    isa.OP_SHL, isa.OP_SHR_S, isa.OP_SHR_U,
    isa.OP_CLZ, isa.OP_CTZ, isa.OP_POPCNT,
    # Issue #105: LOCAL slots are transparent to the FF dispatch — a slot
    # just stores and retrieves whatever stack value was written to it
    # (Poly / BitVec / RationalPoly / IndicatorPoly / SymbolicRemainder),
    # invoking no FF weight matrix. The equivalence claim from #69
    # extends unchanged: each LOCAL_* step routes a single symbolic
    # value through the locals table, so every arithmetic composition
    # still bottoms out at the same M_ADD / M_SUB / B_MUL (etc.) call
    # site as it would under pure stack manipulation.
    isa.OP_LOCAL_GET, isa.OP_LOCAL_SET, isa.OP_LOCAL_TEE,
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
    locals_: Dict[int, RationalStackValue] = {}
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

    def _require_int_ast(v: RationalStackValue, op_name: str) -> SymbolicIntAst:
        """Accept :class:`Poly` or :class:`BitVec` (the i32-valued AST types).

        Used by bit ops and hybrid arithmetic: composition across the
        BitVec boundary is in scope (``AND(SHR_U(k, n))``), but rational
        / indicator operands still raise — those are follow-ups.
        """
        if not isinstance(v, (Poly, BitVec)):
            raise BlockedOpcodeForSymbolic(
                f"{op_name} on rational/indicator stack entries is out of scope"
            )
        return v

    for instr in prog:
        op = instr.op
        if op not in _SCOPE_OPS:
            raise BlockedOpcodeForSymbolic(
                f"op {isa.OP_NAMES.get(op, f'?{op}')!r} is out of scope for the "
                f"bilinear FF dispatch (ADD/SUB/MUL/DIV_S/REM_S + comparisons "
                f"+ bit-vector fragment + stack manip only)"
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
            if isinstance(a, BitVec) or isinstance(b, BitVec):
                ai = _require_int_ast(a, "ADD"); bi = _require_int_ast(b, "ADD")
                stack.append(symbolic_bit_arith(bi, ai, isa.OP_ADD))
            else:
                stack.append(symbolic_add(_require_poly(a, "ADD"),
                                          _require_poly(b, "ADD")))
        elif op == isa.OP_SUB:
            b = _pop(); a = _pop()
            if isinstance(a, BitVec) or isinstance(b, BitVec):
                ai = _require_int_ast(a, "SUB"); bi = _require_int_ast(b, "SUB")
                stack.append(symbolic_bit_arith(bi, ai, isa.OP_SUB))
            else:
                # ``symbolic_sub`` takes ``(pa=top, pb=SP-1)`` and returns
                # ``pb - pa = SP-1 - top`` (WASM SUB). Mirrors the forking
                # wrapper ``FF_ARITHMETIC_OPS.sub = λa,b: symbolic_sub(b, a)``
                # and matches ``DIV_S`` / ``REM_S`` / ``symbolic_cmp``
                # below. Issue #105 exposed this via ``gen_mixed_signs``,
                # whose ``emit_negate`` sequence produces a negative
                # coefficient only when the subtrahend-order is correct.
                stack.append(symbolic_sub(_require_poly(b, "SUB"),
                                          _require_poly(a, "SUB")))
        elif op == isa.OP_MUL:
            b = _pop(); a = _pop()
            if isinstance(a, BitVec) or isinstance(b, BitVec):
                ai = _require_int_ast(a, "MUL"); bi = _require_int_ast(b, "MUL")
                stack.append(symbolic_bit_arith(bi, ai, isa.OP_MUL))
            else:
                stack.append(symbolic_mul(_require_poly(a, "MUL"),
                                          _require_poly(b, "MUL")))
        elif op == isa.OP_DIV_S:
            b = _require_poly(_pop(), "DIV_S"); a = _require_poly(_pop(), "DIV_S")
            # forking executor convention: a = SP-1, b = top; result = a / b.
            # symbolic_div_s uses (pa=top, pb=SP-1) ordering.
            stack.append(symbolic_div_s(b, a))
        elif op == isa.OP_REM_S:
            b = _require_poly(_pop(), "REM_S"); a = _require_poly(_pop(), "REM_S")
            stack.append(symbolic_rem_s(b, a))
        elif op in _OP_RELATION:
            # Binary comparison (issue #76) or BitVec-extended comparison (#77).
            name = isa.OP_NAMES.get(op, f"?{op}")
            b = _pop(); a = _pop()
            if isinstance(a, Poly) and isinstance(b, Poly):
                stack.append(symbolic_cmp(b, a, op))
            else:
                ai = _require_int_ast(a, name); bi = _require_int_ast(b, name)
                # is_power_of_2 composes POPCNT (BitVec) with PUSH 1 (Poly).
                # Build the (vb − va) difference as a BitVec AST in PUSH
                # order (operands[0] = SP-1 = vb, operands[1] = top = va)
                # so ``_apply_bitop("SUB", [vb, va])`` returns ``vb − va``
                # and matches the Poly path's ``symbolic_cmp(b, a).poly =
                # pb − pa = SP-1 − top``. Pre-#108 this stored
                # ``(bi, ai) = (top, SP-1)``, inverting the sign — masked
                # for EQ (``x == 0 ⇔ −x == 0``), but wrong for LT/LE/GT/GE.
                stack.append(IndicatorPoly(
                    poly=BitVec(op="SUB", operands=(ai, bi)),
                    relation=_OP_RELATION[op],
                ))
        elif op == isa.OP_EQZ:
            a = _pop()
            if isinstance(a, Poly):
                stack.append(symbolic_eqz(a))
            else:
                ai = _require_int_ast(a, "EQZ")
                stack.append(IndicatorPoly(poly=ai, relation=REL_EQ))
        elif op in _BITBIN_OP_NAME:
            # Binary bit op (issue #77). ``_pop()`` order: ``b`` = top,
            # ``a`` = SP-1. ``symbolic_bit_binary`` takes ``(pa=top,
            # pb=SP-1)`` and stores ``(pb, pa) = (SP-1, top)`` = PUSH
            # order, so we pass ``(b, a)``. Pre-#108 passed ``(a, b)``,
            # producing ``(top, SP-1)`` — silently wrong for
            # non-commutative ops (SHL ``3 << 2`` returned ``16``).
            name = isa.OP_NAMES.get(op, f"?{op}")
            b = _require_int_ast(_pop(), name)
            a = _require_int_ast(_pop(), name)
            stack.append(symbolic_bit_binary(b, a, op))
        elif op in _BITUN_OP_NAME:
            name = isa.OP_NAMES.get(op, f"?{op}")
            a = _require_int_ast(_pop(), name)
            stack.append(symbolic_bit_unary(a, op))
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
        elif op == isa.OP_LOCAL_GET:
            # FF invariant: the slot holds the identical symbolic value
            # that was written to it — no FF weight matrix is applied on
            # read. The Poly / BitVec / RationalPoly / IndicatorPoly
            # identity is preserved by reference, so downstream ADD / SUB
            # / MUL / … dispatch is unchanged from the pure-stack path.
            slot = int(instr.arg)
            if slot not in locals_:
                raise IndexError(f"LOCAL_GET of uninitialized slot {slot}")
            stack.append(locals_[slot])
        elif op == isa.OP_LOCAL_SET:
            # Store pop-top into the slot. No algebra over the value —
            # the FF dispatch never inspects slot contents; it only
            # routes the write on SET and the read on GET.
            slot = int(instr.arg)
            locals_[slot] = _pop()
        elif op == isa.OP_LOCAL_TEE:
            # Peek-and-store: slot mirrors current top without popping.
            slot = int(instr.arg)
            if not stack:
                raise IndexError("LOCAL_TEE on empty stack")
            locals_[slot] = stack[-1]
        else:  # pragma: no cover — guarded by _SCOPE_OPS above
            raise BlockedOpcodeForSymbolic(f"unreachable: op {op}")

    top: RationalStackValue = stack[-1] if stack else Poly.constant(0)
    return SymbolicForwardResult(top=top, stack=list(stack),
                                 n_heads=n_heads, bindings=bindings)


# ─── Mod-2³² symbolic primitives (issue #78 option (b)) ───────────
#
# ``symbolic_add`` / ``symbolic_sub`` / ``symbolic_mul`` above realise the
# FF bilinear form over ℤ — the equivalence theorem from issue #69 is
# structural over ℤ under a :func:`range_check` assumption. Issue #78
# adds a sibling fragment that carries the i32 wrap semantics through
# the algebra itself: the same spec (M_ADD / M_SUB / B_MUL) interpreted
# over :class:`ModPoly` closes the ``& MASK32`` gap symbolically.
#
# Concretely: ``forward_add / forward_sub / forward_mul`` on float tensors
# compute ``(va + vb) & MASK32`` / etc. — the mask lives *outside* the
# bilinear form in :meth:`executor.CompiledModel.forward`. On the
# symbolic side, ``symbolic_add_mod`` / etc. evaluate the same operator
# tree over :class:`ModPoly`, whose ``__post_init__`` reduces coefficients
# modulo 2³² after every step. That reduction is the same ``& MASK32``,
# now visible as a single line of algebra instead of a numeric-only
# boundary fact.
#
# This is the "real work" option from issue #78 (as opposed to Option (a),
# which pins :func:`range_check` and stops). It's orthogonal to the
# Poly-over-ℤ fragment: both continue to work; ``ModPoly`` is a proof
# artefact for the wrap case, not a replacement.


def symbolic_add_mod(pa: ModPoly, pb: ModPoly) -> ModPoly:
    """ModPoly addition; corresponds to ``forward_add`` over ℤ/2³².

    Same spec as :func:`symbolic_add`, reinterpreted in the quotient
    ring. The coefficient reduction in ``ModPoly.__post_init__`` is
    precisely the ``& MASK32`` step ``CompiledModel.forward`` applies
    after the bilinear form.
    """
    return pa + pb


def symbolic_sub_mod(pa: ModPoly, pb: ModPoly) -> ModPoly:
    """ModPoly subtraction; corresponds to ``forward_sub`` over ℤ/2³².

    Order matches :func:`symbolic_sub`: returns ``pb - pa``.
    """
    return pb - pa


def symbolic_mul_mod(pa: ModPoly, pb: ModPoly) -> ModPoly:
    """ModPoly multiplication; corresponds to ``forward_mul`` over ℤ/2³²."""
    return pa * pb


# Ops this module understands when interpreting the program over ℤ/2³².
# Narrower than ``_SCOPE_OPS``: option (b) only covers the polynomial-
# closed fragment (ADD / SUB / MUL + stack manip). Everything else —
# comparisons (boundary gate), rationals (boundary truncation), bit-ops
# (non-polynomial over ℤ/2³² as well; bit decomposition lives in the
# ``BitVec`` wrapper) — raises ``BlockedOpcodeForSymbolic``. Those
# fragments already apply the correct wrap at their own boundaries.
_SCOPE_OPS_MOD = {
    isa.OP_PUSH, isa.OP_POP, isa.OP_DUP, isa.OP_HALT, isa.OP_NOP,
    isa.OP_ADD, isa.OP_SUB, isa.OP_MUL,
    isa.OP_SWAP, isa.OP_OVER, isa.OP_ROT,
    # Issue #105: LOCAL slots are polynomial-closed — a slot stores and
    # retrieves whatever ModPoly stack value was written. They invoke
    # no bilinear form, so the mod-2³² homomorphism (``ModPoly.from_poly
    # ∘ evaluate_program == evaluate_program_mod`` on the collapsed-Poly
    # catalog) extends unchanged once both drivers carry a locals table.
    isa.OP_LOCAL_GET, isa.OP_LOCAL_SET, isa.OP_LOCAL_TEE,
}


@dataclass
class SymbolicForwardResultMod:
    """Outcome of :func:`evaluate_program_mod` — mod-2³² sibling of
    :class:`SymbolicForwardResult`. ``top`` is a :class:`ModPoly`.
    """
    top: ModPoly
    stack: List[ModPoly]
    n_heads: int
    bindings: Dict[int, int]


def evaluate_program_mod(prog) -> SymbolicForwardResultMod:
    """Run ``prog`` through the bilinear FF interpreters over ℤ/2³².

    Semantics match :func:`evaluate_program` restricted to the polynomial
    fragment {ADD, SUB, MUL, PUSH, POP, DUP, HALT, NOP, SWAP, OVER, ROT}.
    The one difference from the Poly-over-ℤ driver is the value type:
    every stack entry is a :class:`ModPoly` whose coefficients are
    reduced modulo 2³² after every arithmetic operation. Structurally
    the output is ``ModPoly.from_poly(evaluate_program(prog).top)`` — a
    theorem verified by :func:`test_ff_symbolic.test_modpoly_equivalence`.
    """
    stack: List[ModPoly] = []
    locals_: Dict[int, ModPoly] = {}
    bindings: Dict[int, int] = {}
    next_var = 0
    n_heads = 0

    for instr in prog:
        op = instr.op
        if op not in _SCOPE_OPS_MOD:
            raise BlockedOpcodeForSymbolic(
                f"op {isa.OP_NAMES.get(op, f'?{op}')!r} is out of scope for "
                f"the mod-2³² bilinear FF dispatch (ADD/SUB/MUL + stack "
                f"manip only). Comparisons / rationals / bit-ops have their "
                f"own boundary wrap in IndicatorPoly / RationalPoly / BitVec."
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
            stack.append(ModPoly.variable(v))
        elif op == isa.OP_POP:
            if not stack:
                raise IndexError("symbolic stack underflow")
            stack.pop()
        elif op == isa.OP_DUP:
            if not stack:
                raise IndexError("dup on empty stack")
            stack.append(stack[-1])
        elif op == isa.OP_ADD:
            b = stack.pop(); a = stack.pop()
            stack.append(symbolic_add_mod(a, b))
        elif op == isa.OP_SUB:
            b = stack.pop(); a = stack.pop()
            # ``symbolic_sub_mod`` takes ``(pa=top, pb=SP-1)`` and returns
            # ``pb - pa = SP-1 - top``, matching WASM SUB and the
            # ``FF_ARITHMETIC_OPS.sub`` wrapper. Same ordering fix as the
            # Poly driver above (issue #105).
            stack.append(symbolic_sub_mod(b, a))
        elif op == isa.OP_MUL:
            b = stack.pop(); a = stack.pop()
            stack.append(symbolic_mul_mod(a, b))
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
        elif op == isa.OP_LOCAL_GET:
            # Mod-2³² slot read: return the stored ModPoly unchanged.
            # No bilinear form is invoked, so mod-32 reduction is a
            # no-op on the GET edge — the coefficients were already
            # reduced when the value was produced by the arithmetic
            # primitive that wrote it.
            slot = int(instr.arg)
            if slot not in locals_:
                raise IndexError(f"LOCAL_GET of uninitialized slot {slot}")
            stack.append(locals_[slot])
        elif op == isa.OP_LOCAL_SET:
            # Mod-2³² slot write: pop-top into the slot. The ModPoly's
            # invariant (coefficients in [0, 2³²)) is preserved by
            # reference; SET applies no algebra.
            slot = int(instr.arg)
            if not stack:
                raise IndexError("LOCAL_SET on empty stack")
            locals_[slot] = stack.pop()
        elif op == isa.OP_LOCAL_TEE:
            # Mod-2³² peek-and-store: slot mirrors current top.
            slot = int(instr.arg)
            if not stack:
                raise IndexError("LOCAL_TEE on empty stack")
            locals_[slot] = stack[-1]
        else:  # pragma: no cover — guarded by _SCOPE_OPS_MOD above
            raise BlockedOpcodeForSymbolic(f"unreachable: op {op}")

    top = stack[-1] if stack else ModPoly.constant(0)
    return SymbolicForwardResultMod(top=top, stack=list(stack),
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


# ─── Path B (issue #109) — weight-layer realisation of ClosedForm /
# ProductForm. Path B is opt-in via this entrypoint; the default
# ``evaluate_program_forking`` above stays Path A (#107). See
# :mod:`path_b` for the dispatcher and :mod:`ff_symbolic_poly_embedding`
# / :mod:`ff_symbolic_recurrent` / :mod:`algebraic_poly` for the three
# sub-paths.
from path_b import (  # noqa: E402  — late import to avoid a cycle
    PATH_B_OUT_OF_SCOPE_EXCEPTION,
    PathBOutOfScope,
    PathBResult,
    evaluate_program_forking_weight_layer,
    path_b_in_scope,
)


__all__ = [
    "BlockedOpcodeForSymbolic",
    "RangeCheckFailure",
    "I32_MIN", "I32_MAX",
    "E", "E_inv",
    "M_ADD", "M_SUB", "B_MUL", "M_DIV_S", "M_REM_S", "M_CMP", "M_EQZ",
    "M_BITBIN", "M_BITUN",
    "n_parameters",
    "forward_add", "forward_sub", "forward_mul",
    "forward_div_s", "forward_rem_s",
    "forward_cmp", "forward_eqz",
    "forward_bit_binary", "forward_bit_unary",
    "symbolic_add", "symbolic_sub", "symbolic_mul",
    "symbolic_div_s", "symbolic_rem_s",
    "symbolic_cmp", "symbolic_eqz",
    "symbolic_bit_binary", "symbolic_bit_unary", "symbolic_bit_arith",
    "symbolic_add_mod", "symbolic_sub_mod", "symbolic_mul_mod",
    "FF_ARITHMETIC_OPS",
    "SymbolicForwardResult",
    "SymbolicForwardResultMod",
    "evaluate_program",
    "evaluate_program_mod",
    "evaluate_program_forking",
    "range_check",
    # Path B (issue #109)
    "PATH_B_OUT_OF_SCOPE_EXCEPTION",
    "PathBOutOfScope",
    "PathBResult",
    "evaluate_program_forking_weight_layer",
    "path_b_in_scope",
]
