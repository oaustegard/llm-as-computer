"""Tests for ff_symbolic (issue #69 + issue #68 S3).

Verifies the bilinear FF dispatch realises the same polynomial the symbolic
executor emits — not just numerically, but *structurally*. For every
collapsed catalog program, the test asserts
``forward_symbolic(P).top == SymbolicExecutor(P).top`` on canonical Poly
equality (value-compare), alongside a numerical agreement check against
:class:`NumPyExecutor` on concrete inputs.

S3 extends the cross-check past the branchless fragment: guarded programs
(JZ/JNZ on symbolic conditions) and unrolled programs (bounded loops at
concrete inputs) are driven through the forking executor with the bilinear
FF primitives, and compared structurally against the symbolic executor's
native ``run_forking``.

Layout:
  * ``test_primitives_*`` — unit-level checks on E / M_ADD / M_SUB / B_MUL.
  * ``test_range_check`` — Option (a) bound enforcement.
  * ``test_equivalence_*`` — the core equivalence theorem, parametrised
    over every row in ``symbolic_programs_catalog`` whose status is
    ``STATUS_COLLAPSED``. The ``_guarded_*`` / ``_unrolled_*`` variants
    cover ``STATUS_COLLAPSED_GUARDED`` / ``STATUS_COLLAPSED_UNROLLED``.
  * ``test_blocked_opcodes`` — non-polynomial ops are rejected rather than
    silently returning a wrong-but-plausible Poly.
"""

from __future__ import annotations

import sys
import traceback
from typing import List, Tuple

import torch

import ff_symbolic as ff
import isa
from executor import CompiledModel, NumPyExecutor
from isa import DIM_VALUE, program
from symbolic_executor import (
    BitVec,
    GuardedPoly,
    IndicatorPoly,
    ModPoly,
    Poly,
    RationalPoly,
    REL_EQ,
    REL_GE,
    REL_GT,
    REL_LE,
    REL_LT,
    REL_NE,
    SymbolicRemainder,
    run_forking,
    run_symbolic,
)
from symbolic_programs_catalog import (
    STATUS_COLLAPSED,
    STATUS_COLLAPSED_GUARDED,
    STATUS_COLLAPSED_UNROLLED,
    _default_catalog,
    classify_program,
)


# ─── Test harness — tiny, avoids a pytest dep ──────────────────────

_failures: List[str] = []


def _fail(name: str, detail: str):
    _failures.append(f"{name}: {detail}")
    print(f"  FAIL  {name}  {detail}")


def _pass(name: str):
    print(f"  PASS  {name}")


def _check(name: str, cond: bool, detail: str = ""):
    if cond:
        _pass(name)
    else:
        _fail(name, detail)


# ─── Primitive-level tests ────────────────────────────────────────

def test_primitives_add_sub_mul():
    """E / forward_add / forward_sub / forward_mul agree with Python on samples.

    The claim here is narrow: on scalar-embedded inputs the bilinear form
    computes a+b, b-a, and a*b exactly. All eight edge cases for sign
    combinations are covered.
    """
    samples: List[Tuple[int, int]] = [
        (5, 7), (-3, 4), (-6, -9), (0, 12), (12, 0),
        (100, -50), (2**15 - 1, 3), (-7, 2**15),
    ]
    for a, b in samples:
        ea, eb = ff.E(a), ff.E(b)
        _check(f"forward_add({a},{b})",
               ff.E_inv(ff.forward_add(ea, eb)) == a + b,
               f"got {ff.E_inv(ff.forward_add(ea, eb))}, expect {a + b}")
        _check(f"forward_sub({a},{b})",
               ff.E_inv(ff.forward_sub(ea, eb)) == b - a,
               f"got {ff.E_inv(ff.forward_sub(ea, eb))}, expect {b - a}")
        _check(f"forward_mul({a},{b})",
               ff.E_inv(ff.forward_mul(ea, eb)) == a * b,
               f"got {ff.E_inv(ff.forward_mul(ea, eb))}, expect {a * b}")


def test_primitives_matrix_shapes():
    """Analytically-set matrices have the expected shapes and signatures."""
    d = ff.D_MODEL
    _check("M_ADD shape", ff.M_ADD.shape == (d, 2 * d))
    _check("M_SUB shape", ff.M_SUB.shape == (d, 2 * d))
    _check("B_MUL shape", ff.B_MUL.shape == (d, d))

    # M_ADD: only DIM_VALUE slot in both halves is nonzero, both +1.
    _check("M_ADD[DIM_VALUE, DIM_VALUE] == 1",
           float(ff.M_ADD[DIM_VALUE, DIM_VALUE].item()) == 1.0)
    _check("M_ADD[DIM_VALUE, d+DIM_VALUE] == 1",
           float(ff.M_ADD[DIM_VALUE, d + DIM_VALUE].item()) == 1.0)

    # M_SUB: a contributes -1, b contributes +1 (SUB semantics: b - a).
    _check("M_SUB[DIM_VALUE, DIM_VALUE] == -1",
           float(ff.M_SUB[DIM_VALUE, DIM_VALUE].item()) == -1.0)
    _check("M_SUB[DIM_VALUE, d+DIM_VALUE] == +1",
           float(ff.M_SUB[DIM_VALUE, d + DIM_VALUE].item()) == 1.0)

    # B_MUL: rank-1 outer product with a single +1 at [DIM_VALUE, DIM_VALUE].
    nonzero = (ff.B_MUL != 0).sum().item()
    _check("B_MUL rank-1 (one nonzero)", nonzero == 1)
    _check("B_MUL[DIM_VALUE, DIM_VALUE] == 1",
           float(ff.B_MUL[DIM_VALUE, DIM_VALUE].item()) == 1.0)


def test_symbolic_primitives():
    """symbolic_add/sub/mul are the Poly-ring interpretation of the weights."""
    x0, x1 = Poly.variable(0), Poly.variable(1)
    _check("symbolic_add(x0, x1)", ff.symbolic_add(x0, x1) == (x0 + x1))
    _check("symbolic_sub(x0, x1)", ff.symbolic_sub(x0, x1) == (x1 - x0))
    _check("symbolic_mul(x0, x1)", ff.symbolic_mul(x0, x1) == (x0 * x1))

    # Commutativity of ADD/MUL, non-commutativity of SUB — sanity pins.
    _check("symbolic_add commutative",
           ff.symbolic_add(x0, x1) == ff.symbolic_add(x1, x0))
    _check("symbolic_mul commutative",
           ff.symbolic_mul(x0, x1) == ff.symbolic_mul(x1, x0))
    _check("symbolic_sub non-commutative",
           ff.symbolic_sub(x0, x1) != ff.symbolic_sub(x1, x0))


def test_range_check():
    """Option (a) from issue #69 — explicit i32 range assertion."""
    _check("range_check accepts 0", ff.range_check(0) == 0)
    _check("range_check accepts I32_MAX", ff.range_check(ff.I32_MAX) == ff.I32_MAX)
    _check("range_check accepts I32_MIN", ff.range_check(ff.I32_MIN) == ff.I32_MIN)
    try:
        ff.range_check(ff.I32_MAX + 1)
        _fail("range_check rejects overflow", "did not raise")
    except ff.RangeCheckFailure:
        _pass("range_check rejects overflow")
    try:
        ff.range_check(ff.I32_MIN - 1)
        _fail("range_check rejects underflow", "did not raise")
    except ff.RangeCheckFailure:
        _pass("range_check rejects underflow")


# ─── Equivalence theorem ──────────────────────────────────────────

def _collapsed_entries():
    """Catalog entries currently classified STATUS_COLLAPSED.

    The issue projects "15 currently-collapsed catalog programs"; this
    helper actually asks the catalog at runtime so new collapsed rows
    get picked up automatically.
    """
    out = []
    for entry in _default_catalog():
        cr = classify_program(entry.prog)
        if cr.status == STATUS_COLLAPSED and cr.poly is not None:
            out.append(entry)
    return out


def _guarded_entries():
    """Catalog entries classified STATUS_COLLAPSED_GUARDED.

    Exercises the S3 cross-check (issue #68): bilinear FF primitives
    must produce the same :class:`GuardedPoly` as the symbolic executor
    when JZ/JNZ branches fork on symbolic conditions.
    """
    out = []
    for entry in _default_catalog():
        cr = classify_program(entry.prog)
        if cr.status == STATUS_COLLAPSED_GUARDED and cr.guarded is not None:
            out.append((entry, cr))
    return out


def _unrolled_entries():
    """Catalog entries classified STATUS_COLLAPSED_UNROLLED.

    Exercises the S3 cross-check (issue #68): bilinear FF primitives
    under ``input_mode="concrete"`` must produce the same final Poly
    the symbolic executor produces when bounded loops unroll under
    concrete inputs.
    """
    out = []
    for entry in _default_catalog():
        cr = classify_program(entry.prog)
        if cr.status == STATUS_COLLAPSED_UNROLLED:
            out.append((entry, cr))
    return out


def test_equivalence_structural():
    """Core equivalence: forward_symbolic.top == run_symbolic.top, per catalog row.

    Structural Poly equality — two polynomials match as *expressions*
    (canonical monomial dict), not merely as numbers.
    """
    model = CompiledModel()
    entries = _collapsed_entries()
    _check("catalog has collapsed entries", len(entries) > 0,
           f"expected >0, got {len(entries)}")
    for entry in entries:
        sym_result = run_symbolic(entry.prog)
        fs_result = model.forward_symbolic(entry.prog)
        _check(
            f"equivalence[{entry.name}]",
            sym_result.top == fs_result.top,
            f"run_symbolic.top={sym_result.top!r} vs "
            f"forward_symbolic.top={fs_result.top!r}",
        )
        # n_heads agrees too — both interpreters count the same ops.
        _check(
            f"n_heads[{entry.name}]",
            sym_result.n_heads == fs_result.n_heads,
            f"sym={sym_result.n_heads}, fs={fs_result.n_heads}",
        )


def test_equivalence_numeric():
    """Evaluate the Polys at the catalog's bindings — match NumPyExecutor.

    Belt-and-braces: structural Poly equality is the load-bearing claim,
    but the numerical sanity check catches any Poly-internal regression
    that still preserved the term dict.
    """
    model = CompiledModel()
    np_exec = NumPyExecutor()
    for entry in _collapsed_entries():
        fs = model.forward_symbolic(entry.prog)
        sym_val = fs.top.eval_at(fs.bindings) if fs.bindings else fs.top.eval_at({})
        np_trace = np_exec.execute(entry.prog)
        np_top = np_trace.steps[-1].top if np_trace.steps else None
        # All catalog collapsed values are well within i32.
        try:
            ff.range_check(sym_val, context=entry.name)
            _pass(f"range_check[{entry.name}]")
        except ff.RangeCheckFailure as e:
            _fail(f"range_check[{entry.name}]", str(e))
        _check(
            f"numeric[{entry.name}]",
            sym_val == np_top,
            f"sym_val={sym_val}, np_top={np_top}",
        )


# ─── Guarded / unrolled equivalence (issue #68 S3) ────────────────
#
# Structural: the forking executor with the bilinear FF primitives
# plugged in must produce the same ``top`` (a :class:`Poly` or
# :class:`GuardedPoly`) as the symbolic executor's native ``run_forking``.
# ``symbolic_add/sub/mul`` are defined as the Poly-level interpretation
# of ``M_ADD`` / ``M_SUB`` / ``B_MUL``; S3 extends the equivalence past
# branchless straight-line code into JZ/JNZ control flow and bounded
# loop unrolling.
#
# Numeric: the live case's value polynomial, evaluated at the concrete
# bindings, must equal :class:`NumPyExecutor`'s top. The compiled
# transformer's numeric forward path already routes ADD/SUB/MUL through
# ``forward_add/sub/mul`` (see ``executor.py`` lines 826-833), so this
# check completes the three-way agreement per guarded / unrolled row.


def test_equivalence_guarded_structural():
    """Forking FF == forking symbolic, structurally, on every guarded row."""
    model = CompiledModel()
    entries = _guarded_entries()
    _check("catalog has guarded entries", len(entries) > 0,
           f"expected >0, got {len(entries)}")
    for entry, _cr in entries:
        sym = run_forking(entry.prog, input_mode="symbolic")
        ff_res = model.forward_symbolic_forking(entry.prog, input_mode="symbolic")
        _check(
            f"guarded.status[{entry.name}]",
            sym.status == ff_res.status,
            f"sym={sym.status}, ff={ff_res.status}",
        )
        _check(
            f"guarded.top[{entry.name}]",
            sym.top == ff_res.top,
            f"sym.top={sym.top!r} vs ff.top={ff_res.top!r}",
        )
        _check(
            f"guarded.n_heads[{entry.name}]",
            sym.n_heads == ff_res.n_heads,
            f"sym={sym.n_heads}, ff={ff_res.n_heads}",
        )


def test_equivalence_guarded_numeric():
    """For the live case under the catalog's bindings: sym == np == ff-numeric.

    The FF-wired numeric forward path runs each trace step through the
    bilinear matrices; ``TorchExecutor`` drives that loop. The three-way
    agreement here is the numeric counterpart of the structural claim.
    """
    from executor import TorchExecutor
    model = CompiledModel()
    np_exec = NumPyExecutor()
    torch_exec = TorchExecutor(model)
    for entry, cr in _guarded_entries():
        guarded: GuardedPoly = cr.guarded  # type: ignore[assignment]
        bindings = cr.bindings
        if not bindings:
            continue  # no concrete bindings → nothing to evaluate numerically

        np_trace = np_exec.execute(entry.prog)
        np_top = np_trace.steps[-1].top if np_trace.steps else None
        t_trace = torch_exec.execute(entry.prog)
        t_top = t_trace.steps[-1].top if t_trace.steps else None

        try:
            sym_val = guarded.eval_at(bindings)
        except ValueError as e:
            _fail(f"guarded.numeric[{entry.name}]", f"eval_at: {e}")
            continue
        _check(
            f"guarded.sym==np[{entry.name}]",
            sym_val == np_top,
            f"sym={sym_val}, np={np_top}",
        )
        _check(
            f"guarded.np==ff[{entry.name}]",
            np_top == t_top,
            f"np={np_top}, ff(numeric)={t_top}",
        )


def test_equivalence_unrolled_structural():
    """Forking FF == forking symbolic, structurally, on every unrolled row.

    Uses ``input_mode="concrete"`` — matches how ``classify_program``
    collapses unrolled programs.
    """
    model = CompiledModel()
    entries = _unrolled_entries()
    _check("catalog has unrolled entries", len(entries) > 0,
           f"expected >0, got {len(entries)}")
    for entry, _cr in entries:
        sym = run_forking(entry.prog, input_mode="concrete")
        ff_res = model.forward_symbolic_forking(entry.prog, input_mode="concrete")
        _check(
            f"unrolled.status[{entry.name}]",
            sym.status == ff_res.status,
            f"sym={sym.status}, ff={ff_res.status}",
        )
        _check(
            f"unrolled.top[{entry.name}]",
            sym.top == ff_res.top,
            f"sym.top={sym.top!r} vs ff.top={ff_res.top!r}",
        )
        _check(
            f"unrolled.n_heads[{entry.name}]",
            sym.n_heads == ff_res.n_heads,
            f"sym={sym.n_heads}, ff={ff_res.n_heads}",
        )


def test_equivalence_unrolled_numeric():
    """sym (eval'd at concrete bindings) == np == ff-numeric per unrolled row."""
    from executor import TorchExecutor
    model = CompiledModel()
    np_exec = NumPyExecutor()
    torch_exec = TorchExecutor(model)
    for entry, cr in _unrolled_entries():
        np_trace = np_exec.execute(entry.prog)
        np_top = np_trace.steps[-1].top if np_trace.steps else None
        t_trace = torch_exec.execute(entry.prog)
        t_top = t_trace.steps[-1].top if t_trace.steps else None

        if cr.poly is not None:
            # Concrete-mode polys have no free variables (every PUSH
            # specialised to its literal); eval_at({}) collapses them.
            sym_val = cr.poly.eval_at({})
        elif cr.guarded is not None:
            # Rare: unrolled program still produced guarded output.
            sym_val = cr.guarded.eval_at({}) if not cr.guarded.variables() else None
        else:
            sym_val = None
        if sym_val is not None:
            _check(
                f"unrolled.sym==np[{entry.name}]",
                sym_val == np_top,
                f"sym={sym_val}, np={np_top}",
            )
        _check(
            f"unrolled.np==ff[{entry.name}]",
            np_top == t_top,
            f"np={np_top}, ff(numeric)={t_top}",
        )


def test_dup_add_chain_pin():
    """Issue-#69 pinned example: `dup_add_chain_x4` → 16·x0.

    9 heads collapse to a single monomial; the bilinear form exactly
    reproduces that — both structurally and as a scalar at x0=5.
    """
    prog = program(("PUSH", 5), *([("DUP",), ("ADD",)] * 4), ("HALT",))
    model = CompiledModel()
    fs = model.forward_symbolic(prog)
    expected = Poly({((0, 1),): 16})
    _check("dup_add_chain_x4 top == 16·x0", fs.top == expected,
           f"got {fs.top!r}")
    _check("dup_add_chain_x4 eval", fs.top.eval_at({0: 5}) == 80,
           f"got {fs.top.eval_at({0: 5})}")
    _check("dup_add_chain_x4 n_heads == 9", fs.n_heads == 9,
           f"got {fs.n_heads}")


def test_sum_of_squares_pin():
    """Another pin: ``x0² + x3²`` — degree-2 polynomial, two monomials.

    The MUL here rides on ``B_MUL``; the ADD composes the two products.
    """
    prog = program(
        ("PUSH", 3), ("DUP",), ("MUL",),
        ("PUSH", 4), ("DUP",), ("MUL",),
        ("ADD",), ("HALT",),
    )
    model = CompiledModel()
    fs = model.forward_symbolic(prog)
    # forward_symbolic assigns sequential variable ids per PUSH (matching
    # run_symbolic, not run_forking): two PUSHes → x0 and x1.
    expected = Poly({((0, 2),): 1, ((1, 2),): 1})
    _check("sum_of_squares top", fs.top == expected, f"got {fs.top!r}")
    _check("sum_of_squares eval", fs.top.eval_at({0: 3, 1: 4}) == 25)


# ─── Rational primitives (issue #75) ──────────────────────────────

def test_primitives_div_rem_s():
    """forward_div_s / forward_rem_s agree with Python trunc_div / trunc_rem.

    Covers sign combinations and the zero-divisor check deliberately
    skipped: the numeric path raises via ``_trunc_div`` when the divisor
    is zero, so we only exercise non-zero divisors here.
    """
    samples: List[Tuple[int, int]] = [
        # (a, b) where a is top (va), b is stack[SP-1] (vb); op computes
        # trunc_div(vb, va), trunc_rem(vb, va) — matches WASM i32.div_s.
        (3, 10), (-3, 10), (3, -10), (-3, -10),
        (7, 0), (1, 2**15),
    ]
    for a, b in samples:
        if a == 0:
            continue
        ea, eb = ff.E(a), ff.E(b)
        expected_q = isa._trunc_div(b, a)
        expected_r = isa._trunc_rem(b, a)
        _check(f"forward_div_s({a},{b})",
               ff.E_inv(ff.forward_div_s(ea, eb)) == expected_q,
               f"got {ff.E_inv(ff.forward_div_s(ea, eb))}, expect {expected_q}")
        _check(f"forward_rem_s({a},{b})",
               ff.E_inv(ff.forward_rem_s(ea, eb)) == expected_r,
               f"got {ff.E_inv(ff.forward_rem_s(ea, eb))}, expect {expected_r}")


def test_primitives_rational_matrix_shapes():
    """M_DIV_S / M_REM_S are pair-selectors (shape (2, 2*d_model))."""
    d = ff.D_MODEL
    _check("M_DIV_S shape", ff.M_DIV_S.shape == (2, 2 * d))
    _check("M_REM_S shape", ff.M_REM_S.shape == (2, 2 * d))

    # Row 0 pulls va from ea[DIM_VALUE]; row 1 pulls vb from eb[DIM_VALUE].
    for name, W in (("M_DIV_S", ff.M_DIV_S), ("M_REM_S", ff.M_REM_S)):
        _check(f"{name}[0, DIM_VALUE] == 1",
               float(W[0, DIM_VALUE].item()) == 1.0)
        _check(f"{name}[1, d+DIM_VALUE] == 1",
               float(W[1, d + DIM_VALUE].item()) == 1.0)
        nonzero = (W != 0).sum().item()
        _check(f"{name} has exactly 2 nonzeros", nonzero == 2)


def test_symbolic_rational_primitives():
    """symbolic_div_s / rem_s return RationalPoly / SymbolicRemainder."""
    x0, x1 = Poly.variable(0), Poly.variable(1)
    q = ff.symbolic_div_s(x0, x1)           # pa=x0=top, pb=x1=SP-1 → x1 / x0
    r = ff.symbolic_rem_s(x0, x1)
    _check("symbolic_div_s returns RationalPoly", isinstance(q, RationalPoly))
    _check("symbolic_rem_s returns SymbolicRemainder", isinstance(r, SymbolicRemainder))
    _check("symbolic_div_s num == pb", q.num == x1, f"got {q.num!r}")
    _check("symbolic_div_s denom == pa", q.denom == x0, f"got {q.denom!r}")
    _check("symbolic_rem_s num == pb", r.num == x1)
    _check("symbolic_rem_s denom == pa", r.denom == x0)


def test_equivalence_rational():
    """forward_symbolic on DIV_S / REM_S programs matches run_symbolic structurally.

    Both interpreters carry the (num, denom) pair forward — the
    :class:`RationalPoly` / :class:`SymbolicRemainder` wrappers — and
    evaluating either at the catalog's bindings must match the numeric
    ``NumPyExecutor`` top bit-for-bit.
    """
    model = CompiledModel()
    np_exec = NumPyExecutor()

    cases = [
        # (name, prog) — mirrors catalog entries that now collapse. Inputs
        # are kept non-negative so the ℤ-valued rational eval agrees with
        # the masked u32 :class:`NumPyExecutor` top (parity at the boundary
        # per the Option (a) range assumption from issue #69).
        ("native_divmod(2,7)",
         program(("PUSH", 7), ("PUSH", 2), ("DIV_S",), ("HALT",))),
        ("native_remainder(2,7)",
         program(("PUSH", 7), ("PUSH", 2), ("REM_S",), ("HALT",))),
        ("native_divmod(3,10)",
         program(("PUSH", 10), ("PUSH", 3), ("DIV_S",), ("HALT",))),
        ("native_remainder(3,10)",
         program(("PUSH", 10), ("PUSH", 3), ("REM_S",), ("HALT",))),
    ]
    for name, prog_ in cases:
        sym_result = run_symbolic(prog_)
        fs_result = model.forward_symbolic(prog_)
        _check(
            f"rational.struct[{name}]",
            type(sym_result.top) is type(fs_result.top),
            f"sym={type(sym_result.top).__name__}, ff={type(fs_result.top).__name__}",
        )
        _check(
            f"rational.num[{name}]",
            sym_result.top.num == fs_result.top.num,
            f"sym.num={sym_result.top.num!r} vs ff.num={fs_result.top.num!r}",
        )
        _check(
            f"rational.denom[{name}]",
            sym_result.top.denom == fs_result.top.denom,
            f"sym.denom={sym_result.top.denom!r} vs ff.denom={fs_result.top.denom!r}",
        )
        # Three-way numeric match: sym eval == ff eval == NumPy.
        sym_val = sym_result.top.eval_at(sym_result.bindings)
        ff_val = fs_result.top.eval_at(fs_result.bindings)
        np_trace = np_exec.execute(prog_)
        np_top = np_trace.steps[-1].top
        _check(
            f"rational.numeric[{name}]",
            sym_val == ff_val == np_top,
            f"sym={sym_val}, ff={ff_val}, np={np_top}",
        )


def test_rational_catalog_rows_classify_as_collapsed():
    """native_divmod / native_remainder catalog entries collapse (#75)."""
    names_of_interest = {"native_divmod(2,7)", "native_remainder(2,7)"}
    for entry in _default_catalog():
        if entry.name not in names_of_interest:
            continue
        cr = classify_program(entry.prog)
        _check(
            f"catalog.status[{entry.name}] == collapsed",
            cr.status == STATUS_COLLAPSED,
            f"got {cr.status} (blocker={cr.blocker})",
        )
        _check(
            f"catalog.rational[{entry.name}] is set",
            cr.rational is not None,
            f"got {cr.rational!r}",
        )


# ─── Comparison primitives (issue #76) ────────────────────────────

def test_primitives_forward_cmp():
    """forward_cmp(ea, eb, op) returns E(0/1) matching WASM semantics.

    Convention: ``ea`` is va (top of stack), ``eb`` is vb (SP-1). The
    WASM ``LT_S`` opcode computes ``1 if vb < va``, so with va=5, vb=3
    → ``3 < 5`` → 1.
    """
    cases = [
        # (op, va, vb, expected 0/1)
        (isa.OP_LT_S, 5, 3, 1),  # 3 < 5
        (isa.OP_LT_S, 3, 5, 0),
        (isa.OP_LT_S, 4, 4, 0),  # not strict
        (isa.OP_GT_S, 3, 5, 1),  # 5 > 3
        (isa.OP_GT_S, 5, 3, 0),
        (isa.OP_LE_S, 4, 4, 1),  # non-strict
        (isa.OP_LE_S, 3, 4, 0),  # 4 <= 3 → False
        (isa.OP_GE_S, 4, 4, 1),
        (isa.OP_EQ,   7, 7, 1),
        (isa.OP_EQ,   7, 3, 0),
        (isa.OP_NE,   7, 3, 1),
        (isa.OP_NE,   4, 4, 0),
    ]
    for op, va, vb, expected in cases:
        ea, eb = ff.E(va), ff.E(vb)
        got = ff.E_inv(ff.forward_cmp(ea, eb, op))
        _check(
            f"forward_cmp[{isa.OP_NAMES[op]}](va={va},vb={vb})",
            got == expected,
            f"got {got}, expect {expected}",
        )


def test_primitives_forward_eqz():
    """forward_eqz(ea) returns E(1) iff va == 0, else E(0)."""
    for v, expected in [(0, 1), (1, 0), (-5, 0), (100, 0)]:
        got = ff.E_inv(ff.forward_eqz(ff.E(v)))
        _check(f"forward_eqz({v})", got == expected, f"got {got}, expect {expected}")


def test_primitives_cmp_matrix_shapes():
    """M_CMP extracts diff (vb - va) via a 1×(2d) linear form; M_EQZ
    extracts va via a 1×d linear form. Both carry the non-polynomial
    gate at the dispatch boundary, not in their weight tensor."""
    d = ff.D_MODEL
    _check("M_CMP shape", ff.M_CMP.shape == (1, 2 * d))
    _check("M_EQZ shape", ff.M_EQZ.shape == (1, d))

    # M_CMP: ea contributes -1 at DIM_VALUE, eb contributes +1 at d+DIM_VALUE.
    _check("M_CMP[0, DIM_VALUE] == -1",
           float(ff.M_CMP[0, DIM_VALUE].item()) == -1.0)
    _check("M_CMP[0, d+DIM_VALUE] == +1",
           float(ff.M_CMP[0, d + DIM_VALUE].item()) == 1.0)
    nonzero_cmp = (ff.M_CMP != 0).sum().item()
    _check("M_CMP has exactly 2 nonzeros", nonzero_cmp == 2)

    # M_EQZ: single +1 at DIM_VALUE.
    _check("M_EQZ[0, DIM_VALUE] == 1",
           float(ff.M_EQZ[0, DIM_VALUE].item()) == 1.0)
    nonzero_eqz = (ff.M_EQZ != 0).sum().item()
    _check("M_EQZ has exactly 1 nonzero", nonzero_eqz == 1)


def test_symbolic_comparison_primitives():
    """symbolic_cmp / symbolic_eqz return IndicatorPoly with the right
    diff polynomial and relation tag."""
    x0, x1 = Poly.variable(0), Poly.variable(1)

    pairs = [
        (isa.OP_LT_S, REL_LT),
        (isa.OP_GT_S, REL_GT),
        (isa.OP_LE_S, REL_LE),
        (isa.OP_GE_S, REL_GE),
        (isa.OP_EQ,   REL_EQ),
        (isa.OP_NE,   REL_NE),
    ]
    for op, expected_rel in pairs:
        # symbolic_cmp(pa, pb) — pa=top=x0, pb=SP-1=x1 → diff = pb - pa = x1 - x0.
        ind = ff.symbolic_cmp(x0, x1, op)
        _check(f"symbolic_cmp[{isa.OP_NAMES[op]}] type",
               isinstance(ind, IndicatorPoly))
        _check(f"symbolic_cmp[{isa.OP_NAMES[op]}] relation",
               ind.relation == expected_rel,
               f"got {ind.relation}, expect {expected_rel}")
        _check(f"symbolic_cmp[{isa.OP_NAMES[op]}] poly",
               ind.poly == (x1 - x0), f"got {ind.poly!r}")

    ind_eqz = ff.symbolic_eqz(x0)
    _check("symbolic_eqz type", isinstance(ind_eqz, IndicatorPoly))
    _check("symbolic_eqz relation", ind_eqz.relation == REL_EQ)
    _check("symbolic_eqz poly == x0", ind_eqz.poly == x0)


def test_equivalence_comparison_structural():
    """forward_symbolic on comparison programs produces the same
    IndicatorPoly the symbolic executor emits. Structural equality on
    both the underlying poly and the relation tag."""
    model = CompiledModel()
    ops = [isa.OP_LT_S, isa.OP_GT_S, isa.OP_LE_S, isa.OP_GE_S,
           isa.OP_EQ, isa.OP_NE]
    for op in ops:
        prog = program(("PUSH", 3), ("PUSH", 5),
                       (isa.OP_NAMES[op],), ("HALT",))
        sym = run_symbolic(prog)
        fs = model.forward_symbolic(prog)
        _check(
            f"cmp.struct.type[{isa.OP_NAMES[op]}]",
            isinstance(sym.top, IndicatorPoly)
            and isinstance(fs.top, IndicatorPoly),
            f"sym={type(sym.top).__name__}, ff={type(fs.top).__name__}",
        )
        _check(
            f"cmp.struct.poly[{isa.OP_NAMES[op]}]",
            sym.top.poly == fs.top.poly,
            f"sym.poly={sym.top.poly!r} ff.poly={fs.top.poly!r}",
        )
        _check(
            f"cmp.struct.rel[{isa.OP_NAMES[op]}]",
            sym.top.relation == fs.top.relation,
            f"sym.rel={sym.top.relation} ff.rel={fs.top.relation}",
        )


def test_equivalence_eqz_structural():
    """forward_symbolic on EQZ matches run_symbolic.top structurally."""
    model = CompiledModel()
    prog = program(("PUSH", 0), ("EQZ",), ("HALT",))
    sym = run_symbolic(prog)
    fs = model.forward_symbolic(prog)
    _check("eqz.struct.type",
           isinstance(sym.top, IndicatorPoly) and isinstance(fs.top, IndicatorPoly))
    _check("eqz.struct.poly", sym.top.poly == fs.top.poly)
    _check("eqz.struct.rel", sym.top.relation == fs.top.relation == REL_EQ)


def test_equivalence_comparison_numeric():
    """Three-way check: sym.eval_at == NumPy top == TorchExecutor top.

    Covers the WASM convention (``LT_S`` returns ``1 if vb < va``) across
    every signed comparison plus EQZ, on inputs spanning the comparison's
    {<, =, >} (or {=0, ≠0}) regions.
    """
    from executor import TorchExecutor
    model = CompiledModel()
    np_exec = NumPyExecutor()
    torch_exec = TorchExecutor(model)

    binary_cases = [
        (isa.OP_LT_S, (3, 5)), (isa.OP_LT_S, (5, 3)), (isa.OP_LT_S, (4, 4)),
        (isa.OP_GT_S, (3, 5)), (isa.OP_LE_S, (4, 4)), (isa.OP_GE_S, (7, 2)),
        (isa.OP_EQ,   (7, 7)), (isa.OP_NE,   (3, 5)),
    ]
    for op, (a, b) in binary_cases:
        prog = [
            isa.Instruction(isa.OP_PUSH, a),
            isa.Instruction(isa.OP_PUSH, b),
            isa.Instruction(op),
            isa.Instruction(isa.OP_HALT),
        ]
        sym = run_symbolic(prog)
        sym_val = sym.top.eval_at(sym.bindings)
        np_top = np_exec.execute(prog).steps[-1].top
        t_top = torch_exec.execute(prog).steps[-1].top
        _check(
            f"cmp.numeric[{isa.OP_NAMES[op]}({a},{b})]",
            sym_val == np_top == t_top,
            f"sym={sym_val} np={np_top} torch={t_top}",
        )

    for a in [-3, -1, 0, 1, 5]:
        prog = [
            isa.Instruction(isa.OP_PUSH, a),
            isa.Instruction(isa.OP_EQZ),
            isa.Instruction(isa.OP_HALT),
        ]
        sym = run_symbolic(prog)
        sym_val = sym.top.eval_at(sym.bindings)
        np_top = np_exec.execute(prog).steps[-1].top
        t_top = torch_exec.execute(prog).steps[-1].top
        _check(
            f"eqz.numeric({a})",
            sym_val == np_top == t_top,
            f"sym={sym_val} np={np_top} torch={t_top}",
        )


def test_native_max_three_way_numeric():
    """native_max demonstrates the full gated path end-to-end: GT_S →
    IndicatorPoly → JZ hoists relation into Guard pair → GuardedPoly
    dispatch. Symbolic eval, NumPy exec, and TorchExecutor must all agree
    on the computed max at the catalog's concrete inputs."""
    from executor import TorchExecutor
    import programs as P
    model = CompiledModel()
    np_exec = NumPyExecutor()
    torch_exec = TorchExecutor(model)

    for a, b in [(3, 5), (7, 2), (4, 4)]:
        prog, expected = P.make_native_max(a, b)
        sym = run_forking(prog, input_mode="symbolic")
        assert sym.top is not None
        sym_val = sym.top.eval_at(sym.bindings)
        np_top = np_exec.execute(prog).steps[-1].top
        t_top = torch_exec.execute(prog).steps[-1].top
        _check(
            f"native_max.numeric({a},{b})",
            sym_val == np_top == t_top == expected,
            f"sym={sym_val} np={np_top} torch={t_top} expected={expected}",
        )


# ─── Bit-vector primitives (issue #77) ────────────────────────────

def test_primitives_forward_bit_binary():
    """forward_bit_binary(ea, eb, op) returns E(bit_binary(vb, va)).

    Convention: ``ea`` is va (top of stack), ``eb`` is vb (SP-1). WASM
    ``SHL(value, count) = value << count`` with value=SP-1, count=top, so
    ``forward_bit_binary(E(count), E(value), SHL)`` matches
    ``_apply_bitop("SHL", [value, count])``.
    """
    cases = [
        # (op, va=top, vb=SP-1, expected)
        (isa.OP_AND,   10, 12, 12 & 10),
        (isa.OP_OR,    10, 12, 12 | 10),
        (isa.OP_XOR,   10, 12, 12 ^ 10),
        (isa.OP_SHL,   2,  3,  (3 << 2) & 0xFFFFFFFF),
        (isa.OP_SHR_U, 1,  0xFFFFFFFF, 0x7FFFFFFF),
        (isa.OP_SHR_S, 1,  -4 & 0xFFFFFFFF, (-2) & 0xFFFFFFFF),
    ]
    for op, va, vb, expected in cases:
        ea, eb = ff.E(va), ff.E(vb)
        got = ff.E_inv(ff.forward_bit_binary(ea, eb, op))
        _check(
            f"forward_bit_binary[{isa.OP_NAMES[op]}](va={va},vb={vb})",
            got == expected,
            f"got {got}, expect {expected}",
        )


def test_primitives_forward_bit_unary():
    """forward_bit_unary(ea, op) returns E(bit_unary(va)) for CLZ/CTZ/POPCNT."""
    cases = [
        (isa.OP_CLZ,    16, 27),   # 0x00000010
        (isa.OP_CLZ,    0,  32),
        (isa.OP_CLZ,    -1, 0),    # 0xFFFFFFFF
        (isa.OP_CTZ,    8,  3),    # 0b1000
        (isa.OP_CTZ,    0,  32),
        (isa.OP_POPCNT, 13, 3),    # 0b1101
        (isa.OP_POPCNT, 0,  0),
        (isa.OP_POPCNT, -1, 32),
    ]
    for op, n, expected in cases:
        got = ff.E_inv(ff.forward_bit_unary(ff.E(n), op))
        _check(
            f"forward_bit_unary[{isa.OP_NAMES[op]}]({n})",
            got == expected,
            f"got {got}, expect {expected}",
        )


def test_primitives_bitvec_matrix_shapes():
    """``M_BITBIN`` is a (2, 2*d) pair-selector; ``M_BITUN`` is a
    (1, d) single-value extractor. Both are pure weight tensors — the
    non-polynomial bit op lives at the boundary (in ``_apply_bitop``),
    not inside these matrices."""
    d = ff.D_MODEL
    _check("M_BITBIN shape", ff.M_BITBIN.shape == (2, 2 * d))
    _check("M_BITUN shape",  ff.M_BITUN.shape == (1, d))

    # M_BITBIN: row 0 picks va (top) from ea @ DIM_VALUE; row 1 picks vb
    # (SP-1) from eb @ d + DIM_VALUE. Exactly 2 nonzeros total.
    _check("M_BITBIN[0, DIM_VALUE] == 1",
           float(ff.M_BITBIN[0, DIM_VALUE].item()) == 1.0)
    _check("M_BITBIN[1, d+DIM_VALUE] == 1",
           float(ff.M_BITBIN[1, d + DIM_VALUE].item()) == 1.0)
    nonzero_bin = (ff.M_BITBIN != 0).sum().item()
    _check("M_BITBIN has exactly 2 nonzeros", nonzero_bin == 2)

    # M_BITUN: single +1 at DIM_VALUE.
    _check("M_BITUN[0, DIM_VALUE] == 1",
           float(ff.M_BITUN[0, DIM_VALUE].item()) == 1.0)
    nonzero_un = (ff.M_BITUN != 0).sum().item()
    _check("M_BITUN has exactly 1 nonzero", nonzero_un == 1)


def test_symbolic_bitvec_primitives():
    """symbolic_bit_binary / symbolic_bit_unary return BitVec nodes with
    the right op name and operands in natural left-right reading order
    (left=SP-1, right=top)."""
    x0, x1 = Poly.variable(0), Poly.variable(1)

    for op, name in [
        (isa.OP_AND, "AND"), (isa.OP_OR, "OR"), (isa.OP_XOR, "XOR"),
        (isa.OP_SHL, "SHL"), (isa.OP_SHR_S, "SHR_S"), (isa.OP_SHR_U, "SHR_U"),
    ]:
        # pa=top=x0, pb=SP-1=x1 → operands (pb, pa) = (x1, x0).
        bv = ff.symbolic_bit_binary(x0, x1, op)
        _check(f"symbolic_bit_binary[{name}] type", isinstance(bv, BitVec))
        _check(f"symbolic_bit_binary[{name}] op", bv.op == name)
        _check(f"symbolic_bit_binary[{name}] operands[0]==x1 (SP-1)",
               bv.operands[0] == x1)
        _check(f"symbolic_bit_binary[{name}] operands[1]==x0 (top)",
               bv.operands[1] == x0)

    for op, name in [
        (isa.OP_CLZ, "CLZ"), (isa.OP_CTZ, "CTZ"), (isa.OP_POPCNT, "POPCNT"),
    ]:
        bv = ff.symbolic_bit_unary(x0, op)
        _check(f"symbolic_bit_unary[{name}] type", isinstance(bv, BitVec))
        _check(f"symbolic_bit_unary[{name}] op", bv.op == name)
        _check(f"symbolic_bit_unary[{name}] operands==(x0,)",
               bv.operands == (x0,))


def test_equivalence_bitvec_structural():
    """forward_symbolic on each bit-vector program produces the same
    BitVec AST the symbolic executor emits. Structural equality on
    op name + operand tuples (value-based)."""
    model = CompiledModel()
    progs = [
        ("AND",     program(("PUSH", 12), ("PUSH", 10), ("AND",),    ("HALT",))),
        ("OR",      program(("PUSH", 12), ("PUSH", 10), ("OR",),     ("HALT",))),
        ("XOR",     program(("PUSH", 12), ("PUSH", 10), ("XOR",),    ("HALT",))),
        ("SHL",     program(("PUSH", 3),  ("PUSH", 2),  ("SHL",),    ("HALT",))),
        ("SHR_S",   program(("PUSH", -4), ("PUSH", 1),  ("SHR_S",),  ("HALT",))),
        ("SHR_U",   program(("PUSH", -1), ("PUSH", 4),  ("SHR_U",),  ("HALT",))),
        ("CLZ",     program(("PUSH", 16), ("CLZ",),     ("HALT",))),
        ("CTZ",     program(("PUSH", 8),  ("CTZ",),     ("HALT",))),
        ("POPCNT",  program(("PUSH", 13), ("POPCNT",),  ("HALT",))),
    ]
    for name, prog in progs:
        sym = run_symbolic(prog)
        fs = model.forward_symbolic(prog)
        _check(
            f"bitvec.struct.type[{name}]",
            isinstance(sym.top, BitVec) and isinstance(fs.top, BitVec),
            f"sym={type(sym.top).__name__}, ff={type(fs.top).__name__}",
        )
        _check(
            f"bitvec.struct.eq[{name}]",
            sym.top == fs.top,
            f"sym={sym.top!r}  ff={fs.top!r}",
        )


def test_equivalence_bitvec_nested_structural():
    """Nested bit programs (bit_extract = ``AND(1, SHR_U(n, k))``) and
    hybrid arithmetic (log2_floor = ``SUB(31, CLZ(n))``) must produce
    identical BitVec ASTs from both executors."""
    import programs as P
    model = CompiledModel()

    for name, (prog, _expected) in [
        ("bit_extract(5,0)", P.make_bit_extract(5, 0)),
        ("log2_floor(8)",    P.make_log2_floor(8)),
    ]:
        sym = run_symbolic(prog)
        fs = model.forward_symbolic(prog)
        _check(f"bitvec.nested.type[{name}]",
               isinstance(sym.top, BitVec) and isinstance(fs.top, BitVec))
        _check(f"bitvec.nested.eq[{name}]", sym.top == fs.top,
               f"sym={sym.top!r}  ff={fs.top!r}")


def test_equivalence_is_power_of_2_structural():
    """is_power_of_2's IndicatorPoly wraps a BitVec("SUB", ...) diff on
    both paths — sym and FF must produce the same indicator (widened
    IndicatorPoly.poly accepts BitVec per issue #77)."""
    import programs as P
    model = CompiledModel()
    prog, _expected = P.make_is_power_of_2(8)
    sym = run_symbolic(prog)
    fs = model.forward_symbolic(prog)
    _check("is_power_of_2.struct.type",
           isinstance(sym.top, IndicatorPoly) and isinstance(fs.top, IndicatorPoly))
    _check("is_power_of_2.struct.poly", sym.top.poly == fs.top.poly,
           f"sym={sym.top.poly!r}  ff={fs.top.poly!r}")
    _check("is_power_of_2.struct.rel", sym.top.relation == fs.top.relation)


def test_equivalence_bitvec_numeric():
    """Three-way check: ``sym.top.eval_at == NumPy top == TorchExecutor top``
    for every bit-vector row. Ensures M_BITBIN / M_BITUN realise the same
    ops the symbolic AST does."""
    from executor import TorchExecutor
    import programs as P
    model = CompiledModel()
    np_exec = NumPyExecutor()
    torch_exec = TorchExecutor(model)

    cases = [
        ("bitwise_and(12,10)",  P.make_bitwise_binary(isa.OP_AND, 12, 10)),
        ("bitwise_or(12,10)",   P.make_bitwise_binary(isa.OP_OR, 12, 10)),
        ("bitwise_xor(12,10)",  P.make_bitwise_binary(isa.OP_XOR, 12, 10)),
        ("bitwise_shl(3,2)",    P.make_bitwise_binary(isa.OP_SHL, 3, 2)),
        ("bitwise_shr_u(-1,4)", P.make_bitwise_binary(isa.OP_SHR_U, -1, 4)),
        ("native_clz(16)",      P.make_native_clz(16)),
        ("native_ctz(8)",       P.make_native_ctz(8)),
        ("native_popcnt(13)",   P.make_native_popcnt(13)),
        ("bit_extract(5,0)",    P.make_bit_extract(5, 0)),
        ("log2_floor(8)",       P.make_log2_floor(8)),
        ("is_power_of_2(8)",    P.make_is_power_of_2(8)),
    ]
    for name, (prog, expected) in cases:
        sym = run_symbolic(prog)
        try:
            sym_val = sym.top.eval_at(sym.bindings)
        except Exception as e:
            _fail(f"bitvec.numeric[{name}]",
                  f"sym eval raised: {type(e).__name__}: {e}")
            continue
        np_top = np_exec.execute(prog).steps[-1].top
        t_top = torch_exec.execute(prog).steps[-1].top
        _check(
            f"bitvec.numeric[{name}]",
            sym_val == np_top == t_top == expected,
            f"sym={sym_val} np={np_top} torch={t_top} expected={expected}",
        )


def test_bitvec_parameter_count():
    """Issue #77 adds 3 non-zero weights to the FF layer:
    M_BITBIN's 2-entry pair selector + M_BITUN's 1-entry extractor.
    ``ff.n_parameters()`` must reflect this so the budget tracking
    stays honest."""
    n = ff.n_parameters()
    # M_ADD (2) + M_SUB (2) + B_MUL (1) + M_DIV_S (2) + M_REM_S (2)
    # + M_CMP (2) + M_EQZ (1) + M_BITBIN (2) + M_BITUN (1) = 15
    _check("n_parameters == 15", n == 15, f"got {n}")


# ─── Blocked-opcode handling ──────────────────────────────────────

# ─── ModPoly / mod-2³² equivalence (issue #78 option (b)) ────────
#
# Issue #78 asks for the equivalence theorem to be "honest about i32
# wrap" — either pin the range (#69 Option (a), already landed) or
# carry wrap semantics in the algebra (Option (b)). These tests
# exercise Option (b): ``ModPoly`` closes the ``& MASK32`` gap by
# reducing coefficients modulo 2³² after every ADD/SUB/MUL, so the
# equivalence becomes *structural over ℤ/2³²* — not just numeric on
# in-range inputs.
#
# Coverage:
#   1. Primitives on ``ModPoly`` + mod-32 symbolic primitives wrap
#      correctly at the 32-bit boundary.
#   2. Every pure-Poly collapsed catalog row has
#      ``ModPoly.from_poly(run_symbolic(P).top) == evaluate_program_mod(P).top``.
#   3. Two overflow-stressing programs where the ℤ result doesn't fit
#      in i32 still match NumPyExecutor bit-for-bit under wrap.
#   4. Ring-homomorphism check: the lift of the Poly-over-ℤ driver's
#      output equals the mod-32 driver's output on every catalog row.
#   5. Blocked opcodes outside the polynomial fragment (DIV_S, JZ,
#      comparisons, bitwise) raise ``BlockedOpcodeForSymbolic`` from
#      ``evaluate_program_mod`` rather than silently returning a
#      ModPoly that doesn't carry the wrap semantics of the boundary
#      wrappers.


def test_modpoly_primitives():
    """ModPoly constructors, arithmetic, wrap behaviour."""
    # Constants wrap cleanly.
    _check("ModPoly.constant(0) == empty", ModPoly.constant(0) == ModPoly({}))
    _check("ModPoly.constant(2**32) == 0",
           ModPoly.constant(1 << 32) == ModPoly.constant(0))
    _check("ModPoly.constant(-1) == MASK32",
           ModPoly.constant(-1) == ModPoly.constant(0xFFFFFFFF))

    # Variables are the same shape as Poly's.
    x0 = ModPoly.variable(0)
    x1 = ModPoly.variable(1)
    _check("variable structure",
           x0.terms == {((0, 1),): 1})

    # Addition wraps at the coefficient level.
    big = ModPoly.constant(0xFFFFFFFF)
    one = ModPoly.constant(1)
    _check("MASK32 + 1 wraps to 0", (big + one) == ModPoly.constant(0))

    # Subtraction produces a canonical unsigned form.
    _check("0 - 1 == MASK32",
           (ModPoly.constant(0) - ModPoly.constant(1))
           == ModPoly.constant(0xFFFFFFFF))

    # Negation.
    _check("neg(1) == MASK32",
           (-ModPoly.constant(1)) == ModPoly.constant(0xFFFFFFFF))

    # Multiplication wraps. 100_000 * 100_000 = 10**10 ≡ 1_410_065_408 (mod 2³²).
    hundred_k = ModPoly.constant(100_000)
    _check("100k * 100k wraps",
           (hundred_k * hundred_k) == ModPoly.constant(1_410_065_408))

    # 2¹⁶ squared is 2³², which is 0 mod 2³². Structural witness of the
    # zero-divisor behaviour ℤ/2³² exhibits.
    sixteen_bit = ModPoly.constant(1 << 16)
    _check("2^16 * 2^16 == 0 mod 2^32",
           (sixteen_bit * sixteen_bit) == ModPoly.constant(0))

    # eval_at matches a native & MASK32 on an out-of-range value.
    sq = ModPoly.variable(0) * ModPoly.variable(0)
    _check("eval_at wraps 46341^2",
           sq.eval_at({0: 46341}) == (46341 * 46341) & 0xFFFFFFFF)
    _check("eval_at_signed on overflow",
           sq.eval_at_signed({0: 46341}) == -2_147_479_015)
    _check("eval_at handles negative binding",
           sq.eval_at({0: -1}) == 1)


def test_symbolic_primitives_mod():
    """ff.symbolic_{add,sub,mul}_mod = ModPoly +/−/*, order matches numeric."""
    a = ModPoly.variable(0)
    b = ModPoly.variable(1)
    _check("symbolic_add_mod", ff.symbolic_add_mod(a, b) == a + b)
    # SUB order: forward_sub returns b - a (pa = top = a; pb = SP-1 = b).
    _check("symbolic_sub_mod order", ff.symbolic_sub_mod(a, b) == b - a)
    _check("symbolic_mul_mod", ff.symbolic_mul_mod(a, b) == a * b)

    # Composition reduces at every step — the homomorphism property.
    # Building (x+y)² symbolically with constants near the boundary:
    # coefficients stay within [0, 2³²) after every intermediate step.
    k = ModPoly.constant(0xFFFF_FFFE)  # -2 (mod 2³²)
    _check("compose: k + 2 == 0 mod 2^32",
           (ff.symbolic_add_mod(k, ModPoly.constant(2))) == ModPoly.constant(0))


def _collapsed_poly_entries():
    """Catalog rows whose ``run_symbolic.top`` is a plain ``Poly``.

    That's the subset where ModPoly's scope applies directly. Other
    collapsed rows (RationalPoly / IndicatorPoly / BitVec) have their
    own boundary wrap handled inside those wrappers' ``eval_at`` and
    are out of scope for the mod-2³² algebra.
    """
    out = []
    for entry in _default_catalog():
        cr = classify_program(entry.prog)
        if cr.status == STATUS_COLLAPSED and cr.poly is not None:
            out.append(entry)
    return out


def test_modpoly_catalog_structural():
    """For every pure-Poly collapsed row:

    ``ModPoly.from_poly(run_symbolic(P).top) == evaluate_program_mod(P).top``.

    Structural equality in ℤ/2³², not merely numerical — the canonical
    term dict matches after mod-32 reduction. This is the honest
    equivalence theorem option (b) asks for.
    """
    entries = _collapsed_poly_entries()
    _check("mod32 catalog has rows", len(entries) > 0,
           f"expected >0, got {len(entries)}")
    for entry in entries:
        sym = run_symbolic(entry.prog)
        if not isinstance(sym.top, Poly):
            continue
        lifted = ModPoly.from_poly(sym.top)
        mod_res = ff.evaluate_program_mod(entry.prog)
        _check(
            f"mod32.struct[{entry.name}]",
            lifted == mod_res.top,
            f"lifted={lifted!r} mod={mod_res.top!r}",
        )
        _check(
            f"mod32.n_heads[{entry.name}]",
            sym.n_heads == mod_res.n_heads,
            f"sym={sym.n_heads}, mod={mod_res.n_heads}",
        )


def test_modpoly_homomorphism_on_catalog():
    """Ring-homomorphism witness on every collapsed row.

    ``ModPoly.from_poly(evaluate_program(P).top) == evaluate_program_mod(P).top``
    — lifting commutes with the driver. The one-liner proof is that
    each primitive (``symbolic_{add,sub,mul}`` and ``_mod`` variants)
    reduces to the same coefficient-level operation, just in a
    different ring. The test asserts it actually holds programmatically.
    """
    for entry in _collapsed_poly_entries():
        poly_res = ff.evaluate_program(entry.prog)
        if not isinstance(poly_res.top, Poly):
            continue
        lifted = ModPoly.from_poly(poly_res.top)
        mod_res = ff.evaluate_program_mod(entry.prog)
        _check(
            f"mod32.homo[{entry.name}]",
            lifted == mod_res.top,
            f"lifted={lifted!r} mod={mod_res.top!r}",
        )


def test_modpoly_catalog_numeric():
    """Evaluate the mod-32 top at catalog bindings — match NumPyExecutor.

    Unlike ``test_equivalence_numeric`` (which asserts over ℤ and
    requires the value to clear ``range_check``), this evaluates in
    ℤ/2³² directly — the sides agree by construction, range_check
    notwithstanding.
    """
    np_exec = NumPyExecutor()
    for entry in _collapsed_poly_entries():
        mod = ff.evaluate_program_mod(entry.prog)
        mod_val = mod.top.eval_at(mod.bindings) if mod.bindings else mod.top.eval_at({})
        np_trace = np_exec.execute(entry.prog)
        np_top = np_trace.steps[-1].top if np_trace.steps else None
        # NumPyExecutor stores u32-masked values; ModPoly.eval_at returns u32.
        _check(
            f"mod32.numeric[{entry.name}]",
            mod_val == (int(np_top) & 0xFFFFFFFF),
            f"mod={mod_val}, np={np_top}",
        )


def test_modpoly_overflow_stress():
    """Two overflow-stressing cases where ℤ and ℤ/2³² disagree.

    Issue #78 explicitly calls these out as the missing coverage under
    Option (a): "Today's tests live inside the range where both sides
    agree, so a bug in either direction would pass silently." Each
    program here has an honest value > I32_MAX that only agrees with
    the native path under i32 wrap.

    Three-way check per case: run_symbolic over ℤ (out of range; fails
    range_check), evaluate_program_mod (in ℤ/2³²), NumPyExecutor
    (masked). ModPoly ≡ NumPy bit-for-bit; ℤ-evaluated Poly disagrees.
    """
    np_exec = NumPyExecutor()

    # Case 1: MUL on sqrt(I32_MAX)-ish inputs. 46341² = 2_147_488_281
    # overflows signed i32 (I32_MAX = 2_147_483_647) by 4_634. Stays in
    # u32, so NumPyExecutor's masked u32 view matches the unmasked ℤ
    # value here — but the *signed* reading differs, and the structure
    # of the test is to show ModPoly is the one that agrees with
    # NumPyExecutor under ALL i32 inputs, not just in-range ones.
    prog1 = program(
        ("PUSH", 46341), ("DUP",), ("MUL",), ("HALT",),
    )
    sym1 = run_symbolic(prog1)
    mod1 = ff.evaluate_program_mod(prog1)
    np1 = np_exec.execute(prog1).steps[-1].top
    mod_val1 = mod1.top.eval_at(mod1.bindings)
    z_val1 = sym1.top.eval_at(sym1.bindings)
    _check(
        "overflow[46341^2].modpoly==numpy",
        mod_val1 == (int(np1) & 0xFFFFFFFF),
        f"mod={mod_val1}, np={np1}",
    )
    _check(
        "overflow[46341^2].z_matches_too",
        z_val1 == 46341 * 46341,
        f"z={z_val1}",
    )
    # The diagnostic: ℤ value is still equal to masked u32 here (2_147_488_281
    # < 2³² = 4_294_967_296), so both interpretations happen to agree.
    # range_check rejects it though.
    try:
        ff.range_check(z_val1, context="46341^2")
        _fail("overflow[46341^2].range_rejects", "range_check accepted overflow")
    except ff.RangeCheckFailure:
        _pass("overflow[46341^2].range_rejects")

    # Case 2: MUL beyond u32. 100_000 * 100_000 = 10¹⁰ > 2³² (= 4_294_967_296).
    # ℤ-evaluated Poly = 10_000_000_000; ℤ/2³² = 1_410_065_408 = NumPy's
    # masked top. ModPoly is the only side that agrees with NumPyExecutor.
    prog2 = program(
        ("PUSH", 100_000), ("PUSH", 100_000), ("MUL",), ("HALT",),
    )
    sym2 = run_symbolic(prog2)
    mod2 = ff.evaluate_program_mod(prog2)
    np2 = np_exec.execute(prog2).steps[-1].top
    mod_val2 = mod2.top.eval_at(mod2.bindings)
    z_val2 = sym2.top.eval_at(sym2.bindings)
    _check(
        "overflow[100k*100k].modpoly==numpy",
        mod_val2 == (int(np2) & 0xFFFFFFFF),
        f"mod={mod_val2}, np={np2}",
    )
    _check(
        "overflow[100k*100k].z_disagrees",
        z_val2 != (int(np2) & 0xFFFFFFFF) and z_val2 == 10_000_000_000,
        f"z={z_val2}, np={np2}",
    )

    # Case 3 (bonus): 2¹⁶ squared = 2³² ≡ 0 (mod 2³²). Zero-divisor
    # witness inside ℤ/2³²; NumPyExecutor also reports 0.
    prog3 = program(
        ("PUSH", 1 << 16), ("DUP",), ("MUL",), ("HALT",),
    )
    mod3 = ff.evaluate_program_mod(prog3)
    np3 = np_exec.execute(prog3).steps[-1].top
    _check(
        "overflow[2^16 squared == 0]",
        mod3.top.eval_at(mod3.bindings) == 0 and (int(np3) & 0xFFFFFFFF) == 0,
        f"mod={mod3.top.eval_at(mod3.bindings)}, np={np3}",
    )

    # Case 4 (bonus): ADD that overflows. I32_MAX + 2 = -I32_MAX (signed wrap).
    # We do it via PUSHes that themselves exceed i31; evaluate_program_mod
    # still produces x0 + x1 symbolically (coefficients unchanged) — the
    # wrap happens at eval_at.
    prog4 = program(
        ("PUSH", 0x7FFFFFFF), ("PUSH", 2), ("ADD",), ("HALT",),
    )
    mod4 = ff.evaluate_program_mod(prog4)
    np4 = np_exec.execute(prog4).steps[-1].top
    _check(
        "overflow[I32_MAX + 2].modpoly==numpy",
        mod4.top.eval_at(mod4.bindings) == (int(np4) & 0xFFFFFFFF),
        f"mod={mod4.top.eval_at(mod4.bindings)}, np={np4}",
    )


def test_modpoly_blocked_opcodes():
    """Ops outside the polynomial-closed fragment raise in evaluate_program_mod.

    ModPoly is deliberately narrower than ``evaluate_program`` — its
    scope is the ring (ADD/SUB/MUL + stack manip). Everything else
    (comparisons, rationals, bit-ops) keeps its own boundary wrap and
    must NOT silently become a ModPoly lacking those semantics.
    """
    # DIV_S — rational fragment has its own boundary (RationalPoly).
    prog = program(("PUSH", 10), ("PUSH", 3), ("DIV_S",), ("HALT",))
    try:
        ff.evaluate_program_mod(prog)
        _fail("mod32.blocked[DIV_S]", "expected BlockedOpcodeForSymbolic")
    except ff.BlockedOpcodeForSymbolic:
        _pass("mod32.blocked[DIV_S]")

    # LT_S — comparison gate, not polynomial.
    prog = program(("PUSH", 3), ("PUSH", 5), ("LT_S",), ("HALT",))
    try:
        ff.evaluate_program_mod(prog)
        _fail("mod32.blocked[LT_S]", "expected BlockedOpcodeForSymbolic")
    except ff.BlockedOpcodeForSymbolic:
        _pass("mod32.blocked[LT_S]")

    # AND — bitwise, lives in BitVec AST.
    prog = program(("PUSH", 5), ("PUSH", 3), ("AND",), ("HALT",))
    try:
        ff.evaluate_program_mod(prog)
        _fail("mod32.blocked[AND]", "expected BlockedOpcodeForSymbolic")
    except ff.BlockedOpcodeForSymbolic:
        _pass("mod32.blocked[AND]")

    # JZ — control flow; the mod-32 driver is straight-line like its
    # Poly-over-ℤ sibling.
    prog = program(("PUSH", 1), ("JZ", 10), ("HALT",))
    try:
        ff.evaluate_program_mod(prog)
        _fail("mod32.blocked[JZ]", "expected BlockedOpcodeForSymbolic")
    except ff.BlockedOpcodeForSymbolic:
        _pass("mod32.blocked[JZ]")


def test_blocked_opcodes():
    """Ops outside the fragment raise BlockedOpcodeForSymbolic.

    Post issue #75 the fragment is ADD/SUB/MUL/DIV_S/REM_S + stack manip;
    comparisons, bitwise, control-flow-within-branchless, and arithmetic
    composition past a rational stack entry remain non-goals.
    """
    model = CompiledModel()

    # Arithmetic composition past a rational stack entry — DIV_S then ADD.
    # DIV_S is in scope, but ADD on the resulting RationalPoly is not.
    prog = program(
        ("PUSH", 10), ("PUSH", 3), ("DIV_S",),
        ("PUSH", 1), ("ADD",), ("HALT",),
    )
    try:
        model.forward_symbolic(prog)
        _fail("blocked[compose-past-DIV_S]", "expected BlockedOpcodeForSymbolic")
    except ff.BlockedOpcodeForSymbolic:
        _pass("blocked[compose-past-DIV_S]")
    except Exception as e:
        _fail("blocked[compose-past-DIV_S]",
              f"wrong exception: {type(e).__name__}: {e}")

    # JZ: control flow, not this issue's scope (in straight-line forward).
    prog = program(("PUSH", 1), ("JZ", 10), ("HALT",))
    try:
        model.forward_symbolic(prog)
        _fail("blocked[JZ]", "expected BlockedOpcodeForSymbolic")
    except ff.BlockedOpcodeForSymbolic:
        _pass("blocked[JZ]")

    # ABS: unary non-polynomial op outside the bitwise fragment (issue #77
    # brought AND/OR/XOR/SHL/SHR_S/SHR_U/CLZ/CTZ/POPCNT into scope).
    prog = program(("PUSH", -3), ("ABS",), ("HALT",))
    try:
        model.forward_symbolic(prog)
        _fail("blocked[ABS]", "expected BlockedOpcodeForSymbolic")
    except ff.BlockedOpcodeForSymbolic:
        _pass("blocked[ABS]")


# ─── Runner ───────────────────────────────────────────────────────

def main():
    tests = [
        test_primitives_add_sub_mul,
        test_primitives_matrix_shapes,
        test_symbolic_primitives,
        test_range_check,
        test_equivalence_structural,
        test_equivalence_numeric,
        test_equivalence_guarded_structural,
        test_equivalence_guarded_numeric,
        test_equivalence_unrolled_structural,
        test_equivalence_unrolled_numeric,
        test_dup_add_chain_pin,
        test_sum_of_squares_pin,
        test_primitives_div_rem_s,
        test_primitives_rational_matrix_shapes,
        test_symbolic_rational_primitives,
        test_equivalence_rational,
        test_rational_catalog_rows_classify_as_collapsed,
        test_primitives_forward_cmp,
        test_primitives_forward_eqz,
        test_primitives_cmp_matrix_shapes,
        test_symbolic_comparison_primitives,
        test_equivalence_comparison_structural,
        test_equivalence_eqz_structural,
        test_equivalence_comparison_numeric,
        test_native_max_three_way_numeric,
        test_primitives_forward_bit_binary,
        test_primitives_forward_bit_unary,
        test_primitives_bitvec_matrix_shapes,
        test_symbolic_bitvec_primitives,
        test_equivalence_bitvec_structural,
        test_equivalence_bitvec_nested_structural,
        test_equivalence_is_power_of_2_structural,
        test_equivalence_bitvec_numeric,
        test_bitvec_parameter_count,
        test_modpoly_primitives,
        test_symbolic_primitives_mod,
        test_modpoly_catalog_structural,
        test_modpoly_homomorphism_on_catalog,
        test_modpoly_catalog_numeric,
        test_modpoly_overflow_stress,
        test_modpoly_blocked_opcodes,
        test_blocked_opcodes,
    ]
    print("=" * 60)
    print("ff_symbolic tests (issue #69)")
    print("=" * 60)
    for t in tests:
        print(f"\n{t.__name__}:")
        try:
            t()
        except Exception as e:
            _failures.append(f"{t.__name__}: uncaught {type(e).__name__}: {e}")
            print(f"  FAIL  {t.__name__}  uncaught {type(e).__name__}: {e}")
            traceback.print_exc()
    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED {len(_failures)} check(s):")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
