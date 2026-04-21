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
    GuardedPoly,
    Poly,
    RationalPoly,
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


# ─── Blocked-opcode handling ──────────────────────────────────────

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

    # AND: bitwise.
    prog = program(("PUSH", 12), ("PUSH", 10), ("AND",), ("HALT",))
    try:
        model.forward_symbolic(prog)
        _fail("blocked[AND]", "expected BlockedOpcodeForSymbolic")
    except ff.BlockedOpcodeForSymbolic:
        _pass("blocked[AND]")


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
