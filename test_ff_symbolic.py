"""Tests for ff_symbolic (issue #69).

Verifies the bilinear FF dispatch realises the same polynomial the symbolic
executor emits — not just numerically, but *structurally*. For every
collapsed catalog program, the test asserts
``forward_symbolic(P).top == SymbolicExecutor(P).top`` on canonical Poly
equality (value-compare), alongside a numerical agreement check against
:class:`NumPyExecutor` on concrete inputs.

Layout:
  * ``test_primitives_*`` — unit-level checks on E / M_ADD / M_SUB / B_MUL.
  * ``test_range_check`` — Option (a) bound enforcement.
  * ``test_equivalence_*`` — the core equivalence theorem, parametrised
    over every row in ``symbolic_programs_catalog`` whose status is
    ``STATUS_COLLAPSED``.
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
from symbolic_executor import Poly, run_symbolic
from symbolic_programs_catalog import (
    STATUS_COLLAPSED,
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


# ─── Blocked-opcode handling ──────────────────────────────────────

def test_blocked_opcodes():
    """Ops outside the ADD/SUB/MUL fragment raise BlockedOpcodeForSymbolic.

    The issue's scope is explicit: DIV_S / REM_S / comparisons / bitwise
    are non-goals. forward_symbolic refuses them rather than returning a
    wrong answer.
    """
    model = CompiledModel()

    # DIV_S: not polynomial-closed.
    prog = program(("PUSH", 10), ("PUSH", 3), ("DIV_S",), ("HALT",))
    try:
        model.forward_symbolic(prog)
        _fail("blocked[DIV_S]", "expected BlockedOpcodeForSymbolic")
    except ff.BlockedOpcodeForSymbolic:
        _pass("blocked[DIV_S]")
    except Exception as e:
        _fail("blocked[DIV_S]", f"wrong exception: {type(e).__name__}: {e}")

    # JZ: control flow, not this issue's scope.
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
        test_dup_add_chain_pin,
        test_sum_of_squares_pin,
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
