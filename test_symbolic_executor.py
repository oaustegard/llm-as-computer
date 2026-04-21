"""Tests for symbolic_executor (issue #65).

Cross-checks the symbolic executor against ``NumPyExecutor`` on a suite
of branchless programs: for each program we run both executors and
verify the symbolic top-of-stack (after substituting the allocated
variables back to the PUSH constants) equals the numeric top.

Also pins the two PoC examples from the issue so the collapse claims
stay honest: ``PUSH 5; (DUP;ADD)×4`` must remain ``9 heads → 1 monomial``
and ``PUSH 3; PUSH 7; ADD; DUP; ADD`` must remain ``5 heads → 2 monomials``.

Run standalone (phase-file style)::

    python test_symbolic_executor.py
"""

from __future__ import annotations

import sys

import isa
from executor import NumPyExecutor
from isa import program
from symbolic_executor import (
    Poly,
    SymbolicOpNotSupported,
    SymbolicStackUnderflow,
    collapse_report,
    run_symbolic,
)


# ─── Poly unit tests ──────────────────────────────────────────────

def test_poly_constant_zero_is_empty():
    assert Poly.constant(0).terms == {}
    assert Poly.constant(0).n_monomials() == 0


def test_poly_add_combines_like_terms():
    x = Poly.variable(0)
    assert (x + x) == Poly({((0, 1),): 2})
    assert (x + x - x - x) == Poly.constant(0)


def test_poly_mul_distributes():
    x = Poly.variable(0)
    y = Poly.variable(1)
    # (x + y)*(x - y) = x² - y²
    expanded = (x + y) * (x - y)
    assert expanded == Poly({((0, 2),): 1, ((1, 2),): -1})


def test_poly_mul_merges_powers():
    x = Poly.variable(0)
    # x * x * x → x^3 with coefficient 1
    assert (x * x * x) == Poly({((0, 3),): 1})


def test_poly_eval_at_matches_arithmetic():
    x, y = Poly.variable(0), Poly.variable(1)
    p = (x + y) * (x + y) - x * x
    # (x+y)² - x² = 2xy + y²
    assert p.eval_at({0: 3, 1: 5}) == 2 * 3 * 5 + 25


def test_poly_equality_is_value_based():
    a = Poly({((0, 1),): 2, (): 3})
    b = Poly({(): 3, ((0, 1),): 2})
    assert a == b
    # zero-coefficient entries don't change identity
    assert Poly({((0, 1),): 0, (): 1}) == Poly.constant(1)


# ─── SymbolicExecutor unit tests ──────────────────────────────────

def test_poc_1_dup_add_chain():
    """PoC example 1: 9 heads → 1 monomial, top = 16·x0."""
    prog = program(("PUSH", 5), *([("DUP",), ("ADD",)] * 4), ("HALT",))
    r = run_symbolic(prog)
    assert r.n_heads == 9
    assert r.top.n_monomials() == 1
    assert r.top == Poly({((0, 1),): 16})
    assert r.top.eval_at(r.bindings) == 80


def test_poc_2_add_dup_add():
    """PoC example 2: 5 heads → 2 monomials, top = 2·x0 + 2·x1."""
    prog = program(("PUSH", 3), ("PUSH", 7), ("ADD",),
                   ("DUP",), ("ADD",), ("HALT",))
    r = run_symbolic(prog)
    assert r.n_heads == 5
    assert r.top.n_monomials() == 2
    assert r.top == Poly({((0, 1),): 2, ((1, 1),): 2})
    assert r.top.eval_at(r.bindings) == 20


def test_sub_works():
    prog = program(("PUSH", 10), ("PUSH", 3), ("SUB",), ("HALT",))
    r = run_symbolic(prog)
    # 10 - 3 = x0 - x1
    assert r.top == Poly({((0, 1),): 1, ((1, 1),): -1})
    assert r.top.eval_at(r.bindings) == 7


def test_mul_produces_higher_degree():
    prog = program(("PUSH", 3), ("PUSH", 4), ("MUL",), ("HALT",))
    r = run_symbolic(prog)
    # x0 * x1, single monomial, degree 2
    assert r.top == Poly({((0, 1), (1, 1)): 1})
    assert r.top.eval_at(r.bindings) == 12


def test_swap_reorders():
    prog = program(("PUSH", 2), ("PUSH", 7), ("SWAP",), ("SUB",), ("HALT",))
    r = run_symbolic(prog)
    # After swap: stack = [7, 2]; SUB → 7 - 2 → x1 - x0
    assert r.top.eval_at(r.bindings) == 5


def test_over_copies_second():
    prog = program(("PUSH", 11), ("PUSH", 4), ("OVER",), ("HALT",))
    r = run_symbolic(prog)
    # stack = [11, 4, 11]; top = x0
    assert r.top == Poly({((0, 1),): 1})
    assert r.top.eval_at(r.bindings) == 11


def test_rot_three_entries():
    prog = program(("PUSH", 1), ("PUSH", 2), ("PUSH", 3),
                   ("ROT",), ("HALT",))
    r = run_symbolic(prog)
    # [a, b, c] -> [b, c, a] ; top is now x0 (original bottom)
    assert r.top.eval_at(r.bindings) == 1


def test_nop_preserves_stack():
    prog = program(("PUSH", 9), ("NOP",), ("HALT",))
    r = run_symbolic(prog)
    assert r.top.eval_at(r.bindings) == 9


def test_unsupported_op_raises():
    # AND remains outside _POLY_OPS (bitwise — not rational-algebraic).
    # DIV_S / REM_S are in scope per issue #75.
    prog = program(("PUSH", 12), ("PUSH", 10), ("AND",), ("HALT",))
    try:
        run_symbolic(prog)
    except SymbolicOpNotSupported as e:
        assert "AND" in str(e)
    else:
        raise AssertionError("expected SymbolicOpNotSupported for AND")


def test_div_s_composition_raises():
    """DIV_S then ADD — arithmetic on a RationalPoly is out of scope (#75)."""
    prog = program(
        ("PUSH", 10), ("PUSH", 3), ("DIV_S",),
        ("PUSH", 1), ("ADD",), ("HALT",),
    )
    try:
        run_symbolic(prog)
    except SymbolicOpNotSupported as e:
        assert "rational" in str(e).lower() or "DIV_S" in str(e) or "REM_S" in str(e)
    else:
        raise AssertionError(
            "expected SymbolicOpNotSupported for ADD on RationalPoly"
        )


def test_pop_underflow_raises():
    try:
        run_symbolic(program(("POP",), ("HALT",)))
    except SymbolicStackUnderflow:
        pass
    else:
        raise AssertionError("expected SymbolicStackUnderflow")


# ─── Cross-check against NumPyExecutor ────────────────────────────

_BRANCHLESS_SUITE = [
    ("single_push", program(("PUSH", 42), ("HALT",))),
    ("push_dup", program(("PUSH", 7), ("DUP",), ("HALT",))),
    ("simple_add", program(("PUSH", 3), ("PUSH", 4), ("ADD",), ("HALT",))),
    ("poc_16x", program(("PUSH", 5),
                         *([("DUP",), ("ADD",)] * 4), ("HALT",))),
    ("poc_2x_plus_2y", program(("PUSH", 3), ("PUSH", 7), ("ADD",),
                                 ("DUP",), ("ADD",), ("HALT",))),
    ("simple_sub", program(("PUSH", 20), ("PUSH", 8), ("SUB",), ("HALT",))),
    ("simple_mul", program(("PUSH", 6), ("PUSH", 7), ("MUL",), ("HALT",))),
    ("swap_sub", program(("PUSH", 2), ("PUSH", 15), ("SWAP",),
                          ("SUB",), ("HALT",))),
    ("over_add", program(("PUSH", 10), ("PUSH", 5), ("OVER",),
                          ("ADD",), ("HALT",))),
    ("rot_pattern", program(("PUSH", 1), ("PUSH", 2), ("PUSH", 3),
                             ("ROT",), ("HALT",))),
    ("mixed_ops", program(("PUSH", 2), ("PUSH", 3), ("PUSH", 4),
                           ("MUL",), ("ADD",), ("HALT",))),
    ("square_via_dupmul",
        program(("PUSH", 7), ("DUP",), ("MUL",), ("HALT",))),
    ("sum_of_squares",
        program(("PUSH", 3), ("DUP",), ("MUL",),
                ("PUSH", 4), ("DUP",), ("MUL",),
                ("ADD",), ("HALT",))),
]


def test_cross_check_numpy_executor():
    """For every branchless program, the symbolic top substituted with
    the original PUSH constants must match NumPyExecutor's top-of-stack."""
    np_exec = NumPyExecutor()
    failures = []
    for name, prog in _BRANCHLESS_SUITE:
        # Numeric trace
        trace = np_exec.execute(prog)
        numeric_top = trace.steps[-1].top if trace.steps else 0
        # Symbolic
        sym = run_symbolic(prog)
        symbolic_top = sym.top.eval_at(sym.bindings) if sym.bindings else (
            sym.top.eval_at({}) if sym.top.n_monomials() else 0
        )
        if numeric_top != symbolic_top:
            failures.append(
                f"{name}: numeric={numeric_top}  symbolic={symbolic_top}  "
                f"(top poly = {sym.top}, bindings = {sym.bindings})"
            )
    assert not failures, "\n  ".join(["mismatches:"] + failures)


# ─── Collapse report ──────────────────────────────────────────────

def test_collapse_report_format():
    msg = collapse_report(
        program(("PUSH", 5), *([("DUP",), ("ADD",)] * 4), ("HALT",)),
        name="DUP_ADD_x4",
    )
    assert msg.startswith("DUP_ADD_x4: ")
    assert "9 heads" in msg
    assert "1 monomial" in msg
    assert "16·x0" in msg


# ─── Runner ───────────────────────────────────────────────────────

def _collect_tests():
    return {name: obj for name, obj in globals().items()
            if callable(obj) and name.startswith("test_")}


def main():
    tests = _collect_tests()
    print(f"Running {len(tests)} tests from test_symbolic_executor.py")
    passed = 0
    failed = []
    for name, fn in tests.items():
        try:
            fn()
        except Exception as e:
            failed.append((name, e))
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            continue
        passed += 1
        print(f"  ✓ {name}")
    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        return 1

    # Print the collapse report for the PoC programs as a sanity banner.
    print("\nCollapse report:")
    for name, prog in _BRANCHLESS_SUITE:
        print(f"  {collapse_report(prog, name=name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
