"""Tests for the loop-invariant inference path (issue #89).

Pins the closed-form acceptance criteria from
``dev/loop_invariant_inference.md``:

  - ``sum_1_to_n_sym(n)``    → Tier 1, Poly(n(n+1)/2)
  - ``power_of_2_sym(n)``    → Tier 2, ClosedForm with A=[[2]]
  - ``fibonacci_sym(n)``     → Tier 2, ClosedForm with A=[[0,1],[1,1]]
  - ``factorial_sym(n)``     → Tier 3, ProductForm over a Poly factor

For each row:
  - Structural equality on the emitted closed form (shape, sibling type,
    trip count, projection slot).
  - Numeric agreement with :class:`executor.NumPyExecutor` at ≥ 5
    values of ``n`` each.
  - ``run_catalog`` classifies the sym rows as
    ``collapsed_closed_form`` and ``numeric_match=True``.
  - ``run_forking(solve_recurrences=False)`` reproduces the pre-#89
    ``loop_symbolic`` path exactly.

Run standalone (phase-file style)::

    python test_closed_form.py
"""

from __future__ import annotations

import math
import sys

from executor import NumPyExecutor
from programs import (
    make_factorial,
    make_factorial_sym,
    make_fibonacci,
    make_fibonacci_sym,
    make_power_of_2,
    make_power_of_2_sym,
    make_sum_1_to_n,
    make_sum_1_to_n_sym,
)
from symbolic_executor import (
    ClosedForm,
    Poly,
    ProductForm,
    run_forking,
)
from symbolic_programs_catalog import (
    STATUS_COLLAPSED_CLOSED_FORM,
    classify_program,
    run_catalog,
)


_np = NumPyExecutor()


def _numpy_top(prog):
    trace = _np.execute(prog)
    return trace.steps[-1].top


def _fib_ref(n: int) -> int:
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


# ─── Tier 1 — sum_1_to_n_sym ──────────────────────────────────────

def test_sum_1_to_n_sym_classifies_as_tier1_poly():
    # Any ``n`` works for the structural check; pick n=5.
    prog, _ = make_sum_1_to_n_sym(5)
    cr = classify_program(prog, solve_recurrences=True)
    assert cr.status == STATUS_COLLAPSED_CLOSED_FORM
    # Tier 1 stays in Poly; ``closed_form`` is None.
    assert isinstance(cr.poly, Poly)
    assert cr.closed_form is None
    # The closed form is n(n+1)/2 in the counter variable.
    expr = repr(cr.poly)
    assert "1/2·x1" in expr or "1/2·x1^2" in expr
    assert "x1^2" in expr  # quadratic term present


def test_sum_1_to_n_sym_numeric_agrees_with_numpy():
    for n in (1, 2, 5, 10, 20):
        prog, _ = make_sum_1_to_n_sym(n)
        cr = classify_program(prog, solve_recurrences=True)
        assert cr.status == STATUS_COLLAPSED_CLOSED_FORM
        sym_val = int(cr.poly.eval_at(cr.bindings))
        np_val = _numpy_top(prog)
        expected = n * (n + 1) // 2
        assert sym_val == expected, (n, sym_val, expected)
        assert np_val == expected, (n, np_val, expected)


# ─── Tier 2 — power_of_2_sym ──────────────────────────────────────

def test_power_of_2_sym_classifies_as_tier2_closedform():
    prog, _ = make_power_of_2_sym(4)
    cr = classify_program(prog, solve_recurrences=True)
    assert cr.status == STATUS_COLLAPSED_CLOSED_FORM
    assert cr.poly is None
    assert isinstance(cr.closed_form, ClosedForm)
    cf = cr.closed_form
    # Scalar doubling — A=[[2]], b=[0], projection=0, s_0 has length 1.
    assert cf.A == ((2,),)
    assert cf.b == (0,)
    assert len(cf.s_0) == 1
    assert cf.projection == 0


def test_power_of_2_sym_numeric_agrees_with_numpy():
    # Design-doc sample: n ∈ {0, 1, 4, 8, 10}. ``make_power_of_2(0)``
    # short-circuits to a straight-line program (no loop), so at n=0
    # the classifier legitimately returns ``collapsed`` rather than
    # ``collapsed_closed_form`` — both are correct answers, and what
    # this test pins is the numeric agreement.
    for n in (0, 1, 4, 8, 10):
        prog, _ = make_power_of_2_sym(n)
        cr = classify_program(prog, solve_recurrences=True)
        expected = 2 ** n
        np_val = _numpy_top(prog)
        assert np_val == expected, (n, np_val, expected)
        if cr.status == STATUS_COLLAPSED_CLOSED_FORM:
            sym_val = int(cr.closed_form.eval_at(cr.bindings))
        else:
            # Short-circuit path (n=0): the top is already a Poly.
            assert cr.poly is not None, (n, cr.status)
            sym_val = int(cr.poly.eval_at(cr.bindings))
        assert sym_val == expected, (n, sym_val, expected)


# ─── Tier 2 — fibonacci_sym ───────────────────────────────────────

def test_fibonacci_sym_classifies_as_tier2_closedform():
    prog, _ = make_fibonacci_sym(5)
    cr = classify_program(prog, solve_recurrences=True)
    assert cr.status == STATUS_COLLAPSED_CLOSED_FORM
    assert cr.poly is None
    assert isinstance(cr.closed_form, ClosedForm)
    cf = cr.closed_form
    # 2×2 Fibonacci matrix [[0,1],[1,1]] with zero additive vector,
    # projecting out the second slot (b = fib(n)).
    assert cf.A == ((0, 1), (1, 1))
    assert cf.b == (0, 0)
    assert len(cf.s_0) == 2
    assert cf.projection == 1


def test_fibonacci_sym_numeric_agrees_with_numpy():
    # ``make_fibonacci(1)`` short-circuits; its classifier status is
    # ``collapsed`` rather than ``collapsed_closed_form``. Both are
    # correct outputs — the test pins the numeric answer either way.
    for n in (1, 2, 5, 10, 15):
        prog, _ = make_fibonacci_sym(n)
        cr = classify_program(prog, solve_recurrences=True)
        expected = _fib_ref(n)
        np_val = _numpy_top(prog)
        assert np_val == expected, (n, np_val, expected)
        if cr.status == STATUS_COLLAPSED_CLOSED_FORM:
            if cr.closed_form is not None:
                sym_val = int(cr.closed_form.eval_at(cr.bindings))
            else:
                sym_val = int(cr.poly.eval_at(cr.bindings))
        else:
            assert cr.poly is not None, (n, cr.status)
            sym_val = int(cr.poly.eval_at(cr.bindings))
        assert sym_val == expected, (n, sym_val, expected)


# ─── Tier 3 — factorial_sym ───────────────────────────────────────

def test_factorial_sym_classifies_as_tier3_productform():
    prog, _ = make_factorial_sym(4)
    cr = classify_program(prog, solve_recurrences=True)
    assert cr.status == STATUS_COLLAPSED_CLOSED_FORM
    assert cr.poly is None
    assert isinstance(cr.closed_form, ProductForm)
    pf = cr.closed_form
    # Product starts at 1, runs 1..n. The factor poly is the counter
    # slot var itself (factorial's per-step multiplication by k).
    assert pf.init == 1
    assert isinstance(pf.p, Poly)
    # counter_var shows up in p's variable set.
    assert pf.counter_var in pf.p.variables()


def test_factorial_sym_numeric_agrees_with_numpy():
    # ``make_factorial(1)`` short-circuits (returns ``[PUSH 1, HALT]``);
    # the classifier status is ``collapsed``, not ``closed_form`` —
    # the numeric answer is still 1. Mirror the power_of_2 handling.
    for n in (1, 2, 5, 7, 10):
        prog, _ = make_factorial_sym(n)
        cr = classify_program(prog, solve_recurrences=True)
        expected = math.factorial(n)
        np_val = _numpy_top(prog)
        assert np_val == expected, (n, np_val, expected)
        if cr.status == STATUS_COLLAPSED_CLOSED_FORM:
            if cr.closed_form is not None:
                sym_val = int(cr.closed_form.eval_at(cr.bindings))
            else:
                sym_val = int(cr.poly.eval_at(cr.bindings))
        else:
            assert cr.poly is not None, (n, cr.status)
            sym_val = int(cr.poly.eval_at(cr.bindings))
        assert sym_val == expected, (n, sym_val, expected)


# ─── Solver toggle — pre-#89 path is reproducible ────────────────

def test_solve_recurrences_false_reproduces_loop_symbolic():
    """With ``solve_recurrences=False`` every closed-form candidate
    must fall back to the pre-#89 ``loop_symbolic`` status at the
    run_forking layer, proving the flag fully gates the new path.

    Uses n values that exercise the loop (not the short-circuit
    n ≤ 1 branch in fibonacci / factorial / power_of_2).
    """
    for gen, n in [
        (make_sum_1_to_n_sym, 5),
        (make_power_of_2_sym, 4),
        (make_fibonacci_sym, 5),
        (make_factorial_sym, 4),
    ]:
        prog, _ = gen(n)
        r = run_forking(prog, input_mode="symbolic",
                        solve_recurrences=False)
        assert r.status == "loop_symbolic", (gen.__name__, r.status)


# ─── Catalog integration ─────────────────────────────────────────

def test_run_catalog_pins_closed_form_rows():
    """The four sym rows land in ``run_catalog`` as
    ``collapsed_closed_form`` with ``numeric_match=True``."""
    rows = {r.name: r for r in run_catalog()}
    for name in (
        "sum_1_to_n_sym(n)",
        "power_of_2_sym(n)",
        "fibonacci_sym(n)",
        "factorial_sym(n)",
    ):
        r = rows[name]
        assert r.status == STATUS_COLLAPSED_CLOSED_FORM, (name, r.status)
        assert r.is_closed_form is True, name
        assert r.numeric_match is True, name
        assert r.poly_expr, name


def test_format_report_has_closed_form_section():
    """The new summary count and section header both appear in the
    rendered markdown report."""
    from symbolic_programs_catalog import format_report
    rows = run_catalog()
    report = format_report(rows)
    assert "closed-form" in report
    assert "Collapsed (closed form" in report
    # All 4 rows are cited in the closed-form section.
    for name in ("sum_1_to_n_sym", "power_of_2_sym",
                 "fibonacci_sym", "factorial_sym"):
        assert name in report, name


# ─── Phase-file-style entrypoint ─────────────────────────────────

def main() -> int:
    tests = [
        test_sum_1_to_n_sym_classifies_as_tier1_poly,
        test_sum_1_to_n_sym_numeric_agrees_with_numpy,
        test_power_of_2_sym_classifies_as_tier2_closedform,
        test_power_of_2_sym_numeric_agrees_with_numpy,
        test_fibonacci_sym_classifies_as_tier2_closedform,
        test_fibonacci_sym_numeric_agrees_with_numpy,
        test_factorial_sym_classifies_as_tier3_productform,
        test_factorial_sym_numeric_agrees_with_numpy,
        test_solve_recurrences_false_reproduces_loop_symbolic,
        test_run_catalog_pins_closed_form_rows,
        test_format_report_has_closed_form_section,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
