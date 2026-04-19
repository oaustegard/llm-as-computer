"""Tests for symbolic_programs_catalog (issue #65 follow-up).

Covers the three pieces of the bridge:

1. ``poly_to_expr`` emits strings that parse cleanly and evaluate to the
   same integer as ``Poly.eval_at``.
2. ``classify_program`` lands on the right bucket — collapsed, blocked by
   a non-polynomial opcode, or blocked by control flow — for a handful of
   hand-picked programs.
3. ``run_catalog`` produces a row for every entry whose ``numeric_match``
   field is True on the collapsed subset (the LAC ↔ eml-sr pipeline).

The eml-sr cross-checks are skipped when the compiler isn't importable so
this file still passes with just the LAC repo checked out. Run standalone::

    python test_symbolic_programs_catalog.py
"""

from __future__ import annotations

import sys

import isa
from executor import NumPyExecutor
from isa import program
from symbolic_executor import Poly
from symbolic_programs_catalog import (
    _EML_AVAILABLE,
    CatalogEntry,
    classify_program,
    poly_to_expr,
    run_catalog,
)


# ─── poly_to_expr ─────────────────────────────────────────────────

def test_poly_to_expr_empty_is_zero():
    assert poly_to_expr(Poly.constant(0)) == "0"


def test_poly_to_expr_constant_positive():
    assert poly_to_expr(Poly.constant(7)) == "7"


def test_poly_to_expr_single_variable():
    assert poly_to_expr(Poly.variable(0)) == "x0"


def test_poly_to_expr_sum_with_coefficients():
    p = Poly({((0, 1),): 2, ((1, 1),): 2})
    assert poly_to_expr(p) == "2*x0 + 2*x1"


def test_poly_to_expr_negative_coefficient():
    p = Poly({((0, 1),): 1, ((1, 1),): -1})  # x0 - x1
    assert poly_to_expr(p) == "x0 - x1"


def test_poly_to_expr_leading_negative():
    p = Poly({((0, 1),): -3, ((1, 1),): 2})  # -3*x0 + 2*x1
    assert poly_to_expr(p) == "-3*x0 + 2*x1"


def test_poly_to_expr_higher_degree():
    p = Poly({((0, 2),): 1, ((1, 2),): -1})  # x0^2 - x1^2
    assert poly_to_expr(p) == "x0^2 - x1^2"


def test_poly_to_expr_mixed_monomial():
    p = Poly({((0, 1), (1, 1)): 3})
    assert poly_to_expr(p) == "3*x0*x1"


# ─── classify_program ─────────────────────────────────────────────

def test_classify_collapsed_branchless():
    cr = classify_program(program(
        ("PUSH", 5),
        *([("DUP",), ("ADD",)] * 4),
        ("HALT",),
    ))
    assert cr.status == "collapsed"
    assert cr.n_heads == 9
    assert cr.poly == Poly({((0, 1),): 16})


def test_classify_blocks_control_flow():
    # PUSH 3, DUP, JZ 99, HALT — first JZ wins
    prog = [
        isa.Instruction(isa.OP_PUSH, 3),
        isa.Instruction(isa.OP_DUP),
        isa.Instruction(isa.OP_JZ, 99),
        isa.Instruction(isa.OP_HALT),
    ]
    cr = classify_program(prog)
    assert cr.status == "blocked_control"
    assert cr.blocker == "JZ"


def test_classify_blocks_nonpolynomial_opcode():
    # PUSH 2, PUSH 7, DIV_S, HALT
    prog = [
        isa.Instruction(isa.OP_PUSH, 2),
        isa.Instruction(isa.OP_PUSH, 7),
        isa.Instruction(isa.OP_DIV_S),
        isa.Instruction(isa.OP_HALT),
    ]
    cr = classify_program(prog)
    assert cr.status == "blocked_opcode"
    assert cr.blocker == "DIV_S"


def test_classify_first_blocker_wins():
    # LT_S precedes JZ — non-polynomial opcode is reported even though
    # control flow also appears further down.
    prog = [
        isa.Instruction(isa.OP_PUSH, 3),
        isa.Instruction(isa.OP_PUSH, 5),
        isa.Instruction(isa.OP_LT_S),
        isa.Instruction(isa.OP_JZ, 99),
        isa.Instruction(isa.OP_HALT),
    ]
    cr = classify_program(prog)
    assert cr.status == "blocked_opcode"
    assert cr.blocker == "LT_S"


# ─── run_catalog — end-to-end pipeline ────────────────────────────

def test_run_catalog_default_has_collapsed_and_blocked():
    rows = run_catalog()
    n_collapsed = sum(1 for r in rows if r.status == "collapsed")
    n_blocked = sum(1 for r in rows if r.status.startswith("blocked"))
    assert n_collapsed >= 10, f"expected ≥10 collapsed rows, got {n_collapsed}"
    assert n_blocked >= 5, f"expected ≥5 blocked rows, got {n_blocked}"


def test_run_catalog_collapsed_rows_numeric_match():
    """Every collapsed row's 3-way check (NumPy ≡ Poly ≡ EML) must pass.

    Degenerates to a 2-way check (NumPy ≡ Poly) when eml-sr is not on
    the path, which is still a meaningful guard.
    """
    rows = run_catalog()
    failures = [r for r in rows
                if r.status == "collapsed" and r.numeric_match is not True]
    assert not failures, (
        "numeric_match false on: "
        + ", ".join(f"{r.name}={r.numeric_match}" for r in failures)
    )


def test_run_catalog_pins_poc_collapse_counts():
    rows = {r.name: r for r in run_catalog()}
    # These two pins trace back to the issue #65 PoC text and must not drift.
    assert rows["dup_add_chain_x4"].n_heads == 9
    assert rows["dup_add_chain_x4"].n_monomials == 1
    assert rows["dup_add_chain_x4"].poly_expr == "16*x0"
    assert rows["add_dup_add"].n_heads == 5
    assert rows["add_dup_add"].n_monomials == 2
    assert rows["add_dup_add"].poly_expr == "2*x0 + 2*x1"


def test_run_catalog_single_entry_override():
    """Custom entry list is honored instead of the default catalog."""
    entry = CatalogEntry(
        name="x0_plus_x1",
        prog=program(("PUSH", 2), ("PUSH", 5), ("ADD",), ("HALT",)),
        expected=7,
    )
    rows = run_catalog([entry])
    assert len(rows) == 1
    assert rows[0].status == "collapsed"
    assert rows[0].n_heads == 3
    assert rows[0].n_monomials == 2
    assert rows[0].numpy_top == 7
    assert rows[0].numeric_match is True


# ─── Cross-repo bridge — skipped when eml-sr unavailable ──────────

def test_poly_to_expr_round_trips_through_eml_compiler():
    if not _EML_AVAILABLE:
        print("  (eml-sr not available — skipping round-trip test)")
        return
    from eml_compiler import compile_expr, eval_eml

    p = Poly({((0, 1),): 2, ((1, 1),): 2})  # 2*x0 + 2*x1
    tree = compile_expr(poly_to_expr(p), variables=["x0", "x1"])
    val = eval_eml(tree, {"x0": 3, "x1": 7})
    assert abs(val.imag) < 1e-6
    assert int(round(val.real)) == p.eval_at({0: 3, 1: 7}) == 20


def test_collapsed_rows_have_eml_tree_sizes_when_available():
    """EML tree size/depth should be populated for every collapsed row
    when the compiler is importable."""
    if not _EML_AVAILABLE:
        print("  (eml-sr not available — skipping eml-tree test)")
        return
    rows = run_catalog()
    for r in rows:
        if r.status != "collapsed":
            continue
        assert r.eml_size is not None and r.eml_size >= 1, r.name
        assert r.eml_depth is not None and r.eml_depth >= 0, r.name


def test_known_eml_shapes_match_paper():
    """Sanity pins for the two EML identities the paper gives explicitly:

    - ``x*y`` compiles to a 35-node, depth-8 tree (Table 4).
    - ``x + y`` compiles to a 21-node, depth-6 tree.

    Not strictly invariants of *this* module — they're invariants of
    eml_compiler — but a drift here would indicate the upstream
    identity table changed, which is load-bearing for our report
    numbers.
    """
    if not _EML_AVAILABLE:
        print("  (eml-sr not available — skipping paper-shape test)")
        return
    from eml_compiler import compile_expr, tree_depth, tree_size

    mul_tree = compile_expr("x*y", variables=["x", "y"])
    assert tree_size(mul_tree) == 35
    assert tree_depth(mul_tree) == 8

    add_tree = compile_expr("x + y", variables=["x", "y"])
    assert tree_size(add_tree) == 21
    assert tree_depth(add_tree) == 6


# ─── Runner ───────────────────────────────────────────────────────

def _collect_tests():
    return {name: obj for name, obj in globals().items()
            if callable(obj) and name.startswith("test_")}


def main() -> int:
    tests = _collect_tests()
    print(f"Running {len(tests)} tests from test_symbolic_programs_catalog.py")
    print(f"  (eml-sr available: {_EML_AVAILABLE})")
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
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
