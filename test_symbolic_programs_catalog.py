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
from symbolic_executor import GuardedPoly, IndicatorPoly, Poly, REL_EQ, REL_GT, REL_LE, REL_LT
from symbolic_programs_catalog import (
    _EML_AVAILABLE,
    CatalogEntry,
    GuardedCaseEML,
    STATUS_BLOCKED_LOOP_SYM,
    STATUS_BLOCKED_OPCODE,
    STATUS_COLLAPSED,
    STATUS_COLLAPSED_GUARDED,
    STATUS_COLLAPSED_UNROLLED,
    _guard_to_expr,
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


def test_classify_guarded_on_symbolic_branch():
    """JZ on a symbolic input forks into a GuardedPoly (issue #70).

    Before issue #70 this program was reported as ``blocked_control``;
    the forking executor now handles JZ/JNZ when the condition is a
    polynomial in the symbolic inputs.

    Program: ``PUSH a; DUP; JZ 4; HALT; POP; PUSH 0; HALT`` — clamp-to-zero.
    """
    prog = program(
        ("PUSH", 7), ("DUP",), ("JZ", 4), ("HALT",),
        ("POP",), ("PUSH", 0), ("HALT",),
    )
    cr = classify_program(prog)
    assert cr.status == STATUS_COLLAPSED_GUARDED
    assert cr.guarded is not None
    assert cr.n_cases == 2
    # The two cases split on whether x0 is zero.
    guard_polys = [gs[0].poly for gs, _ in cr.guarded.cases]
    assert all(gp == Poly.variable(0) for gp in guard_polys)
    assert {g.eq_zero for gs, _ in cr.guarded.cases for g in gs} == {True, False}


def test_classify_blocks_nonpolynomial_opcode():
    # PUSH, ABS: ABS is outside _POLY_OPS (unary absolute value).
    # DIV_S / REM_S are in scope per issue #75; bitwise ops (AND/OR/XOR/
    # SHL/SHR_S/SHR_U/CLZ/CTZ/POPCNT) are in scope per issue #77.
    prog = [
        isa.Instruction(isa.OP_PUSH, -3),
        isa.Instruction(isa.OP_ABS),
        isa.Instruction(isa.OP_HALT),
    ]
    cr = classify_program(prog)
    assert cr.status == "blocked_opcode"
    assert cr.blocker == "ABS"


def test_classify_rational_top_on_div_s():
    """DIV_S at HALT now collapses to a RationalPoly (issue #75)."""
    prog = [
        isa.Instruction(isa.OP_PUSH, 2),
        isa.Instruction(isa.OP_PUSH, 7),
        isa.Instruction(isa.OP_DIV_S),
        isa.Instruction(isa.OP_HALT),
    ]
    cr = classify_program(prog)
    assert cr.status == "collapsed", cr.status
    assert cr.rational is not None
    assert cr.poly is None


def test_classify_rational_top_on_rem_s():
    """REM_S at HALT collapses to a SymbolicRemainder (issue #75)."""
    prog = [
        isa.Instruction(isa.OP_PUSH, 2),
        isa.Instruction(isa.OP_PUSH, 7),
        isa.Instruction(isa.OP_REM_S),
        isa.Instruction(isa.OP_HALT),
    ]
    cr = classify_program(prog)
    assert cr.status == "collapsed", cr.status
    assert cr.rational is not None
    assert cr.poly is None


def test_classify_first_blocker_wins():
    # ABS precedes JZ — non-polynomial opcode is reported even though
    # control flow also appears further down. (LT_S no longer blocks
    # as of issue #76 — it collapses to an IndicatorPoly. CLZ no longer
    # blocks as of issue #77 — it collapses to a BitVec.)
    prog = [
        isa.Instruction(isa.OP_PUSH, 16),
        isa.Instruction(isa.OP_ABS),
        isa.Instruction(isa.OP_JZ, 99),
        isa.Instruction(isa.OP_HALT),
    ]
    cr = classify_program(prog)
    assert cr.status == "blocked_opcode"
    assert cr.blocker == "ABS"


# ─── Comparison opcodes (issue #76) ───────────────────────────────

def test_classify_compare_lt_s_collapses_to_indicator():
    """LT_S on symbolic inputs collapses to an IndicatorPoly top."""
    prog = [
        isa.Instruction(isa.OP_PUSH, 3),
        isa.Instruction(isa.OP_PUSH, 5),
        isa.Instruction(isa.OP_LT_S),
        isa.Instruction(isa.OP_HALT),
    ]
    cr = classify_program(prog)
    assert cr.status == STATUS_COLLAPSED
    assert cr.indicator is not None
    assert isinstance(cr.indicator, IndicatorPoly)
    assert cr.indicator.relation == REL_LT
    # diff = vb - va = x0 - x1
    assert cr.indicator.poly == Poly({((0, 1),): 1, ((1, 1),): -1})
    assert cr.poly is None and cr.rational is None


def test_classify_compare_eqz_collapses_to_indicator():
    """EQZ on a symbolic input collapses to an IndicatorPoly(poly=x0, REL_EQ)."""
    prog = [
        isa.Instruction(isa.OP_PUSH, 0),
        isa.Instruction(isa.OP_EQZ),
        isa.Instruction(isa.OP_HALT),
    ]
    cr = classify_program(prog)
    assert cr.status == STATUS_COLLAPSED
    assert cr.indicator is not None
    assert cr.indicator.relation == REL_EQ
    assert cr.indicator.poly == Poly.variable(0)


def test_classify_native_max_is_guarded_with_le_gt():
    """GT_S → JZ → POP/HALT dispatch hoists the comparison's relation
    into the Guard pair, producing a two-case GuardedPoly with LE/GT
    guards (not EQ/NE — that was the S1 backward-compat path)."""
    import programs as P
    prog, expected = P.make_native_max(3, 5)
    cr = classify_program(prog)
    assert cr.status == STATUS_COLLAPSED_GUARDED
    assert cr.guarded is not None
    assert cr.n_cases == 2
    relations = {g.relation for gs, _ in cr.guarded.cases for g in gs}
    assert relations == {REL_LE, REL_GT}


def test_guard_to_expr_renders_full_relation_set():
    """Render all six relation symbols correctly."""
    from symbolic_executor import Guard, REL_EQ, REL_NE, REL_LT, REL_LE, REL_GT, REL_GE
    p = Poly.variable(0)
    assert _guard_to_expr(Guard(poly=p, relation=REL_EQ)) == "x0 == 0"
    assert _guard_to_expr(Guard(poly=p, relation=REL_NE)) == "x0 != 0"
    assert _guard_to_expr(Guard(poly=p, relation=REL_LT)) == "x0 < 0"
    assert _guard_to_expr(Guard(poly=p, relation=REL_LE)) == "x0 <= 0"
    assert _guard_to_expr(Guard(poly=p, relation=REL_GT)) == "x0 > 0"
    assert _guard_to_expr(Guard(poly=p, relation=REL_GE)) == "x0 >= 0"


# ─── run_catalog — end-to-end pipeline ────────────────────────────

def test_run_catalog_default_has_collapsed_and_blocked():
    rows = run_catalog()
    n_collapsed = sum(1 for r in rows if r.status == "collapsed")
    n_blocked = sum(1 for r in rows if r.status.startswith("blocked"))
    assert n_collapsed >= 10, f"expected ≥10 collapsed rows, got {n_collapsed}"
    # Floor dropped from ≥5 to ≥4 in issue #76: compare_lt_s + native_max
    # (and compare_eqz on first landing) now classify as collapsed /
    # collapsed_guarded rather than blocked_opcode. Floor dropped again
    # from ≥4 to ≥2 in issue #77: bitwise AND/OR/XOR/SHL/SHR_S/SHR_U +
    # CLZ/CTZ/POPCNT + bit_extract + log2_floor + is_power_of_2 +
    # popcount_loop now classify as collapsed / collapsed_unrolled.
    # Only ABS and NEG remain as unary non-polynomial blockers.
    assert n_blocked >= 2, f"expected ≥2 blocked rows, got {n_blocked}"


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


def test_run_catalog_pins_guarded_rows():
    """Issue #70 pins: the three finite-conditional demos collapse to
    two-case GuardedPolys and cross-check numerically."""
    rows = {r.name: r for r in run_catalog()}

    assert rows["clamp_zero(5)"].status == STATUS_COLLAPSED_GUARDED
    assert rows["clamp_zero(5)"].n_cases == 2
    assert rows["clamp_zero(5)"].numpy_top == 5
    assert rows["clamp_zero(5)"].numeric_match is True

    assert rows["select_by_sign(7)"].status == STATUS_COLLAPSED_GUARDED
    assert rows["select_by_sign(7)"].n_cases == 2
    assert rows["select_by_sign(7)"].numpy_top == 2
    assert rows["select_by_sign(7)"].numeric_match is True

    assert rows["either_or(3,7,1)"].status == STATUS_COLLAPSED_GUARDED
    assert rows["either_or(3,7,1)"].n_cases == 2
    assert rows["either_or(3,7,1)"].numpy_top == 7
    assert rows["either_or(3,7,1)"].numeric_match is True


def test_run_catalog_pins_comparison_rows():
    """Issue #76 pins: comparison rows move from blocked_opcode to
    collapsed (LT_S, EQZ) / collapsed_guarded (native_max with GT_S+JZ
    dispatch). All three numeric-match at the catalog's concrete inputs.
    """
    rows = {r.name: r for r in run_catalog()}

    r = rows["compare_lt_s(3,5)"]
    assert r.status == STATUS_COLLAPSED
    assert r.is_indicator is True
    assert r.poly_expr == "[x0 - x1 < 0]"
    assert r.numpy_top == 1  # 3 < 5 → 1
    assert r.numeric_match is True
    # eml-sr has no sign primitive — indicator rows skip the eml columns.
    assert r.eml_size is None and r.eml_depth is None

    r = rows["compare_eqz(0)"]
    assert r.status == STATUS_COLLAPSED
    assert r.is_indicator is True
    assert r.poly_expr == "[x0 == 0]"
    assert r.numpy_top == 1  # 0 == 0 → 1
    assert r.numeric_match is True

    r = rows["native_max(3,5)"]
    assert r.status == STATUS_COLLAPSED_GUARDED
    assert r.n_cases == 2
    assert r.numpy_top == 5
    assert r.numeric_match is True
    # Both branches render with LE / GT relation symbols.
    joined = " ; ".join(r.case_exprs or [])
    assert " <= 0" in joined and " > 0" in joined


def test_run_catalog_pins_unrolled_rows():
    """Issue #70 pins: bounded loops unroll at concrete inputs and the
    result is a constant Poly equal to the numpy top."""
    rows = {r.name: r for r in run_catalog()}

    # fib(5) = 5, factorial(4) = 24, is_even(6) = 1, power_of_2(4) = 16.
    for name, expected in [
        ("fibonacci(5)", 5),
        ("factorial(4)", 24),
        ("is_even(6)", 1),
        ("power_of_2(4)", 16),
    ]:
        r = rows[name]
        assert r.status == STATUS_COLLAPSED_UNROLLED, (name, r.status)
        assert r.poly_expr == str(expected), (name, r.poly_expr)
        assert r.numpy_top == expected
        assert r.numeric_match is True


def test_guarded_row_case_exprs_are_populated():
    """Guarded rows expose per-case expressions, not just a lumped poly."""
    rows = {r.name: r for r in run_catalog()}
    r = rows["clamp_zero(5)"]
    assert r.case_exprs is not None and len(r.case_exprs) == 2
    # Each case_expr renders as "{guards} → value_poly".
    assert all("→" in ce for ce in r.case_exprs)


# ─── S2: sharper EML accounting for guarded programs ─────────────

def test_guarded_row_case_eml_populated_when_available():
    """Each guarded case exposes its own EML size/depth for both the
    value poly and each guard poly (issue #68 S2)."""
    if not _EML_AVAILABLE:
        print("  (eml-sr not available — skipping guarded case_eml test)")
        return
    rows = {r.name: r for r in run_catalog()}
    for name in ("clamp_zero(5)", "select_by_sign(7)", "either_or(3,7,1)"):
        r = rows[name]
        assert r.case_eml is not None, name
        assert len(r.case_eml) == r.n_cases, name
        for c in r.case_eml:
            assert isinstance(c, GuardedCaseEML), name
            assert c.value_size >= 1, name
            assert c.value_depth >= 0, name
            # Every case in these demos has exactly one guard (single JZ
            # fork on a single symbolic input).
            assert len(c.guard_sizes) == 1 == len(c.guard_depths), name
            assert c.guard_sizes[0] >= 1, name
            assert c.guard_depths[0] >= 0, name


def test_guarded_row_guard_trees_accounted():
    """``eml_guard_size`` / ``eml_guard_depth`` are populated and
    reconcile with the per-case breakdown (issue #68 S2)."""
    if not _EML_AVAILABLE:
        print("  (eml-sr not available — skipping guard accounting test)")
        return
    rows = {r.name: r for r in run_catalog()}
    for name in ("clamp_zero(5)", "select_by_sign(7)", "either_or(3,7,1)"):
        r = rows[name]
        assert r.eml_guard_size is not None, name
        assert r.eml_guard_depth is not None, name
        # Reconcile aggregates against the per-case breakdown.
        expected_size = sum(sum(c.guard_sizes) for c in (r.case_eml or []))
        expected_depth = max(
            (d for c in (r.case_eml or []) for d in c.guard_depths),
            default=0,
        )
        assert r.eml_guard_size == expected_size, name
        assert r.eml_guard_depth == expected_depth, name


def test_guarded_row_value_aggregates_reconcile():
    """``eml_size`` / ``eml_depth`` (value-tree aggregates) match the
    sum / max of ``case_eml`` entries (issue #68 S2 — S1 semantics preserved)."""
    if not _EML_AVAILABLE:
        print("  (eml-sr not available — skipping value reconcile test)")
        return
    rows = {r.name: r for r in run_catalog()}
    for name in ("clamp_zero(5)", "select_by_sign(7)", "either_or(3,7,1)"):
        r = rows[name]
        expected_size = sum(c.value_size for c in (r.case_eml or []))
        expected_depth = max(
            (c.value_depth for c in (r.case_eml or [])),
            default=0,
        )
        assert r.eml_size == expected_size, name
        assert r.eml_depth == expected_depth, name


def test_guarded_row_guard_trees_match_trivial_polys():
    """All three demo guards are single-variable polys (``x0`` or
    ``x2``), which compile to 1-node, depth-0 EML trees — so each
    case contributes exactly 1 to the guard-size total."""
    if not _EML_AVAILABLE:
        print("  (eml-sr not available — skipping trivial-guard shape test)")
        return
    rows = {r.name: r for r in run_catalog()}
    # 2 cases × 1 guard × size-1 tree = 2; depth 0.
    for name in ("clamp_zero(5)", "select_by_sign(7)", "either_or(3,7,1)"):
        r = rows[name]
        assert r.eml_guard_size == 2, (name, r.eml_guard_size)
        assert r.eml_guard_depth == 0, (name, r.eml_guard_depth)


def test_guarded_row_fields_absent_without_eml():
    """When eml-sr is unavailable the S2 fields remain None — the row
    still populates poly/guards text fields but not tree accounting."""
    if _EML_AVAILABLE:
        print("  (eml-sr available — skipping no-eml fallback test)")
        return
    rows = {r.name: r for r in run_catalog()}
    r = rows["clamp_zero(5)"]
    assert r.eml_guard_size is None
    assert r.eml_guard_depth is None
    assert r.case_eml is None
    # But the textual columns still come through.
    assert r.case_exprs is not None


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



# ─── Generated catalog entries (issue #96) ────────────────────────

def test_generated_entries_present_in_default_catalog():
    """At least 6 generated entries appear in the default catalog."""
    rows = run_catalog()
    gen_rows = [r for r in rows if r.name.startswith("gen_")]
    assert len(gen_rows) >= 5, f"expected ≥5 generated rows, got {len(gen_rows)}"


def test_generated_entries_all_collapsed():
    """Every generated entry classifies as STATUS_COLLAPSED."""
    rows = run_catalog()
    gen_rows = [r for r in rows if r.name.startswith("gen_")]
    assert gen_rows, "no generated rows found — catalog not extended"
    non_collapsed = [r for r in gen_rows if r.status != STATUS_COLLAPSED]
    assert not non_collapsed, (
        "generated rows not STATUS_COLLAPSED: "
        + ", ".join(f"{r.name}={r.status}" for r in non_collapsed)
    )


def test_generated_entries_numeric_match():
    """Every generated entry has numeric_match=True (NumPy ≡ Poly at the
    PUSH-0 binding point)."""
    rows = run_catalog()
    gen_rows = [r for r in rows if r.name.startswith("gen_")]
    failures = [r for r in gen_rows if r.numeric_match is not True]
    assert not failures, (
        "numeric_match failed: "
        + ", ".join(f"{r.name}={r.numeric_match}" for r in failures)
    )


def test_generated_entry_shapes():
    """Pin polynomial shape and monomial count for selected generated entries."""
    rows = {r.name: r for r in run_catalog()}

    # x0^5 — 1 monomial, degree 5
    r = rows["gen_x0_fifth"]
    assert r.status == STATUS_COLLAPSED
    assert r.n_monomials == 1
    assert r.poly_expr == "x0^5"
    assert r.numeric_match is True

    # 10*x0 — 1 monomial, large coefficient
    r = rows["gen_10x0"]
    assert r.status == STATUS_COLLAPSED
    assert r.n_monomials == 1
    assert r.poly_expr == "10*x0"
    assert r.numeric_match is True

    # x0^2 + x0*x1 + x1^2 + x0 + x1 — exactly 5 monomials
    r = rows["gen_sum_of_five_terms"]
    assert r.status == STATUS_COLLAPSED
    assert r.n_monomials == 5
    assert r.numeric_match is True

    # 3*x0*x1 - 2*x0^2 + x1 — exactly 3 monomials (mixed signs)
    r = rows["gen_mixed_signs"]
    assert r.status == STATUS_COLLAPSED
    assert r.n_monomials == 3
    assert r.numeric_match is True

    # x0*x1*x2*x3 — 1 monomial, 4 variables
    r = rows["gen_x0x1x2x3"]
    assert r.status == STATUS_COLLAPSED
    assert r.n_monomials == 1
    assert r.numeric_match is True

    # x0*x1 + x2*x3 — 2 monomials, 4 variables
    r = rows["gen_x0x1_plus_x2x3"]
    assert r.status == STATUS_COLLAPSED
    assert r.n_monomials == 2
    assert r.numeric_match is True


def test_generated_entries_eml_when_available():
    """When eml-sr is present, generated collapsed entries get EML tree metrics."""
    if not _EML_AVAILABLE:
        print("  (eml-sr not available — skipping generated eml test)")
        return
    rows = run_catalog()
    gen_rows = [r for r in rows if r.name.startswith("gen_") and r.status == STATUS_COLLAPSED]
    assert gen_rows
    for r in gen_rows:
        assert r.eml_size is not None and r.eml_size >= 1, r.name
        assert r.eml_depth is not None and r.eml_depth >= 0, r.name


def test_generated_catalog_standalone():
    """_generated_catalog() can be run in isolation and produces exactly 6 entries."""
    from symbolic_programs_catalog import _generated_catalog
    entries = _generated_catalog()
    assert len(entries) == 6
    names = {e.name for e in entries}
    expected_names = {
        "gen_x0_fifth",
        "gen_x0x1x2x3",
        "gen_sum_of_five_terms",
        "gen_mixed_signs",
        "gen_10x0",
        "gen_x0x1_plus_x2x3",
    }
    assert names == expected_names, f"unexpected names: {names ^ expected_names}"



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
