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
    BitVec,
    Guard,
    GuardedPoly,
    IndicatorPoly,
    Poly,
    REL_EQ,
    REL_GE,
    REL_GT,
    REL_LE,
    REL_LT,
    REL_NE,
    SymbolicOpNotSupported,
    SymbolicStackUnderflow,
    collapse_report,
    guarded_to_mermaid,
    run_forking,
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
    # ROTL remains outside _POLY_OPS: issue #77 adds AND/OR/XOR/SHL/SHR_S/
    # SHR_U/CLZ/CTZ/POPCNT but leaves ROTL/ROTR as follow-ups. DIV_S /
    # REM_S / comparisons / bit ops in scope per #75 / #76 / #77.
    prog = program(("PUSH", 5), ("PUSH", 1), ("ROTL",), ("HALT",))
    try:
        run_symbolic(prog)
    except SymbolicOpNotSupported as e:
        assert "ROTL" in str(e)
    else:
        raise AssertionError("expected SymbolicOpNotSupported for ROTL")


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


# ─── Comparisons + IndicatorPoly (issue #76) ─────────────────────

def test_indicator_poly_gates_at_boundary():
    """IndicatorPoly.eval_at returns 0/1 by applying the relation to the
    underlying Poly's concrete value — the non-polynomial gate lives at
    the boundary, not inside the polynomial algebra."""
    x, y = Poly.variable(0), Poly.variable(1)
    diff = x - y
    ip_lt = IndicatorPoly(poly=diff, relation=REL_LT)
    assert ip_lt.eval_at({0: 3, 1: 5}) == 1   # 3 - 5 = -2 < 0
    assert ip_lt.eval_at({0: 5, 1: 5}) == 0   # 0 < 0 is False
    assert ip_lt.eval_at({0: 7, 1: 5}) == 0   # 2 < 0 is False

    ip_eq = IndicatorPoly(poly=x, relation=REL_EQ)
    assert ip_eq.eval_at({0: 0}) == 1
    assert ip_eq.eval_at({0: 3}) == 0


def test_comparison_ops_return_indicator_poly():
    """Each signed comparison leaves an IndicatorPoly on top whose poly
    is ``vb - va`` (the WASM-flavored diff, with pa=top / pb=SP-1) and
    whose relation tags the opcode's test."""
    cases = [
        (isa.OP_LT_S, REL_LT),
        (isa.OP_GT_S, REL_GT),
        (isa.OP_LE_S, REL_LE),
        (isa.OP_GE_S, REL_GE),
        (isa.OP_EQ,   REL_EQ),
        (isa.OP_NE,   REL_NE),
    ]
    for op, expected_rel in cases:
        prog = [
            isa.Instruction(isa.OP_PUSH, 3),
            isa.Instruction(isa.OP_PUSH, 5),
            isa.Instruction(op),
            isa.Instruction(isa.OP_HALT),
        ]
        r = run_symbolic(prog)
        assert isinstance(r.top, IndicatorPoly), op
        assert r.top.relation == expected_rel, op
        # pa = top = x1 (pushed second), pb = SP-1 = x0 (pushed first).
        # Diff = pb - pa = x0 - x1.
        assert r.top.poly == Poly({((0, 1),): 1, ((1, 1),): -1}), op


def test_eqz_returns_indicator_with_eq_relation():
    prog = [
        isa.Instruction(isa.OP_PUSH, 0),
        isa.Instruction(isa.OP_EQZ),
        isa.Instruction(isa.OP_HALT),
    ]
    r = run_symbolic(prog)
    assert isinstance(r.top, IndicatorPoly)
    assert r.top.relation == REL_EQ
    assert r.top.poly == Poly.variable(0)


def test_comparison_symbolic_vs_numeric_equivalence():
    """Every signed comparison agrees with NumPyExecutor on sample
    inputs drawn from the {<, =, >} regions of the diff."""
    np_exec = NumPyExecutor()
    failures = []
    for op in (isa.OP_LT_S, isa.OP_GT_S, isa.OP_LE_S, isa.OP_GE_S,
               isa.OP_EQ, isa.OP_NE):
        for a, b in [(3, 5), (5, 3), (7, 7), (-2, 4), (0, 0), (-1, 0)]:
            prog = [
                isa.Instruction(isa.OP_PUSH, a),
                isa.Instruction(isa.OP_PUSH, b),
                isa.Instruction(op),
                isa.Instruction(isa.OP_HALT),
            ]
            trace = np_exec.execute(prog)
            numeric = trace.steps[-1].top
            r = run_symbolic(prog)
            symbolic = r.top.eval_at(r.bindings)
            if numeric != symbolic:
                failures.append(
                    f"op={isa.OP_NAMES[op]} a={a} b={b} "
                    f"numeric={numeric} symbolic={symbolic}"
                )
    assert not failures, "\n  ".join(["mismatches:"] + failures)


def test_eqz_symbolic_vs_numeric_equivalence():
    """EQZ agrees with NumPyExecutor on inputs spanning the {=0, ≠0} split."""
    np_exec = NumPyExecutor()
    for a in [-3, -1, 0, 1, 5]:
        prog = [
            isa.Instruction(isa.OP_PUSH, a),
            isa.Instruction(isa.OP_EQZ),
            isa.Instruction(isa.OP_HALT),
        ]
        trace = np_exec.execute(prog)
        numeric = trace.steps[-1].top
        r = run_symbolic(prog)
        symbolic = r.top.eval_at(r.bindings)
        assert numeric == symbolic, (a, numeric, symbolic)


def test_jz_on_indicator_hoists_relation_into_guards():
    """JZ consuming an IndicatorPoly produces a two-branch fork whose
    Guards carry the *negated* / taken relation — not an EQ wrapping."""
    # native_max: PUSH a, PUSH b, OVER, OVER, GT_S, JZ skip, POP, HALT; skip: SWAP POP HALT
    import programs as P
    prog, _ = P.make_native_max(3, 5)
    r = run_forking(prog, input_mode="symbolic")
    assert r.status == "guarded"
    assert isinstance(r.top, GuardedPoly)
    assert r.top.n_cases() == 2
    relations = {g.relation for gs, _ in r.top.cases for g in gs}
    # GT_S test ⇒ JZ-taken branch gets LE, JZ-skipped gets GT.
    assert relations == {REL_LE, REL_GT}


def test_guard_eq_zero_backward_compat_property():
    """Guard.eq_zero still reads True for REL_EQ, False for others —
    kept as a @property shim so older tests / catalog rows don't break."""
    p = Poly.variable(0)
    assert Guard(poly=p, relation=REL_EQ).eq_zero is True
    assert Guard(poly=p, relation=REL_NE).eq_zero is False
    assert Guard(poly=p, relation=REL_LT).eq_zero is False
    assert Guard(poly=p, relation=REL_GE).eq_zero is False


def test_composition_past_indicator_blocked():
    """Arithmetic composed on top of an IndicatorPoly is out of scope —
    comparisons produce non-polynomial 0/1 values that don't round-trip
    through Poly arithmetic. The executor must raise rather than silently
    pretend the indicator is a Poly."""
    prog = [
        isa.Instruction(isa.OP_PUSH, 3),
        isa.Instruction(isa.OP_PUSH, 5),
        isa.Instruction(isa.OP_LT_S),
        isa.Instruction(isa.OP_PUSH, 1),
        isa.Instruction(isa.OP_ADD),
        isa.Instruction(isa.OP_HALT),
    ]
    try:
        run_symbolic(prog)
    except SymbolicOpNotSupported:
        pass
    else:
        raise AssertionError(
            "expected SymbolicOpNotSupported for ADD on IndicatorPoly"
        )



# ─── Bit-vector AST (issue #77) ──────────────────────────────────

def test_bitvec_binary_op_wraps_operands_in_natural_order():
    """AND/OR/XOR/SHL/SHR_S/SHR_U wrap ``(SP-1, top)`` verbatim — the
    natural left-right reading order. The executor doesn't simplify
    ``AND(x, x) → x``; the AST is intentionally literal."""
    for op_code, name in [
        (isa.OP_AND, "AND"), (isa.OP_OR, "OR"), (isa.OP_XOR, "XOR"),
        (isa.OP_SHL, "SHL"), (isa.OP_SHR_S, "SHR_S"), (isa.OP_SHR_U, "SHR_U"),
    ]:
        prog = [
            isa.Instruction(isa.OP_PUSH, 12),
            isa.Instruction(isa.OP_PUSH, 10),
            isa.Instruction(op_code),
            isa.Instruction(isa.OP_HALT),
        ]
        r = run_symbolic(prog)
        assert isinstance(r.top, BitVec), f"{name}: got {type(r.top).__name__}"
        assert r.top.op == name
        assert len(r.top.operands) == 2
        # SP-1 was variable-from-PUSH-12, top was variable-from-PUSH-10.
        a, b = r.top.operands
        assert r.bindings[a.variables()[0]] == 12  # SP-1
        assert r.bindings[b.variables()[0]] == 10  # top


def test_bitvec_unary_op_single_operand():
    """CLZ/CTZ/POPCNT wrap a single operand."""
    for op_code, name in [
        (isa.OP_CLZ, "CLZ"), (isa.OP_CTZ, "CTZ"), (isa.OP_POPCNT, "POPCNT"),
    ]:
        prog = [
            isa.Instruction(isa.OP_PUSH, 13),
            isa.Instruction(op_code),
            isa.Instruction(isa.OP_HALT),
        ]
        r = run_symbolic(prog)
        assert isinstance(r.top, BitVec)
        assert r.top.op == name
        assert len(r.top.operands) == 1


def test_bitvec_eval_at_matches_numpy_binary():
    """Every binary bit-op's ``BitVec.eval_at`` matches NumPyExecutor's
    numeric result across a spread of i32 inputs."""
    np_exec = NumPyExecutor()
    pairs = [(12, 10), (0, 5), (5, 0), (-1, 3), (0xFF, 0xF0),
             (0x80000000, 1), (7, 2)]
    failures = []
    for op_code in [isa.OP_AND, isa.OP_OR, isa.OP_XOR,
                    isa.OP_SHL, isa.OP_SHR_S, isa.OP_SHR_U]:
        for a, b in pairs:
            prog = [
                isa.Instruction(isa.OP_PUSH, a),
                isa.Instruction(isa.OP_PUSH, b),
                isa.Instruction(op_code),
                isa.Instruction(isa.OP_HALT),
            ]
            numeric = np_exec.execute(prog).steps[-1].top
            r = run_symbolic(prog)
            symbolic = r.top.eval_at(r.bindings)
            if numeric != symbolic:
                failures.append(
                    f"op={isa.OP_NAMES[op_code]} a={a} b={b} "
                    f"numeric={numeric} symbolic={symbolic}"
                )
    assert not failures, "\n  ".join(["mismatches:"] + failures)


def test_bitvec_eval_at_matches_numpy_unary():
    """CLZ/CTZ/POPCNT's ``BitVec.eval_at`` matches NumPyExecutor across
    inputs spanning the zero / low-bit / high-bit regions."""
    np_exec = NumPyExecutor()
    values = [0, 1, 2, 7, 8, 16, 0xFF, 0x80000000, -1, 13]
    failures = []
    for op_code in [isa.OP_CLZ, isa.OP_CTZ, isa.OP_POPCNT]:
        for n in values:
            prog = [
                isa.Instruction(isa.OP_PUSH, n),
                isa.Instruction(op_code),
                isa.Instruction(isa.OP_HALT),
            ]
            numeric = np_exec.execute(prog).steps[-1].top
            r = run_symbolic(prog)
            symbolic = r.top.eval_at(r.bindings)
            if numeric != symbolic:
                failures.append(
                    f"op={isa.OP_NAMES[op_code]} n={n} "
                    f"numeric={numeric} symbolic={symbolic}"
                )
    assert not failures, "\n  ".join(["mismatches:"] + failures)


def test_bitvec_hybrid_arithmetic_lifts_into_ast():
    """``log2_floor(n) = 31 - CLZ(n)`` — SUB with a BitVec operand must
    lift into the BitVec AST rather than widening Poly. The top is
    a BitVec("SUB", (Poly(31), BitVec("CLZ", (n,)))) tree."""
    import programs as P
    prog, expected = P.make_log2_floor(8)
    r = run_symbolic(prog)
    assert isinstance(r.top, BitVec)
    assert r.top.op == "SUB"
    assert r.top.eval_at(r.bindings) == expected


def test_bitvec_nested_ast_bit_extract():
    """``bit_extract(n, k) = (n >>u k) & 1`` — AND wraps an inner
    SHR_U BitVec, exercising nested AST composition."""
    import programs as P
    prog, expected = P.make_bit_extract(5, 0)
    r = run_symbolic(prog)
    assert isinstance(r.top, BitVec)
    assert r.top.op == "AND"
    # Outer op is AND; left operand is a BitVec (inner SHR_U).
    assert isinstance(r.top.operands[0], BitVec)
    assert r.top.operands[0].op == "SHR_U"
    assert r.top.eval_at(r.bindings) == expected


def test_bitvec_wrapped_in_indicator_is_power_of_2():
    """``is_power_of_2(n) = (POPCNT(n) == 1)`` — EQ wraps its BitVec
    diff in an IndicatorPoly. Issue #77 widened IndicatorPoly.poly to
    accept BitVec."""
    import programs as P
    prog, expected = P.make_is_power_of_2(8)
    r = run_symbolic(prog)
    assert isinstance(r.top, IndicatorPoly)
    assert isinstance(r.top.poly, BitVec)
    assert r.top.eval_at(r.bindings) == expected


def test_bitvec_equivalence_across_catalog_rows():
    """Every bit-vector catalog program: ``run_symbolic(...).top.eval_at``
    matches NumPyExecutor's top. Covers AND/OR/XOR/CLZ/CTZ/POPCNT +
    bit_extract + log2_floor. (``is_power_of_2`` covered by
    ``test_bitvec_wrapped_in_indicator_is_power_of_2``; ``popcount_loop``
    needs the forking executor's concrete-mode unroll.)"""
    import programs as P
    np_exec = NumPyExecutor()
    cases = [
        ("bitwise_and(12,10)",    P.make_bitwise_binary(isa.OP_AND, 12, 10)),
        ("bitwise_or(12,10)",     P.make_bitwise_binary(isa.OP_OR, 12, 10)),
        ("bitwise_xor(12,10)",    P.make_bitwise_binary(isa.OP_XOR, 12, 10)),
        ("bitwise_shl(3,2)",      P.make_bitwise_binary(isa.OP_SHL, 3, 2)),
        ("bitwise_shr_u(-1,4)",   P.make_bitwise_binary(isa.OP_SHR_U, -1, 4)),
        ("bitwise_shr_s(-1,4)",   P.make_bitwise_binary(isa.OP_SHR_S, -1, 4)),
        ("native_clz(16)",        P.make_native_clz(16)),
        ("native_ctz(8)",         P.make_native_ctz(8)),
        ("native_popcnt(13)",     P.make_native_popcnt(13)),
        ("bit_extract(5,0)",      P.make_bit_extract(5, 0)),
        ("log2_floor(8)",         P.make_log2_floor(8)),
    ]
    failures = []
    for name, (prog, expected) in cases:
        numeric = np_exec.execute(prog).steps[-1].top
        r = run_symbolic(prog)
        try:
            symbolic = r.top.eval_at(r.bindings)
        except Exception as e:
            failures.append(f"{name}: eval_at raised {type(e).__name__}: {e}")
            continue
        if numeric != symbolic or numeric != expected:
            failures.append(
                f"{name}: numeric={numeric} symbolic={symbolic} "
                f"expected={expected}"
            )
    assert not failures, "\n  ".join(["mismatches:"] + failures)


def test_bitvec_popcount_loop_unrolls_in_concrete_mode():
    """popcount_loop's JZ on a BitVec cond is out-of-scope in symbolic
    mode, but concrete mode reduces the BitVec to a literal at each
    step so the loop unrolls deterministically (issue #77)."""
    import programs as P
    prog, expected = P.make_popcount_loop(5)
    r = run_forking(prog, input_mode="concrete")
    assert r.status == "unrolled"
    # Top may be a BitVec (residual AST with only literals inside).
    if isinstance(r.top, BitVec):
        assert r.top.eval_at({}) == expected
    else:
        assert r.top.eval_at({}) == expected


def test_bitvec_structural_equality():
    """Two BitVec nodes with the same op + operands compare equal —
    the equivalence test for the FF bit-op primitives."""
    a = Poly.variable(0)
    b = Poly.variable(1)
    assert BitVec("AND", (a, b)) == BitVec("AND", (a, b))
    assert BitVec("AND", (a, b)) != BitVec("OR", (a, b))
    assert BitVec("AND", (a, b)) != BitVec("AND", (b, a))  # order matters


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


# ─── guarded_to_mermaid ───────────────────────────────────────────

def _simple_gp() -> GuardedPoly:
    """2-case single-guard GuardedPoly: {x0 == 0} → x5 | {x0 != 0} → x0."""
    x0 = Poly.variable(0)
    x5 = Poly.variable(5)
    return GuardedPoly(cases=(
        ((Guard(x0, REL_EQ),), x5),
        ((Guard(x0, REL_NE),), x0),
    ))


def test_guarded_to_mermaid_flowchart_header():
    """Output starts with the required Mermaid flowchart directive."""
    out = guarded_to_mermaid(_simple_gp())
    assert out.startswith("flowchart TD"), repr(out[:60])


def test_guarded_to_mermaid_covers_every_case_exactly_once():
    """Each case's value_poly appears as a Mermaid leaf label exactly once."""
    gp = _simple_gp()
    out = guarded_to_mermaid(gp)
    for _guards, value in gp.cases:
        # Use the full leaf-node syntax ["value"] to avoid false positives
        # from guard labels that may contain the same substring.
        leaf_pat = f'["{repr(value)}"]'
        count = out.count(leaf_pat)
        assert count == 1, (
            f"{leaf_pat!r} appears {count} times (expected 1) in:\n{out}"
        )


def test_guarded_to_mermaid_two_case_has_one_decision_node():
    """A 2-case single-guard GuardedPoly produces exactly 1 decision diamond."""
    import re
    out = guarded_to_mermaid(_simple_gp())
    # Decision nodes look like D1{...} in the output
    decision_lines = [ln for ln in out.splitlines() if re.search(r'\bD\d+\{', ln)]
    assert len(decision_lines) == 1, (
        f"Expected 1 decision node, got {len(decision_lines)}:\n{out}"
    )


def test_guarded_to_mermaid_true_and_false_edges_present():
    """The output contains both True and False edges for a binary split."""
    out = guarded_to_mermaid(_simple_gp())
    assert "|True|" in out, f"Missing True edge:\n{out}"
    assert "|False|" in out, f"Missing False edge:\n{out}"


def test_guarded_to_mermaid_three_case_has_two_decision_nodes():
    """A 3-case GuardedPoly produces exactly 2 decision nodes."""
    import re
    x0 = Poly.variable(0)
    x1 = Poly.variable(1)
    x2 = Poly.variable(2)
    # Construct a 3-case partition: x0<0, x0>0, else (x0==0)
    gp = GuardedPoly(cases=(
        ((Guard(x0, REL_LT),), x1),
        ((Guard(x0, REL_GT),), x2),
        ((Guard(x0, REL_EQ),), Poly.constant(0)),
    ))
    out = guarded_to_mermaid(gp)
    decision_lines = [ln for ln in out.splitlines() if re.search(r'\bD\d+\{', ln)]
    assert len(decision_lines) == 2, (
        f"Expected 2 decision nodes, got {len(decision_lines)}:\n{out}"
    )
    # All 3 case values appear exactly once as leaf labels.
    for _guards, value in gp.cases:
        leaf_pat = f'["{repr(value)}"]'
        assert out.count(leaf_pat) == 1, (
            f"{leaf_pat!r} should appear exactly once as a leaf:\n{out}"
        )


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
