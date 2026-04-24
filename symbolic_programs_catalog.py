"""LAC program catalog → polynomial → EML tree (issue #65 follow-up).

Bridges :mod:`symbolic_executor` (PR #66) to eml-sr's :mod:`eml_compiler`:
for every branchless program in :mod:`programs`, run the symbolic executor,
emit the top-of-stack polynomial as an elementary expression string, and
compile that to a pure-EML tree — reporting tree size / depth and a
three-way numeric cross-check against ``NumPyExecutor``.

Non-branchless programs are still reported, classified by their blocker
(a non-polynomial opcode like DIV_S, or control flow JZ/JNZ) rather than
forced through the pipeline.

The eml-sr import is optional. When it's on ``PYTHONPATH`` the full
``weights → polynomial → single-operator expression`` story lands; when it
isn't, the polynomial column is still emitted and the EML columns show
``–``. Either way the module stands alone in the LAC repo.

Run standalone::

    PYTHONPATH=../eml-sr python symbolic_programs_catalog.py

or, if the eml-sr spoke is checked out alongside::

    python symbolic_programs_catalog.py

Prints the markdown report to stdout. Tests live in
``test_symbolic_programs_catalog.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import isa
from executor import NumPyExecutor
from isa import program
from symbolic_executor import (
    BitVec,
    ClosedForm,
    ForkingResult,
    Guard,
    GuardedPoly,
    IndicatorPoly,
    Poly,
    ProductForm,
    RationalPoly,
    SymbolicOpNotSupported,
    SymbolicRemainder,
    SymbolicStackUnderflow,
    _REL_SYMBOL,
    _relation_holds,
    run_forking,
    run_symbolic,
)

try:  # eml-sr is a sibling spoke; optional for the cross-repo demo.
    from eml_compiler import compile_expr, eval_eml, tree_depth, tree_size
    _EML_AVAILABLE = True
except ImportError:  # pragma: no cover — flagged in the report
    _EML_AVAILABLE = False


# ─── Poly → expression string emitter ─────────────────────────────

def _mono_to_expr(mono, prefix: str = "x") -> str:
    if not mono:
        return "1"
    parts = [f"{prefix}{v}" if p == 1 else f"{prefix}{v}^{p}"
             for v, p in mono]
    return "*".join(parts)


def poly_to_expr(poly: Poly, var_prefix: str = "x") -> str:
    """Emit a :class:`Poly` as an elementary expression string.

    Output uses ``+ - * ^`` — the grammar :func:`eml_compiler.compile_expr`
    accepts. An empty :class:`Poly` (the canonical zero) renders as ``"0"``.
    Negative coefficients render as ``- term`` rather than ``+ (-1)*term``
    so the output round-trips cleanly through the parser.
    """
    if not poly.terms:
        return "0"
    ordered = sorted(
        poly.terms.items(),
        key=lambda kv: (sum(p for _, p in kv[0]), kv[0]),
    )
    out: List[str] = []
    for i, (mono, coeff) in enumerate(ordered):
        ms = _mono_to_expr(mono, var_prefix)
        negative = coeff < 0
        c = abs(coeff)
        if ms == "1":
            term = str(c)
        elif c == 1:
            term = ms
        else:
            term = f"{c}*{ms}"
        if i == 0:
            out.append(f"-{term}" if negative else term)
        else:
            out.append(f"- {term}" if negative else f"+ {term}")
    return " ".join(out)


# ─── Classification ───────────────────────────────────────────────

_POLY_OPS = {
    isa.OP_PUSH, isa.OP_POP, isa.OP_DUP, isa.OP_HALT,
    isa.OP_ADD, isa.OP_SUB, isa.OP_MUL,
    isa.OP_DIV_S, isa.OP_REM_S,
    isa.OP_SWAP, isa.OP_OVER, isa.OP_ROT, isa.OP_NOP,
    # Comparisons (issue #76) — collapse to IndicatorPoly tops and
    # hoist relations into Guards when consumed by JZ/JNZ.
    isa.OP_EQZ, isa.OP_EQ, isa.OP_NE,
    isa.OP_LT_S, isa.OP_GT_S, isa.OP_LE_S, isa.OP_GE_S,
    # Bit-vector fragment (issue #77) — collapse to BitVec AST tops.
    isa.OP_AND, isa.OP_OR, isa.OP_XOR,
    isa.OP_SHL, isa.OP_SHR_S, isa.OP_SHR_U,
    isa.OP_CLZ, isa.OP_CTZ, isa.OP_POPCNT,
    # Local-variable slots (issues #100 / #102) — polynomial-closed by
    # construction, needed by poly_to_program-generated rows (issue #96).
    # Kept in sync with symbolic_executor._POLY_OPS.
    isa.OP_LOCAL_GET, isa.OP_LOCAL_SET, isa.OP_LOCAL_TEE,
}
# Control flow — handled by the forking executor (issue #70).
_BRANCH_OPS = {isa.OP_JZ, isa.OP_JNZ}
# The forking executor accepts this full set.
_FORKING_OPS = _POLY_OPS | _BRANCH_OPS


# Status codes emitted by classify_program.
STATUS_COLLAPSED              = "collapsed"
STATUS_COLLAPSED_GUARDED      = "collapsed_guarded"
STATUS_COLLAPSED_UNROLLED     = "collapsed_unrolled"
STATUS_COLLAPSED_CLOSED_FORM  = "collapsed_closed_form"
STATUS_BLOCKED_OPCODE         = "blocked_opcode"
STATUS_BLOCKED_UNDERFLOW      = "blocked_underflow"
STATUS_BLOCKED_LOOP_SYM       = "blocked_loop_symbolic"
STATUS_BLOCKED_PATH_EXP       = "blocked_path_explosion"


@dataclass
class ClassificationResult:
    status: str
    blocker: Optional[str] = None
    poly: Optional[Poly] = None           # present when status == collapsed*
    guarded: Optional[GuardedPoly] = None # present when status == collapsed_guarded
    # Rational tops (issue #75) — present when the program ends on a single
    # DIV_S (RationalPoly) or REM_S (SymbolicRemainder).
    rational: Optional[Any] = None        # Union[RationalPoly, SymbolicRemainder]
    # Indicator tops (issue #76) — present when the program ends on a
    # comparison (EQZ/EQ/NE/LT_S/LE_S/GT_S/GE_S) with an unconsumed
    # IndicatorPoly on top of the stack.
    indicator: Optional[IndicatorPoly] = None
    # BitVec tops (issue #77) — present when the program ends on any
    # bit-vector op (AND/OR/XOR/SHL/SHR_S/SHR_U/CLZ/CTZ/POPCNT) or a
    # hybrid arithmetic op lifted into the BitVec AST.
    bitvec: Optional[BitVec] = None
    # Closed-form tops (issue #89) — present when the program ends in a
    # loop that the recurrence solver closed. Tier 1 (affine) returns a
    # Poly via the ``poly`` field with ``closed_form`` left as None.
    # Tier 2 (linear constant-matrix) populates ``closed_form`` with a
    # ClosedForm; Tier 3 (multiplicative) with a ProductForm.
    closed_form: Optional[Any] = None     # Union[ClosedForm, ProductForm]
    n_heads: int = 0
    bindings: Dict[int, int] = field(default_factory=dict)
    n_cases: int = 0                       # number of cases in guarded output


def _poly_or_guarded(result: ForkingResult):
    """Pull the top out of a ForkingResult, already Poly | GuardedPoly | None."""
    return result.top


def classify_program(prog, *, solve_recurrences: bool = False) -> ClassificationResult:
    """Pre-flight check + forking symbolic execute. First blocker wins.

    ``solve_recurrences`` (issue #89): when ``True``, loops that hit a
    symbolic back-edge are routed through the recurrence solver and
    classified as ``collapsed_closed_form`` on success — with the
    appropriate sibling (:class:`Poly` for Tier 1, :class:`ClosedForm`
    for Tier 2, :class:`ProductForm` for Tier 3) populated on the
    result. Default ``False`` preserves the pre-#89 behaviour so
    existing concrete-counter rows (``fibonacci(5)`` etc.) continue to
    classify as ``collapsed_unrolled`` via the concrete-mode fallback.

    Order of checks:
      1. ``blocked_opcode`` — any op outside ``_FORKING_OPS``.
      2. Run :func:`run_forking` in symbolic mode:
         - straight → ``collapsed`` (same semantics as the old executor).
         - guarded → ``collapsed_guarded`` with a GuardedPoly top.
         - closed_form (issue #89, ``solve_recurrences=True``) →
           ``collapsed_closed_form``.
         - unrolled (bounded loop with concrete counter) → ``collapsed_unrolled``.
         - loop_symbolic → retry in concrete mode; if that completes with
           straight/unrolled, return ``collapsed_unrolled``; else
           ``blocked_loop_symbolic``.
         - path_explosion → ``blocked_path_explosion``.
         - blocked_underflow → ``blocked_underflow``.
    """
    for instr in prog:
        op = instr.op
        if op not in _FORKING_OPS:
            return ClassificationResult(
                status=STATUS_BLOCKED_OPCODE,
                blocker=isa.OP_NAMES.get(op, f"?{op}"),
            )

    try:
        r_sym = run_forking(prog, input_mode="symbolic",
                            solve_recurrences=solve_recurrences)
    except SymbolicOpNotSupported as sym_err:
        # Symbolic mode might fail on paths that concrete mode handles
        # (e.g. a bit-op loop where JZ on a BitVec cond is symbolic-only
        # out of scope, but concrete PUSHs reduce the BitVec to a
        # literal — issue #77). Retry in concrete mode before blocking.
        try:
            r_conc = run_forking(prog, input_mode="concrete")
        except (SymbolicOpNotSupported, SymbolicStackUnderflow):
            return ClassificationResult(
                status=STATUS_BLOCKED_OPCODE, blocker=str(sym_err)
            )
        if r_conc.status in ("straight", "unrolled") and isinstance(r_conc.top, Poly):
            return ClassificationResult(
                status=STATUS_COLLAPSED_UNROLLED, poly=r_conc.top,
                n_heads=r_conc.n_heads, bindings={},
            )
        if r_conc.status in ("straight", "unrolled") and isinstance(r_conc.top, BitVec):
            return ClassificationResult(
                status=STATUS_COLLAPSED_UNROLLED, bitvec=r_conc.top,
                n_heads=r_conc.n_heads, bindings={},
            )
        if r_conc.status == "guarded" and isinstance(r_conc.top, GuardedPoly):
            return ClassificationResult(
                status=STATUS_COLLAPSED_UNROLLED, guarded=r_conc.top,
                n_heads=r_conc.n_heads, bindings={},
                n_cases=r_conc.top.n_cases(),
            )
        return ClassificationResult(
            status=STATUS_BLOCKED_OPCODE, blocker=str(sym_err)
        )
    except SymbolicStackUnderflow as e:
        return ClassificationResult(status=STATUS_BLOCKED_UNDERFLOW, blocker=str(e))

    if r_sym.status == "path_explosion":
        return ClassificationResult(status=STATUS_BLOCKED_PATH_EXP)
    if r_sym.status == "blocked_underflow":
        return ClassificationResult(status=STATUS_BLOCKED_UNDERFLOW)

    if r_sym.status == "straight":
        if isinstance(r_sym.top, IndicatorPoly):
            return ClassificationResult(
                status=STATUS_COLLAPSED, indicator=r_sym.top,
                n_heads=r_sym.n_heads, bindings=dict(r_sym.bindings),
            )
        if isinstance(r_sym.top, BitVec):
            return ClassificationResult(
                status=STATUS_COLLAPSED, bitvec=r_sym.top,
                n_heads=r_sym.n_heads, bindings=dict(r_sym.bindings),
            )
        if isinstance(r_sym.top, (RationalPoly, SymbolicRemainder)):
            return ClassificationResult(
                status=STATUS_COLLAPSED, rational=r_sym.top,
                n_heads=r_sym.n_heads, bindings=dict(r_sym.bindings),
            )
        top = r_sym.top if isinstance(r_sym.top, Poly) else None
        return ClassificationResult(
            status=STATUS_COLLAPSED, poly=top,
            n_heads=r_sym.n_heads, bindings=dict(r_sym.bindings),
        )
    if r_sym.status == "guarded":
        if isinstance(r_sym.top, GuardedPoly):
            return ClassificationResult(
                status=STATUS_COLLAPSED_GUARDED, guarded=r_sym.top,
                n_heads=r_sym.n_heads, bindings=dict(r_sym.bindings),
                n_cases=r_sym.top.n_cases(),
            )
        # Single-case degenerate → treat as collapsed.
        return ClassificationResult(
            status=STATUS_COLLAPSED, poly=r_sym.top,
            n_heads=r_sym.n_heads, bindings=dict(r_sym.bindings),
        )
    if r_sym.status == "unrolled":
        # Pure symbolic-mode bounded-loop unroll (rare: requires the
        # counter to become concrete via the polynomial arithmetic).
        top = r_sym.top if isinstance(r_sym.top, Poly) else None
        if top is not None:
            return ClassificationResult(
                status=STATUS_COLLAPSED_UNROLLED, poly=top,
                n_heads=r_sym.n_heads, bindings=dict(r_sym.bindings),
            )
    if r_sym.status == "closed_form":
        # Issue #89: the recurrence solver closed the loop. Tier 1
        # stays in Poly (``poly`` field); Tier 2 / Tier 3 populate
        # ``closed_form`` with a ClosedForm / ProductForm respectively.
        if isinstance(r_sym.top, Poly):
            return ClassificationResult(
                status=STATUS_COLLAPSED_CLOSED_FORM,
                poly=r_sym.top,
                n_heads=r_sym.n_heads, bindings=dict(r_sym.bindings),
            )
        if isinstance(r_sym.top, (ClosedForm, ProductForm)):
            return ClassificationResult(
                status=STATUS_COLLAPSED_CLOSED_FORM,
                closed_form=r_sym.top,
                n_heads=r_sym.n_heads, bindings=dict(r_sym.bindings),
            )
        # Unexpected top type — fall through to concrete retry.

    # loop_symbolic or mixed: retry in concrete mode. Concrete mode
    # specialises every PUSH to its literal value, so loops whose
    # counter is a concrete arg unroll deterministically.
    try:
        r_conc = run_forking(prog, input_mode="concrete")
    except (SymbolicOpNotSupported, SymbolicStackUnderflow):
        return ClassificationResult(status=STATUS_BLOCKED_LOOP_SYM)

    if r_conc.status in ("straight", "unrolled") and isinstance(r_conc.top, Poly):
        return ClassificationResult(
            status=STATUS_COLLAPSED_UNROLLED, poly=r_conc.top,
            n_heads=r_conc.n_heads, bindings={},
        )
    if r_conc.status == "guarded" and isinstance(r_conc.top, GuardedPoly):
        return ClassificationResult(
            status=STATUS_COLLAPSED_UNROLLED, guarded=r_conc.top,
            n_heads=r_conc.n_heads, bindings={},
            n_cases=r_conc.top.n_cases(),
        )
    if r_conc.status == "path_explosion":
        return ClassificationResult(status=STATUS_BLOCKED_PATH_EXP)
    return ClassificationResult(status=STATUS_BLOCKED_LOOP_SYM)


# ─── Catalog ──────────────────────────────────────────────────────

@dataclass
class CatalogEntry:
    name: str
    prog: Any
    expected: Any  # programs.py returns an int; None for trap cases
    # Issue #89: when True, ``classify_program`` runs with recurrence
    # solving enabled and loops close via the Tier 1/2/3 path instead
    # of falling back to concrete-mode unrolling. Default False keeps
    # existing rows (``fibonacci(5)`` etc.) in their pre-#89
    # ``collapsed_unrolled`` classification.
    solve_recurrences: bool = False
    # Issue #109 / Path B motivator gate. Per #107, Path B should not
    # activate silently — building a weight-layer gadget for a program
    # nobody runs is the anti-pattern the issue explicitly warns
    # against. Set this True on a row only when (a) the row is
    # collapsed_closed_form, and (b) there's a concrete reason to want
    # a weight-layer realisation of its eval_at (e.g. blog #82 wants
    # the stronger "weights = computer" claim for this specific row).
    # When True AND ``run_catalog`` confirms the path is in scope, the
    # row's ``ff_equiv`` is upgraded from ``solver_structural`` to
    # ``bilinear_weight_layer``.
    requests_weight_layer: bool = False


def _default_catalog() -> List[CatalogEntry]:
    """Default set of programs piped through the catalog.

    Mix of branchless PoCs (Phase 4 test_* + ``make_native_multiply`` +
    a couple of hand-written squaring/cross-sum polynomials) and blocked
    cases chosen to exercise every branch in :func:`classify_program`.
    """
    import programs as P

    entries: List[CatalogEntry] = []

    # Phase 4 branchless programs
    for name, fn in P.ALL_TESTS:
        prog, expected = fn()
        entries.append(CatalogEntry(name=name, prog=prog, expected=expected))

    # Parametric branchless, multiplicative
    prog, expected = P.make_native_multiply(3, 7)
    entries.append(CatalogEntry("native_multiply(3,7)", prog, expected))

    # Squaring via DUP + MUL — single-monomial, degree 2
    entries.append(CatalogEntry(
        name="square_via_dupmul(9)",
        prog=program(("PUSH", 9), ("DUP",), ("MUL",), ("HALT",)),
        expected=81,
    ))

    # Sum of squares — two monomials, each degree 2
    entries.append(CatalogEntry(
        name="sum_of_squares(3,4)",
        prog=program(
            ("PUSH", 3), ("DUP",), ("MUL",),
            ("PUSH", 4), ("DUP",), ("MUL",),
            ("ADD",), ("HALT",),
        ),
        expected=25,
    ))

    # Issue #65 PoC pins
    entries.append(CatalogEntry(
        name="dup_add_chain_x4",
        prog=program(("PUSH", 5),
                     *([("DUP",), ("ADD",)] * 4), ("HALT",)),
        expected=80,
    ))
    entries.append(CatalogEntry(
        name="add_dup_add",
        prog=program(("PUSH", 3), ("PUSH", 7), ("ADD",),
                     ("DUP",), ("ADD",), ("HALT",)),
        expected=20,
    ))

    # Rational tops (issue #75) — collapse to RationalPoly / SymbolicRemainder.
    entries.append(CatalogEntry("native_divmod(2,7)",   *P.make_native_divmod(2, 7)))
    entries.append(CatalogEntry("native_remainder(2,7)", *P.make_native_remainder(2, 7)))

    # Blocked — non-polynomial opcodes
    entries.append(CatalogEntry("native_abs_unary(-3)", *P.make_native_abs_unary(-3)))
    entries.append(CatalogEntry("native_neg(5)",        *P.make_native_neg(5)))
    # Comparisons (issue #76) — collapse to IndicatorPoly tops under
    # the gated-bilinear-form treatment. These rows used to land in
    # ``blocked_opcode``; after M2 they classify as ``collapsed``.
    entries.append(CatalogEntry(
        "compare_lt_s(3,5)",
        *P.make_compare_binary(isa.OP_LT_S, 3, 5),
    ))
    entries.append(CatalogEntry(
        "compare_eqz(0)",
        *P.make_compare_eqz(0),
    ))
    # Bit-vector fragment (issue #77) — collapse to BitVec AST tops
    # under the bilinear-form treatment. These rows used to land in
    # ``blocked_opcode``; after M3 they classify as ``collapsed``
    # (hybrid arithmetic like ``log2_floor`` lifts into the AST, and
    # the ``is_power_of_2`` comparison wraps its BitVec in an
    # IndicatorPoly).
    entries.append(CatalogEntry(
        "bitwise_and(12,10)",
        *P.make_bitwise_binary(isa.OP_AND, 12, 10),
    ))
    entries.append(CatalogEntry(
        "bitwise_or(12,10)",
        *P.make_bitwise_binary(isa.OP_OR, 12, 10),
    ))
    entries.append(CatalogEntry(
        "bitwise_xor(12,10)",
        *P.make_bitwise_binary(isa.OP_XOR, 12, 10),
    ))
    entries.append(CatalogEntry("native_clz(16)",    *P.make_native_clz(16)))
    entries.append(CatalogEntry("native_ctz(8)",     *P.make_native_ctz(8)))
    entries.append(CatalogEntry("native_popcnt(13)", *P.make_native_popcnt(13)))
    entries.append(CatalogEntry("bit_extract(5,0)",  *P.make_bit_extract(5, 0)))
    entries.append(CatalogEntry("log2_floor(8)",     *P.make_log2_floor(8)))
    entries.append(CatalogEntry("is_power_of_2(8)",  *P.make_is_power_of_2(8)))

    # Bounded loops — unroll cleanly at concrete inputs (issue #70).
    entries.append(CatalogEntry("fibonacci(5)",    *P.make_fibonacci(5)))
    entries.append(CatalogEntry("factorial(4)",    *P.make_factorial(4)))
    entries.append(CatalogEntry("is_even(6)",      *P.make_is_even(6)))
    entries.append(CatalogEntry("power_of_2(4)",   *P.make_power_of_2(4)))
    # Bit-vector loop (issue #77) — concrete-mode unroll with bit ops.
    entries.append(CatalogEntry("popcount_loop(5)", *P.make_popcount_loop(5)))

    # Closed-form loops (issue #89) — symbolic-counter rows that
    # classify as ``collapsed_closed_form`` via the recurrence solver.
    # Parallel to the concrete-counter rows above; same bytecode but
    # tagged ``solve_recurrences=True`` so the forking executor keeps
    # ``n`` symbolic and emits a Poly / ClosedForm / ProductForm top
    # instead of falling back to concrete-mode unrolling.
    entries.append(CatalogEntry(
        "sum_1_to_n_sym(n)", *P.make_sum_1_to_n_sym(5),
        solve_recurrences=True,
    ))
    entries.append(CatalogEntry(
        "power_of_2_sym(n)", *P.make_power_of_2_sym(4),
        solve_recurrences=True,
    ))
    entries.append(CatalogEntry(
        "fibonacci_sym(n)", *P.make_fibonacci_sym(5),
        solve_recurrences=True,
        # Issue #109 motivator: fibonacci_sym is the row that closes
        # Path B's gate per #107. Blog #82's "weights = computer"
        # claim is strongest when it can point at fibonacci's Binet
        # realisation (B.3) — a single-layer algebraic-coefficient
        # gadget that evaluates F(n) for any n without an explicit
        # recurrence loop. The other three closed_form rows stay at
        # ``solver_structural`` until their own motivator appears.
        requests_weight_layer=True,
    ))
    entries.append(CatalogEntry(
        "factorial_sym(n)", *P.make_factorial_sym(4),
        solve_recurrences=True,
    ))

    # Pure finite conditionals — collapse to guarded polys (issue #70).
    entries.append(CatalogEntry("select_by_sign(7)", *P.make_select_by_sign(7)))
    entries.append(CatalogEntry("clamp_zero(5)",     *P.make_clamp_zero(5)))
    entries.append(CatalogEntry("either_or(3,7,1)",  *P.make_either_or(3, 7, 1)))

    # Guarded comparison + dispatch (issue #76) — GT_S followed by JZ
    # hoists the comparison's relation into the Guard pair, producing
    # a two-case GuardedPoly rather than blocking on the opcode.
    entries.append(CatalogEntry("native_max(3,5)", *P.make_native_max(3, 5)))

    # Generated entries (issue #96): poly → program → symbolic round-trip.
    entries.extend(_generated_catalog())

    return entries



def _generated_catalog() -> List[CatalogEntry]:
    """Generated stress-test entries for issue #96.

    Each entry compiles a hand-chosen :class:`Poly` via
    :func:`poly_to_program` and is expected to classify as
    ``STATUS_COLLAPSED`` with ``numeric_match=True``.

    All generated programs use ``PUSH 0; LOCAL_SET i`` for each variable,
    so the symbolic executor binds every variable to 0.  The six shapes
    cover the stress-test matrix in the issue spec:

    * High degree, single variable — deep DUP/MUL chains (x0⁵)
    * Many variables — wide PUSH + local stack (x0·x1·x2·x3)
    * Many monomials — exercises ADD chains (x0² + x0·x1 + x1² + x0 + x1)
    * Mixed signs — negation trick (3·x0·x1 − 2·x0² + x1)
    * Large coefficient — long DUP/ADD scaling (10·x0)
    * Four-variable cross-product — contiguous-index coverage
      (x0·x1 + x2·x3, replacing the non-contiguous x0·x3 sketch
      from the issue which fails the contiguity validator)
    """
    from poly_compiler import poly_to_program  # sibling module in this repo

    def _entry(name: str, poly: Poly) -> CatalogEntry:
        prog = poly_to_program(poly)
        # poly evaluates to 0 at the binding point (all vars bound to 0
        # by the PUSH 0 preamble), so expected=0 is the ground-truth result.
        return CatalogEntry(name=name, prog=prog, expected=0)

    x0, x1, x2, x3 = (Poly.variable(i) for i in range(4))

    return [
        # Shape 1: high degree, 1 var — deep DUP/MUL chains
        _entry("gen_x0_fifth", x0 * x0 * x0 * x0 * x0),
        # Shape 2: many variables — wide PUSH + local slots
        _entry("gen_x0x1x2x3", x0 * x1 * x2 * x3),
        # Shape 3: many monomials — exercises ADD chains
        _entry("gen_sum_of_five_terms", x0 * x0 + x0 * x1 + x1 * x1 + x0 + x1),
        # Shape 4: mixed signs — negation trick exercises emit_negate
        _entry(
            "gen_mixed_signs",
            Poly({((0, 1), (1, 1)): 3, ((0, 2),): -2, ((1, 1),): 1}),
        ),
        # Shape 5: large coefficient — long DUP/ADD scaling chain
        _entry("gen_10x0", Poly({((0, 1),): 10})),
        # Shape 6: four-variable cross-product — covers contiguous indices 0-3
        # (x0·x3 alone would fail the contiguity validator, so we include
        # all four indices via x0·x1 + x2·x3)
        _entry("gen_x0x1_plus_x2x3", x0 * x1 + x2 * x3),
    ]


# ─── Runner + row ────────────────────────────────────────────────

@dataclass
class GuardedCaseEML:
    """Per-case EML accounting for one branch of a :class:`GuardedPoly`.

    ``value_size`` / ``value_depth`` describe the EML tree compiled from
    the case's ``value_poly``. ``guard_sizes`` / ``guard_depths`` describe
    the EML trees compiled from each :class:`Guard` in the case's
    conjunction (one entry per guard, in the order the executor produced
    them). Populated only when eml-sr is importable; ``None`` otherwise.
    """
    value_size: int
    value_depth: int
    guard_sizes: List[int]
    guard_depths: List[int]


@dataclass
class CatalogRow:
    name: str
    status: str
    blocker: Optional[str] = None
    n_heads: int = 0
    n_monomials: Optional[int] = None
    poly_expr: Optional[str] = None
    eml_size: Optional[int] = None           # sum of value-poly sizes for guarded
    eml_depth: Optional[int] = None          # max of value-poly depths for guarded
    numpy_top: Optional[int] = None
    numeric_match: Optional[bool] = None
    # Guarded-specific fields (issue #70).
    n_cases: Optional[int] = None
    case_exprs: Optional[List[str]] = None   # ["guard => value", ...]
    # Sharper guarded accounting (issue #68 S2).
    eml_guard_size: Optional[int] = None     # sum of guard-poly sizes across every case
    eml_guard_depth: Optional[int] = None    # max guard-poly depth across every case
    case_eml: Optional[List[GuardedCaseEML]] = None
    # Rational top flag (issue #75) — True for programs that collapse to
    # a RationalPoly / SymbolicRemainder rather than a pure Poly. eml_* /
    # n_monomials fields stay ``None`` for these rows since eml-sr has no
    # division primitive; ``poly_expr`` renders the rational form.
    is_rational: bool = False
    # Indicator top flag (issue #76) — True for programs that collapse
    # to an IndicatorPoly (gated bilinear form: linear diff + relation
    # gate). eml_* columns stay ``None`` since the single-operator EML
    # family has no sign primitive; the gate lives at the FF-dispatch
    # boundary rather than inside the polynomial.
    is_indicator: bool = False
    # BitVec top flag (issue #77) — True for programs that collapse to a
    # :class:`BitVec` AST (bit-vector fragment). eml_* columns stay
    # ``None`` since the single-operator EML family has no bitwise
    # primitive; the bit ops fire at the FF-dispatch boundary via
    # ``M_BITBIN`` / ``M_BITUN``. ``n_monomials`` reports the AST's
    # node count as a complexity proxy.
    is_bitvec: bool = False
    # Closed-form top flag (issue #89) — True for programs whose loop
    # closed via the Tier 2 :class:`ClosedForm` or Tier 3
    # :class:`ProductForm` path. Tier 1 rows (Poly top) use the
    # regular polynomial filler and leave this flag False. eml_*
    # columns stay ``None`` since matrix exponentiation / bounded
    # products aren't expressible in the single-operator EML family;
    # the closed form evaluates numerically at binding time instead.
    is_closed_form: bool = False
    # FF-equivalence claim strength (issue #90). Three values:
    #   ``"bilinear"`` — the FF layer's weight matrices (M_ADD / M_SUB /
    #     B_MUL / M_DIV_S / M_REM_S / M_CMP / M_EQZ / M_BITBIN / M_BITUN)
    #     compose to realise the symbolic top structurally. Covers every
    #     collapsed / guarded / unrolled row plus Tier 1
    #     ``collapsed_closed_form`` (Poly emitted by Faulhaber).
    #   ``"solver_structural"`` — the recurrence solver emits a sibling
    #     value-equal across drivers, but forward-time ``eval_at``
    #     (matrix power / bounded product) is the non-polynomial boundary
    #     step and has no weight-layer realisation. Tier 2 / Tier 3
    #     ``collapsed_closed_form`` rows only.
    #   ``"n/a"`` — row didn't collapse; no FF-equivalence claim to make.
    # See ``dev/ff_closed_form_equivalence.md``.
    ff_equiv: Optional[str] = None


def _numpy_top(np_exec: NumPyExecutor, prog) -> Optional[int]:
    try:
        trace = np_exec.execute(prog)
    except Exception:  # trap / underflow — don't block the row
        return None
    return trace.steps[-1].top if trace.steps else None


def _guard_to_expr(guard: Guard) -> str:
    """Human-readable rendering of a single Guard (poly relation 0)."""
    sym = _REL_SYMBOL[guard.relation]
    return f"{poly_to_expr(guard.poly)} {sym} 0"


def _guards_to_expr(guards) -> str:
    if not guards:
        return "True"
    return " ∧ ".join(_guard_to_expr(g) for g in guards)


def _eval_poly_safe(p: Poly, bindings: Dict[int, int]) -> int:
    """Evaluate a Poly. Missing variables default to 0 (concrete-mode rows)."""
    try:
        return p.eval_at(bindings)
    except KeyError:
        missing = {v: 0 for v in p.variables() if v not in bindings}
        return p.eval_at({**bindings, **missing})


def _compile_poly(poly: Poly):
    """Compile a Poly to an EML tree. Returns (tree, var_names)."""
    expr = poly_to_expr(poly)
    vars_in_poly = poly.variables()
    if vars_in_poly:
        return compile_expr(expr, variables=[f"x{v}" for v in vars_in_poly]), vars_in_poly
    return compile_expr(expr), []


def _three_way_numeric_match(numpy_top, sym_val, eml_val) -> bool:
    if eml_val is None:
        return numpy_top == sym_val
    if abs(eml_val.imag) > 1e-6:
        return False
    return numpy_top == sym_val == int(round(eml_val.real))


def run_catalog(entries: Optional[List[CatalogEntry]] = None, *,
                np_exec: Optional[NumPyExecutor] = None) -> List[CatalogRow]:
    """Classify every entry, populate EML columns where possible.

    For collapsed / guarded / unrolled rows the three-way cross-check is
    ``NumPy top == Poly.eval_at(bindings) == eval_eml(compiled_tree)`` —
    ``numeric_match`` is True only when all three agree. For guarded
    rows the check runs per-case on each case's `value_poly`, restricted
    to the concrete bindings that satisfy that case's guards.
    """
    entries = entries if entries is not None else _default_catalog()
    np_exec = np_exec or NumPyExecutor()
    rows: List[CatalogRow] = []
    for entry in entries:
        row = CatalogRow(name=entry.name, status="")
        cr = classify_program(entry.prog,
                              solve_recurrences=entry.solve_recurrences)
        row.status = cr.status
        row.blocker = cr.blocker
        row.n_heads = cr.n_heads
        row.numpy_top = _numpy_top(np_exec, entry.prog)

        if cr.status == STATUS_COLLAPSED or cr.status == STATUS_COLLAPSED_UNROLLED:
            if cr.indicator is not None:
                _fill_indicator_row(row, cr.indicator, cr.bindings, row.numpy_top)
            elif cr.bitvec is not None:
                _fill_bitvec_row(row, cr.bitvec, cr.bindings, row.numpy_top)
            elif cr.rational is not None:
                _fill_rational_row(row, cr.rational, cr.bindings, row.numpy_top)
            elif cr.poly is not None:
                _fill_poly_row(row, cr.poly, cr.bindings, row.numpy_top)
            elif cr.guarded is not None:
                # Unrolled + guarded (rare) — fall through to guarded formatter
                _fill_guarded_row(row, cr.guarded, cr.bindings, row.numpy_top)
        elif cr.status == STATUS_COLLAPSED_GUARDED:
            assert cr.guarded is not None
            _fill_guarded_row(row, cr.guarded, cr.bindings, row.numpy_top)
        elif cr.status == STATUS_COLLAPSED_CLOSED_FORM:
            # Issue #89. Tier 1 lands as a Poly via ``cr.poly``;
            # Tier 2 / Tier 3 populate ``cr.closed_form``. Both paths
            # flip ``is_closed_form`` so the report column is
            # consistent, even though Tier 1's top IS a plain Poly.
            if cr.poly is not None:
                _fill_poly_row(row, cr.poly, cr.bindings, row.numpy_top)
                row.is_closed_form = True
            elif cr.closed_form is not None:
                _fill_closed_form_row(row, cr.closed_form, cr.bindings,
                                      row.numpy_top)
        # Otherwise: row stays minimally populated (blocker-only).

        # FF-equivalence tag (issue #90). See the CatalogRow field docs
        # + dev/ff_closed_form_equivalence.md for the full taxonomy.
        if row.status in (STATUS_COLLAPSED, STATUS_COLLAPSED_GUARDED,
                          STATUS_COLLAPSED_UNROLLED):
            row.ff_equiv = "bilinear"
        elif row.status == STATUS_COLLAPSED_CLOSED_FORM:
            # Tier 2 / Tier 3 siblings live in ``cr.closed_form``;
            # Tier 1 (Poly) rides the existing bilinear-form theorem.
            row.ff_equiv = ("solver_structural"
                            if cr.closed_form is not None else "bilinear")
        else:
            row.ff_equiv = "n/a"

        # Issue #109 / Path B upgrade. A row is upgraded from
        # ``solver_structural`` to ``bilinear_weight_layer`` when:
        #   1. The CatalogEntry sets ``requests_weight_layer=True`` —
        #      the motivator gate per #107.
        #   2. The row actually classifies as ``collapsed_closed_form``
        #      (Tier 2 / Tier 3 — the only rows where ``solver_structural``
        #      applies in the first place).
        #   3. ``path_b_in_scope`` says some Path B sub-path covers it.
        # Tier 1 rows (already ``bilinear``) are NOT demoted — the upgrade
        # is one-directional. Path A's claim stands; Path B adds to it.
        if (entry.requests_weight_layer
                and row.ff_equiv == "solver_structural"):
            from path_b import path_b_in_scope
            # A scope query without a specific n: ask "is there ANY n
            # this path covers." For B.2 (the universal sub-path) the
            # answer is True for any catalog row whose top is
            # ClosedForm / ProductForm.
            if path_b_in_scope(_strip_arity(row.name), 1):
                row.ff_equiv = "bilinear_weight_layer"

        rows.append(row)
    return rows


def _strip_arity(name: str) -> str:
    """Map ``"fibonacci_sym(n)"`` → ``"fibonacci_sym"`` for scope lookup.

    The catalog labels rows with their arity for human readability;
    ``path_b_in_scope`` keys on the bare row name.
    """
    return name.split("(", 1)[0]


def _fill_poly_row(row: CatalogRow, poly: Poly,
                   bindings: Dict[int, int],
                   numpy_top: Optional[int]) -> None:
    """Populate a CatalogRow from a single Poly top (collapsed / unrolled)."""
    row.n_monomials = poly.n_monomials()
    row.poly_expr = poly_to_expr(poly)

    sym_val = _eval_poly_safe(poly, bindings)
    eml_val = None
    if _EML_AVAILABLE:
        tree, _ = _compile_poly(poly)
        row.eml_size = tree_size(tree)
        row.eml_depth = tree_depth(tree)
        eml_bindings = {f"x{v}": bindings[v] for v in poly.variables()
                        if v in bindings}
        eml_val = eval_eml(tree, eml_bindings) if eml_bindings or not poly.variables() else None
    row.numeric_match = _three_way_numeric_match(numpy_top, sym_val, eml_val)


def _fill_rational_row(row: CatalogRow, rational,
                       bindings: Dict[int, int],
                       numpy_top: Optional[int]) -> None:
    """Populate a CatalogRow from a rational top (issue #75).

    ``rational`` is a :class:`RationalPoly` (DIV_S) or
    :class:`SymbolicRemainder` (REM_S). Renders as ``"num /ₜ denom"`` /
    ``"num modₜ denom"``; evaluates via the wrapper's ``eval_at`` so
    truncation toward zero (WASM ``i32.div_s`` / ``i32.rem_s`` semantics)
    is applied at the boundary. eml-sr columns stay ``None`` — elementary
    expressions in the single ``eml(x, y) = exp(x) - ln(y)`` operator
    have no division primitive.
    """
    row.is_rational = True
    row.poly_expr = repr(rational)
    # n_monomials summarises the expression complexity: sum of the two
    # underlying Polys' term counts, giving a meaningful size proxy even
    # without an EML tree.
    row.n_monomials = rational.num.n_monomials() + rational.denom.n_monomials()

    try:
        sym_val = rational.eval_at(bindings) if bindings else rational.eval_at({})
    except (KeyError, ValueError, ZeroDivisionError):
        sym_val = None

    if sym_val is None or numpy_top is None:
        row.numeric_match = None
    else:
        row.numeric_match = (sym_val == numpy_top)


def _fill_indicator_row(row: CatalogRow, indicator: IndicatorPoly,
                        bindings: Dict[int, int],
                        numpy_top: Optional[int]) -> None:
    """Populate a CatalogRow from an IndicatorPoly top (issue #76).

    ``indicator`` is a gated bilinear form: a linear ``Poly`` diff
    (``vb - va`` for binary comparisons, the value itself for EQZ) plus a
    relation tag. Renders as ``[poly <rel> 0]`` and evaluates via
    ``indicator.eval_at`` so the non-polynomial gate lands at the
    boundary. eml-sr columns stay ``None`` — the single-operator family
    has no sign primitive; the gate lives in the FF dispatch, not the
    polynomial algebra.

    ``n_monomials`` reports the term count of the underlying linear
    diff polynomial — a meaningful complexity proxy even without a
    value-EML tree. ``poly_expr`` carries the full bracketed indicator.
    """
    row.is_indicator = True
    row.poly_expr = repr(indicator)
    # indicator.poly may be a Poly or (issue #77) a BitVec — count
    # monomials for the former, AST nodes for the latter.
    if isinstance(indicator.poly, BitVec):
        row.n_monomials = _bitvec_node_count(indicator.poly)
    else:
        row.n_monomials = indicator.poly.n_monomials()

    try:
        sym_val = indicator.eval_at(bindings) if bindings else indicator.eval_at({})
    except KeyError:
        sym_val = None

    if sym_val is None or numpy_top is None:
        row.numeric_match = None
    else:
        row.numeric_match = (sym_val == numpy_top)


def _bitvec_node_count(node) -> int:
    """Recursively count nodes in a BitVec / Poly AST."""
    if isinstance(node, BitVec):
        return 1 + sum(_bitvec_node_count(o) for o in node.operands)
    if isinstance(node, Poly):
        return node.n_monomials()
    return 1


def _fill_bitvec_row(row: CatalogRow, bitvec: BitVec,
                     bindings: Dict[int, int],
                     numpy_top: Optional[int]) -> None:
    """Populate a CatalogRow from a BitVec top (issue #77).

    ``bitvec`` is a recursive AST over the bit-vector fragment
    (``AND / OR / XOR / SHL / SHR_S / SHR_U / CLZ / CTZ / POPCNT``) plus
    lifted arithmetic (``ADD / SUB / MUL`` with a :class:`BitVec`
    operand). Renders via ``repr`` (e.g. ``(x0 & x1)`` or ``CLZ(x0)``)
    and evaluates via :meth:`BitVec.eval_at`, which recursively reduces
    every operand to a concrete int and applies :func:`_apply_bitop`.
    The non-polynomial bit op fires only at that boundary, matching the
    :class:`RationalPoly` / :class:`IndicatorPoly` design.

    eml-sr columns stay ``None`` — the single-operator EML family has
    no bitwise primitive; bit ops live at the FF-dispatch boundary via
    ``M_BITBIN`` / ``M_BITUN``. ``n_monomials`` reports the AST's node
    count as a complexity proxy.
    """
    row.is_bitvec = True
    row.poly_expr = repr(bitvec)
    row.n_monomials = _bitvec_node_count(bitvec)

    try:
        sym_val = bitvec.eval_at(bindings) if bindings else bitvec.eval_at({})
    except (KeyError, ValueError):
        sym_val = None

    if sym_val is None or numpy_top is None:
        row.numeric_match = None
    else:
        row.numeric_match = (sym_val == numpy_top)


def _fill_closed_form_row(row: CatalogRow, closed_form,
                          bindings: Dict[int, int],
                          numpy_top: Optional[int]) -> None:
    """Populate a CatalogRow from a Tier 2 :class:`ClosedForm` or Tier 3
    :class:`ProductForm` top (issue #89).

    Renders via ``repr`` and evaluates via ``closed_form.eval_at`` so
    the matrix exponentiation (Tier 2) or bounded product (Tier 3)
    fires only at the binding boundary — same contract as
    :class:`RationalPoly` / :class:`IndicatorPoly` / :class:`BitVec`
    since neither is a polynomial in the input variables. eml-sr
    columns stay ``None`` — the single-operator EML family has no
    matrix-power or bounded-product primitive; the non-polynomial
    step lives at the FF-dispatch boundary, not inside the algebra.

    ``n_monomials`` is repurposed as a structural-size proxy: the
    matrix dimension ``m`` for ClosedForm or the number of distinct
    counter powers in ``p`` for ProductForm.
    """
    row.is_closed_form = True
    row.poly_expr = repr(closed_form)
    if isinstance(closed_form, ClosedForm):
        row.n_monomials = len(closed_form.A)
    elif isinstance(closed_form, ProductForm):
        row.n_monomials = closed_form.p.n_monomials()
    else:
        row.n_monomials = None

    try:
        sym_val = closed_form.eval_at(bindings) if bindings else None
    except (KeyError, ValueError, ZeroDivisionError):
        sym_val = None

    if sym_val is None or numpy_top is None:
        row.numeric_match = None
    else:
        row.numeric_match = (int(sym_val) == int(numpy_top))


def _fill_guarded_row(row: CatalogRow, guarded: GuardedPoly,
                      bindings: Dict[int, int],
                      numpy_top: Optional[int]) -> None:
    """Populate a CatalogRow from a GuardedPoly (collapsed_guarded).

    For each case we compile both the ``value_poly`` and every
    :class:`Guard` poly in its conjunction to their own EML trees. The
    row's ``eml_size`` / ``eml_depth`` continue to describe the **value**
    trees (sum / max) — unchanged since S1 — while ``eml_guard_size`` /
    ``eml_guard_depth`` describe the **guard** trees (sum / max), making
    the dispatch cost explicit rather than free. ``case_eml`` carries the
    per-case breakdown for callers that want to see individual numbers.
    """
    row.n_cases = guarded.n_cases()
    row.n_monomials = sum(v.n_monomials() for _, v in guarded.cases)
    row.case_exprs = [
        f"{{{_guards_to_expr(gs)}}} → {poly_to_expr(v)}"
        for gs, v in guarded.cases
    ]
    row.poly_expr = " ; ".join(row.case_exprs)

    # Per-case three-way check. Exactly one case's guards hold at the
    # concrete bindings we have.
    sym_val = guarded.eval_at(bindings) if bindings else None

    sizes: List[int] = []
    depths: List[int] = []
    guard_sizes_total: List[int] = []
    guard_depths_total: List[int] = []
    eml_vals_per_case = []
    case_eml: List[GuardedCaseEML] = []
    if _EML_AVAILABLE:
        for gs, v in guarded.cases:
            tree, _ = _compile_poly(v)
            v_size = tree_size(tree)
            v_depth = tree_depth(tree)
            sizes.append(v_size)
            depths.append(v_depth)

            # Compile every guard poly in this case's conjunction.
            per_case_gsizes: List[int] = []
            per_case_gdepths: List[int] = []
            for g in gs:
                g_tree, _ = _compile_poly(g.poly)
                per_case_gsizes.append(tree_size(g_tree))
                per_case_gdepths.append(tree_depth(g_tree))
            guard_sizes_total.extend(per_case_gsizes)
            guard_depths_total.extend(per_case_gdepths)

            case_eml.append(GuardedCaseEML(
                value_size=v_size, value_depth=v_depth,
                guard_sizes=per_case_gsizes,
                guard_depths=per_case_gdepths,
            ))

            case_bindings = {f"x{i}": bindings[i] for i in v.variables()
                             if i in bindings}
            if v.variables() and not case_bindings:
                eml_vals_per_case.append(None)
            else:
                eml_vals_per_case.append(eval_eml(tree, case_bindings))
        row.eml_size = sum(sizes)
        row.eml_depth = max(depths) if depths else 0
        row.eml_guard_size = sum(guard_sizes_total)
        row.eml_guard_depth = max(guard_depths_total) if guard_depths_total else 0
        row.case_eml = case_eml

    # numeric_match: for each case pick the one whose guards hold;
    # compare numpy_top, that case's poly eval, and that case's eml eval.
    match = True
    hit = False
    for idx, (gs, v) in enumerate(guarded.cases):
        if not bindings:
            continue
        ok = True
        for g in gs:
            gv = _eval_poly_safe(g.poly, bindings)
            if not _relation_holds(g.relation, gv):
                ok = False; break
        if not ok:
            continue
        hit = True
        case_sym = _eval_poly_safe(v, bindings)
        case_eml = eml_vals_per_case[idx] if _EML_AVAILABLE else None
        if not _three_way_numeric_match(numpy_top, case_sym, case_eml):
            match = False
        break
    row.numeric_match = match and hit if bindings else None


# ─── Reporting ────────────────────────────────────────────────────

_BLOCK_REASON = {
    STATUS_BLOCKED_OPCODE:    "non-polynomial op",
    STATUS_BLOCKED_UNDERFLOW: "stack underflow",
    STATUS_BLOCKED_LOOP_SYM:  "loop with symbolic trip count",
    STATUS_BLOCKED_PATH_EXP:  "path explosion",
}


def _fmt_eml(n):
    return "–" if n is None else str(n)


def _fmt_match(m):
    return "–" if m is None else ("✓" if m else "✗")


def format_report(rows: List[CatalogRow]) -> str:
    """Render a markdown-style catalog report from :func:`run_catalog` rows."""
    n_collapsed = sum(1 for r in rows if r.status == STATUS_COLLAPSED)
    n_guarded = sum(1 for r in rows if r.status == STATUS_COLLAPSED_GUARDED)
    n_unrolled = sum(1 for r in rows if r.status == STATUS_COLLAPSED_UNROLLED)
    n_closed = sum(1 for r in rows if r.status == STATUS_COLLAPSED_CLOSED_FORM)
    n_loop_sym = sum(1 for r in rows if r.status == STATUS_BLOCKED_LOOP_SYM)
    n_opcode = sum(1 for r in rows if r.status == STATUS_BLOCKED_OPCODE)
    n_other = (len(rows) - n_collapsed - n_guarded - n_unrolled - n_closed
               - n_loop_sym - n_opcode)

    summary_parts = [
        f"{n_collapsed} collapsed",
        f"{n_guarded} guarded",
        f"{n_unrolled} unrolled",
        f"{n_closed} closed-form",
        f"{n_loop_sym} loop-symbolic",
        f"{n_opcode} blocked-by-opcode",
    ]
    if n_other:
        summary_parts.append(f"{n_other} other")

    lines = [
        "# LAC program catalog — symbolic collapse report",
        "",
        "_" + " | ".join(summary_parts) + f" (total {len(rows)})._",
        "",
        "**Reading the status columns.** _Collapsed_ rows are straight-line",
        "programs that reduce to a single polynomial (the issue-#65 claim).",
        "_Guarded_ rows contain finite conditionals (JZ/JNZ on symbolic",
        "inputs) and reduce to a `GuardedPoly` — a partitioned case table",
        "whose cases together cover the domain. _Unrolled_ rows contain",
        "bounded loops with concrete trip counts: the executor runs them",
        "in `input_mode=\"concrete\"` (every PUSH is specialised to its",
        "literal arg) so the loop unrolls by execution rather than by",
        "invariant inference. \"Unrolled at n=5\" is therefore a claim",
        "about a specific input, **not** a symbolic proof over all n.",
        "",
    ]
    if not _EML_AVAILABLE:
        lines += ["_eml-sr not on PYTHONPATH — eml tree columns show `–`._", ""]

    lines += [
        "## Collapsed (branchless, polynomial-closed)",
        "",
        "| Program | k heads | # mono | poly | eml size | eml depth | match |",
        "|---|---:|---:|---|---:|---:|:-:|",
    ]
    for r in rows:
        if r.status != STATUS_COLLAPSED:
            continue
        lines.append(
            f"| `{r.name}` | {r.n_heads} | {r.n_monomials} | "
            f"`{r.poly_expr}` | {_fmt_eml(r.eml_size)} | "
            f"{_fmt_eml(r.eml_depth)} | {_fmt_match(r.numeric_match)} |"
        )

    lines += [
        "",
        "## Collapsed (guarded — finite conditionals)",
        "",
        "Each case is `guards ⇒ value_poly`. Guarded dispatch has two",
        "EML costs: the **value** trees (one per case's `value_poly`)",
        "and the **guard** trees (one per `Guard` in every case's",
        "conjunction). Both are reported separately rather than rolled",
        "together — so the \"what does one execution cost?\" number and",
        "the \"what does it take to realise the whole case table?\"",
        "number stay distinguishable. _value Σ size_ / _value max depth_",
        "sum and max across cases' value trees; _guard Σ size_ / _guard",
        "max depth_ do the same across every guard tree.",
        "",
        "| Program | k heads | # cases | cases | value Σ size | value max depth | guard Σ size | guard max depth | match |",
        "|---|---:|---:|---|---:|---:|---:|---:|:-:|",
    ]
    for r in rows:
        if r.status != STATUS_COLLAPSED_GUARDED:
            continue
        cases_md = "<br>".join(f"`{c}`" for c in (r.case_exprs or []))
        lines.append(
            f"| `{r.name}` | {r.n_heads} | {r.n_cases} | {cases_md} | "
            f"{_fmt_eml(r.eml_size)} | {_fmt_eml(r.eml_depth)} | "
            f"{_fmt_eml(r.eml_guard_size)} | {_fmt_eml(r.eml_guard_depth)} | "
            f"{_fmt_match(r.numeric_match)} |"
        )

    lines += [
        "",
        "## Collapsed (unrolled at the catalog's concrete inputs)",
        "",
        "| Program | k heads | # mono | poly | eml size | eml depth | match |",
        "|---|---:|---:|---|---:|---:|:-:|",
    ]
    for r in rows:
        if r.status != STATUS_COLLAPSED_UNROLLED:
            continue
        lines.append(
            f"| `{r.name}` | {r.n_heads} | {r.n_monomials} | "
            f"`{r.poly_expr}` | {_fmt_eml(r.eml_size)} | "
            f"{_fmt_eml(r.eml_depth)} | {_fmt_match(r.numeric_match)} |"
        )

    lines += [
        "",
        "## Collapsed (closed form from symbolic loop — issue #89)",
        "",
        "Rows whose loop body is an affine / linear / multiplicative",
        "recurrence on the loop-carried stack slice. The recurrence",
        "solver emits a `Poly` (Tier 1, via Faulhaber), `ClosedForm`",
        "(Tier 2, constant integer matrix), or `ProductForm` (Tier 3,",
        "bounded product of a Poly factor). Unlike the _unrolled_ rows",
        "above, this is a **symbolic proof** that holds at every `n`,",
        "not a single-input execution trace. eml-sr columns stay `–`",
        "— matrix power and bounded products aren't expressible in the",
        "single-operator EML family.",
        "",
        "The `FF equiv` column (issue #90) records the strength of the",
        "FF-layer equivalence claim: `bilinear` for Tier 1 (the solver",
        "emits a Poly realised by composed M_ADD / B_MUL);",
        "`solver_structural` for Tier 2 / Tier 3 (the emitted",
        "ClosedForm / ProductForm is structurally equal across drivers,",
        "but forward-time `eval_at` — matrix power / bounded product —",
        "has no weight-layer realisation). See",
        "`dev/ff_closed_form_equivalence.md`.",
        "",
        "| Program | k heads | size | closed form | FF equiv | match |",
        "|---|---:|---:|---|:-:|:-:|",
    ]
    for r in rows:
        if r.status != STATUS_COLLAPSED_CLOSED_FORM:
            continue
        size = r.n_monomials if r.n_monomials is not None else "–"
        lines.append(
            f"| `{r.name}` | {r.n_heads} | {size} | "
            f"`{r.poly_expr}` | `{r.ff_equiv or '–'}` | "
            f"{_fmt_match(r.numeric_match)} |"
        )

    lines += [
        "",
        "## Blocked (out of symbolic-executor scope)",
        "",
        "| Program | reason | blocker |",
        "|---|---|---|",
    ]
    for r in rows:
        if r.status in (STATUS_COLLAPSED, STATUS_COLLAPSED_GUARDED,
                         STATUS_COLLAPSED_UNROLLED,
                         STATUS_COLLAPSED_CLOSED_FORM) or r.status == "":
            continue
        reason = _BLOCK_REASON.get(r.status, r.status)
        blocker = r.blocker if r.blocker else "–"
        lines.append(f"| `{r.name}` | {reason} | `{blocker}` |")

    return "\n".join(lines) + "\n"


# ─── CLI ──────────────────────────────────────────────────────────

def main() -> int:
    rows = run_catalog()
    print(format_report(rows))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(main())


__all__ = [
    "CatalogEntry",
    "CatalogRow",
    "ClassificationResult",
    "GuardedCaseEML",
    "STATUS_COLLAPSED",
    "STATUS_COLLAPSED_GUARDED",
    "STATUS_COLLAPSED_UNROLLED",
    "STATUS_COLLAPSED_CLOSED_FORM",
    "STATUS_BLOCKED_OPCODE",
    "STATUS_BLOCKED_UNDERFLOW",
    "STATUS_BLOCKED_LOOP_SYM",
    "STATUS_BLOCKED_PATH_EXP",
    "classify_program",
    "format_report",
    "poly_to_expr",
    "run_catalog",
    "_generated_catalog",
]
