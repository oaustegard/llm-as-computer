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
    ForkingResult,
    Guard,
    GuardedPoly,
    Poly,
    SymbolicOpNotSupported,
    SymbolicStackUnderflow,
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
    isa.OP_SWAP, isa.OP_OVER, isa.OP_ROT, isa.OP_NOP,
}
# Control flow — handled by the forking executor (issue #70).
_BRANCH_OPS = {isa.OP_JZ, isa.OP_JNZ}
# The forking executor accepts this full set.
_FORKING_OPS = _POLY_OPS | _BRANCH_OPS


# Status codes emitted by classify_program.
STATUS_COLLAPSED          = "collapsed"
STATUS_COLLAPSED_GUARDED  = "collapsed_guarded"
STATUS_COLLAPSED_UNROLLED = "collapsed_unrolled"
STATUS_BLOCKED_OPCODE     = "blocked_opcode"
STATUS_BLOCKED_UNDERFLOW  = "blocked_underflow"
STATUS_BLOCKED_LOOP_SYM   = "blocked_loop_symbolic"
STATUS_BLOCKED_PATH_EXP   = "blocked_path_explosion"


@dataclass
class ClassificationResult:
    status: str
    blocker: Optional[str] = None
    poly: Optional[Poly] = None           # present when status == collapsed*
    guarded: Optional[GuardedPoly] = None # present when status == collapsed_guarded
    n_heads: int = 0
    bindings: Dict[int, int] = field(default_factory=dict)
    n_cases: int = 0                       # number of cases in guarded output


def _poly_or_guarded(result: ForkingResult):
    """Pull the top out of a ForkingResult, already Poly | GuardedPoly | None."""
    return result.top


def classify_program(prog) -> ClassificationResult:
    """Pre-flight check + forking symbolic execute. First blocker wins.

    Order of checks:
      1. ``blocked_opcode`` — any op outside ``_FORKING_OPS``.
      2. Run :func:`run_forking` in symbolic mode:
         - straight → ``collapsed`` (same semantics as the old executor).
         - guarded → ``collapsed_guarded`` with a GuardedPoly top.
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
        r_sym = run_forking(prog, input_mode="symbolic")
    except SymbolicOpNotSupported as e:
        return ClassificationResult(status=STATUS_BLOCKED_OPCODE, blocker=str(e))
    except SymbolicStackUnderflow as e:
        return ClassificationResult(status=STATUS_BLOCKED_UNDERFLOW, blocker=str(e))

    if r_sym.status == "path_explosion":
        return ClassificationResult(status=STATUS_BLOCKED_PATH_EXP)
    if r_sym.status == "blocked_underflow":
        return ClassificationResult(status=STATUS_BLOCKED_UNDERFLOW)

    if r_sym.status == "straight":
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

    # Blocked — non-polynomial opcodes
    entries.append(CatalogEntry("native_divmod(2,7)",   *P.make_native_divmod(2, 7)))
    entries.append(CatalogEntry("native_clz(16)",       *P.make_native_clz(16)))
    entries.append(CatalogEntry("native_abs_unary(-3)", *P.make_native_abs_unary(-3)))
    entries.append(CatalogEntry("native_neg(5)",        *P.make_native_neg(5)))
    entries.append(CatalogEntry(
        "compare_lt_s(3,5)",
        *P.make_compare_binary(isa.OP_LT_S, 3, 5),
    ))
    entries.append(CatalogEntry(
        "bitwise_and(12,10)",
        *P.make_bitwise_binary(isa.OP_AND, 12, 10),
    ))

    # Bounded loops — unroll cleanly at concrete inputs (issue #70).
    entries.append(CatalogEntry("fibonacci(5)",  *P.make_fibonacci(5)))
    entries.append(CatalogEntry("factorial(4)",  *P.make_factorial(4)))
    entries.append(CatalogEntry("is_even(6)",    *P.make_is_even(6)))
    entries.append(CatalogEntry("power_of_2(4)", *P.make_power_of_2(4)))

    # Pure finite conditionals — collapse to guarded polys (issue #70).
    entries.append(CatalogEntry("select_by_sign(7)", *P.make_select_by_sign(7)))
    entries.append(CatalogEntry("clamp_zero(5)",     *P.make_clamp_zero(5)))
    entries.append(CatalogEntry("either_or(3,7,1)",  *P.make_either_or(3, 7, 1)))

    # Still blocked — non-polynomial op.
    entries.append(CatalogEntry("native_max(3,5)", *P.make_native_max(3, 5)))

    return entries


# ─── Runner + row ────────────────────────────────────────────────

@dataclass
class CatalogRow:
    name: str
    status: str
    blocker: Optional[str] = None
    n_heads: int = 0
    n_monomials: Optional[int] = None
    poly_expr: Optional[str] = None
    eml_size: Optional[int] = None           # sum across cases for guarded
    eml_depth: Optional[int] = None          # max across cases for guarded
    numpy_top: Optional[int] = None
    numeric_match: Optional[bool] = None
    # Guarded-specific fields (issue #70).
    n_cases: Optional[int] = None
    case_exprs: Optional[List[str]] = None   # ["guard => value", ...]


def _numpy_top(np_exec: NumPyExecutor, prog) -> Optional[int]:
    try:
        trace = np_exec.execute(prog)
    except Exception:  # trap / underflow — don't block the row
        return None
    return trace.steps[-1].top if trace.steps else None


def _guard_to_expr(guard: Guard) -> str:
    """Human-readable rendering of a single Guard (poly op 0)."""
    op = "== 0" if guard.eq_zero else "!= 0"
    return f"{poly_to_expr(guard.poly)} {op}"


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
        cr = classify_program(entry.prog)
        row.status = cr.status
        row.blocker = cr.blocker
        row.n_heads = cr.n_heads
        row.numpy_top = _numpy_top(np_exec, entry.prog)

        if cr.status == STATUS_COLLAPSED or cr.status == STATUS_COLLAPSED_UNROLLED:
            if cr.poly is not None:
                _fill_poly_row(row, cr.poly, cr.bindings, row.numpy_top)
            elif cr.guarded is not None:
                # Unrolled + guarded (rare) — fall through to guarded formatter
                _fill_guarded_row(row, cr.guarded, cr.bindings, row.numpy_top)
        elif cr.status == STATUS_COLLAPSED_GUARDED:
            assert cr.guarded is not None
            _fill_guarded_row(row, cr.guarded, cr.bindings, row.numpy_top)
        # Otherwise: row stays minimally populated (blocker-only).

        rows.append(row)
    return rows


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


def _fill_guarded_row(row: CatalogRow, guarded: GuardedPoly,
                      bindings: Dict[int, int],
                      numpy_top: Optional[int]) -> None:
    """Populate a CatalogRow from a GuardedPoly (collapsed_guarded).

    For each case we compile ``value_poly`` to its own EML tree. The row's
    ``eml_size`` is the sum across cases (honest total work) and
    ``eml_depth`` is the max across cases (honest worst case).
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
    eml_vals_per_case = []
    if _EML_AVAILABLE:
        for gs, v in guarded.cases:
            tree, _ = _compile_poly(v)
            sizes.append(tree_size(tree))
            depths.append(tree_depth(tree))
            case_bindings = {f"x{i}": bindings[i] for i in v.variables()
                             if i in bindings}
            if v.variables() and not case_bindings:
                eml_vals_per_case.append(None)
            else:
                eml_vals_per_case.append(eval_eml(tree, case_bindings))
        row.eml_size = sum(sizes)
        row.eml_depth = max(depths) if depths else 0

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
            if g.eq_zero and gv != 0:
                ok = False; break
            if not g.eq_zero and gv == 0:
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
    n_loop_sym = sum(1 for r in rows if r.status == STATUS_BLOCKED_LOOP_SYM)
    n_opcode = sum(1 for r in rows if r.status == STATUS_BLOCKED_OPCODE)
    n_other = len(rows) - n_collapsed - n_guarded - n_unrolled - n_loop_sym - n_opcode

    summary_parts = [
        f"{n_collapsed} collapsed",
        f"{n_guarded} guarded",
        f"{n_unrolled} unrolled",
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
        "Each case is `guards ⇒ value_poly`. The `eml size` column sums",
        "across cases (total EML nodes to realise every branch); the",
        "`eml depth` column reports the deepest single case.",
        "",
        "| Program | k heads | # cases | cases | eml size | eml depth | match |",
        "|---|---:|---:|---|---:|---:|:-:|",
    ]
    for r in rows:
        if r.status != STATUS_COLLAPSED_GUARDED:
            continue
        cases_md = "<br>".join(f"`{c}`" for c in (r.case_exprs or []))
        lines.append(
            f"| `{r.name}` | {r.n_heads} | {r.n_cases} | {cases_md} | "
            f"{_fmt_eml(r.eml_size)} | {_fmt_eml(r.eml_depth)} | "
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
        "## Blocked (out of symbolic-executor scope)",
        "",
        "| Program | reason | blocker |",
        "|---|---|---|",
    ]
    for r in rows:
        if r.status in (STATUS_COLLAPSED, STATUS_COLLAPSED_GUARDED,
                         STATUS_COLLAPSED_UNROLLED) or r.status == "":
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
    "STATUS_COLLAPSED",
    "STATUS_COLLAPSED_GUARDED",
    "STATUS_COLLAPSED_UNROLLED",
    "STATUS_BLOCKED_OPCODE",
    "STATUS_BLOCKED_UNDERFLOW",
    "STATUS_BLOCKED_LOOP_SYM",
    "STATUS_BLOCKED_PATH_EXP",
    "classify_program",
    "format_report",
    "poly_to_expr",
    "run_catalog",
]
