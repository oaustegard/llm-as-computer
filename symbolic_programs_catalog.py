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
    Poly,
    SymbolicStackUnderflow,
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
# Control flow — symbolic-executor scope excludes these even though the
# taken path would technically evaluate. Reported separately so the
# blocked rows split cleanly into "needs piecewise polys" (control flow)
# vs "out of the polynomial ring" (comparisons, bitwise, division).
_BRANCH_OPS = {isa.OP_JZ, isa.OP_JNZ}


@dataclass
class ClassificationResult:
    status: str                         # "collapsed" | "blocked_opcode"
                                        # | "blocked_control" | "blocked_underflow"
    blocker: Optional[str] = None
    poly: Optional[Poly] = None
    n_heads: int = 0
    bindings: Dict[int, int] = field(default_factory=dict)


def classify_program(prog) -> ClassificationResult:
    """Pre-flight check + symbolic execute. First blocker wins.

    Walks the instruction list once before invoking :func:`run_symbolic` so
    the blocked case reports the distinct ``blocked_control`` label — the
    executor itself only knows "not polynomial-closed".
    """
    for instr in prog:
        op = instr.op
        if op in _BRANCH_OPS:
            return ClassificationResult(
                status="blocked_control",
                blocker=isa.OP_NAMES.get(op, f"?{op}"),
            )
        if op not in _POLY_OPS:
            return ClassificationResult(
                status="blocked_opcode",
                blocker=isa.OP_NAMES.get(op, f"?{op}"),
            )
    try:
        r = run_symbolic(prog)
    except SymbolicStackUnderflow as e:
        return ClassificationResult(status="blocked_underflow", blocker=str(e))
    return ClassificationResult(
        status="collapsed", poly=r.top,
        n_heads=r.n_heads, bindings=r.bindings,
    )


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

    # Blocked — control flow
    entries.append(CatalogEntry("fibonacci(5)",  *P.make_fibonacci(5)))
    entries.append(CatalogEntry("factorial(4)",  *P.make_factorial(4)))
    entries.append(CatalogEntry("is_even(6)",    *P.make_is_even(6)))
    entries.append(CatalogEntry("power_of_2(4)", *P.make_power_of_2(4)))
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
    eml_size: Optional[int] = None
    eml_depth: Optional[int] = None
    numpy_top: Optional[int] = None
    numeric_match: Optional[bool] = None


def _numpy_top(np_exec: NumPyExecutor, prog) -> Optional[int]:
    try:
        trace = np_exec.execute(prog)
    except Exception:  # trap / underflow — don't block the row
        return None
    return trace.steps[-1].top if trace.steps else None


def run_catalog(entries: Optional[List[CatalogEntry]] = None, *,
                np_exec: Optional[NumPyExecutor] = None) -> List[CatalogRow]:
    """Classify every entry, populate EML columns where possible.

    For ``collapsed`` rows the three-way cross-check is
    ``NumPy top == Poly.eval_at(bindings) == eval_eml(compiled_tree)`` —
    ``numeric_match`` is True only when all three agree.
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

        if cr.status != "collapsed":
            rows.append(row)
            continue

        assert cr.poly is not None
        row.n_monomials = cr.poly.n_monomials()
        row.poly_expr = poly_to_expr(cr.poly)

        sym_val = cr.poly.eval_at(cr.bindings) if cr.bindings else 0
        if cr.poly.n_monomials() and not cr.bindings:
            # constant Poly — bindings may still be empty, evaluate at {}
            sym_val = cr.poly.eval_at({})
        checks = [row.numpy_top == sym_val]

        if _EML_AVAILABLE:
            vars_in_poly = cr.poly.variables()
            if vars_in_poly:
                tree = compile_expr(
                    row.poly_expr,
                    variables=[f"x{v}" for v in vars_in_poly],
                )
                eml_bindings = {f"x{v}": cr.bindings[v] for v in vars_in_poly}
            else:
                tree = compile_expr(row.poly_expr)
                eml_bindings = {}
            row.eml_size = tree_size(tree)
            row.eml_depth = tree_depth(tree)
            eml_val = eval_eml(tree, eml_bindings)
            checks.append(
                abs(eml_val.imag) < 1e-6
                and int(round(eml_val.real)) == sym_val
            )

        row.numeric_match = all(checks)
        rows.append(row)
    return rows


# ─── Reporting ────────────────────────────────────────────────────

_BLOCK_REASON = {
    "blocked_control":   "control flow",
    "blocked_opcode":    "non-polynomial op",
    "blocked_underflow": "stack underflow",
}


def format_report(rows: List[CatalogRow]) -> str:
    """Render a markdown-style catalog report from :func:`run_catalog` rows."""
    n_collapsed = sum(1 for r in rows if r.status == "collapsed")
    n_control = sum(1 for r in rows if r.status == "blocked_control")
    n_opcode = sum(1 for r in rows if r.status == "blocked_opcode")
    n_other = len(rows) - n_collapsed - n_control - n_opcode

    lines = [
        "# LAC program catalog — symbolic collapse report",
        "",
        f"_{n_collapsed} collapsed, {n_control} blocked-by-branch, "
        f"{n_opcode} blocked-by-opcode"
        + (f", {n_other} other" if n_other else "")
        + f" (total {len(rows)})._",
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
        if r.status != "collapsed":
            continue
        size = "–" if r.eml_size is None else str(r.eml_size)
        depth = "–" if r.eml_depth is None else str(r.eml_depth)
        ok = "–" if r.numeric_match is None else ("✓" if r.numeric_match else "✗")
        lines.append(
            f"| `{r.name}` | {r.n_heads} | {r.n_monomials} | "
            f"`{r.poly_expr}` | {size} | {depth} | {ok} |"
        )

    lines += [
        "",
        "## Blocked (out of symbolic-executor scope)",
        "",
        "| Program | reason | blocker |",
        "|---|---|---|",
    ]
    for r in rows:
        if r.status == "collapsed" or r.status == "":
            continue
        reason = _BLOCK_REASON.get(r.status, r.status)
        lines.append(f"| `{r.name}` | {reason} | `{r.blocker}` |")

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
    "classify_program",
    "format_report",
    "poly_to_expr",
    "run_catalog",
]
