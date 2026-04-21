"""Symbolic executor for LAC programs (issues #65 / #70).

Claim: for branchless straight-line programs over {PUSH, POP, DUP, SWAP,
OVER, ROT, ADD, SUB, MUL}, the k-instruction sequence executed by LAC's
k attention heads collapses to a single polynomial in the PUSH constants.
This module makes the collapse mechanical: walk the program, carry a
symbolic stack whose entries are `Poly` expressions, and emit whatever's
on top when HALT fires.

Each PUSH allocates a fresh symbolic variable (`x0`, `x1`, ...) so the
output generalises across the concrete PUSH constants. `eval_at` plugs
the real integers back in to verify against ``NumPyExecutor``.

Issue #70 extends the executor past straight-line code:

  - **Guarded traces.** On a JZ/JNZ whose condition polynomial still
    carries variables, the executor forks into two paths with
    complementary guards (``cond == 0`` vs ``cond != 0``) and carries
    them to HALT independently. Halted tops are combined into a
    ``GuardedPoly`` — a partitioned case table over the input domain.
  - **Bounded-loop unrolling.** Running in ``input_mode="concrete"``
    pushes the raw PUSH args onto the symbolic stack (no variables)
    so every branch collapses deterministically and loops unroll by
    normal execution. The polynomial has no variables in this mode;
    it's a single integer, honest for "unrolled at the catalog input".
  - **Loop-symbolic detection.** If a path revisits ``(pc, sp)`` on a
    back-edge while the controlling condition still has variables, the
    path halts with ``loop_symbolic`` — we don't attempt invariant
    inference.

Out of scope (file as follow-ups):
  - Non-polynomial opcodes (DIV_S, REM_S, comparisons, bitwise).
  - Loop-invariant inference for truly symbolic loops.
  - Locals, heap, memory — no symbolic address model yet.
  - Emitting W_Q/W_K/W_V themselves as expression trees (the issue's
    longer-horizon export story). The claim about composition collapse
    lives at the semantic level and doesn't need the tree export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union

import isa
from isa import _trunc_div, _trunc_rem


# ─── Poly ──────────────────────────────────────────────────────────
#
# Polynomial over integer-indexed symbolic variables with rational
# (``int`` or :class:`fractions.Fraction`) coefficients. Canonical form:
# ``terms`` is a dict keyed by a monomial (tuple of (var_idx, power)
# pairs, sorted by var_idx, powers > 0). The empty tuple ``()`` is the
# constant monomial. Zero-coefficient terms are dropped on construction
# so comparisons are value-equal when the polynomials are mathematically
# equal.
#
# Rational coefficients land via issue #75 (symbolic DIV_S / REM_S): the
# bilinear forms for ADD/SUB/MUL produce polynomials over ℤ, but DIV_S
# introduces rational polynomials (``a/b`` with integer ``a, b``). To keep
# one canonical type, coefficients accept ``int | Fraction`` and
# normalise to ``int`` whenever the denominator is 1 — so every Poly
# produced by ADD/SUB/MUL still has literal ``int`` coefficients and
# existing structural-equality tests remain green.

Monomial = Tuple[Tuple[int, int], ...]


def _norm_coeff(c):
    """Normalise a coefficient to ``int`` when integral, else ``Fraction``.

    Accepts ``int`` or ``Fraction`` input. The canonical form keeps
    integer coefficients as ``int`` so value-compare against the
    pre-#75 Polys still works and ``repr`` output stays unchanged for
    the ADD/SUB/MUL fragment.
    """
    if isinstance(c, Fraction):
        if c.denominator == 1:
            return int(c.numerator)
        return c
    return int(c)


def _mono_mul(a: Monomial, b: Monomial) -> Monomial:
    """Merge two monomials. Powers add; result is sorted by var index."""
    if not a:
        return b
    if not b:
        return a
    merged: Dict[int, int] = {}
    for v, p in a:
        merged[v] = merged.get(v, 0) + p
    for v, p in b:
        merged[v] = merged.get(v, 0) + p
    return tuple(sorted(merged.items()))


def _mono_str(mono: Monomial) -> str:
    if not mono:
        return "1"
    parts = []
    for v, p in mono:
        parts.append(f"x{v}" if p == 1 else f"x{v}^{p}")
    return "·".join(parts)


@dataclass(frozen=True)
class Poly:
    """Multivariate polynomial with rational coefficients.

    ``terms`` maps a monomial (canonical-form tuple) to its coefficient,
    which is ``int`` when the coefficient is integral and
    :class:`fractions.Fraction` when the denominator is >1. Zero-
    coefficient entries are never stored.
    """

    terms: Mapping[Monomial, Union[int, Fraction]]

    @staticmethod
    def _normalise(terms: Mapping[Monomial, Union[int, Fraction]]
                   ) -> Dict[Monomial, Union[int, Fraction]]:
        return {m: _norm_coeff(c) for m, c in terms.items() if c != 0}

    def __post_init__(self):
        # Freeze a normalised copy. Doing it this way so callers can pass
        # any mapping and still get the value-equality guarantee.
        object.__setattr__(self, "terms", self._normalise(dict(self.terms)))

    # ── Constructors ──────────────────────────────────────────

    @classmethod
    def constant(cls, c: Union[int, Fraction]) -> "Poly":
        if c == 0:
            return cls({})
        return cls({(): _norm_coeff(c)})

    @classmethod
    def variable(cls, idx: int) -> "Poly":
        return cls({((int(idx), 1),): 1})

    # ── Arithmetic ────────────────────────────────────────────

    def __add__(self, other: "Poly") -> "Poly":
        out: Dict[Monomial, int] = dict(self.terms)
        for m, c in other.terms.items():
            out[m] = out.get(m, 0) + c
        return Poly(out)

    def __sub__(self, other: "Poly") -> "Poly":
        out: Dict[Monomial, int] = dict(self.terms)
        for m, c in other.terms.items():
            out[m] = out.get(m, 0) - c
        return Poly(out)

    def __neg__(self) -> "Poly":
        return Poly({m: -c for m, c in self.terms.items()})

    def __mul__(self, other: "Poly") -> "Poly":
        out: Dict[Monomial, int] = {}
        for ma, ca in self.terms.items():
            for mb, cb in other.terms.items():
                m = _mono_mul(ma, mb)
                out[m] = out.get(m, 0) + ca * cb
        return Poly(out)

    # ── Inspection ────────────────────────────────────────────

    def n_monomials(self) -> int:
        return len(self.terms)

    def variables(self) -> List[int]:
        """Variable indices referenced by any monomial, sorted."""
        seen = set()
        for m in self.terms:
            for v, _ in m:
                seen.add(v)
        return sorted(seen)

    def eval_at(self, bindings: Mapping[int, int]) -> Union[int, Fraction]:
        """Substitute ``bindings[i]`` for each ``x_i`` and reduce.

        Returns ``int`` when the result is integral (the common case for
        ADD/SUB/MUL Polys), otherwise returns :class:`fractions.Fraction`
        (after a DIV_S introduces a rational coefficient). Missing
        variables raise ``KeyError`` — symbolic executors that emit a
        variable per PUSH should pass one binding per PUSH.
        """
        total: Union[int, Fraction] = 0
        for mono, coeff in self.terms.items():
            term: Union[int, Fraction] = coeff
            for v, p in mono:
                term *= bindings[v] ** p
            total += term
        return _norm_coeff(total)

    # ── Equality / display ────────────────────────────────────

    def __eq__(self, other) -> bool:
        if not isinstance(other, Poly):
            return NotImplemented
        return self.terms == other.terms

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.terms.items())))

    def __repr__(self) -> str:  # deterministic for tests
        if not self.terms:
            return "0"
        # sort by (total degree, monomial) for readable output
        def _key(item):
            m, _ = item
            return (sum(p for _, p in m), m)

        pieces = []
        for mono, coeff in sorted(self.terms.items(), key=_key):
            ms = _mono_str(mono)
            if ms == "1":
                pieces.append(str(coeff))
                continue
            if coeff == 1:
                pieces.append(ms)
            elif coeff == -1:
                pieces.append(f"-{ms}")
            else:
                pieces.append(f"{coeff}·{ms}")
        # join with explicit signs
        out = pieces[0]
        for p in pieces[1:]:
            if p.startswith("-"):
                out += f" - {p[1:]}"
            else:
                out += f" + {p}"
        return out


# ─── Rational + remainder forms (issue #75) ───────────────────────
#
# DIV_S / REM_S break out of the polynomial ring: integer division is
# not polynomial, and even the underlying rational a/b lands outside
# :class:`Poly` without coefficient generalisation. Per the issue's
# design note ("Probably the latter, since WASM i32.div_s is truncating
# and we can model the rational inside the ring while the i32 rounding
# lives at the boundary"), we keep the symbolic form rational and apply
# ``trunc_div`` / ``trunc_rem`` only at ``eval_at`` — the same boundary
# pattern :func:`ff_symbolic.range_check` uses for i32 wrap on ADD/SUB/MUL.
#
# Two minimal types, one per op: :class:`RationalPoly` for DIV_S and
# :class:`SymbolicRemainder` for REM_S. Algebra past the op itself is
# deliberately not closed — the catalog rows this unblocks
# (``native_divmod``, ``native_remainder``) consist of ``PUSH b; PUSH a;
# DIV_S/REM_S; HALT``, so no composition with Poly arithmetic is
# required. Composing DIV_S with further ADD/SUB/MUL would need a full
# rational-function algebra, listed as a follow-up (the issue's
# non-goal list calls this out explicitly).


@dataclass(frozen=True)
class RationalPoly:
    """Symbolic quotient ``num / denom`` under WASM ``i32.div_s`` semantics.

    Stores the two operand polynomials verbatim. ``eval_at`` reduces
    them to integers under the given bindings and then applies
    truncating-toward-zero division (:func:`isa._trunc_div`) — the same
    semantic the compiled transformer's nonlinear path applies. Trapping
    on ``denom == 0`` is the caller's responsibility at the bindings
    site; ``eval_at`` raises :class:`ZeroDivisionError` in that case.

    Structural equality is value-based on ``(num, denom)``, so two
    symbolic executors that emit the same operand polys produce equal
    tops — the equivalence test the issue asks for.
    """
    num: Poly
    denom: Poly

    def variables(self) -> List[int]:
        return sorted(set(self.num.variables()) | set(self.denom.variables()))

    def eval_at(self, bindings: Mapping[int, int]) -> int:
        """Integer result: ``trunc(num(bindings) / denom(bindings))``.

        ``denom`` evaluating to 0 raises :class:`ZeroDivisionError` — the
        catalog's `native_divmod(0, b)` variants already produce a trap
        in :class:`executor.NumPyExecutor` rather than a value, so the
        symbolic side mirrors that failure mode.
        """
        n = self.num.eval_at(bindings)
        d = self.denom.eval_at(bindings)
        if d == 0:
            raise ZeroDivisionError(
                f"RationalPoly.eval_at: denom {self.denom!r} = 0 at bindings={dict(bindings)}"
            )
        return _trunc_div(int(n), int(d))

    def __repr__(self) -> str:
        return f"({self.num}) /ₜ ({self.denom})"


@dataclass(frozen=True)
class SymbolicRemainder:
    """Symbolic remainder ``num % denom`` under WASM ``i32.rem_s`` semantics.

    Stored as a ``(num, denom)`` pair rather than reduced to a closed
    polynomial form, because ``b mod a`` is not rational in ``(a, b)``
    — the truncation that defines it is piecewise, not algebraic.
    ``eval_at`` applies :func:`isa._trunc_rem` at the boundary, matching
    the compiled transformer.
    """
    num: Poly
    denom: Poly

    def variables(self) -> List[int]:
        return sorted(set(self.num.variables()) | set(self.denom.variables()))

    def eval_at(self, bindings: Mapping[int, int]) -> int:
        n = self.num.eval_at(bindings)
        d = self.denom.eval_at(bindings)
        if d == 0:
            raise ZeroDivisionError(
                f"SymbolicRemainder.eval_at: denom {self.denom!r} = 0 at bindings={dict(bindings)}"
            )
        return _trunc_rem(int(n), int(d))

    def __repr__(self) -> str:
        return f"({self.num}) %ₜ ({self.denom})"


# Union covering every "top of symbolic stack" type run_symbolic /
# run_forking might emit for a branchless polynomial-plus-rational program.
RationalStackValue = Union[Poly, RationalPoly, SymbolicRemainder]


# ─── Guard + GuardedPoly ──────────────────────────────────────────
#
# A guard is a polynomial we assert is either "== 0" or "!= 0". A
# conjunction is a tuple of guards that must all hold simultaneously.
# GuardedPoly is a case table — one (conjunction, value_poly) entry per
# partition of the input domain.
#
# Guards are value-compared on (poly, eq_zero), so two paths that derive
# the same guard chain in different orders merge cleanly after sorting.


@dataclass(frozen=True)
class Guard:
    """Assertion that ``poly == 0`` (eq_zero=True) or ``poly != 0``."""
    poly: Poly
    eq_zero: bool

    def __repr__(self) -> str:
        op = "==" if self.eq_zero else "!="
        return f"({self.poly} {op} 0)"


def _canonical_guards(guards: Tuple[Guard, ...]) -> Tuple[Guard, ...]:
    """Deduplicate + sort a guard conjunction for value-based equality."""
    # Use hash-based dedupe; Guard is frozen so hashable.
    return tuple(sorted(set(guards), key=lambda g: (repr(g.poly), g.eq_zero)))


def _guards_complementary(a: Tuple[Guard, ...], b: Tuple[Guard, ...]) -> bool:
    """True iff a and b differ on exactly one guard by `eq_zero` flip."""
    if len(a) != len(b):
        return False
    diff = 0
    for ga, gb in zip(a, b):
        if ga == gb:
            continue
        if ga.poly == gb.poly and ga.eq_zero != gb.eq_zero:
            diff += 1
        else:
            return False
    return diff == 1


@dataclass(frozen=True)
class GuardedPoly:
    """Partitioned case table: ``[(guards, value_poly), ...]``.

    Each case's ``guards`` tuple is a conjunction that must hold for
    that case's ``value_poly`` to apply. The set of cases is expected
    to partition the domain — i.e. for any concrete bindings, exactly
    one guard conjunction evaluates to True.
    """
    cases: Tuple[Tuple[Tuple[Guard, ...], Poly], ...]

    def __post_init__(self):
        canonical = tuple(
            (_canonical_guards(gs), v) for gs, v in self.cases
        )
        # Sort cases deterministically for equality.
        canonical = tuple(sorted(canonical,
                                 key=lambda c: (tuple(repr(g) for g in c[0]), repr(c[1]))))
        object.__setattr__(self, "cases", canonical)

    def n_cases(self) -> int:
        return len(self.cases)

    def variables(self) -> List[int]:
        seen = set()
        for gs, v in self.cases:
            for g in gs:
                seen.update(g.poly.variables())
            seen.update(v.variables())
        return sorted(seen)

    def eval_at(self, bindings: Mapping[int, int]) -> int:
        """Pick the unique case whose guards all hold at ``bindings``."""
        hits: List[int] = []
        for gs, v in self.cases:
            ok = True
            for g in gs:
                try:
                    val = g.poly.eval_at(bindings)
                except KeyError:
                    ok = False
                    break
                if g.eq_zero and val != 0:
                    ok = False
                    break
                if not g.eq_zero and val == 0:
                    ok = False
                    break
            if ok:
                hits.append(v.eval_at(bindings))
        if len(hits) != 1:
            raise ValueError(
                f"GuardedPoly.eval_at: {len(hits)} cases hit (expected 1) "
                f"at bindings={dict(bindings)}"
            )
        return hits[0]

    def __repr__(self) -> str:
        body = ", ".join(
            f"{{{' ∧ '.join(repr(g) for g in gs) or 'True'}}} → {v}"
            for gs, v in self.cases
        )
        return f"Guarded[{body}]"


# ─── SymbolicExecutor ──────────────────────────────────────────────

# Opcodes the *branchless* fragment understands — preserved for the
# legacy run_symbolic entry point that the original PoC tests exercise.
# Issue #75 adds DIV_S / REM_S: they break the polynomial-ring closure
# (outputs are :class:`RationalPoly` / :class:`SymbolicRemainder`) but
# the executor still accepts them as long as nothing downstream tries to
# compose a Poly op against a rational stack entry.
_POLY_OPS = {
    isa.OP_PUSH, isa.OP_POP, isa.OP_DUP, isa.OP_HALT,
    isa.OP_ADD, isa.OP_SUB, isa.OP_MUL,
    isa.OP_DIV_S, isa.OP_REM_S,
    isa.OP_SWAP, isa.OP_OVER, isa.OP_ROT, isa.OP_NOP,
}
# Branch ops the forking executor additionally handles.
_BRANCH_OPS = {isa.OP_JZ, isa.OP_JNZ}
# Union: what run_forking accepts.
_FORKING_OPS = _POLY_OPS | _BRANCH_OPS


@dataclass
class SymbolicResult:
    """Outcome of running a program symbolically.

    ``top`` is the top-of-stack value after HALT (or at the end of the
    trace if HALT is absent). For the ADD/SUB/MUL fragment it's a
    :class:`Poly`; for the DIV_S / REM_S rows added in issue #75 it can
    also be :class:`RationalPoly` or :class:`SymbolicRemainder`. ``stack``
    is the full final stack (bottom at index 0). ``n_heads`` is the
    number of instructions executed — the "k heads" the issue talks
    about. ``bindings`` maps the allocated variable indices back to the
    original PUSH constants.
    """

    top: RationalStackValue
    stack: List[RationalStackValue]
    n_heads: int
    bindings: Dict[int, int]

    def collapse_ratio(self) -> float:
        """k_heads ÷ n_monomials in the top expression, after simplification.

        Matches the issue's "9 heads → 1 monomial" style report. Returns
        ``inf`` when the top collapses to zero (no monomials) — flagged
        by callers who want a cleaner representation. For rational
        outputs (DIV_S / REM_S) the denominator counts as an additional
        monomial bundle; we sum the two sides' monomial counts to keep
        the "one number" shape of the ratio.
        """
        if isinstance(self.top, Poly):
            n = self.top.n_monomials()
        elif isinstance(self.top, (RationalPoly, SymbolicRemainder)):
            n = self.top.num.n_monomials() + self.top.denom.n_monomials()
        else:
            return float("inf")
        if n == 0:
            return float("inf")
        return self.n_heads / n


class SymbolicStackUnderflow(RuntimeError):
    pass


class SymbolicOpNotSupported(NotImplementedError):
    pass


class SymbolicLoopSymbolic(RuntimeError):
    """Raised when a path hits a back-edge whose cond is still symbolic."""
    pass


class SymbolicPathExplosion(RuntimeError):
    """Raised when the live path count exceeds the configured cap."""
    pass


def run_symbolic(prog: List[isa.Instruction]) -> SymbolicResult:
    """Execute ``prog`` symbolically. One variable allocated per PUSH.

    Polynomial composition happens eagerly so the final form is already
    simplified — no separate simplification pass needed.

    Branch ops raise :class:`SymbolicOpNotSupported` to preserve the
    issue-#65 "branchless only" contract. Use :func:`run_forking` for
    programs with JZ/JNZ.
    """
    stack: List[Poly] = []
    bindings: Dict[int, int] = {}
    next_var = 0
    n_heads = 0

    def _pop() -> Poly:
        if not stack:
            raise SymbolicStackUnderflow("pop from empty stack")
        return stack.pop()

    for instr in prog:
        op = instr.op
        arg = instr.arg
        if op not in _POLY_OPS:
            name = isa.OP_NAMES.get(op, f"?{op}")
            raise SymbolicOpNotSupported(
                f"op {name!r} is not polynomial-closed; issue #65 scope is "
                f"branchless straight-line programs"
            )
        if op == isa.OP_HALT:
            # HALT terminates the trace; the issue's head counts match
            # the number of instructions executed *before* HALT fires.
            break
        n_heads += 1
        if op == isa.OP_NOP:
            continue

        if op == isa.OP_PUSH:
            v = next_var
            next_var += 1
            bindings[v] = int(arg)
            stack.append(Poly.variable(v))
        elif op == isa.OP_POP:
            _pop()
        elif op == isa.OP_DUP:
            if not stack:
                raise SymbolicStackUnderflow("dup on empty stack")
            stack.append(stack[-1])
        elif op == isa.OP_ADD:
            b = _pop(); a = _pop()
            if not isinstance(a, Poly) or not isinstance(b, Poly):
                raise SymbolicOpNotSupported(
                    "ADD on rational stack entries is out of scope "
                    "(composition past one DIV_S/REM_S is a follow-up)"
                )
            stack.append(a + b)
        elif op == isa.OP_SUB:
            b = _pop(); a = _pop()
            if not isinstance(a, Poly) or not isinstance(b, Poly):
                raise SymbolicOpNotSupported(
                    "SUB on rational stack entries is out of scope "
                    "(composition past one DIV_S/REM_S is a follow-up)"
                )
            stack.append(a - b)
        elif op == isa.OP_MUL:
            b = _pop(); a = _pop()
            if not isinstance(a, Poly) or not isinstance(b, Poly):
                raise SymbolicOpNotSupported(
                    "MUL on rational stack entries is out of scope "
                    "(composition past one DIV_S/REM_S is a follow-up)"
                )
            stack.append(a * b)
        elif op == isa.OP_DIV_S:
            b = _pop(); a = _pop()
            if not isinstance(a, Poly) or not isinstance(b, Poly):
                raise SymbolicOpNotSupported(
                    "DIV_S on rational stack entries is out of scope "
                    "(composition past one DIV_S/REM_S is a follow-up)"
                )
            stack.append(RationalPoly(num=a, denom=b))
        elif op == isa.OP_REM_S:
            b = _pop(); a = _pop()
            if not isinstance(a, Poly) or not isinstance(b, Poly):
                raise SymbolicOpNotSupported(
                    "REM_S on rational stack entries is out of scope "
                    "(composition past one DIV_S/REM_S is a follow-up)"
                )
            stack.append(SymbolicRemainder(num=a, denom=b))
        elif op == isa.OP_SWAP:
            if len(stack) < 2:
                raise SymbolicStackUnderflow("swap needs 2 entries")
            stack[-1], stack[-2] = stack[-2], stack[-1]
        elif op == isa.OP_OVER:
            if len(stack) < 2:
                raise SymbolicStackUnderflow("over needs 2 entries")
            stack.append(stack[-2])
        elif op == isa.OP_ROT:
            if len(stack) < 3:
                raise SymbolicStackUnderflow("rot needs 3 entries")
            # [a, b, c] -> [b, c, a] (matches test_algorithm semantics)
            a, b, c = stack[-3], stack[-2], stack[-1]
            stack[-3], stack[-2], stack[-1] = b, c, a
        else:  # pragma: no cover — guarded by _POLY_OPS
            raise SymbolicOpNotSupported(f"unreachable: op {op}")

    top = stack[-1] if stack else Poly.constant(0)
    return SymbolicResult(top=top, stack=list(stack),
                          n_heads=n_heads, bindings=bindings)


# ─── Arithmetic hook (issue #68 S3) ───────────────────────────────
#
# ``run_forking`` calls three arithmetic primitives in its inner loop —
# one each for ADD, SUB, MUL. The default primitives are ``Poly``'s
# native ``+ - *``; the forking driver is indifferent to which
# implementation is plugged in as long as the signature is ``(Poly,
# Poly) -> Poly`` and the algebra is the same.
#
# The hook exists so :mod:`ff_symbolic` can drive the same forking
# executor with its bilinear-FF interpretation of the primitives
# (``symbolic_add/sub/mul``) and demonstrate the equivalence claim
# from issue #69 extends across JZ/JNZ control flow — not just
# branchless straight-line programs.


ArithFn = Callable[["Poly", "Poly"], "Poly"]
DivFn = Callable[["Poly", "Poly"], "RationalPoly"]
RemFn = Callable[["Poly", "Poly"], "SymbolicRemainder"]


@dataclass(frozen=True)
class ArithmeticOps:
    """Operator spec the forking executor consumes for the arithmetic
    fragment of the ISA.

    ``sub(a, b)`` must return ``a - b`` (executor computes ``a - b``
    where ``a`` is the second-from-top, matching the existing Poly
    order). ``ff_symbolic.symbolic_sub`` matches this spec.

    ``div_s(a, b)`` / ``rem_s(a, b)`` (issue #75) must return the
    symbolic quotient / remainder of ``a / b`` under WASM
    ``i32.div_s`` / ``i32.rem_s`` semantics — ``a`` is stack[SP-1] (the
    dividend) and ``b`` is top (the divisor), matching the numeric
    path's ``_trunc_div(vb, va)`` convention (``executor.py:835-838``).
    """
    add: ArithFn
    sub: ArithFn
    mul: ArithFn
    div_s: DivFn = None  # type: ignore[assignment]
    rem_s: RemFn = None  # type: ignore[assignment]


DEFAULT_ARITHMETIC_OPS = ArithmeticOps(
    add=lambda a, b: a + b,
    sub=lambda a, b: a - b,
    mul=lambda a, b: a * b,
    div_s=lambda a, b: RationalPoly(num=a, denom=b),
    rem_s=lambda a, b: SymbolicRemainder(num=a, denom=b),
)


# ─── Forking executor (issue #70) ─────────────────────────────────

# Default caps. Catalog programs stay under these comfortably; the
# symbolic-loop exit trips long before we get close.
DEFAULT_MAX_PATHS = 64
DEFAULT_MAX_STEPS = 50_000


def _as_concrete_int(p: "RationalStackValue") -> Optional[int]:
    """Return integer value if ``p`` has no variables; else None.

    Rational stack values (``RationalPoly`` / ``SymbolicRemainder``) never
    collapse to a concrete int for branching purposes — DIV_S/REM_S past a
    subsequent JZ/JNZ is out of scope for issue #75, so return ``None``
    and let the caller fall into the symbolic-cond path (which will then
    raise when it tries to wrap the value in a Guard).
    """
    if not isinstance(p, Poly):
        return None
    if not p.terms:
        return 0
    if len(p.terms) == 1 and () in p.terms:
        return int(p.terms[()])
    for mono in p.terms:
        if mono:
            return None
    # Only the constant monomial appears (possibly not present).
    return int(p.terms.get((), 0))


@dataclass
class _Path:
    """One symbolic execution thread.

    ``visited_branches`` records ``(pc, sp)`` pairs where this path took
    a symbolic branch; revisiting such a pair with a still-symbolic cond
    at the same site is the loop-symbolic signal.

    Variables are indexed by the PC of the PUSH instruction that
    allocated them, so forked paths sharing a prefix also share the
    variable ids of pre-fork PUSHes — and post-fork PUSHes at distinct
    static sites get distinct ids even across paths.
    """
    pc: int
    stack: Tuple["RationalStackValue", ...]
    guards: Tuple[Guard, ...]
    bindings: Dict[int, int]
    n_heads: int = 0
    visited_branches: frozenset = field(default_factory=frozenset)
    loop_unrolled: bool = False  # True if this path ever took a back-edge
    halted_top: Optional["RationalStackValue"] = None

    def with_(self, **kwargs) -> "_Path":
        """Return a copy with selected fields replaced."""
        base = dict(
            pc=self.pc, stack=self.stack, guards=self.guards,
            bindings=dict(self.bindings),
            n_heads=self.n_heads, visited_branches=self.visited_branches,
            loop_unrolled=self.loop_unrolled, halted_top=self.halted_top,
        )
        base.update(kwargs)
        return _Path(**base)


@dataclass
class ForkingResult:
    """Outcome of running a program via the forking executor.

    ``top`` collapses:
      - to a single :class:`Poly` if all halted paths agree;
      - to a :class:`GuardedPoly` when paths disagree;
      - to ``None`` if no path halted (loop_symbolic-only outcome).

    ``status`` is one of ``"straight" | "guarded" | "unrolled" |
    "loop_symbolic" | "path_explosion" | "blocked_underflow"``.
    """

    top: Optional[Union[Poly, GuardedPoly]]
    status: str
    n_heads: int                  # max k_heads across halted paths
    bindings: Dict[int, int]      # union of all paths' bindings
    n_halted: int = 0
    n_loop_symbolic: int = 0
    paths_explored: int = 0


def _eq_guard(p: Poly, eq_zero: bool) -> Guard:
    return Guard(poly=p, eq_zero=eq_zero)


def run_forking(prog: List[isa.Instruction], *,
                input_mode: str = "symbolic",
                max_paths: int = DEFAULT_MAX_PATHS,
                max_steps: int = DEFAULT_MAX_STEPS,
                arithmetic_ops: Optional[ArithmeticOps] = None) -> ForkingResult:
    """Forking symbolic executor with finite-conditional + bounded-loop support.

    ``input_mode``:
      - ``"symbolic"``: each PUSH allocates a fresh variable. Branches on
        symbolic conditions fork the path. Suitable for ``collapsed_guarded``.
      - ``"concrete"``: each PUSH pushes its literal arg (no variables).
        All branches collapse deterministically; loops unroll naturally.
        Suitable for ``collapsed_unrolled``.

    ``arithmetic_ops``: override the ADD/SUB/MUL primitives applied to
    Poly stack entries. Defaults to :data:`DEFAULT_ARITHMETIC_OPS` (plain
    Poly ``+ - *``). :mod:`ff_symbolic` passes its bilinear-FF
    interpretation here (issue #68 S3) to demonstrate equivalence across
    control flow.

    The executor uses a worklist. Each fork splits the path into two new
    paths carrying complementary guards. When a path's top polynomial is
    concrete at a branch, the branch is followed deterministically. A
    symbolic back-edge that revisits ``(pc, sp)`` halts the path with
    ``loop_symbolic``.
    """
    if input_mode not in ("symbolic", "concrete"):
        raise ValueError(f"unknown input_mode {input_mode!r}")
    ops = arithmetic_ops if arithmetic_ops is not None else DEFAULT_ARITHMETIC_OPS

    # Pre-flight: reject programs with non-polynomial, non-branch opcodes.
    for instr in prog:
        if instr.op not in _FORKING_OPS:
            name = isa.OP_NAMES.get(instr.op, f"?{instr.op}")
            raise SymbolicOpNotSupported(
                f"op {name!r} is out of scope for the forking executor "
                f"(polynomial + JZ/JNZ only)"
            )

    init = _Path(
        pc=0, stack=(), guards=(),
        bindings={}, n_heads=0,
        visited_branches=frozenset(), loop_unrolled=False,
    )
    worklist: List[_Path] = [init]
    halted: List[_Path] = []
    loop_symbolic_paths: List[_Path] = []
    paths_explored = 0
    total_steps = 0
    underflow_seen = False

    def _spawn(new: _Path):
        if len(worklist) + len(halted) + len(loop_symbolic_paths) + 1 > max_paths:
            raise SymbolicPathExplosion(
                f"path count exceeds max_paths={max_paths}"
            )
        worklist.append(new)

    try:
        while worklist:
            path = worklist.pop()
            paths_explored += 1
            # Step this path until it halts, forks, or loops symbolically.
            while True:
                total_steps += 1
                if total_steps > max_steps:
                    raise SymbolicPathExplosion(
                        f"total step count exceeds max_steps={max_steps}"
                    )
                if path.pc < 0 or path.pc >= len(prog):
                    # Implicit fall-off-end acts as HALT with current top.
                    path = path.with_(
                        halted_top=path.stack[-1] if path.stack
                        else Poly.constant(0),
                    )
                    halted.append(path)
                    break
                instr = prog[path.pc]
                op = instr.op
                if op == isa.OP_HALT:
                    path = path.with_(
                        halted_top=path.stack[-1] if path.stack
                        else Poly.constant(0),
                    )
                    halted.append(path)
                    break

                # Non-branch, non-halt → advance n_heads and apply op.
                if op != isa.OP_JZ and op != isa.OP_JNZ:
                    try:
                        stack = _apply_poly_op(path, instr, input_mode, ops)
                    except SymbolicStackUnderflow:
                        underflow_seen = True
                        # drop this path; don't propagate partial result
                        break
                    new_bindings = path.bindings
                    if op == isa.OP_PUSH and input_mode == "symbolic":
                        # Variable id = PUSH's pc (stable across forked paths).
                        new_bindings = dict(path.bindings)
                        new_bindings[path.pc] = int(instr.arg)
                    path = path.with_(
                        pc=path.pc + 1,
                        stack=stack,
                        n_heads=path.n_heads + (0 if op == isa.OP_NOP else 1),
                        bindings=new_bindings,
                    )
                    continue

                # JZ / JNZ: pop cond, decide branch.
                if not path.stack:
                    underflow_seen = True
                    break
                cond = path.stack[-1]
                popped_stack = path.stack[:-1]
                path = path.with_(
                    n_heads=path.n_heads + 1,
                    stack=popped_stack,
                )
                sp = len(popped_stack)
                target = int(instr.arg)
                fall_through = path.pc + 1
                is_back_edge = target <= path.pc

                concrete = _as_concrete_int(cond)
                if concrete is not None:
                    taken = (concrete == 0) if op == isa.OP_JZ else (concrete != 0)
                    new_pc = target if taken else fall_through
                    path = path.with_(
                        pc=new_pc,
                        loop_unrolled=path.loop_unrolled or (is_back_edge and taken),
                    )
                    continue

                # Symbolic condition → fork. Check for symbolic back-edge revisit.
                site = (path.pc, sp, op)
                if is_back_edge and site in path.visited_branches:
                    # This path already forked at this back-edge once with a
                    # symbolic cond — seeing it again means no progress.
                    loop_symbolic_paths.append(path)
                    break
                new_visited = path.visited_branches | {site}

                eq_guard = _eq_guard(cond, eq_zero=True)
                ne_guard = _eq_guard(cond, eq_zero=False)
                # JZ: take when cond == 0. JNZ: take when cond != 0.
                take_guard = eq_guard if op == isa.OP_JZ else ne_guard
                skip_guard = ne_guard if op == isa.OP_JZ else eq_guard

                take_path = path.with_(
                    pc=target,
                    guards=path.guards + (take_guard,),
                    visited_branches=new_visited,
                )
                skip_path = path.with_(
                    pc=fall_through,
                    guards=path.guards + (skip_guard,),
                    visited_branches=new_visited,
                )
                _spawn(skip_path)
                _spawn(take_path)
                break  # current thread is replaced by the two new ones
    except SymbolicPathExplosion:
        # Collect what we have and report.
        return ForkingResult(
            top=None, status="path_explosion",
            n_heads=max((p.n_heads for p in halted + loop_symbolic_paths), default=0),
            bindings={}, n_halted=len(halted),
            n_loop_symbolic=len(loop_symbolic_paths),
            paths_explored=paths_explored,
        )

    # Combine halted tops.
    if not halted:
        if loop_symbolic_paths:
            return ForkingResult(
                top=None, status="loop_symbolic", n_heads=0,
                bindings={}, n_halted=0,
                n_loop_symbolic=len(loop_symbolic_paths),
                paths_explored=paths_explored,
            )
        if underflow_seen:
            return ForkingResult(
                top=None, status="blocked_underflow", n_heads=0,
                bindings={}, n_halted=0, n_loop_symbolic=0,
                paths_explored=paths_explored,
            )
        return ForkingResult(
            top=None, status="blocked_underflow", n_heads=0,
            bindings={}, n_halted=0, n_loop_symbolic=0,
            paths_explored=paths_explored,
        )

    merged_bindings: Dict[int, int] = {}
    for p in halted:
        merged_bindings.update(p.bindings)

    tops = [(p.guards, p.halted_top) for p in halted]
    # Determine status.
    any_forked = any(p.guards for p in halted)
    any_looped = any(p.loop_unrolled for p in halted) or bool(loop_symbolic_paths)

    # Build the top: a single Poly if all agree and no guards, else GuardedPoly.
    unique_values = {v for _, v in tops}
    if not any_forked and len(unique_values) == 1:
        top_val: Union[Poly, GuardedPoly] = next(iter(unique_values))
    else:
        top_val = _build_guarded_poly(tops)

    if loop_symbolic_paths and not halted:
        status = "loop_symbolic"
    elif loop_symbolic_paths:
        # Partial collapse: some paths halted, others hit symbolic loops.
        status = "loop_symbolic"
    elif any_looped:
        status = "unrolled"
    elif any_forked:
        status = "guarded"
    else:
        status = "straight"

    n_heads = max((p.n_heads for p in halted), default=0)
    return ForkingResult(
        top=top_val, status=status, n_heads=n_heads,
        bindings=merged_bindings, n_halted=len(halted),
        n_loop_symbolic=len(loop_symbolic_paths),
        paths_explored=paths_explored,
    )


def _apply_poly_op(path: _Path, instr: isa.Instruction,
                   input_mode: str,
                   arithmetic_ops: ArithmeticOps = DEFAULT_ARITHMETIC_OPS) -> Tuple[RationalStackValue, ...]:
    """Apply a non-branch opcode to ``path.stack`` and return the new stack.

    ``arithmetic_ops`` picks the ADD/SUB/MUL/DIV_S/REM_S implementations.
    Defaults to Poly's native operators (plus RationalPoly/SymbolicRemainder
    wrappers for DIV_S/REM_S); :mod:`ff_symbolic` passes its bilinear-FF
    primitives.
    """
    op = instr.op
    stack = list(path.stack)

    def _pop() -> RationalStackValue:
        if not stack:
            raise SymbolicStackUnderflow(f"pop from empty stack at pc={path.pc}")
        return stack.pop()

    if op == isa.OP_PUSH:
        if input_mode == "symbolic":
            stack.append(Poly.variable(path.pc))
        else:
            stack.append(Poly.constant(int(instr.arg)))
    elif op == isa.OP_POP:
        _pop()
    elif op == isa.OP_DUP:
        if not stack:
            raise SymbolicStackUnderflow(f"dup on empty stack at pc={path.pc}")
        stack.append(stack[-1])
    elif op == isa.OP_ADD:
        b = _pop(); a = _pop()
        if not isinstance(a, Poly) or not isinstance(b, Poly):
            raise SymbolicOpNotSupported(
                "ADD on rational stack entries is out of scope "
                "(composition past one DIV_S/REM_S is a follow-up)"
            )
        stack.append(arithmetic_ops.add(a, b))
    elif op == isa.OP_SUB:
        b = _pop(); a = _pop()
        if not isinstance(a, Poly) or not isinstance(b, Poly):
            raise SymbolicOpNotSupported(
                "SUB on rational stack entries is out of scope "
                "(composition past one DIV_S/REM_S is a follow-up)"
            )
        stack.append(arithmetic_ops.sub(a, b))
    elif op == isa.OP_MUL:
        b = _pop(); a = _pop()
        if not isinstance(a, Poly) or not isinstance(b, Poly):
            raise SymbolicOpNotSupported(
                "MUL on rational stack entries is out of scope "
                "(composition past one DIV_S/REM_S is a follow-up)"
            )
        stack.append(arithmetic_ops.mul(a, b))
    elif op == isa.OP_DIV_S:
        b = _pop(); a = _pop()
        if not isinstance(a, Poly) or not isinstance(b, Poly):
            raise SymbolicOpNotSupported(
                "DIV_S on rational stack entries is out of scope "
                "(composition past one DIV_S/REM_S is a follow-up)"
            )
        if arithmetic_ops.div_s is None:
            raise SymbolicOpNotSupported(
                "arithmetic_ops.div_s is not wired; pass a div_s primitive"
            )
        stack.append(arithmetic_ops.div_s(a, b))
    elif op == isa.OP_REM_S:
        b = _pop(); a = _pop()
        if not isinstance(a, Poly) or not isinstance(b, Poly):
            raise SymbolicOpNotSupported(
                "REM_S on rational stack entries is out of scope "
                "(composition past one DIV_S/REM_S is a follow-up)"
            )
        if arithmetic_ops.rem_s is None:
            raise SymbolicOpNotSupported(
                "arithmetic_ops.rem_s is not wired; pass a rem_s primitive"
            )
        stack.append(arithmetic_ops.rem_s(a, b))
    elif op == isa.OP_SWAP:
        if len(stack) < 2:
            raise SymbolicStackUnderflow(f"swap needs 2 entries at pc={path.pc}")
        stack[-1], stack[-2] = stack[-2], stack[-1]
    elif op == isa.OP_OVER:
        if len(stack) < 2:
            raise SymbolicStackUnderflow(f"over needs 2 entries at pc={path.pc}")
        stack.append(stack[-2])
    elif op == isa.OP_ROT:
        if len(stack) < 3:
            raise SymbolicStackUnderflow(f"rot needs 3 entries at pc={path.pc}")
        a, b, c = stack[-3], stack[-2], stack[-1]
        stack[-3], stack[-2], stack[-1] = b, c, a
    elif op == isa.OP_NOP:
        pass
    else:  # pragma: no cover
        raise SymbolicOpNotSupported(f"op {op} unexpected in _apply_poly_op")

    return tuple(stack)


def _build_guarded_poly(
    tops: List[Tuple[Tuple[Guard, ...], Poly]],
) -> Union[Poly, GuardedPoly]:
    """Merge per-path ``(guards, value)`` pairs into a single GuardedPoly.

    Paths with the same value polynomial are combined by merging their
    guard chains. If all paths produce the same value *and* their guards
    together span the full domain (the trivial case of a single path
    with empty guards), we return the bare Poly.
    """
    # Group by value poly.
    by_value: Dict[Poly, List[Tuple[Guard, ...]]] = {}
    for gs, v in tops:
        by_value.setdefault(v, []).append(_canonical_guards(gs))

    # If only one distinct value and at least one path has no guards, it's unconditional.
    if len(by_value) == 1:
        sole_value = next(iter(by_value))
        guard_chains = by_value[sole_value]
        if any(len(gs) == 0 for gs in guard_chains):
            return sole_value

    cases: List[Tuple[Tuple[Guard, ...], Poly]] = []
    for value, guard_chains in by_value.items():
        for gs in guard_chains:
            cases.append((gs, value))
    return GuardedPoly(cases=tuple(cases))


# ─── Reporting helper ─────────────────────────────────────────────

def collapse_report(prog: List[isa.Instruction], *,
                    name: str = "") -> str:
    """Run ``prog`` symbolically and return a one-line collapse summary.

    Example::

        PUSH 5; DUP;ADD;DUP;ADD;DUP;ADD;DUP;ADD  →  9 heads, 1 monomial, top = 16·x0
    """
    r = run_symbolic(prog)
    prefix = f"{name}: " if name else ""
    return (f"{prefix}{r.n_heads} heads → {r.top.n_monomials()} "
            f"monomials, top = {r.top}")


__all__ = [
    "Poly",
    "Guard",
    "GuardedPoly",
    "SymbolicResult",
    "ForkingResult",
    "SymbolicStackUnderflow",
    "SymbolicOpNotSupported",
    "SymbolicLoopSymbolic",
    "SymbolicPathExplosion",
    "ArithmeticOps",
    "DEFAULT_ARITHMETIC_OPS",
    "DEFAULT_MAX_PATHS",
    "DEFAULT_MAX_STEPS",
    "run_symbolic",
    "run_forking",
    "collapse_report",
]
