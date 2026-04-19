"""Symbolic executor for branchless LAC programs (issue #65).

Claim: for branchless straight-line programs over {PUSH, POP, DUP, SWAP,
OVER, ROT, ADD, SUB, MUL}, the k-instruction sequence executed by LAC's
k attention heads collapses to a single polynomial in the PUSH constants.
This module makes the collapse mechanical: walk the program, carry a
symbolic stack whose entries are `Poly` expressions, and emit whatever's
on top when HALT fires.

Each PUSH allocates a fresh symbolic variable (`x0`, `x1`, ...) so the
output generalises across the concrete PUSH constants. `eval_at` plugs
the real integers back in to verify against ``NumPyExecutor``.

Out of scope (file as follow-ups):
  - Control flow (JZ/JNZ/CALL/RETURN) — branching creates piecewise polys.
  - Bitwise / division / comparisons — not a polynomial operation.
  - Locals, heap, memory — no symbolic address model yet.
  - Emitting W_Q/W_K/W_V themselves as expression trees (the issue's
    longer-horizon export story). The claim about composition collapse
    lives at the semantic level and doesn't need the tree export.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

import isa


# ─── Poly ──────────────────────────────────────────────────────────
#
# Polynomial over integer-indexed symbolic variables with integer
# coefficients. Canonical form: terms is a dict keyed by a monomial
# (tuple of (var_idx, power) pairs, sorted by var_idx, powers > 0).
# The empty tuple `()` is the constant monomial. Zero-coefficient
# terms are dropped on construction so comparisons are value-equal
# when the polynomials are mathematically equal.

Monomial = Tuple[Tuple[int, int], ...]


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
    """Multivariate polynomial with integer coefficients.

    `terms` maps a monomial (canonical-form tuple) to its coefficient.
    Zero-coefficient entries are never stored.
    """

    terms: Mapping[Monomial, int]

    @staticmethod
    def _normalise(terms: Mapping[Monomial, int]) -> Dict[Monomial, int]:
        return {m: int(c) for m, c in terms.items() if c != 0}

    def __post_init__(self):
        # Freeze a normalised copy. Doing it this way so callers can pass
        # any mapping and still get the value-equality guarantee.
        object.__setattr__(self, "terms", self._normalise(dict(self.terms)))

    # ── Constructors ──────────────────────────────────────────

    @classmethod
    def constant(cls, c: int) -> "Poly":
        if c == 0:
            return cls({})
        return cls({(): int(c)})

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

    def eval_at(self, bindings: Mapping[int, int]) -> int:
        """Substitute `bindings[i]` for each ``x_i`` and reduce to an int.

        Missing variables raise KeyError — symbolic executors that emit
        a variable per PUSH should pass one binding per PUSH.
        """
        total = 0
        for mono, coeff in self.terms.items():
            term = coeff
            for v, p in mono:
                term *= bindings[v] ** p
            total += term
        return int(total)

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


# ─── SymbolicExecutor ──────────────────────────────────────────────

# Opcodes this executor understands. Branchless straight-line polynomial-
# closed ops only; anything else raises on encounter so the caller sees
# a clean "out of scope" error instead of a silent mismatch.
_POLY_OPS = {
    isa.OP_PUSH, isa.OP_POP, isa.OP_DUP, isa.OP_HALT,
    isa.OP_ADD, isa.OP_SUB, isa.OP_MUL,
    isa.OP_SWAP, isa.OP_OVER, isa.OP_ROT, isa.OP_NOP,
}


@dataclass
class SymbolicResult:
    """Outcome of running a program symbolically.

    ``top`` is the polynomial left on top of the stack after HALT (or at
    the end of the trace if HALT is absent). ``stack`` is the full final
    stack (bottom at index 0). ``n_heads`` is the number of instructions
    executed — the "k heads" the issue talks about. ``bindings`` maps the
    allocated variable indices back to the original PUSH constants.
    """

    top: Poly
    stack: List[Poly]
    n_heads: int
    bindings: Dict[int, int]

    def collapse_ratio(self) -> float:
        """k_heads ÷ n_monomials in the top expression, after simplification.

        Matches the issue's "9 heads → 1 monomial" style report. Returns
        `inf` when the top collapses to zero (no monomials) — flagged by
        callers who want a cleaner representation.
        """
        n = self.top.n_monomials()
        if n == 0:
            return float("inf")
        return self.n_heads / n


class SymbolicStackUnderflow(RuntimeError):
    pass


class SymbolicOpNotSupported(NotImplementedError):
    pass


def run_symbolic(prog: List[isa.Instruction]) -> SymbolicResult:
    """Execute ``prog`` symbolically. One variable allocated per PUSH.

    Polynomial composition happens eagerly so the final form is already
    simplified — no separate simplification pass needed.
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
            stack.append(a + b)
        elif op == isa.OP_SUB:
            b = _pop(); a = _pop()
            stack.append(a - b)
        elif op == isa.OP_MUL:
            b = _pop(); a = _pop()
            stack.append(a * b)
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
    "SymbolicExecutor",
    "SymbolicResult",
    "SymbolicStackUnderflow",
    "SymbolicOpNotSupported",
    "run_symbolic",
    "collapse_report",
]
