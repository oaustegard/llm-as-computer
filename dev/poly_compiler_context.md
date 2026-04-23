# poly_compiler_context.md

Pre-digested reference for poly_to_program implementation.
DO NOT read symbolic_executor.py (50K+ tokens). This file has everything needed.

## Opcode constants (from isa.py)

```python
OP_PUSH = 1
OP_POP  = 2
OP_ADD  = 3
OP_DUP  = 4
OP_HALT = 5
OP_SUB = 6
OP_JZ  = 7
OP_JNZ = 8
OP_NOP = 9
OP_SWAP = 10
OP_OVER = 11
OP_ROT  = 12
OP_MUL   = 13
OP_DIV_S = 14
OP_DIV_U = 15
OP_REM_S = 16
OP_REM_U = 17
OP_EQZ   = 18
OP_EQ    = 19
OP_NE    = 20
OP_LT_S  = 21
OP_LT_U  = 22
OP_GT_S  = 23
OP_GT_U  = 24
OP_LE_S  = 25
OP_LE_U  = 26
OP_GE_S  = 27
OP_GE_U  = 28
OP_AND   = 29
OP_OR    = 30
OP_XOR   = 31
OP_SHL   = 32
OP_SHR_S = 33
OP_SHR_U = 34
OP_ROTL  = 35
OP_ROTR  = 36
OP_CLZ    = 37
OP_CTZ    = 38
OP_POPCNT = 39
OP_ABS    = 40
OP_NEG    = 41
OP_SELECT = 42
OP_LOCAL_GET = 43
OP_LOCAL_SET = 44
OP_LOCAL_TEE = 45
OP_I32_LOAD    = 46
OP_I32_STORE   = 47
OP_I32_LOAD8_U = 48
OP_I32_LOAD8_S = 49
OP_I32_LOAD16_U = 50
OP_I32_LOAD16_S = 51
OP_I32_STORE8  = 52
OP_I32_STORE16 = 53
OP_CALL   = 54
OP_RETURN = 55
OP_TRAP  = 99
```

## Instruction class (from isa.py)

```python
class Instruction:
    op: int
    arg: int = 0

    def __repr__(self):
        name = OP_NAMES.get(self.op, f"?{self.op}")
        if self.op in (OP_PUSH, OP_JZ, OP_JNZ,
                        OP_LOCAL_GET, OP_LOCAL_SET, OP_LOCAL_TEE,
                        OP_CALL):
            return f"{name} {self.arg}"
        return name
```

## Poly class (from symbolic_executor.py)

```python
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
```

## _POLY_OPS (from symbolic_executor.py)

```python
_POLY_OPS = {
    isa.OP_PUSH, isa.OP_POP, isa.OP_DUP, isa.OP_HALT,
    isa.OP_ADD, isa.OP_SUB, isa.OP_MUL,
    isa.OP_DIV_S, isa.OP_REM_S,
    isa.OP_SWAP, isa.OP_OVER, isa.OP_ROT, isa.OP_NOP,
} | _CMP_OPS | _BITVEC_OPCODES
```

## SymbolicResult (from symbolic_executor.py)

```python
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
```

## run_symbolic semantics (key behaviors)

```
PUSH arg  → allocate fresh variable x_i (next_var++), push Poly.variable(i)
            Two PUSHes with same concrete value produce DISTINCT variables.
POP       → pop top
DUP       → copy top (same Poly reference, NOT a new variable)
SWAP      → swap top two
OVER      → copy second-from-top to top
ROT       → [a, b, c] → [b, c, a]  (depth-2 element to top)
ADD       → pop b, pop a, push a + b  (Poly addition)
SUB       → pop b, pop a, push a - b  (Poly subtraction)
MUL       → pop b, pop a, push a * b  (Poly multiplication)
HALT      → stop execution, top of stack is result
```

## Round-trip invariant

```python
run_symbolic(poly_to_program(p)).top == p
```
