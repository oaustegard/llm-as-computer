# Bit-vector extension for the symbolic executor + FF layer

_Issue #77 writeup. Follow-up to #69 (bilinear ADD/SUB/MUL), #75 (rational DIV_S/REM_S), and #76 (gated-bilinear comparisons)._

## The claim, in one sentence

For the bit-vector fragment `{AND, OR, XOR, SHL, SHR_S, SHR_U, CLZ, CTZ, POPCNT}` of the ISA, the symbolic executor carries a **`BitVec` AST** through the stack — same pattern `RationalPoly` / `IndicatorPoly` already use — and the FF layer realises each op as a **linear extractor + boundary bit step**, mirroring the DIV_S / REM_S / comparison construction.

## Why these ops are not polynomial

ADD/SUB/MUL close the integer polynomial ring `ℤ[x₀, x₁, …]`; that's what made #69 clean. The bit-vector ops don't:

- **Bitwise `AND`, `OR`, `XOR`** are polynomial over `ℤ/2ℤ` (the Boolean ring), **not** over `ℤ`. Moving to `ℤ` requires either a one-hot-per-bit embedding (`E_bits(v) = (v₀, v₁, …, v₃₁)`) or a boundary step back to the integer value. We pick the boundary route to keep the weight budget small; one-hot embedding is listed as a follow-up (it would make AND/OR/XOR closed in `ℤ[bits]`).
- **Shifts `SHL`, `SHR_S`, `SHR_U`** are *linear in the value* at a fixed shift amount — so `SHL(a, k) = 2ᵏ · a` on `ℤ`. But the shift amount `k` is itself the second operand, and raising 2 to a symbolic power leaves the polynomial ring. A bilinear form in `(a, 2ᵏ)` would close it, but only at the cost of an exponent-lookup path in the FF — another follow-up.
- **Bit counting `CLZ`, `CTZ`, `POPCNT`** is piecewise over every 32-bit input. `POPCNT(v) = Σᵢ bᵢ(v)` is linear in the bit decomposition, not in `v`. `CLZ` / `CTZ` are not linear anywhere. Table lookup (one-hot embedding + a constant matrix) works but is 32× wider than the scalar embedding; we leave that as a discussion point in §6 and ship the boundary-step version.

The pattern that rescued comparisons and division also rescues bit-vector ops: **keep the polynomial ring closed at the expression level, push the non-polynomial step to the boundary, make the boundary first-class rather than pretending it isn't there.**

## The three moves

### 1. Symbolic executor — new sibling type `BitVec`

`BitVec(op, operands)` is a frozen dataclass whose `eval_at` recursively evaluates each operand to an `int` and then applies the named bit op.

```python
@dataclass(frozen=True)
class BitVec:
    op: str                                   # "AND"/"OR"/"XOR"/"SHL"/...
    operands: Tuple[Union[Poly, "BitVec"], ...]

    def eval_at(self, bindings: Mapping[int, int]) -> int:
        vals = [o.eval_at(bindings) for o in self.operands]
        return _apply_bitop(self.op, vals)
```

Key design choices:

- **Recursive AST.** `operands` may contain nested `BitVec` nodes so that programs like `bit_extract = SHR_U; PUSH 1; AND` collapse to a single expression `BitVec("AND", (BitVec("SHR_U", (k, n)), Poly.constant(1)))`.
- **Structural equality.** `BitVec` is `@dataclass(frozen=True)`, so `==` compares `(op, operands)` tuples — the same value-based equality `Poly` / `RationalPoly` / `IndicatorPoly` already provide. Two symbolic executors emitting the same AST produce equal tops.
- **Boundary evaluation only.** The bit op fires once per `eval_at` call, on concrete integers. The AST is never simplified symbolically — a `BitVec("AND", (x, x))` is not auto-rewritten to `x`. That's intentional: simplification belongs to a Z/2ℤ algebra we're not building in this issue.
- **Composition closure.** Arithmetic on a `BitVec` top is *out of scope* for the polynomial ring — but `log2_floor` needs `SUB(31, CLZ(n))`. Rather than adding every arithmetic op to the `BitVec` AST, we do the minimal thing: treat `ADD`/`SUB`/`MUL` on a `BitVec` operand as lifting the whole expression into the `BitVec` AST (the result op string records the arithmetic). `log2_floor(n)` therefore collapses to `BitVec("SUB", (Poly.constant(31), BitVec("CLZ", (Poly.variable(0),))))`. The `SymbolicIntAst = Union[Poly, BitVec]` annotation makes the hybrid explicit; downstream comparisons and branches accept either.

### 2. `ArithmeticOps` gains nine bit primitives

Parallel to the `cmp_*` / `div_s` / `rem_s` hooks `run_forking`'s `arithmetic_ops` already carries, we add:

```python
@dataclass(frozen=True)
class ArithmeticOps:
    ...
    bit_and:   Callable[[SymbolicIntAst, SymbolicIntAst], BitVec] = None
    bit_or:    Callable[[SymbolicIntAst, SymbolicIntAst], BitVec] = None
    bit_xor:   Callable[[SymbolicIntAst, SymbolicIntAst], BitVec] = None
    bit_shl:   Callable[[SymbolicIntAst, SymbolicIntAst], BitVec] = None
    bit_shr_s: Callable[[SymbolicIntAst, SymbolicIntAst], BitVec] = None
    bit_shr_u: Callable[[SymbolicIntAst, SymbolicIntAst], BitVec] = None
    bit_clz:   Callable[[SymbolicIntAst], BitVec] = None
    bit_ctz:   Callable[[SymbolicIntAst], BitVec] = None
    bit_popcnt:Callable[[SymbolicIntAst], BitVec] = None
```

The default `DEFAULT_ARITHMETIC_OPS` instance wires each to a trivial `BitVec(...)` constructor. `ff_symbolic.FF_ARITHMETIC_OPS` wires them to the same symbolic primitives after (conceptually) flowing through the analytically-set FF matrices. The equivalence theorem at the `BitVec` level is structural equality of the resulting ASTs, same shape as #69's Poly equality.

### 3. `ff_symbolic` — linear extractor matrices plus boundary bit ops

Each bit-vector op decomposes into **linear extraction** (identical shape to DIV_S's pair-selector) followed by a **boundary bit step**:

| Op      | Weight tensor    | Shape        | Non-zero | Boundary step                                 |
| ------- | ---------------- | ------------ | -------: | --------------------------------------------- |
| Binary bit (`AND`/`OR`/`XOR`/`SHL`/`SHR_S`/`SHR_U`) | `M_BITBIN[0, DIM_VALUE] = 1`, `M_BITBIN[1, d+DIM_VALUE] = 1` | `(2, 2d)` | 2 | `_apply_bitop(op, [va, vb])` |
| Unary bit (`CLZ`/`CTZ`/`POPCNT`)                    | `M_BITUN[0, DIM_VALUE] = 1`                                 | `(1, d)`  | 1 | `_apply_bitop(op, [va])`     |

All six binary bit ops share one matrix (`M_BITBIN`); all three unary ops share one matrix (`M_BITUN`). The weight budget is **3 non-zero entries**, bringing the running total after #69 + #75 + #76 + #77 to **15**.

The boundary bit step reuses the existing `isa.py` helpers (`_to_i32`, `_shr_s`, `_shr_u`, `_clz32`, `_ctz32`, `_popcnt32`), which already encode WASM i32 semantics. The numeric path in `CompiledModel.forward` (lines 867-879) already applied those helpers; what this issue does is make it explicit at the FF boundary that they are the non-polynomial leaves of the operator tree, on par with `_trunc_div` (DIV_S) and `_relation_holds` (comparisons).

## Worked example: `bit_extract(n, bit_pos)`

Program: `PUSH n; PUSH bit_pos; SHR_U; PUSH 1; AND; HALT`.

```
After PUSH n:          stack = [x₀]                                      (Poly)
After PUSH bit_pos:    stack = [x₀, x₁]                                  (Poly)
After SHR_U:           stack = [BitVec("SHR_U", (x₁, x₀))]               (BitVec)
After PUSH 1:          stack = [BitVec(…), 1]                            (Poly const)
After AND:             stack = [BitVec("AND", (Poly.constant(1),
                                               BitVec("SHR_U", (x₁, x₀))))]
HALT                   top = AND(1, SHR_U(bit_pos, n))
```

`top.eval_at({0: n, 1: bit_pos})` computes `SHR_U(bit_pos, n)` (= `(n >> bit_pos) & 0xFFFFFFFF`) and then ANDs with 1 — the exact WASM i32 value of `(n >> bit_pos) & 1`, matching `NumPyExecutor`.

## Worked example: `log2_floor(n)`

Program: `PUSH n; CLZ; PUSH 31; SWAP; SUB; HALT`.

```
After PUSH n:          stack = [x₀]                                      (Poly)
After CLZ:             stack = [BitVec("CLZ", (x₀,))]                    (BitVec)
After PUSH 31:         stack = [BitVec("CLZ", (x₀,)), Poly.constant(31)] (hybrid)
After SWAP:            stack = [Poly.constant(31), BitVec("CLZ", (x₀,))]
After SUB:             stack = [BitVec("SUB", (BitVec("CLZ", (x₀,)),
                                               Poly.constant(31)))]
HALT                   top = SUB(31, CLZ(n))  (in WASM order: vb − va = 31 − CLZ)
```

The `SUB` lifts into the `BitVec` AST because one operand is a `BitVec`. This is the **SymbolicIntAst** composition closure — it mirrors the way `IndicatorPoly` consumes a non-Poly operand by wrapping it in a `Guard`.

`top.eval_at({0: n})` evaluates `CLZ(n)`, subtracts from 31, and returns the integer — matching `NumPyExecutor`.

## Catalog impact

Nine rows move from `blocked_opcode` to `collapsed`:

| Row                    | Program                              | Symbolic top             |
| ---------------------- | ------------------------------------ | ------------------------ |
| `bitwise_and(12, 10)`  | `PUSH 12; PUSH 10; AND; HALT`        | `AND(x₁, x₀)`            |
| `bitwise_or(12, 10)`   | `PUSH 12; PUSH 10; OR; HALT`         | `OR(x₁, x₀)`             |
| `bitwise_xor(12, 10)`  | `PUSH 12; PUSH 10; XOR; HALT`        | `XOR(x₁, x₀)`            |
| `native_clz(16)`       | `PUSH 16; CLZ; HALT`                 | `CLZ(x₀)`                |
| `native_ctz(8)`        | `PUSH 8; CTZ; HALT`                  | `CTZ(x₀)`                |
| `native_popcnt(13)`    | `PUSH 13; POPCNT; HALT`              | `POPCNT(x₀)`             |
| `bit_extract(5, 0)`    | `PUSH 5; PUSH 0; SHR_U; PUSH 1; AND; HALT` | `AND(1, SHR_U(x₁, x₀))` |
| `log2_floor(8)`        | `PUSH 8; CLZ; PUSH 31; SWAP; SUB; HALT`    | `SUB(CLZ(x₀), 31)` (WASM) |
| `is_power_of_2(8)`     | `PUSH 8; POPCNT; PUSH 1; EQ; HALT`   | `[POPCNT(x₀) − 1 == 0]` (IndicatorPoly wrapping a BitVec diff) |

`popcount_loop(5)` is a bounded loop: after concrete-mode unrolling it becomes a straight-line composition of `AND`, `SHR_U`, `ADD` at literal inputs, landing as `STATUS_COLLAPSED_UNROLLED`.

Width variations beyond i32 (i64 ops) and floating-point bit tricks remain out of scope, matching the issue's non-goals.

## Equivalence theorem for the bit-vector fragment

> For every bit-vector opcode `op ∈ {AND, OR, XOR, SHL, SHR_S, SHR_U, CLZ, CTZ, POPCNT}` and every integer-valued polynomial input, `run_symbolic` / `run_forking` using `DEFAULT_ARITHMETIC_OPS` produces a `BitVec` AST that is **structurally equal** to the one `ff_symbolic.evaluate_program` / `evaluate_program_forking` produces using `FF_ARITHMETIC_OPS`. Evaluating either AST at the catalog's concrete bindings returns the same integer that `NumPyExecutor` reports, modulo the standard i32 wrap caveat (`range_check` asserts in-range inputs for the equivalence claim).

The proof is the usual two-level argument:

1. **Unit level.** `symbolic_and(a, b) == BitVec("AND", (a, b))` on literal inputs; the analytically-set `M_BITBIN` matrix extracts `(va, vb)` exactly, and the boundary `_apply_bitop("AND", [va, vb])` returns `(va & vb)` on `ℤ_{i32}`. Same for the other eight bit primitives.
2. **Compositional level.** `run_forking`'s dispatch loop is indifferent to the concrete identity of the primitives — it just calls `ops.bit_and(a, b)`, `ops.bit_clz(a)`, etc. Plugging either `DEFAULT_ARITHMETIC_OPS` or `FF_ARITHMETIC_OPS` produces the same AST because both wire to a common `BitVec(...)` constructor.

The `test_ff_symbolic.test_equivalence_bitvec_*` tests pin this structural equality on every new catalog row. Numeric agreement with `NumPyExecutor` is the separate `_numeric` test family.

## Non-goals (explicit follow-ups)

- **Polynomial algebra over `ℤ/2ℤ`.** A one-hot-per-bit embedding `E_bits(v) = (b₀, b₁, …, b₃₁)` would make `AND`, `OR`, `XOR` closed under Poly multiplication / addition / negation respectively, turning the current AST into a genuine polynomial over `(ℤ/2ℤ)[bits]`. Worth doing if a later catalog program needs structural simplification of `AND`/`OR` chains. 32× width increase in the embedding.
- **Bilinear shifts with exponent lookup.** `SHL(a, k) = 2ᵏ · a` is bilinear in `(a, 2ᵏ)`. A bilinear form would realise this if the FF had an exponent-lookup path emitting `2ᵏ` from the shift-amount embedding. Straightforward mechanically but another matrix + a 32-entry lookup table.
- **One-hot CLZ / CTZ / POPCNT.** The same one-hot-per-bit embedding above would let a single bilinear form compute all three without a boundary step — POPCNT becomes a sum, CLZ / CTZ become the index of the first / last set bit. Delayed for the same reason: cost is 32× embedding width.
- **ROTL / ROTR.** Not in the roadmap's listed acceptance criteria and not in any currently-catalogued program. Mechanically the same as SHL / SHR_U with an extra boundary helper.
- **64-bit (i64) variants.** Out of scope per the issue's "Non-goals" section.

## What this issue *is* worth, in one line

The bit-vector fragment now fits the same spec shape that carried ADD/SUB/MUL, DIV_S/REM_S, and the comparisons: **linear extraction at the weight layer, non-polynomial step at the boundary, a first-class sibling type on the symbolic stack.** Shipping this fragment is the fourth and largest confirmation that the FF-layer-as-polynomial-evaluator story scales past its original polynomial-ring scope — without pretending the non-polynomial steps aren't there.
