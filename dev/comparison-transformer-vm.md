# LAC vs. Percepta-Core/transformer-vm — Comparative Review

**Repos compared**

- **LAC** — `oaustegard/llm-as-computer` @ `main` (73 files, 650 symbols)
- **TVM** — `Percepta-Core/transformer-vm` @ `main` (39 files, 102 symbols)

LAC is an *independent re-derivation* built to validate the claim in
Percepta's "Can LLMs Be Computers?" blog post. TVM is Percepta's own
reference implementation released alongside the blog. Both compile
programs into transformer weights and execute them via 2D parabolic-key
attention, but they differ substantially in scope, architecture, and
engineering style.

---

## 1. Origin and intent

| | LAC | TVM |
|---|---|---|
| Role | Independent validation | Canonical reference |
| Posture | Research repo, phase-numbered exploration | Productized library + CLI |
| Documentation | `README`, `HOW-IT-WORKS`, `WRITEUP`, `CLAUDE.md`, `dev/FINDINGS.md`, per-phase write-ups | `README` only (mermaid file map) |
| Story arc | 13 phases, including a productive wrong-turn through gradient training | Direct: build the analytical machine |

LAC's narrative payoff is the discovery that compiling beats training
(Phases 5–11 walk through training failure → compile success). TVM
skips that arc — the whole repo presupposes the answer.

## 2. Front-end / source language

| | LAC | TVM |
|---|---|---|
| ISA | Custom 55-opcode stack machine (WASM-flavored) | Standard WebAssembly MVP, 35 opcodes |
| Compiler input | C → WAT → ISA (`c_pipeline.py`, `wat_parser.py`, `assembler.py`) | C → wasm32 binary → token prefix (`compilation/decoder.py`, `compile_wasm.py`) |
| Lowering | Native opcodes for MUL/DIV/AND/OR/XOR/SHL/SHR/CLZ/CTZ/POPCNT/ROTL/ROTR | `compilation/lower.py` rewrites MUL/DIV/MOD/AND/OR/XOR/SHL/SHR into ADD/SUB sequences |
| Toolchain | Hand-written WAT parser; no external toolchain dependency | Requires `clang --target=wasm32` + `lld`; uses real WASM binaries |

TVM is the more *honest* claim — it runs unmodified `clang -Owhatever`
WASM. LAC takes the shorter path of compiling C to WAT and then to its
own richer ISA, which avoids the lowering complexity but means it
isn't running "real" WebAssembly.

## 3. Execution architecture

Both use the same core trick — 2D parabolic keys
`k = (2j, −j²)` so that `q · k` peaks at the queried address — but
they wire it up very differently.

### LAC — explicit hand-wired model

`executor.py` defines three flat executors: `NumPyExecutor`,
`CompiledModel` (real `nn.Linear`), and `TorchExecutor`. Each opcode
class has a hand-coded handler. Address spaces (program, stack,
locals, heap, call stack) are kept as parallel `keys[]/vals[]` lists,
each tagged with its own dimension flag (`DIM_IS_PROG`,
`DIM_IS_STACK`, `DIM_IS_HEAP`, …). Attention heads (4–5 active,
14 reserved slots) read the program counter, opcode argument, top of
stack, SP-1, SP-2. FF dispatch is a bilinear gate (opcode one-hot ×
value routing matrix). Total: 964 compiled parameters.

There's also a constellation of *symbolic* siblings —
`symbolic_executor.py`, `forking_executor.py`, `ff_symbolic.py`,
`ff_symbolic_recurrent.py`, `closed_form.py`, `path_b.py`,
`symbolic_programs_catalog.py` — exploring symbolic execution and
"the FF weights *are* the polynomial" equivalence proofs. None of
this exists in TVM.

### TVM — graph DSL + MILP scheduler

The architecture is layered:

1. **`graph/core.py`** — a small symbolic algebra DSL: `Expression`,
   `Dimension`, plus five primitive `Dimension` subclasses
   (`InputDimension`, `ReGLUDimension`, `PersistDimension`,
   `LookUpDimension`, `CumSumDimension`) and helper builders
   (`reglu`, `stepglu`, `persist`, `fetch`, `fetch_sum`).
2. **`wasm/interpreter.py`** — defines the *behavior* of all 35
   WASM opcodes by composing graph primitives. There is no
   per-opcode `if/elif`; the whole machine is a DAG of dimensions.
   Notably, opcode dispatch uses **64 points on a circle of
   radius² = 32045** (`pointsR2`, lines 14–84) so each opcode can be
   detected by a single ReGLU neuron via dot product.
3. **`scheduler/milp.py`** — a MILP solver (`uv` pulls in
   pulp/HiGHS-style deps) assigns gate-to-layer to minimize layers
   subject to dependency and pathwidth constraints.
4. **`model/weights.py`** — emits actual PyTorch tensors from the
   scheduled graph. `HARD_K = 1e10` rescales softmax to approximate
   argmax (transformer-vm stays softmax everywhere; LAC uses literal
   `argmax`).
5. **`model/transformer.py` + `transformer.cpp`** — runs the result
   either in PyTorch or in a standalone C++ engine with BLAS.

The conceptual gap is large: LAC writes interpreters, TVM writes a
compiler from a computation-graph IR to transformer weights.

## 4. Hardmax cache (the O(log n) trick)

Both repos implement the same convex-hull KV cache.

| | LAC | TVM |
|---|---|---|
| Implementation | Python (`forking_executor.py` and friends) + Mojo backend (`src/executor.mojo`) | C++ (`attention/hull2d_cht.h`, `hull_ext.cpp`) bound via pybind11; Python wrapper (`hull_cache.py`) |
| Algorithm | Ternary search over parabolic keys | Convex Hull Trick (CHT) with insert/query in `O(log n)` |
| Tie-break | `eps * write_count` injected into key.y | Explicit `LATEST_ALPHA = 0.3` on `inv_log_pos`, plus `tie_break={"latest","average"}` API |

Both maintain a true incremental hull. TVM's data structure is
slightly more general (it exposes both "latest" and "average"
tie-break modes; LAC always picks latest). LAC ships a Mojo backend
(67–126 M steps/sec) that TVM has no equivalent of — TVM relies on
C++ + BLAS instead (≈30 K tok/s end-to-end, but tokens-per-step is
much higher because each token is one byte of state).

## 5. Specialization (Futamura projection)

Both repos implement first-Futamura specialization — baking a
specific program into the FFN weights and removing the program prefix
from the input.

| | LAC | TVM |
|---|---|---|
| File | `specialize.py` (~70 lines, NumPy buffer overlay) | `specialize.py` + `wasm/interpreter.py` (graph rebuilt with `program=...`) |
| Mechanism | `SpecializationFFN` overlays compile-time fetches; `build_specialized_model()` snaps weights | Graph builder takes `program` arg; `op_dot()` becomes constant; whole circuit collapses |
| CLI | `test_specialize.py` script | `wasm-specialize` CLI command, saves binary weights for the C++ engine |

TVM's approach is cleaner because specialization falls out of the
graph DSL almost for free — the same `build()` is reused with a
constant program. LAC's specialization is bolted onto the hand-wired
executor.

## 6. Tokens and tracing

This is the most subtle architectural divergence.

- **LAC** records executions as a `Trace` of `TraceStep(op, arg, sp,
  top)` — 4 tokens per instruction. The transformer is run in lockstep
  with that trace format. It is *not* doing autoregressive byte-level
  generation in the transformer-as-CPU sense.
- **TVM** generates one token per **byte** of machine state (stack
  values, memory, output). A program that prints "hello" autoregresses
  through tens of thousands of byte tokens. Carry propagation between
  bytes is part of the graph (`carry` InputDimension, `byte_number`).
  Sudoku takes ~900 K tokens.

TVM is closer to the spirit of "the transformer *is* the CPU" because
the entire microarchitectural state (including carries between bytes
of a 32-bit add) lives in the token stream. LAC operates on whole
i32s per step, which is faster and simpler to reason about, but
hides arithmetic inside the FF rather than letting it ride the
attention mechanism.

## 7. Engineering style

| | LAC | TVM |
|---|---|---|
| Layout | Mostly flat at root, ~30 top-level files; `dev/` for phase scripts; `src/` for Mojo | Clean package: `graph/`, `wasm/`, `model/`, `scheduler/`, `attention/`, `compilation/`, `examples/`, `tests/` |
| Build/deps | `requirements.txt`, ad-hoc | `pyproject.toml`, `uv.lock`, `pre-commit-config`, GitHub CI badge, `ruff`, CMake for pybind11 |
| Test runner | Eight `test_*.py` scripts each with their own `main()` | `pytest`, `pytest -m "not slow"` split |
| Entry points | Direct `python file.py` invocations | `wasm-run`, `wasm-eval`, `wasm-compile`, `wasm-build`, `wasm-specialize`, `wasm-reference` console scripts |
| LOC density | 73 files / 650 symbols (~9/file) | 39 files / 102 symbols (~2.6/file) |

TVM is the more polished library. LAC is denser per file and shows
its research-repo origins (multiple parallel executors, symbolic
sidelines, phase artifacts in `dev/phases/`).

## 8. Examples and benchmarks

Overlapping: **Sudoku, Fibonacci, Hungarian/min-cost matching**
appear in both.

LAC-only: FNV-1a hash, bubble sort, sum-of-primes, parity, true
Mojo-vs-Python-vs-native benchmarks (`llm_vs_native.py`,
`benchmark_scaling.py`), million-step scaling test.

TVM-only: Collatz, "addition" (long-arithmetic carry showcase),
hello-world via printf, end-to-end Sudoku via 900 K-token autoregress.

LAC publishes raw throughput numbers (67–126 M steps/sec in Mojo,
2.1–3.1 M in Python); TVM reports ~30 K tok/s end-to-end through its
C++ engine — apples-and-oranges because of the tokens-per-step
difference noted above.

## 9. What each repo does that the other does not

**Only in LAC**

- Mojo backend with massive throughput claims.
- Symbolic execution stack (`symbolic_executor.py`,
  `forking_executor.py`, `ff_symbolic.py`, `closed_form.py`,
  `algebraic_poly.py`, `modpoly.py`, `poly_compiler.py`).
- Documented training-vs-compiling investigation (Phases 5–10
  showing what doesn't work, then Phase 11 showing what does).
- Per-phase test files and a "symbolic collapse report" classifier.
- `viz/` React visualizations.
- Honest LLM-vs-native benchmark.

**Only in TVM**

- General-purpose computation-graph DSL (`graph/core.py`) — programs
  are graphs, not handlers.
- MILP scheduler that minimizes layer count.
- Real WASM binary input (clang/wasm32) instead of a custom ISA.
- Standalone C++ inference engine with hull-cache pybind11 bindings.
- Byte-level tokenization with explicit carry propagation.
- Production-grade packaging (uv, ruff, pre-commit, CI, console
  scripts, mermaid file map).
- Circle-point opcode dispatch (single-neuron detection per opcode).

## 10. Verdict

The two implementations are complementary, not duplicative:

- **TVM is the formal demonstration**: a clean compiler from a
  computation-graph IR to weights, running real WASM, with a serious
  C++ inference engine. Read it to learn the *theory*.
- **LAC is the empirical investigation**: many parallel executors
  (numpy, torch, Mojo, symbolic, forking), a documented exploration
  of what compiles vs. what doesn't, and aggressive throughput
  engineering. Read it to learn what *fails* and what *scales*.

LAC validates Percepta's claim and then takes it further in two
directions TVM doesn't touch: (a) high-throughput compiled execution
(Mojo) and (b) symbolic-equivalence analysis of the compiled FF
(showing the weights literally *are* the polynomial they compute).
TVM's ceiling is higher in scope (full WASM, byte-level autoregress,
sudoku in 900 K tokens of one autoregressive forward pass); LAC's
ceiling is higher in throughput per "step" because it abstracts above
byte-granularity.

If the goal is "convince a skeptic the claim is real," TVM is the
artifact to point at. If the goal is "understand why and where this
breaks down vs. when compiled," LAC is the better-documented
journey.

## 11. Honest reckoning — the symbolic stack

Section 10 frames LAC's "symbolic-equivalence analysis of the compiled
FF" as one of two clear wins over TVM. After actually reading the
~6,600 lines that make up the symbolic stack
(`symbolic_executor.py`, `forking_executor.py`, `ff_symbolic.py`,
`closed_form.py`, `algebraic_poly.py`, `modpoly.py`, `poly.py`,
`poly_compiler.py`, `symbolic_programs_catalog.py`, `path_b.py`,
`guarded.py`, `symbolic_types.py`, `bitvec.py`, plus the recurrent
and poly-embedding variants), that framing needs a softer
qualification. The stack is good research output, but most of its
"value-add over TVM" disappears once you look at TVM's graph DSL.

### 11.1 What it really delivers

Three things are load-bearing. Everything else is chrome.

**(a) Closing the Python-arithmetic gap.** Before issue #69, the FF
dispatch in `executor.CompiledModel.forward` had this fallback:

```python
nonlinear[OPCODE_IDX[OP_ADD]] = float((va + vb) & MASK32)
nonlinear[OPCODE_IDX[OP_SUB]] = float((vb - va) & MASK32)
nonlinear[OPCODE_IDX[OP_MUL]] = float((va * vb) & MASK32)
```

The transformer was *routing* arithmetic that CPython performed.
`ff_symbolic.py` replaced that with `M_ADD`, `M_SUB`, `B_MUL` —
analytically-set linear and bilinear forms — and proved (by
construction plus catalog cross-check) that the same operator tree,
re-interpreted over `Poly` instead of `torch.Tensor`, yields the
polynomial that `run_symbolic` emits. Without `ff_symbolic`, the
compile claim has a Python-shaped hole. **This is the one piece that
genuinely earns the "weights ARE the polynomial" slogan.**

**(b) A regression suite with canonical normal forms.** The catalog
runner classifies 43 programs into collapsed (28) / guarded (4) /
unrolled (5) / closed-form (4) / blocked (2) and prints the canonical
`Poly` / `GuardedPoly` / `ClosedForm` / `ProductForm` for each. If
you change FF dispatch, embedding scheme, or add an opcode, the
catalog flags movement at the *semantic* level, not just the trace
level. This is the most practically useful piece day-to-day.

**(c) A real closed-form loop solver.** `forking_executor` walks a
symbolic loop body, classifies the recurrence (affine → Faulhaber
polynomial inside `Poly`; constant-integer matrix → `ClosedForm`;
multiplicative over a Poly factor → `ProductForm`), and emits a top
that evaluates without unrolling. `fibonacci_sym(n)`,
`power_of_2_sym(n)`, `factorial_sym(n)`, `sum_1_to_n_sym(n)` all
collapse this way. This is genuine algorithmic content — actual
computer-algebra plumbing, not just bookkeeping.

### 11.2 What it doesn't deliver, and the chrome

- It doesn't make execution faster. `eval_at` walks the loop in
  Python; matrix-power squaring isn't even used because the catalog
  caps trip counts at n ≤ 32.
- It doesn't extend the ISA. It runs *parallel* to `executor.py`,
  not as an enabler.
- It doesn't generate weights — it *characterises* them.
- It doesn't help training (training was abandoned for compile).

The chrome:

- `poly_compiler.poly_to_program` (Poly → branchless LAC program with
  round-trip `run_symbolic(poly_to_program(p)).top == p`) is a
  curiosity. The catalog already tells you which polynomials map to
  which programs.
- `algebraic_poly` (Path B.3, Binet's formula via ℚ(√5)) deliberately
  reopens a "no" decision from #89 to get a single-layer Fibonacci
  realisation. Cute but narrow — Fibonacci only.
- `bitvec`, `modpoly`, `RationalPoly`, `IndicatorPoly` extend the
  type ring to bitwise / mod / division / comparison opcodes, but
  composition past one such op raises `BlockedOpcodeForSymbolic`.
  The symbolic ring closes one layer thick, not arbitrarily.
- `ff_symbolic_recurrent.py`, `ff_symbolic_poly_embedding.py`,
  `path_b.py` are exploratory sub-paths.

The honest characterisation: this is a *verification and
characterisation* layer, not an *enablement* layer.

### 11.3 Would it port to transformer-vm?

Mostly no, with one piece worth lifting — and not from LAC.

**TVM already has the formal spec for free.** LAC needed
`ff_symbolic` because `CompiledModel.forward` had a Python-arithmetic
fallback. TVM has no such fallback. `transformer_vm/graph/core.py`
*is* a symbolic algebra DSL — `Expression`, `ReGLUDimension`,
`LookUpDimension`, `PersistDimension`, `CumSumDimension` —
and `transformer_vm/model/weights.py` analytically materialises
tensors from that graph. The "weights ARE the spec" property is
*structural* in TVM, not a side proof. There is no
Python-arithmetic hole to plug.

**Granularity mismatch breaks the polynomial framing.** LAC operates
at i32-value granularity, so `PUSH 3 ; PUSH 5 ; ADD ; HALT` cleanly
becomes `x0 + x1`. TVM tokenises one token per *byte*, with explicit
carry propagation (`carry` is a top-level `InputDimension`, see
`transformer_vm/wasm/interpreter.py:223`). A 32-bit add is a
sequence of byte tokens with carries chained between them. Recovering
a value-level polynomial in TVM means either (a) re-deriving it from
the *original* WASM (in which case you've reimplemented Binaryen /
wasm-opt / Souper), or (b) writing a substantial abstraction layer
over `graph/core.py` to lift byte semantics back to value semantics.
Neither is small.

**TVM's lowered ISA loses the polynomial-closed core.** TVM's
`compilation/lower.py` rewrites `MUL`, `DIV`, `MOD`, `AND`, `OR`,
`XOR`, `SHL`, `SHR` into `ADD/SUB` sequences at compile time. The
polynomial-closed core in LAC's catalog (which includes native `MUL`)
doesn't translate at all. You'd be analysing a long-arithmetic
unrolled trace, not a polynomial.

**The one piece worth lifting, at the WASM level not the transformer
level.** The recurrence-classifier-and-solver in `forking_executor`
is the only piece of LAC's symbolic stack that does nontrivial
*computation* rather than recording structure. If TVM wants to serve
`factorial(20)` *without* 900 K tokens of autoregress, it should
recognise the loop as a `ProductForm`, replace it with the constant
`2432902008176640000`, and emit. But the right place for that is the
*compilation pipeline* (`transformer_vm/compilation/`), operating on
WASM IR before lowering — not on the transformer graph after weights
are built. And it's a job for an existing optimiser pass (Souper,
LLVM polly, Binaryen) more than a custom Python-and-Poly stack
ported wholesale.

### 11.4 Revised verdict

Section 10 listed two things LAC does that TVM doesn't:
"(a) high-throughput compiled execution (Mojo) and (b)
symbolic-equivalence analysis of the compiled FF."

(a) holds up. Mojo at 67–126 M steps/sec is a real, measurable win
that TVM has no equivalent of.

(b) needs revision. The symbolic-equivalence analysis was *necessary*
for LAC because LAC had a Python-arithmetic gap to close. TVM didn't
have that gap to begin with. Saying "LAC does this and TVM doesn't"
is technically true but misleading — it's like noting that a guard
dog has a job a guard rail doesn't, while ignoring that the guard
rail makes the dog unnecessary. The catalog regression suite and the
closed-form loop solver are useful research artefacts, but they're
verification machinery, not capability extensions, and the closed-form
solver is misplaced — it would do more good upstream of TVM, in
standard WASM optimisation passes, than as a port.

The corrected single-sentence summary: **the symbolic stack is good
*for LAC* because it patches a real gap and gives a regression
harness; it doesn't compose cleanly with TVM's architecture, and
porting it would mostly reimplement structure TVM already has.**

---

## Appendix — file pointers

**LAC core**: `isa.py:1`, `executor.py:56` (`NumPyExecutor`),
`executor.py:524` (`CompiledModel`), `executor.py:1043`
(`TorchExecutor`), `assembler.py:52` (`compile_structured`),
`c_pipeline.py:168`, `wat_parser.py:508` (`parse_wat`),
`specialize.py:104` (`specialize`), `forking_executor.py:221`
(`run_forking`), `src/executor.mojo` (Mojo backend).

**TVM core**: `transformer_vm/graph/core.py:23` (`Expression`),
`transformer_vm/graph/core.py:328` (`fetch`),
`transformer_vm/wasm/interpreter.py:14` (circle points),
`transformer_vm/wasm/interpreter.py:198` (`build`),
`transformer_vm/scheduler/milp.py`,
`transformer_vm/model/weights.py:22` (`HARD_K`),
`transformer_vm/attention/hull2d_cht.h:144` (`add_line`),
`transformer_vm/specialize.py:86` (`specialize`),
`transformer_vm/runner.py:24` (`run_model_program`),
`transformer_vm/compilation/lower.py:1159` (`lower_hard_ops`).
