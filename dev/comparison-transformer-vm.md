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
