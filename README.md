# llm-as-computer

Testing [Percepta's claim](https://percepta.ai/blog/can-llms-be-computers) that transformers can execute programs internally via 2D convex hull attention, achieving O(log t) per-step decoding over million-step execution traces.

**Writeup:** [WRITEUP.md](WRITEUP.md)

## What is this?

Percepta published a blog post (Mar 11, 2026) describing how they embedded a WebAssembly interpreter inside a vanilla transformer by:

1. Restricting attention head dimension to 2D
2. Using parabolic key encoding k_j = (2j, −j²) for memory lookups
3. Replacing linear attention scans with convex hull queries for O(log t) decoding
4. Tracking state (instruction pointer, stack depth) via cumulative sum attention

They released no code or weights. This repo independently tests whether the core primitives work as described — and ultimately validates the approach by building a working compiled transformer executor.

## Status

### Foundation: Validating the Primitives (Phases 1–4)

| Phase | Description | Status | Key Finding |
|-------|------------|--------|-------------|
| 1 | Convex hull KV cache scaling | Done | O(log t) confirmed via ternary search. 35x speedup at 50K steps. |
| 2 | Parabolic key encoding precision | Done | Float32 breaks at index ~4K (revised). Float64 safe to ~200K+. |
| 2b | Extended addressing | Done | Residual (bit-split) addressing: 25M range from 2 heads. |
| 3 | Cumulative sum via attention | Done | Rock-solid — zero integer errors at 100K in float32. |
| 4 | Hand-wired stack machine | Done | Primitives compose. 10/10 test programs execute correctly via attention only. |

### Detour: Can Gradient Descent Learn Execution? (Phases 5–9)

Phases 5–9 explored whether a small transformer could *learn* to execute programs from training data alone. This was a deliberate departure from Percepta's compile approach. The journey was instructive — it precisely characterized *why* compilation is necessary.

| Phase | Description | Status | Key Finding |
|-------|------------|--------|-------------|
| 5 | Trained micro-executor | Done | Learns execution structure (56% token acc, 112x chance) but 0/50 perfect traces. Width > depth. |
| 6 | Curriculum learning | Done | 56%→85% acc, 0→39/50 perfect traces. Solves copy bottleneck and non-arithmetic ops. |
| 7 | Percepta architecture test | Done | Percepta's d=36/h=18/L=7 matches Phase 6 (84.6% acc) but doesn't break the DIFF+ADD wall (0%). |
| 8 | Micro-op trace diagnostics | Done | **THE key diagnostic phase.** Retrieval is 100% solved; arithmetic (a+b for a≠b) is the sole bottleneck. |
| 9 | Weighted arithmetic loss | Done | Weighted loss perfects doubling (2a→100%) but true addition (a+b, a≠b) stays at 0%. It's representational, not gradient signal. |

**Conclusion from Phases 5–9:** Training alone cannot learn true integer addition within execution context. The DIFF+ADD wall is a representational limit of FF layers trying to simultaneously learn arithmetic and routing. This sent us back to Percepta's original insight: *compile, don't train.*

### Back on Track: Compiled Execution (Phases 11–13)

Rereading Percepta's blog post clarified the divergence. Percepta **compiles** interpreter logic directly into weight matrices; we had been **training** via gradient descent. Phases 11–13 return to the compile path and validate it completely.

| Phase | Description | Status | Key Finding |
|-------|------------|--------|-------------|
| 11 | Compiled executor (numpy) | Done | 100% correct traces. Extended ISA with SUB/JZ/JNZ enables loops and control flow. |
| 12 | Real PyTorch compiled transformer | Done | 100% correct via real nn.Linear weight matrices and tensor ops (matmul, argmax). 758 compiled params. |
| 13 | ISA completeness | Done | SWAP/OVER/ROT + Fibonacci, multiply, power-of-2, sum, parity. Forth-equivalent ISA. 964 compiled params. |

## Files

```
CLAUDE.md                       # Project instructions and full phase history
RD-PLAN.md                      # R&D plan and evolution
FINDINGS.md                     # Detailed findings from all phases
phase1_hull_cache.py            # Convex hull vs brute force benchmarks
phase2_parabolic.py             # Parabolic encoding precision tests
phase2b_address_limits.py       # Extended addressing exploration
phase3_cumsum.py                # Cumulative sum stability tests
phase4_stack_machine.py         # Stack machine via attention primitives
phase5_training.py              # Training experiments (3 model configs)
phase6_curriculum.py            # Curriculum learning experiment
phase7_percepta_arch.py         # Percepta architecture (d=36,h=18,L=7) test
phase8_microop_traces.py        # Micro-op decomposition diagnostics
phase9_weighted_arithmetic.py   # Weighted loss experiments
phase10_digit_decomposition.py  # Digit decomposition (exploratory)
phase11_compile_executor.py     # Compiled executor with numpy primitives
phase12_percepta_model.py       # Full PyTorch compiled transformer
phase13_isa_completeness.py     # ISA completeness: SWAP/OVER/ROT + algorithms
viz/phase1-results.jsx          # Phase 1 interactive visualization (React)
```

## Running

```bash
# Foundation phases
python3 phase1_hull_cache.py       # ~60s, benchmarks query scaling
python3 phase2_parabolic.py        # ~10s, finds float32 breakpoint
python3 phase3_cumsum.py           # ~10s, tests numerical drift
python3 phase4_stack_machine.py    # ~1s, stack machine composition test

# Training experiments (Phases 5-9, requires torch)
python3 phase5_training.py         # ~5min+ CPU, trains 3 model configs
python3 phase6_curriculum.py       # ~2min CPU, 3-stage curriculum
python3 phase7_percepta_arch.py    # Percepta architecture comparison
python3 phase8_microop_traces.py   # Micro-op diagnostics
python3 phase9_weighted_arithmetic.py  # Weighted loss experiments

# Compiled execution (Phases 11-13)
python3 phase11_compile_executor.py    # ~1s, compiled executor tests
python3 phase12_percepta_model.py      # ~1s, PyTorch compiled transformer
python3 phase13_isa_completeness.py    # ~1s, ISA completeness + algorithms
```

Requires: numpy, scipy (for convex hull verification), torch (for Phases 5-9 and 12-13).

## Key Takeaways

- **The geometry works.** Parabolic keys + ternary search give exact O(log n) index lookup.
- **Float32 is the bottleneck — but solvable.** ~4K addressable indices per parabolic head. Residual (bit-split) addressing with 2 heads extends this to 25M.
- **Cumsum is not the weak link.** Both the mean×t trick and sequential lookback are stable far beyond expected.
- **The primitives compose.** Phase 4 proves parabolic indexing, recency bias, and sequential state tracking work together as a stack machine executor.
- **Training alone hits a wall.** Phases 5–9 showed that gradient descent can learn execution *structure* (85% accuracy, 39/50 perfect traces) but cannot learn true integer addition (a+b for a≠b) within execution context. The bottleneck is representational, not capacity or gradient signal.
- **Compile, don't train.** Returning to Percepta's original approach in Phases 11–13 validates their core claim: when arithmetic and routing logic are compiled directly into weight matrices, the transformer executes correctly — including true addition, subtraction, branching, and loops.
- **The result is a general-purpose stack computer.** Phase 13's 12-opcode, 5-head, 964-parameter compiled transformer is Forth-equivalent, executing Fibonacci, multiplication, power-of-2, summation, and parity checks correctly.
