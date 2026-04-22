# LLM-as-Computer

Research repo exploring whether vanilla transformer primitives can implement a working computer (stack machine). Inspired by Percepta's "Can LLMs Be Computers?" blog post (Mar 2026).

## Muninn Boot

This repository is developed by Oskar Austegard using Claude sessions sharing persistent memory via Muninn. Boot loads profile, operational context, and prior findings into the session.

**Boot is automatic.** The SessionStart hook (`.claude/hooks/session-start.sh`) runs `boot()` at the beginning of every Claude Code on the web session.

Credentials auto-detect from environment or well-known paths (`/mnt/project/turso.env`, `/mnt/project/muninn.env`, `~/.muninn/.env`). If boot fails, the hook logs a warning and continues.

### Decision Traces

After completing meaningful work (new phase, training run, key finding), store a memory:

```python
remember(
    "Phase N result: [what was found]. Key insight: [what it means]. "
    "Constraint: [if any]. Next: [what follows].",
    "analysis",
    tags=["LLM", "architecture", "research", "phase-N"],
    priority=1
)
```

Lead with *why*, not *what* — the diff shows what. Include surprises, rejected approaches, and architectural implications.

### Regular Memory Updates

Store progress updates to Muninn **regularly during work**, not just at the end. Any time you complete a meaningful sub-task, hit a blocker, or discover something unexpected, call `remember()` so future sessions have context. Aim for at least one memory store per significant milestone (e.g., "tests passing", "new opcode working", "design decision made"). This protects against session cutoffs losing context.

## Project Context

### What This Is

A bottom-up validation of whether transformer attention + FF layers can implement program execution. Each phase isolates a primitive, tests it numerically, then composes with prior phases.

### Key Architectural Insight

**Attention is lookup; feed-forward is routing.** Attention is cheap and reliable (pattern matching, memory addressing). FF layers must learn crisp categorical decisions (opcode dispatch) — this is the hard part. Width > depth for learning execution.

### Parabolic Encoding

The workhorse primitive: `k = (2j, -j²)` encodes position j such that dot-product attention peaks sharply at the target position. Same encoding addresses both program memory and stack memory without interference. Phase 2b extended this past float32 limits via residual (bit-split) addressing.

## Phases

| Phase | File | Status | What It Proves |
|-------|------|--------|----------------|
| 1 | phase1_hull_cache.py | Complete | O(log t) lookup via ternary search on parabolic keys |
| 2 | phase2_parabolic.py | Complete | Parabolic encoding as exact memory addressing |
| 2b | phase2b_address_limits.py | Complete | Residual addressing scales to 25M+ range |
| 3 | phase3_cumsum.py | Complete | Cumulative sum tracks instruction pointer / stack pointer |
| 4 | phase4_stack_machine.py | Complete | Hand-wired transformer executes PUSH/POP/ADD/DUP/HALT correctly |
| 5 | phase5_training.py | Complete | Tiny model learns execution grammar (56% acc) but not perfect traces |
| 6 | phase6_curriculum.py | Complete | Curriculum learning: 56%→85% acc, 0→39/50 perfect traces |
| 7 | phase7_percepta_arch.py | Complete | Percepta architecture (d=36,h=18,L=7): 84.6% acc, DIFF+ADD still 0% |
| 8 | phase8_microop_traces.py | Complete | Micro-op decomposition proves retrieval is solved; arithmetic is bottleneck |
| 9 | phase9_weighted_arithmetic.py | Complete | Weighted loss perfects doubling (100%) but DIFF+ADD stays 0% |
| 11 | phase11_compile_executor.py | Complete | Compiled execution: correct traces, extended ISA (SUB/JZ/JNZ), O(log t) path |
| 12 | phase12_percepta_model.py | Complete | Full PyTorch compiled transformer with real nn.Linear weight matrices |
| 13 | phase13_isa_completeness.py | Complete | ISA completeness: SWAP/OVER/ROT + Fibonacci, multiply, power-of-2, sum, parity |

### Phase 5 Key Finding

Wide model (d=64, heads=4, layers=2, 137K params) reaches 56% token accuracy (vs 0.5% chance) but 0/50 perfect traces. The model learns *structure* but not *precise routing*. This is the attention-vs-FF gap made concrete: good at finding operands, bad at dispatching operations.

### Phase 6 Key Findings

Curriculum learning confirms the hypothesis: staged instruction complexity (PUSH-only → +POP/DUP → full set) improves accuracy from 56% to 85% with 39/50 perfect traces.

**Three deep diagnostics revealed the bottleneck progression:**
1. **Copy bottleneck (solved):** The model couldn't copy values from program memory. Fix: more data (5K samples) — the copy mechanism IS learnable, just data-starved at 1K.
2. **Non-arithmetic execution (solved):** Stage 2 (PUSH/POP/DUP) achieves 50/50 perfect traces. The model fully learns stack operations that don't require arithmetic.
3. **Two-operand retrieval (current frontier):** ADD requires reading stack[SP-1] and stack[SP-2] simultaneously. The model gets 97% on DUP+ADD (one lookup + double) but only 3% on PUSH a, PUSH b, ADD where a≠b. Doubling heads (h=4→8) doesn't help — per-head capacity drops, negating the extra heads.

**Key insight:** Content-addressable memory lookup (parabolic indexing via gradient descent) is THE fundamental bottleneck in learning execution. Once single-value copy converges, everything follows except two-value retrieval + arithmetic.

### Phase 7 Key Finding

Percepta's architecture (d_model=36, n_heads=18, n_layers=7, head_dim=2) performs comparably to Phase 6's wider model (84.6% vs 85%) but does NOT break the DIFF+ADD wall (0% vs 3%). More depth and heads don't help — the bottleneck is elsewhere.

### Phase 8 Key Findings — THE RETRIEVAL/ARITHMETIC SEPARATION

**Phase 8 is the most important diagnostic phase.** It definitively separates retrieval from arithmetic as bottlenecks.

Micro-op trace format expands each step from 4 to 6 tokens: `[OP, ARG, FETCH1, FETCH2, SP, TOP]`. For ADD, FETCH1 and FETCH2 explicitly contain the two operands before TOP must be predicted.

**Results:**
1. **Retrieval is SOLVED.** Token-by-token analysis shows FETCH1 and FETCH2 are correct 100% of the time for DIFF+ADD. The model perfectly retrieves both operands from different stack positions.
2. **Arithmetic is the sole bottleneck.** The ONLY errors are on `TOP = FETCH1 + FETCH2`. Even with both operands already in context, the model predicts doubling (2*a) or copying instead of a+b.
3. **The model CAN learn addition in isolation** — 98% accuracy on bare `a+b` task with 500 epochs. But within execution traces, addition receives too little gradient signal (~15% of tokens involve ADD).
4. **Wider FF doesn't help.** Even d_ff=1024 (4x baseline) produces 0% DIFF+ADD. The bottleneck isn't capacity — it's the proportion of arithmetic gradient updates.
5. **ADD-enriched data marginally helps** — 50% DIFF+ADD training data yields 1/30 (3%), barely breaking the wall.

**Revised architectural insight:** "Attention is lookup; feed-forward is arithmetic." Attention reliably learns content-addressable retrieval via gradient descent. But FF layers struggle to learn even simple arithmetic (integer addition) when it's a minority of the training signal. This suggests real transformer execution requires either (a) arithmetic pre-training, (b) massive over-sampling of arithmetic examples, or (c) external arithmetic modules.

### Phase 9 Key Findings — WEIGHTED LOSS

Upweighting arithmetic tokens in the loss (10x-50x on ADD's TOP position) tests whether gradient signal alone explains the DIFF+ADD wall.

**Results:**
- **DUP+ADD: 83% → 100%.** Weighted loss perfects doubling (f(a)=2a). This was learnable but under-trained.
- **SAME+ADD: 83% → 100%.** Same story — doubling perfected.
- **DIFF+ADD: 0% → 0%.** True addition (f(a,b)=a+b for a≠b) not learned at ANY weight (10x, 20x, 50x).
- Higher weights (50x) actually hurt overall accuracy (87.4% vs 88.3% at 10x) by destabilizing non-ADD tokens.

**Conclusion:** The DIFF+ADD wall is NOT a gradient signal problem. It's a representational problem: the FF layers cannot learn integer addition in token embedding space while simultaneously handling execution logic. The model can learn addition in isolation (98% in Phase 8's bare test) but not within the multi-task execution context.

**The bottleneck progression is now fully characterized:**
1. Copy mechanism — solved by more data (Phase 6)
2. Stack retrieval — solved by micro-op decomposition (Phase 8)
3. Doubling (2a) — solved by weighted loss (Phase 9)
4. True addition (a+b, a≠b) — **unsolved via training**: requires compilation into weights (Percepta's approach)

### Phase 11 Key Findings — RETURN TO THE COMPILE PATH

After rereading Percepta's blog post, we identified that Phases 5-10 diverged from the original approach. Percepta **compiles** interpreter logic into weights; we were **training** via gradient descent. Phase 11 returns to the compile path.

**Results:**
1. **Compiled executor matches reference 100%.** All 10 Phase 4 test programs produce identical traces when executed via compiled attention primitives (parabolic indexing + cumsum).
2. **Extended ISA works.** SUB, JZ/JNZ (conditional branching), and NOP enable loops and control flow. A countdown loop (JNZ jumping backward) executes correctly — the first looping program in this repo.
3. **HullKVCache integration preserves correctness.** All traces match when stack memory uses the Phase 1 hull cache.
4. **Scaling advantage confirmed.** Dict-based O(1) stack access gives 20-170x speedup over parabolic numpy scan on programs with 100-2000 steps.

**Revised architectural insight:** The DIFF+ADD wall from Phases 5-10 was a *training* limitation, not an *architectural* one. When arithmetic is compiled into FF weights (rather than learned via gradient descent), the transformer executes correctly — including true a+b addition. This validates Percepta's core claim: compile, don't train.

**The bottleneck progression is now fully characterized:**
1. Copy mechanism — solved by more data (Phase 6)
2. Stack retrieval — solved by micro-op decomposition (Phase 8)
3. Doubling (2a) — solved by weighted loss (Phase 9)
4. True addition (a+b, a≠b) — solved by **compilation** (Phase 11); unsolvable via training alone
5. Control flow (branching, loops) — solved by compiled JZ/JNZ (Phase 11)

### Phase 12 Key Findings — REAL PYTORCH COMPILED TRANSFORMER

Phase 12 implements the Percepta model as a real PyTorch `nn.Module` with actual `nn.Linear` weight matrices set analytically. Programs execute via real tensor operations (`matmul`, `argmax`), not simulated numpy primitives.

**Architecture:**
- `d_model=36`, `head_dim=2` (2D parabolic key space, matching Percepta)
- 4 active attention heads: program opcode fetch, program arg fetch, stack read at SP, stack read at SP-1
- Hard-max attention (argmax, not softmax)
- FF dispatch: bilinear gating (opcode one-hot × value routing matrix)
- 758 total compiled parameters (722 in attention heads + 36 in dispatch buffers)
- Float64 precision for parabolic addressing correctness

**Results:**
1. **100% trace match** on all Phase 4 test programs (10/10).
2. **Extended ISA works** — SUB, JZ/JNZ, NOP all correct (8/8), including countdown loop.
3. **Full-sequence attention** confirmed — the compiled weights work in standard transformer Q@K^T → argmax → V framework.
4. **Address verification** needed for stack reads — pure parabolic attention can select wrong-address entries; a second head reads the address and the FF gates the value. This is architecturally valid (two heads cooperating via FF).

**Key insight:** The transition from Phase 11 (numpy simulation) to Phase 12 (PyTorch matmul) required solving **address verification** — parabolic scores are positive even for wrong addresses when the query address exceeds the stored address. The fix: an address-checking head reads `key[0]/2` and the FF layer gates the value (output 0 if address mismatch). This is what multiple heads in a real transformer would do.

### Phase 13 Key Findings — ISA COMPLETENESS

Phase 13 proves the compiled transformer is a general-purpose stack computer, not just a special-case executor.

**New opcodes:** SWAP (exchange top two), OVER (copy second to top), ROT (rotate top 3: [a,b,c]→[b,c,a]). ROT required a new attention head (Head 4) reading SP-2, using one of 14 reserved head slots.

**Algorithm suite (all correct on both numpy + PyTorch, trace-level match):**
1. **Fibonacci(n):** Iterative with 3-value cycling via SWAP+OVER+ADD+ROT. fib(10)=55 in 111 steps from 19 instructions.
2. **Multiply(a,b):** Repeated addition. mul(12,10)=120 in 119 steps.
3. **Power of 2 (2^n):** Repeated doubling via DUP+ADD. 2^7=128 in 76 steps.
4. **Sum(1..n):** Accumulation loop. sum(1..15)=120 in 156 steps.
5. **Parity (is_even):** Conditional branching via repeated subtraction-by-2.

**Architecture:** d_model=36, 5 active heads (of 18 slots), 12 opcodes, 964 compiled parameters.

**Key insight:** Adding SWAP/OVER/ROT transforms the ISA from "theoretically Turing-complete but practically limited" to "Forth-equivalent and practically programmable." The attention head for SP-2 was architecturally trivial (same Q bias pattern as SP-1), confirming the parabolic addressing scheme generalizes cleanly to arbitrary stack depths.

## Development Notes

### File Reading Discipline

**Use tree-sitting to navigate, then Read scoped line ranges.** Several files in this repo exceed the Read tool's token cap (executor.py, symbolic_executor.py, ff_symbolic.py, symbolic_programs_catalog.py). Don't try to read them whole — it fails. Use the `tree-sitting` skill to find the exact symbol and line range, then Read only that window.

```bash
TREESIT=/mnt/skills/user/tree-sitting/scripts/treesit.py
PY=/home/claude/.venv/bin/python

$PY $TREESIT .                              # repo overview
$PY $TREESIT . 'find:NumPyExecutor'          # locate a symbol
$PY $TREESIT . 'source:NumPyExecutor.execute' # print source + line range
$PY $TREESIT . 'refs:CompiledModel'          # find references before editing
```

**Pattern — find-then-target:** `find:Symbol` → Read file with the returned `offset`/`limit`.

**Anti-pattern — blind chunking:** Read executor.py (fails) → Read offset=0 limit=5000 → Read offset=5000 limit=5000 → lose track of what's where.

**Anti-pattern — line-by-line reads:** Read lines 1-100 → think → Read 100-200 → think → repeat.

For files under the Read cap (phase files, programs.py, isa.py, test_consolidated.py), a full read is fine and preferred.

### Environment Setup

Use `uv` for dependency installation — it's pre-installed in Claude Code environments and 10-50× faster than pip.

```bash
uv pip install -r requirements.txt --system
```

The `--system` flag installs into the system Python (required in containers where venv creation is unnecessary). If `uv` is not available, fall back to:

```bash
pip install -r requirements.txt --break-system-packages
```

**Do not pin exact versions.** This is a research repo; loose lower bounds (`>=`) in requirements.txt avoid unnecessary breakage across environments.

### Mojo Availability

**Mojo is available in this repo.** The `llm-as-computer` skill (installed at `/mnt/skills/user/llm-as-computer/`) includes a Mojo executor. The SessionStart hook installs Mojo and compiles the executor binary automatically. To use it:

```python
import sys
sys.path.insert(0, '/mnt/skills/user/llm-as-computer/src')
from runner import run, setup
setup()  # idempotent — installs Mojo + compiles if needed
```

If Mojo is not yet installed, run `cd /mnt/skills/user/llm-as-computer/src && bash setup.sh`. The skill falls back to a pure-Python executor if Mojo is unavailable.

### Container Constraints
- Claude.ai containers: ~200s bash timeout, ~15 min session limit, 8GB RAM
- CCotw (Claude Code on the web): 600s bash timeout, longer sessions, 16GB RAM — better for compute
- Store checkpoint memories every ~5 min during training to survive cutoffs
- Install torch/numpy at session start (`uv pip install torch numpy --system`); not pre-installed in CCotw

### Commit and Push on Every File Write

**⚠️ MANDATORY: Commit and push after EVERY meaningful file write. This is non-negotiable.** Sessions can be cut off at any time with zero warning. To ensure work is never lost, you MUST commit and push to the working branch on GitHub every time you write or edit a file. Do NOT batch up multiple file changes into a single commit at the end — commit and push incrementally as you go. If you write a file and do not immediately commit+push, you are violating this rule. This way, if the session dies mid-task, the next session can pick up from the last push rather than starting over.

**Pattern:** Write/edit file → run tests → `git add` + `git commit` + `git push` — every single time. No exceptions.

### Testing
Always run phase scripts and verify output before committing. Each phase file is self-contained with its own test harness.

### Recall Tags
Use `recall("llm-as-computer", n=10)` or `recall("percepta", n=5)` to load prior context. Key tags: `LLM`, `architecture`, `research`, `percepta`, `transformer-executor`, `phase-N`.
