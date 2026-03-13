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
4. True addition (a+b, a≠b) — **unsolved**: requires either joint arithmetic training, digit-level decomposition, or dedicated arithmetic modules

## Development Notes

### Container Constraints
- Claude.ai containers: ~200s bash timeout, ~15 min session limit, 8GB RAM
- CCotw (Claude Code on the web): 600s bash timeout, longer sessions, 16GB RAM — better for compute
- Store checkpoint memories every ~5 min during training to survive cutoffs
- Install torch/numpy at session start (`pip install torch numpy`); not pre-installed in CCotw

### Testing
Always run phase scripts and verify output before committing. Each phase file is self-contained with its own test harness.

### Recall Tags
Use `recall("llm-as-computer", n=10)` or `recall("percepta", n=5)` to load prior context. Key tags: `LLM`, `architecture`, `research`, `percepta`, `transformer-executor`, `phase-N`.
