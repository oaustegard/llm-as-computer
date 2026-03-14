# R&D Plan: Prototyping 2D Convex Hull Attention for In-Model Execution

**Context:** The Percepta blog post "Can LLMs Be Computers?" claims that restricting transformer attention heads to 2D enables O(log t) decoding via convex hull queries, making million-step execution traces feasible inside a transformer. No code or weights are published. This plan tests the core claims from first principles.

**Writeup:** [WRITEUP.md](WRITEUP.md)

**Environment:** Claude skills container — CPU-only, Python 3.12, PyTorch available, scipy available. No GPU. This is fine; the whole point of their approach is CPU-friendly execution.

---

## Phase 1: Convex Hull KV Cache — Does the Geometry Work? ✅

**Goal:** Validate that 2D convex hull lookups actually achieve O(log t) per query vs O(t) brute force, and measure the constant-factor overhead.

**Result:** YES. Ternary search over parabolic keys scales as O(log n). 35x speedup at 50K entries. Crossover at ~1-2K entries.

**Key question answered:** Is the convex hull approach actually faster in practice on CPU, and at what trace length does it break even?

---

## Phase 2: Parabolic Key Encoding — Does Index Lookup Work? ✅

**Goal:** Verify that the encoding k_j = (2j, -j²) with query q = (i, 1) correctly retrieves the value stored at index i via hard-max attention.

**Result:** YES. 100% accuracy in float64. Float32 breaks at ~4K indices (revised from initial ~7K). Recency bias trick handles overwrites cleanly.

---

## Phase 2b: Extended Addressing ✅

**Goal:** The float32 limit (~4K addresses) constrains WASM-scale execution. Find encoding tricks to extend addressable range.

**Result:** Residual (bit-split) addressing solves this. Split addr = (block, offset), each resolved by a separate head. B=5000 → 25M addressable range from 2 heads, zero errors.

---

## Phase 3: Cumulative Sum Attention — Tracking Running State ✅

**Goal:** Verify the claim that attention can compute cumulative sums (used for instruction pointer, stack depth, etc.).

**Result:** Surprisingly robust. Zero integer errors at 100K steps in float32. Sequential lookback (attend to t-1) is simpler and likely what Percepta uses.

---

## Phase 4: Minimal Stack Machine via Attention ✅

**Goal:** Build a hand-wired transformer (weights set analytically, not trained) that executes a trivial instruction set using the primitives from Phases 1-3.

**Result:** YES — the primitives compose. 10/10 test programs execute identically in the attention executor and the reference interpreter. Instruction set: PUSH, POP, ADD, DUP, HALT.

**Key finding:** The FF layer is the hard part. Attention heads have clean, separable roles; opcode-dependent routing in the FF network is where model capacity goes.

---

## Phases 5–9: The Training Detour ✅

Phases 5–9 explored whether gradient descent could learn execution from data — a deliberate departure from Percepta's compile approach. While this path didn't achieve perfect execution, it produced essential findings about what transformers can and cannot learn.

### Phase 5: Trained Micro-Executor ✅

**Goal:** Train a small transformer to learn execution of the Phase 4 instruction set from trace examples.

**Result:** 56% token accuracy (112x above chance), 0/50 perfect traces. Width > depth confirms FF routing is the bottleneck.

### Phase 6: Curriculum Learning ✅

**Goal:** Test whether staged instruction complexity improves learnability.

**Result:** 56%→85% accuracy, 0→39/50 perfect traces. Copy bottleneck solved with more data. Non-arithmetic execution (PUSH/POP/DUP) achieves 50/50 perfect. Two-operand retrieval (ADD a+b, a≠b) remains at 3%.

### Phase 7: Percepta Architecture Test ✅

**Goal:** Test Percepta's published architecture (d=36, h=18, L=7) on our training task.

**Result:** 84.6% accuracy — comparable to Phase 6's 85%. DIFF+ADD stays at 0%. More depth and heads don't help.

### Phase 8: Micro-Op Trace Diagnostics ✅

**Goal:** Definitively separate retrieval from arithmetic as bottlenecks.

**Result:** THE key diagnostic phase. Retrieval is 100% solved (FETCH1 and FETCH2 are always correct). Arithmetic is the sole bottleneck — the model cannot compute a+b even with both operands in context.

### Phase 9: Weighted Arithmetic Loss ✅

**Goal:** Test whether gradient signal alone explains the DIFF+ADD wall.

**Result:** Weighted loss perfects doubling (2a→100%) but true addition (a+b, a≠b) stays at 0% at any weight. The wall is representational, not gradient signal.

### Conclusion from the Training Detour

The bottleneck progression, fully characterized:
1. Copy mechanism — solved by more data (Phase 6)
2. Stack retrieval — solved by micro-op decomposition (Phase 8)
3. Doubling (2a) — solved by weighted loss (Phase 9)
4. True addition (a+b, a≠b) — **unsolvable via training alone**

This sent us back to Percepta's original insight: compile, don't train.

---

## Phase 11: Compiled Executor (Numpy) ✅

**Goal:** Return to the compile path. Implement arithmetic and routing logic as compiled attention primitives rather than learned weights.

**Result:** 100% correct traces on all test programs. Extended ISA with SUB, JZ/JNZ (conditional branching), and NOP. First looping program (countdown via JNZ). Dict-based stack access gives 20-170x speedup over parabolic scan.

**Key insight:** The DIFF+ADD wall was a training limitation, not an architectural one. Compiled arithmetic works perfectly.

---

## Phase 12: Real PyTorch Compiled Transformer ✅

**Goal:** Implement the compiled executor as a real PyTorch `nn.Module` with actual `nn.Linear` weight matrices set analytically.

**Result:** 100% correct. d_model=36, head_dim=2, 4 active heads, 758 compiled parameters. Programs execute via real tensor ops (matmul, argmax). Address verification solved via cooperating heads + FF gating.

---

## Phase 13: ISA Completeness ✅

**Goal:** Prove the compiled transformer is a general-purpose stack computer.

**Result:** Added SWAP, OVER, ROT (5 active heads now). 12-opcode ISA. Fibonacci, multiply, power-of-2, sum, and parity all execute correctly on both numpy and PyTorch backends. 964 compiled parameters. Forth-equivalent.

---

## Success Criteria — Final Status

| Phase | Success looks like | Status |
|-------|-------------------|--------|
| 1 | Clear log vs linear scaling plot with crossover point identified | ✅ Done |
| 2 | 100% retrieval accuracy up to some numerical limit, limit characterized | ✅ Done |
| 3 | Cumsum within ±1 of true value over 100K+ steps | ✅ Done |
| 4 | Hand-wired transformer correctly executes 10+ test programs | ✅ Done |
| 5–9 | Characterize learnability limits of execution via gradient descent | ✅ Done (essential negative result) |
| 11 | Compiled executor produces correct traces | ✅ Done |
| 12 | Real PyTorch matmul execution produces correct traces | ✅ Done |
| 13 | General-purpose algorithms (Fibonacci, multiply) execute correctly | ✅ Done |

## Overall Conclusion

Percepta's core claim is validated: vanilla transformer primitives (2D parabolic attention + compiled FF routing) can implement a working stack machine computer. The key insight is **compile, don't train** — gradient descent can learn execution structure but cannot learn integer arithmetic within execution context. When arithmetic is compiled directly into weight matrices, the transformer executes correctly, including branching, loops, and complex algorithms.
