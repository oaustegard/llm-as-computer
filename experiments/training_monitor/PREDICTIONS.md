# Pre-registered predictions — compiler as a training monitor

Written into the handoff spec on 2026-08-11 (Muninn memory `0234cf29`), before any code
in this directory existed. Copied here verbatim in substance on 2026-09-05, the day the
experiment ran. Graded in `RESULTS.md`.

## Question

Initialize a transformer AT the compiled LAC weights — a point in weight space whose
program is known exactly — run ordinary gradient descent, and run the blind analyst every
N steps. Output is a decay curve: how long the ISA stays recoverable, whether degradation
is graceful or cliff-shaped, and which structure dies first.

## Predictions

**P1. Overshadow, not erase.** The compiled structure stays recoverable well past the
point where the machine stops computing correctly. This matches what the fine-tuning
literature claims (post-training MI survey, arXiv 2407.02646: fine-tuning enhances
existing mechanisms rather than replacing them; forgetting is overshadowing, not
erasure) and would be the first test of it where the prior mechanism was *known* rather
than hypothesized.

**P2. Dispatch dies before addressing.** Addressing has margin — scores scale as x² with
a winner/runner-up gap of 1, so small perturbations do not flip the argmax at small
addresses. The dispatch matrices are a small linear map with NO margin: any perturbation
corrupts the written value directly and proportionally. Expect corrupted arithmetic while
control flow and memory access still work. Same shape as P2/P3 of the superposition run:
exact-argmax structure is robust, exact-value structure is not.

**P3. Failure is cliff-shaped in correctness and graceful in recoverability.** Per-step
exactness compounds over hundreds of steps, so program correctness should fall off
sharply; the analyst averages over rows and should degrade smoothly. If both are smooth,
P1 is probably wrong too.

## Delta vs prior art (stated before running)

- **Tracr** (arXiv 2301.05062): compiled models as interpretability ground truth. Does
  not train from compiled weights.
- **InterpBench / SIIT** (arXiv 2407.14494): DOES train from Tracr models, with a loss
  that preserves the circuit while making weights realistic. This experiment is the exact
  complement: no preservation term, measure decay. They have both endpoints; nobody has
  the path between them.
- **Tracr-Injection** (arXiv 2505.10719, found 2026-09-05 after the spec was written):
  injects a compiled circuit into GPT-2-large with an alignment loss. Also a
  preservation-by-loss endpoint, not the path.
- **ALTA** (arXiv 2410.18077): compiles to Universal Transformers; interpretable-by-design
  training. Adjacent, not this.

## Gotchas fixed in advance

- No circuit-preservation loss term. That is InterpBench and it is the opposite experiment.
- The four oracle programs are the correctness measure. Do not invent new ones for scoring.
- The machine trained is the numpy compiler of `experiments/blind_recovery` (dispatch
  entirely in tensors), not upstream `executor.CompiledModel` (dispatch hard-coded in
  Python, no gradient). Kept consistent across every checkpoint.
- The analyst's tolerance threshold is swept and reported, not tuned.
