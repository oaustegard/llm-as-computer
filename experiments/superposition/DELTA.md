# Delta against Tracr §5

Required reading before this experiment runs. Lindner et al., *Tracr: Compiled
Transformers as a Laboratory for Interpretability* (arXiv 2301.05062), §5
"Compressing Compiled Transformers", already uses compiled transformers to study
superposition — it is in the abstract. The spec for this experiment was written
without checking, and this file is the correction: what Tracr §5 does, what is left,
and what has to be cited rather than claimed.

## What Tracr §5 actually does

- **Learned compression.** A single linear projection `W ∈ R^{D×d}` is trained by SGD
  to compress the disentangled residual stream. `W^T` is applied on every read from
  the stream and `W` on every write; all layer weights are frozen and only `W` trains.
- **Loss.** `L_out` (output match against the compiled model) plus `L_layer`, a
  regularizer summing squared per-layer activation differences, so the compressed
  model is pushed to reproduce the original's intermediate states, not just its answer.
- **Case studies.** `frac_prevs` compresses from `D=14` to `d=6` "without hurting
  performance", averaged over 10 seeds; `sort_unique` is the second.
- **What the compression learned.** `W^T W` at `d=8` shows task-critical features
  (`tokens:x`, `is_x`, `frac_prevs`) keeping near-separate dimensions, unnecessary
  features (`tokens:a/b/c`) discarded outright, and index embeddings placed in
  non-orthogonal directions — read as consistent with Elhage et al. because those
  features are sparse and do not co-occur.
- **Same computation?** §5.3 checks per-layer cosine similarity. `frac_prevs` stays
  near 1. `sort_unique` reaches almost perfect accuracy with layer cosine stuck around
  0.8, and the paper attributes it to the compressed model learning a *different
  numerical encoding* for `target_pos` — i.e. solving the task another way.
- **Stated as not done.** Superposition "has not been studied in models deeper than
  two layers", and reversing the induced superposition — "using ideas from compressed
  sensing and dictionary learning" — is named as future work.

## What this experiment does that §5 does not

**1. Compression is stipulated, not learned.** Tracr's `W` is fit by SGD against the
task, so its results measure what gradient descent finds as much as what packing
costs. `sort_unique` is the clean example: the compressed model re-encoded a variable
and stopped matching the original's intermediates. Here the code is drawn at random
and frozen — each feature gets a random unit vector `u_f`, and the readout is either
`u_f` itself or the pseudo-inverse. Nothing adapts, nothing is discarded, and the
ground truth is stipulated rather than inferred. That is a strictly weaker compressor
and the numbers here should be read as a lower bound (see Limitations).

**2. Stateful iterated computation, not a feed-forward program.** Tracr compresses a
RASP program of fixed depth; error passes through a bounded number of layers once.
LAC iterates one weight set over hundreds of steps, writing rows into a memory it
then re-reads, with a hard argmax at every step. `sum_1_to_100` is 803 sequential
exact-argmax steps over a memory that grows past 1500 rows. Tracr's own framing notes
superposition has not been studied past two layers; this is the regime where a single
flipped argmax is not a small numeric error but a different program.

**3. Two curves, not one.** Tracr asks whether the compressed model still solves the
task, and (§5.3) whether it solves it the same way. Neither question is whether the
*ISA is still recoverable by an analyst who was not told it*. This experiment measures
computation and recovery on the same axis, which is what P1 is about. The analyst here
is an instance of the "reverse the superposition" direction Tracr names as future
work — analytic (geometry + least squares + replay) rather than learned, and applied
to a stack machine rather than a RASP program.

**4. P2 and P3 have no analogue in §5.** Tracr's capacity story is about feature
count, importance and sparsity: which variables matter, which get dropped, which share
directions. P2 predicts that is not the binding constraint here — that capacity is set
by numeric dynamic range, because the parabolic key gap is exactly 1 while interference
scales as `stored_value / sqrt(d)`. Tracr has no quantity like `-j^2` spanning four
orders of magnitude within one feature, and nothing like the `1e-6` write-order
tiebreak of P3. Both predictions are specific to sequential exact computation over a
numeric key scheme, and nothing in §5 speaks to either.

## What must be cited, not claimed

- "Compiled transformers as a testbed for studying superposition" is Tracr's, from the
  abstract. Not a framing this experiment gets to introduce.
- "Superposition in a compiled model has not been studied" would be false.
- Reversing compression-induced superposition is named in Tracr's future work; the
  analyst here is one instance of it, not the first suggestion of it.
- Related, for the *base* experiment rather than this one: Thurnherr and Riesen,
  *Neural Decompiling of Tracr Transformers* (arXiv 2410.00061), trained a seq2seq
  decompiler on Tracr weight/RASP pairs (~30% exact, ~73% functionally equivalent).
  Decompiling a compiled transformer is done; `experiments/blind_recovery` differs in
  kind (analytic, exact, permuted basis, stack machine, and it measures an
  identifiability limit) but not in existence.

## Limitations this delta creates

A random frozen code is a weaker compressor than a trained one. Tracr shows SGD both
discards features the task does not need and re-encodes the ones it keeps, so a
learned code should hold this ISA at a smaller `d` than any random code does. Every
threshold measured here is therefore an upper bound on `d`, not a capacity limit of
the ISA. Closing that gap means training a `W` against the LAC executor the way §5
trains against a RASP program, which is a follow-up and not this experiment.

One post-hoc arm is reported alongside the two pre-registered ones and labelled as
post-hoc throughout: a **scaled** code, where each feature's direction is divided by
that feature's typical magnitude so every feature contributes comparably to the
residual norm. It is not learned — it is one normalization constant per feature — but
it is the sharpest available test of P2. If dynamic range is what sets capacity, that
single change should move the threshold by orders of magnitude while leaving the
feature count untouched.
