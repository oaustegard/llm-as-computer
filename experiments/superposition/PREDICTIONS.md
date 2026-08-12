# Pre-registered predictions — compiled superposition

Written 2026-08-12, **before** the sweep was run. Graded in `RESULTS.md`.

## Setup being tested

`experiments/blind_recovery/` compiles the LAC core-12 ISA to explicit tensors in an
axis-aligned residual stream: 51 dims, 24 used, one dim per semantic feature, zero
interference. The blind analyst recovered 12/12 opcodes from permuted tensors. That
result does not transfer to trained nets, for exactly one reason — trained nets
superpose features, and the compiled artifact does not.

This experiment replaces the axis-aligned basis with a random code. Each of the 24
semantic features (`is_prog`, `is_stack`, `is_state`, `prog_k0/1`, `stack_k0/1`,
`opcode`, `value`, `ip`, `sp`, `one`, and 12 opcode indicators) gets a random unit
vector `u_f` in `R^d`; a row embeds as `sum_f value_f * u_f`. Weight matrices stop
being selectors and become readouts. Two readout arms:

- **dot** — `w_f = u_f` (Elhage toy-models convention). Readout of `f` returns
  `value_f + sum_{g != f} value_g (u_f . u_g)`.
- **pinv** — `W = pinv(U^T)`, least-squares optimal, exact for `d >= 24`, lossy below.

Sweep `d`, `>= 20` seeds per `d`, both arms. Two measurements per configuration:
**computes** (fraction of the four oracle programs returning the exact expected value)
and **recovers** (superposition-aware blind analyst scores exact-ISA recovery 12/12).

Scope: this measures whether the **state encoding** survives compression. The dispatch
tensors (`ffn_A/B/C`) are unaffected by a residual-stream basis change and are handed to
the analyst intact, as in the base experiment.

## Predictions

**P1. `d_compute > d_recover`.** The machine needs per-step exactness; the analyst needs
only statistical structure across many rows and can least-squares away interference.
Expect the analyst to keep recovering the ISA at compression ratios where the machine has
already stopped computing. This inverts the naive expectation and is the point of the
experiment.

**P2. Failure is driven by numeric dynamic range, not feature count.** The parabolic
winner/runner-up gap is exactly 1 (score at `j=x` is `x^2`, at `j=x±1` it is `x^2-1`),
while interference scales as `stored_value / sqrt(d)`. So `sum_1_to_100` (values ~5050)
should break at far larger `d` than `countdown_5` (values <= 5). If true, the headline is
that superposition capacity in a computer is set by dynamic range, not by how many
features are packed.

**P3. The `EPS=1e-6` recency tiebreak dies first.** Stack keys carry
`key_1 = -addr^2 + 1e-6 * write_order`. That is the smallest signal in the system and
should be destroyed at essentially any `d`. Expect append-only stack semantics to break
before addressing does. If everything appears broken even at large `d`, this is why —
the fallback is to switch the stack to overwrite-in-place keyed on address and re-run.

## Grading rules (fixed now)

- **P1 confirmed** if, per program and arm, the smallest `d` at which recovery still
  succeeds is below the smallest `d` at which the machine still computes — i.e. the
  recovery curve extends to more aggressive compression than the compute curve.
- **P2 confirmed** if `d_compute(sum_1_to_100) > d_compute(countdown_5)` by a wide
  margin under the dot arm, and if the measured breakpoints track
  `max|value| / sqrt(d) ~ 1` rather than tracking feature count.
- **P3 confirmed** if the first observed divergence from the reference machine is
  attributable to a stack-key tie (two rows sharing an address, the stale one winning)
  before any addressing or opcode-decode failure appears.
