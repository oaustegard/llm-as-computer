"""Phase 2c: The addressing capacity law across dtypes (successor to #120).

#120 established: flat parabolic addressing k_j = (2j, -j²), q = (j, 1)
is exact iff j² ≤ 2^(significand bits) — the winner's margin is exactly 1
and must be representable at score magnitude j². float32 (24-bit): wall
at j = 4096. Paged (two-level) addressing keeps each level under the
wall: capacity P² in pure float32 (PR #121).

This study extends the law down the precision ladder and off the
exact-retrieval corner. Findings, each asserted below:

1. BF16 WALL AT 16. bf16 has an 8-bit significand → exact iff j ≤ 16.
   Measured: first failure at j = 17; misaddress count is N-17 modulo
   LUCKY TIES — above the wall, scores collide exactly and argmax
   tie-break occasionally lands on the right index by accident (j = 51
   at N = 128). Genuine resolution above 16: zero.

2. HIERARCHY COMPOSES. Paged bf16, P = 16: two levels exact over 256,
   three levels exact over 4096. Depth L → capacity 16^L.

3. INT4 DICHOTOMY. Post-hoc quantization of parabolic keys to int4 is
   catastrophic under every scaling scheme (symmetric per-tensor fails
   from N = 4; affine per-group-16 still misaddresses ~85%): the margin
   is 1 and no scale can make 15 steps span thousands of units at unit
   resolution. But RE-ENCODING onto the grid — address as int4 digits
   (j÷16, j mod 16), two parabolic digit terms, high term dominant —
   is exact to 16^L with zero quantization loss. The encoding must live
   natively on the representation grid; scales cannot rescue a
   grid-incompatible circuit.

4. SMEARED CLIFF — AND SOFT SELF-CORRECTION. Replacing hard argmax
   with softmax readout (scores in bf16, softmax in fp32) does not just
   smear the cliff; it PARTIALLY REPAIRS it. Above the wall, ~30% of
   indices still read out nearly exactly (33/111 at N=128), because
   score collisions are symmetric around the target and the value
   function is linear — the expectation averages the tie back onto j
   even where argmax fails. Precision loss under soft attention with
   LINEAR readout manifests as blur (variance), not bias. CATEGORICAL
   readout — copying an exact token id, what induction heads need —
   cannot average and inherits the hard cliff. Silhouette for trained
   models: low-precision long-context degradation should hit exact-copy
   tasks (needle, induction) much harder than smooth-value regression,
   and errors should be near-neighbor confusions, non-monotone in
   position.

General law: addressing capacity per attention level is the rate budget
of the score representation; hierarchical decomposition converts depth
into exponential capacity; encodings must be grid-native. Wild-model
prediction: post-training int4 quantization should damage long-range
retrieval (needle / induction) far more than perplexity, and QAT should
recover it by finding grid-native encodings.

Requires: numpy, ml_dtypes (no torch).
"""

import sys
import numpy as np

try:
    from ml_dtypes import bfloat16 as bf16
except ImportError:  # pragma: no cover
    print("SKIP: pip install ml_dtypes"); sys.exit(0)


# ─── shared machinery ─────────────────────────────────────────────

def flat_bad_indices(dtype, n):
    """Indices that misaddress under flat parabolic keys at size n."""
    j64 = np.arange(n, dtype=np.float64)
    keys = np.stack([(2 * j64).astype(dtype), (-(j64 ** 2)).astype(dtype)],
                    axis=1)
    bad = []
    for q in range(n):
        s = ((keys[:, 0] * dtype(q)).astype(dtype) + keys[:, 1]).astype(dtype)
        if int(np.argmax(s.astype(np.float32))) != q:
            bad.append(q)
    return bad


# ─── 1. bf16 wall at 16, with lucky ties ──────────────────────────

def test_bf16_wall_at_17():
    assert flat_bad_indices(bf16, 17) == []
    bad24 = flat_bad_indices(bf16, 24)
    assert bad24 == list(range(17, 24)), bad24


def test_bf16_survivors_are_lucky_ties():
    n = 128
    bad = set(flat_bad_indices(bf16, n))
    survivors = [q for q in range(17, n) if q not in bad]
    j64 = np.arange(n, dtype=np.float64)
    keys = np.stack([(2 * j64).astype(bf16), (-(j64 ** 2)).astype(bf16)],
                    axis=1)
    for q in survivors:
        s = ((keys[:, 0] * bf16(q)).astype(bf16) + keys[:, 1]).astype(np.float32)
        top = np.sort(s)[-2:]
        assert top[0] == top[1], f"j={q} resolved genuinely?!"  # must be a tie


# ─── 2. hierarchy composes: bf16 paged, L levels → 16^L ───────────

def _digits(a, base, L):
    out = []
    for _ in range(L):
        out.append(a % base); a //= base
    return out[::-1]  # most significant first


class MultiLevelPaged:
    """L-level parabolic addressing; every level under the dtype wall."""

    def __init__(self, base, levels, dtype):
        self.B, self.L, self.dtype = base, levels, dtype
        self.entries = []  # (digit tuple, value); layout only

    def write(self, addr, val):
        self.entries.append((tuple(_digits(addr, self.B, self.L)), val))

    def read(self, addr):
        want = _digits(addr, self.B, self.L)
        cand = list(range(len(self.entries)))
        for lvl in range(self.L):
            # one hard-attention call over the survivors of prior levels
            ds = np.array([self.entries[i][0][lvl] for i in cand],
                          dtype=np.float64)
            k0 = (2 * ds).astype(self.dtype)
            k1 = (-(ds ** 2)).astype(self.dtype)
            s = ((k0 * self.dtype(want[lvl])).astype(self.dtype)
                 + k1).astype(np.float32)
            best_d = self.entries[cand[int(np.argmax(s))]][0][lvl]
            if best_d != want[lvl]:
                return 0
            cand = [i for i in cand if self.entries[i][0][lvl] == want[lvl]]
        return self.entries[cand[-1]][1]  # latest write wins


def test_bf16_two_levels_exact_256():
    m = MultiLevelPaged(16, 2, bf16)
    for a in range(256):
        m.write(a, a * 7 + 3)
    assert all(m.read(a) == a * 7 + 3 for a in range(256))


def test_bf16_three_levels_exact_4096():
    m = MultiLevelPaged(16, 3, bf16)
    import random
    rng = random.Random(120)
    addrs = [0, 15, 16, 255, 256, 4095] + [rng.randrange(4096)
                                           for _ in range(400)]
    for a in addrs:
        m.write(a, a * 3 + 1)
    assert all(m.read(a) == a * 3 + 1 for a in addrs)


# ─── 3. int4 dichotomy ────────────────────────────────────────────

def _parab(n):
    j = np.arange(n, dtype=np.float64)
    return np.stack([2 * j, -j * j], axis=1)

def _misaddr(Kq, n):
    return sum(1 for q in range(n)
               if int(np.argmax(Kq[:, 0] * q + Kq[:, 1])) != q)

def test_int4_posthoc_quant_catastrophic():
    for n in (16, 64, 256):
        K = _parab(n)
        s = np.abs(K).max() / 7.0                       # symmetric per-tensor
        sym = np.clip(np.round(K / s), -8, 7) * s
        assert _misaddr(sym, n) >= n - 2, n             # near-total failure
        aff = np.empty_like(K)                          # affine per-group-16
        for a in range(0, n, 16):
            blk = K[a:a + 16]
            lo, hi = blk.min(axis=0), blk.max(axis=0)
            step = (hi - lo) / 15.0
            step[step == 0] = 1.0
            aff[a:a + 16] = np.round((blk - lo) / step) * step + lo
        assert _misaddr(aff, n) > 0.6 * n, n            # still mostly broken

def test_int4_digit_native_exact():
    for n in (64, 256):
        hi, lo = np.arange(n) // 16, np.arange(n) % 16  # int4-native digits
        bad = 0
        for q in range(n):
            s = 1024.0 * (2 * (q // 16) * hi - hi * hi) \
                + (2 * (q % 16) * lo - lo * lo)
            if int(np.argmax(s)) != q:
                bad += 1
        assert bad == 0, (n, bad)


# ─── 4. smeared cliff under softmax readout ──────────────────────

def softmax_readout_error(n, temp=1.0):
    """|readout − j| per index: bf16 scores, fp32 softmax over values=j."""
    j64 = np.arange(n, dtype=np.float64)
    keys = np.stack([(2 * j64).astype(bf16), (-(j64 ** 2)).astype(bf16)],
                    axis=1)
    vals = j64.astype(np.float32)
    errs = np.empty(n)
    for q in range(n):
        s = ((keys[:, 0] * bf16(q)).astype(bf16)
             + keys[:, 1]).astype(np.float32) / temp
        w = np.exp(s - s.max()); w /= w.sum()
        errs[q] = abs(float(w @ vals) - q)
    return errs

def test_smeared_cliff():
    errs = softmax_readout_error(128)
    assert errs[:17].max() < 0.5          # below the wall: exact readout
    above = errs[17:]
    assert (above > 0.5).mean() > 0.5     # above: majority broken...
    assert (above < 0.5).sum() >= 20      # ...but ~30% self-correct via
                                          # symmetric-tie averaging (linear
                                          # readout repairs what argmax can't)
    d = np.diff(above)
    assert (d < 0).any() and (d > 0).any()  # non-monotone degradation


def main():
    tests = [
        test_bf16_wall_at_17,
        test_bf16_survivors_are_lucky_ties,
        test_bf16_two_levels_exact_256,
        test_bf16_three_levels_exact_4096,
        test_int4_posthoc_quant_catastrophic,
        test_int4_digit_native_exact,
        test_smeared_cliff,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
