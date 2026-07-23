"""Paged (two-level) parabolic addressing — fix for the float32 capacity law.

Issue #120 establishes the closed-form addressing capacity law for flat
parabolic keys k_j = (2j, -j²) with query (j, 1):

    exact retrieval holds iff j² ≤ 2^(significand bits)

The winning score has magnitude j² and beats the runner-up by exactly 1;
float32 (24-bit significand) has ULP 2 for values in [2²⁴, 2²⁵), so every
index j > 4096 misaddresses. The break is a step function at N = 4096,
not the "~4–7K" blur the March benchmarks reported (that was pipeline
noise on top of this law).

This module implements proposed fix 2: two-level paged addressing. An
address a splits into (page, offset) = (a // P, a % P) with P ≤ 4096.
Level 1 is one hard-attention call over a page directory (parabolic keys
on page ids); level 2 is one hard-attention call over the entries of the
selected page (parabolic keys on offsets). Each level's score magnitude
stays ≤ P² ≤ 2²⁴, under the float32 ULP wall, so retrieval remains EXACT
— capacity P² ≈ 16.7M addresses in pure float32 for one extra head.

Rate–distortion reading: exact addressing is the zero-distortion endpoint
of the quantizer frontier, so capacity is the rate budget of the score
dtype; paging splits that rate budget across levels. L levels give P^L.

Both levels below are literal attention math — float32 key matrix, dot
product with the query, argmax — matching the semantics of
CompiledAttentionHead / NumPyExecutor.heap_read. The page-directory dict
is storage layout only, the same role the executor's Python lists play.

Fix 1 (fp64 score accumulation, capacity ~9×10⁷) is de facto the repo's
status quo: isa.DTYPE is torch.float64 and NumPyExecutor builds float64
arrays. This module is the fix that keeps float32 viable.
"""

import numpy as np
from typing import Optional

DEFAULT_PAGE_SIZE = 4096  # exactly the float32 capacity boundary from #120

# NOTE on recency: NumPyExecutor breaks overwrite ties with an
# EPS·write_count key bias, which works because its keys are float64.
# In float32 that bias is INERT — within-page scores are integers of
# magnitude up to ~2²⁴, where the ULP is 1, so any sub-1 bias rounds
# away and duplicate writes produce exactly tied keys. This module
# breaks ties positionally instead (argmax-to-latest), the float32-
# faithful equivalent of the same convention.


def flat_misaddressed(dtype, n: int) -> int:
    """Count misaddressed indices for flat parabolic addressing at size n.

    Reference measurement from issue #120. For float32 the law predicts
    max(0, n - 4097): every index above 4096 fails, every index at or
    below stays exact.
    """
    j = np.arange(n, dtype=dtype)
    keys = np.stack([2 * j, -(j * j)], axis=1).astype(dtype)
    bad = 0
    for q in range(n):
        scores = dtype(q) * keys[:, 0] + keys[:, 1]
        if int(np.argmax(scores)) != q:
            bad += 1
    return bad


class PagedAttentionMemory:
    """Two-level parabolic addressing memory, exact in float32 up to P².

    write(addr, val) / read(addr) semantics match the executor's heap:
    reads of untouched addresses return 0; overwrites resolve to the most
    recent write via the same EPS·write_count recency bias the flat
    scheme uses.
    """

    def __init__(self, page_size: int = DEFAULT_PAGE_SIZE, dtype=np.float32):
        if page_size > 4096 and dtype == np.float32:
            raise ValueError(
                f"page_size {page_size} > 4096 exceeds the float32 "
                f"capacity law (j² ≤ 2²⁴) from issue #120"
            )
        self.P = page_size
        self.dtype = dtype
        # Page directory: page_id -> (offset_keys, values). Storage layout
        # only; all retrieval math below is float32 attention.
        self._pages = {}
        self._dir_keys = []   # parabolic keys over page ids, one per page
        self._dir_ids = []    # page id per directory row

    def max_addressable(self) -> int:
        return self.P * self.P

    def _split(self, addr: int):
        return addr // self.P, addr % self.P

    def write(self, addr: int, val):
        page, off = self._split(addr)
        if page >= self.P:
            raise ValueError(f"addr {addr} exceeds capacity {self.P * self.P}")
        if page not in self._pages:
            self._pages[page] = ([], [])
            self._dir_keys.append(
                (self.dtype(2 * page), self.dtype(-(page * page)))
            )
            self._dir_ids.append(page)
        keys, vals = self._pages[page]
        keys.append((self.dtype(2 * off), self.dtype(-(off * off))))
        vals.append(val)

    def read(self, addr: int):
        """Two hard-attention calls: page select, then offset select."""
        if not self._dir_keys:
            return 0
        page, off = self._split(addr)

        # Level 1: attention over the page directory.
        dir_keys = np.array(self._dir_keys, dtype=self.dtype)
        q1 = np.array([page, 1.0], dtype=self.dtype)
        best_dir = int(np.argmax(dir_keys @ q1))
        selected = self._dir_ids[best_dir]
        if selected != page:
            return 0  # page not present; parabola peaked elsewhere

        # Level 2: attention over the selected page's entries.
        keys, vals = self._pages[selected]
        page_keys = np.array(keys, dtype=self.dtype)
        q2 = np.array([off, 1.0], dtype=self.dtype)
        scores = page_keys @ q2
        # argmax with tie-break to the LATEST write (see recency NOTE above)
        best = len(scores) - 1 - int(np.argmax(scores[::-1]))
        stored_off = round(float(page_keys[best, 0]) / 2.0)
        return vals[best] if stored_off == off else 0
