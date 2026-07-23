"""Tests for the #120 addressing capacity law and paged addressing fix.

The law: flat parabolic addressing in float32 is exact iff j ≤ 4096 —
misaddressed count at size N is exactly max(0, N - 4097), a step
function. Paged addressing keeps every level under the wall and stays
exact in float32 across the full P² address space.
"""

import sys
import random

import numpy as np

from paged_addressing import (
    PagedAttentionMemory, flat_misaddressed, DEFAULT_PAGE_SIZE,
)


# ─── The capacity law itself (regression-pins issue #120) ─────────

def test_flat_float32_exact_at_4096():
    assert flat_misaddressed(np.float32, 4096) == 0


def test_flat_float32_cliff_is_step_function():
    # every index above 4096 misaddresses: count == N - 4097
    for n in (4608, 5120, 6144):
        assert flat_misaddressed(np.float32, n) == n - 4097, n


def test_flat_float64_exact_at_65536():
    assert flat_misaddressed(np.float64, 65536) == 0


# ─── Paged addressing: exact float32 retrieval beyond the wall ────

def test_paged_exact_across_full_range():
    m = PagedAttentionMemory()  # P=4096, float32, capacity 16.7M
    rng = random.Random(120)
    hi = m.max_addressable()
    addrs = ([0, 1, 4095, 4096, 4097, 8191, 8192, hi - 1]
             + [rng.randrange(hi) for _ in range(500)])
    written = {}
    for a in addrs:
        v = rng.randrange(1, 10**6)
        m.write(a, v)
        written[a] = v
    for a, v in written.items():
        assert m.read(a) == v, f"addr {a}"


def test_paged_exact_where_flat_float32_fails():
    # indices 4097..8192 all misaddress under flat float32 (the law);
    # under paging every one reads back exactly.
    m = PagedAttentionMemory()
    for a in range(4097, 8193, 41):
        m.write(a, a * 3 + 1)
    for a in range(4097, 8193, 41):
        assert m.read(a) == a * 3 + 1, f"addr {a}"


def test_untouched_address_reads_zero():
    m = PagedAttentionMemory()
    m.write(5000, 7)
    assert m.read(5001) == 0        # same page, absent offset
    assert m.read(9_000_000) == 0   # absent page


def test_overwrite_resolves_to_latest():
    m = PagedAttentionMemory()
    m.write(123456, 1)
    m.write(123456, 2)
    assert m.read(123456) == 2


def test_page_size_over_wall_rejected():
    try:
        PagedAttentionMemory(page_size=5000)
    except ValueError:
        return
    raise AssertionError("page_size > 4096 in float32 must be rejected")


def test_capacity():
    assert PagedAttentionMemory().max_addressable() == DEFAULT_PAGE_SIZE ** 2


def main():
    tests = [
        test_flat_float32_exact_at_4096,
        test_flat_float32_cliff_is_step_function,
        test_flat_float64_exact_at_65536,
        test_paged_exact_across_full_range,
        test_paged_exact_where_flat_float32_fails,
        test_untouched_address_reads_zero,
        test_overwrite_resolves_to_latest,
        test_page_size_over_wall_rejected,
        test_capacity,
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
