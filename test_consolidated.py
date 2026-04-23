"""Equivalence tests: verify consolidated modules match phase14 behavior exactly.

Imports ONLY from the new modules (isa, executor, programs).
Cross-validates against old Phase14Executor and Phase14PyTorchExecutor.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isa import (
    Instruction, compare_traces, _test_algorithm, _test_trap_algorithm,
    OP_PUSH, OP_POP, OP_ADD, OP_DUP, OP_HALT,
    OP_SUB, OP_JZ, OP_JNZ, OP_NOP,
    OP_SWAP, OP_OVER, OP_ROT,
    OP_MUL, OP_DIV_S, OP_DIV_U, OP_REM_S, OP_REM_U,
    OP_EQZ, OP_EQ, OP_NE,
    OP_LT_S, OP_LT_U, OP_GT_S, OP_GT_U,
    OP_LE_S, OP_LE_U, OP_GE_S, OP_GE_U,
    OP_AND, OP_OR, OP_XOR,
    OP_SHL, OP_SHR_S, OP_SHR_U, OP_ROTL, OP_ROTR,
    OP_CLZ, OP_CTZ, OP_POPCNT, OP_ABS, OP_NEG, OP_SELECT,
    OP_TRAP,
)
from executor import NumPyExecutor, TorchExecutor
from programs import (
    ALL_TESTS,
    make_fibonacci, make_power_of_2, make_sum_1_to_n, make_multiply, make_is_even,
    make_native_multiply, make_native_divmod, make_native_remainder,
    make_native_is_even, make_factorial, make_gcd,
    make_compare_eqz, make_compare_binary,
    make_native_max, make_native_abs, make_native_clamp,
    make_bitwise_binary, make_popcount_loop, make_bit_extract,
    make_native_clz, make_native_ctz, make_native_popcnt,
    make_native_abs_unary, make_native_neg,
    make_select, make_select_max, make_log2_floor, make_is_power_of_2,
    fib,
)

# Cross-validation imports from old phase files
from phase14_extended_isa import (
    Phase14Executor, Phase14PyTorchExecutor,
)


def test_numpy_equivalence():
    """Verify NumPyExecutor matches Phase14Executor on all programs."""
    print("=" * 60)
    print("Test 1: NumPy Executor Equivalence")
    print("=" * 60)

    old = Phase14Executor()
    new = NumPyExecutor()

    passed = 0
    total = 0

    # Phase 4 tests
    for name, test_fn in ALL_TESTS:
        prog, expected_top = test_fn()
        old_trace = old.execute(prog)
        new_trace = new.execute(prog)
        match, detail = compare_traces(old_trace, new_trace)
        old_top = old_trace.steps[-1].top if old_trace.steps else None
        new_top = new_trace.steps[-1].top if new_trace.steps else None
        ok = match and old_top == expected_top and new_top == expected_top
        if ok: passed += 1
        total += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:25s}  expected={expected_top:>6}  old={old_top}  new={new_top}  match={'Y' if match else 'N'}")
        if not ok and not match:
            print(f"         Detail: {detail}")

    # Phase 11 extended tests
    ext_tests = [
        ("sub_basic",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 3),
          Instruction(OP_SUB), Instruction(OP_HALT)], 7),
        ("loop_countdown",
         [Instruction(OP_PUSH, 3), Instruction(OP_DUP),
          Instruction(OP_PUSH, 1), Instruction(OP_SUB),
          Instruction(OP_DUP), Instruction(OP_JNZ, 1),
          Instruction(OP_HALT)], 0),
        ("jz_taken",
         [Instruction(OP_PUSH, 0), Instruction(OP_JZ, 3),
          Instruction(OP_HALT),
          Instruction(OP_PUSH, 42), Instruction(OP_HALT)], 42),
    ]
    for name, prog, expected in ext_tests:
        old_trace = old.execute(prog)
        new_trace = new.execute(prog)
        match, detail = compare_traces(old_trace, new_trace)
        old_top = old_trace.steps[-1].top if old_trace.steps else None
        new_top = new_trace.steps[-1].top if new_trace.steps else None
        ok = match and old_top == expected and new_top == expected
        if ok: passed += 1
        total += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:25s}  expected={expected:>6}  old={old_top}  new={new_top}  match={'Y' if match else 'N'}")

    # Phase 13 algorithms
    p13_algos = [
        ("fib(10)", *make_fibonacci(10)),
        ("fib(7)", *make_fibonacci(7)),
        ("sum(1..10)", *make_sum_1_to_n(10)),
        ("power(2^5)", *make_power_of_2(5)),
        ("mul(7,8)", *make_multiply(7, 8)),
        ("is_even(10)", *make_is_even(10)),
        ("is_even(7)", *make_is_even(7)),
    ]
    for name, prog, expected in p13_algos:
        old_trace = old.execute(prog)
        new_trace = new.execute(prog)
        match, detail = compare_traces(old_trace, new_trace)
        old_top = old_trace.steps[-1].top if old_trace.steps else None
        new_top = new_trace.steps[-1].top if new_trace.steps else None
        ok = match and old_top == expected and new_top == expected
        if ok: passed += 1
        total += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:25s}  expected={expected:>6}  old={old_top}  new={new_top}  match={'Y' if match else 'N'}")

    # Phase 14 arithmetic
    arith_tests = [
        ("native_mul(7,8)", *make_native_multiply(7, 8)),
        ("native_mul(0,5)", *make_native_multiply(0, 5)),
        ("native_divmod(3,10)", *make_native_divmod(3, 10)),
        ("native_rem(3,10)", *make_native_remainder(3, 10)),
        ("factorial(5)", *make_factorial(5)),
        ("factorial(10)", *make_factorial(10)),
        ("gcd(12,8)", *make_gcd(12, 8)),
        ("gcd(100,75)", *make_gcd(100, 75)),
        ("native_is_even(42)", *make_native_is_even(42)),
        ("native_is_even(15)", *make_native_is_even(15)),
    ]
    for name, prog, expected in arith_tests:
        old_trace = old.execute(prog)
        new_trace = new.execute(prog)
        match, detail = compare_traces(old_trace, new_trace)
        old_top = old_trace.steps[-1].top if old_trace.steps else None
        new_top = new_trace.steps[-1].top if new_trace.steps else None
        ok = match and old_top == expected and new_top == expected
        if ok: passed += 1
        total += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:25s}  expected={expected:>6}  old={old_top}  new={new_top}  match={'Y' if match else 'N'}")

    # Phase 14 comparisons
    cmp_tests = [
        ("eqz(0)", *make_compare_eqz(0)),
        ("eqz(5)", *make_compare_eqz(5)),
        ("eq(5,5)", *make_compare_binary(OP_EQ, 5, 5)),
        ("lt_s(3,7)", *make_compare_binary(OP_LT_S, 3, 7)),
        ("gt_s(10,2)", *make_compare_binary(OP_GT_S, 10, 2)),
        ("max(3,7)", *make_native_max(3, 7)),
        ("max(10,2)", *make_native_max(10, 2)),
        ("abs(42)", *make_native_abs(42)),
        ("clamp(5,0,10)", *make_native_clamp(5, 0, 10)),
        ("clamp(15,0,10)", *make_native_clamp(15, 0, 10)),
    ]
    for name, prog, expected in cmp_tests:
        old_trace = old.execute(prog)
        new_trace = new.execute(prog)
        match, detail = compare_traces(old_trace, new_trace)
        old_top = old_trace.steps[-1].top if old_trace.steps else None
        new_top = new_trace.steps[-1].top if new_trace.steps else None
        ok = match and old_top == expected and new_top == expected
        if ok: passed += 1
        total += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:25s}  expected={expected:>6}  old={old_top}  new={new_top}  match={'Y' if match else 'N'}")

    # Phase 14 bitwise
    bit_tests = [
        ("and(0xFF,0x0F)", *make_bitwise_binary(OP_AND, 0xFF, 0x0F)),
        ("or(0xF0,0x0F)", *make_bitwise_binary(OP_OR, 0xF0, 0x0F)),
        ("xor(0xFF,0xFF)", *make_bitwise_binary(OP_XOR, 0xFF, 0xFF)),
        ("shl(1,4)", *make_bitwise_binary(OP_SHL, 1, 4)),
        ("shr_u(16,1)", *make_bitwise_binary(OP_SHR_U, 16, 1)),
        ("rotl(1,1)", *make_bitwise_binary(OP_ROTL, 1, 1)),
        ("rotr(2,1)", *make_bitwise_binary(OP_ROTR, 2, 1)),
        ("popcount(255)", *make_popcount_loop(255)),
        ("bit(0xFF,4)", *make_bit_extract(0xFF, 4)),
    ]
    for name, prog, expected in bit_tests:
        old_trace = old.execute(prog)
        new_trace = new.execute(prog)
        match, detail = compare_traces(old_trace, new_trace)
        old_top = old_trace.steps[-1].top if old_trace.steps else None
        new_top = new_trace.steps[-1].top if new_trace.steps else None
        ok = match and old_top == expected and new_top == expected
        if ok: passed += 1
        total += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:25s}  expected={expected:>6}  old={old_top}  new={new_top}  match={'Y' if match else 'N'}")

    # Phase 14 unary + parametric
    unary_tests = [
        ("clz(0)", *make_native_clz(0)),
        ("clz(255)", *make_native_clz(255)),
        ("ctz(8)", *make_native_ctz(8)),
        ("popcnt(255)", *make_native_popcnt(255)),
        ("abs_native(42)", *make_native_abs_unary(42)),
        ("abs_native(-7)", *make_native_abs_unary(-7)),
        ("neg(5)", *make_native_neg(5)),
        ("neg(-3)", *make_native_neg(-3)),
        ("select(10,20,1)", *make_select(10, 20, 1)),
        ("select(10,20,0)", *make_select(10, 20, 0)),
        ("select_max(10,25)", *make_select_max(10, 25)),
        ("log2(8)", *make_log2_floor(8)),
        ("ispow2(8)", *make_is_power_of_2(8)),
        ("ispow2(7)", *make_is_power_of_2(7)),
    ]
    for name, prog, expected in unary_tests:
        old_trace = old.execute(prog)
        new_trace = new.execute(prog)
        match, detail = compare_traces(old_trace, new_trace)
        old_top = old_trace.steps[-1].top if old_trace.steps else None
        new_top = new_trace.steps[-1].top if new_trace.steps else None
        ok = match and old_top == expected and new_top == expected
        if ok: passed += 1
        total += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:25s}  expected={expected:>6}  old={old_top}  new={new_top}  match={'Y' if match else 'N'}")

    # Division by zero traps
    trap_progs = [
        ("div_by_zero",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 0),
          Instruction(OP_DIV_S), Instruction(OP_HALT)]),
        ("rem_by_zero",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 0),
          Instruction(OP_REM_S), Instruction(OP_HALT)]),
    ]
    for name, prog in trap_progs:
        old_trace = old.execute(prog)
        new_trace = new.execute(prog)
        match, detail = compare_traces(old_trace, new_trace)
        old_trapped = old_trace.steps and old_trace.steps[-1].op == OP_TRAP
        new_trapped = new_trace.steps and new_trace.steps[-1].op == OP_TRAP
        ok = match and old_trapped and new_trapped
        if ok: passed += 1
        total += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:25s}  old_trap={old_trapped}  new_trap={new_trapped}  match={'Y' if match else 'N'}")

    print(f"\n  NumPy equivalence: {passed}/{total} passed")
    return passed == total


def test_torch_equivalence():
    """Verify TorchExecutor matches Phase14PyTorchExecutor on all programs."""
    print("\n" + "=" * 60)
    print("Test 2: Torch Executor Equivalence")
    print("=" * 60)

    old = Phase14PyTorchExecutor()
    new = TorchExecutor()

    passed = 0
    total = 0

    # Phase 4 tests
    for name, test_fn in ALL_TESTS:
        prog, expected_top = test_fn()
        old_trace = old.execute(prog)
        new_trace = new.execute(prog)
        match, detail = compare_traces(old_trace, new_trace)
        old_top = old_trace.steps[-1].top if old_trace.steps else None
        new_top = new_trace.steps[-1].top if new_trace.steps else None
        ok = match and old_top == expected_top and new_top == expected_top
        if ok: passed += 1
        total += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:25s}  expected={expected_top:>6}  old={old_top}  new={new_top}  match={'Y' if match else 'N'}")
        if not ok and not match:
            print(f"         Detail: {detail}")

    # Phase 13 algorithms
    p13_algos = [
        ("fib(10)", *make_fibonacci(10)),
        ("sum(1..10)", *make_sum_1_to_n(10)),
        ("power(2^5)", *make_power_of_2(5)),
        ("mul(7,8)", *make_multiply(7, 8)),
    ]
    for name, prog, expected in p13_algos:
        old_trace = old.execute(prog)
        new_trace = new.execute(prog)
        match, detail = compare_traces(old_trace, new_trace)
        old_top = old_trace.steps[-1].top if old_trace.steps else None
        new_top = new_trace.steps[-1].top if new_trace.steps else None
        ok = match and old_top == expected and new_top == expected
        if ok: passed += 1
        total += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:25s}  expected={expected:>6}  old={old_top}  new={new_top}  match={'Y' if match else 'N'}")

    # Phase 14 arithmetic + comparison + bitwise + unary
    p14_tests = [
        ("native_mul(12,10)", *make_native_multiply(12, 10)),
        ("factorial(7)", *make_factorial(7)),
        ("gcd(48,36)", *make_gcd(48, 36)),
        ("eqz(0)", *make_compare_eqz(0)),
        ("gt_s(10,2)", *make_compare_binary(OP_GT_S, 10, 2)),
        ("max(10,25)", *make_native_max(10, 25)),
        ("and(0xAA,0x55)", *make_bitwise_binary(OP_AND, 0xAA, 0x55)),
        ("popcount(15)", *make_popcount_loop(15)),
        ("clz(1)", *make_native_clz(1)),
        ("popcnt_native(7)", *make_native_popcnt(7)),
        ("abs_native(-7)", *make_native_abs_unary(-7)),
        ("neg(5)", *make_native_neg(5)),
        ("select(10,20,1)", *make_select(10, 20, 1)),
        ("select_max(25,10)", *make_select_max(25, 10)),
        ("log2(1024)", *make_log2_floor(1024)),
        ("ispow2(1024)", *make_is_power_of_2(1024)),
    ]
    for name, prog, expected in p14_tests:
        old_trace = old.execute(prog)
        new_trace = new.execute(prog)
        match, detail = compare_traces(old_trace, new_trace)
        old_top = old_trace.steps[-1].top if old_trace.steps else None
        new_top = new_trace.steps[-1].top if new_trace.steps else None
        ok = match and old_top == expected and new_top == expected
        if ok: passed += 1
        total += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:25s}  expected={expected:>6}  old={old_top}  new={new_top}  match={'Y' if match else 'N'}")

    # Trap tests
    trap_progs = [
        ("div_by_zero",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 0),
          Instruction(OP_DIV_S), Instruction(OP_HALT)]),
        ("rem_by_zero",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 0),
          Instruction(OP_REM_U), Instruction(OP_HALT)]),
    ]
    for name, prog in trap_progs:
        old_trace = old.execute(prog)
        new_trace = new.execute(prog)
        match, detail = compare_traces(old_trace, new_trace)
        old_trapped = old_trace.steps and old_trace.steps[-1].op == OP_TRAP
        new_trapped = new_trace.steps and new_trace.steps[-1].op == OP_TRAP
        ok = match and old_trapped and new_trapped
        if ok: passed += 1
        total += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:25s}  old_trap={old_trapped}  new_trap={new_trapped}  match={'Y' if match else 'N'}")

    print(f"\n  Torch equivalence: {passed}/{total} passed")
    return passed == total


def test_new_np_vs_new_pt():
    """Verify new NumPyExecutor matches new TorchExecutor (internal consistency)."""
    print("\n" + "=" * 60)
    print("Test 3: New NumPy vs New Torch (Internal Consistency)")
    print("=" * 60)

    np_exec = NumPyExecutor()
    pt_exec = TorchExecutor()

    passed = 0
    total = 0

    all_progs = []

    # Phase 4
    for name, test_fn in ALL_TESTS:
        prog, expected = test_fn()
        all_progs.append((name, prog, expected))

    # Phase 13 algorithms
    all_progs.extend([
        ("fib(10)", *make_fibonacci(10)),
        ("sum(1..15)", *make_sum_1_to_n(15)),
        ("power(2^7)", *make_power_of_2(7)),
        ("mul(12,10)", *make_multiply(12, 10)),
    ])

    # Phase 14 full coverage
    all_progs.extend([
        ("native_mul(100,200)", *make_native_multiply(100, 200)),
        ("factorial(10)", *make_factorial(10)),
        ("gcd(17,13)", *make_gcd(17, 13)),
        ("native_is_even(100)", *make_native_is_even(100)),
        ("eq(5,5)", *make_compare_binary(OP_EQ, 5, 5)),
        ("ne(3,7)", *make_compare_binary(OP_NE, 3, 7)),
        ("le_s(3,7)", *make_compare_binary(OP_LE_S, 3, 7)),
        ("ge_s(10,2)", *make_compare_binary(OP_GE_S, 10, 2)),
        ("max(5,5)", *make_native_max(5, 5)),
        ("clamp(0,5,10)", *make_native_clamp(0, 5, 10)),
        ("or(0xAA,0x55)", *make_bitwise_binary(OP_OR, 0xAA, 0x55)),
        ("xor(12,10)", *make_bitwise_binary(OP_XOR, 12, 10)),
        ("shl(0xFF,4)", *make_bitwise_binary(OP_SHL, 0xFF, 4)),
        ("shr_s(16,1)", *make_bitwise_binary(OP_SHR_S, 16, 1)),
        ("popcount(0xFFFF)", *make_popcount_loop(0xFFFF)),
        ("bit(42,3)", *make_bit_extract(42, 3)),
        ("clz(65536)", *make_native_clz(65536)),
        ("ctz(1024)", *make_native_ctz(1024)),
        ("popcnt_native(1023)", *make_native_popcnt(1023)),
        ("abs_native(0)", *make_native_abs_unary(0)),
        ("neg(1000)", *make_native_neg(1000)),
        ("select(5,9,-1)", *make_select(5, 9, -1)),
        ("select_max(7,7)", *make_select_max(7, 7)),
        ("log2(255)", *make_log2_floor(255)),
        ("ispow2(0)", *make_is_power_of_2(0)),
    ])

    for name, prog, expected in all_progs:
        np_trace = np_exec.execute(prog)
        pt_trace = pt_exec.execute(prog)
        match, detail = compare_traces(np_trace, pt_trace)
        np_top = np_trace.steps[-1].top if np_trace.steps else None
        pt_top = pt_trace.steps[-1].top if pt_trace.steps else None
        ok = match and np_top == expected and pt_top == expected
        if ok: passed += 1
        total += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:25s}  expected={expected:>10}  np={np_top}  pt={pt_top}  match={'Y' if match else 'N'}")
        if not ok:
            if not match:
                print(f"         Detail: {detail}")

    print(f"\n  Internal consistency: {passed}/{total} passed")
    return passed == total


def main():
    print("=" * 60)
    print("Consolidated Module Equivalence Tests")
    print("=" * 60)
    print()

    t0 = time.time()

    results = []
    results.append(("NumPy equivalence", test_numpy_equivalence()))
    results.append(("Torch equivalence", test_torch_equivalence()))
    results.append(("Internal consistency", test_new_np_vs_new_pt()))

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok: all_pass = False
        print(f"  {status}  {name}")

    print(f"\n  Time: {elapsed:.2f}s")

    if all_pass:
        print("\n  All tests pass. Consolidated modules are equivalent to phase files.")
    else:
        print("\n  FAILURES detected. See details above.")

    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
