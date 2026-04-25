"""Specialization parity tests.

Validates the partial-evaluation pass in `specialize.py` on two axes:

  1. STATIC parity — for every cursor 0 <= t < N, specialized.fetch(t) is
     bit-exact with (prog[t].op, prog[t].arg). Exhaustive over the program
     table; stronger than any sampled dynamic check.

  2. EXECUTION equivalence — wrap the program so each NumPyExecutor read
     goes through specialized.fetch(ip) instead of direct indexing, then
     verify the full trace (every TraceStep) matches the universal
     interpreter step-for-step, on the full 42-opcode ISA.

Coverage spans countdown (the specialization demo), `ALL_TESTS` (Phase 4
baseline), and every Phase 13+ generator used by the forking/closed-form
paths — arithmetic, comparison, bitwise, call/return, select. If this
passes, the partial-evaluation pass is safe to drop in anywhere the
universal fetch path is used.
"""

import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isa import (
    Instruction,
    OP_EQ, OP_LT_S,
    OP_AND, OP_OR, OP_XOR, OP_SHL,
)
from executor import NumPyExecutor
from programs import (
    ALL_TESTS,
    make_countdown,
    make_fibonacci, make_factorial, make_power_of_2,
    make_sum_1_to_n, make_multiply, make_gcd, make_is_even,
    make_log2_floor, make_compare_binary, make_bitwise_binary,
    make_select_max,
)
from specialize import (
    SpecializedProgram, specialize, verify_fetch_parity, fetch_fn_from_program,
    build_specialized_model,
)


# ─── Program wrapper: routes reads through specialized.fetch ────────

class SpecializedProgram_AsList:
    """Drop-in replacement for `List[Instruction]` whose `__getitem__`
    reconstructs the Instruction via `SpecializedProgram.fetch(ip)`.

    NumPyExecutor accesses the program table exclusively through `prog[ip]`
    and `len(prog)`, so routing reads through this wrapper is enough to run
    the executor against the FFN-specialized representation with zero
    changes to executor.py.
    """

    def __init__(self, sp: SpecializedProgram):
        self._sp = sp

    def __len__(self):
        return self._sp.n

    def __getitem__(self, ip: int) -> Instruction:
        op, arg = self._sp.fetch(ip)
        return Instruction(op, arg)

    def __iter__(self):
        for i in range(self._sp.n):
            yield self[i]


# ─── Static parity ──────────────────────────────────────────────────

def _static_parity(name: str, prog: List[Instruction]) -> None:
    verify_fetch_parity(prog)
    sp = specialize(prog)
    assert sp.n == len(prog)
    # Telescoping sanity: prefix-sum of coefficients recovers original field
    for field in ("op", "arg"):
        running = 0
        for i, c in enumerate(sp.coefficients[field]):
            running += c
            assert running == int(getattr(prog[i], field)), (
                f"{name}: telescoping mismatch for {field} at i={i}"
            )


# ─── Execution equivalence ──────────────────────────────────────────

def _trace_tuples(trace) -> List[Tuple[int, int, int, int]]:
    return [(s.op, s.arg, s.sp, s.top) for s in trace.steps]


def _execution_equiv(name: str, prog: List[Instruction], max_steps: int = 50000) -> int:
    universal = NumPyExecutor().execute(prog, max_steps=max_steps)
    specialized_wrapper = SpecializedProgram_AsList(specialize(prog))
    specialized = NumPyExecutor().execute(specialized_wrapper, max_steps=max_steps)

    u = _trace_tuples(universal)
    s = _trace_tuples(specialized)
    assert len(u) == len(s), (
        f"{name}: step count mismatch: universal={len(u)} specialized={len(s)}"
    )
    for i, (a, b) in enumerate(zip(u, s)):
        assert a == b, f"{name}: trace divergence at step {i}: universal={a} specialized={b}"
    return len(u)


# ─── Token savings sanity ───────────────────────────────────────────

def _token_savings(prog: List[Instruction]) -> dict:
    sp = specialize(prog)
    ts = sp.token_savings()
    assert ts["program_tokens_saved"] == 2 * len(prog)
    assert ts["universal_prompt_tokens"] == 2 * len(prog)
    assert ts["specialized_prompt_tokens"] == 0
    return ts


# ─── Runner ─────────────────────────────────────────────────────────

def _unpack(p):
    return p[0] if isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], list) else p


def _check(name: str, prog_or_test) -> None:
    prog = _unpack(prog_or_test() if callable(prog_or_test) else prog_or_test)
    _static_parity(name, prog)
    ts = _token_savings(prog)
    steps = _execution_equiv(name, prog)
    print(
        f"  {name:34s}  N={len(prog):3d}  steps={steps:5d}  "
        f"saved={ts['program_tokens_saved']:3d} toks"
    )


# ─── Torch parity: specialized CompiledModel matches universal ──────

def _torch_parity(name: str, prog: List[Instruction], max_steps: int = 50000) -> int:
    """Run the universal TorchExecutor against a specialized one and
    assert step-by-step trace equality at TraceStep granularity.

    Specialized executor:
      - has no program prefix (prog_embs=None inside CompiledModel.forward)
      - reads (op, arg) at each step from the 2N step-function FFN
        whose output lives in residual dims DIM_FETCHED_OP /
        DIM_FETCHED_ARG.

    Acceptance for issue #115. Bit-exact trace parity is the strongest
    available signal that the analytically constructed weights replay
    the program correctly.
    """
    # Lazy: torch import only when this section runs.
    from executor import TorchExecutor, CompiledModel, D_MODEL, D_MODEL_SPECIALIZED

    universal_model = CompiledModel()
    universal = TorchExecutor(universal_model).execute(prog, max_steps=max_steps)

    spec_model = build_specialized_model(prog)

    # Acceptance: shape sanity in the constructor.
    assert spec_model.d_model == D_MODEL_SPECIALIZED == D_MODEL + 2, (
        f"{name}: specialized d_model = {spec_model.d_model}, expected {D_MODEL + 2}"
    )
    assert spec_model.d_ffn == 2 * len(prog), (
        f"{name}: specialized d_ffn = {spec_model.d_ffn}, expected {2 * len(prog)}"
    )
    assert spec_model.specialization_ffn is not None
    # The base (non-specialized) model has no FFN.
    assert universal_model.specialization_ffn is None
    assert universal_model.d_model == D_MODEL
    assert universal_model.d_ffn == 0

    # prog_embs=None should be tolerated by the model in specialized mode.
    specialized = TorchExecutor(spec_model).execute(prog, max_steps=max_steps)

    # Also run with prog=None to confirm the executor can recover the
    # display program from sp.originals.
    specialized_no_prog = TorchExecutor(spec_model).execute(None, max_steps=max_steps)

    u = _trace_tuples(universal)
    s = _trace_tuples(specialized)
    s2 = _trace_tuples(specialized_no_prog)
    assert len(u) == len(s) == len(s2), (
        f"{name}: step counts: universal={len(u)} specialized={len(s)} "
        f"specialized(no prog)={len(s2)}"
    )
    for i, (a, b) in enumerate(zip(u, s)):
        assert a == b, f"{name}: trace divergence at step {i}: universal={a} specialized={b}"
    for i, (a, b) in enumerate(zip(u, s2)):
        assert a == b, (
            f"{name}: trace divergence (no-prog mode) at step {i}: "
            f"universal={a} specialized={b}"
        )
    return len(u)


def _torch_check(name: str, prog_or_test) -> None:
    prog = _unpack(prog_or_test() if callable(prog_or_test) else prog_or_test)
    steps = _torch_parity(name, prog)
    print(f"  {name:34s}  N={len(prog):3d}  steps={steps:5d}  torch-parity OK")


def main() -> int:
    print("=" * 72)
    print("Specialization parity: static fetch + execution equivalence (42-op ISA)")
    print("=" * 72)

    print("\n── countdown (canonical demo) ──")
    _check("countdown(5)",  make_countdown(5))
    _check("countdown(20)", make_countdown(20))

    print("\n── ALL_TESTS (Phase 4 baseline) ──")
    for tname, tfn in ALL_TESTS:
        _check(f"all_tests/{tname}", tfn)

    print("\n── Phase 13+ generators ──")
    _check("fibonacci(10)",       make_fibonacci(10))
    _check("factorial(5)",        make_factorial(5))
    _check("power_of_2(6)",       make_power_of_2(6))
    _check("sum_1_to_n(10)",      make_sum_1_to_n(10))
    _check("multiply(7, 6)",      make_multiply(7, 6))
    _check("gcd(48, 18)",         make_gcd(48, 18))
    _check("is_even(14)",         make_is_even(14))
    _check("log2_floor(1024)",    make_log2_floor(1024))
    _check("compare_binary eq",   make_compare_binary(OP_EQ, 3, 3))
    _check("compare_binary lt_s", make_compare_binary(OP_LT_S, -1, 1))
    _check("bitwise_binary xor",  make_bitwise_binary(OP_XOR, 0xF0F0, 0x0FF0))
    _check("bitwise_binary shl",  make_bitwise_binary(OP_SHL, 1, 5))
    _check("select_max(5, 9)",    make_select_max(5, 9))

    print("\n── torch_parity (issue #115: specialized CompiledModel) ──")
    # Issue #115 acceptance set: countdown(5/20), fibonacci(10),
    # factorial(5), sum_1_to_n(10), plus the Phase 4 baseline.
    _torch_check("countdown(5)",       make_countdown(5))
    _torch_check("countdown(20)",      make_countdown(20))
    _torch_check("fibonacci(10)",      make_fibonacci(10))
    _torch_check("factorial(5)",       make_factorial(5))
    _torch_check("sum_1_to_n(10)",     make_sum_1_to_n(10))
    for tname, tfn in ALL_TESTS:
        _torch_check(f"all_tests/{tname}", tfn)

    print("\n" + "=" * 72)
    print("OK — all specialization parity tests passed.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
