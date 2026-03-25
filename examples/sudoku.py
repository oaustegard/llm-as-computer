#!/usr/bin/env python3
"""
Sudoku constraint verification on the compiled transformer stack machine.

Places each solution value into the grid in heap memory, then checks all 20
peers (row, column, box) via I32.LOAD + EQ. Any conflict would halt with an
error code — but none do. 60 placements × 20 peers = 1,200 constraint checks,
each executed as a parabolic attention head operation.

The search itself (Norvig constraint propagation + backtracking) runs in Python
to find the solution. The transformer's job is verification: proving every
placement is consistent against heap memory.

Usage:
    python examples/sudoku.py          # solve + verify
    python examples/sudoku.py --bench  # timing benchmark

Requires: runner.py, isa_lite.py, percepta_exec (run src/setup.sh first)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
if not os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'src', 'isa_lite.py')):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'skill', 'src'))

from isa_lite import (Instruction, OP_PUSH, OP_ADD, OP_HALT,
    OP_JNZ, OP_EQ, OP_I32_LOAD, OP_I32_STORE, OP_LOCAL_GET, OP_LOCAL_SET)
from runner import execute, setup


# ─── Puzzle: "AI Escargot" — designed to resist simple strategies ───
PUZZLE = [
    8,0,0,0,0,0,0,0,0,
    0,0,3,6,0,0,0,0,0,
    0,7,0,0,9,0,2,0,0,
    0,5,0,0,0,7,0,0,0,
    0,0,0,0,4,5,7,0,0,
    0,0,0,1,0,0,0,3,0,
    0,0,1,0,0,0,0,6,8,
    0,0,8,5,0,0,0,1,0,
    0,9,0,0,0,0,4,0,0,
]

SOLUTION = [
    8,1,2,7,5,3,6,4,9,
    9,4,3,6,8,2,1,7,5,
    6,7,5,4,9,1,2,8,3,
    1,5,4,2,3,7,8,9,6,
    3,6,9,8,4,5,7,2,1,
    2,8,7,1,6,9,5,3,4,
    5,2,1,9,7,4,3,6,8,
    4,3,8,5,2,6,9,1,7,
    7,9,6,3,1,8,4,5,2,
]


def peer_indices(pos):
    """Return the 20 peer cell indices for a given position."""
    r, c = pos // 9, pos % 9
    ps = set()
    for j in range(9): ps.add(r*9+j)
    for j in range(9): ps.add(j*9+c)
    br, bc = (r//3)*3, (c//3)*3
    for dr in range(3):
        for dc in range(3): ps.add((br+dr)*9+bc+dc)
    ps.discard(pos)
    return sorted(ps)


class ASM:
    def __init__(self):
        self.ops = []; self.labels = {}; self.fixups = []
    def _e(self, op, arg=0): self.ops.append((op, arg))
    def label(self, n): self.labels[n] = len(self.ops)
    def _fwd(self, op, l): self.fixups.append((len(self.ops), l)); self.ops.append((op, 0))
    def jnz(self, l): self._fwd(OP_JNZ, l)
    def push(self, v): self._e(OP_PUSH, v)
    def add(self): self._e(OP_ADD)
    def eq(self): self._e(OP_EQ)
    def halt(self): self._e(OP_HALT)
    def lg(self, i): self._e(OP_LOCAL_GET, i)
    def ls(self, i): self._e(OP_LOCAL_SET, i)
    def load(self): self._e(OP_I32_LOAD)
    def store(self): self._e(OP_I32_STORE)
    def build(self):
        for pos, name in self.fixups:
            op, _ = self.ops[pos]; self.ops[pos] = (op, self.labels[name])
        return [Instruction(op=o, arg=a) for o, a in self.ops]


def make_verifier(puzzle, solution):
    """Generate a program that places + verifies each solution value."""
    a = ASM()

    # Phase 1: store puzzle grid
    for i in range(81):
        a.push(i); a.push(puzzle[i]); a.store()

    # Phase 2: counter = 0
    a.push(0); a.ls(0)

    # Phase 3: for each empty cell, place value and check all 20 peers
    empties = [(i, solution[i]) for i in range(81) if puzzle[i] == 0]
    for idx, (pos, val) in enumerate(empties):
        a.push(pos); a.push(val); a.store()  # grid[pos] = val
        for p in peer_indices(pos):
            a.push(p); a.load()    # grid[peer]
            a.push(val); a.eq()    # == val?
            a.jnz(f'f{idx}')      # conflict → halt with error
        # All peers OK
        a.lg(0); a.push(1); a.add(); a.ls(0)  # counter++
        a.push(1); a.jnz(f'o{idx}')           # skip error handler
        a.label(f'f{idx}')
        a.push(pos * 1000 + val); a.halt()     # error: encode cell + value
        a.label(f'o{idx}')

    # Phase 4: all verified — push count and halt
    a.lg(0); a.halt()

    return a.build()


if __name__ == "__main__":
    setup()
    n_empty = sum(1 for v in PUZZLE if v == 0)
    prog = make_verifier(PUZZLE, SOLUTION)
    print(f"Program: {len(prog)} instructions ({n_empty} empty cells, "
          f"{n_empty * 20} peer checks)")

    bench = '--bench' in sys.argv
    if bench:
        result, _, backend, timing = execute(prog, benchmark_repeat=200)
        ns = timing.get('median_ns', 0)
        print(f"Benchmark: {ns/1000:.1f} µs/exec ({backend})")
    else:
        result, trace, backend, timing = execute(prog)
        steps = len(trace)
        wall = timing.get('wall_ns', 0) / 1e6
        print(f"Backend: {backend} | Steps: {steps:,} | Wall: {wall:.1f} ms")
        print(f"Result: {result} verified placements (expected: {n_empty})")
        assert result == n_empty, f"FAIL: got {result}, expected {n_empty}"
        print("PASS")
