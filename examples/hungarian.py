#!/usr/bin/env python3
"""
Hungarian Algorithm (Kuhn-Munkres) on the compiled transformer stack machine.

Solves min-cost perfect matching on an n×n cost matrix. The entire algorithm —
potentials, augmenting paths, traceback — runs as parabolic attention head
operations on the Mojo executor.

Usage:
    python examples/hungarian.py          # run + verify
    python examples/hungarian.py --bench  # run timing benchmark

Requires: runner.py, isa_lite.py, percepta_exec (run src/setup.sh first)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
if not os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'src', 'isa_lite.py')):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'skill', 'src'))

from isa_lite import (Instruction, OP_PUSH, OP_POP, OP_ADD, OP_DUP, OP_HALT,
    OP_SUB, OP_JZ, OP_JNZ, OP_MUL, OP_LT_S, OP_NE, OP_EQ,
    OP_I32_LOAD, OP_I32_STORE, OP_LOCAL_GET, OP_LOCAL_SET, OP_SWAP)
from runner import execute, setup

INF = 999999

# ─── Memory layout ───
# Cost matrix at 0..n*n-1, potentials u[] at 100+, v[] at 120+,
# assignment p[] at 140+, way[] at 160+, minv[] at 180+, used[] at 200+
COST=0; U=100; V=120; P=140; WAY=160; MINV=180; USED=200

# Locals
L_I=0; L_J0=1; L_J=2; L_I0=3; L_DELTA=4; L_J1=5; L_CUR=6; L_TMP=7; L_TMP2=8


class ASM:
    """Stack machine assembler with label resolution."""
    def __init__(self):
        self.ops = []; self.labels = {}; self.fixups = []
    def _e(self, op, arg=0): self.ops.append((op, arg))
    def label(self, n): self.labels[n] = len(self.ops)
    def _fwd(self, op, l): self.fixups.append((len(self.ops), l)); self.ops.append((op, 0))
    def jz(self, l):  self._fwd(OP_JZ, l)
    def jnz(self, l): self._fwd(OP_JNZ, l)
    def jmp(self, l):  self._e(OP_PUSH, 1); self._fwd(OP_JNZ, l)
    def push(self, v): self._e(OP_PUSH, v)
    def pop(self):     self._e(OP_POP)
    def dup(self):     self._e(OP_DUP)
    def swap(self):    self._e(OP_SWAP)
    def add(self):     self._e(OP_ADD)
    def sub(self):     self._e(OP_SUB)
    def mul(self):     self._e(OP_MUL)
    def lt(self):      self._e(OP_LT_S)
    def eq(self):      self._e(OP_EQ)
    def ne(self):      self._e(OP_NE)
    def halt(self):    self._e(OP_HALT)
    def lg(self, i):   self._e(OP_LOCAL_GET, i)
    def ls(self, i):   self._e(OP_LOCAL_SET, i)
    def load(self):    self._e(OP_I32_LOAD)
    def store(self):   self._e(OP_I32_STORE)

    def arr_load(self, base, idx_local):
        self.push(base); self.lg(idx_local); self.add(); self.load()
    def arr_store(self, base, idx_local):
        self.ls(L_TMP2); self.push(base); self.lg(idx_local); self.add()
        self.lg(L_TMP2); self.store()
    def arr_store_local(self, base, idx_local, val_local):
        self.push(base); self.lg(idx_local); self.add(); self.lg(val_local); self.store()
    def build(self):
        for pos, name in self.fixups:
            op, _ = self.ops[pos]; self.ops[pos] = (op, self.labels[name])
        return [Instruction(op=o, arg=a) for o, a in self.ops]


def make_hungarian(matrix):
    """Generate a stack machine program for min-cost perfect matching."""
    a = ASM()
    n = len(matrix)

    # Store cost matrix
    for i in range(n):
        for j in range(n):
            a.push(COST + i*n + j); a.push(matrix[i][j]); a.store()

    # Initialize u, v, p = 0
    for k in range(n+1):
        a.push(U+k); a.push(0); a.store()
        a.push(V+k); a.push(0); a.store()
        a.push(P+k); a.push(0); a.store()

    # Main loop: for i = 1 to n
    a.push(1); a.ls(L_I)
    a.label('row_loop')
    a.lg(L_I); a.push(n+1); a.lt(); a.jz('done')

    a.push(P+0); a.lg(L_I); a.store()  # p[0] = i
    a.push(0); a.ls(L_J0)              # j0 = 0

    # minv[1..n] = INF, used[1..n] = 0
    a.push(1); a.ls(L_J)
    a.label('init_loop')
    a.lg(L_J); a.push(n+1); a.lt(); a.jz('init_end')
    a.push(INF); a.arr_store(MINV, L_J)
    a.push(0);   a.arr_store(USED, L_J)
    a.lg(L_J); a.push(1); a.add(); a.ls(L_J)
    a.jmp('init_loop')
    a.label('init_end')
    a.push(USED+0); a.push(0); a.store()

    # Augmenting path loop
    a.label('aug_loop')
    a.push(1); a.arr_store(USED, L_J0)
    a.arr_load(P, L_J0); a.ls(L_I0)
    a.push(INF); a.ls(L_DELTA)
    a.push(0); a.ls(L_J1)

    # Inner loop: for j = 1 to n
    a.push(1); a.ls(L_J)
    a.label('col_loop')
    a.lg(L_J); a.push(n+1); a.lt(); a.jz('col_end')
    a.arr_load(USED, L_J); a.jnz('col_next')

    # cur = C[(i0-1)*n + (j-1)] - u[i0] - v[j]
    a.lg(L_I0); a.push(1); a.sub(); a.push(n); a.mul()
    a.lg(L_J); a.push(1); a.sub(); a.add()
    a.push(COST); a.add(); a.load()
    a.arr_load(U, L_I0); a.sub()
    a.arr_load(V, L_J); a.sub()
    a.ls(L_CUR)

    a.lg(L_CUR); a.arr_load(MINV, L_J); a.lt(); a.jz('skip_minv')
    a.lg(L_CUR); a.arr_store(MINV, L_J)
    a.lg(L_J0);  a.arr_store(WAY, L_J)
    a.label('skip_minv')

    a.arr_load(MINV, L_J); a.lg(L_DELTA); a.lt(); a.jz('col_next')
    a.arr_load(MINV, L_J); a.ls(L_DELTA)
    a.lg(L_J); a.ls(L_J1)

    a.label('col_next')
    a.lg(L_J); a.push(1); a.add(); a.ls(L_J)
    a.jmp('col_loop')
    a.label('col_end')

    # Update potentials
    a.push(0); a.ls(L_J)
    a.label('upd_loop')
    a.lg(L_J); a.push(n+1); a.lt(); a.jz('upd_end')
    a.arr_load(USED, L_J); a.jz('upd_else')
    a.arr_load(P, L_J); a.ls(L_TMP)
    a.arr_load(U, L_TMP); a.lg(L_DELTA); a.add(); a.arr_store(U, L_TMP)
    a.arr_load(V, L_J); a.lg(L_DELTA); a.sub(); a.arr_store(V, L_J)
    a.jmp('upd_next')
    a.label('upd_else')
    a.arr_load(MINV, L_J); a.lg(L_DELTA); a.sub(); a.arr_store(MINV, L_J)
    a.label('upd_next')
    a.lg(L_J); a.push(1); a.add(); a.ls(L_J)
    a.jmp('upd_loop')
    a.label('upd_end')

    a.lg(L_J1); a.ls(L_J0)
    a.arr_load(P, L_J0); a.push(0); a.ne(); a.jnz('aug_loop')

    # Traceback
    a.label('trace_loop')
    a.arr_load(WAY, L_J0); a.ls(L_J1)
    a.arr_load(P, L_J1); a.arr_store(P, L_J0)
    a.lg(L_J1); a.ls(L_J0)
    a.lg(L_J0); a.push(0); a.ne(); a.jnz('trace_loop')

    a.lg(L_I); a.push(1); a.add(); a.ls(L_I)
    a.jmp('row_loop')

    # Compute result: sum C[p[j]-1][j-1]
    a.label('done')
    a.push(0); a.ls(L_TMP)
    a.push(1); a.ls(L_J)
    a.label('sum_loop')
    a.lg(L_J); a.push(n+1); a.lt(); a.jz('sum_end')
    a.arr_load(P, L_J); a.push(1); a.sub(); a.push(n); a.mul()
    a.lg(L_J); a.push(1); a.sub(); a.add()
    a.push(COST); a.add(); a.load()
    a.lg(L_TMP); a.add(); a.ls(L_TMP)
    a.lg(L_J); a.push(1); a.add(); a.ls(L_J)
    a.jmp('sum_loop')
    a.label('sum_end')
    a.lg(L_TMP); a.halt()

    return a.build()


# ─── Test matrix ───
MATRIX = [
    [61,58,35,86,32,39,41,27,21,42],
    [59,77,97,99,78,21,89,72,35,63],
    [88,85,37,57,59,97,37,29,69,94],
    [32,82,53,20,77,96,21,70,50,61],
    [15,44,81,10,64,36,56,78,20,69],
    [76,35,87,69,16,55,26,37,30,66],
    [86,32,74,94,32,14,24,12,31,70],
    [97,63,20,64,90,21,28,49,89,10],
    [58,52,27,76,61,35,17,91,37,66],
    [42,79,61,26,55,98,70,17,26,86],
]
EXPECTED = 206  # Verified via scipy.optimize.linear_sum_assignment


if __name__ == "__main__":
    setup()
    prog = make_hungarian(MATRIX)
    print(f"Program: {len(prog)} instructions")

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
        print(f"Result: {result} (expected: {EXPECTED})")
        assert result == EXPECTED, f"FAIL: got {result}, expected {EXPECTED}"
        print("PASS")
