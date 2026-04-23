"""ISA definition for the compiled transformer stack machine.

Consolidated from phase4, phase12, phase13, phase14. Contains:
  - Types: Instruction, Trace, TraceStep
  - Constants: D_MODEL, DTYPE, EPS, N_OPCODES, all DIM_* layout
  - Opcodes: OP_PUSH through OP_SELECT, OP_TRAP
  - Maps: OP_NAMES, OPCODE_DIM_MAP, OPCODE_IDX, NONLINEAR_OPS
  - Math helpers: _trunc_div, _trunc_rem, bitwise ops
  - CompiledAttentionHead: nn.Module for hard-max attention
  - Embedding functions: embed_program_token, embed_stack_entry, embed_state
  - Test utilities: compare_traces, test_algorithm, test_trap_algorithm
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional
from dataclasses import dataclass, field


# ─── Types (from phase4) ──────────────────────────────────────────

@dataclass
class Instruction:
    op: int
    arg: int = 0

    def __repr__(self):
        name = OP_NAMES.get(self.op, f"?{self.op}")
        if self.op in (OP_PUSH, OP_JZ, OP_JNZ,
                        OP_LOCAL_GET, OP_LOCAL_SET, OP_LOCAL_TEE,
                        OP_CALL):
            return f"{name} {self.arg}"
        return name


def program(*instrs) -> List[Instruction]:
    """Convenience: program(('PUSH', 3), ('PUSH', 5), ('ADD',), ('HALT',))"""
    result = []
    _name_to_op = {
        "PUSH": 1, "POP": 2, "ADD": 3, "DUP": 4, "HALT": 5,
        "SUB": 6, "JZ": 7, "JNZ": 8, "NOP": 9,
        "SWAP": 10, "OVER": 11, "ROT": 12,
        "MUL": 13, "DIV_S": 14, "DIV_U": 15, "REM_S": 16, "REM_U": 17,
        "EQZ": 18, "EQ": 19, "NE": 20,
        "LT_S": 21, "LT_U": 22, "GT_S": 23, "GT_U": 24,
        "LE_S": 25, "LE_U": 26, "GE_S": 27, "GE_U": 28,
        "AND": 29, "OR": 30, "XOR": 31,
        "SHL": 32, "SHR_S": 33, "SHR_U": 34, "ROTL": 35, "ROTR": 36,
        "CLZ": 37, "CTZ": 38, "POPCNT": 39, "ABS": 40, "NEG": 41,
        "SELECT": 42,
        "LOCAL.GET": 43, "LOCAL.SET": 44, "LOCAL.TEE": 45,
        "I32.LOAD": 46, "I32.STORE": 47,
        "I32.LOAD8_U": 48, "I32.LOAD8_S": 49,
        "I32.LOAD16_U": 50, "I32.LOAD16_S": 51,
        "I32.STORE8": 52, "I32.STORE16": 53,
        "CALL": 54, "RETURN": 55,
    }
    for instr in instrs:
        if isinstance(instr, Instruction):
            result.append(instr)
            continue
        name = instr[0].upper()
        arg = instr[1] if len(instr) > 1 else 0
        op = _name_to_op[name]
        result.append(Instruction(op, arg))
    return result


@dataclass
class TraceStep:
    """One instruction's execution record."""
    op: int
    arg: int
    sp: int      # stack pointer AFTER execution
    top: int     # top-of-stack value AFTER execution

    def tokens(self) -> List[int]:
        return [self.op, self.arg, self.sp, self.top]


@dataclass
class Trace:
    """Full execution trace: program prefix + step records."""
    program: List[Instruction]
    steps: List[TraceStep] = field(default_factory=list)

    def format_trace(self) -> str:
        """Human-readable trace."""
        lines = []
        lines.append(f"Program: {' ; '.join(str(i) for i in self.program)}")
        lines.append(f"{'Step':>4}  {'Instruction':<10} {'SP':>3}  {'TOP':>5}")
        lines.append("-" * 35)
        for i, s in enumerate(self.steps):
            name = OP_NAMES.get(s.op, "?")
            instr_str = f"{name} {s.arg}" if s.op in (
                OP_PUSH, OP_JZ, OP_JNZ, OP_LOCAL_GET, OP_LOCAL_SET, OP_LOCAL_TEE,
                OP_CALL
            ) else name
            lines.append(f"{i:4d}  {instr_str:<10} {s.sp:3d}  {s.top:5d}")
        return "\n".join(lines)


# ─── Constants ────────────────────────────────────────────────────

D_MODEL = 51
DTYPE = torch.float64
EPS = 1e-6

# Token roles (from phase4, used in training phases)
TOKENS_PER_STEP = 4


# ─── Embedding Dimension Layout (all 36 dims) ────────────────────

DIM_IS_PROG     = 0
DIM_IS_STACK    = 1
DIM_IS_STATE    = 2
DIM_PROG_KEY_0  = 3
DIM_PROG_KEY_1  = 4
DIM_STACK_KEY_0 = 5
DIM_STACK_KEY_1 = 6
DIM_OPCODE      = 7
DIM_VALUE       = 8
DIM_IP          = 9
DIM_SP          = 10
DIM_ONE         = 11
DIM_IS_PUSH     = 12
DIM_IS_POP      = 13
DIM_IS_ADD      = 14
DIM_IS_DUP      = 15
DIM_IS_HALT     = 16
DIM_IS_SUB      = 17
DIM_IS_JZ       = 18
DIM_IS_JNZ      = 19
DIM_IS_NOP      = 20
DIM_IS_SWAP     = 21
DIM_IS_OVER     = 22
DIM_IS_ROT      = 23
DIM_IS_MUL      = 24
DIM_IS_DIV_S    = 25
DIM_IS_DIV_U    = 26
DIM_IS_REM_S    = 27
DIM_IS_REM_U    = 28
DIM_IS_EQZ      = 29
DIM_IS_EQ       = 30
DIM_IS_NE       = 31
DIM_IS_LT       = 32   # shared by LT_S and LT_U
DIM_IS_GT       = 33   # shared by GT_S and GT_U
DIM_IS_LE       = 34   # shared by LE_S and LE_U
DIM_IS_GE       = 35   # shared by GE_S and GE_U

# Phase 15: local variables address space
DIM_IS_LOCAL      = 36
DIM_LOCAL_KEY_0   = 37
DIM_LOCAL_KEY_1   = 38
DIM_IS_LOCAL_GET  = 39
DIM_IS_LOCAL_SET  = 40
DIM_IS_LOCAL_TEE  = 41

# Phase 16: linear memory (heap) address space
DIM_IS_HEAP       = 42
DIM_HEAP_KEY_0    = 43
DIM_HEAP_KEY_1    = 44

# Phase 17: call stack address space
DIM_IS_CALL_STACK    = 45
DIM_CALL_KEY_0       = 46
DIM_CALL_KEY_1       = 47
DIM_CALL_RET_ADDR    = 48
DIM_CALL_SAVED_SP    = 49
DIM_CALL_LOCALS_BASE = 50


# ─── Opcodes ─────────────────────────────────────────────────────

# Phase 4 base
OP_PUSH = 1
OP_POP  = 2
OP_ADD  = 3
OP_DUP  = 4
OP_HALT = 5

# Phase 11 extended
OP_SUB = 6
OP_JZ  = 7
OP_JNZ = 8
OP_NOP = 9

# Phase 13 stack manipulation
OP_SWAP = 10
OP_OVER = 11
OP_ROT  = 12

# Phase 14 Chunk 1: arithmetic
OP_MUL   = 13
OP_DIV_S = 14
OP_DIV_U = 15
OP_REM_S = 16
OP_REM_U = 17

# Phase 14 Chunk 2: comparisons
OP_EQZ   = 18
OP_EQ    = 19
OP_NE    = 20
OP_LT_S  = 21
OP_LT_U  = 22
OP_GT_S  = 23
OP_GT_U  = 24
OP_LE_S  = 25
OP_LE_U  = 26
OP_GE_S  = 27
OP_GE_U  = 28

# Phase 14 Chunk 3: bitwise
OP_AND   = 29
OP_OR    = 30
OP_XOR   = 31
OP_SHL   = 32
OP_SHR_S = 33
OP_SHR_U = 34
OP_ROTL  = 35
OP_ROTR  = 36

# Phase 14 Chunk 4: unary + parametric
OP_CLZ    = 37
OP_CTZ    = 38
OP_POPCNT = 39
OP_ABS    = 40
OP_NEG    = 41
OP_SELECT = 42

# Phase 15: local variables
OP_LOCAL_GET = 43
OP_LOCAL_SET = 44
OP_LOCAL_TEE = 45

# Phase 16: linear memory
OP_I32_LOAD    = 46
OP_I32_STORE   = 47
OP_I32_LOAD8_U = 48
OP_I32_LOAD8_S = 49
OP_I32_LOAD16_U = 50
OP_I32_LOAD16_S = 51
OP_I32_STORE8  = 52
OP_I32_STORE16 = 53

# Phase 17: function calls
OP_CALL   = 54
OP_RETURN = 55

# Trap
OP_TRAP  = 99

N_OPCODES = 55  # 53 base + 2 call ops


# ─── Maps ─────────────────────────────────────────────────────────

OP_NAMES = {
    OP_PUSH: "PUSH", OP_POP: "POP", OP_ADD: "ADD", OP_DUP: "DUP", OP_HALT: "HALT",
    OP_SUB: "SUB", OP_JZ: "JZ", OP_JNZ: "JNZ", OP_NOP: "NOP",
    OP_SWAP: "SWAP", OP_OVER: "OVER", OP_ROT: "ROT",
    OP_MUL: "MUL", OP_DIV_S: "DIV_S", OP_DIV_U: "DIV_U",
    OP_REM_S: "REM_S", OP_REM_U: "REM_U",
    OP_EQZ: "EQZ", OP_EQ: "EQ", OP_NE: "NE",
    OP_LT_S: "LT_S", OP_LT_U: "LT_U", OP_GT_S: "GT_S", OP_GT_U: "GT_U",
    OP_LE_S: "LE_S", OP_LE_U: "LE_U", OP_GE_S: "GE_S", OP_GE_U: "GE_U",
    OP_AND: "AND", OP_OR: "OR", OP_XOR: "XOR",
    OP_SHL: "SHL", OP_SHR_S: "SHR_S", OP_SHR_U: "SHR_U",
    OP_ROTL: "ROTL", OP_ROTR: "ROTR",
    OP_CLZ: "CLZ", OP_CTZ: "CTZ", OP_POPCNT: "POPCNT",
    OP_ABS: "ABS", OP_NEG: "NEG", OP_SELECT: "SELECT",
    OP_LOCAL_GET: "LOCAL.GET", OP_LOCAL_SET: "LOCAL.SET", OP_LOCAL_TEE: "LOCAL.TEE",
    OP_I32_LOAD: "I32.LOAD", OP_I32_STORE: "I32.STORE",
    OP_I32_LOAD8_U: "I32.LOAD8_U", OP_I32_LOAD8_S: "I32.LOAD8_S",
    OP_I32_LOAD16_U: "I32.LOAD16_U", OP_I32_LOAD16_S: "I32.LOAD16_S",
    OP_I32_STORE8: "I32.STORE8", OP_I32_STORE16: "I32.STORE16",
    OP_CALL: "CALL", OP_RETURN: "RETURN",
    OP_TRAP: "TRAP",
}

OPCODE_DIM_MAP = {
    OP_PUSH: DIM_IS_PUSH, OP_POP: DIM_IS_POP, OP_ADD: DIM_IS_ADD,
    OP_DUP: DIM_IS_DUP, OP_HALT: DIM_IS_HALT, OP_SUB: DIM_IS_SUB,
    OP_JZ: DIM_IS_JZ, OP_JNZ: DIM_IS_JNZ, OP_NOP: DIM_IS_NOP,
    OP_SWAP: DIM_IS_SWAP, OP_OVER: DIM_IS_OVER, OP_ROT: DIM_IS_ROT,
    OP_MUL: DIM_IS_MUL, OP_DIV_S: DIM_IS_DIV_S, OP_DIV_U: DIM_IS_DIV_U,
    OP_REM_S: DIM_IS_REM_S, OP_REM_U: DIM_IS_REM_U,
    OP_EQZ: DIM_IS_EQZ, OP_EQ: DIM_IS_EQ, OP_NE: DIM_IS_NE,
    OP_LT_S: DIM_IS_LT, OP_LT_U: DIM_IS_LT,
    OP_GT_S: DIM_IS_GT, OP_GT_U: DIM_IS_GT,
    OP_LE_S: DIM_IS_LE, OP_LE_U: DIM_IS_LE,
    OP_GE_S: DIM_IS_GE, OP_GE_U: DIM_IS_GE,
    OP_LOCAL_GET: DIM_IS_LOCAL_GET,
    OP_LOCAL_SET: DIM_IS_LOCAL_SET,
    OP_LOCAL_TEE: DIM_IS_LOCAL_TEE,
}

OPCODE_IDX = {
    OP_PUSH: 0, OP_POP: 1, OP_ADD: 2, OP_DUP: 3, OP_HALT: 4,
    OP_SUB: 5, OP_JZ: 6, OP_JNZ: 7, OP_NOP: 8,
    OP_SWAP: 9, OP_OVER: 10, OP_ROT: 11,
    OP_MUL: 12, OP_DIV_S: 13, OP_DIV_U: 14, OP_REM_S: 15, OP_REM_U: 16,
    OP_EQZ: 17, OP_EQ: 18, OP_NE: 19,
    OP_LT_S: 20, OP_LT_U: 21, OP_GT_S: 22, OP_GT_U: 23,
    OP_LE_S: 24, OP_LE_U: 25, OP_GE_S: 26, OP_GE_U: 27,
    OP_AND: 28, OP_OR: 29, OP_XOR: 30,
    OP_SHL: 31, OP_SHR_S: 32, OP_SHR_U: 33, OP_ROTL: 34, OP_ROTR: 35,
    OP_CLZ: 36, OP_CTZ: 37, OP_POPCNT: 38, OP_ABS: 39, OP_NEG: 40,
    OP_SELECT: 41,
    OP_LOCAL_GET: 42, OP_LOCAL_SET: 43, OP_LOCAL_TEE: 44,
    OP_I32_LOAD: 45, OP_I32_STORE: 46,
    OP_I32_LOAD8_U: 47, OP_I32_LOAD8_S: 48,
    OP_I32_LOAD16_U: 49, OP_I32_LOAD16_S: 50,
    OP_I32_STORE8: 51, OP_I32_STORE16: 52,
    OP_CALL: 53, OP_RETURN: 54,
}

NONLINEAR_OPS = {
    OP_MUL, OP_DIV_S, OP_DIV_U, OP_REM_S, OP_REM_U,
    OP_EQZ, OP_EQ, OP_NE,
    OP_LT_S, OP_LT_U, OP_GT_S, OP_GT_U,
    OP_LE_S, OP_LE_U, OP_GE_S, OP_GE_U,
    OP_AND, OP_OR, OP_XOR,
    OP_SHL, OP_SHR_S, OP_SHR_U,
    OP_ROTL, OP_ROTR,
    OP_CLZ, OP_CTZ, OP_POPCNT, OP_ABS, OP_NEG, OP_SELECT,
    OP_I32_LOAD8_U, OP_I32_LOAD8_S,
    OP_I32_LOAD16_U, OP_I32_LOAD16_S,
}


# ─── Math Helpers ─────────────────────────────────────────────────

MASK32 = 0xFFFFFFFF


def _trunc_div(b, a):
    """Signed integer division truncating toward zero (WASM semantics)."""
    return int(b / a)


def _trunc_rem(b, a):
    """Signed remainder matching truncated division: b - trunc(b/a)*a."""
    return b - _trunc_div(b, a) * a


def _to_i32(val):
    """Cast to 32-bit signed integer from potentially float stack value."""
    return int(val) & MASK32


def _shr_u(b, a):
    """Logical (unsigned) right shift of b by a positions."""
    return (_to_i32(b) >> (int(a) & 31))


def _shr_s(b, a):
    """Arithmetic (signed) right shift of b by a positions."""
    val = _to_i32(b)
    if val >= 0x80000000:
        val -= 0x100000000
    shift = int(a) & 31
    result = val >> shift
    return result & MASK32 if result < 0 else result


def _rotl32(b, a):
    """Left-rotate b by a positions within 32-bit word."""
    val = _to_i32(b)
    shift = int(a) & 31
    return ((val << shift) | (val >> (32 - shift))) & MASK32 if shift else val


def _rotr32(b, a):
    """Right-rotate b by a positions within 32-bit word."""
    val = _to_i32(b)
    shift = int(a) & 31
    return ((val >> shift) | (val << (32 - shift))) & MASK32 if shift else val


def _clz32(val):
    """Count leading zeros in 32-bit representation."""
    v = _to_i32(val)
    if v == 0:
        return 32
    n = 0
    if v <= 0x0000FFFF: n += 16; v <<= 16
    if v <= 0x00FFFFFF: n += 8;  v <<= 8
    if v <= 0x0FFFFFFF: n += 4;  v <<= 4
    if v <= 0x3FFFFFFF: n += 2;  v <<= 2
    if v <= 0x7FFFFFFF: n += 1
    return n


def _ctz32(val):
    """Count trailing zeros in 32-bit representation."""
    v = _to_i32(val)
    if v == 0:
        return 32
    n = 0
    if (v & 0x0000FFFF) == 0: n += 16; v >>= 16
    if (v & 0x000000FF) == 0: n += 8;  v >>= 8
    if (v & 0x0000000F) == 0: n += 4;  v >>= 4
    if (v & 0x00000003) == 0: n += 2;  v >>= 2
    if (v & 0x00000001) == 0: n += 1
    return n


def _popcnt32(val):
    """Population count (number of set bits) in 32-bit representation."""
    return bin(_to_i32(val)).count('1')


def _sign_extend_8(val):
    """Sign-extend an 8-bit value to a signed integer."""
    v = int(val) & 0xFF
    return v - 0x100 if v >= 0x80 else v


def _sign_extend_16(val):
    """Sign-extend a 16-bit value to a signed integer."""
    v = int(val) & 0xFFFF
    return v - 0x10000 if v >= 0x8000 else v


# ─── Token Vocabulary and Embedding Tables ────────────────────────

class TokenVocab:
    """Fixed token vocabulary for the compiled transformer.

    Covers all token types that appear in execution traces:
      - Special tokens: PAD, COMMIT, BRANCH_TAKEN, BRANCH_NOT_TAKEN
      - Opcodes: 55 ISA opcodes + TRAP (56 tokens)
      - Byte values: integers 0-255 (256 tokens)
      - SP deltas: -3 to +3 (7 tokens)

    Provides encode(token) -> int and decode(int) -> token,
    plus compiled nn.Embedding and nn.Linear (unembedding) tables.
    """

    # --- Token type tags (used in encode/decode) ---
    OPCODE = "op"
    VALUE  = "val"
    SP_DELTA = "sp_delta"
    SPECIAL = "special"

    # --- Special token names ---
    PAD = "PAD"
    COMMIT = "COMMIT"
    BRANCH_TAKEN = "BRANCH_TAKEN"
    BRANCH_NOT_TAKEN = "BRANCH_NOT_TAKEN"

    # --- Layout: reserved ranges ---
    _SPECIAL_BASE = 0
    _SPECIAL_COUNT = 4
    _OPCODE_BASE = _SPECIAL_BASE + _SPECIAL_COUNT       # 4
    _OPCODE_COUNT = N_OPCODES + 1                        # 55 opcodes + TRAP = 56
    _VALUE_BASE = _OPCODE_BASE + _OPCODE_COUNT           # 60
    _VALUE_COUNT = 256                                    # byte values 0-255
    _SP_DELTA_BASE = _VALUE_BASE + _VALUE_COUNT           # 316
    _SP_DELTA_MIN = -3
    _SP_DELTA_MAX = 3
    _SP_DELTA_COUNT = _SP_DELTA_MAX - _SP_DELTA_MIN + 1  # 7

    VOCAB_SIZE = _SP_DELTA_BASE + _SP_DELTA_COUNT         # 323

    # --- Special token IDs ---
    PAD_ID = 0
    COMMIT_ID = 1
    BRANCH_TAKEN_ID = 2
    BRANCH_NOT_TAKEN_ID = 3

    _SPECIAL_NAMES = {
        PAD_ID: PAD,
        COMMIT_ID: COMMIT,
        BRANCH_TAKEN_ID: BRANCH_TAKEN,
        BRANCH_NOT_TAKEN_ID: BRANCH_NOT_TAKEN,
    }
    _SPECIAL_IDS = {v: k for k, v in _SPECIAL_NAMES.items()}

    # --- Opcode <-> token ID mapping ---
    # OP codes are 1-55 (contiguous) + 99 (TRAP).
    # Map them to sequential slots starting at _OPCODE_BASE.
    _OPCODE_TO_SLOT = {}  # built in __init_subclass__ / class body
    _SLOT_TO_OPCODE = {}

    def __init__(self):
        # Build opcode mappings from the canonical OPCODE_IDX map
        self._opcode_to_tid = {}
        self._tid_to_opcode = {}
        for op_code, idx in OPCODE_IDX.items():
            tid = self._OPCODE_BASE + idx
            self._opcode_to_tid[op_code] = tid
            self._tid_to_opcode[tid] = op_code
        # TRAP gets the last opcode slot
        trap_tid = self._OPCODE_BASE + N_OPCODES  # slot 59
        self._opcode_to_tid[OP_TRAP] = trap_tid
        self._tid_to_opcode[trap_tid] = OP_TRAP

        self.vocab_size = self.VOCAB_SIZE

    # ---- encode / decode ----

    def encode(self, token):
        """Encode a token to its vocabulary ID.

        Args:
            token: one of:
              - ("op", opcode_int)       e.g. ("op", 1) for PUSH
              - ("val", int_0_to_255)    e.g. ("val", 42)
              - ("sp_delta", int)        e.g. ("sp_delta", -1)
              - ("special", name_str)    e.g. ("special", "COMMIT")
              - str for special names    e.g. "PAD", "COMMIT"

        Returns:
            int: token ID in [0, vocab_size)
        """
        if isinstance(token, str):
            # Shorthand: bare string = special token
            if token in self._SPECIAL_IDS:
                return self._SPECIAL_IDS[token]
            raise ValueError(f"Unknown special token: {token!r}")

        tag, val = token
        if tag == self.OPCODE:
            if val not in self._opcode_to_tid:
                raise ValueError(f"Unknown opcode: {val}")
            return self._opcode_to_tid[val]
        elif tag == self.VALUE:
            if not (0 <= val <= 255):
                raise ValueError(f"Value out of byte range: {val}")
            return self._VALUE_BASE + val
        elif tag == self.SP_DELTA:
            if not (self._SP_DELTA_MIN <= val <= self._SP_DELTA_MAX):
                raise ValueError(
                    f"SP delta {val} out of range [{self._SP_DELTA_MIN}, {self._SP_DELTA_MAX}]"
                )
            return self._SP_DELTA_BASE + (val - self._SP_DELTA_MIN)
        elif tag == self.SPECIAL:
            if val in self._SPECIAL_IDS:
                return self._SPECIAL_IDS[val]
            raise ValueError(f"Unknown special token: {val!r}")
        else:
            raise ValueError(f"Unknown token tag: {tag!r}")

    def decode(self, tid):
        """Decode a token ID back to its structured representation.

        Returns:
            tuple: (tag, value) — e.g. ("op", 1), ("val", 42), ("special", "PAD")
        """
        if not (0 <= tid < self.vocab_size):
            raise ValueError(f"Token ID {tid} out of range [0, {self.vocab_size})")

        if tid < self._OPCODE_BASE:
            # Special token
            return (self.SPECIAL, self._SPECIAL_NAMES[tid])
        elif tid < self._VALUE_BASE:
            # Opcode
            return (self.OPCODE, self._tid_to_opcode[tid])
        elif tid < self._SP_DELTA_BASE:
            # Byte value
            return (self.VALUE, tid - self._VALUE_BASE)
        else:
            # SP delta
            return (self.SP_DELTA, (tid - self._SP_DELTA_BASE) + self._SP_DELTA_MIN)

    # ---- Compiled embedding / unembedding ----

    def compile_embedding(self, d_model=None):
        """Build nn.Embedding with analytically set weights.

        Each token's embedding row is set so that:
          - Opcode tokens: DIM_OPCODE = opcode_int, DIM_IS_* = 1.0, DIM_ONE = 1.0
          - Value tokens: DIM_VALUE = value_int, DIM_ONE = 1.0
          - SP delta tokens: DIM_SP = delta_int, DIM_ONE = 1.0
          - Special tokens: DIM_ONE = 1.0 (PAD is all zeros)

        Returns:
            nn.Embedding with shape (vocab_size, d_model), weights frozen.
        """
        if d_model is None:
            d_model = D_MODEL
        emb = nn.Embedding(self.vocab_size, d_model)
        W = torch.zeros(self.vocab_size, d_model, dtype=DTYPE)

        # Special tokens
        # PAD: all zeros (row 0 stays zero)
        # COMMIT: unique signature via DIM_OPCODE = -1
        W[self.COMMIT_ID, DIM_ONE] = 1.0
        W[self.COMMIT_ID, DIM_OPCODE] = -1.0
        # BRANCH_TAKEN: unique signature via DIM_OPCODE = -2
        W[self.BRANCH_TAKEN_ID, DIM_ONE] = 1.0
        W[self.BRANCH_TAKEN_ID, DIM_OPCODE] = -2.0
        # BRANCH_NOT_TAKEN: unique signature via DIM_OPCODE = -3
        W[self.BRANCH_NOT_TAKEN_ID, DIM_ONE] = 1.0
        W[self.BRANCH_NOT_TAKEN_ID, DIM_OPCODE] = -3.0

        # Opcode tokens — match embed_program_token(pos=0, Instruction(op, 0))
        for op_code, tid in self._opcode_to_tid.items():
            W[tid, DIM_IS_PROG] = 1.0
            W[tid, DIM_OPCODE] = float(op_code)
            W[tid, DIM_ONE] = 1.0
            dim = OPCODE_DIM_MAP.get(op_code)
            if dim is not None and dim < d_model:
                W[tid, dim] = 1.0

        # Value tokens — match embed_stack_entry(addr=0, value=v, write_order=0)
        # DIM_IS_STACK marks these as value-type (matching embed_stack_entry)
        for v in range(256):
            tid = self._VALUE_BASE + v
            W[tid, DIM_IS_STACK] = 1.0
            W[tid, DIM_VALUE] = float(v)
            W[tid, DIM_ONE] = 1.0

        # SP delta tokens — DIM_IS_STATE marks these as state-type (matching embed_state)
        for delta in range(self._SP_DELTA_MIN, self._SP_DELTA_MAX + 1):
            tid = self._SP_DELTA_BASE + (delta - self._SP_DELTA_MIN)
            W[tid, DIM_IS_STATE] = 1.0
            W[tid, DIM_SP] = float(delta)
            W[tid, DIM_ONE] = 1.0

        emb.weight = nn.Parameter(W, requires_grad=False)
        return emb

    def compile_unembedding(self, embedding=None, d_model=None):
        """Build nn.Linear unembedding head s.t. argmax(unembed(embed(tok))) == tok.

        Uses the transpose of the embedding matrix as the unembedding weight,
        with per-token bias correction to handle varying embedding norms.

        Args:
            embedding: nn.Embedding from compile_embedding(). If None, builds one.
            d_model: model dimension (used only if embedding is None).

        Returns:
            nn.Linear(d_model, vocab_size) with frozen weights.
        """
        if d_model is None:
            d_model = D_MODEL
        if embedding is None:
            embedding = self.compile_embedding(d_model)

        E = embedding.weight.data  # (vocab_size, d_model)
        unembed = nn.Linear(d_model, self.vocab_size, bias=True)

        # W = E, bias = -0.5 * ||e_i||^2  →  score_i = e_i · x - 0.5||e_i||^2
        # This is equivalent to finding the nearest embedding by dot product,
        # corrected for norm differences.
        norms_sq = (E * E).sum(dim=1)  # (vocab_size,)
        unembed.weight = nn.Parameter(E.clone(), requires_grad=False)
        unembed.bias = nn.Parameter(-0.5 * norms_sq, requires_grad=False)

        return unembed

    def opcode_name(self, op_code):
        """Human-readable name for an opcode."""
        return OP_NAMES.get(op_code, f"?{op_code}")

    def token_name(self, tid):
        """Human-readable name for a token ID."""
        tag, val = self.decode(tid)
        if tag == self.SPECIAL:
            return val
        elif tag == self.OPCODE:
            return self.opcode_name(val)
        elif tag == self.VALUE:
            return f"V{val}"
        elif tag == self.SP_DELTA:
            return f"SP{'+' if val >= 0 else ''}{val}"
        return f"?{tid}"

    def __repr__(self):
        return (f"TokenVocab(vocab_size={self.vocab_size}, "
                f"opcodes={len(self._opcode_to_tid)}, "
                f"values=256, sp_deltas={self._SP_DELTA_COUNT}, "
                f"specials={self._SPECIAL_COUNT})")


# ─── Compiled Attention Head (from phase12) ───────────────────────

class CompiledAttentionHead(nn.Module):
    """Hard-max attention head with analytically set W_Q, W_K, W_V.

    Computes:
      q = W_Q @ query_embedding           (head_dim,)
      K = W_K @ memory_embeddings          (N, head_dim)
      V = W_V @ memory_embeddings          (N, v_dim)
      scores = K @ q                       (N,)
      output = V[argmax(scores)]           (v_dim,)

    head_dim=2 for parabolic key space.
    v_dim=1 for scalar value extraction.
    """

    def __init__(self, d_model=D_MODEL, head_dim=2, v_dim=1, use_bias_q=False):
        super().__init__()
        self.W_Q = nn.Linear(d_model, head_dim, bias=use_bias_q)
        self.W_K = nn.Linear(d_model, head_dim, bias=False)
        self.W_V = nn.Linear(d_model, v_dim, bias=False)
        self.double()

    def forward(self, query_emb, memory_embs):
        """Hard-max attention lookup.

        Args:
            query_emb: (D,) single query embedding (float64)
            memory_embs: (N, D) memory entries to attend over (float64)

        Returns:
            value: (v_dim,) extracted from the best-matching entry
            score: scalar, the winning attention score
            idx: int, the index of the selected entry
        """
        if memory_embs.shape[0] == 0:
            return torch.zeros(self.W_V.out_features, dtype=DTYPE), \
                   torch.tensor(-float('inf'), dtype=DTYPE), -1

        q = self.W_Q(query_emb)
        K = self.W_K(memory_embs)
        V = self.W_V(memory_embs)

        scores = K @ q
        best = scores.argmax().item()

        return V[best], scores[best], best


# ─── Embedding Functions (phase14 versions with OPCODE_DIM_MAP) ──

def embed_program_token(pos, instr):
    """Create 36-dim embedding for a program instruction."""
    emb = torch.zeros(D_MODEL, dtype=DTYPE)
    emb[DIM_IS_PROG]     = 1.0
    emb[DIM_PROG_KEY_0]  = 2.0 * pos
    emb[DIM_PROG_KEY_1]  = -float(pos * pos)
    emb[DIM_OPCODE]      = float(instr.op)
    emb[DIM_VALUE]       = float(instr.arg)
    emb[DIM_ONE]         = 1.0
    dim = OPCODE_DIM_MAP.get(instr.op)
    if dim is not None:
        emb[dim] = 1.0
    return emb


def embed_stack_entry(addr, value, write_order):
    """Create 36-dim embedding for a stack write record."""
    emb = torch.zeros(D_MODEL, dtype=DTYPE)
    emb[DIM_IS_STACK]     = 1.0
    emb[DIM_STACK_KEY_0]  = 2.0 * addr
    emb[DIM_STACK_KEY_1]  = -float(addr * addr) + EPS * write_order
    emb[DIM_VALUE]        = float(value)
    emb[DIM_ONE]          = 1.0
    return emb


def embed_local_entry(local_idx, value, write_order):
    """Create embedding for a local variable write record."""
    emb = torch.zeros(D_MODEL, dtype=DTYPE)
    emb[DIM_IS_LOCAL]     = 1.0
    emb[DIM_LOCAL_KEY_0]  = 2.0 * local_idx
    emb[DIM_LOCAL_KEY_1]  = -float(local_idx * local_idx) + EPS * write_order
    emb[DIM_VALUE]        = float(value)
    emb[DIM_ONE]          = 1.0
    return emb


def embed_heap_entry(addr, value, write_order):
    """Create embedding for a heap memory write record."""
    emb = torch.zeros(D_MODEL, dtype=DTYPE)
    emb[DIM_IS_HEAP]      = 1.0
    emb[DIM_HEAP_KEY_0]   = 2.0 * addr
    emb[DIM_HEAP_KEY_1]   = -float(addr * addr) + EPS * write_order
    emb[DIM_VALUE]        = float(value)
    emb[DIM_ONE]          = 1.0
    return emb


def embed_call_frame(depth, ret_addr, saved_sp, locals_base, write_order):
    """Create embedding for a call stack frame."""
    emb = torch.zeros(D_MODEL, dtype=DTYPE)
    emb[DIM_IS_CALL_STACK]    = 1.0
    emb[DIM_CALL_KEY_0]       = 2.0 * depth
    emb[DIM_CALL_KEY_1]       = -float(depth * depth) + EPS * write_order
    emb[DIM_CALL_RET_ADDR]    = float(ret_addr)
    emb[DIM_CALL_SAVED_SP]    = float(saved_sp)
    emb[DIM_CALL_LOCALS_BASE] = float(locals_base)
    emb[DIM_ONE]              = 1.0
    return emb


def embed_state(ip, sp):
    """Create 36-dim query embedding encoding current execution state."""
    emb = torch.zeros(D_MODEL, dtype=DTYPE)
    emb[DIM_IS_STATE] = 1.0
    emb[DIM_IP]       = float(ip)
    emb[DIM_SP]       = float(sp)
    emb[DIM_ONE]      = 1.0
    return emb


# ─── Test Utilities (from phase13/phase14) ────────────────────────

def compare_traces(trace_a, trace_b):
    """Compare two traces token by token. Returns (match: bool, detail: str)."""
    if len(trace_a.steps) != len(trace_b.steps):
        return False, f"length mismatch: {len(trace_a.steps)} vs {len(trace_b.steps)}"
    for i, (a, b) in enumerate(zip(trace_a.steps, trace_b.steps)):
        if a.tokens() != b.tokens():
            return False, f"step {i}: {a.tokens()} vs {b.tokens()}"
    return True, "match"


def _test_algorithm(name, prog, expected, np_exec, pt_exec, verbose=False):
    """Run an algorithm on both executors and verify."""
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)

    np_top = np_trace.steps[-1].top if np_trace.steps else None
    pt_top = pt_trace.steps[-1].top if pt_trace.steps else None
    match, detail = compare_traces(np_trace, pt_trace)

    np_ok = (np_top == expected)
    pt_ok = (pt_top == expected)
    all_ok = np_ok and pt_ok and match

    status = "PASS" if all_ok else "FAIL"
    print(f"  {status}  {name:30s}  expected={expected:>6}  "
          f"numpy={np_top:>6}  torch={pt_top:>6}  "
          f"steps={len(np_trace.steps):>4}  trace_match={'Y' if match else 'N'}")

    if not all_ok and verbose:
        if not match:
            print(f"         Trace mismatch: {detail}")
        if not np_ok:
            print(f"         NumPy wrong: got {np_top}, expected {expected}")
        if not pt_ok:
            print(f"         PyTorch wrong: got {pt_top}, expected {expected}")

    return all_ok, len(np_trace.steps)


def _test_trap_algorithm(name, prog, np_exec, pt_exec, verbose=False):
    """Run a program expected to TRAP on both executors. Returns True if both trap."""
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)

    np_trapped = np_trace.steps and np_trace.steps[-1].op == OP_TRAP
    pt_trapped = pt_trace.steps and pt_trace.steps[-1].op == OP_TRAP
    match, detail = compare_traces(np_trace, pt_trace)
    all_ok = np_trapped and pt_trapped and match

    status = "PASS" if all_ok else "FAIL"
    np_label = f"TRAP@{len(np_trace.steps)}" if np_trapped else f"top={np_trace.steps[-1].top if np_trace.steps else '?'}"
    pt_label = f"TRAP@{len(pt_trace.steps)}" if pt_trapped else f"top={pt_trace.steps[-1].top if pt_trace.steps else '?'}"
    print(f"  {status}  {name:30s}  numpy={np_label:>10}  torch={pt_label:>10}  "
          f"trace_match={'Y' if match else 'N'}")

    if not all_ok and verbose:
        if not np_trapped:
            print(f"         NumPy did not trap")
        if not pt_trapped:
            print(f"         PyTorch did not trap")
        if not match:
            print(f"         Trace mismatch: {detail}")

    return all_ok
