"""Flattened executor classes for the compiled transformer stack machine.

Consolidated from phase11 → phase12 → phase13 → phase14. No inheritance chain.
Contains:
  - NumPyExecutor: full 42-opcode numpy executor (flattened from Phase14Executor)
  - CompiledModel: PyTorch nn.Module with compiled weights (flattened from Phase14Model)
  - TorchExecutor: executes programs via CompiledModel (flattened from Phase14PyTorchExecutor)
"""

import numpy as np
import torch
import torch.nn as nn

import ff_symbolic
from isa import (
    Instruction, Trace, TraceStep,
    CompiledAttentionHead,
    embed_program_token, embed_stack_entry, embed_state,
    embed_local_entry, embed_heap_entry, embed_call_frame,
    D_MODEL, DTYPE, EPS, N_OPCODES,
    DIM_IS_PROG, DIM_IS_STACK, DIM_IS_STATE,
    DIM_PROG_KEY_0, DIM_PROG_KEY_1,
    DIM_STACK_KEY_0, DIM_STACK_KEY_1,
    DIM_IS_LOCAL, DIM_LOCAL_KEY_0, DIM_LOCAL_KEY_1,
    DIM_IS_HEAP, DIM_HEAP_KEY_0, DIM_HEAP_KEY_1,
    DIM_IS_CALL_STACK, DIM_CALL_KEY_0, DIM_CALL_KEY_1,
    DIM_CALL_RET_ADDR, DIM_CALL_SAVED_SP, DIM_CALL_LOCALS_BASE,
    DIM_OPCODE, DIM_VALUE, DIM_IP, DIM_SP, DIM_ONE,
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
    OP_LOCAL_GET, OP_LOCAL_SET, OP_LOCAL_TEE,
    OP_I32_LOAD, OP_I32_STORE,
    OP_I32_LOAD8_U, OP_I32_LOAD8_S,
    OP_I32_LOAD16_U, OP_I32_LOAD16_S,
    OP_I32_STORE8, OP_I32_STORE16,
    OP_CALL, OP_RETURN,
    OP_TRAP,
    OPCODE_DIM_MAP, OPCODE_IDX, NONLINEAR_OPS,
    _trunc_div, _trunc_rem, _to_i32, MASK32,
    _shr_u, _shr_s, _rotl32, _rotr32,
    _clz32, _ctz32, _popcnt32,
    _sign_extend_8, _sign_extend_16,
)


# ─── NumPy Executor ───────────────────────────────────────────────

class NumPyExecutor:
    """Compiled numpy executor with full 42-opcode ISA.

    Flattened from Phase14Executor <- Phase13Executor <- ExtendedExecutor <- CompiledExecutorNumpy.
    """

    def execute(self, prog, max_steps=50000):
        trace = Trace(program=prog)

        stack_keys = []
        stack_vals = []
        write_count = 0
        eps = 1e-10

        # Local variables address space (separate from stack)
        locals_keys = []
        locals_vals = []
        local_write_count = 0

        # Heap (linear memory) address space
        heap_keys = []
        heap_vals = []
        heap_write_count = 0

        # Call stack
        call_stack = []  # list of (ret_addr, saved_sp, saved_locals_base)
        locals_base = 0

        ip = 0
        sp = 0

        def stack_write(addr, val):
            nonlocal write_count
            stack_keys.append((2.0*addr, -float(addr*addr) + eps*write_count))
            stack_vals.append(val)
            write_count += 1

        def stack_read(addr):
            if not stack_keys:
                return 0
            keys = np.array(stack_keys)
            q = np.array([addr, 1.0])
            scores = keys @ q
            best = np.argmax(scores)
            stored_addr = round(keys[best, 0] / 2.0)
            return stack_vals[best] if stored_addr == addr else 0

        def local_write(local_idx, val):
            nonlocal local_write_count
            actual_idx = locals_base + local_idx
            locals_keys.append((2.0*actual_idx, -float(actual_idx*actual_idx) + eps*local_write_count))
            locals_vals.append(val)
            local_write_count += 1

        def local_read(local_idx):
            if not locals_keys:
                return 0
            actual_idx = locals_base + local_idx
            keys = np.array(locals_keys)
            q = np.array([actual_idx, 1.0])
            scores = keys @ q
            best = np.argmax(scores)
            stored_idx = round(keys[best, 0] / 2.0)
            return locals_vals[best] if stored_idx == actual_idx else 0

        def heap_write(addr, val):
            nonlocal heap_write_count
            heap_keys.append((2.0*addr, -float(addr*addr) + eps*heap_write_count))
            heap_vals.append(val)
            heap_write_count += 1

        def heap_read(addr):
            if not heap_keys:
                return 0
            keys = np.array(heap_keys)
            q = np.array([addr, 1.0])
            scores = keys @ q
            best = np.argmax(scores)
            stored_addr = round(keys[best, 0] / 2.0)
            return heap_vals[best] if stored_addr == addr else 0

        for step in range(max_steps):
            if ip >= len(prog):
                break

            op = prog[ip].op
            arg = prog[ip].arg
            next_ip = ip + 1
            top = 0

            # ── Phase 4 base ops ──
            if op == OP_PUSH:
                sp += 1
                stack_write(sp, arg)
                top = arg
            elif op == OP_POP:
                sp -= 1
                top = stack_read(sp) if sp > 0 else 0
            elif op == OP_ADD:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = (val_a + val_b) & MASK32
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_SUB:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = (val_b - val_a) & MASK32
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_DUP:
                val = stack_read(sp)
                sp += 1
                stack_write(sp, val)
                top = val

            # ── Phase 13 stack manipulation ──
            elif op == OP_SWAP:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                stack_write(sp, val_b)
                stack_write(sp - 1, val_a)
                top = val_b
            elif op == OP_OVER:
                val_b = stack_read(sp - 1)
                sp += 1
                stack_write(sp, val_b)
                top = val_b
            elif op == OP_ROT:
                val_top    = stack_read(sp)
                val_second = stack_read(sp - 1)
                val_third  = stack_read(sp - 2)
                stack_write(sp, val_third)
                stack_write(sp - 1, val_top)
                stack_write(sp - 2, val_second)
                top = val_third

            # ── Phase 14 arithmetic ──
            elif op == OP_MUL:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = (val_a * val_b) & MASK32
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op in (OP_DIV_S, OP_DIV_U):
                val_a = stack_read(sp)
                if val_a == 0:
                    trace.steps.append(TraceStep(OP_TRAP, 0, sp, 0))
                    break
                val_b = stack_read(sp - 1)
                result = _trunc_div(val_b, val_a) & MASK32
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op in (OP_REM_S, OP_REM_U):
                val_a = stack_read(sp)
                if val_a == 0:
                    trace.steps.append(TraceStep(OP_TRAP, 0, sp, 0))
                    break
                val_b = stack_read(sp - 1)
                result = _trunc_rem(val_b, val_a) & MASK32
                sp -= 1
                stack_write(sp, result)
                top = result

            # ── Phase 14 Chunk 2: comparison ops ──
            elif op == OP_EQZ:
                val_a = stack_read(sp)
                result = 1 if val_a == 0 else 0
                stack_write(sp, result)
                top = result
            elif op in (OP_EQ, OP_NE, OP_LT_S, OP_LT_U, OP_GT_S, OP_GT_U,
                        OP_LE_S, OP_LE_U, OP_GE_S, OP_GE_U):
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                if op in (OP_EQ,):
                    result = 1 if val_a == val_b else 0
                elif op in (OP_NE,):
                    result = 1 if val_a != val_b else 0
                elif op in (OP_LT_S, OP_LT_U):
                    result = 1 if val_b < val_a else 0
                elif op in (OP_GT_S, OP_GT_U):
                    result = 1 if val_b > val_a else 0
                elif op in (OP_LE_S, OP_LE_U):
                    result = 1 if val_b <= val_a else 0
                elif op in (OP_GE_S, OP_GE_U):
                    result = 1 if val_b >= val_a else 0
                sp -= 1
                stack_write(sp, result)
                top = result

            # ── Phase 14 Chunk 3: bitwise ops ──
            elif op == OP_AND:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = _to_i32(val_a) & _to_i32(val_b)
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_OR:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = _to_i32(val_a) | _to_i32(val_b)
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_XOR:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = _to_i32(val_a) ^ _to_i32(val_b)
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_SHL:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = (_to_i32(val_b) << (int(val_a) & 31)) & MASK32
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_SHR_S:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = _shr_s(val_b, val_a)
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_SHR_U:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = _shr_u(val_b, val_a)
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_ROTL:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = _rotl32(val_b, val_a)
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_ROTR:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = _rotr32(val_b, val_a)
                sp -= 1
                stack_write(sp, result)
                top = result

            # ── Phase 14 Chunk 4: unary + parametric ops ──
            elif op == OP_CLZ:
                val_a = stack_read(sp)
                result = _clz32(val_a)
                stack_write(sp, result)
                top = result
            elif op == OP_CTZ:
                val_a = stack_read(sp)
                result = _ctz32(val_a)
                stack_write(sp, result)
                top = result
            elif op == OP_POPCNT:
                val_a = stack_read(sp)
                result = _popcnt32(val_a)
                stack_write(sp, result)
                top = result
            elif op == OP_ABS:
                val_a = stack_read(sp)
                result = abs(int(val_a))
                stack_write(sp, result)
                top = result
            elif op == OP_NEG:
                val_a = stack_read(sp)
                result = (-int(val_a)) & MASK32
                stack_write(sp, result)
                top = result
            elif op == OP_SELECT:
                val_a = stack_read(sp)       # c (condition)
                val_b = stack_read(sp - 1)   # b (false value)
                val_c = stack_read(sp - 2)   # a (true value)
                result = val_c if val_a != 0 else val_b
                sp -= 2
                stack_write(sp, result)
                top = result

            # ── Phase 15: local variables ──
            elif op == OP_LOCAL_GET:
                val = local_read(arg)
                sp += 1
                stack_write(sp, val)
                top = val
            elif op == OP_LOCAL_SET:
                val = stack_read(sp)
                sp -= 1
                local_write(arg, val)
                top = stack_read(sp) if sp > 0 else 0
            elif op == OP_LOCAL_TEE:
                val = stack_read(sp)
                local_write(arg, val)
                top = val

            # ── Phase 16: linear memory ──
            elif op == OP_I32_LOAD:
                addr = stack_read(sp)
                val = heap_read(int(addr))
                stack_write(sp, val)
                top = val
            elif op == OP_I32_STORE:
                val = stack_read(sp)
                addr = stack_read(sp - 1)
                heap_write(int(addr), val)
                sp -= 2
                top = stack_read(sp) if sp > 0 else 0
            elif op == OP_I32_LOAD8_U:
                addr = stack_read(sp)
                val = heap_read(int(addr))
                result = int(val) & 0xFF
                stack_write(sp, result)
                top = result
            elif op == OP_I32_LOAD8_S:
                addr = stack_read(sp)
                val = heap_read(int(addr))
                result = _sign_extend_8(val)
                stack_write(sp, result)
                top = result
            elif op == OP_I32_LOAD16_U:
                addr = stack_read(sp)
                val = heap_read(int(addr))
                result = int(val) & 0xFFFF
                stack_write(sp, result)
                top = result
            elif op == OP_I32_LOAD16_S:
                addr = stack_read(sp)
                val = heap_read(int(addr))
                result = _sign_extend_16(val)
                stack_write(sp, result)
                top = result
            elif op == OP_I32_STORE8:
                val = stack_read(sp)
                addr = stack_read(sp - 1)
                heap_write(int(addr), int(val) & 0xFF)
                sp -= 2
                top = stack_read(sp) if sp > 0 else 0
            elif op == OP_I32_STORE16:
                val = stack_read(sp)
                addr = stack_read(sp - 1)
                heap_write(int(addr), int(val) & 0xFFFF)
                sp -= 2
                top = stack_read(sp) if sp > 0 else 0

            # ── Phase 17: function calls ──
            elif op == OP_CALL:
                call_stack.append((ip + 1, sp, locals_base))
                locals_base = len(locals_keys)
                top = stack_read(sp) if sp > 0 else 0
                next_ip = arg
            elif op == OP_RETURN:
                if not call_stack:
                    trace.steps.append(TraceStep(OP_TRAP, 0, sp, 0))
                    break
                ret_val = stack_read(sp)
                ret_addr, saved_sp, saved_locals_base = call_stack.pop()
                sp = saved_sp + 1
                stack_write(sp, ret_val)
                locals_base = saved_locals_base
                top = ret_val
                next_ip = ret_addr

            # ── Control flow ──
            elif op == OP_JZ:
                cond = stack_read(sp)
                sp -= 1
                top = stack_read(sp) if sp > 0 else 0
                if cond == 0:
                    next_ip = arg
            elif op == OP_JNZ:
                cond = stack_read(sp)
                sp -= 1
                top = stack_read(sp) if sp > 0 else 0
                if cond != 0:
                    next_ip = arg
            elif op == OP_NOP:
                top = stack_read(sp) if sp > 0 else 0
            elif op == OP_HALT:
                top = stack_read(sp) if sp > 0 else 0
                trace.steps.append(TraceStep(op, arg, sp, top))
                break
            else:
                # Unknown opcode — treat as NOP
                top = stack_read(sp) if sp > 0 else 0

            trace.steps.append(TraceStep(op, arg, sp, top))
            ip = next_ip

        return trace


# ─── Compiled PyTorch Model ──────────────────────────────────────

class CompiledModel(nn.Module):
    """Compiled transformer with 10 attention heads and linear+nonlinear FF dispatch.

    Architecture:
      d_model=51, head_dim=2 (2D parabolic key space)
      10 active attention heads:
        Head 0: program opcode fetch
        Head 1: program arg fetch
        Head 2: stack read at SP
        Head 3: stack read at SP-1
        Head 4: stack read at SP-2
        Head 5: local value fetch
        Head 6: local address verify
        Head 7: heap value fetch
        Head 8: heap address verify
        Head 9: call stack read (v_dim=3: ret_addr, saved_sp, locals_base)
      FF dispatch:
        M_top: linear routing matrix
        Nonlinear override: explicit computation for complex ops
      sp_deltas: per-opcode stack pointer delta
    """

    def __init__(self, d_model=D_MODEL):
        nn.Module.__init__(self)
        self.d_model = d_model

        # Heads 0-4
        self.head_prog_op  = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        self.head_prog_arg = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        self.head_stack_a  = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        self.head_stack_b  = CompiledAttentionHead(d_model, head_dim=2, v_dim=1, use_bias_q=True)
        self.head_stack_c  = CompiledAttentionHead(d_model, head_dim=2, v_dim=1, use_bias_q=True)
        # Heads 5-6: local variable access
        self.head_local_val   = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        self.head_local_addr  = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        # Heads 7-8: heap (linear memory) access
        self.head_heap_val    = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        self.head_heap_addr   = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        # Head 9: call stack read (v_dim=3 for ret_addr, saved_sp, locals_base)
        self.head_call_stack  = CompiledAttentionHead(d_model, head_dim=2, v_dim=3)

        # FF dispatch: N_OPCODES opcodes, 6 value inputs (arg, va, vb, vc, local_val, heap_val)
        self.register_buffer('M_top', torch.zeros(N_OPCODES, 6, dtype=DTYPE))
        self.register_buffer('sp_deltas', torch.zeros(N_OPCODES, dtype=DTYPE))

        self._compile_weights()

    def _compile_weights(self):
        """Set all weight matrices analytically."""
        with torch.no_grad():
            # ── Head 0: Program opcode fetch ──
            W = torch.zeros(2, self.d_model)
            W[0, DIM_IP]  = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_prog_op.W_Q.weight.copy_(W)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_PROG_KEY_0] = 1.0
            W[1, DIM_PROG_KEY_1] = 1.0
            self.head_prog_op.W_K.weight.copy_(W)

            W = torch.zeros(1, self.d_model)
            W[0, DIM_OPCODE] = 1.0
            self.head_prog_op.W_V.weight.copy_(W)

            # ── Head 1: Program argument fetch ──
            W = torch.zeros(2, self.d_model)
            W[0, DIM_IP]  = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_prog_arg.W_Q.weight.copy_(W)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_PROG_KEY_0] = 1.0
            W[1, DIM_PROG_KEY_1] = 1.0
            self.head_prog_arg.W_K.weight.copy_(W)

            W = torch.zeros(1, self.d_model)
            W[0, DIM_VALUE] = 1.0
            self.head_prog_arg.W_V.weight.copy_(W)

            # ── Head 2: Stack read at SP ──
            W = torch.zeros(2, self.d_model)
            W[0, DIM_SP]  = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_stack_a.W_Q.weight.copy_(W)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_STACK_KEY_0] = 1.0
            W[1, DIM_STACK_KEY_1] = 1.0
            self.head_stack_a.W_K.weight.copy_(W)

            W = torch.zeros(1, self.d_model)
            W[0, DIM_VALUE] = 1.0
            self.head_stack_a.W_V.weight.copy_(W)

            # ── Head 3: Stack read at SP-1 ──
            W = torch.zeros(2, self.d_model)
            W[0, DIM_SP]  = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_stack_b.W_Q.weight.copy_(W)
            b = torch.zeros(2)
            b[0] = -1.0
            self.head_stack_b.W_Q.bias.copy_(b)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_STACK_KEY_0] = 1.0
            W[1, DIM_STACK_KEY_1] = 1.0
            self.head_stack_b.W_K.weight.copy_(W)

            W = torch.zeros(1, self.d_model)
            W[0, DIM_VALUE] = 1.0
            self.head_stack_b.W_V.weight.copy_(W)

            # ── Head 4: Stack read at SP-2 ──
            W = torch.zeros(2, self.d_model)
            W[0, DIM_SP]  = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_stack_c.W_Q.weight.copy_(W)
            b = torch.zeros(2)
            b[0] = -2.0
            self.head_stack_c.W_Q.bias.copy_(b)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_STACK_KEY_0] = 1.0
            W[1, DIM_STACK_KEY_1] = 1.0
            self.head_stack_c.W_K.weight.copy_(W)

            W = torch.zeros(1, self.d_model)
            W[0, DIM_VALUE] = 1.0
            self.head_stack_c.W_V.weight.copy_(W)

            # ── Head 5: Local value fetch ──
            # Query uses arg from instruction (local index), searches locals space
            W = torch.zeros(2, self.d_model)
            W[0, DIM_VALUE]  = 1.0   # query = instruction arg (local index)
            W[1, DIM_ONE]    = 1.0
            self.head_local_val.W_Q.weight.copy_(W)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_LOCAL_KEY_0] = 1.0
            W[1, DIM_LOCAL_KEY_1] = 1.0
            self.head_local_val.W_K.weight.copy_(W)

            W = torch.zeros(1, self.d_model)
            W[0, DIM_VALUE] = 1.0
            self.head_local_val.W_V.weight.copy_(W)

            # ── Head 6: Local address verify ──
            # Same Q/K as head 5, but V extracts the key[0]/2 for address checking
            W = torch.zeros(2, self.d_model)
            W[0, DIM_VALUE]  = 1.0
            W[1, DIM_ONE]    = 1.0
            self.head_local_addr.W_Q.weight.copy_(W)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_LOCAL_KEY_0] = 1.0
            W[1, DIM_LOCAL_KEY_1] = 1.0
            self.head_local_addr.W_K.weight.copy_(W)

            W = torch.zeros(1, self.d_model)
            W[0, DIM_LOCAL_KEY_0] = 0.5  # extracts addr = key[0]/2
            self.head_local_addr.W_V.weight.copy_(W)

            # ── Head 7: Heap value fetch ──
            # Query uses the address from stack (val_a), constructed dynamically in forward()
            W = torch.zeros(2, self.d_model)
            W[0, DIM_VALUE]  = 1.0   # query = heap address
            W[1, DIM_ONE]    = 1.0
            self.head_heap_val.W_Q.weight.copy_(W)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_HEAP_KEY_0] = 1.0
            W[1, DIM_HEAP_KEY_1] = 1.0
            self.head_heap_val.W_K.weight.copy_(W)

            W = torch.zeros(1, self.d_model)
            W[0, DIM_VALUE] = 1.0
            self.head_heap_val.W_V.weight.copy_(W)

            # ── Head 8: Heap address verify ──
            W = torch.zeros(2, self.d_model)
            W[0, DIM_VALUE]  = 1.0
            W[1, DIM_ONE]    = 1.0
            self.head_heap_addr.W_Q.weight.copy_(W)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_HEAP_KEY_0] = 1.0
            W[1, DIM_HEAP_KEY_1] = 1.0
            self.head_heap_addr.W_K.weight.copy_(W)

            W = torch.zeros(1, self.d_model)
            W[0, DIM_HEAP_KEY_0] = 0.5  # extracts addr = key[0]/2
            self.head_heap_addr.W_V.weight.copy_(W)

            # ── Head 9: Call stack read ──
            # Query uses call_depth to find most recent frame
            # Q extracts (call_depth, 1) from a synthetic query constructed in executor
            W = torch.zeros(2, self.d_model)
            W[0, DIM_VALUE]  = 1.0   # query = call_depth
            W[1, DIM_ONE]    = 1.0
            self.head_call_stack.W_Q.weight.copy_(W)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_CALL_KEY_0] = 1.0
            W[1, DIM_CALL_KEY_1] = 1.0
            self.head_call_stack.W_K.weight.copy_(W)

            # V extracts [ret_addr, saved_sp, locals_base] (v_dim=3)
            W = torch.zeros(3, self.d_model)
            W[0, DIM_CALL_RET_ADDR]    = 1.0
            W[1, DIM_CALL_SAVED_SP]    = 1.0
            W[2, DIM_CALL_LOCALS_BASE] = 1.0
            self.head_call_stack.W_V.weight.copy_(W)

            # ── FF dispatch: linear routing ──
            # M_top maps [arg, val_a, val_b, val_c, local_val, heap_val] -> candidate top per opcode
            #                         arg  va   vb   vc   lv   hv
            self.M_top[0]  = torch.tensor([ 1.,  0.,  0.,  0.,  0.,  0.])  # PUSH: top = arg
            self.M_top[1]  = torch.tensor([ 0.,  0.,  1.,  0.,  0.,  0.])  # POP:  top = val_b
            self.M_top[2]  = torch.tensor([ 0.,  0.,  0.,  0.,  0.,  0.])  # ADD:  nonlinear (i32-masked)
            self.M_top[3]  = torch.tensor([ 0.,  1.,  0.,  0.,  0.,  0.])  # DUP:  top = va
            self.M_top[4]  = torch.tensor([ 0.,  1.,  0.,  0.,  0.,  0.])  # HALT: top = va
            self.M_top[5]  = torch.tensor([ 0.,  0.,  0.,  0.,  0.,  0.])  # SUB:  nonlinear (i32-masked)
            self.M_top[6]  = torch.tensor([ 0.,  0.,  1.,  0.,  0.,  0.])  # JZ:   top = vb
            self.M_top[7]  = torch.tensor([ 0.,  0.,  1.,  0.,  0.,  0.])  # JNZ:  top = vb
            self.M_top[8]  = torch.tensor([ 0.,  1.,  0.,  0.,  0.,  0.])  # NOP:  top = va
            self.M_top[9]  = torch.tensor([ 0.,  0.,  1.,  0.,  0.,  0.])  # SWAP: top = vb
            self.M_top[10] = torch.tensor([ 0.,  0.,  1.,  0.,  0.,  0.])  # OVER: top = vb
            self.M_top[11] = torch.tensor([ 0.,  0.,  0.,  1.,  0.,  0.])  # ROT:  top = vc

            # Nonlinear ops: M_top rows stay zero — results computed in forward()

            # LOCAL ops use linear routing via the local_val input
            self.M_top[OPCODE_IDX[OP_LOCAL_GET]] = torch.tensor([ 0.,  0.,  0.,  0.,  1.,  0.])  # LOCAL.GET: top = local_val
            self.M_top[OPCODE_IDX[OP_LOCAL_SET]] = torch.tensor([ 0.,  0.,  1.,  0.,  0.,  0.])  # LOCAL.SET: top = val_b (new stack top after pop)
            self.M_top[OPCODE_IDX[OP_LOCAL_TEE]] = torch.tensor([ 0.,  1.,  0.,  0.,  0.,  0.])  # LOCAL.TEE: top = val_a (stack top unchanged)

            # Memory ops: I32.LOAD uses heap_val, I32.STORE uses vc
            self.M_top[OPCODE_IDX[OP_I32_LOAD]]  = torch.tensor([ 0.,  0.,  0.,  0.,  0.,  1.])  # I32.LOAD: top = heap_val
            self.M_top[OPCODE_IDX[OP_I32_STORE]]  = torch.tensor([ 0.,  0.,  0.,  1.,  0.,  0.])  # I32.STORE: top = vc (after pop 2)
            # Width load variants: nonlinear (handled in forward)
            # Width store variants: top = vc (linear)
            self.M_top[OPCODE_IDX[OP_I32_STORE8]]  = torch.tensor([ 0.,  0.,  0.,  1.,  0.,  0.])
            self.M_top[OPCODE_IDX[OP_I32_STORE16]] = torch.tensor([ 0.,  0.,  0.,  1.,  0.,  0.])

            # CALL/RETURN: top = val_a (current stack top)
            self.M_top[OPCODE_IDX[OP_CALL]]   = torch.tensor([ 0.,  1.,  0.,  0.,  0.,  0.])
            self.M_top[OPCODE_IDX[OP_RETURN]] = torch.tensor([ 0.,  1.,  0.,  0.,  0.,  0.])

            # SP deltas
            # Indices 0-44: same as before
            # Indices 45-52: I32.LOAD(0), I32.STORE(-2), LOAD8_U(0), LOAD8_S(0),
            #                LOAD16_U(0), LOAD16_S(0), STORE8(-2), STORE16(-2)
            # CALL and RETURN sp_deltas are 0 — executor handles sp changes directly
            self.sp_deltas.copy_(torch.tensor(
                [1., -1., -1., 1., 0., -1., -1., -1., 0., 0., 1., 0.,
                 -1., -1., -1., -1., -1.,
                 0.,  -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                 -1., -1., -1., -1., -1., -1., -1., -1.,
                 0., 0., 0., 0., 0., -2.,
                 1., -1., 0.,
                 0., -2., 0., 0., 0., 0., -2., -2.,
                 0., 0.]))

    def forward(self, query_emb, prog_embs, stack_embs, local_embs=None, heap_embs=None,
                call_embs=None, locals_base=0):
        """Execute one step.

        Returns (opcode, arg, sp_delta, top, opcode_one_hot, val_a, val_b, val_c, local_val, heap_val).
        """
        if local_embs is None:
            local_embs = torch.zeros(0, self.d_model, dtype=DTYPE)
        if heap_embs is None:
            heap_embs = torch.zeros(0, self.d_model, dtype=DTYPE)
        if call_embs is None:
            call_embs = torch.zeros(0, self.d_model, dtype=DTYPE)

        # Head 0: Fetch opcode
        opcode_val, _, _ = self.head_prog_op(query_emb, prog_embs)
        # Head 1: Fetch argument
        arg_val, _, _ = self.head_prog_arg(query_emb, prog_embs)

        # Head 2: Read stack[SP]
        if stack_embs.shape[0] > 0:
            val_a_raw, _, idx_a = self.head_stack_a(query_emb, stack_embs)
            stored_addr_a = round(stack_embs[idx_a, DIM_STACK_KEY_0].item() / 2.0)
            queried_sp = round(query_emb[DIM_SP].item())
            val_a = val_a_raw[0] if stored_addr_a == queried_sp else torch.tensor(0.0, dtype=DTYPE)
        else:
            val_a = torch.tensor(0.0, dtype=DTYPE)

        # Head 3: Read stack[SP-1]
        if stack_embs.shape[0] > 0:
            val_b_raw, _, idx_b = self.head_stack_b(query_emb, stack_embs)
            stored_addr_b = round(stack_embs[idx_b, DIM_STACK_KEY_0].item() / 2.0)
            queried_sp_m1 = round(query_emb[DIM_SP].item()) - 1
            val_b = val_b_raw[0] if stored_addr_b == queried_sp_m1 else torch.tensor(0.0, dtype=DTYPE)
        else:
            val_b = torch.tensor(0.0, dtype=DTYPE)

        # Head 4: Read stack[SP-2]
        if stack_embs.shape[0] > 0:
            val_c_raw, _, idx_c = self.head_stack_c(query_emb, stack_embs)
            stored_addr_c = round(stack_embs[idx_c, DIM_STACK_KEY_0].item() / 2.0)
            queried_sp_m2 = round(query_emb[DIM_SP].item()) - 2
            val_c = val_c_raw[0] if stored_addr_c == queried_sp_m2 else torch.tensor(0.0, dtype=DTYPE)
        else:
            val_c = torch.tensor(0.0, dtype=DTYPE)

        # Heads 5-6: Read local variable
        # Query uses locals_base + arg to find the correct local
        arg_raw = round(arg_val[0].item())
        actual_local_idx = locals_base + arg_raw
        if local_embs.shape[0] > 0:
            local_query = torch.zeros(self.d_model, dtype=DTYPE)
            local_query[DIM_VALUE] = float(actual_local_idx)
            local_query[DIM_ONE] = 1.0
            local_val_raw, _, idx_l = self.head_local_val(local_query, local_embs)
            local_addr_raw, _, _ = self.head_local_addr(local_query, local_embs)
            stored_local_addr = round(local_addr_raw[0].item())
            local_val = local_val_raw[0] if stored_local_addr == actual_local_idx else torch.tensor(0.0, dtype=DTYPE)
        else:
            local_val = torch.tensor(0.0, dtype=DTYPE)

        # Decode opcode early so we can use val_a for heap query
        opcode = round(opcode_val[0].item())
        arg = arg_raw

        # Heads 7-8: Read heap memory
        # For LOAD ops, address comes from val_a (stack top)
        # For STORE ops, address comes from val_b (stack[SP-1])
        # We always query with val_a; STORE ops don't use heap_val
        heap_query_addr = round(val_a.item())
        if heap_embs.shape[0] > 0:
            heap_query = torch.zeros(self.d_model, dtype=DTYPE)
            heap_query[DIM_VALUE] = float(heap_query_addr)
            heap_query[DIM_ONE] = 1.0
            heap_val_raw, _, idx_h = self.head_heap_val(heap_query, heap_embs)
            heap_addr_raw, _, _ = self.head_heap_addr(heap_query, heap_embs)
            stored_heap_addr = round(heap_addr_raw[0].item())
            heap_val = heap_val_raw[0] if stored_heap_addr == heap_query_addr else torch.tensor(0.0, dtype=DTYPE)
        else:
            heap_val = torch.tensor(0.0, dtype=DTYPE)

        # FF Dispatch — linear path
        opcode_one_hot = torch.zeros(N_OPCODES, dtype=DTYPE)
        idx = OPCODE_IDX.get(opcode, -1)
        if idx >= 0:
            opcode_one_hot[idx] = 1.0

        values = torch.stack([
            torch.tensor(float(arg), dtype=DTYPE),
            val_a, val_b, val_c, local_val, heap_val
        ])
        candidates = self.M_top @ values
        top_linear = (opcode_one_hot * candidates).sum()

        # FF Dispatch — nonlinear path
        va = round(val_a.item())
        vb = round(val_b.item())

        nonlinear = torch.zeros(N_OPCODES, dtype=DTYPE)
        # ADD / SUB / MUL: route through the bilinear FF forms from
        # ff_symbolic (issue #69). E(v) is the scalar-at-DIM_VALUE
        # embedding, so we lift (va, vb) back into the model's embedding
        # space, evaluate the linear / bilinear forms, and read the
        # scalar back. The & MASK32 step lives here, outside the form,
        # for i32-wrap parity with NumPyExecutor — the form itself
        # computes over Z (see ff_symbolic.range_check / equivalence proof).
        ea = ff_symbolic.E(va, self.d_model)
        eb = ff_symbolic.E(vb, self.d_model)
        nonlinear[OPCODE_IDX[OP_ADD]] = float(ff_symbolic.E_inv(
            ff_symbolic.forward_add(ea, eb)) & MASK32)
        nonlinear[OPCODE_IDX[OP_SUB]] = float(ff_symbolic.E_inv(
            ff_symbolic.forward_sub(ea, eb)) & MASK32)
        nonlinear[OPCODE_IDX[OP_MUL]] = float(ff_symbolic.E_inv(
            ff_symbolic.forward_mul(ea, eb)) & MASK32)
        if va != 0:
            nonlinear[OPCODE_IDX[OP_DIV_S]] = float(_trunc_div(vb, va) & MASK32)
            nonlinear[OPCODE_IDX[OP_DIV_U]] = float(_trunc_div(vb, va) & MASK32)
            nonlinear[OPCODE_IDX[OP_REM_S]] = float(_trunc_rem(vb, va) & MASK32)
            nonlinear[OPCODE_IDX[OP_REM_U]] = float(_trunc_rem(vb, va) & MASK32)

        # Comparison ops
        nonlinear[OPCODE_IDX[OP_EQZ]]  = 1.0 if va == 0 else 0.0
        nonlinear[OPCODE_IDX[OP_EQ]]   = 1.0 if va == vb else 0.0
        nonlinear[OPCODE_IDX[OP_NE]]   = 1.0 if va != vb else 0.0
        nonlinear[OPCODE_IDX[OP_LT_S]] = 1.0 if vb < va else 0.0
        nonlinear[OPCODE_IDX[OP_LT_U]] = 1.0 if vb < va else 0.0
        nonlinear[OPCODE_IDX[OP_GT_S]] = 1.0 if vb > va else 0.0
        nonlinear[OPCODE_IDX[OP_GT_U]] = 1.0 if vb > va else 0.0
        nonlinear[OPCODE_IDX[OP_LE_S]] = 1.0 if vb <= va else 0.0
        nonlinear[OPCODE_IDX[OP_LE_U]] = 1.0 if vb <= va else 0.0
        nonlinear[OPCODE_IDX[OP_GE_S]] = 1.0 if vb >= va else 0.0
        nonlinear[OPCODE_IDX[OP_GE_U]] = 1.0 if vb >= va else 0.0

        # Bitwise ops
        nonlinear[OPCODE_IDX[OP_AND]]   = float(_to_i32(va) & _to_i32(vb))
        nonlinear[OPCODE_IDX[OP_OR]]    = float(_to_i32(va) | _to_i32(vb))
        nonlinear[OPCODE_IDX[OP_XOR]]   = float(_to_i32(va) ^ _to_i32(vb))
        nonlinear[OPCODE_IDX[OP_SHL]]   = float((_to_i32(vb) << (int(va) & 31)) & MASK32)
        nonlinear[OPCODE_IDX[OP_SHR_S]] = float(_shr_s(vb, va))
        nonlinear[OPCODE_IDX[OP_SHR_U]] = float(_shr_u(vb, va))
        nonlinear[OPCODE_IDX[OP_ROTL]]  = float(_rotl32(vb, va))
        nonlinear[OPCODE_IDX[OP_ROTR]]  = float(_rotr32(vb, va))

        # Unary ops
        nonlinear[OPCODE_IDX[OP_CLZ]]    = float(_clz32(va))
        nonlinear[OPCODE_IDX[OP_CTZ]]    = float(_ctz32(va))
        nonlinear[OPCODE_IDX[OP_POPCNT]] = float(_popcnt32(va))
        nonlinear[OPCODE_IDX[OP_ABS]]    = float(abs(int(va)))
        nonlinear[OPCODE_IDX[OP_NEG]]    = float((-int(va)) & MASK32)

        # Parametric: SELECT
        vc = round(val_c.item())
        nonlinear[OPCODE_IDX[OP_SELECT]] = float(vc if va != 0 else vb)

        # Width load variants (nonlinear masking/sign-extension of heap_val)
        hv = round(heap_val.item())
        nonlinear[OPCODE_IDX[OP_I32_LOAD8_U]]  = float(int(hv) & 0xFF)
        nonlinear[OPCODE_IDX[OP_I32_LOAD8_S]]  = float(_sign_extend_8(hv))
        nonlinear[OPCODE_IDX[OP_I32_LOAD16_U]] = float(int(hv) & 0xFFFF)
        nonlinear[OPCODE_IDX[OP_I32_LOAD16_S]] = float(_sign_extend_16(hv))

        top_nonlinear = (opcode_one_hot * nonlinear).sum()
        top = top_linear + top_nonlinear

        sp_delta = (opcode_one_hot * self.sp_deltas).sum()

        return (opcode, arg, int(sp_delta.item()), round(top.item()),
                opcode_one_hot, round(val_a.item()), round(val_b.item()), round(val_c.item()),
                round(local_val.item()), round(heap_val.item()))

    def forward_symbolic(self, prog):
        """Run ``prog`` through the bilinear FF dispatch over :class:`Poly`.

        The numeric :meth:`forward` evaluates the same ``M_ADD / M_SUB /
        B_MUL`` operator tree on float tensors; this method evaluates it
        on :class:`Poly` values. Scope is the branchless polynomial
        fragment (see :data:`ff_symbolic._SCOPE_OPS`); anything else
        raises :class:`ff_symbolic.BlockedOpcodeForSymbolic`.

        Returns a :class:`ff_symbolic.SymbolicForwardResult`.

        The equivalence theorem (issue #69) states: for every collapsed
        catalog program P, ``forward_symbolic(P).top`` is structurally
        equal to :func:`symbolic_executor.run_symbolic` ``(P).top``. The
        proof is that both interpreters apply the same polynomial operator
        tree to inputs from the same source — the symbolic stack.
        """
        return ff_symbolic.evaluate_program(prog)


# ─── PyTorch Executor ────────────────────────────────────────────

class TorchExecutor:
    """Executes programs using CompiledModel.

    Flattened from Phase14PyTorchExecutor.
    """

    def __init__(self, model=None):
        self.model = model or CompiledModel()
        self.model.eval()

    def execute(self, prog, max_steps=50000):
        trace = Trace(program=prog)

        prog_embs = torch.stack([
            embed_program_token(i, instr)
            for i, instr in enumerate(prog)
        ])

        stack_embs_list = []
        local_embs_list = []
        heap_embs_list = []
        call_embs_list = []
        write_count = 0
        local_write_count = 0
        heap_write_count = 0
        call_write_count = 0
        call_stack = []  # list of (ret_addr, saved_sp, saved_locals_base)
        locals_base = 0
        ip = 0
        sp = 0

        with torch.no_grad():
            for step in range(max_steps):
                if ip >= len(prog):
                    break

                query = embed_state(ip, sp)
                stack_embs = (torch.stack(stack_embs_list)
                              if stack_embs_list
                              else torch.zeros(0, D_MODEL, dtype=DTYPE))
                local_embs = (torch.stack(local_embs_list)
                              if local_embs_list
                              else torch.zeros(0, D_MODEL, dtype=DTYPE))
                heap_embs = (torch.stack(heap_embs_list)
                             if heap_embs_list
                             else torch.zeros(0, D_MODEL, dtype=DTYPE))
                call_embs = (torch.stack(call_embs_list)
                             if call_embs_list
                             else torch.zeros(0, D_MODEL, dtype=DTYPE))

                opcode, arg, sp_delta, top, _, val_a, val_b, val_c, local_val, heap_val = \
                    self.model.forward(query, prog_embs, stack_embs, local_embs, heap_embs,
                                       call_embs, locals_base)

                if opcode == OP_HALT:
                    trace.steps.append(TraceStep(opcode, arg, sp, top))
                    break

                # Trap: division by zero
                if opcode in (OP_DIV_S, OP_DIV_U, OP_REM_S, OP_REM_U) and val_a == 0:
                    trace.steps.append(TraceStep(OP_TRAP, 0, sp, 0))
                    break

                # For JZ/JNZ: read condition BEFORE updating SP
                cond_val = None
                if opcode in (OP_JZ, OP_JNZ):
                    cond_val = val_a

                new_sp = sp + sp_delta

                # Stack writes per opcode
                if opcode in (OP_PUSH, OP_DUP, OP_OVER):
                    stack_embs_list.append(
                        embed_stack_entry(new_sp, top, write_count))
                    write_count += 1
                elif opcode in (OP_ADD, OP_SUB, OP_MUL, OP_DIV_S, OP_DIV_U,
                                OP_REM_S, OP_REM_U,
                                OP_EQ, OP_NE,
                                OP_LT_S, OP_LT_U, OP_GT_S, OP_GT_U,
                                OP_LE_S, OP_LE_U, OP_GE_S, OP_GE_U,
                                OP_AND, OP_OR, OP_XOR,
                                OP_SHL, OP_SHR_S, OP_SHR_U,
                                OP_ROTL, OP_ROTR):
                    stack_embs_list.append(
                        embed_stack_entry(new_sp, top, write_count))
                    write_count += 1
                elif opcode in (OP_EQZ, OP_CLZ, OP_CTZ, OP_POPCNT, OP_ABS, OP_NEG):
                    stack_embs_list.append(
                        embed_stack_entry(new_sp, top, write_count))
                    write_count += 1
                elif opcode == OP_SELECT:
                    stack_embs_list.append(
                        embed_stack_entry(new_sp, top, write_count))
                    write_count += 1
                elif opcode == OP_SWAP:
                    stack_embs_list.append(
                        embed_stack_entry(sp, val_b, write_count))
                    write_count += 1
                    stack_embs_list.append(
                        embed_stack_entry(sp - 1, val_a, write_count))
                    write_count += 1
                elif opcode == OP_ROT:
                    stack_embs_list.append(
                        embed_stack_entry(sp, val_c, write_count))
                    write_count += 1
                    stack_embs_list.append(
                        embed_stack_entry(sp - 1, val_a, write_count))
                    write_count += 1
                    stack_embs_list.append(
                        embed_stack_entry(sp - 2, val_b, write_count))
                    write_count += 1
                elif opcode == OP_LOCAL_GET:
                    # Push fetched local value onto stack
                    stack_embs_list.append(
                        embed_stack_entry(new_sp, top, write_count))
                    write_count += 1
                elif opcode == OP_LOCAL_SET:
                    # Pop stack top, write to locals space (with base offset)
                    local_embs_list.append(
                        embed_local_entry(locals_base + arg, val_a, local_write_count))
                    local_write_count += 1
                elif opcode == OP_LOCAL_TEE:
                    # Copy stack top to locals space (no pop, with base offset)
                    local_embs_list.append(
                        embed_local_entry(locals_base + arg, val_a, local_write_count))
                    local_write_count += 1

                # Memory ops: stack writes + heap side effects
                elif opcode == OP_I32_LOAD:
                    # Overwrite stack[SP] with loaded value (sp_delta=0)
                    stack_embs_list.append(
                        embed_stack_entry(new_sp, top, write_count))
                    write_count += 1
                elif opcode in (OP_I32_LOAD8_U, OP_I32_LOAD8_S,
                                OP_I32_LOAD16_U, OP_I32_LOAD16_S):
                    stack_embs_list.append(
                        embed_stack_entry(new_sp, top, write_count))
                    write_count += 1
                elif opcode == OP_I32_STORE:
                    # Write val_a (stack top) to heap at addr val_b (stack[SP-1])
                    heap_embs_list.append(
                        embed_heap_entry(int(val_b), val_a, heap_write_count))
                    heap_write_count += 1
                elif opcode == OP_I32_STORE8:
                    heap_embs_list.append(
                        embed_heap_entry(int(val_b), int(val_a) & 0xFF, heap_write_count))
                    heap_write_count += 1
                elif opcode == OP_I32_STORE16:
                    heap_embs_list.append(
                        embed_heap_entry(int(val_b), int(val_a) & 0xFFFF, heap_write_count))
                    heap_write_count += 1

                # CALL/RETURN: complex state changes handled here
                elif opcode == OP_CALL:
                    call_stack.append((ip + 1, sp, locals_base))
                    call_embs_list.append(
                        embed_call_frame(len(call_stack) - 1, ip + 1, sp, locals_base, call_write_count))
                    call_write_count += 1
                    locals_base = len(local_embs_list)
                    # sp unchanged, top = current stack top
                    trace.steps.append(TraceStep(opcode, arg, sp, top))
                    ip = arg
                    continue
                elif opcode == OP_RETURN:
                    if not call_stack:
                        trace.steps.append(TraceStep(OP_TRAP, 0, sp, 0))
                        break
                    ret_val = val_a  # current stack top
                    ret_addr, saved_sp, saved_locals_base = call_stack.pop()
                    sp = saved_sp + 1
                    stack_embs_list.append(
                        embed_stack_entry(sp, ret_val, write_count))
                    write_count += 1
                    locals_base = saved_locals_base
                    trace.steps.append(TraceStep(opcode, arg, sp, int(ret_val)))
                    ip = ret_addr
                    continue

                trace.steps.append(TraceStep(opcode, arg, new_sp, top))
                sp = new_sp

                # IP update
                if opcode == OP_JZ:
                    ip = arg if cond_val == 0 else ip + 1
                elif opcode == OP_JNZ:
                    ip = arg if cond_val != 0 else ip + 1
                else:
                    ip += 1

        return trace
