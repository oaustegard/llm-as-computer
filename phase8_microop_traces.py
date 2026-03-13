"""
Phase 8: Micro-Op Trace Decomposition

The DIFF+ADD wall (0% in Phase 7, 3% in Phase 6) exists because the model must
predict a+b in a single token, requiring SIMULTANEOUS retrieval of two values
from different stack positions. This is the hardest thing we ask the model to do.

Insight: decompose multi-operand operations into sequential single-operand steps.
Instead of 4 tokens per step [OP, ARG, SP, TOP], use 6 tokens:

  [OP, ARG, FETCH1, FETCH2, SP, TOP]

For each operation:
  PUSH val:  [1, val,   0,     0,     new_SP, val]
  POP:       [2, 0,     new_top, 0,    new_SP, new_top]    -- single lookup
  DUP:       [4, 0,     dup_val, 0,    new_SP, dup_val]    -- single lookup
  ADD:       [3, 0,     a,       b,    new_SP, a+b]        -- TWO sequential lookups
  HALT:      [5, 0,     top_val, 0,    SP,     top_val]    -- single lookup

Now for ADD, the model predicts:
  1. OP_ADD (opcode dispatch -- already works)
  2. ARG=0 (trivial)
  3. FETCH1=a (single lookup at stack[SP] -- proven to work)
  4. FETCH2=b (single lookup at stack[SP-1] -- proven to work, conditioned on FETCH1)
  5. SP=new_SP (arithmetic on known SP -- trivial)
  6. TOP=a+b (arithmetic on a,b already in context -- should be easy)

Each prediction requires at most ONE stack lookup. The parallel retrieval bottleneck
is gone. If this works, it proves the architectural hypothesis: the model CAN execute,
it just needs the right trace granularity.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import time
import json
import os
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4_stack_machine import (
    program, Instruction, ReferenceExecutor, Trace, TraceStep,
    OP_PUSH, OP_POP, OP_ADD, OP_DUP, OP_HALT, OP_NAMES,
    TOK_PROG_START, TOK_PROG_END, TOK_TRACE_START,
)
from phase5_training import (
    MicroTransformer, TraceDataset,
    encode_token, decode_token,
    VOCAB_SIZE, PAD_TOKEN, MAX_VAL
)
from phase6_curriculum import (
    constrained_random_program,
    CheckpointMeta, save_checkpoint, load_checkpoint,
    STAGES
)

# ─── Micro-Op Trace Format ───────────────────────────────────────

MICROOP_TOKENS_PER_STEP = 6  # [OP, ARG, FETCH1, FETCH2, SP, TOP]


@dataclass
class MicroOpStep:
    """One instruction's execution record with explicit fetch tokens."""
    op: int
    arg: int
    fetch1: int   # first operand fetched from stack (0 if none)
    fetch2: int   # second operand fetched from stack (0 if none)
    sp: int       # stack pointer AFTER execution
    top: int      # top-of-stack value AFTER execution

    def tokens(self) -> List[int]:
        return [self.op, self.arg, self.fetch1, self.fetch2, self.sp, self.top]


@dataclass
class MicroOpTrace:
    """Full execution trace with micro-op format."""
    program_instrs: List[Instruction]
    steps: List[MicroOpStep]

    def to_token_sequence(self) -> List[int]:
        tokens = [TOK_PROG_START]
        for instr in self.program_instrs:
            tokens.extend([instr.op, instr.arg])
        tokens.append(TOK_PROG_END)
        tokens.append(TOK_TRACE_START)
        for step in self.steps:
            tokens.extend(step.tokens())
        return tokens


class MicroOpExecutor:
    """Generates traces in micro-op format (6 tokens per step)."""

    def execute(self, prog: List[Instruction], max_steps: int = 1000) -> MicroOpTrace:
        stack = []
        trace = MicroOpTrace(program_instrs=prog, steps=[])
        ip = 0

        for _ in range(max_steps):
            if ip >= len(prog):
                break

            instr = prog[ip]
            op, arg = instr.op, instr.arg

            if op == OP_PUSH:
                stack.append(arg)
                trace.steps.append(MicroOpStep(
                    op=op, arg=arg,
                    fetch1=0, fetch2=0,
                    sp=len(stack), top=arg
                ))

            elif op == OP_POP:
                stack.pop()
                new_top = stack[-1] if stack else 0
                trace.steps.append(MicroOpStep(
                    op=op, arg=0,
                    fetch1=new_top, fetch2=0,  # fetch the new top
                    sp=len(stack), top=new_top
                ))

            elif op == OP_ADD:
                b = stack.pop()
                a = stack.pop()
                result = a + b
                stack.append(result)
                trace.steps.append(MicroOpStep(
                    op=op, arg=0,
                    fetch1=b, fetch2=a,  # top-of-stack first, then second
                    sp=len(stack), top=result
                ))

            elif op == OP_DUP:
                val = stack[-1]
                stack.append(val)
                trace.steps.append(MicroOpStep(
                    op=op, arg=0,
                    fetch1=val, fetch2=0,  # fetch the value being duplicated
                    sp=len(stack), top=val
                ))

            elif op == OP_HALT:
                top = stack[-1] if stack else 0
                trace.steps.append(MicroOpStep(
                    op=op, arg=0,
                    fetch1=top, fetch2=0,
                    sp=len(stack), top=top
                ))
                break

            ip += 1

        return trace


# ─── Data Generation ─────────────────────────────────────────────

def generate_microop_data(
    allowed_ops: Set[int],
    n_samples: int,
    min_len: int = 3,
    max_len: int = 8,
    max_push_val: int = 30
) -> List[List[int]]:
    """Generate encoded micro-op trace sequences."""
    executor = MicroOpExecutor()
    seqs = []
    attempts = 0
    max_attempts = n_samples * 5

    while len(seqs) < n_samples and attempts < max_attempts:
        attempts += 1
        prog = constrained_random_program(allowed_ops, min_len, max_len, max_push_val)
        try:
            trace = executor.execute(prog)
            tokens = trace.to_token_sequence()
            encoded = [encode_token(t) for t in tokens]
            if all(0 <= t < VOCAB_SIZE for t in encoded):
                seqs.append(encoded)
        except Exception:
            continue

    return seqs


# ─── Evaluation for Micro-Op Traces ─────────────────────────────

def evaluate_microop_execution(
    model: MicroTransformer,
    test_progs: List[List[Instruction]],
    verbose: bool = False
) -> Dict:
    """Evaluate model on micro-op trace format."""
    executor = MicroOpExecutor()
    model.eval()

    results = {
        'total': len(test_progs),
        'perfect': 0,
        'final_correct': 0,
        'token_errors': [],
        'examples': [],
    }

    with torch.no_grad():
        for prog in test_progs:
            ref_trace = executor.execute(prog)
            ref_tokens = ref_trace.to_token_sequence()
            ref_encoded = [encode_token(t) for t in ref_tokens]

            trace_start_tok = encode_token(TOK_TRACE_START)
            try:
                trace_start_idx = ref_encoded.index(trace_start_tok)
            except ValueError:
                continue

            prefix = ref_encoded[:trace_start_idx + 1]
            generated = list(prefix)
            n_trace_tokens = len(ref_encoded) - len(prefix)

            for _ in range(n_trace_tokens + 6):
                inp = torch.tensor([generated], dtype=torch.long)
                logits = model(inp)
                next_tok = logits[0, -1].argmax().item()
                generated.append(next_tok)
                if len(generated) >= len(ref_encoded):
                    break

            gen_trace = generated[trace_start_idx + 1:]
            ref_trace_tokens = ref_encoded[trace_start_idx + 1:]

            min_len_t = min(len(gen_trace), len(ref_trace_tokens))
            gen_trace = gen_trace[:min_len_t]
            ref_trace_tokens = ref_trace_tokens[:min_len_t]

            errors = sum(1 for g, r in zip(gen_trace, ref_trace_tokens) if g != r)
            results['token_errors'].append(errors)

            if errors == 0:
                results['perfect'] += 1

            # Check final TOP value (last token in trace)
            if min_len_t >= MICROOP_TOKENS_PER_STEP:
                # Last step's TOP is at position -1 (last token of last 6-token step)
                ref_final = ref_trace_tokens[-1]
                gen_final = gen_trace[-1] if len(gen_trace) >= len(ref_trace_tokens) else -1
                if gen_final == ref_final:
                    results['final_correct'] += 1

            # Store examples
            if len(results['examples']) < 5:
                prog_str = ' ; '.join(str(i) for i in prog)
                results['examples'].append({
                    'program': prog_str,
                    'ref_trace': ref_trace_tokens[:18],
                    'gen_trace': gen_trace[:18],
                    'errors': errors,
                })

    return results


# ─── ADD Diagnostic (Micro-Op) ───────────────────────────────────

def run_add_diagnostic_microop(
    model: MicroTransformer,
    n_tests: int = 30,
    verbose: bool = True
) -> Dict:
    """Test the three ADD patterns with micro-op trace format."""
    results = {}

    patterns = {
        'DUP+ADD (PUSH a, DUP, ADD)': [],
        'SAME+ADD (PUSH a, PUSH a, ADD)': [],
        'DIFF+ADD (PUSH a, PUSH b, ADD)': [],
    }

    random.seed(99)

    for _ in range(n_tests):
        a = random.randint(1, 25)
        b = random.randint(1, 25)
        while b == a:
            b = random.randint(1, 25)

        patterns['DUP+ADD (PUSH a, DUP, ADD)'].append(
            [Instruction(OP_PUSH, a), Instruction(OP_DUP, 0),
             Instruction(OP_ADD, 0), Instruction(OP_HALT, 0)])

        patterns['SAME+ADD (PUSH a, PUSH a, ADD)'].append(
            [Instruction(OP_PUSH, a), Instruction(OP_PUSH, a),
             Instruction(OP_ADD, 0), Instruction(OP_HALT, 0)])

        patterns['DIFF+ADD (PUSH a, PUSH b, ADD)'].append(
            [Instruction(OP_PUSH, a), Instruction(OP_PUSH, b),
             Instruction(OP_ADD, 0), Instruction(OP_HALT, 0)])

    if verbose:
        print("\n  ADD Diagnostic (micro-op traces):")
        print(f"  {'Pattern':<45} {'Perfect':>10} {'Final OK':>10}")
        print("  " + "-" * 65)

    for name, progs in patterns.items():
        res = evaluate_microop_execution(model, progs)
        results[name] = {
            'perfect': res['perfect'],
            'total': res['total'],
            'final_correct': res['final_correct'],
            'pct_perfect': res['perfect'] / res['total'] * 100,
        }
        if verbose:
            pf = f"{res['perfect']}/{res['total']}"
            fc = f"{res['final_correct']}/{res['total']}"
            print(f"  {name:<45} {pf:>10} {fc:>10}")

    return results


# ─── Training (reuse Phase 7's infrastructure) ──────────────────

def train_stage(
    model: MicroTransformer,
    train_data: TraceDataset,
    val_data: TraceDataset,
    stage: int,
    max_epochs: int = 80,
    lr: float = 3e-4,
    batch_size: int = 64,
    patience: int = 20,
    max_wall_time: float = 500.0,
    checkpoint_prefix: str = "phase8",
    checkpoint_dir: str = ".",
    resume: bool = True,
    verbose: bool = True
) -> CheckpointMeta:
    """Train one curriculum stage with wall-clock safety and checkpointing."""

    ckpt_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_ckpt_stage{stage}.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    start_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state = None
    total_wall = 0.0

    if resume and os.path.exists(ckpt_path):
        meta = load_checkpoint(ckpt_path, model, optimizer)
        start_epoch = meta['epoch'] + 1
        history = meta['history']
        best_val_loss = meta['best_val_loss']
        best_val_acc = meta['best_val_acc']
        total_wall = meta['wall_time_s']
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max_epochs, last_epoch=start_epoch - 1 if start_epoch > 0 else -1
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    no_improve = 0
    wall_start = time.time()

    for epoch in range(start_epoch, max_epochs):
        elapsed = time.time() - wall_start
        if elapsed > max_wall_time:
            if verbose:
                print(f"  Wall-clock limit ({max_wall_time:.0f}s) reached at epoch {epoch}")
            break

        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch, lengths in train_loader:
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            logits = model(inp)

            mask = torch.zeros_like(tgt, dtype=torch.bool)
            for i, l in enumerate(lengths):
                mask[i, :l-1] = True

            loss = F.cross_entropy(logits[mask], tgt[mask])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        n_val = 0
        with torch.no_grad():
            for batch, lengths in val_loader:
                inp = batch[:, :-1]
                tgt = batch[:, 1:]
                logits = model(inp)

                mask = torch.zeros_like(tgt, dtype=torch.bool)
                for i, l in enumerate(lengths):
                    mask[i, :l-1] = True

                loss = F.cross_entropy(logits[mask], tgt[mask])
                val_loss += loss.item()
                n_val += 1

                preds = logits.argmax(dim=-1)
                val_correct += (preds[mask] == tgt[mask]).sum().item()
                val_total += mask.sum().item()

        val_loss /= max(n_val, 1)
        val_acc = val_correct / max(val_total, 1)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if verbose and (epoch % 10 == 0 or epoch == max_epochs - 1):
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                  f"[{time.time()-wall_start:.0f}s]")

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}")
                break

    total_wall += time.time() - wall_start
    final_epoch = len(history['val_acc']) - 1

    if best_state:
        model.load_state_dict(best_state)

    meta = CheckpointMeta(
        stage=stage,
        epoch=final_epoch,
        best_val_acc=best_val_acc,
        best_val_loss=best_val_loss,
        total_epochs_trained=len(history['val_acc']),
        wall_time_s=total_wall,
        history=history,
    )
    save_checkpoint(model, optimizer, meta, ckpt_path)
    if verbose:
        print(f"  Checkpoint saved: stage={stage}, epochs={meta.total_epochs_trained}, "
              f"best_acc={best_val_acc:.4f}, wall={total_wall:.1f}s")

    return meta


# ─── Stage Runner ────────────────────────────────────────────────

def run_stage(
    stage: int,
    model: MicroTransformer,
    n_train: int = 5000,
    n_val: int = 500,
    n_test: int = 50,
    checkpoint_prefix: str = "phase8",
    checkpoint_dir: str = ".",
    verbose: bool = True,
) -> Dict:
    """Run a single curriculum stage with micro-op traces."""

    cfg = STAGES[stage]
    if verbose:
        print(f"\n{'='*60}")
        print(f"Stage {stage}: {cfg['name']} (micro-op traces)")
        print(f"  {cfg['description']}")
        print(f"  Target: >{cfg['target_acc']*100:.0f}% token accuracy")
        print(f"{'='*60}\n")

    random.seed(42 + stage)
    np.random.seed(42 + stage)
    torch.manual_seed(42 + stage)

    gen_ops = cfg['ops'] - {OP_HALT}

    if verbose:
        print("  Generating micro-op data...")
    train_seqs = generate_microop_data(gen_ops, n_train, max_push_val=cfg['max_push_val'])
    val_seqs = generate_microop_data(gen_ops, n_val, max_push_val=cfg['max_push_val'])

    max_seq_len = max(
        max(len(s) for s in train_seqs),
        max(len(s) for s in val_seqs)
    )

    if verbose:
        print(f"  {len(train_seqs)} train, {len(val_seqs)} val, max_len={max_seq_len}")

    train_data = TraceDataset(train_seqs, max_len=max_seq_len)
    val_data = TraceDataset(val_seqs, max_len=max_seq_len)

    meta = train_stage(
        model, train_data, val_data,
        stage=stage,
        max_epochs=cfg['max_epochs'],
        patience=cfg['patience'],
        checkpoint_prefix=checkpoint_prefix,
        checkpoint_dir=checkpoint_dir,
        verbose=verbose,
    )

    if verbose:
        print(f"\n  Execution evaluation...")

    test_progs = [
        constrained_random_program(gen_ops, max_push_val=cfg['max_push_val'])
        for _ in range(n_test)
    ]
    exec_results = evaluate_microop_execution(model, test_progs, verbose=verbose)

    if verbose:
        print(f"  Token accuracy (validation): {meta.best_val_acc:.4f}")
        print(f"  Perfect traces: {exec_results['perfect']}/{exec_results['total']}")
        print(f"  Final value correct: {exec_results['final_correct']}/{exec_results['total']}")
        target_met = "YES" if meta.best_val_acc >= cfg['target_acc'] else "NO"
        print(f"  Target met (>{cfg['target_acc']*100:.0f}%): {target_met}")

        if exec_results.get('examples'):
            print(f"\n  Sample traces:")
            for ex in exec_results['examples'][:3]:
                print(f"    {ex['program']}")
                print(f"      ref: {ex['ref_trace'][:18]}...")
                print(f"      gen: {ex['gen_trace'][:18]}...")
                print(f"      errors: {ex['errors']}")

    return {
        'stage': stage,
        'name': cfg['name'],
        'meta': meta.to_dict(),
        'execution': {
            'total': exec_results['total'],
            'perfect': exec_results['perfect'],
            'final_correct': exec_results['final_correct'],
            'avg_token_errors': float(np.mean(exec_results['token_errors'])) if exec_results['token_errors'] else 0,
        },
    }


# ─── Main: Micro-Op Curriculum ──────────────────────────────────

def run_microop_curriculum(checkpoint_dir: str = ".") -> Dict:
    """Run full curriculum with micro-op trace format.

    Tests both Percepta architecture (d=36, h=18, L=7) and Phase 6 config (d=64, h=4, L=2).
    """

    print("=" * 60)
    print("Phase 8: Micro-Op Trace Decomposition")
    print("=" * 60)
    print()
    print("Hypothesis: decomposing two-operand retrieval into sequential")
    print("single-operand predictions breaks the DIFF+ADD wall.")
    print()
    print("Trace format: [OP, ARG, FETCH1, FETCH2, SP, TOP]")
    print("  ADD: FETCH1=stack[SP], FETCH2=stack[SP-1], TOP=FETCH1+FETCH2")
    print("  Each prediction requires at most ONE stack lookup.")
    print()

    # Use Phase 6 config (wider model, proven to learn structure)
    # Percepta arch (d=36, h=18, L=7) was actually WORSE in Phase 7
    cfg = {
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'max_len': 200,
        'dropout': 0.1,
    }

    model = MicroTransformer(
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'],
        d_ff=cfg['d_ff'],
        max_len=cfg['max_len'],
        dropout=cfg['dropout'],
    )
    print(f"Model config: d_model={cfg['d_model']}, n_heads={cfg['n_heads']}, "
          f"n_layers={cfg['n_layers']}, d_ff={cfg['d_ff']}")
    print(f"Parameters: {model.n_params:,}")
    print()

    all_results = {'config': cfg, 'config_name': 'phase6_wide'}
    total_start = time.time()

    for stage in [1, 2, 3]:
        if stage > 1:
            prev_ckpt = os.path.join(checkpoint_dir, f"phase8_ckpt_stage{stage-1}.pt")
            if os.path.exists(prev_ckpt):
                print(f"\n  Loading Stage {stage-1} weights as initialization...")
                load_checkpoint(prev_ckpt, model)
            else:
                print(f"\n  WARNING: No Stage {stage-1} checkpoint found.")

        results = run_stage(
            stage, model,
            n_train=5000, n_val=500, n_test=50,
            checkpoint_prefix="phase8",
            checkpoint_dir=checkpoint_dir,
        )
        all_results[f'stage_{stage}'] = results

        results_path = os.path.join(checkpoint_dir, "phase8_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved to {results_path}")

    # ADD diagnostic
    print(f"\n{'='*60}")
    print("ADD Diagnostic (micro-op traces)")
    print(f"{'='*60}")
    add_results = run_add_diagnostic_microop(model, n_tests=30, verbose=True)
    all_results['add_diagnostic'] = add_results

    total_time = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print("Phase 8 Summary: Micro-Op Trace Decomposition")
    print(f"{'='*60}")
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\n{'Stage':<30} {'Val Acc':>8} {'Perfect':>10} {'Final OK':>10}")
    print("-" * 60)
    for key in ['stage_1', 'stage_2', 'stage_3']:
        if key in all_results:
            res = all_results[key]
            name = res['name']
            acc = res['meta']['best_val_acc']
            pf = f"{res['execution']['perfect']}/{res['execution']['total']}"
            fc = f"{res['execution']['final_correct']}/{res['execution']['total']}"
            print(f"{name:<30} {acc:>8.4f} {pf:>10} {fc:>10}")

    print(f"\nBaselines (4-token traces, same model config):")
    print(f"  Phase 6: 85% val acc, 39/50 perfect, 44/50 final correct")
    print(f"  Phase 6 ADD: DUP+ADD 97%, SAME+ADD 57%, DIFF+ADD 3%")
    print(f"  Phase 7 ADD: DUP+ADD 63%, SAME+ADD 57%, DIFF+ADD 0%")

    if 'add_diagnostic' in all_results:
        diff_add = all_results['add_diagnostic'].get('DIFF+ADD (PUSH a, PUSH b, ADD)', {})
        diff_pct = diff_add.get('pct_perfect', 0)
        dup_add = all_results['add_diagnostic'].get('DUP+ADD (PUSH a, DUP, ADD)', {})
        dup_pct = dup_add.get('pct_perfect', 0)
        same_add = all_results['add_diagnostic'].get('SAME+ADD (PUSH a, PUSH a, ADD)', {})
        same_pct = same_add.get('pct_perfect', 0)

        print(f"\nPhase 8 ADD diagnostic (micro-op traces):")
        print(f"  DUP+ADD:  {dup_pct:.0f}%")
        print(f"  SAME+ADD: {same_pct:.0f}%")
        print(f"  DIFF+ADD: {diff_pct:.0f}%")

        if diff_pct > 10:
            print(f"\n  >>> MICRO-OP DECOMPOSITION BREAKS THE DIFF+ADD WALL! <<<")
            print(f"  Sequential single-lookups overcome parallel retrieval bottleneck.")
        elif diff_pct > 3:
            print(f"\n  Micro-ops show improvement on DIFF+ADD ({diff_pct:.0f}% vs 0-3%).")
        else:
            print(f"\n  DIFF+ADD wall persists. Bottleneck may not be retrieval parallelism.")

    all_results['total_time_s'] = total_time
    results_path = os.path.join(checkpoint_dir, "phase8_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFinal results saved to {results_path}")

    return all_results


# ─── Sanity Check ────────────────────────────────────────────────

def sanity_check():
    """Verify micro-op executor matches reference executor on final values."""
    ref = ReferenceExecutor()
    micro = MicroOpExecutor()

    tests = [
        [Instruction(OP_PUSH, 3), Instruction(OP_PUSH, 5), Instruction(OP_ADD, 0), Instruction(OP_HALT, 0)],
        [Instruction(OP_PUSH, 7), Instruction(OP_DUP, 0), Instruction(OP_ADD, 0), Instruction(OP_HALT, 0)],
        [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 20), Instruction(OP_POP, 0), Instruction(OP_HALT, 0)],
        [Instruction(OP_PUSH, 1), Instruction(OP_PUSH, 2), Instruction(OP_PUSH, 3),
         Instruction(OP_ADD, 0), Instruction(OP_ADD, 0), Instruction(OP_HALT, 0)],
    ]

    print("Sanity check: micro-op vs reference executor")
    all_pass = True
    for prog in tests:
        ref_trace = ref.execute(prog)
        micro_trace = micro.execute(prog)

        ref_top = ref_trace.steps[-1].top
        micro_top = micro_trace.steps[-1].top

        prog_str = ' ; '.join(str(i) for i in prog)
        ok = ref_top == micro_top
        if not ok:
            all_pass = False
        print(f"  {'OK' if ok else 'FAIL'}: {prog_str} -> ref={ref_top}, micro={micro_top}")

        # Show micro-op trace
        for step in micro_trace.steps:
            op_name = OP_NAMES.get(step.op, "?")
            print(f"    {op_name:5s} arg={step.arg} f1={step.fetch1} f2={step.fetch2} sp={step.sp} top={step.top}")

    print(f"\n  All pass: {all_pass}")
    return all_pass


# ─── Entry Point ─────────────────────────────────────────────────

if __name__ == "__main__":
    if sanity_check():
        print()
        results = run_microop_curriculum(checkpoint_dir=".")
