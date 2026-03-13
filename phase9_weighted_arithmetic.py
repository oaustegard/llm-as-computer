"""
Phase 9: Weighted Arithmetic Loss

Phase 8 proved retrieval is solved and arithmetic is the sole bottleneck.
The model gets FETCH1/FETCH2 correct 100% of the time for DIFF+ADD but
can't compute TOP = FETCH1 + FETCH2.

Root cause: addition is ~15% of tokens, so the FF layers receive too little
gradient signal to learn f(a,b) = a+b. The model can learn addition in
isolation (98% in 500 epochs) but not within execution traces.

Fix: upweight the cross-entropy loss on arithmetic-critical tokens.
For ADD steps in micro-op traces [OP, ARG, F1, F2, SP, TOP]:
  - F1 (operand retrieval): moderate weight (already works, but reinforce)
  - F2 (operand retrieval): moderate weight
  - TOP (the addition result): HIGH weight — this is the bottleneck

This is the simplest possible intervention that directly targets the
diagnosed problem. If it works, the bottleneck was purely gradient
allocation. If it fails, the issue is deeper than signal strength.
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
from typing import List, Dict, Set
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4_stack_machine import (
    Instruction, OP_PUSH, OP_POP, OP_ADD, OP_DUP, OP_HALT, OP_NAMES,
    TOK_PROG_START, TOK_PROG_END, TOK_TRACE_START,
)
from phase5_training import (
    MicroTransformer, TraceDataset,
    encode_token, decode_token,
    VOCAB_SIZE, PAD_TOKEN, MAX_VAL
)
from phase6_curriculum import (
    constrained_random_program, CheckpointMeta, save_checkpoint, load_checkpoint, STAGES
)
from phase8_microop_traces import (
    MicroOpExecutor, MicroOpTrace, MicroOpStep,
    MICROOP_TOKENS_PER_STEP,
    generate_microop_data, evaluate_microop_execution,
    run_add_diagnostic_microop,
)


# ─── Weighted Loss Computation ───────────────────────────────────

def compute_arithmetic_weights(
    batch: torch.Tensor,
    lengths: torch.Tensor,
    add_top_weight: float = 20.0,
    add_fetch_weight: float = 5.0,
) -> torch.Tensor:
    """Compute per-token loss weights that upweight ADD arithmetic positions.

    For each sequence in the batch, find ADD opcode tokens in the trace,
    then upweight the F1, F2, and TOP positions of that ADD step.

    In micro-op format (6 tokens/step): [OP, ARG, F1, F2, SP, TOP]
    The target for prediction is shifted by 1, so:
      - When we predict token at position P, we use input[:P] and target[P-1]
      - ADD OP at position P means F1 is at P+2, F2 at P+3, TOP at P+5
      - In the target tensor (shifted by 1): F1 at P+1, F2 at P+2, TOP at P+4

    Returns weight tensor same shape as target (batch[:, 1:]).
    """
    B, L = batch.shape
    # Target is batch[:, 1:], so target position idx corresponds to batch position idx+1
    weights = torch.ones(B, L - 1, dtype=torch.float32)

    # Encoded ADD opcode
    add_encoded = encode_token(OP_ADD)
    trace_start_encoded = encode_token(TOK_TRACE_START)

    for b in range(B):
        seq = batch[b].tolist()
        seq_len = int(lengths[b].item())

        # Find TRACE_START
        try:
            ts_idx = seq[:seq_len].index(trace_start_encoded)
        except ValueError:
            continue

        # Scan trace for ADD opcodes (every 6 tokens from ts_idx+1)
        trace_start = ts_idx + 1
        pos = trace_start
        while pos + 5 < seq_len:
            if seq[pos] == add_encoded:
                # This is an ADD step starting at position `pos` in the sequence
                # In the target tensor (offset by 1):
                #   F1 target is at target index pos+1  (predicting batch[pos+2] from batch[:pos+2])
                #   F2 target is at target index pos+2
                #   TOP target is at target index pos+4
                for offset, w in [(1, add_fetch_weight), (2, add_fetch_weight), (4, add_top_weight)]:
                    target_idx = pos + offset
                    if target_idx < L - 1:
                        weights[b, target_idx] = w
            pos += MICROOP_TOKENS_PER_STEP

    return weights


# ─── Training with Weighted Loss ─────────────────────────────────

def train_stage_weighted(
    model: MicroTransformer,
    train_data: TraceDataset,
    val_data: TraceDataset,
    stage: int,
    add_top_weight: float = 20.0,
    add_fetch_weight: float = 5.0,
    max_epochs: int = 80,
    lr: float = 3e-4,
    batch_size: int = 64,
    patience: int = 25,
    max_wall_time: float = 500.0,
    checkpoint_prefix: str = "phase9",
    checkpoint_dir: str = ".",
    resume: bool = True,
    verbose: bool = True
) -> CheckpointMeta:
    """Train with weighted loss that upweights arithmetic tokens."""

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

            # Standard validity mask (ignore padding)
            mask = torch.zeros_like(tgt, dtype=torch.bool)
            for i, l in enumerate(lengths):
                mask[i, :l-1] = True

            # Compute arithmetic weights (only for stages with ADD)
            if stage >= 3:
                arith_weights = compute_arithmetic_weights(
                    batch, lengths,
                    add_top_weight=add_top_weight,
                    add_fetch_weight=add_fetch_weight,
                )
            else:
                arith_weights = torch.ones_like(tgt, dtype=torch.float32)

            # Weighted cross-entropy: per-token loss * weight, then mean
            per_token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                reduction='none'
            ).reshape(tgt.shape)

            weighted_loss = (per_token_loss * arith_weights * mask.float()).sum() / mask.sum()

            optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += weighted_loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= max(n_batches, 1)

        # Validate (unweighted, standard metrics)
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
    add_top_weight: float = 20.0,
    add_fetch_weight: float = 5.0,
    n_train: int = 5000,
    n_val: int = 500,
    n_test: int = 50,
    checkpoint_prefix: str = "phase9",
    checkpoint_dir: str = ".",
    verbose: bool = True,
) -> Dict:
    """Run a single curriculum stage with weighted arithmetic loss."""

    cfg = STAGES[stage]
    if verbose:
        print(f"\n{'='*60}")
        print(f"Stage {stage}: {cfg['name']} (weighted arithmetic loss)")
        if stage >= 3:
            print(f"  ADD TOP weight: {add_top_weight}x, FETCH weight: {add_fetch_weight}x")
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

    meta = train_stage_weighted(
        model, train_data, val_data,
        stage=stage,
        add_top_weight=add_top_weight,
        add_fetch_weight=add_fetch_weight,
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


# ─── Weight Sweep ────────────────────────────────────────────────

def run_weight_sweep(checkpoint_dir: str = ".") -> Dict:
    """Test multiple arithmetic weight configurations.

    Sweep: TOP weight in [10, 20, 50], FETCH weight in [3, 5].
    Only train Stage 3 (full instruction set) since that's where ADD lives.
    Use Stage 2 checkpoint from Phase 8 as initialization.
    """

    print("=" * 60)
    print("Phase 9: Weighted Arithmetic Loss Sweep")
    print("=" * 60)
    print()
    print("Hypothesis: upweighting ADD's TOP token in the loss gives the")
    print("FF layers enough gradient signal to learn integer addition.")
    print()

    configs = [
        {'add_top_weight': 10.0, 'add_fetch_weight': 3.0, 'name': 'w10_f3'},
        {'add_top_weight': 20.0, 'add_fetch_weight': 5.0, 'name': 'w20_f5'},
        {'add_top_weight': 50.0, 'add_fetch_weight': 5.0, 'name': 'w50_f5'},
    ]

    all_results = {'configs': [c.copy() for c in configs]}
    total_start = time.time()

    for cfg_idx, wcfg in enumerate(configs):
        name = wcfg['name']
        print(f"\n{'='*60}")
        print(f"Config {cfg_idx+1}/{len(configs)}: {name}")
        print(f"  TOP weight={wcfg['add_top_weight']}, FETCH weight={wcfg['add_fetch_weight']}")
        print(f"{'='*60}")

        # Fresh model for each config
        model = MicroTransformer(
            d_model=64, n_heads=4, n_layers=2, d_ff=256,
            max_len=200, dropout=0.1,
        )
        if cfg_idx == 0:
            print(f"  Model: d=64, h=4, L=2, params={model.n_params:,}")

        # Train stages 1-2 with uniform weights (no ADD in these stages)
        for stage in [1, 2]:
            stage_cfg = STAGES[stage]
            gen_ops = stage_cfg['ops'] - {OP_HALT}

            random.seed(42 + stage)
            np.random.seed(42 + stage)
            torch.manual_seed(42 + stage)

            train_seqs = generate_microop_data(gen_ops, 5000, max_push_val=stage_cfg['max_push_val'])
            val_seqs = generate_microop_data(gen_ops, 500, max_push_val=stage_cfg['max_push_val'])
            max_seq_len = max(max(len(s) for s in train_seqs), max(len(s) for s in val_seqs))

            train_data = TraceDataset(train_seqs, max_len=max_seq_len)
            val_data = TraceDataset(val_seqs, max_len=max_seq_len)

            prefix = f"phase9_{name}"
            meta = train_stage_weighted(
                model, train_data, val_data,
                stage=stage,
                add_top_weight=1.0,  # uniform for non-ADD stages
                add_fetch_weight=1.0,
                max_epochs=stage_cfg['max_epochs'],
                patience=stage_cfg['patience'],
                checkpoint_prefix=prefix,
                checkpoint_dir=checkpoint_dir,
                verbose=True,
            )

            if stage == 1:
                # Load stage 1 for stage 2
                ckpt_path = os.path.join(checkpoint_dir, f"{prefix}_ckpt_stage1.pt")
                if os.path.exists(ckpt_path):
                    load_checkpoint(ckpt_path, model)

        # Load stage 2 weights
        ckpt_path = os.path.join(checkpoint_dir, f"phase9_{name}_ckpt_stage2.pt")
        if os.path.exists(ckpt_path):
            print(f"\n  Loading Stage 2 weights...")
            load_checkpoint(ckpt_path, model)

        # Stage 3 with weighted loss
        results = run_stage(
            stage=3,
            model=model,
            add_top_weight=wcfg['add_top_weight'],
            add_fetch_weight=wcfg['add_fetch_weight'],
            n_train=5000,
            n_val=500,
            n_test=50,
            checkpoint_prefix=f"phase9_{name}",
            checkpoint_dir=checkpoint_dir,
        )

        # ADD diagnostic
        print(f"\n  ADD Diagnostic for {name}:")
        add_results = run_add_diagnostic_microop(model, n_tests=30, verbose=True)

        all_results[name] = {
            'stage_3': results,
            'add_diagnostic': add_results,
        }

        # Save incremental results
        results_path = os.path.join(checkpoint_dir, "phase9_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    total_time = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print("Phase 9 Summary: Weighted Arithmetic Loss")
    print(f"{'='*60}")
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"\n{'Config':<15} {'Val Acc':>8} {'Perfect':>8} {'DUP+ADD':>10} {'SAME+ADD':>10} {'DIFF+ADD':>10}")
    print("-" * 65)

    for wcfg in configs:
        name = wcfg['name']
        if name in all_results:
            res = all_results[name]
            s3 = res['stage_3']
            acc = s3['meta']['best_val_acc']
            pf = f"{s3['execution']['perfect']}/50"
            add_diag = res['add_diagnostic']
            dup = f"{add_diag.get('DUP+ADD (PUSH a, DUP, ADD)', {}).get('pct_perfect', 0):.0f}%"
            same = f"{add_diag.get('SAME+ADD (PUSH a, PUSH a, ADD)', {}).get('pct_perfect', 0):.0f}%"
            diff = f"{add_diag.get('DIFF+ADD (PUSH a, PUSH b, ADD)', {}).get('pct_perfect', 0):.0f}%"
            print(f"{name:<15} {acc:>8.4f} {pf:>8} {dup:>10} {same:>10} {diff:>10}")

    print(f"\nPhase 8 baseline (uniform loss): 88.4% val acc, DUP+ADD 83%, SAME+ADD 83%, DIFF+ADD 0%")

    best_diff = 0
    best_name = None
    for wcfg in configs:
        name = wcfg['name']
        if name in all_results:
            diff_pct = all_results[name]['add_diagnostic'].get(
                'DIFF+ADD (PUSH a, PUSH b, ADD)', {}).get('pct_perfect', 0)
            if diff_pct > best_diff:
                best_diff = diff_pct
                best_name = name

    if best_diff > 10:
        print(f"\n>>> WEIGHTED LOSS BREAKS THE WALL! Best: {best_name} with {best_diff:.0f}% DIFF+ADD")
    elif best_diff > 0:
        print(f"\n>>> Marginal improvement: {best_name} with {best_diff:.0f}% DIFF+ADD")
    else:
        print(f"\n>>> Weighted loss alone insufficient. Bottleneck is deeper than gradient signal.")

    all_results['total_time_s'] = total_time
    results_path = os.path.join(checkpoint_dir, "phase9_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return all_results


# ─── Entry Point ─────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_weight_sweep(checkpoint_dir=".")
