"""Tolerance-mode analyst: how much of the compiled ISA is still readable out of a
parameter dict that gradient descent has made dense?

``experiments/blind_recovery/analyst.py`` assumes exact sparse structure -- it reads
nonzero columns and fits residuals to zero tolerance. Ten steps of SGD make every
matrix dense, so that analyst returns nothing. This one:

  1. THRESHOLDS every weight at ``tau`` (absolute, since the compiled lattice is
     {0, +-0.5, +-1, +-2}) and SNAPS survivors to the nearest half-integer. The
     snapped dict is the *recovered machine*.
  2. SCORES the recovered machine against the compiled ground truth structurally,
     with partial credit per opcode (delta 0.25, control 0.25, write coefficients
     0.5) and per head (Q, K, V, b each 0.25).
  3. REPLAYS the recovered machine on the reference ROMs and requires it to
     reproduce every captured memory snapshot exactly, searching the 12 cyclic
     opcode-id -> dispatch-row alignments as the blind analyst did. The replay runs
     on the SNAPPED weights, never the trained ones: a recovery procedure must not
     itself run on the degraded substrate (RESULTS.md of the superposition run).

The threshold at which recovery breaks IS the noise-tolerance measurement, so the
caller sweeps ``tau`` and reports the curve rather than picking a value.
"""
import numpy as np

import machine as M

TAUS = (0.05, 0.1, 0.2, 0.3, 0.45)
SNAP_ROWS = (2, 9, 17)          # steps whose memory is captured, as in compile_artifact


def snap(p, tau):
    """Threshold at tau, snap to the half-integer lattice. n_write is copied."""
    s = {}
    for k in M.HEAD_KEYS + ['W_write']:
        W = np.asarray(p[k], dtype=float)
        s[k] = np.where(np.abs(W) > tau, np.round(W * 2.0) / 2.0, 0.0)
    s['sp_delta'] = np.round(np.asarray(p['sp_delta'], dtype=float))
    s['ctrl'] = (np.asarray(p['ctrl'], dtype=float) > 0.5).astype(float)
    s['n_write'] = np.asarray(p['n_write'], dtype=float)
    return s


def structure_scores(s, truth):
    """Per-opcode and per-head partial credit of the snapped machine vs. truth."""
    per_op = {}
    for i, op in enumerate(M.OPS):
        sc = 0.0
        sc += 0.25 * float(s['sp_delta'][i] == truth['sp_delta'][i])
        sc += 0.25 * float(np.array_equal(s['ctrl'][i], truth['ctrl'][i]))
        n = int(truth['n_write'][i])
        sc += 0.5 * float(np.array_equal(s['W_write'][i, :n], truth['W_write'][i, :n]))
        per_op[op] = sc
    per_head = {}
    for h in M.HEAD_ORDER:
        sc = 0.0
        for m in ('W_Q', 'W_K', 'W_V', 'b_Q'):
            sc += 0.25 * float(np.array_equal(s[f'{h}.{m}'], truth[f'{h}.{m}']))
        per_head[h] = sc
    return per_op, per_head


def reference_captures(truth):
    """ROMs, queries and memory snapshots from the COMPILED machine's own runs.
    These are the activations the blind analyst was handed; they do not change
    across checkpoints because embeddings are fixed functions, not weights."""
    caps = []
    for name, prog, _ in M.ORACLE[:3]:
        tr = []
        M.run(truth, prog, trace=tr)
        snaps = [(tr[s]['q'], tr[s]['mem']) for s in SNAP_ROWS if s < len(tr)]
        caps.append(dict(name=name, prog=prog, rom=M.rom_of(prog), snaps=snaps))
    return caps


def _replay(s, prog, row_of, targets, steps=250):
    """Run the snapped machine with an opcode-id -> row map, checking that every
    target memory snapshot is reproduced exactly. Returns (result, all_seen)."""
    rom = M.rom_of(prog)
    rows, wo, sp, ip = [], 0, 0, 0
    seen = [False] * len(targets)
    for _ in range(steps):
        q = M.embed_state(ip, sp)
        mem = np.stack(rows) if rows else np.zeros((0, M.D))
        for ti, (_, t) in enumerate(targets):
            if not seen[ti] and mem.shape == t.shape and np.allclose(mem, t):
                seen[ti] = True
        opv, _ = M.attend(s, 'prog_op', q, rom)
        arg, _ = M.attend(s, 'prog_arg', q, rom)
        op = int(round(opv))
        if op not in row_of:
            return None, seen
        r = row_of[op]
        vals = []
        for h, off in M.READ_HEADS:
            v, i = M.attend(s, h, q, mem)
            if i < 0:
                vals.append(0.0)
                continue
            hit = round(mem[i, M.STACK_K0] / 2.0) == sp + off
            vals.append(v if hit else 0.0)
        u = np.array([arg] + vals)
        new_sp = int(round(sp + s['sp_delta'][r]))
        if new_sp < 0:
            return None, seen
        for c in range(int(s['n_write'][r])):
            rows.append(M.embed_stack(new_sp - c, float(s['W_write'][r, c] @ u), wo))
            wo += 1
        jz, jnz, halt = s['ctrl'][r] > 0.5
        if halt:
            return vals[0], seen
        ip = int(round(arg)) if ((jz and vals[0] == 0) or (jnz and vals[0] != 0)) else ip + 1
        if not 0 <= ip < len(prog):
            return None, seen
        sp = new_sp
    return None, seen


def replay_alignment(s, caps):
    """Which cyclic opcode-id -> row alignments reproduce every captured snapshot?"""
    survivors = []
    for shift in range(M.N_OPS):
        row_of = {v: (v - 1 + shift) % M.N_OPS for v in range(1, M.N_OPS + 1)}
        good = True
        for cap in caps:
            res, seen = _replay(s, cap['prog'], row_of, cap['snaps'])
            if res is None or not all(seen):
                good = False
                break
        if good:
            survivors.append(shift)
    return survivors


def analyze(p, truth, caps, tau):
    s = snap(p, tau)
    per_op, per_head = structure_scores(s, truth)
    survivors = replay_alignment(s, caps)
    return dict(tau=tau,
                isa_score=float(np.mean(list(per_op.values()))),
                addr_score=float(np.mean(list(per_head.values()))),
                per_op=per_op, per_head=per_head,
                replay_ok=survivors == [0],
                survivors=survivors)


def analyze_sweep(p, truth, caps, taus=TAUS):
    return [analyze(p, truth, caps, tau) for tau in taus]


if __name__ == '__main__':
    truth = M.compile_params()
    caps = reference_captures(truth)
    for r in analyze_sweep(truth, truth, caps):
        print(f"tau={r['tau']:.2f} isa={r['isa_score']:.3f} addr={r['addr_score']:.3f} "
              f"replay={r['replay_ok']} survivors={r['survivors']}")
    rng = np.random.default_rng(0)
    noisy = {k: (v + rng.normal(0, 0.15, v.shape) if k in M.TRAINABLE else v)
             for k, v in truth.items()}
    print('noise sigma=0.15:')
    for r in analyze_sweep(noisy, truth, caps):
        print(f"tau={r['tau']:.2f} isa={r['isa_score']:.3f} addr={r['addr_score']:.3f} "
              f"replay={r['replay_ok']} survivors={r['survivors']}")
