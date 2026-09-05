"""BLIND discovery on the decayed checkpoints.

``analyst_tol.py`` is a STRUCTURAL verifier: it snaps the trained weights to the
half-integer lattice and compares them to the compiled ground truth it already
holds. It answers "is this still the machine I compiled?".

``experiments/blind_recovery/analyst.py`` is a DISCOVERER: handed an anonymized
artifact (d_model axis permuted, head order shuffled, no names, no ISA) plus a
few reference activations, it works out the head geometry, the parabolic
addressing law, the opcode block and the dispatch table from scratch. It assumes
EXACT sparsity and returns nothing once gradient descent has made the matrices
dense.

This module ports that discoverer to tolerance mode and runs it on every
checkpoint of the decay arms:

  * ``sel`` keeps entries with |w| > tau and snaps survivors to the nearest
    half-integer; every stage downstream reads the snapped matrices.
  * the addressing-law fit reports a max residual and an R^2 instead of
    asserting integrality.
  * head-geometry, region-split, read-port and opcode-block detection each
    return None when their preconditions fail. None is the only failure value;
    no stage falls back to a default.
  * the alignment search replays the SNAPPED weights, never the trained ones. A
    recovery must not run on the degraded substrate.

The artifact is rebuilt per checkpoint exactly as ``compile_artifact.py`` builds
it -- same permutation seed 20260811, same head shuffle, same three ROMs, same
capture steps -- with the checkpoint's weights in place of the compiled ones.
The activations stay those of the COMPILED machine: embeddings are fixed
functions rather than parameters, and the August analyst was handed
compiled-machine activations.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import machine as M  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, 'ckpt')
ARMS = ['rival', 'neutral_adam', 'aux', 'aux_preserve_1e-5']
TAUS = (0.05, 0.1, 0.2, 0.3, 0.45)
CAPTURE_STEPS = (2, 9, 17)
PERM_SEED = 20260811
EPSQ = 1e-6
N_ARTIFACT_PROGS = 3


# ---------------- the anonymized artifact ------------------------------------
def anonymization(seed=PERM_SEED):
    """The permutation of the d_model axis and the head shuffle, drawn in the same
    order as ``compile_artifact.py`` so the artifact is byte-identical there."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M.D)
    horder = rng.permutation(len(M.HEAD_ORDER))
    return perm, horder


def compiled_activations(truth):
    """ROMs, queries and stack snapshots from the compiled machine's own runs on
    the three artifact programs, captured at steps 2, 9 and 17."""
    acts = []
    for _, prog, _ in M.ORACLE[:N_ARTIFACT_PROGS]:
        tr = []
        M.run(truth, prog, trace=tr)
        snaps = [(tr[s]['q'], tr[s]['mem']) for s in CAPTURE_STEPS if s < len(tr)]
        acts.append(dict(rom=M.rom_of(prog), snaps=snaps))
    return acts


def build_artifact(p, acts, perm=None, horder=None):
    """Anonymized tensors for parameter dict ``p``, activations from ``acts``."""
    if perm is None or horder is None:
        perm, horder = anonymization()
    P = lambda W: np.asarray(W, dtype=float)[..., perm]  # noqa: E731
    out = {}
    for new_i, old_i in enumerate(horder):
        h = M.HEAD_ORDER[old_i]
        out[f'head{new_i:02d}_Wq'] = P(p[f'{h}.W_Q'])
        out[f'head{new_i:02d}_Wk'] = P(p[f'{h}.W_K'])
        out[f'head{new_i:02d}_Wv'] = P(p[f'{h}.W_V'])
        out[f'head{new_i:02d}_bq'] = np.asarray(p[f'{h}.b_Q'], dtype=float)
    out['ffn_A'] = np.asarray(p['W_write'], dtype=float).reshape(M.N_OPS, 3 * M.NU)
    out['ffn_B'] = np.stack([np.asarray(p['n_write'], dtype=float),
                             np.asarray(p['sp_delta'], dtype=float)]).T
    out['ffn_C'] = np.asarray(p['ctrl'], dtype=float)
    for pi, a in enumerate(acts):
        out[f'act_rom_{pi}'] = P(a['rom'])
        for k, (q, mem) in enumerate(a['snaps']):
            out[f'act_query_{pi}_{k}'] = P(q)
            out[f'act_mem_{pi}_{k}'] = P(mem)
    return out


# ---------------- tolerance-mode discovery -----------------------------------
def snap_mat(W, tau):
    W = np.asarray(W, dtype=float)
    return np.where(np.abs(W) > tau, np.round(W * 2.0) / 2.0, 0.0)


def sel(W):
    """Per-row [(column, weight)] of the surviving entries of a snapped matrix."""
    return [[(int(c), float(W[r, c])) for c in np.flatnonzero(W[r] != 0.0)]
            for r in range(W.shape[0])]


def _first_col(rows, i):
    """Column of the first surviving entry in row ``i``, or None."""
    if i >= len(rows) or not rows[i]:
        return None
    return rows[i][0][0]


def _fit_law(Mact, c0, c1):
    """Fit col_c1 = -(col_c0 / 2)^2 over the activation rows. Reports residual and
    R^2 rather than asserting integrality."""
    j = Mact[:, c0] / 2.0
    y = Mact[:, c1]
    pred = -j ** 2
    resid = y - pred
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
    return dict(c0=int(c0), c1=int(c1), resid_max=float(np.abs(resid).max()),
                r2=float(r2),
                integral=bool(np.allclose(j, np.round(j), atol=1e-6)))


def _find_opcode_col(ROM, live_rom):
    """The integer ROM column in bijection with the one-hot indicator block, or
    None when no such column exists."""
    binary = [c for c in sorted(live_rom)
              if set(np.unique(ROM[:, c])) <= {0.0, 1.0} and ROM[:, c].sum() < len(ROM)]
    onehot = [c for c in binary if ROM[:, c].sum() > 0]
    if not onehot:
        return None, []
    onehot = [c for c in onehot if np.all(ROM[:, onehot].sum(1) == 1)]
    if not onehot:
        return None, []
    ids = [c for c in sorted(live_rom)
           if c not in binary and np.allclose(ROM[:, c], np.round(ROM[:, c]))]
    for c in ids:
        groups = {v: set(np.flatnonzero(np.abs(ROM[:, onehot][ROM[:, c] == v]).sum(0)))
                  for v in np.unique(ROM[:, c])}
        if (all(len(g) == 1 for g in groups.values())
                and len({tuple(g) for g in groups.values()}) == len(groups)):
            return int(c), onehot
    return None, onehot


def _blank():
    return dict(heads_found=0, region_split_ok=False, law=None, law_fit_ok=None,
                opcode_col_found=None, alignment_survivors=None,
                opcodes_recovered=0.0, replay_ok=False, stage_failed=None)


def discover(artifact, tau, truth=None):
    """Tolerance-mode blind discovery. Returns a GRADED dict, never an assertion.

    ``truth`` is used only to score the recovered instruction table afterwards;
    no stage of the discovery reads it.
    """
    A = artifact
    out = _blank()
    out['tau'] = float(tau)

    heads = sorted({k.split('_')[0] for k in A if k.startswith('head')})
    d_model = A[heads[0] + '_Wq'].shape[1]
    roms = sorted(k for k in A if k.startswith('act_rom'))
    mems = sorted(k for k in A if k.startswith('act_mem'))
    qs = sorted(k for k in A if k.startswith('act_query'))
    ROM = np.vstack([A[k] for k in roms])
    MEM = np.vstack([A[k] for k in mems])
    QRY = np.vstack([A[k] for k in qs])
    live = {'rom': set(np.flatnonzero(np.abs(ROM).max(0) > 0)),
            'mem': set(np.flatnonzero(np.abs(MEM).max(0) > 0)),
            'qry': set(np.flatnonzero(np.abs(QRY).max(0) > 0))}

    # --- 1. head geometry from the snapped key matrices ---
    S = {}
    H = {}
    for h in heads:
        Wq, Wk, Wv = (snap_mat(A[f'{h}_W{m}'], tau) for m in 'qkv')
        bq = snap_mat(np.asarray(A[f'{h}_bq'], float)[None, :], tau)[0]
        S[h] = dict(Wq=Wq, Wk=Wk, Wv=Wv, bq=bq)
        kq, kk, kv = sel(Wq), sel(Wk), sel(Wv)
        kcols = [c for row in kk for c, _ in row]
        region = None
        if len(kcols) == 2:
            region = 'rom' if set(kcols) <= live['rom'] - live['mem'] else 'mem'
        H[h] = dict(q=kq, k=kk, v=kv, b=bq.tolist(), region=region, kcols=kcols)
    out['heads_found'] = sum(1 for h in heads if len(H[h]['kcols']) == 2)

    rom_heads = [h for h in heads if H[h]['region'] == 'rom']
    mem_heads = [h for h in heads if H[h]['region'] == 'mem']
    out['region_split_ok'] = bool(rom_heads and mem_heads)
    if not out['region_split_ok']:
        out['stage_failed'] = 'region_split'
        return out

    # --- 2. the addressing law, fitted from activations ---
    law = {}
    for region, Mact, hs in (('rom', ROM, rom_heads), ('mem', MEM, mem_heads)):
        c0, c1 = H[hs[0]]['kcols']
        law[region] = _fit_law(Mact, c0, c1)
    out['law'] = law
    out['law_fit_ok'] = bool(all(law[r]['integral'] and law[r]['r2'] > 0.99
                                 and law[r]['resid_max'] < 1e-3 for r in law))

    # --- 3. read ports ---
    ONE_COL = next((_first_col(H[h]['q'], 1) for h in heads
                    if _first_col(H[h]['q'], 1) is not None), None)
    IP_COL = _first_col(H[rom_heads[0]]['q'], 0)
    SP_COL = _first_col(H[mem_heads[0]]['q'], 0)
    c0m, c1m = law['mem']['c0'], law['mem']['c1']
    addr_head = [h for h in mem_heads if _first_col(H[h]['v'], 0) == c0m]
    val_heads = sorted([h for h in mem_heads if h not in addr_head],
                       key=lambda h: -H[h]['b'][0])
    VAL_COL = _first_col(H[val_heads[0]]['v'], 0) if val_heads else None
    is_mem = [c for c in live['mem'] - live['rom']
              if set(np.unique(MEM[:, c])) == {1.0}]
    is_state = sorted(live['qry'] - live['rom'] - live['mem'])
    if (ONE_COL is None or IP_COL is None or SP_COL is None or VAL_COL is None
            or not is_mem or not is_state):
        out['stage_failed'] = 'read_ports'
        return out
    IS_MEM, IS_STATE = is_mem[0], is_state[0]

    # --- 4. the opcode block in the ROM ---
    opcode_col, _ = _find_opcode_col(ROM, live['rom'])
    out['opcode_col_found'] = opcode_col is not None
    if opcode_col is None:
        out['stage_failed'] = 'opcode_block'
        return out
    op_reader = [h for h in rom_heads if _first_col(H[h]['v'], 0) == opcode_col]
    arg_reader = [h for h in rom_heads if h not in op_reader
                  and _first_col(H[h]['v'], 0) is not None]
    if not op_reader or not arg_reader:
        out['stage_failed'] = 'rom_read_ports'
        return out
    op_head, arg_head = op_reader[0], arg_reader[0]

    # --- 5. dispatch tensors, snapped ---
    n_ops = A['ffn_A'].shape[0]
    Wwrite = snap_mat(A['ffn_A'], tau).reshape(n_ops, 3, M.NU)
    nwrite = np.asarray(A['ffn_B'], float)[:, 0]          # not a trained parameter
    spdelta = np.round(np.asarray(A['ffn_B'], float)[:, 1])
    ctrl = (np.asarray(A['ffn_C'], float) > 0.5).astype(float)

    def mkrow(addr, val, wo):
        e = np.zeros(d_model)
        e[IS_MEM] = 1.0
        e[c0m] = 2.0 * addr
        e[c1m] = -float(addr * addr) + EPSQ * wo
        e[VAL_COL] = float(val)
        e[ONE_COL] = 1.0
        return e

    def mkq(ip, sp):
        e = np.zeros(d_model)
        e[IS_STATE] = 1.0
        e[IP_COL] = float(ip)
        e[SP_COL] = float(sp)
        e[ONE_COL] = 1.0
        return e

    def attend(h, q, mem):
        if len(mem) == 0:
            return 0.0, -1
        qq = S[h]['Wq'] @ q + S[h]['bq']
        i = int(np.argmax((mem @ S[h]['Wk'].T) @ qq))
        return float((S[h]['Wv'] @ mem[i])[0]), i

    def simulate(rom, row_of, targets=(), steps=250):
        rows, wo, sp, ip = [], 0, 0, 0
        seen = [False] * len(targets)
        for _ in range(steps):
            q = mkq(ip, sp)
            mem = np.stack(rows) if rows else np.zeros((0, d_model))
            for ti, t in enumerate(targets):
                if not seen[ti] and mem.shape == t.shape and np.allclose(mem, t):
                    seen[ti] = True
            opv, _ = attend(op_head, q, rom)
            arg, _ = attend(arg_head, q, rom)
            if not (np.isfinite(opv) and np.isfinite(arg)):
                return None, seen
            if int(round(opv)) not in row_of:
                return None, seen
            r = row_of[int(round(opv))]
            vals = []
            for k, h in enumerate(val_heads):
                v, i = attend(h, q, mem)
                if i < 0:
                    vals.append(0.0)
                    continue
                hit = round(mem[i, c0m] / 2.0) == sp - k
                vals.append(v if hit else 0.0)
            while len(vals) < M.NU - 1:
                vals.append(0.0)
            u = np.array([arg] + vals[:M.NU - 1])
            if not np.all(np.isfinite(u)):
                return None, seen
            new_sp = int(round(sp + spdelta[r]))
            if new_sp < 0:
                return None, seen
            for c in range(int(nwrite[r])):
                w = float(Wwrite[r, c] @ u)
                if not np.isfinite(w) or abs(w) > 1e12:
                    return None, seen
                rows.append(mkrow(new_sp - c, w, wo))
                wo += 1
            jz, jnz, halt = ctrl[r] > 0.5
            if halt:
                return vals[0], seen
            ip = int(round(arg)) if ((jz and vals[0] == 0)
                                     or (jnz and vals[0] != 0)) else ip + 1
            if not 0 <= ip < len(rom):
                return None, seen
            sp = new_sp
        return None, seen

    # --- 6. opcode-id -> dispatch-row alignment, replayed on the snapped weights ---
    survivors = []
    for shift in range(n_ops):
        row_of = {v: (v - 1 + shift) % n_ops for v in range(1, n_ops + 1)}
        good = True
        for pi, rk in enumerate(roms):
            snaps = [A[k] for k in mems if k.startswith(f'act_mem_{pi}_')]
            res, seen = simulate(A[rk], row_of, targets=snaps)
            if res is None or not all(seen):
                good = False
                break
        if good:
            survivors.append(shift)
    out['alignment_survivors'] = survivors
    out['replay_ok'] = survivors == [0]

    # --- 7. score the recovered instruction table ---
    if truth is not None:
        out['opcodes_recovered'] = (
            score_table(Wwrite, spdelta, ctrl, nwrite, survivors[0], truth)
            if survivors else 0.0)
        out['opcodes_recovered_best_shift'] = max(
            score_table(Wwrite, spdelta, ctrl, nwrite, sh, truth)
            for sh in range(n_ops))
    return out


def score_table(Wwrite, spdelta, ctrl, nwrite, shift, truth):
    """Partial credit of the recovered table against the compiled one, opcode id
    v against dispatch row (v-1+shift) % 12: delta 0.25, ctrl 0.25, writes 0.5.
    Sums to 0..12."""
    total = 0.0
    n = len(truth['sp_delta'])
    for v in range(1, n + 1):
        r = (v - 1 + shift) % n
        t = v - 1
        sc = 0.25 * float(spdelta[r] == truth['sp_delta'][t])
        sc += 0.25 * float(np.array_equal(ctrl[r], truth['ctrl'][t]))
        k = int(truth['n_write'][t])
        sc += 0.5 * float(int(nwrite[r]) == k
                          and np.array_equal(Wwrite[r, :k], truth['W_write'][t, :k]))
        total += sc
    return float(total)


# ---------------- the sweep over checkpoints ---------------------------------
def load_ckpt(arm, step):
    f = os.path.join(CKPT, f'{arm}_{step:04d}.npz')
    z = np.load(f)
    return {k: z[k] for k in z.files}


def run_sweep(arms=ARMS, taus=TAUS, verbose=True):
    truth = M.compile_params()
    acts = compiled_activations(truth)
    perm, horder = anonymization()
    structural = json.load(open(os.path.join(HERE, 'results.json')))
    steps = structural['config']['checkpoints']
    out = dict(config=dict(arms=list(arms), taus=list(taus), steps=steps,
                           perm_seed=PERM_SEED, capture_steps=list(CAPTURE_STEPS)),
               arms={})
    for arm in arms:
        rows = []
        struct_curve = {c['step']: c for c in structural['arms'][arm]}
        for step in steps:
            p = load_ckpt(arm, step)
            art = build_artifact(p, acts, perm, horder)
            res = [discover(art, tau, truth) for tau in taus]
            sc = struct_curve[step]
            rows.append(dict(
                step=step,
                blind=[{k: r[k] for k in ('tau', 'heads_found', 'region_split_ok',
                                          'law_fit_ok', 'opcode_col_found',
                                          'alignment_survivors', 'opcodes_recovered',
                                          'replay_ok', 'stage_failed')} for r in res],
                law=[r['law'] for r in res],
                structural=[dict(tau=s['tau'], isa=s['isa'], addr=s['addr'],
                                 replay=s['replay']) for s in sc['sweep']],
                oracle=sc['oracle'], oracle_overwrite=sc['oracle_overwrite'],
                l2_heads=sc['l2_heads'], l2_dispatch=sc['l2_dispatch']))
            if verbose:
                b = {r['tau']: r['opcodes_recovered'] for r in res}
                s = {r['tau']: r['isa'] * 12 for r in sc['sweep']}
                print(f'[{arm}] step {step:5d} blind '
                      + ' '.join(f'{t}:{b[t]:.2f}' for t in taus)
                      + ' | structural '
                      + ' '.join(f'{t}:{s[t]:.2f}' for t in taus), flush=True)
        out['arms'][arm] = rows
    return out


def first_failure(rows, tau, field='opcodes_recovered', full=12.0):
    for r in rows:
        b = next(x for x in r['blind'] if x['tau'] == tau)
        ok = b[field] >= full if field == 'opcodes_recovered' else b[field]
        if not ok:
            return r['step']
    return None


def plot(res, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms = res['config']['arms']
    fig, axes = plt.subplots(1, len(arms), figsize=(4.0 * len(arms), 3.6),
                             sharey=True)
    for ax, arm in zip(np.atleast_1d(axes), arms):
        rows = res['arms'][arm]
        x = [max(r['step'], 0.5) for r in rows]
        for tau, style in ((0.2, '-'), (0.45, '--')):
            blind = [next(b['opcodes_recovered'] for b in r['blind']
                          if b['tau'] == tau) / 12.0 for r in rows]
            struct = [next(s['isa'] for s in r['structural'] if s['tau'] == tau)
                      for r in rows]
            ax.plot(x, blind, style, color='#c1272d', marker='o', ms=3,
                    label=f'blind, τ={tau}')
            ax.plot(x, struct, style, color='#0b6e99', marker='s', ms=3,
                    label=f'structural, τ={tau}')
        ax.set_xscale('log')
        ax.set_xlabel('training step')
        ax.set_title(arm)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel('ISA recovered (fraction of 12)')
    np.atleast_1d(axes)[-1].legend(fontsize=7, loc='center left')
    fig.suptitle('Blind discovery vs structural verification on the decayed checkpoints',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == '__main__':
    res = run_sweep()
    json.dump(res, open(os.path.join(HERE, 'blind_decay.json'), 'w'), indent=1)
    plot(res, os.path.join(HERE, 'blind_vs_structural.png'))
    print('\nfirst step below 12/12 (blind) per arm and tau:')
    for arm in res['config']['arms']:
        rows = res['arms'][arm]
        print(f'  {arm:20s} ' + ' '.join(
            f'tau={t}:{first_failure(rows, t)}' for t in res['config']['taus']))
    print('wrote blind_decay.json and blind_vs_structural.png')
