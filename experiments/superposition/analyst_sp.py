"""SUPERPOSITION-AWARE BLIND ANALYST.

Input: the anonymized tensors of a packed machine (weights + a handful of captured
activations). No access to the compiler, the ISA, the codebook, or any name.

The axis-aligned analyst in ``experiments/blind_recovery/analyst.py`` leans on
sparsity everywhere -- ``sel()`` reads the nonzero columns of each weight matrix, the
region split asks which columns a head keys on, and the opcode detector looks for a
one-hot block of columns. Under superposition every weight matrix is dense and every
one of those moves dies. The replacements here are geometric and statistical:

  region split      which activation family a head's W_K actually projects onto,
                    by projected magnitude, not by column support
  addressing law    project rows through W_K into 2D and fit k1 against k0 there;
                    the fit runs over many rows, so it averages interference away
  head roles        the address head is the one whose W_V readout tracks the
                    recovered address; value heads sort by their b_Q offsets
  basis recovery    least-squares the embedding directions out of the captured
                    snapshots, given the addresses recovered above
  opcode block      no one-hot columns survive packing; the two ROM heads are tried
                    both ways round and the replay decides

Kept unchanged from the axis-aligned analyst: the strongest check is still replaying
the recovered machine against every captured memory snapshot. Under superposition the
replay cannot be bit-exact -- the machine stores values it read through a lossy
readout -- so the criterion becomes an exact match of the decoded address sequence
plus a relative-error threshold on the embeddings, and the best-fitting candidate
wins. At d >= n_features under the pinv arm the readouts are exact and this reduces
to the original bit-exact check.
"""
import numpy as np

REL_TOL = 0.5          # relative Frobenius error a candidate replay may carry
EPSQ = 1e-6            # write-order tiebreak quantum, recovered from the data below


def _project(M, Wk):
    return M @ Wk.T if len(M) else np.zeros((0, Wk.shape[0]))


def _decode_addresses(a, b, unique=False):
    """Joint integer decode of the address from both projected coordinates.

    The law is score(x, j) = 2xj - j^2, so a row's projection is (2j, -j^2) plus
    interference of comparable scale on each coordinate. Rounding a/2 alone throws
    away half the evidence; picking the integer j that minimises
    (a - 2j)^2 + (b + j^2)^2 uses both measurements. Equal weights are right here --
    both coordinates are readouts of the same row and carry the same interference.
    """
    lo, hi = int(np.floor(a.min() / 2)) - 2, int(np.ceil(a.max() / 2)) + 2
    cand = np.arange(lo, hi + 1)
    cost = (a[:, None] - 2 * cand[None, :]) ** 2 + (b[:, None] + cand[None, :] ** 2) ** 2
    if unique and len(cand) >= len(a):
        # A program holds one instruction per address, so the ROM's addresses are a
        # permutation, not n independent guesses. Solving the assignment instead of
        # rounding each row on its own is where the analyst gets to average
        # interference out -- a row whose own projection is ambiguous is pinned by
        # every other row having claimed its address.
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(cost)
        j = np.empty(len(a), dtype=int)
        j[ri] = cand[ci]
        return j, np.sqrt(cost[np.arange(len(a)), [list(cand).index(x) for x in j]])
    j = cand[np.argmin(cost, axis=1)]
    return j, np.sqrt(cost.min(axis=1))


def _parabola_fit(P2, unique=False):
    """Best (k0, k1) coordinate assignment, decoded addresses, and residuals.

    One projected coordinate is 2j and the other is -j^2; try both orders and keep
    the one the parabola actually fits.
    """
    best = None
    for a, b in ((0, 1), (1, 0)):
        j, resid = _decode_addresses(P2[:, a], P2[:, b], unique=unique)
        scale = max(float(np.abs(P2[:, b]).max()), 1.0)
        norm = float(resid.max() / scale)
        if best is None or norm < best['resid']:
            best = dict(k0=a, k1=b, resid=norm, j=j, r=P2[:, b] + j.astype(float) ** 2,
                        int_err=float(np.abs(P2[:, a] / 2.0 - j).max()))
    return best


def recover(A, rel_tol=REL_TOL):
    """Blind recovery. Returns the recovered machine plus diagnostics.

    Never sees ground truth -- scoring happens in the caller.
    """
    diag = {}
    heads = sorted({k.split('_')[0] for k in A.files if k.startswith('head')}
                   if hasattr(A, 'files') else
                   {k.split('_')[0] for k in A if k.startswith('head')})
    d_model = A[heads[0] + '_Wq'].shape[1]
    keys = list(A.files) if hasattr(A, 'files') else list(A)
    roms = sorted(k for k in keys if k.startswith('act_rom'))
    mems = sorted(k for k in keys if k.startswith('act_mem'))
    # act_query_* ships in the artifact but goes unused: once the replay runs
    # arithmetically the analyst never has to synthesize a query embedding
    mems = [k for k in mems if len(A[k])]
    if not roms or not mems:
        return dict(ok=False, why='no activations captured -- the machine died first',
                    diag=diag)
    ROM, MEM = np.vstack([A[k] for k in roms]), np.vstack([A[k] for k in mems])
    n_ops = A['ffn_A'].shape[0]
    diag.update(d_model=d_model, n_heads=len(heads), n_ops=n_ops,
                head_dim=A[heads[0] + '_Wq'].shape[0], v_dim=A[heads[0] + '_Wv'].shape[0])

    # ── 1. region split: which family does each head's W_K actually see? ─────────
    # A head that genuinely keys a family projects it at a magnitude comparable to
    # the rows themselves; on the other family it sees only interference. Raw
    # magnitude does not separate the two -- ROM rows simply have larger norms --
    # so normalize each family's projection by that family's row scale.
    H = {}
    for h in heads:
        Wk = A[h + '_Wk']
        mag = {fam: float(np.linalg.norm(_project(M, Wk)) /
                          max(np.linalg.norm(M), 1e-12))
               for fam, M in (('rom', ROM), ('mem', MEM))}
        region = max(mag, key=mag.get)
        H[h] = dict(region=region, mag=mag, b=A[h + '_bq'].tolist())
    rom_heads = [h for h in heads if H[h]['region'] == 'rom']
    mem_heads = [h for h in heads if H[h]['region'] == 'mem']
    diag['region_split'] = {h: H[h]['region'] for h in heads}
    if len(rom_heads) < 2 or len(mem_heads) < 4:
        return dict(ok=False, why='region split degenerate', diag=diag)

    # ── 2. the addressing law, fitted from the projections ──────────────────────
    # Each ROM is its own address space, so the one-instruction-per-address
    # constraint applies within a single ROM, never across the concatenation.
    laws = {}
    for region, M, hs in (('rom', A[roms[0]], rom_heads), ('mem', MEM, mem_heads)):
        fits = [_parabola_fit(_project(M, A[h + '_Wk']), unique=(region == 'rom'))
                for h in hs]
        laws[region] = min(fits, key=lambda f: f['resid'])
        diag[f'law_{region}'] = dict(resid=laws[region]['resid'],
                                     int_err=laws[region]['int_err'])
    # the write-order tiebreak lives in the mem residual; its quantum sets the ceiling
    r = laws['mem']['r']
    diag['tiebreak_quantum'] = float(np.min(np.diff(np.unique(np.round(r, 12))))) \
        if len(np.unique(np.round(r, 12))) > 1 else 0.0

    # ── 3. head roles ───────────────────────────────────────────────────────────
    # the address head reads back the winning row's address (V ~ k0/2 = j); the rest
    # return stored data. Sort the data heads by their query offsets: 0, -1, -2.
    j_mem = laws['mem']['j']
    addr_score = {}
    for h in mem_heads:
        v = (A[h + '_Wv'] @ MEM.T)[0]
        addr_score[h] = float(np.abs(v - j_mem).mean() / max(np.abs(j_mem).mean(), 1e-9))
    addr_head = min(addr_score, key=addr_score.get)
    val_heads = sorted([h for h in mem_heads if h != addr_head], key=lambda h: -H[h]['b'][0])
    diag['addr_head_margin'] = float(
        sorted(addr_score.values())[1] - sorted(addr_score.values())[0])
    if len(val_heads) < 3:
        return dict(ok=False, why='fewer than three data ports', diag=diag)


    # ── 4. decode the evidence: ROM contents and the captured snapshots ─────────
    # Everything below works in decoded (address, value) space. The axis-aligned
    # analyst replayed the recovered machine THROUGH the weights, which under
    # superposition would make recovery inherit the machine's own fragility -- the
    # replay would face the same per-step exact argmax the machine is failing at.
    # A decompiler that has recovered the ISA runs it in its own interpreter
    # instead, and only has to decode the fixed, small evidence set.
    def decode_rom(rk, op_head, arg_head, limit=32):
        """Decoded ROM, plus a short list of alternates for the rows whose readout
        did not land cleanly on an integer.

        Rounding a scalar that arrived through a lossy readout is the one place a
        single row gets no help from the others. Rather than commit, hand the
        genuinely ambiguous rows to the replay as alternatives and let the snapshot
        check decide -- the same move the analyst already makes for the two ROM head
        roles and the twelve cyclic offsets.
        """
        rom = A[rk]
        fit = _parabola_fit(_project(rom, A[rom_heads[0] + '_Wk']), unique=True)
        ops = (A[op_head + '_Wv'] @ rom.T)[0]
        args = (A[arg_head + '_Wv'] @ rom.T)[0]
        base = {int(a): (int(round(o)), int(round(g)))
                for a, o, g in zip(fit['j'], ops, args)}
        amb = []
        for a, o, g in zip(fit['j'], ops, args):
            for slot, x in ((0, o), (1, g)):
                frac = abs(x - round(x))
                if frac > 0.15:
                    alt = int(round(x)) + (1 if x > round(x) else -1)
                    amb.append((frac, int(a), slot, alt))
        amb.sort(reverse=True)
        variants = [base]
        for _, addr, slot, alt in amb[:int(np.log2(limit))]:
            variants += [{**v, addr: tuple(alt if i == slot else v[addr][i]
                                           for i in range(2))} for v in variants]
            if len(variants) >= limit:
                break
        return variants

    def decode_mem(mk):
        M = A[mk]
        fit = _parabola_fit(_project(M, A[val_heads[0] + '_Wk']))
        vals = (A[val_heads[0] + '_Wv'] @ M.T)[0]
        return [(int(a), float(v)) for a, v in zip(fit['j'], vals)]

    targets = {rk: [decode_mem(mk) for mk in mems
                    if mk.startswith(f'act_mem_{rk.split("_")[-1]}_')] for rk in roms}
    diag['n_snapshots'] = sum(len(v) for v in targets.values())

    Wwrite = A['ffn_A'].reshape(n_ops, 3, 4)
    nwrite, spdelta = A['ffn_B'][:, 0], A['ffn_B'][:, 1]
    ctrl = A['ffn_C']

    def replay(prog_map, row_of, snaps, steps=120):
        """Run the recovered machine arithmetically. Append-only memory, latest write
        at an address wins, three read ports at offsets 0/-1/-2."""
        rows, sp, ip = [], 0, 0
        seen = [False] * len(snaps)
        for _ in range(steps):
            for si, t in enumerate(snaps):
                # addresses are integers pinned by the joint decode, so they must
                # match exactly; the stored values arrived through a lossy readout,
                # so they are compared as floats against what the replay predicts
                if seen[si] or len(rows) != len(t):
                    continue
                if all(a == b for (a, _), (b, _) in zip(rows, t)) and \
                        max((abs(v - w) for (_, v), (_, w) in zip(rows, t)), default=0) < 0.5:
                    seen[si] = True
            if snaps and all(seen):
                return 'matched', seen
            if ip not in prog_map:
                return None, seen
            op, arg = prog_map[ip]
            if op not in row_of:
                return None, seen
            r = row_of[op]
            vals = []
            for k in range(3):
                hit = [v for a, v in rows if a == sp - k]
                vals.append(hit[-1] if hit else 0)
            u = np.array([arg] + vals, dtype=float)
            new_sp = int(sp + spdelta[r])
            for c in range(int(nwrite[r])):
                rows.append((new_sp - c, int(round(float(Wwrite[r, c] @ u)))))
            jz, jnz, halt = ctrl[r]
            if halt:
                return vals[0], seen
            ip = arg if ((jz and vals[0] == 0) or (jnz and vals[0] != 0)) else ip + 1
            sp = new_sp
        return None, seen

    # ── 5. opcode-id -> dispatch-row alignment ──────────────────────────────────
    # Two unknowns at once: which ROM head reads the opcode (the one-hot block that
    # used to settle it is gone), and the cyclic offset between opcode ids and
    # dispatch rows. 2 x n_ops candidates, all decided by replay.
    cands, detail = [], []
    for op_head, arg_head in ((rom_heads[0], rom_heads[1]), (rom_heads[1], rom_heads[0])):
        variants = {rk: decode_rom(rk, op_head, arg_head) for rk in roms}
        for shift in range(n_ops):
            row_of = {v: (v - 1 + shift) % n_ops for v in range(1, n_ops + 1)}
            hit = tot = 0
            chosen = {}
            for rk in roms:
                best = (-1, variants[rk][0])
                for pm in variants[rk]:
                    _, seen = replay(pm, row_of, targets[rk])
                    if sum(seen) > best[0]:
                        best = (sum(seen), pm)
                    if best[0] == len(targets[rk]):
                        break
                chosen[rk] = best[1]
                hit += best[0]
                tot += len(targets[rk])
            detail.append(dict(op_head=op_head, shift=shift, matched=hit, of=tot))
            if tot and hit == tot:
                cands.append(dict(op_head=op_head, arg_head=arg_head, shift=shift,
                                  progs=chosen))
    diag['n_candidates'] = len(cands)
    diag['best_partial'] = max(detail, key=lambda c: c['matched'])
    if not cands:
        return dict(ok=False, why='no alignment reproduces the captured snapshots', diag=diag)
    if len(cands) > 1:
        return dict(ok=False, why=f'{len(cands)} alignments fit the evidence equally',
                    diag=diag)

    win = cands[0]
    row_of = {v: (v - 1 + win['shift']) % n_ops for v in range(1, n_ops + 1)}
    results = {rk: replay(win['progs'][rk], row_of, [], steps=400)[0] for rk in roms}
    return dict(ok=True, row_of=row_of, shift=win['shift'], op_head=win['op_head'],
                addr_head=addr_head, val_heads=val_heads, replays=results, diag=diag)


def instruction_table(A, row_of):
    """The recovered ISA, in the same shape the axis-aligned analyst printed."""
    n_ops = A['ffn_A'].shape[0]
    Wwrite = A['ffn_A'].reshape(n_ops, 3, 4)
    nwrite, spdelta = A['ffn_B'][:, 0], A['ffn_B'][:, 1]
    ctrl = A['ffn_C']
    names = ['imm', 'top', 'top-1', 'top-2']
    table = {}
    for v in range(1, n_ops + 1):
        r = row_of[v]
        parts = []
        for c in range(int(nwrite[r])):
            terms = [f'{x:+g}*{names[k]}' for k, x in enumerate(Wwrite[r, c]) if x]
            parts.append(f'[sp{f"-{c}" if c else ""}] = ' + (' '.join(terms) or '0'))
        jz, jnz, halt = ctrl[r]
        if halt:
            parts.append('HALT')
        if jz:
            parts.append('if top==0: ip=imm')
        if jnz:
            parts.append('if top!=0: ip=imm')
        table[v] = dict(delta=float(spdelta[r]), cells=int(nwrite[r]), effect='; '.join(parts))
    return table
