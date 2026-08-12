"""BLIND ANALYST.

Input: /home/claude/exp/artifact.npz — anonymized tensors, permuted d_model basis.
No access to the compiler, the ISA, or any name. Recovers the machine and prints
a reconstructed instruction table.
"""
import numpy as np, itertools, json

A = np.load('/home/claude/exp/artifact.npz')
report = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    report.append(s); print(s)

# ── 1. inventory ────────────────────────────────────────────────
heads = sorted({k.split('_')[0] for k in A.files if k.startswith('head')})
d_model = A[heads[0] + '_Wq'].shape[1]
roms = sorted(k for k in A.files if k.startswith('act_rom'))
mems = sorted(k for k in A.files if k.startswith('act_mem'))
qs = sorted(k for k in A.files if k.startswith('act_query'))
say(f'[1] d_model={d_model}  heads={len(heads)}  '
    f'head_dim={A[heads[0]+"_Wq"].shape[0]}  v_dim={A[heads[0]+"_Wv"].shape[0]}')
say(f'    activation families: {len(roms)} rom, {len(mems)} mem, {len(qs)} query')

def sel(W):
    return [[(int(c), float(W[r, c])) for c in np.flatnonzero(np.abs(W[r]) > 1e-12)]
            for r in range(W.shape[0])]

# ── 2. which activation family does each head key against? ──────
ROM = np.vstack([A[k] for k in roms])
MEM = np.vstack([A[k] for k in mems])
QRY = np.vstack([A[k] for k in qs])
live = {'rom': set(np.flatnonzero(np.abs(ROM).max(0) > 0)),
        'mem': set(np.flatnonzero(np.abs(MEM).max(0) > 0)),
        'qry': set(np.flatnonzero(np.abs(QRY).max(0) > 0))}
say(f'[2] live columns: rom={len(live["rom"])} mem={len(live["mem"])} qry={len(live["qry"])}'
    f'  overlap rom∩mem={len(live["rom"] & live["mem"])}')

H = {}
for h in heads:
    kq, kk, kv = sel(A[h + '_Wq']), sel(A[h + '_Wk']), sel(A[h + '_Wv'])
    kcols = [c for row in kk for c, _ in row]
    region = 'rom' if set(kcols) <= live['rom'] - live['mem'] else 'mem'
    H[h] = dict(q=kq, k=kk, v=kv, b=A[h + '_bq'].tolist(), region=region, kcols=kcols)
    say(f'    {h}: keys on cols {kcols} -> region {region};  '
        f'q={kq} b={H[h]["b"]}  v={kv}')

# ── 3. the addressing law, fitted from activations ──────────────
laws = {}
for region, M in (('rom', ROM), ('mem', MEM)):
    c0, c1 = next(H[h]['kcols'] for h in heads if H[h]['region'] == region)
    j = M[:, c0] / 2.0
    resid = M[:, c1] + j ** 2
    ok_int = np.allclose(j, np.round(j))
    laws[region] = dict(c0=c0, c1=c1, resid_max=float(np.abs(resid).max()))
    say(f'[3] {region}: col{c0} = 2j (integer j: {ok_int}), '
        f'col{c1} = -j^2 + r, max|r|={np.abs(resid).max():.2e}')
    if np.abs(resid).max() > 0:
        big = max((A[k] for k in mems), key=len)
        rr = big[:, c1] + (big[:, c0] / 2.0) ** 2
        d = np.diff(np.unique(np.round(rr, 12)))
        say(f'    r nonzero only in {region}; within one snapshot r increases with row '
            f'index: {bool(np.all(np.diff(rr) > 0))}; quantum {d.min():.1e} '
            f'-> append-only log, later write wins the tie (recency)')
say('    => score(q=(x,1), key_j) = 2xj - j^2 = -(j-x)^2 + x^2 : hard-argmax at j = x.')
say('    Content-addressable read by exact integer address. Exact while j^2 is')
say('    representable; the tiebreak quantum sets the write-count ceiling.')

# ── 4. read ports ───────────────────────────────────────────────
qcol_of = lambda h: H[h]['q'][0][0][0]
for region in ('rom', 'mem'):
    hs = [h for h in heads if H[h]['region'] == region]
    say(f'[4] {region}: {len(hs)} heads, query col {sorted({qcol_of(h) for h in hs})}, '
        f'offsets {sorted(H[h]["b"][0] for h in hs)}, '
        f'value cols {sorted({H[h]["v"][0][0][0] for h in hs})}')

ONE_COL = [c for c, _ in H[heads[0]]['q'][1]][0]
IP_COL = qcol_of([h for h in heads if H[h]['region'] == 'rom'][0])
SP_COL = qcol_of([h for h in heads if H[h]['region'] == 'mem'][0])
rom_heads = [h for h in heads if H[h]['region'] == 'rom']
mem_heads = [h for h in heads if H[h]['region'] == 'mem']
# the mem head whose V reads a KEY column at coeff 0.5 is an address read-back (hit check)
addr_head = [h for h in mem_heads if H[h]['v'][0][0][0] in (laws['mem']['c0'],)]
val_heads = sorted([h for h in mem_heads if h not in addr_head], key=lambda h: -H[h]['b'][0])
VAL_COL = H[val_heads[0]]['v'][0][0][0]
say(f'    inferred: ip col={IP_COL} sp col={SP_COL} one col={ONE_COL} value col={VAL_COL}')
say(f'    one mem head returns key/2 = the winning row address -> hit/miss check, not data')

# ── 5. opcode block in the ROM ──────────────────────────────────
binary_cols = [c for c in sorted(live['rom'])
               if set(np.unique(ROM[:, c])) <= {0.0, 1.0} and ROM[:, c].sum() < len(ROM)]
onehot_cols = [c for c in binary_cols if ROM[:, c].sum() > 0]
onehot_cols = [c for c in onehot_cols if np.all(ROM[:, onehot_cols].sum(1) == 1)]
id_cols = [c for c in sorted(live['rom']) if c not in binary_cols
           and np.allclose(ROM[:, c], np.round(ROM[:, c]))]
opcode_col = None
for c in id_cols:
    groups = {v: set(np.flatnonzero(np.abs(ROM[:, onehot_cols][ROM[:, c] == v]).sum(0)))
              for v in np.unique(ROM[:, c])}
    if all(len(g) == 1 for g in groups.values()) and len({tuple(g) for g in groups.values()}) == len(groups):
        opcode_col = c; break
obs = sorted(np.unique(ROM[:, opcode_col]).astype(int).tolist())
say(f'[5] one-hot indicator block: {len(onehot_cols)} cols; integer col {opcode_col} '
    f'is in bijection with it -> opcode id. observed ids {obs}')
n_ops = A['ffn_A'].shape[0]
say(f'    dispatch tensors carry {n_ops} rows -> {n_ops}-opcode ISA '
    f'({len(obs)} exercised by the shipped ROMs)')
rom_heads = sorted(rom_heads, key=lambda h: H[h]['v'][0][0][0] != opcode_col)
say(f'    rom head {rom_heads[0]} reads col {opcode_col} (opcode); '
    f'{rom_heads[1]} reads col {H[rom_heads[1]]["v"][0][0][0]} -> immediate operand field')

# ── 6. reconstruct the machine, then fix opcode-id -> dispatch-row ─
Wwrite = A['ffn_A'].reshape(n_ops, 3, 4)
nwrite, spdelta = A['ffn_B'][:, 0], A['ffn_B'][:, 1]
ctrl = A['ffn_C']
IS_MEM = [c for c in live['mem'] - live['rom']
          if set(np.unique(MEM[:, c])) == {1.0}][0]
c0m, c1m = laws['mem']['c0'], laws['mem']['c1']
EPSQ = 1e-6
IS_STATE = [c for c in live['qry'] - live['rom'] - live['mem']][0]

def mkrow(addr, val, wo):
    e = np.zeros(d_model)
    e[IS_MEM] = 1.0; e[c0m] = 2.0 * addr; e[c1m] = -float(addr * addr) + EPSQ * wo
    e[VAL_COL] = float(val); e[ONE_COL] = 1.0
    return e

def mkq(ip, sp):
    e = np.zeros(d_model)
    e[IS_STATE] = 1.0; e[IP_COL] = ip; e[SP_COL] = sp; e[ONE_COL] = 1.0
    return e

def attend(h, q, mem):
    W = A[h + '_Wq'], A[h + '_Wk'], A[h + '_Wv']
    if len(mem) == 0:
        return 0.0, -1
    qq = W[0] @ q + A[h + '_bq']
    i = int(np.argmax((mem @ W[1].T) @ qq))
    return float((W[2] @ mem[i])[0]), i

def simulate(rom, row_of, targets=(), steps=250):
    rows, wo, sp, ip, seen = [], 0, 0, 0, [False]*len(targets)
    for _ in range(steps):
        q = mkq(ip, sp)
        mem = np.stack(rows) if rows else np.zeros((0, d_model))
        for ti, t in enumerate(targets):
            if not seen[ti] and mem.shape == t.shape and np.allclose(mem, t):
                seen[ti] = True
        opv, _ = attend(rom_heads[0], q, rom)
        arg, _ = attend(rom_heads[-1], q, rom)
        if int(round(opv)) not in row_of:
            return None, seen
        r = row_of[int(round(opv))]
        vals = []
        for k, h in enumerate(val_heads):
            v, i = attend(h, q, mem)
            if i < 0:
                vals.append(0.0); continue
            hit = round(mem[i, c0m] / 2.0) == sp - k
            vals.append(v if hit else 0.0)
        u = np.array([arg] + vals)
        new_sp = sp + spdelta[r]
        for c in range(int(nwrite[r])):
            rows.append(mkrow(int(new_sp) - c, float(Wwrite[r, c] @ u), wo)); wo += 1
        jz, jnz, halt = ctrl[r]
        if halt:
            return vals[0], seen
        ip = int(round(arg)) if ((jz and vals[0] == 0) or (jnz and vals[0] != 0)) else ip + 1
        sp = int(new_sp)
    return None, seen

survivors = []
for shift in range(n_ops):
    row_of = {v: (v - 1 + shift) % n_ops for v in range(1, n_ops + 1)}
    good = True
    for pi, rk in enumerate(roms):
        snaps = [A[k] for k in mems if k.startswith(f'act_mem_{pi}_')]
        res, seen = simulate(A[rk], row_of, targets=snaps)
        if res is None or not all(seen):
            good = False; break
    if good:
        survivors.append(shift)
say(f'[6] opcode-id -> dispatch-row alignment: searched {n_ops} cyclic offsets, '
    f'survivors that reproduce every captured memory snapshot exactly: {survivors}')
shift = survivors[0]
row_of = {v: (v - 1 + shift) % n_ops for v in range(1, n_ops + 1)}
for pi, rk in enumerate(roms):
    res, _ = simulate(A[rk], row_of)
    say(f'    replay {rk}: halts, result = {res}')

# ── 7. reconstructed instruction table ──────────────────────────
IN = ['imm', 'top', 'top-1', 'top-2']
def describe(r):
    parts = []
    for c in range(int(nwrite[r])):
        terms = [f'{v:+g}*{IN[k]}' for k, v in enumerate(Wwrite[r, c]) if v]
        parts.append(f'[sp{c and f"-{c}" or ""}] = ' + (' '.join(terms) or '0'))
    jz, jnz, halt = ctrl[r]
    if halt: parts.append('HALT')
    if jz: parts.append('if top==0: ip=imm')
    if jnz: parts.append('if top!=0: ip=imm')
    return '; '.join(parts)

say('\n[7] RECOVERED INSTRUCTION SET')
say(f'{"id":>3} {"Δsp":>4} {"cells":>5}  effect')
table = {}
for v in range(1, n_ops + 1):
    r = row_of[v]
    table[v] = dict(delta=float(spdelta[r]), cells=int(nwrite[r]), effect=describe(r))
    say(f'{v:>3} {spdelta[r]:>+4g} {int(nwrite[r]):>5}  {describe(r)}')

json.dump(dict(d_model=d_model, n_heads=len(heads), n_ops=n_ops, shift=shift,
               table=table), open('/home/claude/exp/recovered.json', 'w'), indent=1)
open('/home/claude/exp/analyst_report.txt', 'w').write('\n'.join(report))
