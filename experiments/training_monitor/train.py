"""Compiler as a training monitor.

Initialize a transformer AT the compiled LAC weights -- a point in weight space whose
program is known exactly -- run ordinary gradient descent, and run the tolerance-mode
analyst every few steps. Output: a decay curve of ISA recoverability against program
correctness, per arm.

Arms
  neutral   the machine's own task: next-state prediction over its own execution
            traces, teacher-forced. Loss starts near zero (softmax at BETA is not a
            hard argmax), so gradients are near zero -- only weight decay and
            optimizer noise move it. This is the DRIFT CONTROL, not a real arm.
  rival     a task the compiled machine does not solve: the same traces with targets
            from a RIVAL ISA -- dispatch rows cyclically shifted by one and the three
            stack reads at SP-1, SP-2, SP-3 instead of SP, SP-1, SP-2. Real gradient
            lands on the dispatch table, the stack heads' offsets, and the value
            readouts. This is where the decay curve comes from.
  random    the same architecture from random init on the neutral task. Not a decay
            arm; it is the "naturally trained model of the same shape" against which
            the realism proxy is measured, InterpBench's axis.

What is NOT here, deliberately: a circuit-preservation loss. That is InterpBench/SIIT
and it is the opposite experiment.

Which machine: the numpy compiler of experiments/blind_recovery (dispatch entirely in
tensors), re-expressed in torch -- NOT upstream executor.CompiledModel, whose forward
hard-codes every opcode in Python and so carries no dispatch gradient at all.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyst_tol as AT  # noqa: E402
import machine as M  # noqa: E402

torch.set_default_dtype(torch.float32)
BETA = 20.0            # key scaling: softmax weight on a runner-up at gap 1 is e^-20
CHECKPOINTS = [0, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')


# ---------------- data: teacher-forced traces of the compiled machine ----------
def harvest(truth):
    recs = []
    for name, prog, _ in M.ORACLE:
        tr = []
        M.run(truth, prog, trace=tr, overwrite=True)
        rom = M.rom_of(prog)
        for t in tr:
            t['rom'] = rom
            t['prog'] = name
            recs.append(t)
    return recs


def _mem_value_at(mem, addr):
    """(value, hit) for the row at integer address ``addr`` in an overwrite snapshot."""
    for row in mem:
        if round(row[M.STACK_K0] / 2.0) == addr:
            return float(row[M.VALUE]), 1.0
    return 0.0, 0.0


def make_batches(recs, truth):
    N = len(recs)
    Lm = max(len(r['mem']) for r in recs)
    Lr = max(len(r['rom']) for r in recs)
    Q = np.zeros((N, M.D))
    MEM = np.zeros((N, Lm, M.D))
    MMASK = np.zeros((N, Lm), bool)
    ROM = np.zeros((N, Lr, M.D))
    RMASK = np.zeros((N, Lr), bool)
    op = np.zeros(N, int)
    arg = np.zeros(N)
    vals = np.zeros((N, 3))
    hits = np.zeros((N, 3))
    writes = np.zeros((N, 3))
    wmask = np.zeros((N, 3))
    delta = np.zeros(N)
    ctrl = np.zeros((N, 3))
    r_vals = np.zeros((N, 3))
    r_hits = np.zeros((N, 3))
    r_writes = np.zeros((N, 3))
    r_delta = np.zeros(N)
    r_ctrl = np.zeros((N, 3))
    for i, r in enumerate(recs):
        Q[i] = r['q']
        m = r['mem']
        MEM[i, :len(m)] = m
        MMASK[i, :len(m)] = True
        ROM[i, :len(r['rom'])] = r['rom']
        RMASK[i, :len(r['rom'])] = True
        op[i] = r['op']
        arg[i] = r['arg']
        vals[i] = r['vals']
        hits[i] = r['hits']
        n = int(truth['n_write'][r['op']])
        writes[i, :n] = r['writes']
        wmask[i, :n] = 1.0
        delta[i] = truth['sp_delta'][r['op']]
        ctrl[i] = truth['ctrl'][r['op']]
        # rival ISA targets
        rr = (r['op'] + 1) % M.N_OPS
        for k in range(3):
            r_vals[i, k], r_hits[i, k] = _mem_value_at(m, r['sp'] - 1 - k)
        u_riv = np.array([r['arg'], *r_vals[i]])
        for c in range(n):
            r_writes[i, c] = truth['W_write'][rr, c] @ u_riv
        r_delta[i] = truth['sp_delta'][rr]
        r_ctrl[i] = truth['ctrl'][rr]
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)  # noqa: E731
    return dict(Q=T(Q), MEM=T(MEM), MMASK=T(MMASK, torch.bool), ROM=T(ROM),
                RMASK=T(RMASK, torch.bool), op=T(op, torch.long), arg=T(arg),
                vals=T(vals), hits=T(hits), writes=T(writes), wmask=T(wmask),
                delta=T(delta), ctrl=T(ctrl), r_vals=T(r_vals), r_hits=T(r_hits),
                r_writes=T(r_writes), r_delta=T(r_delta), r_ctrl=T(r_ctrl))


# ---------------- the differentiable machine ----------------------------------
class SoftMachine(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.p = torch.nn.ParameterDict()
        for k in M.TRAINABLE:
            self.p[k.replace('.', '__')] = torch.nn.Parameter(
                torch.as_tensor(np.asarray(params[k], dtype=np.float32)))
        self.register_buffer('n_write', torch.as_tensor(params['n_write'], dtype=torch.float32))

    def w(self, k):
        return self.p[k.replace('.', '__')]

    def read(self, h, q, mem, mask):
        qq = q @ self.w(f'{h}.W_Q').T + self.w(f'{h}.b_Q')            # [B,2]
        keys = mem @ self.w(f'{h}.W_K').T                              # [B,L,2]
        s = torch.einsum('blk,bk->bl', keys, qq) * BETA
        s = s.masked_fill(~mask, -1e30)
        a = torch.softmax(s, dim=1)
        a = a * mask.any(1, keepdim=True)                              # empty memory -> 0
        v = (mem @ self.w(f'{h}.W_V').T)[..., 0]                       # [B,L]
        return (a * v).sum(1)

    def forward(self, b, idx):
        q, mem, mmask, rom, rmask = (b['Q'][idx], b['MEM'][idx], b['MMASK'][idx],
                                     b['ROM'][idx], b['RMASK'][idx])
        opv = self.read('prog_op', q, rom, rmask)
        arg = self.read('prog_arg', q, rom, rmask)
        reads = torch.stack([self.read(h, q, mem, mmask) for h, _ in M.READ_HEADS], 1)
        return opv, arg, reads

    def dispatch(self, op, u):
        Ww = self.w('W_write')[op]                                     # [B,3,4]
        writes = torch.einsum('bck,bk->bc', Ww, u)
        return writes, self.w('sp_delta')[op], self.w('ctrl')[op]

    def numpy_params(self):
        out = {k: self.w(k).detach().cpu().numpy().astype(float) for k in M.TRAINABLE}
        out['n_write'] = self.n_write.cpu().numpy().astype(float)
        return out


def loss_fn(model, b, idx, arm):
    opv, arg, reads = model(b, idx)
    op = b['op'][idx]
    if arm == 'rival':
        vt, ht, wt, dt, ct = (b[k][idx] for k in ('r_vals', 'r_hits', 'r_writes',
                                                   'r_delta', 'r_ctrl'))
    else:
        vt, ht, wt, dt, ct = (b[k][idx] for k in ('vals', 'hits', 'writes', 'delta', 'ctrl'))
    u = torch.cat([arg[:, None], reads * ht], 1)                       # teacher-forced hits
    writes, delta, ctrl = model.dispatch(op, u)
    wm = b['wmask'][idx]
    mse = torch.nn.functional.mse_loss
    L = (mse(opv, op.float() + 1.0) + mse(arg, b['arg'][idx])
         + mse(reads * ht, vt * ht)
         + ((writes - wt) ** 2 * wm).sum() / wm.sum()
         + mse(delta, dt) + mse(ctrl, ct))
    return L


# ---------------- measurements at a checkpoint ---------------------------------
def l2_from(p, truth, keys):
    return float(np.sqrt(sum(((p[k] - truth[k]) ** 2).sum() for k in keys)))


def pooled(p):
    return np.concatenate([np.abs(np.asarray(p[k])).ravel() for k in M.TRAINABLE])


def hybrid(p, truth, keys):
    """truth everywhere except ``keys``, which come from p."""
    h = dict(truth)
    h.update({k: p[k] for k in keys})
    return h


def measure(p, truth, caps, realism_ref):
    # main correctness: the spec's append-only machine, read scalars re-digitized
    frac, verdicts = M.oracle_score(p, quantize=True)
    frac_raw, _ = M.oracle_score(p, quantize=False)
    frac_ow, _ = M.oracle_score(p, quantize=True, overwrite=True)
    # P2 diagnostic: which component ALONE breaks correctness? Overwrite-in-place
    # machine, so the 1e-6 recency tiebreak (the smallest signal in the system,
    # known to die at any perturbation) does not stand in for the component.
    hyb = {name: M.oracle_score(hybrid(p, truth, keys), quantize=True, overwrite=True)[0]
           for name, keys in (('addr', M.ADDR_KEYS), ('values', M.VALUE_KEYS),
                              ('dispatch', M.DISPATCH_KEYS))}
    sweep = AT.analyze_sweep(p, truth, caps)
    w = pooled(p)
    return dict(oracle=frac, oracle_raw=frac_raw, oracle_overwrite=frac_ow,
                oracle_only=hyb, verdicts=verdicts,
                sweep=[dict(tau=r['tau'], isa=r['isa_score'], addr=r['addr_score'],
                            replay=r['replay_ok'], per_op=r['per_op'],
                            per_head=r['per_head']) for r in sweep],
                l2_heads=l2_from(p, truth, M.HEAD_KEYS),
                l2_dispatch=l2_from(p, truth, M.DISPATCH_KEYS),
                sparsity=float((w < 1e-3).mean()),
                wasserstein_vs_random=(None if realism_ref is None
                                       else float(wasserstein_distance(w, realism_ref))))


def random_params(truth, seed=0):
    rng = np.random.default_rng(seed)
    p = {k: rng.normal(0, 0.1, np.shape(truth[k])) for k in M.TRAINABLE}
    p['n_write'] = truth['n_write'].copy()
    return p


ARMS = {   # name: (task, init, optimizer, lr)
    'random': ('neutral', 'random', 'adamw', 1e-3),
    'neutral_sgd': ('neutral', 'compiled', 'sgd', 1e-8),   # 1e-3 diverges in 3 steps
    'neutral_adam': ('neutral', 'compiled', 'adamw', 1e-3),
    'rival': ('rival', 'compiled', 'adamw', 1e-3),
    'rival_slow': ('rival', 'compiled', 'adamw', 1e-4),
}


def train_arm(arm, init, batches, truth, caps, realism_ref, steps, wd, seed, log):
    torch.manual_seed(seed)
    task, _, optname, lr = ARMS[arm]
    model = SoftMachine(init)
    if optname == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    N = batches['Q'].shape[0]
    g = torch.Generator().manual_seed(seed)
    curve = []
    t0 = time.time()
    for step in range(steps + 1):
        if step in CHECKPOINTS:
            p = model.numpy_params()
            full = loss_fn(model, batches, torch.arange(N), task).item()
            m = measure(p, truth, caps, realism_ref)
            m.update(step=step, loss=full)
            curve.append(m)
            best = max(m['sweep'], key=lambda r: r['isa'])
            o = m['oracle_only']
            log(f"[{arm}] step {step:5d} loss {full:.3e} computes append/overwrite "
                f"{m['oracle']:.2f}/{m['oracle_overwrite']:.2f} (raw {m['oracle_raw']:.2f}) "
                f"only-addr/values/dispatch "
                f"{o['addr']:.2f}/{o['values']:.2f}/{o['dispatch']:.2f} "
                f"isa@best-tau {best['isa']:.3f} (tau {best['tau']}) "
                f"addr {best['addr']:.3f} replay {best['replay']} "
                f"L2 heads {m['l2_heads']:.3f} dispatch {m['l2_dispatch']:.3f} "
                f"[{time.time() - t0:.0f}s]")
        if step == steps:
            break
        idx = torch.randint(0, N, (128,), generator=g)
        opt.zero_grad()
        L = loss_fn(model, batches, idx, task)
        L.backward()
        opt.step()
    return curve, model.numpy_params()


def main(steps=3000, wd=1e-2, seed=0, arms=''):
    truth = M.compile_params()
    caps = AT.reference_captures(truth)
    recs = harvest(truth)
    batches = make_batches(recs, truth)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f'{len(recs)} trace steps from {len(M.ORACLE)} programs; '
        f'max memory rows {batches["MEM"].shape[1]}')
    # step-0 exactness of the soft forward against the reference reads
    model = SoftMachine(truth)
    with torch.no_grad():
        idx = torch.arange(len(recs))
        opv, arg, reads = model(batches, idx)
        e_op = (opv - (batches['op'].float() + 1)).abs().max().item()
        e_arg = (arg - batches['arg']).abs().max().item()
        e_rd = ((reads - batches['vals']) * batches['hits']).abs().max().item()
    log(f'step-0 soft-forward max abs error: opcode {e_op:.2e} arg {e_arg:.2e} reads {e_rd:.2e}')
    assert max(e_op, e_arg, e_rd) < 0.25, 'soft forward is not exact at the compiled point'

    results = dict(config=dict(steps=steps, wd=wd, seed=seed, beta=BETA, arms=ARMS,
                               checkpoints=CHECKPOINTS, taus=list(AT.TAUS)),
                   step0=dict(err_opcode=e_op, err_arg=e_arg, err_reads=e_rd), arms={})
    # realism reference first: the random-init model trained on the neutral task
    curve_r, p_rand = train_arm('random', random_params(truth, seed), batches, truth, caps,
                                None, steps, wd, seed, log)
    ref = pooled(p_rand)
    results['arms']['random'] = curve_r
    np.savez(os.path.join(os.path.dirname(OUT), 'final_random.npz'), **p_rand)
    only = [a for a in arms.split(',') if a]
    if only and os.path.exists(OUT):        # rerun a subset, keep the rest
        prev = json.load(open(OUT))['arms']
        results['arms'].update({k: v for k, v in prev.items() if k not in only})
    for arm in ARMS:
        if arm == 'random' or (only and arm not in only):
            continue
        curve, p_fin = train_arm(arm, truth, batches, truth, caps, ref, steps, wd, seed, log)
        results['arms'][arm] = curve
        np.savez(os.path.join(os.path.dirname(OUT), f'final_{arm}.npz'), **p_fin)
    results['log'] = lines
    json.dump(results, open(OUT, 'w'), indent=1)
    log(f'wrote {OUT}')


if __name__ == '__main__':
    kw = {}
    for a in sys.argv[1:]:
        k, v = a.split('=')
        kw[k] = type(main.__defaults__[['steps', 'wd', 'seed', 'arms'].index(k)])(v)
    main(**kw)
