"""Guards for the tolerance-mode blind discoverer: exact on the compiled artifact
at every tau, and unable to reach a full recovery from pure noise."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'blind_recovery'))
import blind_decay as B  # noqa: E402
import compile_artifact as CA  # noqa: E402
import machine as M  # noqa: E402


def _compiled_artifact():
    truth = M.compile_params()
    return truth, B.build_artifact(truth, B.compiled_activations(truth))


def test_artifact_matches_the_blind_recovery_compiler():
    """The rebuilt artifact is the August one, tensor for tensor."""
    truth, art = _compiled_artifact()
    rng = np.random.default_rng(B.PERM_SEED)
    perm = rng.permutation(CA.D)
    horder = rng.permutation(len(CA.HEAD_ORDER))
    for new_i, old_i in enumerate(horder):
        h = CA.HEADS[CA.HEAD_ORDER[old_i]]
        for a, b in (('Wq', 'W_Q'), ('Wk', 'W_K'), ('Wv', 'W_V')):
            assert np.array_equal(art[f'head{new_i:02d}_{a}'], h[b][..., perm])
        assert np.array_equal(art[f'head{new_i:02d}_bq'], h['b_Q'])
    assert np.array_equal(art['ffn_A'], CA.W_write.reshape(12, 12))
    assert np.array_equal(art['ffn_B'], np.stack([CA.n_write, CA.sp_delta]).T)
    assert np.array_equal(art['ffn_C'], CA.ctrl)
    progs = [CA.sum_1_to_n(15)[0], CA.countdown(5)[0], CA.rot_jz_nop()[0]]
    for pi, prog in enumerate(progs):
        cap = {s: None for s in B.CAPTURE_STEPS}
        CA.run(prog, capture=cap)
        rom = np.stack([CA.embed_prog(p, o, a) for p, (o, a) in enumerate(prog)])
        assert np.array_equal(art[f'act_rom_{pi}'], rom[..., perm])
        for k, s in enumerate(sorted(cap)):
            if cap[s] is None:
                continue
            assert np.array_equal(art[f'act_query_{pi}_{k}'], cap[s]['q'][perm])
            assert np.array_equal(art[f'act_mem_{pi}_{k}'], cap[s]['stack'][..., perm])


def test_blind_discovery_is_exact_on_the_compiled_artifact_every_tau():
    truth, art = _compiled_artifact()
    for tau in B.TAUS:
        r = B.discover(art, tau, truth)
        assert r['heads_found'] == len(M.HEAD_ORDER), (tau, r['heads_found'])
        assert r['region_split_ok'] and r['law_fit_ok'] and r['opcode_col_found'], tau
        assert r['alignment_survivors'] == [0], (tau, r['alignment_survivors'])
        assert r['replay_ok'], tau
        assert r['opcodes_recovered'] == 12.0, (tau, r['opcodes_recovered'])


def test_blind_discovery_never_reaches_full_recovery_from_noise():
    """invariant: 12/12 is not reachable from weights that carry no machine."""
    truth = M.compile_params()
    acts = B.compiled_activations(truth)
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        noise = {k: rng.normal(0, 1.0, np.shape(truth[k])) for k in M.TRAINABLE}
        noise['n_write'] = truth['n_write'].copy()
        art = B.build_artifact(noise, acts)
        for tau in B.TAUS:
            r = B.discover(art, tau, truth)
            assert r['opcodes_recovered'] < 12.0, (seed, tau)
            assert not r['replay_ok'], (seed, tau)


def test_every_discovery_stage_reports_none_rather_than_a_default():
    """A head bank of zeros has no geometry to find; the discoverer says so."""
    truth = M.compile_params()
    acts = B.compiled_activations(truth)
    dead = {k: np.zeros_like(np.asarray(truth[k], float)) for k in M.TRAINABLE}
    dead['n_write'] = truth['n_write'].copy()
    art = B.build_artifact(dead, acts)
    r = B.discover(art, 0.2, truth)
    assert r['heads_found'] == 0
    assert r['stage_failed'] == 'region_split'
    assert r['alignment_survivors'] is None
    assert r['law'] is None
    assert r['opcodes_recovered'] == 0.0
