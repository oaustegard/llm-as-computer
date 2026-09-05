"""Guards: this directory's machine is the blind-recovery compiler, and the
tolerance analyst is exact on the compiled point and blind to pure noise."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'blind_recovery'))
import analyst_tol as AT  # noqa: E402
import compile_artifact as CA  # noqa: E402
import machine as M  # noqa: E402


def test_params_match_blind_recovery_compiler():
    p = M.compile_params()
    for h in M.HEAD_ORDER:
        for m in ('W_Q', 'W_K', 'W_V', 'b_Q'):
            assert np.array_equal(p[f'{h}.{m}'], CA.HEADS[h][m]), (h, m)
    assert np.array_equal(p['W_write'], CA.W_write)
    assert np.array_equal(p['sp_delta'], CA.sp_delta)
    assert np.array_equal(p['ctrl'], CA.ctrl)
    assert np.array_equal(p['n_write'], CA.n_write)


def test_oracle_programs_agree_with_blind_recovery_executor():
    p = M.compile_params()
    for name, prog, expect in M.ORACLE:
        ref, ref_steps, _ = CA.run(prog)
        for ow in (False, True):
            for qz in (False, True):
                got, steps = M.run(p, prog, overwrite=ow, quantize=qz)
                assert got == ref == expect, (name, ow, qz, got, ref)
                assert steps == ref_steps, (name, ow, qz)


def test_failure_is_none_not_a_number():
    p = M.compile_params()
    p = dict(p)
    p['ctrl'] = np.zeros_like(p['ctrl'])          # no HALT anywhere
    got, steps = M.run(p, M.countdown(3)[0], max_steps=50)
    assert got is None and steps <= 50


def test_analyst_exact_on_compiled_point_every_tau():
    truth = M.compile_params()
    caps = AT.reference_captures(truth)
    for r in AT.analyze_sweep(truth, truth, caps):
        assert r['isa_score'] == 1.0 and r['addr_score'] == 1.0 and r['replay_ok'], r['tau']


def test_analyst_degrades_under_noise():
    """invariant: a recovery score of 1.0 is not reachable from pure noise."""
    truth = M.compile_params()
    caps = AT.reference_captures(truth)
    rng = np.random.default_rng(1)
    noisy = {k: (rng.normal(0, 1.0, np.shape(v)) if k in M.TRAINABLE else v)
             for k, v in truth.items()}
    for r in AT.analyze_sweep(noisy, truth, caps):
        assert r['isa_score'] < 1.0 and not r['replay_ok']
