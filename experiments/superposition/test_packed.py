"""Anchors for the packed machine and the superposition-aware analyst.

The identity codebook has to reproduce the axis-aligned machine exactly, and the
pseudo-inverse readout has to be exact whenever the code has full column rank. Those
two facts are what make the compressed results below d = n_features interpretable as
compression damage rather than as a bug in the packing.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyst_sp as AN
import packed as P


def test_identity_codebook_is_the_axis_aligned_machine():
    U = P.identity_codebook()
    H = P.build_heads(U)
    for name, prog, expect in P.programs():
        got, _ = P.run(prog, U, U, H)
        assert got == pytest.approx(expect), name


@pytest.mark.parametrize('d', [24, 32, 51])
def test_pinv_readout_is_exact_at_full_rank(d):
    U = P.codebook(d, 1)
    R = P.readout(U, 'pinv')
    assert np.abs(R @ U.T - np.eye(P.NF)).max() < 1e-9


@pytest.mark.parametrize('d', [24, 32])
def test_packed_machine_computes_when_the_readout_is_exact(d):
    U = P.codebook(d, 1)
    R = P.readout(U, 'pinv')
    H = P.build_heads(R)
    for name, prog, expect in P.programs():
        got, _ = P.run(prog, U, R, H)
        assert got == pytest.approx(expect), f'{name} at d={d}'


@pytest.mark.parametrize('seed', [1, 7])
def test_blind_analyst_recovers_the_isa_at_full_rank(seed):
    U = P.codebook(24, seed)
    R = P.readout(U, 'pinv')
    H = P.build_heads(R)
    out = AN.recover(P.make_artifact(U, R, H, seed=seed, ideal=True))
    assert out['ok'], out.get('why')
    assert out['row_of'] == P.TRUE_ROW_OF
    # the recovered machine, run in the analyst's own interpreter, reproduces the
    # three shipped programs
    assert sorted(out['replays'].values()) == [0, 1, 120]


def test_analyst_never_sees_ground_truth():
    """The artifact carries no names, no codebook, and no permutation key."""
    U = P.codebook(24, 3)
    R = P.readout(U, 'pinv')
    A = P.make_artifact(U, R, P.build_heads(R), seed=3)
    for k in A:
        assert k.startswith(('head', 'ffn_', 'act_')), k
    assert not any(f in ' '.join(A) for f in P.FEATURES)


def test_divergence_is_detected_against_the_reference():
    refs = P.reference_traces()
    U = P.codebook(8, 2)
    R = P.readout(U, 'dot')
    H = P.build_heads(R)
    name, prog, _ = P.programs()[0]
    tr = []
    P.run(prog, U, R, H, max_steps=60, trace=tr)
    assert P.first_divergence(tr, refs[name]) is not None
