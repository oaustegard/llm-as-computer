"""Anchors for the categorical-address machine and the machine-generic trainer.

The comparison in `addressing_sweep.py` is only meaningful if `packed_cat` is the
same machine as `packed` with the addressing swapped, and if `learned_generic` fits
`packed` the way `learned` does. These are those two equivalences.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import learned as L
import learned_generic as LG
import packed as P
import packed_cat as C

TRACE_FIELDS = ('ip', 'sp', 'n_rows', 'op', 'arg', 'vals')


def _exact(M):
    U = M.identity_codebook()
    return U, M.build_heads(U)


@pytest.mark.parametrize('overwrite', [False, True])
def test_categorical_traces_match_the_parabolic_machine(overwrite):
    Up, Hp = _exact(P)
    Uc, Hc = _exact(C)
    for (name, prog, expect), (_n2, prog2, _e2) in zip(P.programs(), C.programs()):
        assert prog == prog2
        tp, tc = [], []
        gp, _ = P.run(prog, Up, Up, Hp, trace=tp, overwrite=overwrite)
        gc, _ = C.run(prog, Uc, Uc, Hc, trace=tc, overwrite=overwrite)
        assert gp == pytest.approx(expect), name
        assert gc == pytest.approx(expect), name
        assert len(tp) == len(tc), name
        for s, (a, b) in enumerate(zip(tp, tc)):
            assert tuple(a[f] for f in TRACE_FIELDS) == tuple(b[f] for f in TRACE_FIELDS), \
                f'{name} step {s}'
            assert a.get('wrote') == b.get('wrote'), f'{name} step {s} writes'


def test_returns_are_the_four_oracle_values():
    Uc, Hc = _exact(C)
    got = [C.run(prog, Uc, Uc, Hc, overwrite=True)[0] for _n, prog, _e in C.programs()]
    assert got == [120.0, 0.0, 1.0, 5050.0]


def test_address_spans_cover_every_program():
    """A_ROM and A_ST are measured, not assumed: fail if a program outgrows them."""
    Up, Hp = _exact(P)
    max_rom = max(len(prog) for _n, prog, _e in P.programs())
    max_addr = 0
    for _n, prog, _e in P.programs():
        tr = []
        P.run(prog, Up, Up, Hp, trace=tr, overwrite=True)
        for rec in tr:
            for addr, _v, _w in rec.get('wrote', ()):
                max_addr = max(max_addr, addr)
    assert C.A_ROM == max_rom
    assert C.A_ST == max_addr + 1


def test_onehot_score_gap_is_one_on_every_read():
    """Winner scores 1 and every other row 0, so the gap is exactly 1 -- the same gap
    the parabolic machine gets from its keys. Reads that hit no live row score 0
    everywhere and are discarded by the address check; they are excluded here."""
    eye = np.eye(C.NF)
    H = C.build_heads(eye)
    seen = 0
    for _name, prog, _e in C.programs():
        rom = [C.embed_prog(eye, p, o, a) for p, (o, a) in enumerate(prog)]
        tr = []
        C.run(prog, eye, eye, H, trace=tr, overwrite=True)
        stack, saddr = [], []
        for rec in tr:
            q = C.embed_state(eye, rec['ip'], rec['sp'])
            for h, region in C.ARGMAX_HEADS:
                pool = rom if region == 'rom' else stack
                if len(pool) < 2:
                    continue
                s = LG._ref_scores(C, h, q, pool)
                order = np.sort(s)[::-1]
                assert set(np.unique(s)) <= {0.0, 1.0}, h
                if order[0] == 0.0:
                    continue          # miss: no live row at that address
                assert order[0] == 1.0 and order[1] == 0.0, (h, order[:2])
                seen += 1
            for addr, val, wo in rec.get('wrote', ()):
                y = C.embed_stack(eye, addr, val, wo)
                if addr in saddr:
                    stack[saddr.index(addr)] = y
                else:
                    stack.append(y)
                    saddr.append(addr)
    assert seen > 1000


def test_generic_harvest_reproduces_the_hardcoded_one_on_packed():
    g = LG.harvest(P)
    h = L.harvest()
    assert g[6] == 0                       # no degenerate pair on the parabolic machine
    assert len(g[0]) == len(h[0])
    assert np.allclose(np.sort(g[5]), np.sort(h[5]))


@pytest.mark.parametrize('M', [P, C], ids=['parabolic', 'categorical'])
def test_identity_code_is_an_exact_solution_at_full_width(M):
    data = LG.harvest(M)
    _code, fin = LG.train(M, M.NF, 0, data, iters=5, init=np.eye(M.NF))
    assert fin['margin'] == pytest.approx(0.0, abs=1e-12)
    assert fin['tol'] == pytest.approx(0.0, abs=1e-12)
    assert fin['viol_structural'] == pytest.approx(0.0)


def test_dense_and_used_feature_sets_are_disjoint_from_the_indicators():
    for M in (P, C):
        assert all(not f.startswith('op_') for f in M.DENSE)
        assert set(M.DENSE) <= set(M.FEATURES)
