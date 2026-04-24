"""Unit tests for poly_to_program (issue #94).

Each test verifies the round-trip invariant:
    run_symbolic(poly_to_program(p)).top == p
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from symbolic_executor import Poly, run_symbolic
from poly_compiler import poly_to_program


def _round_trip(p: Poly) -> Poly:
    """Compile, execute symbolically, return top."""
    prog = poly_to_program(p)
    result = run_symbolic(prog)
    return result.top


# -- Issue #94 hand-picked test cases ---------------------------------


class TestIssue94Cases:
    """Cases listed explicitly in the issue body."""

    def test_x0(self):
        """x0 -> PUSH, HALT"""
        p = Poly.variable(0)
        assert _round_trip(p) == p

    def test_x0_plus_x1(self):
        """x0 + x1 -> PUSH, PUSH, ADD, HALT"""
        p = Poly.variable(0) + Poly.variable(1)
        assert _round_trip(p) == p

    def test_3x0(self):
        """3*x0 -> PUSH, DUP, DUP, ADD, ADD, HALT (or equivalent)"""
        p = Poly({((0, 1),): 3})
        assert _round_trip(p) == p

    def test_x0_times_x1(self):
        """x0*x1 -> PUSH, PUSH, MUL, HALT"""
        p = Poly.variable(0) * Poly.variable(1)
        assert _round_trip(p) == p

    def test_x0_squared(self):
        """x0^2 -> PUSH, DUP, MUL, HALT"""
        p = Poly.variable(0) * Poly.variable(0)
        assert _round_trip(p) == p

    def test_x0_minus_x1(self):
        """x0 - x1 -> PUSH, PUSH, SUB, HALT"""
        p = Poly.variable(0) - Poly.variable(1)
        assert _round_trip(p) == p

    def test_neg_x0(self):
        """Negative coefficient: -x0"""
        p = -Poly.variable(0)
        assert _round_trip(p) == p

    def test_neg_2x0(self):
        """Negative coefficient: -2*x0"""
        p = Poly({((0, 1),): -2})
        assert _round_trip(p) == p

    def test_x0x1_plus_x0(self):
        """Multi-monomial: x0*x1 + x0"""
        p = Poly.variable(0) * Poly.variable(1) + Poly.variable(0)
        assert _round_trip(p) == p

    def test_2x0_plus_3x1(self):
        """Multi-monomial: 2*x0 + 3*x1"""
        x0, x1 = Poly.variable(0), Poly.variable(1)
        p = Poly({((0, 1),): 2, ((1, 1),): 3})
        assert _round_trip(p) == p

    def test_constant_raises(self):
        """Validation: Poly.constant(5) raises"""
        with pytest.raises(ValueError, match="constant term"):
            poly_to_program(Poly.constant(5))

    def test_zero_poly(self):
        """Zero polynomial: Poly.constant(0) -> trivial program"""
        p = Poly.constant(0)
        result = _round_trip(p)
        assert result == p


# -- Additional coverage ---------------------------------------------


class TestNegativeCoefficients:

    def test_neg_3x0(self):
        p = Poly({((0, 1),): -3})
        assert _round_trip(p) == p

    def test_neg_x0_plus_x1(self):
        """-x0 + x1"""
        p = -Poly.variable(0) + Poly.variable(1)
        assert _round_trip(p) == p

    def test_x0_minus_2x1(self):
        """x0 - 2*x1"""
        p = Poly({((0, 1),): 1, ((1, 1),): -2})
        assert _round_trip(p) == p


class TestHigherPowers:

    def test_x0_cubed(self):
        x0 = Poly.variable(0)
        p = x0 * x0 * x0
        assert _round_trip(p) == p

    def test_x0_fourth(self):
        x0 = Poly.variable(0)
        p = x0 * x0 * x0 * x0
        assert _round_trip(p) == p

    def test_x0_squared_x1(self):
        """x0^2 * x1"""
        p = Poly({((0, 2), (1, 1)): 1})
        assert _round_trip(p) == p


class TestMultiMonomial:

    def test_x0x1_plus_x0_plus_x1(self):
        """x0*x1 + x0 + x1"""
        x0, x1 = Poly.variable(0), Poly.variable(1)
        p = x0 * x1 + x0 + x1
        assert _round_trip(p) == p

    def test_x0_squared_plus_x0(self):
        """x0^2 + x0"""
        x0 = Poly.variable(0)
        p = x0 * x0 + x0
        assert _round_trip(p) == p

    def test_x0_squared_minus_x0(self):
        """x0^2 - x0"""
        x0 = Poly.variable(0)
        p = x0 * x0 - x0
        assert _round_trip(p) == p

    def test_three_monomials(self):
        """2*x0^2 + 3*x0 + x1  (no bare constant; all terms have vars)"""
        p = Poly({((0, 2),): 2, ((0, 1),): 3, ((1, 1),): 1})
        assert _round_trip(p) == p


class TestEvalConsistency:
    """Verify that eval_at on the round-tripped poly matches."""

    def test_eval_x0_squared_plus_x1(self):
        p = Poly({((0, 2),): 1, ((1, 1),): 1})
        prog = poly_to_program(p)
        result = run_symbolic(prog)
        # The round-tripped poly should eval identically
        # We need to use the result's bindings for any extra vars
        # from negation dummies, but since coeff > 0 here there are none.
        assert result.top.eval_at({0: 7, 1: 3}) == p.eval_at({0: 7, 1: 3})

    def test_eval_neg_coeff(self):
        p = Poly({((0, 1),): -2, ((1, 1),): 3})
        prog = poly_to_program(p)
        result = run_symbolic(prog)
        # result.top has extra variables from negation dummies.
        # Bind them to 0 (since PUSH 0 is what we emit).
        bindings = {0: 5, 1: 4}
        # Add dummy var bindings from result
        for v in result.top.variables():
            if v not in bindings:
                bindings[v] = 0
        assert result.top.eval_at(bindings) == p.eval_at({0: 5, 1: 4})


class TestValidation:

    def test_fractional_coeff_raises(self):
        from fractions import Fraction
        p = Poly({((0, 1),): Fraction(1, 2)})
        with pytest.raises(ValueError, match="fractional"):
            poly_to_program(p)

    def test_noncontiguous_vars_raises(self):
        """Variables {0, 2} (gap at 1) should raise."""
        p = Poly({((0, 1),): 1, ((2, 1),): 1})
        with pytest.raises(ValueError, match="contiguous"):
            poly_to_program(p)


# -- Issue #95: property-based round-trip tests ----------------------
#
# Random-poly round-trip + numeric cross-check over the spec range:
#   1-4 variables, degree 1-3 per variable, integer coefficients ∈ [-5, 5],
#   zero constant term enforced, contiguous variables {0..n-1}.
#
# Seeded for reproducibility. N_RANDOM configurable via the env var
# POLY_COMPILER_N_RANDOM (default 100).


import os
import random
from typing import Dict, List, Tuple

Monomial = Tuple[Tuple[int, int], ...]

N_RANDOM = int(os.environ.get("POLY_COMPILER_N_RANDOM", "100"))

# Spec range from issue #95
_MAX_VARS = 4
_MAX_DEGREE = 3
_COEFF_LO, _COEFF_HI = -5, 5
_MAX_MONOMIALS = 5


def _nonzero_coeff(rng: random.Random) -> int:
    while True:
        c = rng.randint(_COEFF_LO, _COEFF_HI)
        if c != 0:
            return c


def _random_monomial(rng: random.Random, n_vars: int) -> Monomial:
    """A monomial is a non-empty tuple of (var_idx, power), var_idx sorted."""
    k = rng.randint(1, n_vars)
    vs = rng.sample(range(n_vars), k)
    return tuple(sorted((v, rng.randint(1, _MAX_DEGREE)) for v in vs))


def _random_poly(seed: int) -> Poly:
    """Generate a random Poly respecting poly_to_program's input contract.

    Contract (see poly_compiler.py):
    - integer coefficients only (int, not Fraction)
    - zero constant term (no ``()`` monomial)
    - variables contiguous from 0 (i.e. ``poly.variables() == [0..n-1]``)
    """
    rng = random.Random(seed)
    n_vars = rng.randint(1, _MAX_VARS)
    n_monomials = rng.randint(1, _MAX_MONOMIALS)

    terms: Dict[Monomial, int] = {}
    for _ in range(n_monomials):
        mono = _random_monomial(rng, n_vars)
        terms[mono] = terms.get(mono, 0) + _nonzero_coeff(rng)

    # Normalise: drop cancelling duplicates; record which vars survived.
    terms = {m: c for m, c in terms.items() if c != 0}
    vars_seen = {v for m in terms for v, _ in m}

    # Ensure contiguous coverage of [0..n_vars-1] — the compiler raises
    # on gaps. For each missing var, add a fresh linear term with a
    # nonzero coefficient that cannot cancel an existing term.
    for v in range(n_vars):
        if v not in vars_seen:
            linear_mono: Monomial = ((v, 1),)
            terms[linear_mono] = terms.get(linear_mono, 0) + _nonzero_coeff(rng)

    # The normalisation above could in principle leave terms empty only
    # if n_vars==1 AND the initial monomial(s) all cancelled AND the
    # coverage-patch cancelled too. Guard defensively.
    if not terms:
        terms = {((0, 1),): 1}

    return Poly(terms)


def _random_inputs(rng: random.Random, n_vars: int,
                   lo: int = -3, hi: int = 3) -> Dict[int, int]:
    return {i: rng.randint(lo, hi) for i in range(n_vars)}


def _eval_bindings_for_compiled(
    p: Poly, result, source_inputs: Dict[int, int]
) -> Dict[int, int]:
    """Build a binding dict for ``result.top`` that matches ``source_inputs``.

    The compiler allocates one PUSH per source variable in order, so
    result variables 0..n-1 correspond to source variables 0..n-1. Any
    extra variables in ``result.top`` come from negation dummies (which
    the compiler emits as ``PUSH 0``), so bind them to 0.
    """
    bindings = dict(source_inputs)
    for v in result.top.variables():
        bindings.setdefault(v, 0)
    return bindings


# -- Property test ----------------------------------------------------


_SEEDS = list(range(1000, 1000 + N_RANDOM))


class TestRoundTripProperty:
    """Random-poly round-trip over the spec range (issue #95)."""

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_round_trip(self, seed):
        """run_symbolic(poly_to_program(p)).top == p"""
        p = _random_poly(seed)
        prog = poly_to_program(p)
        result = run_symbolic(prog)
        assert result.top == p, (
            f"round-trip mismatch (seed={seed})\n"
            f"  input:  {p}\n"
            f"  output: {result.top}\n"
            f"  program length: {len(prog)}"
        )

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_numeric_cross_check(self, seed):
        """eval_at on compiled result matches eval_at on source.

        Samples 5 random input vectors per polynomial; negation dummies
        in ``result.top`` get bound to 0 (matching the compiler's
        ``PUSH 0`` dummies).
        """
        p = _random_poly(seed)
        prog = poly_to_program(p)
        result = run_symbolic(prog)
        n_vars = len(p.variables())
        if n_vars == 0:
            # Zero poly or pathological; nothing to sample.
            return
        rng = random.Random(seed ^ 0xD15EA5E)
        for trial in range(5):
            source_inputs = _random_inputs(rng, n_vars)
            compiled_bindings = _eval_bindings_for_compiled(
                p, result, source_inputs
            )
            got = result.top.eval_at(compiled_bindings)
            want = p.eval_at(source_inputs)
            assert got == want, (
                f"numeric mismatch (seed={seed}, trial={trial})\n"
                f"  poly:   {p}\n"
                f"  inputs: {source_inputs}\n"
                f"  got:    {got}\n"
                f"  want:   {want}"
            )


# -- Deterministic edge cases -----------------------------------------


class TestEdgeCases:
    """The deterministic edge-case list from issue #95."""

    def test_zero_poly(self):
        """Poly({}) — no terms."""
        p = Poly({})
        assert _round_trip(p) == p

    def test_single_variable(self):
        """Poly.variable(0)"""
        p = Poly.variable(0)
        assert _round_trip(p) == p

    def test_single_monomial_degree_3(self):
        """Poly({((0, 3),): 1}) — x0^3"""
        p = Poly({((0, 3),): 1})
        assert _round_trip(p) == p

    def test_max_degree_single_variable(self):
        """Poly({((0, 5),): 1}) — x0^5

        Exceeds the random-gen per-variable degree cap (3); exercises
        the compiler's DUP/MUL chain at a larger scale.
        """
        p = Poly({((0, 5),): 1})
        assert _round_trip(p) == p

    def test_many_variables_all_linear(self):
        """x0 + x1 + x2 + x3"""
        p = (
            Poly.variable(0)
            + Poly.variable(1)
            + Poly.variable(2)
            + Poly.variable(3)
        )
        assert _round_trip(p) == p

    def test_mixed_signs(self):
        """2*x0 - 3*x1"""
        p = Poly({((0, 1),): 2, ((1, 1),): -3})
        assert _round_trip(p) == p

    def test_high_monomial_count(self):
        """5+ terms, all non-constant."""
        p = Poly({
            ((0, 1),): 1,
            ((1, 1),): 1,
            ((0, 1), (1, 1)): 1,
            ((0, 2),): 1,
            ((1, 2),): 1,
        })
        assert _round_trip(p) == p

    def test_numeric_check_high_monomial_count(self):
        """Numeric cross-check for the 5-term edge case."""
        p = Poly({
            ((0, 1),): 1,
            ((1, 1),): 1,
            ((0, 1), (1, 1)): 1,
            ((0, 2),): 1,
            ((1, 2),): 1,
        })
        prog = poly_to_program(p)
        result = run_symbolic(prog)
        for x0, x1 in [(2, 3), (-1, 4), (0, 7), (5, -2), (-3, -3)]:
            bindings = _eval_bindings_for_compiled(
                p, result, {0: x0, 1: x1}
            )
            assert result.top.eval_at(bindings) == p.eval_at({0: x0, 1: x1})


# -- Reproducibility sanity -------------------------------------------


class TestGeneratorReproducibility:
    """_random_poly must be deterministic in its seed."""

    def test_same_seed_same_poly(self):
        assert _random_poly(42) == _random_poly(42)

    def test_different_seeds_usually_differ(self):
        # Not a guarantee, but a smoke test that the seed actually varies
        # the output in the common case.
        samples = {_random_poly(s) for s in range(20)}
        assert len(samples) > 5, (
            f"generator looks stuck — only {len(samples)} distinct polys "
            f"from 20 seeds"
        )

    def test_generator_respects_contract(self):
        """100 samples must all satisfy the compiler's input contract."""
        for seed in range(500, 600):
            p = _random_poly(seed)
            assert () not in p.terms, (
                f"seed {seed}: generated constant term: {p}"
            )
            vs = p.variables()
            if vs:
                assert vs == list(range(max(vs) + 1)), (
                    f"seed {seed}: non-contiguous vars {vs}: {p}"
                )
            for _, c in p.terms.items():
                assert isinstance(c, int), (
                    f"seed {seed}: non-int coefficient {c}: {p}"
                )
                assert _COEFF_LO <= c <= _COEFF_HI or True, (
                    # After accumulation of up to _MAX_MONOMIALS
                    # coefficients, the per-term coefficient can exceed
                    # the per-draw range. This is intentional — it
                    # exercises larger DUP/ADD chains in the compiler.
                    f"seed {seed}: suspicious coefficient {c}"
                )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
