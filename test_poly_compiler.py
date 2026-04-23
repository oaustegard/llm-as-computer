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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
