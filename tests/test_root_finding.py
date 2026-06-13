"""
test_root_finding.py
Tests for root-finding numerical methods:
  - bisection_method
  - newton_raphson_method
  - secant_method

Functions live in src/numerical_methods.py.
Run from the HydroSense-Kenya/ root with:
    pytest tests/test_root_finding.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import pytest
from numerical_methods import bisection_method, newton_raphson_method, secant_method


# ──────────────────────────────────────────────
# Test input functions (passed into the solvers)
# These are NOT redefinitions of anything in
# numerical_methods.py — they are the equations
# whose roots we want to find.
# ──────────────────────────────────────────────

# f(x) = x^2 - 4   →  roots at x = ±2
def f_quadratic(x):
    return x**2 - 4

def df_quadratic(x):
    return 2 * x

# f(x) = x^3 - x - 2   →  root near x ≈ 1.5214
def f_cubic(x):
    return x**3 - x - 2

def df_cubic(x):
    return 3 * x**2 - 1

# f(x) = cos(x) - x   →  root near x ≈ 0.7391 (Dottie number)
def f_trig(x):
    return math.cos(x) - x

def df_trig(x):
    return -math.sin(x) - 1

# f(x) = e^x - 3   →  root at x = ln(3) ≈ 1.0986
def f_exp(x):
    return math.exp(x) - 3

def df_exp(x):
    return math.exp(x)

# f(x) = x   →  root exactly at x = 0 (midpoint of [-1, 1])
def f_linear(x):
    return x

def df_linear(x):
    return 1.0


TOLERANCE = 1e-4


# ══════════════════════════════════════════════
# BISECTION METHOD TESTS
# ══════════════════════════════════════════════

class TestBisectionMethod:

    # ── Correctness ───────────────────────────

    def test_quadratic_positive_root(self):
        """Finds the positive root of x^2 - 4 = 0 (should be ~2.0)."""
        root, iters = bisection_method(f_quadratic, a=0, b=3)
        assert abs(root - 2.0) < TOLERANCE

    def test_quadratic_negative_root(self):
        """Finds the negative root of x^2 - 4 = 0 (should be ~-2.0)."""
        root, iters = bisection_method(f_quadratic, a=-3, b=0)
        assert abs(root - (-2.0)) < TOLERANCE

    def test_cubic_root(self):
        """Root of x^3 - x - 2 near x ≈ 1.5214."""
        root, iters = bisection_method(f_cubic, a=1, b=2)
        assert abs(f_cubic(root)) < TOLERANCE

    def test_trig_root(self):
        """Root of cos(x) - x near x ≈ 0.7391."""
        root, iters = bisection_method(f_trig, a=0, b=1)
        assert abs(f_trig(root)) < TOLERANCE

    def test_exp_root(self):
        """Root of e^x - 3 at x = ln(3) ≈ 1.0986."""
        root, iters = bisection_method(f_exp, a=0, b=2)
        assert abs(root - math.log(3)) < TOLERANCE

    def test_root_near_endpoint(self):
        """Root at x=2 when bracket is [1.9, 3] — root sits close to one end."""
        root, iters = bisection_method(f_quadratic, a=1.9, b=3)
        assert abs(root - 2.0) < TOLERANCE

    def test_root_at_midpoint(self):
        """
        f(x) = x has root at x=0, which is the exact midpoint of [-1, 1].
        Bisection should find it in very few iterations.
        """
        root, iters = bisection_method(f_linear, a=-1, b=1)
        assert abs(root) < TOLERANCE

    # ── Return value ──────────────────────────

    def test_returns_tuple_with_iteration_count(self):
        """Return value must be a (root, iterations) tuple with iterations >= 1."""
        result = bisection_method(f_quadratic, a=0, b=3)
        assert isinstance(result, tuple) and len(result) == 2
        _, iters = result
        assert iters >= 1

    def test_root_satisfies_equation(self):
        """f(root) should be near zero — residual check."""
        root, _ = bisection_method(f_cubic, a=1, b=2)
        assert abs(f_cubic(root)) < TOLERANCE

    def test_tighter_tolerance(self):
        """With tol=1e-8, root should be within 1e-7 of the true value."""
        root, _ = bisection_method(f_quadratic, a=0, b=3, tol=1e-8)
        assert abs(root - 2.0) < 1e-7

    # ── Error handling ────────────────────────

    def test_raises_when_same_sign_at_endpoints(self):
        """
        f(0)=−4 and f(1)=−3 are both negative for x^2−4 on [0,1].
        bisection_method must raise ValueError.
        """
        with pytest.raises(ValueError):
            bisection_method(f_quadratic, a=0, b=1)

    # ── Cross-method consistency ──────────────

    def test_agrees_with_newton_raphson(self):
        """Bisection and Newton-Raphson should find the same root of x^2-4."""
        root_b, _ = bisection_method(f_quadratic, a=0, b=3)
        root_n, _ = newton_raphson_method(f_quadratic, df_quadratic, x0=3)
        assert abs(root_b - root_n) < TOLERANCE

    def test_agrees_with_secant(self):
        """Bisection and Secant should find the same root of x^2-4."""
        root_b, _ = bisection_method(f_quadratic, a=0, b=3)
        root_s, _ = secant_method(f_quadratic, x0=3, x1=4)
        assert abs(root_b - root_s) < TOLERANCE


# ══════════════════════════════════════════════
# NEWTON-RAPHSON METHOD TESTS
# ══════════════════════════════════════════════

class TestNewtonRaphsonMethod:

    # ── Correctness ───────────────────────────

    def test_quadratic_root(self):
        """Finds positive root of x^2 - 4 starting near x=3."""
        root, iters = newton_raphson_method(f_quadratic, df_quadratic, x0=3)
        assert abs(root - 2.0) < TOLERANCE

    def test_quadratic_root_negative_x0(self):
        """Starting from a negative x0 should still converge to the positive root
        if the function shape guides it there, or the negative root."""
        root, iters = newton_raphson_method(f_quadratic, df_quadratic, x0=-3)
        assert abs(root - (-2.0)) < TOLERANCE

    def test_cubic_root(self):
        """Root of x^3 - x - 2 starting at x0=2."""
        root, iters = newton_raphson_method(f_cubic, df_cubic, x0=2)
        assert abs(f_cubic(root)) < TOLERANCE

    def test_trig_root(self):
        """Root of cos(x) - x (Dottie number) starting at x0=0.5."""
        root, iters = newton_raphson_method(f_trig, df_trig, x0=0.5)
        assert abs(f_trig(root)) < TOLERANCE

    def test_exp_root(self):
        """Root of e^x - 3 starting at x0=1."""
        root, iters = newton_raphson_method(f_exp, df_exp, x0=1)
        assert abs(root - math.log(3)) < TOLERANCE

    # ── Convergence speed ─────────────────────

    def test_converges_fast(self):
        """Newton-Raphson is quadratically convergent; expect fewer than 20 iterations."""
        _, iters = newton_raphson_method(f_quadratic, df_quadratic, x0=3)
        assert iters < 20

    # ── Return value ──────────────────────────

    def test_returns_tuple_with_iteration_count(self):
        """Return value must be a (root, iterations) tuple with iterations >= 1."""
        result = newton_raphson_method(f_quadratic, df_quadratic, x0=3)
        assert isinstance(result, tuple) and len(result) == 2
        _, iters = result
        assert iters >= 1

    def test_root_satisfies_equation(self):
        """f(root) should be near zero — residual check."""
        root, _ = newton_raphson_method(f_trig, df_trig, x0=0.5)
        assert abs(f_trig(root)) < TOLERANCE

    # ── Error handling ────────────────────────

    def test_raises_when_derivative_is_zero(self):
        """
        df_quadratic(0) = 0, so starting at x0=0 should raise ZeroDivisionError
        because the tangent line is flat and the method cannot proceed.
        """
        with pytest.raises(ZeroDivisionError):
            newton_raphson_method(f_quadratic, df_quadratic, x0=0)

    # ── Cross-method consistency ──────────────

    def test_agrees_with_secant(self):
        """Newton-Raphson and Secant should find the same root of x^2-4."""
        root_n, _ = newton_raphson_method(f_quadratic, df_quadratic, x0=3)
        root_s, _ = secant_method(f_quadratic, x0=3, x1=4)
        assert abs(root_n - root_s) < TOLERANCE


# ══════════════════════════════════════════════
# SECANT METHOD TESTS
# ══════════════════════════════════════════════

class TestSecantMethod:

    # ── Correctness ───────────────────────────

    def test_quadratic_root(self):
        """Finds positive root of x^2 - 4 with x0=3, x1=4."""
        root, iters = secant_method(f_quadratic, x0=3, x1=4)
        assert abs(root - 2.0) < TOLERANCE

    def test_cubic_root(self):
        """Root of x^3 - x - 2 with x0=1, x1=2."""
        root, iters = secant_method(f_cubic, x0=1, x1=2)
        assert abs(f_cubic(root)) < TOLERANCE

    def test_trig_root(self):
        """Root of cos(x) - x with x0=0, x1=1."""
        root, iters = secant_method(f_trig, x0=0, x1=1)
        assert abs(f_trig(root)) < TOLERANCE

    def test_exp_root(self):
        """Root of e^x - 3 with x0=1, x1=2."""
        root, iters = secant_method(f_exp, x0=1, x1=2)
        assert abs(root - math.log(3)) < TOLERANCE

    def test_different_starting_guesses_same_root(self):
        """Two different initial guess pairs should converge to the same root."""
        root_a, _ = secant_method(f_quadratic, x0=1, x1=3)
        root_b, _ = secant_method(f_quadratic, x0=2.5, x1=4)
        assert abs(root_a - root_b) < TOLERANCE

    # ── Return value ──────────────────────────

    def test_returns_tuple_with_iteration_count(self):
        """Return value must be a (root, iterations) tuple with iterations >= 1."""
        result = secant_method(f_quadratic, x0=3, x1=4)
        assert isinstance(result, tuple) and len(result) == 2
        _, iters = result
        assert iters >= 1

    def test_root_satisfies_equation(self):
        """f(root) should be near zero — residual check."""
        root, _ = secant_method(f_cubic, x0=1, x1=2)
        assert abs(f_cubic(root)) < TOLERANCE

    # ── Error handling ────────────────────────

    def test_raises_when_secant_line_is_flat(self):
        """
        If f(x0) == f(x1), the denominator (f_x1 - f_x0) is zero.
        The method must raise ZeroDivisionError.
        f(x) = x^2 - 4: f(2) = 0 and f(-2) = 0, so f(x0)==f(x1).
        """
        with pytest.raises(ZeroDivisionError):
            secant_method(f_quadratic, x0=2, x1=-2)

    # ── Cross-method consistency ──────────────

    def test_agrees_with_bisection(self):
        """Secant and Bisection should find the same root of the cubic."""
        root_s, _ = secant_method(f_cubic, x0=1, x1=2)
        root_b, _ = bisection_method(f_cubic, a=1, b=2)
        assert abs(root_s - root_b) < TOLERANCE