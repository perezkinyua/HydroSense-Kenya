"""
test_linear_solver.py
Tests for the linear system solver:
  - gaussian_elimination(A, b)  →  returns x such that Ax = b

Function lives in src/numerical_methods.py.
Run from the HydroSense-Kenya/ root with:
    pytest tests/test_linear_systems.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import pytest
from numerical_methods import gaussian_elimination # pyright: ignore[reportMissingImports]


TOLERANCE = 1e-6


# ══════════════════════════════════════════════
# GAUSSIAN ELIMINATION TESTS
# ══════════════════════════════════════════════

class TestGaussianElimination:

    # ── Basic correctness ─────────────────────

    def test_2x2_simple(self):
        """
        2x + y  = 5
         x + 3y = 10
        Solution: x=1, y=3
        """
        A = [[2, 1],
             [1, 3]]
        b = [5, 10]
        x = gaussian_elimination(A, b)
        assert abs(x[0] - 1.0) < TOLERANCE
        assert abs(x[1] - 3.0) < TOLERANCE

    def test_3x3_standard(self):
        """
        2x +  y - z  =  8
        -3x - y + 2z = -11
        -2x + y + 2z = -3
        Solution: x=2, y=3, z=-1
        """
        A = [[ 2,  1, -1],
             [-3, -1,  2],
             [-2,  1,  2]]
        b = [8, -11, -3]
        x = gaussian_elimination(A, b)
        assert abs(x[0] - 2.0) < TOLERANCE
        assert abs(x[1] - 3.0) < TOLERANCE
        assert abs(x[2] - (-1.0)) < TOLERANCE

    def test_identity_matrix(self):
        """Ax = b with A = I should return x = b exactly."""
        A = [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        b = [4.0, -2.0, 7.5]
        x = gaussian_elimination(A, b)
        for xi, bi in zip(x, b):
            assert abs(xi - bi) < TOLERANCE

    def test_diagonal_matrix(self):
        """
        3x = 9
        5y = 20
        2z = 6
        Solution: x=3, y=4, z=3
        """
        A = [[3, 0, 0],
             [0, 5, 0],
             [0, 0, 2]]
        b = [9, 20, 6]
        x = gaussian_elimination(A, b)
        assert abs(x[0] - 3.0) < TOLERANCE
        assert abs(x[1] - 4.0) < TOLERANCE
        assert abs(x[2] - 3.0) < TOLERANCE

    # ── Verify solution by back-substitution ──

    def test_residual_2x2(self):
        """The computed x should satisfy Ax = b (residual check)."""
        A = [[4, 1],
             [2, 3]]
        b = [9, 8]
        x = gaussian_elimination(A, b)
        for i in range(len(b)):
            row_sum = sum(A[i][j] * x[j] for j in range(len(x)))
            assert abs(row_sum - b[i]) < TOLERANCE

    def test_residual_3x3(self):
        """Residual check for a 3×3 system."""
        A = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 10]]   # slightly perturbed so it's non-singular
        b = [14, 32, 53]
        x = gaussian_elimination(A, b)
        for i in range(len(b)):
            row_sum = sum(A[i][j] * x[j] for j in range(len(x)))
            assert abs(row_sum - b[i]) < TOLERANCE

    # ── Partial pivoting behaviour ─────────────

    def test_pivoting_needed(self):
        """
        System where the first pivot is 0 — requires row swapping.
        0x + 2y = 4
        3x + 4y = 10
        Solution: x=2/3, y=2
        """
        A = [[0, 2],
             [3, 4]]
        b = [4, 10]
        x = gaussian_elimination(A, b)
        assert abs(x[0] - (2 / 3)) < TOLERANCE
        assert abs(x[1] - 2.0) < TOLERANCE

    def test_large_coefficient_difference(self):
        """
        System with coefficients that differ by orders of magnitude —
        partial pivoting prevents catastrophic cancellation.
        """
        A = [[0.0001, 1],
             [1,      1]]
        b = [1.0001, 2]
        # True solution: x ≈ 1, y ≈ 1
        x = gaussian_elimination(A, b)
        assert abs(x[0] - 1.0) < 1e-4
        assert abs(x[1] - 1.0) < 1e-4

    # ── Return type ────────────────────────────

    def test_returns_list_of_correct_length(self):
        """Output should be a list of the same length as b."""
        A = [[2, 1], [5, 7]]
        b = [11, 13]
        x = gaussian_elimination(A, b)
        assert isinstance(x, list)
        assert len(x) == len(b)

    def test_4x4_system(self):
        """
        Slightly larger system to confirm scalability.
        Uses A = 2I so solution is x = b/2.
        """
        n = 4
        A = [[2 if i == j else 0 for j in range(n)] for i in range(n)]
        b = [2.0, 4.0, 6.0, 8.0]
        x = gaussian_elimination(A, b)
        expected = [1.0, 2.0, 3.0, 4.0]
        for xi, ei in zip(x, expected):
            assert abs(xi - ei) < TOLERANCE