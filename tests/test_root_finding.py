import pytest
from numerical_methods import bisection_method, newton_raphson_method

def test_bisection_method():
    # Test finding root of x^2 - 4 = 0 (Root is 2.0)
    func = lambda x: x**2 - 4
    root, _ = bisection_method(func, 0, 5)
    assert root == pytest.approx(2.0, rel=1e-4)

def test_newton_raphson_method():
    # Test finding root of x^2 - 4 = 0 (Root is 2.0)
    func = lambda x: x**2 - 4
    deriv = lambda x: 2*x
    root, _ = newton_raphson_method(func, deriv, 3.0)
    assert root == pytest.approx(2.0, rel=1e-4)