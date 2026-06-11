import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('../src'))
from src.numerical_methods import rk4_integrate

def test_rk4_linear_function():
    """RK4 should perfectly solve the derivative dS/dt = 2t (Integral is t^2 + C)."""
    def derivative(t, S):
        return 2 * t
    
    t_array = np.linspace(0, 1, 11) # 0.0 to 1.0
    S0 = 0.0
    
    # Expected result: S = t^2. At t=1.0, S should be 1.0
    results = rk4_integrate(derivative, S0, t_array)
    assert results[-1] == pytest.approx(1.0, rel=1e-3)

def test_physical_constraint():
    """Ensure moisture never drops below 0 even under extreme evaporation."""
    def high_evap(t, S):
        return -1000.0 # Extreme loss
    
    t_array = np.array([0, 1])
    results = rk4_integrate(high_evap, 5.0, t_array)
    assert results[-1] == 0.0