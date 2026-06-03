import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('../src'))
from simulation import simulate_and_optimize

def test_irrigation_logic():
    """Test that irrigation triggers when moisture drops below S_min."""
    # Scenario: 1 day, no rain, high ET (moisture will drop)
    rain = np.array([0.0])
    et = np.array([10.0])
    
    # S_initial=30, S_min=25, S_target=40, S_cap=50
    # Expected: moisture will drop to 20, which is < S_min. 
    # Algorithm should irrigate 20 (40 - 20)
    irrigation, moisture = simulate_and_optimize(rain, et, 30, 25, 40, 50)
    
    assert irrigation[0] == 20.0
    assert moisture[1] == 40.0

def test_no_irrigation_when_sufficient():
    """Test that irrigation does NOT trigger if moisture stays above S_min."""
    rain = np.array([0.0])
    et = np.array([1.0])
    
    # Start at 40, S_min=25. Should drop to 39, no irrigation needed.
    irrigation, moisture = simulate_and_optimize(rain, et, 40, 25, 40, 50)
    
    assert irrigation[0] == 0.0
    assert moisture[1] == 39.0