"""
numerical.py
Contains numerical ODE solvers (Euler and RK4) for simulating the 
continuous soil-water balance differential equations over time.
"""
import numpy as np

def euler_integrate(deriv_func, S0, t_array):
    """
    Solves an ODE using the First-Order Forward Euler Method.
    
    Parameters:
    - deriv_func: The differential equation function dS/dt = f(t, S)
    - S0: Initial soil moisture at t=0
    - t_array: Array of time steps (can be fractional for higher resolution)
    
    Returns:
    - S: Array of simulated soil moisture values
    """
    S = np.zeros(len(t_array))
    S[0] = S0
    
    for i in range(1, len(t_array)):
        dt = t_array[i] - t_array[i-1]
        # S_{t+1} = S_t + dt * f(t, S_t)
        S[i] = S[i-1] + dt * deriv_func(t_array[i-1], S[i-1])
        
        # Hard physical constraint: Soil moisture cannot drop below 0
        S[i] = max(0.0, S[i])
        
    return S

def rk4_integrate(deriv_func, S0, t_array):
    """
    Solves an ODE using the Fourth-Order Runge-Kutta (RK4) Method.
    Provides higher numerical stability and accuracy than Euler.
    """
    S = np.zeros(len(t_array))
    S[0] = S0
    
    for i in range(1, len(t_array)):
        dt = t_array[i] - t_array[i-1]
        t = t_array[i-1]
        current_S = S[i-1]
        
        # Calculate the 4 Runge-Kutta slopes
        k1 = deriv_func(t, current_S)
        k2 = deriv_func(t + dt/2.0, current_S + dt * k1 / 2.0)
        k3 = deriv_func(t + dt/2.0, current_S + dt * k2 / 2.0)
        k4 = deriv_func(t + dt, current_S + dt * k3)
        
        # Weighted average of slopes
        S[i] = current_S + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        S[i] = max(0.0, S[i])
        
    return S