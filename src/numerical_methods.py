"""
numerical.py
Contains numerical ODE solvers (Euler and RK4) for simulating the 
continuous soil-water balance differential equations over time.
"""
import numpy as np
import math

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



# ==========================================
# 1. ROOT FINDING ALGORITHMS
# ==========================================

def bisection_method(func, a, b, tol=1e-5, max_iter=100):
    if func(a) * func(b) >= 0:
        raise ValueError("Function must have opposite signs at endpoints a and b.")

    for i in range(max_iter):
        c = (a + b) / 2.0
        
        # FIX: Now returning 'c' (the root) AND 'i + 1' (the iteration count)
        if abs(func(c)) < tol or (b - a) / 2.0 < tol:
            return c, i + 1 

        if func(c) * func(a) < 0:
            b = c
        else:
            a = c

    raise TimeoutError("Bisection method did not converge.")

def newton_raphson_method(func, deriv_func, x0, tol=1e-5, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = func(x)
        
        # FIX: Now returning 'x' (the root) AND 'i + 1' (the iteration count)
        if abs(fx) < tol:
            return x, i + 1

        dfx = deriv_func(x)
        if dfx == 0:
            raise ZeroDivisionError("Derivative is zero. Newton-Raphson failed.")

        x = x - fx / dfx

    raise TimeoutError("Newton-Raphson method did not converge.")
def secant_method(func, x0, x1, tol=1e-5, max_iter=100):
    """
    Finds the root using the Secant method.
    Requires two initial guesses but no analytical derivative.
    
    Returns:
    - root: The x-value where func(x) == 0
    - iterations: Number of steps taken to converge
    """
    for i in range(max_iter):
        f_x0 = func(x0)
        f_x1 = func(x1)
        
        if f_x1 - f_x0 == 0:
            raise ZeroDivisionError("Division by zero in Secant method.")
            
        x_new = x1 - f_x1 * ((x1 - x0) / (f_x1 - f_x0))
        
        if abs(x_new - x1) < tol:
            return x_new, i + 1
            
        x0, x1 = x1, x_new
        
    raise TimeoutError("Secant method did not converge.")

def trapezoidal_rule(func, a, b, n):
    """
    Approximates definite integral using the Trapezoidal Rule.
    n is the number of sub-intervals.
    """
    h = (b - a) / n
    integration = func(a) + func(b)
    for i in range(1, n):
        k = a + i * h
        integration += 2 * func(k)
    return integration * h / 2.0

def simpsons_rule(func, a, b, n):
    """
    Approximates definite integral using Simpson's 1/3 Rule.
    n must be an even integer.
    """
    if n % 2 != 0:
        raise ValueError("Number of sub-intervals (n) must be even for Simpson's rule.")
    h = (b - a) / n
    integration = func(a) + func(b)
    for i in range(1, n):
        k = a + i * h
        if i % 2 == 0:
            integration += 2 * func(k)
        else:
            integration += 4 * func(k)
    return integration * h / 3.0

def gaussian_elimination(A, b):
    """
    Solves a system of linear equations Ax = b using Gaussian Elimination 
    with partial pivoting.
    A is a 2D list (matrix), b is a 1D list (vector).
    """
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    for i in range(n):
        max_el = abs(M[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > max_el:
                max_el = abs(M[k][i])
                max_row = k
        M[i], M[max_row] = M[max_row], M[i]
        
        if M[i][i] == 0:
            raise ValueError("Matrix is singular or nearly singular.")
            
        for k in range(i + 1, n):
            factor = -M[k][i] / M[i][i]
            for j in range(i, n + 1):
                if i == j:
                    M[k][j] = 0
                else:
                    M[k][j] += factor * M[i][j]
                    
    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n] / M[i][i]
        for k in range(i - 1, -1, -1):
            M[k][n] -= M[k][i] * x[i]
    return x