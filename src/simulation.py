"""
simulation.py
Contains algorithmic solvers for simulating and optimizing irrigation schedules.
Uses a Model Predictive Control approach to simulate future soil states 
and minimize water usage while adhering to biological crop thresholds.
"""
import numpy as np

def simulate_and_optimize(rain_array, et_array, S_initial, S_min, S_target, S_cap):
    """
    Simulates the daily moisture balance and computes the optimal irrigation schedule.
    
    Parameters:
    - rain_array: 1D numpy array of daily rainfall (mm)
    - et_array: 1D numpy array of daily evapotranspiration (mm)
    - S_initial: Initial soil moisture at Day 0 (%)
    - S_min: Critical minimum soil moisture threshold (%)
    - S_target: Target optimal soil moisture after irrigation (%)
    - S_cap: Soil field capacity (maximum moisture retention %)
    
    Returns:
    - optimized_irrigation: 1D array of daily irrigation applied (mm)
    - simulated_moisture: 1D array of resulting soil moisture (%)
    """
    N_days = len(rain_array)
    optimized_irrigation = np.zeros(N_days)
    
    # Moisture array is N+1 to hold the state at the start of each day, plus the final end state
    simulated_moisture = np.zeros(N_days + 1)
    simulated_moisture[0] = S_initial

    for t in range(N_days):
        # 1. Predict tomorrow's moisture without any irrigation
        projected_S = simulated_moisture[t] + rain_array[t] - et_array[t]
        
        # 2. Check Constraint: Will moisture drop below the minimum threshold?
        if projected_S < S_min:
            # 3. Calculate exact water needed to hit the optimal target
            water_needed = S_target - projected_S
            optimized_irrigation[t] = water_needed
            
            # Apply the irrigation
            simulated_moisture[t+1] = projected_S + water_needed
        else:
            # No irrigation needed
            optimized_irrigation[t] = 0.0
            simulated_moisture[t+1] = projected_S
            
        # 4. Drainage Cap: If massive rain pushes moisture above field capacity, it drains away
        if simulated_moisture[t+1] > S_cap:
            simulated_moisture[t+1] = S_cap
            
    return optimized_irrigation, simulated_moisture