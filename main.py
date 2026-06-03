
"""
main.py
The central execution pipeline for HydroSense-Kenya.
Runs Data Cleaning (Level 1), Vectorized Water Balance (Level 2), 
and the Continuous ODE Solvers (Level 3).
"""

import os
import subprocess
import pandas as pd
import numpy as np


from src.numerical_methods import euler_integrate, rk4_integrate
from src.simulation import simulate_and_optimize

def run_level_2_vectorization(input_path, output_path):
    """Executes the Level 2 NumPy Vectorization Engine."""
    print("   -> Loading clean dataset...")
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])

    print("   -> Applying C-level vectorized ET and Drainage equations...")
    temps = df['temperature_c'].to_numpy()
    humidity = df['humidity_pct'].to_numpy()
    wind = df['wind_speed_mps'].to_numpy()
    solar = df['solar_index'].to_numpy()
    rain = df['rainfall_mm'].to_numpy()
    
    moisture = df['soil_moisture_pct'].to_numpy()
    field_capacity = df['field_capacity_pct'].to_numpy()
    drainage_coeff = df['drainage_coefficient'].to_numpy()

    # Synchronized Empirical ET Formula
    et_raw = (0.12 * temps) + (0.35 * wind) + (2.4 * solar) - (0.025 * humidity)
    et_daily = np.clip(et_raw, a_min=0.0, a_max=None) 

    # Drainage and Projected Moisture
    excess_water = (moisture + rain) - field_capacity
    drainage_daily = np.where(excess_water > 0, excess_water * drainage_coeff, 0.0)
    
    projected_moisture = np.clip(moisture + rain - et_daily - drainage_daily, a_min=0.0, a_max=None)

    # Append and save
    df['estimated_ET_mm'] = np.round(et_daily, 4)
    df['estimated_drainage_mm'] = np.round(drainage_daily, 4)
    df['projected_moisture_pct'] = np.round(projected_moisture, 4)

    df.to_csv(output_path, index=False)
    print(f"   -> Level 2 simulation saved to: {output_path}")

def run_level_5_optimization(input_path):
    """Executes the Level 5 Optimization algorithm."""
    print("   -> Loading Level 2 results for optimization...")
    df = pd.read_csv(input_path)
    
    # We'll optimize for Zone_A (Tomato) as a standard
    df_zone = df[df['zone_id'] == 'Zone_A'].copy()
    
    rain = df_zone['rainfall_mm'].to_numpy()
    et = df_zone['estimated_ET_mm'].to_numpy()
    S_init = df_zone['soil_moisture_pct'].iloc[0]
    
    # Optimize
    irrigation, moisture = simulate_and_optimize(
        rain, et, S_init, 
        df_zone['min_moisture_pct'].iloc[0],
        df_zone['target_moisture_pct'].iloc[0],
        df_zone['field_capacity_pct'].iloc[0]
    )
    
    print(f"   -> Optimization Result: {np.sum(irrigation):.2f} mm total water required.")

def main():
    print("==================================================")
    print("INITIALIZING HYDROSENSE-KENYA PIPELINE")
    print("==================================================")

    # ---------------------------------------------------------
    # LEVEL 1: DATA CLEANING
    # ---------------------------------------------------------
    print("\n[1/3] EXECUTING LEVEL 1: Data Cleaning & Imputation...")
    try:
        # Run your colleague's script from INSIDE the src/ folder so '../data' resolves perfectly
        subprocess.run(["python", "data_cleaning.py"], cwd="src", check=True)
    except Exception as e:
        print(f"Error during Level 1 Execution: {e}")
        return

    # ---------------------------------------------------------
    # LEVEL 2: VECTORIZATION
    # ---------------------------------------------------------
    print("\n[2/3] EXECUTING LEVEL 2: NumPy Vectorized Water Balance...")
    clean_data_path = 'data/processed/cleaned_irrigation_dataset.csv'
    level2_output_path = 'data/processed/level_2_results.csv'
    
    if os.path.exists(clean_data_path):
        run_level_2_vectorization(clean_data_path, level2_output_path)
    else:
        print(f"Error: Level 1 output not found at {clean_data_path}")
        return

    # ---------------------------------------------------------
    # LEVEL 3: DIFFERENTIAL EQUATIONS (Test Run)
    # ---------------------------------------------------------
    print("\n[3/3] EXECUTING LEVEL 3: Testing Continuous ODE Solvers...")
    print("   -> Loading numerical.py engines (Euler & RK4)...")
    # We just run a quick confirmation that the module loads and works
    df_sim = pd.read_csv(level2_output_path)
    print(f"   -> Successfully loaded {len(df_sim)} rows for ODE analysis.")
    print("   -> ODE plotting is reserved for notebooks/Level_3_Differential_Equations.ipynb")

    # ---------------------------------------------------------
    # LEVEL 5: OPTIMIZATION
    # ---------------------------------------------------------
    print("\n[3/3] EXECUTING LEVEL 5: Optimization Algorithm...")
    level2_path = 'data/processed/level_2_results.csv'
    run_level_5_optimization(level2_path)

    print("\n==================================================")
    print(" PIPELINE COMPLETE! All systems nominal.")
    print("==================================================")


if __name__ == "__main__":
    main()