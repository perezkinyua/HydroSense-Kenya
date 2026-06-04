"""
clean_data.py
Cleans the three HydroSense-Kenya raw datasets and merges them into a single
processed file: ../data/processed/cleaned_irrigation_dataset.csv

Each row in the output = one zone on one day (90 rows total: 30 days × 3 zones).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
RAW       = Path('../data/raw')
PROCESSED = Path('../data/processed')
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────────────────
weather = pd.read_csv(RAW / 'weather_daily.csv',      parse_dates=['date'])
soil    = pd.read_csv(RAW / 'soil_sensor_data.csv',   parse_dates=['timestamp'])
crop    = pd.read_csv(RAW / 'crop_zone_parameters.csv')

# ── Clean weather_daily ───────────────────────────────────────────────────────
# W1: rainfall_mm NaN → median (right-skewed, median more robust than mean)
rainfall_median = weather['rainfall_mm'].median()
weather['rainfall_mm'] = weather['rainfall_mm'].fillna(rainfall_median)

# W2: humidity_pct NaN → mean (roughly symmetric distribution)
humidity_mean = round(weather['humidity_pct'].mean(), 2)
weather['humidity_pct'] = weather['humidity_pct'].fillna(humidity_mean)

# W3: temperature_c = 45.8 °C (physically implausible for Kenya) → median excl. outlier
temp_median = weather.loc[weather['temperature_c'] != 45.8, 'temperature_c'].median()
weather['temperature_c'] = weather['temperature_c'].replace(45.8, temp_median)

# ── Clean soil_sensor_data ────────────────────────────────────────────────────
# S1: soil_moisture_pct NaN (Zone_B) → Zone_B median
zone_b_moisture_median = soil.loc[soil['zone_id'] == 'Zone_B', 'soil_moisture_pct'].median()
mask_s1 = soil['soil_moisture_pct'].isna()
soil.loc[mask_s1, 'soil_moisture_pct'] = zone_b_moisture_median

# S2: soil_moisture_pct = 8.5 % (Zone_B) → sensor misread, replace with Zone_B median excl. outlier

zone_b_moisture_median2 = soil.loc[
    (soil['zone_id'] == 'Zone_B') & (soil['soil_moisture_pct'] != 8.5),
    'soil_moisture_pct'
].median()
mask_s2 = (soil['zone_id'] == 'Zone_B') & (soil['soil_moisture_pct'] == 8.5)
soil.loc[mask_s2, 'soil_moisture_pct'] = zone_b_moisture_median2

# S3: tank_level_liters = 9900 (Zone_C) → likely digit error, replace with Zone_C median excl. outlier
zone_c_tank_median = soil.loc[
    (soil['zone_id'] == 'Zone_C') & (soil['tank_level_liters'] != 9900),
    'tank_level_liters'
].median()
mask_s3 = (soil['zone_id'] == 'Zone_C') & (soil['tank_level_liters'] == 9900)
soil.loc[mask_s3, 'tank_level_liters'] = zone_c_tank_median

# S4: pump_flow_lpm = 0.0 with sensor_status=CHECK (Zone_B) → replace with Zone_B median excl. zero
zone_b_flow_median = soil.loc[
    (soil['zone_id'] == 'Zone_B') & (soil['pump_flow_lpm'] != 0.0),
    'pump_flow_lpm'
].median()
mask_s4 = (soil['zone_id'] == 'Zone_B') & (soil['sensor_status'] == 'CHECK')
soil.loc[mask_s4, 'pump_flow_lpm']    = zone_b_flow_median
soil.loc[mask_s4, 'sensor_status']    = 'IMPUTED'

# ── Merge ─────────────────────────────────────────────────────────────────────
# Extract date from soil timestamp for joining with weather
soil['date'] = soil['timestamp'].dt.date.astype('datetime64[ns]')

# Merge soil + weather on date (weather columns broadcast to all 3 zones per day)
merged = soil.merge(weather, on='date', how='left')

# Merge in crop zone parameters on zone_id
merged = merged.merge(crop, on='zone_id', how='left')

# ── Final column order ────────────────────────────────────────────────────────
cols = [
    'date', 'zone_id', 'crop_type',
    'soil_moisture_pct', 'tank_level_liters', 'pump_flow_lpm',
    'pump_power_watts', 'sensor_status',
    'rainfall_mm', 'temperature_c',
    'humidity_pct', 'wind_speed_mps', 'solar_index',
    'area_m2', 'min_moisture_pct', 'target_moisture_pct',
    'field_capacity_pct', 'drainage_coefficient',
]
merged = merged[cols].sort_values(['date', 'zone_id']).reset_index(drop=True)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = PROCESSED / 'cleaned_irrigation_dataset.csv'
merged.to_csv(out_path, index=False)

print(f"Saved: {out_path}")
print(f"Shape: {merged.shape}  ({merged['date'].nunique()} days × {merged['zone_id'].nunique()} zones)")
print(f"Remaining NaNs: {merged.isnull().sum().sum()}")