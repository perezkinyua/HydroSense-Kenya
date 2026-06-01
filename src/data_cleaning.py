import pandas as pd
import numpy as np
import os

def process_hydrosense_data(weather_path, sensor_path, crop_path, output_path):
    """
    Reads raw HydroSense IoT data, handles missing values via interpolation,
    removes hardware outliers using IQR, and merges into a master dataset.
    """
    print("Initializing HydroSense Data Pipeline...")

    # 1. Load Data
    df_weather = pd.read_csv(weather_path)
    df_sensor = pd.read_csv(sensor_path)
    df_crops = pd.read_csv(crop_path)

    df_crops.rename(columns={'Zone_ID': 'zone_id'}, inplace=True)  

    df_weather.columns = df_weather.columns.str.strip()
    df_sensor.columns = df_sensor.columns.str.strip()
    df_crops.columns = df_crops.columns.str.strip()

    # 2. Process Weather (15-min to Daily)
    df_weather['ts'] = pd.to_datetime(df_weather['ts'])
    df_weather['date'] = df_weather['ts'].dt.date
    
    weather_daily = df_weather.groupby('date').agg({
        'temp_sht': 'mean',
        'humidity_sht': 'mean',
        'rg1': 'sum'
    }).reset_index()
    
    weather_daily.rename(columns={'temp_sht': 'avg_temp', 'humidity_sht': 'avg_humidity', 'rg1': 'total_rain'}, inplace=True)
    weather_daily['date'] = pd.to_datetime(weather_daily['date'])

    # 3. Process Sensor Data
    df_sensor['timestamp'] = pd.to_datetime(df_sensor['timestamp'])
    df_sensor['date'] = df_sensor['timestamp'].dt.normalize()
    
    # Interpolate missing values per zone
    df_sensor['soil_moisture_pct'] = df_sensor.groupby('zone_id')['soil_moisture_pct'].transform(lambda x: x.interpolate(method='linear'))

    # IQR Outlier Detection per zone
    records = []
    for zone, group in df_sensor.groupby('zone_id'):
        group = group.copy()
        Q1 = group['soil_moisture_pct'].quantile(0.25)
        Q3 = group['soil_moisture_pct'].quantile(0.75)
        IQR = Q3 - Q1
        lower = max(0, Q1 - 1.5 * IQR)
        upper = min(100, Q3 + 1.5 * IQR)
        group['is_outlier'] = (
            (group['soil_moisture_pct'] < lower) |
            (group['soil_moisture_pct'] > upper) |
            (group['sensor_status'] != 'OK')
        )
        records.append(group)

    df_sensor = pd.concat(records, ignore_index=True)
    df_sensor_clean = df_sensor[df_sensor['is_outlier'] == False].drop(columns=['is_outlier'])

    # 4. Master Merge (Clean Calendar Merge on 'date')
    df_master = pd.merge(df_sensor_clean, df_crops, on='zone_id', how='left')
    df_master = pd.merge(df_master, weather_daily, on='date', how='left') # Merging purely on date now!
    
    df_master.drop(columns=['timestamp', 'sensor_status'], inplace=True)

    df_master['avg_temp'] = df_master['avg_temp'].round(2)
    df_master['avg_humidity'] = df_master['avg_humidity'].round(2)
    df_master['total_rain'] = df_master['total_rain'].round(1)

    # 5. Export
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_master.to_csv(output_path, index=False)
    print(f"Pipeline complete! Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    process_hydrosense_data(
        weather_path=os.path.join(base_dir, '..', 'data', 'raw', 'weather_daily.csv'),
        sensor_path=os.path.join(base_dir, '..', 'data', 'raw', 'soil_sensor_data.csv'),
        crop_path=os.path.join(base_dir, '..', 'data', 'raw', 'crop_zone_parameters.csv'),
        output_path=os.path.join(base_dir, '..', 'data', 'processed', 'cleaned_irrigation_dataset.csv')
    )