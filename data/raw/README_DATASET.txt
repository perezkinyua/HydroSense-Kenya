HydroSense-Kenya Dataset Package for ICS 2207 Scientific Computing

Files:
1. weather_daily.csv - Daily synthetic weather data for 30 days.
2. soil_sensor_data.csv - Daily noon sensor readings for three farm zones.
3. crop_zone_parameters.csv - Zone-level crop and soil-water parameters.

Important notes:
- The data is synthetic and designed for educational use.
- Some missing values and outliers are deliberate. Students must detect, document, and handle them.
- NA values should be treated as missing values in Pandas.
- Recommended loading command: pd.read_csv('weather_daily.csv', na_values=['NA', ''])
