import pandas as pd
import json
import os

# Load the datasets
df_air_quality = pd.read_csv('C:/Users/aimen/Desktop/mlops/course-project-aimenSaf/data/air_quality_data.csv')
df_weather = pd.read_csv('C:/Users/aimen/Desktop/mlops/course-project-aimenSaf/data/weather_data.csv')

# Extract and clean air quality data
def extract_air_quality(row):
    data = json.loads(row.replace("'", '"'))  # Convert to valid JSON by replacing single quotes with double quotes
    # Convert timestamp and remove timezone information directly
    timestamp = pd.to_datetime(data['current']['pollution']['ts'])
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)

    return {
        'timestamp': timestamp,
        'aqi_us': data['current']['pollution']['aqius'],
        'main_pollutant_us': data['current']['pollution']['mainus'],
        'aqi_cn': data['current']['pollution']['aqicn'],
        'main_pollutant_cn': data['current']['pollution']['maincn'],
        'temperature': data['current']['weather']['tp'],
        'pressure': data['current']['weather']['pr'],
        'humidity': data['current']['weather']['hu'],
        'wind_speed': data['current']['weather']['ws'],
        'wind_direction': data['current']['weather']['wd'],
    }

df_air_quality_clean = pd.DataFrame([extract_air_quality(row) for row in df_air_quality['data']])

# Extract and clean weather data
def extract_weather(row):
    main = json.loads(row['main'].replace("'", '"'))
    timestamp = pd.to_datetime(row['dt'], unit='s')  # Convert Unix timestamp to datetime
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return {
        'timestamp': timestamp,
        'temp': main['temp'],
        # Add other fields as needed
    }

df_weather_clean = pd.DataFrame([extract_weather(row) for index, row in df_weather.iterrows()])

# Merge the datasets
df_combined = pd.merge_asof(df_air_quality_clean.sort_values('timestamp'), 
                            df_weather_clean.sort_values('timestamp'), 
                            on='timestamp', tolerance=pd.Timedelta('1 hour'), direction='nearest')

output_dir = 'C:/Users/aimen/Desktop/mlops/course-project-aimenSaf/output'
output_file_path = os.path.join(output_dir, 'combined_data.csv')

df_combined.to_csv(output_file_path, index=False)
print(f"File saved successfully at {output_file_path}")
print(df_combined.head())
