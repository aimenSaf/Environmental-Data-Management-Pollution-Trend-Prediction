import os
from dotenv import load_dotenv, find_dotenv
import requests
import pandas as pd
from datetime import datetime
import time
import logging

load_dotenv(find_dotenv())

logging.basicConfig(filename='data_collection.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

if "OPENWEATHERMAP_API_KEY" not in os.environ or "AIRVISUAL_API_KEY" not in os.environ:
    logging.critical("API keys are not set in environment variables")
    raise ValueError("API keys are not set in environment variables")

# API configuration
API_KEYS = {
    "OpenWeatherMap": os.getenv('OPENWEATHERMAP_API_KEY'),
    "AirVisual": os.getenv('AIRVISUAL_API_KEY')
}
URLS = {
    "OpenWeatherMap": "http://api.openweathermap.org/data/2.5/weather",
    "AirVisual": "https://api.airvisual.com/v2/city"
}

# Parameters for the API calls
PARAMETERS = {
    "OpenWeatherMap": {"q": "London,uk", "appid": API_KEYS["OpenWeatherMap"]},
    "AirVisual" : {"city": "London", "state": "England", "country": "United Kingdom","key": API_KEYS["AirVisual"]
}

}

def fetch_data(url, params):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Error fetching data from {url}: {str(e)}")
        return None

def save_data(data, filename):
    try:
        df = pd.DataFrame([data])
        df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)
        logging.info(f"Data saved successfully to {filename}")
    except Exception as e:
        logging.error(f"Error saving data: {str(e)}")

def main():
    weather_data = fetch_data(URLS["OpenWeatherMap"], PARAMETERS["OpenWeatherMap"])
    if weather_data:
        save_data(weather_data, "data/weather_data.csv")
    
    air_quality_data = fetch_data("https://api.airvisual.com/v2/city", PARAMETERS["AirVisual"])
    if air_quality_data:
        save_data(air_quality_data, "data/air_quality_data.csv")

if __name__ == "__main__":
    while True:
        main()
        sleeping_time = 300
        logging.info(f"Sleeping for {sleeping_time // 60} minutes")
        time.sleep(sleeping_time)   # 5 min for testing
