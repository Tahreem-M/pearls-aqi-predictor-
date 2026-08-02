import os
import csv
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from src.config import LAT, LON, CITY_NAME
from src.hopsworks_client import insert_feature_row
from src.utils import compute_aqi, aqi_category

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")

FIELDNAMES = [
    "timestamp", "city", "hour", "day", "month", "day_of_week",
    "pm2_5", "pm10", "co", "no2", "so2", "o3",
    "temp", "humidity", "pressure", "wind_speed",
    "aqi", "aqi_category", "aqi_change_rate",
]


def fetch_raw_data(lat, lon, api_key):
    pollution_url = "http://api.openweathermap.org/data/2.5/air_pollution"
    weather_url = "https://api.openweathermap.org/data/2.5/weather"

    pollution = requests.get(pollution_url, params={"lat": lat, "lon": lon, "appid": api_key}).json()
    weather = requests.get(weather_url, params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}).json()

    return pollution, weather


def get_last_aqi():
    if not os.path.exists(FEATURES_CSV):
        return None
    with open(FEATURES_CSV, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return float(rows[-1]["aqi"])


def build_feature_row(pollution_data, weather_data, city_name, last_aqi=None):
    now = datetime.now(timezone.utc)
    components = pollution_data["list"][0]["components"]

    aqi = compute_aqi(components["pm2_5"], components["pm10"])
    aqi_change_rate = float(aqi - last_aqi) if last_aqi is not None else 0.0

    return {
        "timestamp": now.isoformat(),
        "city": city_name,
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "day_of_week": now.weekday(),
        "pm2_5": components["pm2_5"],
        "pm10": components["pm10"],
        "co": components["co"],
        "no2": components["no2"],
        "so2": components["so2"],
        "o3": components["o3"],
        "temp": float(weather_data["main"]["temp"]),
        "humidity": float(weather_data["main"]["humidity"]),
        "pressure": float(weather_data["main"]["pressure"]),
        "wind_speed": float(weather_data["wind"]["speed"]),
        "aqi": aqi,
        "aqi_category": aqi_category(aqi),
        "aqi_change_rate": aqi_change_rate,
    }


def save_row_locally(row):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_exists = os.path.exists(FEATURES_CSV)
    with open(FEATURES_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def push_to_hopsworks(row):
    """Push one feature row to the aqi_features Feature Group in Hopsworks."""
    insert_feature_row(row)


def run():
    load_dotenv()

    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        print("ERROR: set OPENWEATHER_API_KEY first.")
        return

    last_aqi = get_last_aqi()
    pollution_data, weather_data = fetch_raw_data(LAT, LON, api_key)
    row = build_feature_row(pollution_data, weather_data, CITY_NAME, last_aqi)

    save_row_locally(row)
    push_to_hopsworks(row)

    print(f"[{row['timestamp']}] {CITY_NAME} AQI: {row['aqi']} ({row['aqi_category']}) "
          f"| change: {row['aqi_change_rate']:+.1f}")
    print(f"Saved locally to {FEATURES_CSV}")
    print("Pushed to Hopsworks feature group 'aqi_features'")


if __name__ == "__main__":
    run()