"""
Backfill pipeline — builds historical training data by combining:
  - OpenWeather air pollution history (free, back to Nov 2020)
  - Open-Meteo historical weather (free, no API key needed)

Pushes all rows to Hopsworks in a single batch insert (one login, one
materialization job) instead of row-by-row, which would trigger hundreds
of logins and take hours.

Run: python -m src.backfill_pipeline
"""

import os
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

from src.config import LAT, LON, CITY_NAME
from src.utils import compute_aqi, aqi_category
from src.hopsworks_client import insert_feature_rows_batch

load_dotenv()

# How far back to backfill — start small (7 days) to test, then increase
DAYS_BACK = 90


def fetch_pollution_history(lat, lon, start_ts, end_ts, api_key):
    """OpenWeather pollution history — returns hourly readings."""
    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {"lat": lat, "lon": lon, "start": start_ts, "end": end_ts, "appid": api_key}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()["list"]


def fetch_weather_history(lat, lon, start_date, end_date):
    """Open-Meteo historical weather — no API key needed, hourly data."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "timezone": "UTC",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()["hourly"]


def build_backfill_dataset():
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        print("ERROR: set OPENWEATHER_API_KEY first.")
        return

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=DAYS_BACK)

    print(f"Fetching pollution history: {start_dt.date()} to {end_dt.date()}...")
    pollution_records = fetch_pollution_history(
        LAT, LON, int(start_dt.timestamp()), int(end_dt.timestamp()), api_key
    )

    print("Fetching weather history from Open-Meteo...")
    weather_data = fetch_weather_history(
        LAT, LON, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
    )

    # Build a lookup: hour-rounded timestamp -> weather values
    weather_lookup = {}
    for i, time_str in enumerate(weather_data["time"]):
        weather_lookup[time_str] = {
            "temp": weather_data["temperature_2m"][i],
            "humidity": weather_data["relative_humidity_2m"][i],
            "pressure": weather_data["surface_pressure"][i],
            "wind_speed": weather_data["wind_speed_10m"][i],
        }

    rows = []
    last_aqi = None
    for record in pollution_records:
        dt = datetime.fromtimestamp(record["dt"], tz=timezone.utc)
        hour_key = dt.strftime("%Y-%m-%dT%H:00")  # match Open-Meteo's hourly format

        weather = weather_lookup.get(hour_key)
        if weather is None:
            continue  # skip if we don't have matching weather for this hour

        components = record["components"]
        aqi = compute_aqi(components["pm2_5"], components["pm10"])
        aqi_change_rate = float(aqi - last_aqi) if last_aqi is not None else 0.0
        last_aqi = aqi

        rows.append({
            "timestamp": dt.isoformat(),
            "city": CITY_NAME,
            "hour": dt.hour,
            "day": dt.day,
            "month": dt.month,
            "day_of_week": dt.weekday(),
            "pm2_5": components["pm2_5"],
            "pm10": components["pm10"],
            "co": components["co"],
            "no2": components["no2"],
            "so2": components["so2"],
            "o3": components["o3"],
            "temp": float(weather["temp"]),
            "humidity": float(weather["humidity"]),
            "pressure": float(weather["pressure"]),
            "wind_speed": float(weather["wind_speed"]),
            "aqi": aqi,
            "aqi_category": aqi_category(aqi),
            "aqi_change_rate": aqi_change_rate,
        })

    print(f"Built {len(rows)} historical rows. Pushing to Hopsworks as one batch...")
    insert_feature_rows_batch(rows)
    print("Backfill complete.")


if __name__ == "__main__":
    build_backfill_dataset()