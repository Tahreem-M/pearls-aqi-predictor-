"""
Inference pipeline — loads the latest trained model from the Hopsworks
Model Registry and the most recent feature row, and predicts AQI 3 days
(72 hours) from now.

Run: python -m src.inference_pipeline
"""

import os
import joblib
import pandas as pd
from dotenv import load_dotenv

from src.hopsworks_client import connect
from src.utils import aqi_category

load_dotenv()

FEATURE_GROUP_VERSION = 5  # keep in sync with hopsworks_client.py and training_pipeline.py

FEATURE_COLUMNS = [
    "aqi",
    "hour", "day", "month", "day_of_week",
    "pm2_5", "pm10", "co", "no2", "so2", "o3",
    "temp", "humidity", "pressure", "wind_speed", "aqi_change_rate",
]


def load_latest_features():
    """Pull the single most recent row from the feature group."""
    project = connect()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df.iloc[[-1]]  # most recent row, kept as a 1-row DataFrame


def download_latest_model():
    """
    Download the most recent version of the aqi_3day_forecaster model
    from the Hopsworks Model Registry and load it with joblib.

    We deliberately pick the HIGHEST version number, not the "best" by
    RMSE across all history — older model versions were trained with a
    different, smaller feature set (before DAYS_BACK was increased and
    before 'aqi' was added as a feature), so their RMSE isn't comparable
    to newer versions. Picking by RMSE alone previously grabbed an old
    model that didn't expect the current feature columns, causing a
    ValueError at prediction time.
    """
    project = connect()
    mr = project.get_model_registry()
    models = mr.get_models("aqi_3day_forecaster")
    latest_model = max(models, key=lambda m: m.version)
    model_dir = latest_model.download()

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError(f"No .pkl model file found in {model_dir}")

    model_path = os.path.join(model_dir, model_files[0])
    return joblib.load(model_path), latest_model.version


def run():
    print("Loading latest features from Hopsworks...")
    latest_row = load_latest_features()
    current_aqi = latest_row["aqi"].values[0]
    current_timestamp = latest_row["timestamp"].values[0]

    print("Loading latest model from Model Registry...")
    model, model_version = download_latest_model()
    print(f"Using model version {model_version}.")

    X = latest_row[FEATURE_COLUMNS]
    predicted_aqi = model.predict(X)[0]
    predicted_aqi = round(float(predicted_aqi))
    category = aqi_category(predicted_aqi)

    print("\n--- 3-Day AQI Forecast ---")
    print(f"Latest data from: {current_timestamp}")
    print(f"Current AQI: {round(float(current_aqi))}")
    print(f"Predicted AQI in 3 days: {predicted_aqi} ({category})")

    return {
        "current_aqi": round(float(current_aqi)),
        "predicted_aqi": predicted_aqi,
        "predicted_category": category,
        "based_on_timestamp": str(current_timestamp),
        "model_version": model_version,
    }


if __name__ == "__main__":
    run()