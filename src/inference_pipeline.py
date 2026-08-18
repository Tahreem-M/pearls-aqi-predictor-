"""
Inference pipeline — loads the latest trained model from the Hopsworks
Model Registry, pulls enough recent feature history to compute rolling
features correctly, and predicts AQI 3 days (72 hours) from now.

Run: python -m src.inference_pipeline
"""

import os
import joblib
import pandas as pd
from dotenv import load_dotenv

from src.hopsworks_client import connect, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
from src.utils import add_engineered_features, aqi_category, FEATURE_COLUMNS

load_dotenv()

# Needs to comfortably cover the longest rolling window (72h) plus buffer,
# regardless of current collection frequency.
HISTORY_HOURS_NEEDED = 96


def load_recent_features():
    """
    Pull enough recent rows to compute rolling features correctly, then
    return the single most recent row with those features attached.
    """
    project = connect()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    cutoff = df["timestamp"].max() - pd.Timedelta(hours=HISTORY_HOURS_NEEDED)
    recent = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    recent = add_engineered_features(recent)
    return recent.iloc[[-1]]  # most recent row, with rolling features computed from its history


def download_latest_model():
    """
    Download the most recent version of the aqi_3day_forecaster model
    from the Hopsworks Model Registry and load it with joblib.

    We deliberately pick the HIGHEST version number, not the "best" by
    RMSE across all history — older model versions were trained with a
    different, smaller feature set, so their RMSE isn't comparable to
    newer versions.
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
    print("Loading recent features from Hopsworks...")
    latest_row = load_recent_features()
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