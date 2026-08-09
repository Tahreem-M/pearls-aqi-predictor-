"""
Training pipeline — pulls features from Hopsworks, builds a 3-day-ahead
target, trains multiple models, evaluates against a naive persistence
baseline, and saves the best real model to the Model Registry.

Run: python -m src.training_pipeline
"""

import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.hopsworks_client import connect

load_dotenv()

FEATURE_GROUP_VERSION = 5  # keep in sync with hopsworks_client.py

FEATURE_COLUMNS = [
    "aqi",  # current AQI — the strongest single predictor of near-future AQI
    "hour", "day", "month", "day_of_week",
    "pm2_5", "pm10", "co", "no2", "so2", "o3",
    "temp", "humidity", "pressure", "wind_speed", "aqi_change_rate",
]
TARGET_COLUMN = "aqi_target_3d"


def load_training_data():
    """Pull all rows from the Hopsworks feature group as a pandas DataFrame."""
    project = connect()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_target(df):
    """
    Shift AQI forward by 72 hours to create the '3 days ahead' target.
    Each row's target = the AQI value 72 hours later in the same city.
    Rows near the end won't have a future value yet, so we drop those.
    """
    df[TARGET_COLUMN] = df["aqi"].shift(-72)
    df = df.dropna(subset=[TARGET_COLUMN])
    return df


def evaluate(y_true, y_pred, model_name):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
    return {"model": model_name, "rmse": rmse, "mae": mae, "r2": r2}


def run():
    print("Loading data from Hopsworks...")
    df = load_training_data()
    print(f"Loaded {len(df)} rows.")

    df = build_target(df)
    print(f"{len(df)} rows remain after building 3-day-ahead target.")

    if len(df) < 50:
        print("WARNING: very little training data. Run backfill with more DAYS_BACK first.")
        return

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Time-based split: train on earlier data, test on later data
    # (shuffling randomly would leak future info into training — not valid for time series)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print("\n--- Baseline: naive persistence (predict future AQI = current AQI) ---")
    persistence_pred = X_test["aqi"]
    evaluate(y_test, persistence_pred, "Persistence")
    print("Any real model below should beat this to be worth using.\n")

    results = []

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    results.append((evaluate(y_test, ridge.predict(X_test), "Ridge"), ridge))

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    results.append((evaluate(y_test, rf.predict(X_test), "RandomForest"), rf))

    # Pick the model with the lowest RMSE
    best_metrics, best_model = min(results, key=lambda r: r[0]["rmse"])
    print(f"\nBest model: {best_metrics['model']} (RMSE={best_metrics['rmse']:.2f})")

    os.makedirs("models", exist_ok=True)
    model_path = f"models/{best_metrics['model'].lower()}_aqi.pkl"
    joblib.dump(best_model, model_path)
    print(f"Saved best model locally to {model_path}")

    # Push to Hopsworks Model Registry
    project = connect()
    mr = project.get_model_registry()
    model = mr.python.create_model(
        name="aqi_3day_forecaster",
        metrics={"rmse": best_metrics["rmse"], "mae": best_metrics["mae"], "r2": best_metrics["r2"]},
        description=f"Best model ({best_metrics['model']}) for 3-day-ahead AQI forecast",
    )
    model.save(model_path)
    print("Pushed model to Hopsworks Model Registry.")


if __name__ == "__main__":
    run()