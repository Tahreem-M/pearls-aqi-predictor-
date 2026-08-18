"""
Training pipeline — pulls features from Hopsworks, adds engineered
features (cyclical time encoding + rolling averages), builds a time-based
3-day-ahead target, trains multiple models, evaluates against a naive
persistence baseline, and saves the best real model to the Model Registry.

Run: python -m src.training_pipeline
"""

import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.hopsworks_client import connect
from src.utils import add_engineered_features, FEATURE_COLUMNS

load_dotenv()

FEATURE_GROUP_VERSION = 5  # keep in sync with hopsworks_client.py
TARGET_COLUMN = "aqi_target_3d"
FORECAST_HORIZON_HOURS = 72
TARGET_MATCH_TOLERANCE_HOURS = 2  # how close a row's timestamp must be to "now + 72h" to count


def load_training_data():
    """Pull all rows from the Hopsworks feature group as a pandas DataFrame."""
    project = connect()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_target(df):
    """
    Build the '3 days ahead' target by matching each row to the AQI reading
    closest to (its own timestamp + 72 hours), within a tolerance window.

    This is deliberately time-based rather than a fixed row-count shift:
    the feature pipeline's collection frequency has changed during this
    project (hourly, then every 4 hours, to manage Hopsworks compute
    budget — see README), so "72 rows ahead" would mean a different time
    span depending on when a row was collected. merge_asof matches on
    actual elapsed time instead, so this stays correct regardless of
    collection frequency.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    future = df[["timestamp", "aqi"]].copy()
    future["timestamp"] = (future["timestamp"] - pd.Timedelta(hours=FORECAST_HORIZON_HOURS)).astype(df["timestamp"].dtype)
    future = future.rename(columns={"aqi": TARGET_COLUMN}).sort_values("timestamp")

    merged = pd.merge_asof(
        df, future, on="timestamp", direction="nearest",
        tolerance=pd.Timedelta(hours=TARGET_MATCH_TOLERANCE_HOURS),
    )
    return merged.dropna(subset=[TARGET_COLUMN])


def evaluate(y_true, y_pred, model_name):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    category_acc = category_accuracy(y_true, y_pred)
    print(f"{model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}, Category Accuracy={category_acc:.1%}")
    return {"model": model_name, "rmse": rmse, "mae": mae, "r2": r2, "category_accuracy": category_acc}


def category_accuracy(y_true, y_pred):
    """
    A more intuitive companion metric to RMSE/MAE/R²: what fraction of
    predictions fall in the same EPA AQI category (Good, Moderate,
    Unhealthy, etc.) as the actual value? Unlike R², this has a plain-
    English interpretation and doesn't get distorted by trending data.
    """
    from src.utils import aqi_category
    true_categories = [aqi_category(v) for v in y_true]
    pred_categories = [aqi_category(v) for v in y_pred]
    matches = sum(t == p for t, p in zip(true_categories, pred_categories))
    return matches / len(true_categories)


def run():
    print("Loading data from Hopsworks...")
    df = load_training_data()
    print(f"Loaded {len(df)} rows.")

    print("Adding engineered features (cyclical time + rolling averages)...")
    df = add_engineered_features(df)
    df = df.dropna(subset=FEATURE_COLUMNS)  # drop rows without enough history for 72h rolling window
    print(f"{len(df)} rows remain after feature engineering.")

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
    evaluate(y_test, X_test["aqi"], "Persistence")
    print("Any real model below should beat this to be worth using.\n")

    results = []

    # Ridge is sensitive to feature scale (day_of_week: 0-6 vs pm10: 0-100+),
    # so it's wrapped in a scaling pipeline. Random Forest is scale-invariant
    # and doesn't need this, but scaling doesn't hurt it either.
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(X_train, y_train)
    results.append((evaluate(y_test, ridge.predict(X_test), "Ridge"), ridge))

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    results.append((evaluate(y_test, rf.predict(X_test), "RandomForest"), rf))

    best_metrics, best_model = min(results, key=lambda r: r[0]["rmse"])
    print(f"\nBest model: {best_metrics['model']} (RMSE={best_metrics['rmse']:.2f})")

    os.makedirs("models", exist_ok=True)
    model_path = f"models/{best_metrics['model'].lower()}_aqi.pkl"
    joblib.dump(best_model, model_path)
    print(f"Saved best model locally to {model_path}")

    project = connect()
    mr = project.get_model_registry()
    model = mr.python.create_model(
        name="aqi_3day_forecaster",
        metrics={"rmse": best_metrics["rmse"], "mae": best_metrics["mae"], "r2": best_metrics["r2"]},
        description=f"Best model ({best_metrics['model']}) for 3-day-ahead AQI forecast, "
                     f"with cyclical time encoding and rolling-average features",
    )
    model.save(model_path)
    print("Pushed model to Hopsworks Model Registry.")


if __name__ == "__main__":
    run()