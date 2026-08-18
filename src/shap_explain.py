"""
SHAP explainability — explains which features drove the current 3-day
AQI forecast, and by how much.

Run standalone: python -m src.shap_explain
Also reused by the dashboard for a live "why this forecast" chart.
"""

import pandas as pd
import shap

from src.inference_pipeline import load_recent_features, download_latest_model
from src.training_pipeline import load_training_data, build_target
from src.utils import add_engineered_features, FEATURE_COLUMNS


def explain_latest_prediction(background_sample_size=100):
    """
    Returns a DataFrame of (feature, shap_value) for the latest prediction,
    sorted by how much each feature pushed the prediction up or down.

    A background sample from training data is needed because SHAP explains
    each feature's contribution relative to what the model would predict
    "on average" — not in isolation.
    """
    model, model_version = download_latest_model()

    latest_row = load_recent_features()
    X_latest = latest_row[FEATURE_COLUMNS]

    df = load_training_data()
    df = add_engineered_features(df)
    df = df.dropna(subset=FEATURE_COLUMNS)
    df = build_target(df)
    background = df[FEATURE_COLUMNS].sample(
        min(background_sample_size, len(df)), random_state=42
    )

    explainer = shap.Explainer(model.predict, background)
    shap_values = explainer(X_latest)

    contributions = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "shap_value": shap_values.values[0],
    })
    contributions["abs_value"] = contributions["shap_value"].abs()
    contributions = contributions.sort_values("abs_value", ascending=False).drop(columns="abs_value")

    return contributions, model_version


def run():
    print("Computing SHAP explanation for the latest prediction...")
    contributions, model_version = explain_latest_prediction()

    print(f"\nModel version: {model_version}")
    print("\n--- Feature contributions to the 3-day AQI forecast ---")
    print("(positive = pushed prediction UP, negative = pushed prediction DOWN)\n")
    for _, row in contributions.iterrows():
        direction = "+" if row["shap_value"] >= 0 else ""
        print(f"  {row['feature']:<24} {direction}{row['shap_value']:.2f}")


if __name__ == "__main__":
    run()