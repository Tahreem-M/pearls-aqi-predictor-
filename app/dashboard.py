"""
Streamlit dashboard for Pearls AQI Predictor.

Shows:
  - Current AQI + category
  - 3-day-ahead forecast (reuses inference_pipeline's logic directly)
  - Historical AQI trend chart
  - A hazard alert banner when AQI crosses unhealthy thresholds

Run from the project root (not from inside app/):
    streamlit run app/dashboard.py
"""

import os
import sys

# Make `src` importable when Streamlit runs this file directly from app/,
# since Streamlit only adds this script's own folder to sys.path by default.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st

from src.hopsworks_client import connect, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
from src.inference_pipeline import run as get_forecast
from src.utils import aqi_category

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌫️", layout="centered")


@st.cache_data(ttl=300)  # re-fetch at most every 5 minutes, avoid hammering Hopsworks on every rerun
def load_forecast():
    return get_forecast()


@st.cache_data(ttl=300)
def load_recent_history(hours=72):
    """Pull the last `hours` rows for the trend chart."""
    project = connect()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df.tail(hours)[["timestamp", "aqi"]]


def hazard_banner(aqi_value, label):
    """Show a colored banner based on how unhealthy the AQI level is."""
    if aqi_value > 200:
        st.error(f"🚨 {label}: {aqi_value} — Very Unhealthy / Hazardous. Limit outdoor exposure.")
    elif aqi_value > 150:
        st.warning(f"⚠️ {label}: {aqi_value} — Unhealthy. Sensitive groups should stay indoors.")
    elif aqi_value > 100:
        st.warning(f"⚠️ {label}: {aqi_value} — Unhealthy for Sensitive Groups.")
    else:
        st.success(f"✅ {label}: {aqi_value} — {aqi_category(aqi_value)}.")


st.title("🌫️ Pearls AQI Predictor")
st.caption("3-day AQI forecast for Karachi — 100% serverless ML pipeline (Hopsworks + GitHub Actions)")

with st.spinner("Loading latest forecast..."):
    try:
        forecast = load_forecast()
    except Exception as e:
        st.error(f"Couldn't load forecast: {e}")
        st.stop()

col1, col2 = st.columns(2)
with col1:
    st.metric("Current AQI", forecast["current_aqi"])
with col2:
    delta = forecast["predicted_aqi"] - forecast["current_aqi"]
    st.metric("Predicted AQI (in 3 days)", forecast["predicted_aqi"], delta=delta, delta_color="inverse")

st.divider()
hazard_banner(forecast["predicted_aqi"], "3-day forecast")

st.caption(
    f"Based on data from {forecast['based_on_timestamp']} · "
    f"Model version {forecast['model_version']}"
)

st.divider()
st.subheader("Recent AQI trend")

with st.spinner("Loading recent history..."):
    try:
        history = load_recent_history()
        history["timestamp"] = pd.to_datetime(history["timestamp"])
        history = history.set_index("timestamp")
        st.line_chart(history["aqi"])
    except Exception as e:
        st.warning(f"Couldn't load history chart: {e}")

st.divider()
if st.button("🔄 Refresh now"):
    st.cache_data.clear()
    st.rerun()