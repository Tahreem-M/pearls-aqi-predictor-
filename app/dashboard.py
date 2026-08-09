"""
Streamlit dashboard for Pearls AQI Predictor.

Shows:
  - Current AQI + 3-day forecast as gauge charts, color-coded by severity
  - A hazard alert banner when AQI crosses unhealthy thresholds
  - Historical AQI trend chart with shaded EPA category bands
  - A breakdown of the raw pollutants behind the current reading

Run from the project root (not from inside app/):
    streamlit run app/dashboard.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.hopsworks_client import connect, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
from src.inference_pipeline import run as get_forecast
from src.utils import aqi_category

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌫️", layout="wide")

# --- AQI category colors, used consistently across gauges, banners, and chart bands ---
AQI_BANDS = [
    (0, 50, "#00e400", "Good"),
    (50, 100, "#ffff00", "Moderate"),
    (100, 150, "#ff7e00", "Unhealthy for Sensitive Groups"),
    (150, 200, "#ff0000", "Unhealthy"),
    (200, 300, "#8f3f97", "Very Unhealthy"),
    (300, 500, "#7e0023", "Hazardous"),
]


def aqi_color(value):
    for low, high, color, _ in AQI_BANDS:
        if low <= value < high:
            return color
    return AQI_BANDS[-1][2]


st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 2.8rem; }
    .aqi-caption { color: #9ca3af; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("# 🌫️ Pearls AQI Predictor")
st.markdown(
    "<p class='aqi-caption'>3-day AQI forecast for Karachi &nbsp;·&nbsp; "
    "100% serverless ML pipeline &nbsp;·&nbsp; Hopsworks + GitHub Actions</p>",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300)
def load_forecast():
    return get_forecast()


@st.cache_data(ttl=300)
def load_recent_history(hours=168):
    project = connect()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df.tail(hours)


def make_gauge(value, title):
    color = aqi_color(value)
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 18}},
        number={"font": {"size": 40, "color": color}},
        gauge={
            "axis": {"range": [0, 300], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [{"range": [low, high], "color": c} for low, high, c, _ in AQI_BANDS if low < 300],
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.9, "value": value},
        },
    )).update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e5e7eb"})


with st.spinner("Loading latest forecast..."):
    try:
        forecast = load_forecast()
    except Exception as e:
        st.error(f"Couldn't load forecast: {e}")
        st.stop()

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(make_gauge(forecast["current_aqi"], "Current AQI"), use_container_width=True)
with col2:
    st.plotly_chart(make_gauge(forecast["predicted_aqi"], "Predicted AQI · 3 days"), use_container_width=True)

# --- Hazard banner ---
predicted = forecast["predicted_aqi"]
category = aqi_category(predicted)
if predicted > 200:
    st.error(f"🚨 **{category}** forecast ({predicted}) — limit outdoor exposure over the next 3 days.")
elif predicted > 150:
    st.warning(f"⚠️ **{category}** forecast ({predicted}) — sensitive groups should stay indoors.")
elif predicted > 100:
    st.warning(f"⚠️ **{category}** forecast ({predicted}) — sensitive groups should take precautions.")
else:
    st.success(f"✅ **{category}** forecast ({predicted}) — air quality expected to stay safe.")

st.markdown(
    f"<p class='aqi-caption'>Based on data from {forecast['based_on_timestamp']} · "
    f"Model version {forecast['model_version']}</p>",
    unsafe_allow_html=True,
)

st.divider()

# --- Trend chart with shaded EPA category bands ---
st.subheader("📈 7-day AQI trend")
with st.spinner("Loading recent history..."):
    try:
        history = load_recent_history()
        history["timestamp"] = pd.to_datetime(history["timestamp"])

        fig = go.Figure()
        for low, high, color, label in AQI_BANDS:
            fig.add_hrect(y0=low, y1=high, fillcolor=color, opacity=0.12, line_width=0)
        fig.add_trace(go.Scatter(
            x=history["timestamp"], y=history["aqi"],
            mode="lines", line=dict(color="#60a5fa", width=2.5), name="AQI",
        ))
        fig.update_layout(
            height=350, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e5e7eb"}, yaxis_title="AQI", xaxis_title=None,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Couldn't load history chart: {e}")
        history = None

st.divider()

# --- Pollutant breakdown for the latest reading ---
st.subheader("🧪 Current pollutant levels")
if history is not None and len(history) > 0:
    latest = history.iloc[-1]
    pollutants = [
        ("PM2.5", latest.get("pm2_5"), "µg/m³"),
        ("PM10", latest.get("pm10"), "µg/m³"),
        ("O₃", latest.get("o3"), "µg/m³"),
        ("NO₂", latest.get("no2"), "µg/m³"),
        ("SO₂", latest.get("so2"), "µg/m³"),
        ("CO", latest.get("co"), "µg/m³"),
    ]
    cols = st.columns(len(pollutants))
    for col, (name, value, unit) in zip(cols, pollutants):
        with col:
            st.metric(name, f"{value:.1f}" if value is not None else "—", help=unit)

st.divider()
if st.button("🔄 Refresh now"):
    st.cache_data.clear()
    st.rerun()