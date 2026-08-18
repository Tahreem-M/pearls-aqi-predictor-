"""
Helper functions shared across pipelines.

OpenWeather gives raw pollutant concentrations (µg/m3), NOT the standard
0-500 AQI scale. We convert PM2.5 and PM10 to the real US EPA AQI ourselves,
since that's the standard everyone (AQICN, govt dashboards) actually uses.

Note: full EPA AQI also considers O3, CO, SO2, NO2 breakpoints. PM2.5/PM10
are the dominant pollutants in most South Asian cities, so starting with
just these two is a reasonable v1 — you can extend to the other pollutants
later if you want a more complete implementation (mention this as a
deliberate scoping choice in your report).
"""

import numpy as np
import pandas as pd

# EPA breakpoints: (concentration_low, concentration_high, aqi_low, aqi_high)
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]


def _linear_aqi(concentration, breakpoints):
    """Standard EPA linear interpolation formula."""
    for c_low, c_high, aqi_low, aqi_high in breakpoints:
        if c_low <= concentration <= c_high:
            return round(
                ((aqi_high - aqi_low) / (c_high - c_low)) * (concentration - c_low) + aqi_low
            )
    # Above the highest breakpoint — cap at 500 (hazardous, off the charts)
    return 500


def compute_aqi(pm2_5, pm10):
    """
    Real AQI = the WORSE (higher) of the PM2.5 sub-index and PM10 sub-index.
    This is how EPA actually defines overall AQI — not an average.
    """
    aqi_pm25 = _linear_aqi(pm2_5, PM25_BREAKPOINTS)
    aqi_pm10 = _linear_aqi(pm10, PM10_BREAKPOINTS)
    return max(aqi_pm25, aqi_pm10)


def aqi_category(aqi):
    """Human-readable EPA category — useful for the dashboard and alerts."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def add_engineered_features(df):
    """
    Adds cyclical time encodings and rolling-window features to a feature
    DataFrame. Used identically by training and inference so the model
    always sees the same feature shape.

    Why cyclical encoding: a raw 'month' column treats December (12) and
    January (1) as maximally different, even though they're seasonally
    adjacent. Encoding as sin/cos pairs fixes this for hour, day_of_week,
    and month.

    Why time-based rolling windows (not row-count lag): the feature
    pipeline's collection frequency has changed over the project's
    lifetime (was hourly, now every 4 hours, for Hopsworks compute-budget
    reasons — see README). A lag defined by row count (e.g. "24 rows back")
    would silently mean a different time span depending on when the row
    was collected. Using pandas' time-based .rolling('24h') is correct
    regardless of how densely rows are spaced.

    Requires: df has 'timestamp' (parseable to datetime), 'aqi', 'pm2_5',
    'pm10', 'hour', 'day_of_week', 'month' columns. Returns a new,
    timestamp-sorted DataFrame with additional feature columns. Rows near
    the start of the input won't have enough history for the longest
    rolling window (72h) and will contain NaNs there — drop those before
    training.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df_ts = df.set_index("timestamp")
    df["aqi_rolling_mean_24h"] = df_ts["aqi"].rolling("24h").mean().values
    df["aqi_rolling_mean_72h"] = df_ts["aqi"].rolling("72h").mean().values
    df["pm2_5_rolling_mean_24h"] = df_ts["pm2_5"].rolling("24h").mean().values
    df["pm10_rolling_mean_24h"] = df_ts["pm10"].rolling("24h").mean().values

    return df


# The feature set used by training, inference, and SHAP — defined once here
# so all three always stay in sync.
FEATURE_COLUMNS = [
    "aqi",
    "pm2_5", "pm10", "co", "no2", "so2", "o3",
    "temp", "humidity", "pressure", "wind_speed",
    "aqi_change_rate",
    "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
    "aqi_rolling_mean_24h", "aqi_rolling_mean_72h",
    "pm2_5_rolling_mean_24h", "pm10_rolling_mean_24h",
]