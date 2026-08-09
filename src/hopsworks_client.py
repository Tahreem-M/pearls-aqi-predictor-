"""Hopsworks Feature Store connection and aqi_features Feature Group helpers."""

import os

import hopsworks
import pandas as pd

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 5


def _get_credentials():
    """Read Hopsworks credentials from environment variables."""
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    project_name = os.environ.get("HOPSWORKS_PROJECT")
    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY is not set.")
    if not project_name:
        raise ValueError("HOPSWORKS_PROJECT is not set.")
    return api_key, project_name


def connect():
    """Log in to Hopsworks Serverless and return the active project."""
    api_key, project_name = _get_credentials()
    return hopsworks.login(api_key_value=api_key, project=project_name)


def get_or_create_aqi_feature_group(feature_store):
    """
    Return the aqi_features Feature Group, creating it on first use.

    primary_key + event_time let Hopsworks keep full offline history per city.
    Online store is intentionally disabled — this project reads features in
    batch (training/inference pipelines), not via low-latency lookups, and
    the online store's Kafka/SSL setup was causing certificate errors on
    Windows. time_travel_format is set explicitly to HUDI (Hopsworks' own
    default) since leaving it unset caused Hopsworks to fall back to DELTA,
    which needs an extra library we don't have installed.

    Version bumped to 5: earlier versions ended up with mixed int/float
    schemas for pressure, wind_speed, and humidity because the very first
    row pushed to each version determined the locked-in column type. All
    numeric weather fields are now explicitly floated in both
    feature_pipeline.py and backfill_pipeline.py to prevent this recurring.
    """
    return feature_store.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Karachi AQI features from OpenWeather (weather + pollution)",
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=False,
        time_travel_format="HUDI",
    )


def insert_feature_row(row):
    """
    Insert one feature row into the aqi_features Feature Group.
    Use this for single live rows (feature_pipeline.py's hourly runs).

    Hopsworks expects event_time as a datetime column, not an ISO string,
    so we convert timestamp before insert.
    """
    project = connect()
    feature_store = project.get_feature_store()
    feature_group = get_or_create_aqi_feature_group(feature_store)

    df = pd.DataFrame([row])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    feature_group.insert(df)
    return feature_group


def insert_feature_rows_batch(rows):
    """
    Insert many feature rows at once — one login, one materialization job.
    Use this for backfill (backfill_pipeline.py). Inserting hundreds of rows
    one at a time via insert_feature_row() would trigger hundreds of logins
    and materialization job attempts, taking hours instead of seconds.
    """
    project = connect()
    feature_store = project.get_feature_store()
    feature_group = get_or_create_aqi_feature_group(feature_store)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    feature_group.insert(df)
    return feature_group