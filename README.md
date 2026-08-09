# Pearls AQI Predictor

End-to-end serverless ML system that forecasts Air Quality Index (AQI) for Karachi 3 days ahead.

## Setup

1. Create a virtual environment with Python 3.11 and install dependencies:

   ```bash
   py -3.11 -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root:

   ```env
   OPENWEATHER_API_KEY=your_key_here
   HOPSWORKS_API_KEY=your_key_here
   HOPSWORKS_PROJECT=sharingan_aqi
   ```

3. Run the feature pipeline:

   ```bash
   python -m src.feature_pipeline
   ```

## Progress Checklist

- [x] Feature pipeline — fetch OpenWeather data, engineer features, compute EPA AQI
- [x] Local CSV storage (`data/features.csv`)
- [x] Hopsworks integration — `aqi_features` Feature Group with dual-write
- [ ] Historical backfill script
- [ ] Training pipeline (Ridge, Random Forest, TensorFlow)
- [ ] GitHub Actions automation (hourly feature, daily training)
- [ ] Inference pipeline (3-day forecast)
- [ ] Streamlit dashboard
- [ ] SHAP explainability
- [ ] Hazardous AQI alerts
- [ ] EDA notebook
- [ ] Final report
