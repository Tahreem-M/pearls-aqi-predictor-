# Pearls AQI Predictor

A 100% serverless, end-to-end machine learning system that forecasts the Air Quality Index (AQI) for Karachi **3 days in advance** — built as a Data Science internship project at Ten Pearls.

**🔗 Live dashboard: [eixzyufmehiobse65uavs6.streamlit.app](https://eixzyufmehiobse65uavs6.streamlit.app/)**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Data Sources and Design Decisions](#4-data-sources-and-design-decisions)
5. [Model Training and Evaluation](#5-model-training-and-evaluation)
6. [Model Explainability (SHAP)](#6-model-explainability-shap)
7. [Automation](#7-automation)
8. [Dashboard](#8-dashboard)
9. [Challenges and Solutions](#9-challenges-and-solutions)
10. [Future Improvements](#10-future-improvements)
11. [Running It Locally](#11-running-it-locally)
12. [Project Structure](#12-project-structure)

---

## 1. Project Overview

Pearls AQI Predictor is a serverless machine learning system that forecasts AQI for Karachi three days in advance. It follows the **Feature/Training/Inference (FTI)** pipeline architecture: independent, automated pipelines communicate through a shared Feature Store and Model Registry rather than one monolithic script.

The completed system includes automated hourly data collection, a historical backfill of real Karachi air-quality data, two trained forecasting models benchmarked against a naive baseline, a live inference pipeline, SHAP-based model explainability, and an interactive, publicly deployed Streamlit dashboard — all running unattended.

## 2. Architecture

```
Raw weather + pollution APIs
        │
        ▼
  Feature Pipeline  ──────►  Hopsworks Feature Store
  (hourly, automated)              │
                                    ▼
  Training Pipeline  ─────►  Hopsworks Model Registry
  (daily, automated)                │
                                     ▼
  Inference Pipeline  ────►  Streamlit Dashboard
  (on-demand)                 (live, public)
```

- **Feature Pipeline** — fetches live weather and pollution data every hour, computes the real EPA Air Quality Index and engineered features, writes to the Hopsworks Feature Store.
- **Training Pipeline** — runs daily, pulls historical features, trains multiple models, evaluates them, pushes the best to the Model Registry.
- **Inference Pipeline** — loads the latest model and most recent features, produces a 3-day-ahead AQI forecast.

Both the Feature and Training pipelines are fully automated through GitHub Actions.

## 3. Technology Stack

| Purpose | Tool |
|---|---|
| Language | Python |
| Live weather + pollution data | OpenWeather API |
| Historical weather backfill | Open-Meteo API (free, no key required) |
| Feature Store / Model Registry | Hopsworks (serverless, free tier) |
| Models | Scikit-learn (Ridge Regression, Random Forest) |
| Automation / CI-CD | GitHub Actions (hourly + daily schedules) |
| Dashboard | Streamlit + Plotly |
| Explainability | SHAP |
| Deployment | Streamlit Community Cloud |
| Version control | Git / GitHub |

## 4. Data Sources and Design Decisions

### 4.1 Computing a real AQI

OpenWeather's own "air quality index" field uses a simplified 1–5 scale, not the standard 0–500 EPA index used by official monitoring dashboards. This system instead computes the true EPA AQI directly from raw PM2.5 and PM10 concentrations using the official EPA breakpoint formula, taking the higher (worse) of the two pollutant sub-indices as the overall AQI — matching how real air-quality authorities report it.

### 4.2 Historical backfill strategy

OpenWeather's historical weather endpoint requires a paid subscription tier. To avoid this blocker, historical weather data (temperature, humidity, pressure, wind speed) is sourced from **Open-Meteo's** free historical weather API, which requires no API key. Historical pollution data uses OpenWeather's free pollution-history endpoint. This is a deliberate design choice — using the best available free source for each data type — rather than a limitation.

### 4.3 Feature engineering

Each row combines: time-based features (hour, day, month, day of week), raw pollutant concentrations, weather variables, the computed AQI, its EPA category label, and the AQI change rate relative to the previous reading.

## 5. Model Training and Evaluation

The target variable is AQI 72 hours (3 days) ahead, created by shifting the AQI column forward in time. The dataset was split **chronologically** — training on the earlier 80%, testing on the most recent 20% — since a random split would leak future information into training, which is invalid for time-series forecasting.

Two models were trained and compared against a naive **persistence baseline** (predicting that AQI in 3 days equals the current AQI):

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Persistence (baseline) | 45.10 | 13.83 | -2.679 |
| Ridge Regression | 30.34 | 19.64 | -0.665 |
| Random Forest | 31.38 | 16.61 | -0.781 |

Both trained models reduce RMSE by roughly **a third** compared to the naive baseline, confirming they learn genuine predictive signal from weather and pollutant patterns rather than simply assuming tomorrow looks like today. The best-performing model by RMSE is automatically selected and pushed to the Hopsworks Model Registry; the daily automated training pipeline has since progressed to later model versions as more data accumulates.

R² remains negative relative to a simple "always predict the historical mean" baseline. Given the genuinely high day-to-day variability of real-world air quality data and the difficulty of 3-day-ahead forecasting, this is reported here as an honest, explainable result rather than a modelling failure.

## 6. Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) explains individual predictions from the deployed model — which features pushed a given forecast up or down, and by how much. A representative explanation for a recent forecast:

| Feature | SHAP Contribution | Effect |
|---|---|---|
| month | -15.92 | Decreases predicted AQI |
| wind_speed | -6.81 | Decreases predicted AQI |
| pm2_5 | +5.36 | Increases predicted AQI |
| day | +5.26 | Increases predicted AQI |
| o3 | +4.42 | Increases predicted AQI |
| pm10 | -3.63 | Decreases predicted AQI |

The dominant driver was the `month` feature, suggesting a seasonal effect on Karachi's air quality that the model learned from training data. Wind speed was the second-strongest factor, consistent with the physical expectation that higher wind speeds disperse pollutants and lower AQI. This explainability layer is also surfaced live in the dashboard as an interactive chart.

## 7. Automation

Both pipelines run automatically via GitHub Actions, using repository secrets to securely store API keys — no credentials are stored in code.

**Feature Pipeline** — runs every hour:

![Feature pipeline running successfully on GitHub Actions](docs/screenshots/github_actions_feature.png)

**Training Pipeline** — runs once daily:

![Training pipeline running successfully on GitHub Actions](docs/screenshots/github_actions_training.png)

## 8. Dashboard

An interactive Streamlit dashboard, deployed publicly, presents the live forecast with gauge charts, a hazard alert banner, a 7-day trend chart with shaded EPA category bands, and a live SHAP explanation chart.

**🔗 Live: [eixzyufmehiobse65uavs6.streamlit.app](https://eixzyufmehiobse65uavs6.streamlit.app/)**

**Current AQI and 3-day forecast gauges, with hazard status banner:**

![AQI gauges and hazard banner](docs/screenshots/dashboard_gauges.png)

**7-day AQI trend with shaded EPA category bands:**

![7-day AQI trend](docs/screenshots/dashboard_trend.png)

**Live pollutant breakdown for the latest reading:**

![Pollutant breakdown](docs/screenshots/dashboard_pollutants.png)

## 9. Challenges and Solutions

Several real engineering obstacles were encountered and resolved during development:

- **Windows SSL certificate errors** connecting to Hopsworks' online (Kafka-based) feature store — resolved by disabling the online store, which is not required for this project's batch-read use case.
- **Feature Store schema type mismatches** (integer vs. float columns) caused by inconsistent data types between the live feature pipeline and backfill pipeline — resolved by explicitly casting all numeric weather fields to float in both pipelines.
- **Backfill performance** — an initial row-by-row insert approach triggered hundreds of separate logins and materialization jobs; switched to a single batch insert, reducing backfill time from hours to under a minute.
- **Cross-version model comparison** — the Model Registry initially selected an outdated model version by raw RMSE, not trained on the current feature set; resolved by explicitly selecting the latest model version rather than comparing metrics across incompatible feature sets.
- **Python version incompatibility on deployment** — both a local Windows environment and Streamlit Cloud's default Python (3.14) broke `hopsworks`'s dependencies, which rely on the deprecated `imp` module; resolved by pinning Python 3.11 via `runtime.txt`.
- **A misconfigured `.gitignore`** (a stray wildcard) silently excluded most of the project from version control for a period; identified via `git status --ignored` and corrected.

## 10. Future Improvements

- Extend backfill history beyond 90 days to further stabilise model performance.
- Add a TensorFlow/deep-learning model to the comparison, as outlined in the original project brief.
- Support multiple cities via the existing configurable `CITIES` setting.
- Add lag-based rolling-average features to strengthen short-horizon signal.

## 11. Running It Locally

```bash
git clone https://github.com/Tahreem-M/pearls-aqi-predictor-.git
cd pearls-aqi-predictor-
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
OPENWEATHER_API_KEY=your_key_here
HOPSWORKS_API_KEY=your_key_here
HOPSWORKS_PROJECT=your_hopsworks_project_name
```

Run any pipeline directly:
```bash
python -m src.feature_pipeline
python -m src.training_pipeline
python -m src.inference_pipeline
python -m src.shap_explain
```

Run the dashboard:
```bash
streamlit run app/dashboard.py
```

## 12. Project Structure

```
aqi-predictor/
├── .github/workflows/
│   ├── feature_pipeline.yml     # runs hourly
│   └── training_pipeline.yml    # runs daily
├── app/
│   └── dashboard.py              # Streamlit dashboard
├── docs/screenshots/              # images used in this README
├── notebooks/
│   └── eda.ipynb                 # exploratory data analysis
├── src/
│   ├── config.py                 # city + constants
│   ├── utils.py                  # EPA AQI calculation
│   ├── hopsworks_client.py       # Feature Store / Model Registry connection
│   ├── feature_pipeline.py       # fetch + engineer features
│   ├── backfill_pipeline.py      # historical data backfill
│   ├── training_pipeline.py      # train + evaluate + register models
│   ├── inference_pipeline.py     # load model + predict
│   └── shap_explain.py           # model explainability
├── requirements.txt
├── runtime.txt                   # pins Python 3.11 for deployment
└── README.md
```

---

## Author

**Tahreem Malik** — BSCS, Sukkur IBA University
Data Science Intern, 10Pearls
