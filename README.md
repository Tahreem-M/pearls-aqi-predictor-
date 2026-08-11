#  Pearls AQI Predictor

A 100% serverless, end-to-end machine learning system that forecasts the Air Quality Index (AQI) for Karachi **3 days in advance** — built as a Data Science internship project at Ten Pearls.

**🔗 Live dashboard: [eixzyufmehiobse65uavs6.streamlit.app](https://eixzyufmehiobse65uavs6.streamlit.app/)**

---

## What it does

- Fetches live weather and pollution data every hour
- Computes the real EPA Air Quality Index from raw pollutant concentrations
- Trains and evaluates multiple forecasting models against a naive baseline
- Predicts AQI 72 hours ahead using the best-performing model
- Explains each prediction with SHAP (which factors drove it, and by how much)
- Runs entirely unattended — no servers, no manual steps

## Architecture

This project follows the **FTI (Feature / Training / Inference)** pipeline pattern used in production ML systems:

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

Both the Feature Pipeline and Training Pipeline run automatically via **GitHub Actions** — no manual intervention required.

## Tech stack

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

## Results

Both trained models were evaluated against a naive **persistence baseline** (predicting that AQI in 3 days equals AQI today), using a chronological train/test split appropriate for time-series data:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Persistence (baseline) | 45.10 | 13.83 | -2.679 |
| Ridge Regression | 30.34 | 19.64 | -0.665 |
| Random Forest | 31.38 | 16.61 | -0.781 |

Both trained models reduce RMSE by roughly **a third** compared to the naive baseline, confirming they learn genuine predictive signal from weather and pollutant patterns rather than assuming tomorrow looks like today.

## Project structure

```
aqi-predictor/
├── .github/workflows/
│   ├── feature_pipeline.yml     # runs hourly
│   └── training_pipeline.yml    # runs daily
├── app/
│   └── dashboard.py              # Streamlit dashboard
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

## Running it locally

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

## Automation

- **Feature Pipeline** — runs every hour via GitHub Actions, fetching live data and pushing engineered features to the Hopsworks Feature Store
- **Training Pipeline** — runs daily via GitHub Actions, retraining on the latest data and pushing the best model to the Model Registry

Both use GitHub repository secrets for credentials — no keys are stored in code.

## Author

**Tahreem Malik ** — BSCS, Sukkur IBA University
Data Science Intern, 10Pearls