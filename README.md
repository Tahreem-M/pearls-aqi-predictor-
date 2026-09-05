# AQI Predictor

A 100% serverless, end-to-end machine learning system that forecasts the Air Quality Index (AQI) for Karachi **3 days in advance** — built as a Data Science internship project at Ten Pearls.

**🔗 Live dashboard: [eixzyufmehiobse65uavs6.streamlit.app](https://eixzyufmehiobse65uavs6.streamlit.app/)**

## Running It Locally

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

## Project Structure

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
