# Vietnam EV Charging Visual Analytics Dashboard

Interactive Python Shiny dashboard for exploring nationwide electric-vehicle charging sessions and identifying smart-charging opportunities from operational charging logs.

This project was developed for **COMP5120 Data Visualization — Project 2: Data Stories with Python Shiny**. The application combines spatial visualization, temporal demand analysis, behavioral analytics, anomaly detection, infrastructure utilization analysis, and forecasting-based validation.

## 1. Project overview

The dashboard answers the following analytical question:

> How can real-world EV charging session data be transformed into an interactive visual analytics tool that helps operators understand charging demand, detect abnormal sessions, evaluate infrastructure usage, and identify regions suitable for future smart-charging interventions?

The application is designed around a visual analytics workflow:

1. Load monthly charging-session CSV files.
2. Clean raw operational fields and engineer reusable analytical features.
3. Apply rule-based and optional machine-learning anomaly detection.
4. Visualize spatial, temporal, behavioral, and infrastructure patterns.
5. Support interactive filtering by time, geography, station, charger, venue, and anomaly mode.
6. Forecast future charging demand and compare predictions against actual observed data when available.

## 2. Main features

### Interactive dashboard

The app includes global filters for:

- Date range
- Region
- Province
- Station name
- Charger type
- Station type
- Venue type
- Anomaly mode: show all, normal sessions only, or anomalies only

The dashboard also displays KPI cards for:

- Total charging sessions
- Total energy delivered
- Infrastructure coverage, measured as stations / chargers
- Anomaly rate

### Visualization modules

The implemented dashboard contains ten visual modules:

1. **Spatial Demand Map** — station-level bubble map using latitude/longitude, session count, and energy delivered.
2. **Temporal Load Profile** — hourly demand line chart and weekday-hour heatmap.
3. **Behavioral Analysis** — session duration versus delivered energy, colored by charger type and marked by anomaly status.
4. **Charger Type Mix** — sessions and energy distribution by charger type.
5. **Infrastructure Utilization Ranking** — top stations ranked by a utilization proxy using sessions, energy, and duration.
6. **Smart Charging Opportunity Matrix** — province-level priority map based on peak demand intensity, scheduling flexibility, and energy demand.
7. **Forecast: Number of EVs** — time-series forecast of future charging-session counts.
8. **Forecast: Total Energy Demand** — time-series forecast of aggregate energy demand.
9. **Forecast: Energy per Session** — per-session prediction versus actual energy values.
10. **Forecast: Charging Duration per Session** — per-session prediction versus actual duration values.

### Forecasting and validation

The forecasting module supports both statistical and machine-learning methods:

- Seasonal Naive / Mean
- Seasonal profile / Group mean
- Recency-weighted seasonal model
- Ridge Regression
- Random Forest
- HistGradientBoosting Regression

Forecast settings include:

- Cutoff date
- Cutoff hour
- Preset horizons: 1 hour, 4 hours, 1 day, 1 week, 1 month
- Custom horizon length and unit
- Training window: all history, last 30 days, last 60 days, or last 90 days
- Multi-method selection and ensemble mean

The forecast charts display predicted values together with actual future observations when those observations exist in the dataset. Error summaries such as MAE and MAPE are shown in the legend for model comparison.

## 3. Dataset

The application expects monthly CSV files containing EV charging session logs. Each row represents one charging session. Important fields include:

- `thoi_gian_bat_dau`: charging start time
- `thoi_gian_dung_sac`: charging stop time
- `thoi_gian_ket_thuc`: transaction end time
- `dien_nang_tieu_thu`: delivered energy
- `muc_pin_bat_dau`: starting state of charge
- `muc_pin_ket_thuc`: ending state of charge
- `ma_tram`: station ID
- `ma_tru`: charger ID
- `loai_tru`: charger type
- `ten_tram`: station name
- `mien`: region
- `tinh_thanh`: province
- `loai_tram`: station type
- `loai_mat_bang`: venue type
- `vi_do`, `kinh_do`: station coordinates

The repository may include small sample CSV files for reproducibility and demonstration. For full-scale analysis, place the complete monthly CSV files in the `data/` directory. Sensitive or proprietary raw data should not be committed publicly unless permitted.

## 4. Project structure

```text
.
├── app.py
├── modules/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── ml.py
│   ├── metrics.py
│   ├── forecasting.py
│   └── charts.py
├── www/
│   └── styles.css
├── data/
│   ├── 2024-01.csv
│   ├── 2024-02.csv
│   └── 2024-03.csv
├── README.md
└── requirements.txt
```

Recommended organization:

- `app.py`: Python Shiny UI and server logic.
- `modules/data_loader.py`: safe CSV loading with multiple encodings.
- `modules/preprocessing.py`: data cleaning, type conversion, feature engineering, and rule-based anomaly flags.
- `modules/ml.py`: optional Isolation Forest anomaly scoring.
- `modules/metrics.py`: KPI and decision-support summaries.
- `modules/forecasting.py`: statistical and machine-learning forecasting engine.
- `modules/charts.py`: Plotly chart functions.
- `www/styles.css`: dashboard styling.
- `data/`: monthly CSV data files.

## 5. Installation

### 5.1 Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell
```

### 5.2 Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the core packages manually:

```bash
pip install shiny shinywidgets pandas numpy plotly scikit-learn
```

## 6. Running the application locally

From the project root, run:

```bash
shiny run --reload --launch-browser app.py
```

The application will open in a local browser window. If it does not open automatically, copy the local URL printed in the terminal.

## 7. Data setup

Create a `data/` folder in the project root and place monthly CSV files inside it:

```text
data/
├── 2024-01.csv
├── 2024-02.csv
├── 2024-03.csv
└── ...
```

The loader automatically reads all `.csv` files in the folder and concatenates them into one dataframe. The app adds a `source_file` column so sessions can still be traced to the original monthly export.

## 8. Notes on preprocessing

The preprocessing pipeline performs the following steps:

- Standardizes column names.
- Parses Vietnamese-style date/time strings with day-first priority.
- Converts numeric operational columns to numeric types.
- Cleans categorical text fields.
- Creates dashboard features:
  - `energy_kwh`
  - `duration_minutes`
  - `soc_delta`
  - `scheduling_flexibility_raw`
  - `avg_power_estimated_kw`
  - `hour`, `weekday`, `month`, and `is_weekend`
- Flags invalid or suspicious sessions using rule-based anomaly conditions.
- Checks whether coordinates fall inside a plausible Vietnam latitude/longitude range.

If enough valid records are available and `scikit-learn` is installed, the app also adds an optional Isolation Forest anomaly score.

## 9. Forecasting behavior

Forecasting intentionally uses a different filtering logic from the descriptive charts:

- Descriptive charts follow all global filters, including the Date Range filter.
- Forecasting follows all non-date filters and the anomaly toggle.
- Forecasting ignores the Date Range filter because validation requires future records after the selected cutoff date.

This design allows the app to train on data before the cutoff and compare predictions with actual records after the cutoff when those records are available.

## 10. Deployment note

The app can be deployed to shinyapps.io after confirming that:

1. The repository has the correct structure.
2. Required packages are listed in `requirements.txt`.
3. Data files are small enough for deployment or replaced with a reproducible sample.
4. Sensitive operational data is excluded or anonymized.

A typical deployment workflow is:

```bash
pip install rsconnect-python
rsconnect deploy shiny . --name <account-name> --title vietnam-ev-charging-dashboard
```

Replace `<account-name>` with the shinyapps.io account name used for the project.

## 11. Limitations

- The dashboard depends on the quality and completeness of the charging-session logs.
- Charging status monitoring was not retained as a final module because the available status-related fields did not provide enough variation for a meaningful visualization.
- Forecasting models are intentionally lightweight so that they can run interactively inside the dashboard.
- Per-session forecasts are harder than aggregate demand forecasts and should be interpreted as exploratory model diagnostics rather than production-grade predictions.
- The current smart-charging opportunity score is a visual analytics proxy, not a full grid-constrained optimization model.

## 12. Future improvements

Potential extensions include:

- Integrating live or streaming charging-session updates.
- Adding station-level capacity and charger availability data.
- Incorporating electricity price or grid-load signals.
- Testing advanced forecasting models such as LSTM, temporal convolutional networks, or probabilistic forecasting.
- Adding user-defined smart-charging scenarios.
- Exporting filtered charts and summary tables.
- Improving deployment with a public anonymized dataset and reproducible data-generation scripts.

## 13. Author

Nguyen Van Cuong  
COMP5120 Data Visualization, Spring 2026  
VinUniversity
