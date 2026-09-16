# Grid Intelligence — Real PJM + NOAA Load Forecasting V6.1

This version replaces V6's synthetic load/weather generator with a reproducible real-data pipeline:

**PJM metered hourly demand + NOAA GHCNh hourly observations → leakage-safe ML load forecast → IEEE-14-bus power-flow scenario.**

## What is real?

- **Electricity demand:** PJM Data Miner `hrl_load_metered`.
- **Weather:** NOAA/NCEI Global Historical Climatology Network hourly (GHCNh) station files.
- **Grid model:** IEEE 14-bus benchmark supplied by pandapower (not a real PJM network model).

PJM documents `hrl_load_metered` fields such as UTC/EPT timestamps, load area, MW, and company verification status. PJM's current API guide says API queries require a subscription key, date ranges cannot exceed 366 days, and a single query is limited to 50,000 rows. The downloader therefore requests one year at a time and paginates. Do not commit your PJM API key or raw PJM data to a public repository. See the PJM Data Miner terms for redistribution restrictions.

NOAA's current GHCNh product replaces the legacy Integrated Surface Dataset and provides hourly weather observations. This project downloads period-of-record station files and filters them locally to the requested years.

## Setup

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Set the PJM API key outside the repository:

### PowerShell

```powershell
$env:PJM_API_KEY="YOUR_KEY_HERE"
$env:GRID_START_YEAR="2023"
$env:GRID_END_YEAR="2025"
$env:GRID_LOAD_AREA="RTO"
python run_pipeline.py
```

### macOS/Linux

```bash
export PJM_API_KEY="YOUR_KEY_HERE"
export GRID_START_YEAR=2023
export GRID_END_YEAR=2025
export GRID_LOAD_AREA=RTO
python run_pipeline.py
```

PJM's API guide says a free API key can be obtained through its API portal. The automated API is intended for internal use; follow PJM's current acceptable-use/redistribution terms.

## Pipeline

1. Download PJM metered hourly demand.
2. Download NOAA GHCNh station files.
3. Parse UTC timestamps and remove invalid/duplicate demand rows.
4. Interpolate only short weather gaps (never the load target).
5. Average selected representative weather stations into a PJM-footprint weather signal.
6. Merge demand and weather on UTC hour.
7. Build load lags, rolling statistics, weather, and calendar features.
8. Use a chronological 70/15/15 train/validation/test split.
9. Train a HistGradientBoostingRegressor and compare against a 168-hour seasonal-naive baseline.
10. Produce metrics, feature importance, forecast plots, and a 24-hour recursive forecast.
11. Scale an IEEE-14-bus benchmark load to the forecast and run AC power flow.

## Important forecasting caveat

The included 24-hour recursive forecast holds the latest observed weather constant. That is deliberate for a reproducible demonstration, but it is **not** a real operational weather forecast. A stronger portfolio version should feed an actual NOAA/NWS numerical weather prediction forecast into the load model for the future horizon.

## Important grid-model caveat

PJM public data does not provide nodal load at individual locations; PJM states that nodal load is confidential. Therefore this project does **not** claim to reconstruct the PJM transmission network. The IEEE-14-bus step is a benchmark scenario demonstrating how a forecasted system-demand trajectory can be coupled to a power-flow model.

## Output

```text
data/raw/pjm/                  # ignored by git
data/raw/noaa/                 # ignored by git
data/processed/pjm_noaa_hourly.csv
models/load_forecaster.joblib
results/metrics.json
results/model_comparison.csv
results/test_predictions.csv
results/feature_importance.csv
results/feature_importance.png
results/forecast_vs_actual.png
results/forecast_error_distribution.png
results/next_24h_forecast.csv
results/grid_forecast_scenario.csv
```

## Recruiter-facing bullets after running the project

- Developed a Python machine-learning pipeline using multi-year PJM metered hourly electricity demand and NOAA GHCNh weather observations to forecast system load with leakage-safe temporal, weather, and calendar features.
- Benchmarked gradient-boosting forecasts against a seasonal-naive baseline using chronological backtesting and MAE, RMSE, MAPE, and R² metrics.
- Integrated forecasted demand scenarios with an IEEE-14-bus AC power-flow model to quantify forecast-driven changes in bus voltage, transmission loading, and active-power losses.

## Sources

- PJM Data Miner: https://www.pjm.com/markets-and-operations/etools/data-miner-2
- PJM Data Miner API Guide: https://www.pjm.com/-/media/DotCom/etools/data-miner-2/data-miner-2-api-guide.ashx
- NOAA GHCNh: https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly
