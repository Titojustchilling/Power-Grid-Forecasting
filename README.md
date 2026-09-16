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

The included 24-hour recursive forecast holds the latest observed weather constant. That is deliberate for a reproducible demonstration, but it is not a real operational weather forecast. 
## Important grid-model caveat

PJM public data does not provide nodal load at individual locations; PJM states that nodal load is confidential. Therefore this project does nog claim to reconstruct the PJM transmission network. The IEEE-14-bus step is a benchmark scenario demonstrating how a forecasted system-demand trajectory can be coupled to a power-flow model.

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


## Sources

- PJM Data Miner: https://www.pjm.com/markets-and-operations/etools/data-miner-2
- PJM Data Miner API Guide: https://www.pjm.com/-/media/DotCom/etools/data-miner-2/data-miner-2-api-guide.ashx
- NOAA GHCNh: https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly
