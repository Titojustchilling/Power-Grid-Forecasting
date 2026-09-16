from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.config import START_YEAR, END_YEAR, DATA_PROCESSED, MODELS, RESULTS, NOAA_STATIONS
from src.data import download_pjm, download_noaa, build_merged_dataset
from src.model import train_and_evaluate, recursive_forecast
from src.grid import run_power_flow


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"Downloading PJM metered hourly demand: {START_YEAR}-{END_YEAR}")
    pjm = download_pjm()
    print("Downloading NOAA GHCNh station data:", ", ".join(NOAA_STATIONS))
    noaa = download_noaa()
    merged_path = build_merged_dataset(pjm, noaa)
    df = pd.read_csv(merged_path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    print(f"Merged rows: {len(df):,}")

    model, metrics, test_table, pred = train_and_evaluate(df, MODELS / "load_forecaster.joblib", RESULTS)
    history = df.tail(1000).copy()
    future = recursive_forecast(history, model, hours=24)
    future.to_csv(RESULTS / "next_24h_forecast.csv", index=False)

    # Forecast visualization
    plt.figure(figsize=(12, 5))
    plt.plot(test_table.timestamp, test_table.actual_mw, label="Actual")
    plt.plot(test_table.timestamp, test_table.predicted_mw, label="ML Forecast")
    plt.plot(test_table.timestamp, test_table.seasonal_naive_mw, label="Seasonal Naive")
    plt.title("PJM Hourly Load Forecast — Test Set")
    plt.xlabel("UTC time"); plt.ylabel("MW"); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS / "forecast_vs_actual.png", dpi=160); plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(test_table.actual_mw - test_table.predicted_mw, bins=50)
    plt.title("Forecast Error Distribution")
    plt.xlabel("Actual - Predicted (MW)"); plt.ylabel("Count"); plt.tight_layout()
    plt.savefig(RESULTS / "forecast_error_distribution.png", dpi=160); plt.close()

    plt.figure(figsize=(9, 5))
    fi = pd.read_csv(RESULTS / "feature_importance.csv").head(15).sort_values("importance")
    plt.barh(fi.feature, fi.importance)
    plt.title("Top Permutation Feature Importance")
    plt.xlabel("Importance"); plt.tight_layout()
    plt.savefig(RESULTS / "feature_importance.png", dpi=160); plt.close()

    grid_result = run_power_flow(future, RESULTS / "grid_forecast_scenario.csv")
    print(json.dumps({"metrics": metrics, "grid": grid_result}, indent=2))
    print("Pipeline complete. See data/processed, models/, and results/.")

if __name__ == "__main__":
    main()
