"""Download, clean, and align real PJM load and NOAA GHCNh weather data."""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .config import (
    DATA_RAW_NOAA,
    DATA_RAW_PJM,
    DATA_PROCESSED,
    END_YEAR,
    NOAA_BASE_URL,
    NOAA_STATIONS,
    PJM_API_KEY,
    PJM_API_URL,
    PJM_LOAD_AREA,
    START_YEAR,
)

PJM_FIELDS = (
    "datetime_beginning_utc,datetime_beginning_ept,datetime_ending_utc,"
    "datetime_ending_ept,nerc_region,mkt_region,zone,load_area,mw,is_verified"
)


def _require_pjm_key() -> None:
    if not PJM_API_KEY:
        raise RuntimeError(
            "PJM_API_KEY is not set. Create a PJM Data Miner/API account and "
            "export PJM_API_KEY before running the real-data pipeline."
        )


def download_pjm(start_year: int = START_YEAR, end_year: int = END_YEAR) -> list[Path]:
    """Use the manually downloaded PJM CSV when available."""
    manual_file = DATA_RAW_PJM / "hrl_load_metered.csv"

    if manual_file.exists():
        print(f"Using manual PJM file: {manual_file}")
        return [manual_file]

    raise FileNotFoundError(
        f"Could not find the manual PJM file at: {manual_file}\n"
        "Place hrl_load_metered.csv in data/raw/."
    )


def _ghcnh_url(station: str) -> str:
    return f"{NOAA_BASE_URL}/GHCNh_{station}_por.psv"


def download_noaa_station(station: str, force: bool = False) -> Path:
    """Download a NOAA GHCNh period-of-record station file."""
    DATA_RAW_NOAA.mkdir(parents=True, exist_ok=True)
    out = DATA_RAW_NOAA / f"GHCNh_{station}_por.psv"
    if out.exists() and not force:
        return out
    r = requests.get(_ghcnh_url(station), timeout=120)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out


def download_noaa(stations: Iterable[str] = NOAA_STATIONS, force: bool = False) -> list[Path]:
    return [download_noaa_station(s, force=force) for s in stations]


def clean_pjm(files: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip().lower() for c in df.columns]
        df["timestamp"] = pd.to_datetime(df["datetime_beginning_utc"], utc=True, errors="coerce")
        df["load_mw"] = pd.to_numeric(df["mw"], errors="coerce")
        if "load_area" in df.columns:
            df["load_area"] = df["load_area"].astype(str).str.strip()
        frames.append(df[["timestamp", "load_mw", "load_area", "is_verified"]])
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["timestamp", "load_mw"])
    out = out[out["load_mw"] > 0]
    out = out.sort_values("timestamp")
    # Keep one row per timestamp. For RTO data this should normally already be unique.
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    return out.reset_index(drop=True)


def _parse_ghcnh_psv(path: Path) -> pd.DataFrame:
    # GHCNh is pipe-separated. Column names contain spaces/case that we normalize.
    df = pd.read_csv(path, sep="|", low_memory=False)
    cols = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols)
    return df


def clean_noaa(files: Iterable[Path], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for f in files:
        station = f.name.split("_")[1]
        df = _parse_ghcnh_psv(f)
        # GHCNh's date field is ISO-like in the current format; errors become NaT.
        date_col = next((c for c in df.columns if c.lower() == "date"), None)
        temp_col = next((c for c in df.columns if "temperature" == c or c == "dry_bulb_temperature"), None)
        dew_col = next((c for c in df.columns if "dew_point_temperature" in c), None)
        wind_col = next((c for c in df.columns if c == "wind_speed"), None)
        rh_col = next((c for c in df.columns if c == "relative_humidity"), None)
        precip_col = next((c for c in df.columns if c == "precipitation"), None)
        if date_col is None or temp_col is None:
            raise RuntimeError(f"Could not identify DATE/Temperature columns in {f.name}.")

        keep = [date_col, temp_col]
        for c in [dew_col, wind_col, rh_col, precip_col]:
            if c and c not in keep:
                keep.append(c)
        x = df[keep].copy()
        x["timestamp"] = pd.to_datetime(x[date_col], utc=True, errors="coerce")
        rename = {temp_col: "temperature", date_col: "source_date"}
        if dew_col: rename[dew_col] = "dew_point"
        if wind_col: rename[wind_col] = "wind_speed"
        if rh_col: rename[rh_col] = "relative_humidity"
        if precip_col: rename[precip_col] = "precipitation"
        x = x.rename(columns=rename)
        for c in ["temperature", "dew_point", "wind_speed", "relative_humidity", "precipitation"]:
            if c in x:
                x[c] = pd.to_numeric(x[c], errors="coerce")
        if "precipitation" in x:
            x["precipitation"] = x["precipitation"].fillna(0)
        # GHCNh commonly uses tenths of degrees C for temperature/dew point and tenths
        # of m/s for wind. The downloader stores raw values; conversion is explicit here.
        x["temperature_c"] = x["temperature"] / 10.0
        x["dew_point_c"] = x.get("dew_point", pd.Series(index=x.index, dtype=float)) / 10.0
        x["wind_speed_mps"] = x.get("wind_speed", pd.Series(index=x.index, dtype=float)) / 10.0
        x["relative_humidity_pct"] = x.get("relative_humidity", pd.Series(index=x.index, dtype=float)) / 10.0
        x["precipitation_mm"] = x.get("precipitation", pd.Series(index=x.index, dtype=float)) / 10.0
        x = x[(x["timestamp"] >= start) & (x["timestamp"] <= end)]
        x["station"] = station
        frames.append(x[["timestamp", "station", "temperature_c", "dew_point_c", "wind_speed_mps", "relative_humidity_pct", "precipitation_mm"]])

    weather = pd.concat(frames, ignore_index=True)
    # Each station is first regularized independently. Only short gaps are interpolated.
    pieces = []
    for station, g in weather.groupby("station"):
        g = g.sort_values("timestamp").drop_duplicates("timestamp")
        idx = pd.date_range(g.timestamp.min(), g.timestamp.max(), freq="1h", tz="UTC")
        g = g.set_index("timestamp").reindex(idx)
        g["station"] = station
        for c in ["temperature_c", "dew_point_c", "wind_speed_mps", "relative_humidity_pct", "precipitation_mm"]:
            if c in g:
                g[c] = g[c].interpolate(limit=3, limit_direction="both")
        pieces.append(g.reset_index(names="timestamp"))
    weather = pd.concat(pieces, ignore_index=True)
    # Footprint average: weather observations from selected representative stations.
    return weather.groupby("timestamp", as_index=False)[
        ["temperature_c", "dew_point_c", "wind_speed_mps", "relative_humidity_pct", "precipitation_mm"]
    ].mean()


def build_merged_dataset(
    pjm_files: Iterable[Path],
    noaa_files: Iterable[Path],
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
) -> Path:
    start = pd.Timestamp(f"{start_year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{end_year}-12-31 23:00:00", tz="UTC")
    load = clean_pjm(pjm_files)
    weather = clean_noaa(noaa_files, start, end)
    load = load[(load.timestamp >= start) & (load.timestamp <= end)]
    merged = pd.merge(load[["timestamp", "load_mw"]], weather, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").drop_duplicates("timestamp")
    # Only retain rows where all model inputs exist. Do not interpolate the target load.
    merged = merged.dropna(subset=["load_mw", "temperature_c"])
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "pjm_noaa_hourly.csv"
    merged.to_csv(out, index=False)
    return out
