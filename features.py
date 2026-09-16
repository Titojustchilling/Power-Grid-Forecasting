from __future__ import annotations
import numpy as np
import pandas as pd

TARGET = "load_mw"
BASE_FEATURES = [
    "temperature_c", "dew_point_c", "wind_speed_mps", "relative_humidity_pct", "precipitation_mm",
    "hour", "day_of_week", "month", "day_of_year", "is_weekend",
    "load_lag_1", "load_lag_2", "load_lag_3", "load_lag_24", "load_lag_48", "load_lag_72", "load_lag_168",
    "rolling_mean_24", "rolling_std_24", "rolling_mean_168", "rolling_std_168",
]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values("timestamp")
    x["hour"] = x.timestamp.dt.hour
    x["day_of_week"] = x.timestamp.dt.dayofweek
    x["month"] = x.timestamp.dt.month
    x["day_of_year"] = x.timestamp.dt.dayofyear
    x["is_weekend"] = (x["day_of_week"] >= 5).astype(int)
    for lag in [1, 2, 3, 24, 48, 72, 168]:
        x[f"load_lag_{lag}"] = x[TARGET].shift(lag)
    shifted = x[TARGET].shift(1)
    x["rolling_mean_24"] = shifted.rolling(24).mean()
    x["rolling_std_24"] = shifted.rolling(24).std()
    x["rolling_mean_168"] = shifted.rolling(168).mean()
    x["rolling_std_168"] = shifted.rolling(168).std()
    return x


def model_matrix(df: pd.DataFrame):
    x = make_features(df).dropna(subset=BASE_FEATURES + [TARGET])
    return x[BASE_FEATURES], x[TARGET], x
