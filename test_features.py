import pandas as pd
from src.features import make_features, BASE_FEATURES

def sample():
    n = 250
    ts = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "load_mw": range(1000, 1000+n),
        "temperature_c": 10.0, "dew_point_c": 5.0, "wind_speed_mps": 3.0,
        "relative_humidity_pct": 60.0, "precipitation_mm": 0.0,
    })

def test_lags_are_past_only():
    x = make_features(sample()).dropna()
    row = x.iloc[10]
    assert row.load_lag_1 == x.iloc[9].load_mw
    assert row.load_lag_168 == x.iloc[10].load_mw - 168

def test_features_exist():
    x = make_features(sample())
    assert set(BASE_FEATURES).issubset(x.columns)
