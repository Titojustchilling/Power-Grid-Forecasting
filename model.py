from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import BASE_FEATURES, model_matrix, make_features


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return {
        "MAE_MW": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_MW": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_pct": float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


def chronological_split(df, train_frac=.70, val_frac=.15):
    n = len(df); a = int(n * train_frac); b = int(n * (train_frac + val_frac))
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def train_and_evaluate(df: pd.DataFrame, model_path: Path, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True); model_path.parent.mkdir(parents=True, exist_ok=True)
    train, val, test = chronological_split(df)
    Xtr, ytr, _ = model_matrix(train)
    Xv, yv, _ = model_matrix(pd.concat([train.tail(168), val], ignore_index=True))
    Xt, yt, xtab = model_matrix(pd.concat([pd.concat([train.tail(168), val.tail(168)]), test], ignore_index=True))

    model = HistGradientBoostingRegressor(max_iter=350, learning_rate=.06, max_leaf_nodes=31, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xt)
    naive = xtab["load_lag_168"].to_numpy()
    m = {"gradient_boosting": metrics(yt, pred), "seasonal_naive_168h": metrics(yt, naive), "n_test": int(len(yt))}
    (results_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    pd.DataFrame([m["gradient_boosting"], m["seasonal_naive_168h"]], index=["gradient_boosting", "seasonal_naive_168h"]).to_csv(results_dir / "model_comparison.csv")
    pd.DataFrame({"timestamp": xtab.timestamp, "actual_mw": yt, "predicted_mw": pred, "seasonal_naive_mw": naive}).to_csv(results_dir / "test_predictions.csv", index=False)

    pi = permutation_importance(model, Xt, yt, n_repeats=5, random_state=42, scoring="neg_mean_absolute_error")
    imp = pd.DataFrame({"feature": BASE_FEATURES, "importance": pi.importances_mean}).sort_values("importance", ascending=False)
    imp.to_csv(results_dir / "feature_importance.csv", index=False)

    joblib.dump({"model": model, "features": BASE_FEATURES}, model_path)

    test_results = pd.DataFrame({
    "timestamp": xtab.timestamp.to_numpy(),
    "actual_mw": yt,
    "predicted_mw": pred,
    "seasonal_naive_mw": naive,
})

    return model, m, test_results, pred


def recursive_forecast(history: pd.DataFrame, model, hours=24):
    work = history.copy().sort_values("timestamp")
    future = []
    for i in range(hours):
        ts = work.timestamp.iloc[-1] + pd.Timedelta(hours=1)
        # Weather is held at the latest observed values for a reproducible scenario.
        row = {"timestamp": ts, "load_mw": np.nan}
        for c in ["temperature_c", "dew_point_c", "wind_speed_mps", "relative_humidity_pct", "precipitation_mm"]:
            row[c] = float(work[c].iloc[-1])
        temp = pd.concat([work, pd.DataFrame([row])], ignore_index=True)
        feat = make_features(temp).iloc[[-1]][BASE_FEATURES]
        yhat = float(model.predict(feat)[0])
        row["load_mw"] = yhat
        work = pd.concat([work, pd.DataFrame([row])], ignore_index=True)
        future.append(row)
    return pd.DataFrame(future)
