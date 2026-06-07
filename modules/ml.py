# \
from __future__ import annotations

import numpy as np
import pandas as pd


def add_ml_anomaly_score(df: pd.DataFrame) -> pd.DataFrame:
    """Optional Isolation Forest anomaly score.

    If scikit-learn is unavailable or the dataset is too small, the function
    silently returns the existing rule-based anomaly column.
    """
    out = df.copy()
    features = ["duration_minutes", "energy_kwh", "avg_power_estimated_kw", "soc_delta"]

    if out.empty or not all(c in out.columns for c in features):
        out["ml_anomaly_score"] = np.nan
        return out

    model_df = out[features].replace([np.inf, -np.inf], np.nan).dropna()
    if len(model_df) < 200:
        out["ml_anomaly_score"] = np.nan
        return out

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import RobustScaler

        # Sampling keeps dashboard startup fast on millions of sessions.
        train = model_df.sample(min(len(model_df), 100_000), random_state=42)
        scaler = RobustScaler()
        x_train = scaler.fit_transform(train)

        model = IsolationForest(
            n_estimators=100,
            contamination=0.03,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train)

        x_all = scaler.transform(out[features].replace([np.inf, -np.inf], np.nan).fillna(model_df.median()))
        scores = -model.score_samples(x_all)
        threshold = np.nanpercentile(scores, 97)

        out["ml_anomaly_score"] = scores
        out["is_ml_anomaly"] = scores >= threshold
        out["is_anomaly"] = out["is_anomaly"] | out["is_ml_anomaly"]
    except Exception:
        out["ml_anomaly_score"] = np.nan

    return out
