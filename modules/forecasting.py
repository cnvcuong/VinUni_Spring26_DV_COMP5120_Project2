"""
modules/forecasting.py
======================
Forecasting engine for the Vgreen dashboard.

Methods (selectable in the UI):
  Statistical baselines (fast)
    - "naive"          : Seasonal Naive (TS) / Global Mean (per-session)
    - "seasonal_mean"  : Seasonal profile mean (TS) / Group Mean backoff (per-session)
    - "ewm_seasonal"   : Recency-weighted seasonal mean (TS) / recency-weighted
                         group mean (per-session). Weights recent data more, so it
                         tracks drift -- useful for online / continuously-updated data.
  Classical machine learning, tabular feature-engineered (fast)
    - "ridge"          : Ridge regression
    - "rf"             : Random Forest
    - "hgb"            : HistGradientBoosting (fast gradient boosting)

Horizon: a preset key ("1H".."1M") OR a custom spec string "CUSTOM:<n>:<unit>"
with unit in {hour, day, week, month}. The step (resolution) is derived adaptively.

Training window (lookback_days): if set, only data in
[cutoff - lookback_days, cutoff] is used for learning. If None, all history up to
the cutoff is used.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- config

METHOD_LABELS = {
    "naive": "Seasonal Naive / Mean",
    "seasonal_mean": "Seasonal profile / Group mean",
    "ewm_seasonal": "Recency-weighted seasonal",
    "ridge": "Ridge Regression",
    "rf": "Random Forest",
    "hgb": "Gradient Boosting (HGB)",
}
STAT_METHODS = ["naive", "seasonal_mean", "ewm_seasonal"]
ML_METHODS = ["ridge", "rf", "hgb"]
ALL_METHODS = STAT_METHODS + ML_METHODS

# Method families for grouped UI. Only families with a non-empty list are shown.
# Add deep-learning / probabilistic methods here later and they appear automatically.
METHOD_FAMILIES = {
    "stat": ("Statistical baselines", STAT_METHODS),
    "ml": ("Classical ML (tabular, feature-engineered)", ML_METHODS),
    "dl": ("Deep-learning time-series models", []),          # e.g. ["lstm", "tcn", "transformer"]
    "prob": ("Probabilistic forecasting", []),               # e.g. ["quantile_gbm", "conformal"]
}

HORIZON_TS = {
    "1H": ("15min", 4),
    "4H": ("30min", 8),
    "1D": ("1h", 24),
    "1W": ("3h", 56),
    "1M": ("1D", 30),
}
HORIZON_DELTA = {
    "1H": pd.Timedelta(hours=1),
    "4H": pd.Timedelta(hours=4),
    "1D": pd.Timedelta(days=1),
    "1W": pd.Timedelta(days=7),
    "1M": pd.Timedelta(days=30),
}
SEASON_BY_FREQ = {"15min": 96, "30min": 48, "1h": 24, "3h": 8, "1D": 7}
_STEP_HOURS = {"15min": 0.25, "30min": 0.5, "1h": 1.0, "3h": 3.0, "1D": 24.0}
_UNIT_HOURS = {"hour": 1, "day": 24, "week": 168, "month": 720}
MAX_STEPS = 400  # guard so a huge custom horizon cannot freeze the UI


def _resolve_custom(n, unit):
    hours = max(1, int(n)) * _UNIT_HOURS.get(unit, 24)
    window = pd.Timedelta(hours=hours)
    if hours <= 6:
        freq = "15min"
    elif hours <= 48:
        freq = "1h"
    elif hours <= 24 * 21:
        freq = "3h"
    else:
        freq = "1D"
    steps = int(round(hours / _STEP_HOURS[freq]))
    steps = max(1, min(MAX_STEPS, steps))
    return freq, steps, window


def parse_horizon(horizon):
    """Return (freq, steps, window_timedelta) for a preset key or CUSTOM spec."""
    if isinstance(horizon, str) and horizon.startswith("CUSTOM:"):
        _, n, unit = horizon.split(":")
        return _resolve_custom(int(n), unit)
    if horizon in HORIZON_TS:
        freq, steps = HORIZON_TS[horizon]
        return freq, steps, HORIZON_DELTA[horizon]
    freq, steps = HORIZON_TS["1D"]
    return freq, steps, HORIZON_DELTA["1D"]


# ---------------------------------------------------------------- shared

def _end_of_day(cutoff_date):
    return pd.Timestamp(cutoff_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


def _resolve_cutoff(cutoff_date, cutoff_hour=None):
    """Cutoff timestamp. If an hour (0-23) is given, the cutoff is the END of that
    hour on the chosen date; otherwise it is the end of the whole day."""
    base = pd.Timestamp(cutoff_date)
    if cutoff_hour is None or cutoff_hour == "":
        return base + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return base + pd.Timedelta(hours=int(cutoff_hour) + 1) - pd.Timedelta(seconds=1)


def _apply_lookback(obj, time_index, cutoff, lookback_days):
    if not lookback_days:
        return obj
    start = cutoff - pd.Timedelta(days=int(lookback_days))
    return obj[time_index >= start]


def make_series(df, metric, freq):
    if df.empty or "thoi_gian_bat_dau" not in df.columns:
        return pd.Series(dtype=float)
    d = df.dropna(subset=["thoi_gian_bat_dau"]).set_index("thoi_gian_bat_dau").sort_index()
    if d.empty:
        return pd.Series(dtype=float)
    if metric == "sessions":
        s = d["energy_kwh"].resample(freq).size().astype(float)
    else:
        if "energy_kwh" not in d.columns:
            return pd.Series(dtype=float)
        s = d["energy_kwh"].resample(freq).sum().astype(float)
    full = pd.date_range(s.index.min(), s.index.max(), freq=freq)
    return s.reindex(full, fill_value=0.0)


def _future_index(series, cutoff, freq, steps):
    off = pd.tseries.frequencies.to_offset(freq)
    after = series.index[series.index > cutoff]
    if len(after) >= steps:
        return after[:steps]
    start = (series.index[series.index <= cutoff].max() if (series.index <= cutoff).any()
             else series.index.min())
    return pd.date_range(start + off, periods=steps, freq=freq)


# ---------------------------------------------------------------- TS methods

def _seasonal_key(ts_index, freq):
    k = pd.DataFrame(index=ts_index)
    k["weekday"] = ts_index.weekday
    if freq != "1D":
        k["hour"] = ts_index.hour
    if freq in ("15min", "30min"):
        k["minute"] = ts_index.minute
    return k


def _ts_seasonal_naive(train, future_idx, freq):
    y = train.to_numpy(dtype=float)
    steps = len(future_idx)
    if len(y) == 0:
        return np.zeros(steps)
    m = min(SEASON_BY_FREQ.get(freq, 24), len(y))
    return np.tile(y[-m:], int(np.ceil(steps / m)))[:steps]


def _ts_seasonal_mean(train, future_idx, freq, weights=None):
    if train.empty:
        return np.zeros(len(future_idx))
    key = _seasonal_key(train.index, freq)
    cols = list(key.columns)
    y = train.to_numpy(dtype=float)
    if weights is None:
        weights = np.ones(len(y))
    key["__wy"] = weights * y
    key["__w"] = weights
    num = key.groupby(cols)["__wy"].sum()
    den = key.groupby(cols)["__w"].sum()
    profile = (num / den).rename("v").reset_index()
    gmean = float(np.average(y, weights=weights))
    kf = _seasonal_key(future_idx, freq).reset_index(drop=True)
    merged = kf.merge(profile, on=cols, how="left")
    return merged["v"].fillna(gmean).to_numpy()


def _ts_ewm_seasonal(train, future_idx, freq, half_life_cycles=4):
    if train.empty:
        return np.zeros(len(future_idx))
    m = SEASON_BY_FREQ.get(freq, 24)
    n = len(train)
    cycles_from_end = (n - 1 - np.arange(n)) // m
    w = 0.5 ** (cycles_from_end / max(1, half_life_cycles))
    return _ts_seasonal_mean(train, future_idx, freq, weights=w)


def _ts_build_features(series, freq):
    m = SEASON_BY_FREQ.get(freq, 24)
    f = pd.DataFrame(index=series.index)
    f["y"] = series.values
    f["weekday"] = series.index.weekday
    f["hour"] = series.index.hour
    f["is_weekend"] = (series.index.weekday >= 5).astype(int)
    f["lag1"] = series.shift(1)
    f["lag_season"] = series.shift(m)
    f["roll_mean"] = series.shift(1).rolling(m, min_periods=1).mean()
    return f


def _ts_ml(train, future_idx, freq, model):
    m = SEASON_BY_FREQ.get(freq, 24)
    feat = _ts_build_features(train, freq).dropna()
    cols = ["weekday", "hour", "is_weekend", "lag1", "lag_season", "roll_mean"]
    if len(feat) < max(10, m // 2):
        return _ts_seasonal_naive(train, future_idx, freq)
    model.fit(feat[cols].to_numpy(), feat["y"].to_numpy())
    history = train.copy()
    preds = []
    for t in future_idx:
        lag1 = history.iloc[-1] if len(history) else 0.0
        lag_season = history.iloc[-m] if len(history) >= m else float(history.mean() if len(history) else 0.0)
        roll = history.iloc[-m:].mean() if len(history) else 0.0
        x = np.array([[t.weekday(), t.hour, int(t.weekday() >= 5), lag1, lag_season, roll]])
        yhat = max(float(model.predict(x)[0]), 0.0)
        preds.append(yhat)
        history = pd.concat([history, pd.Series([yhat], index=[t])])
    return np.array(preds)


def _ts_method(train, future_idx, freq, method):
    if method == "naive":
        return _ts_seasonal_naive(train, future_idx, freq)
    if method == "seasonal_mean":
        return _ts_seasonal_mean(train, future_idx, freq)
    if method == "ewm_seasonal":
        return _ts_ewm_seasonal(train, future_idx, freq)
    if method == "ridge":
        from sklearn.linear_model import Ridge
        return _ts_ml(train, future_idx, freq, Ridge(alpha=1.0))
    if method == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return _ts_ml(train, future_idx, freq,
                      RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1))
    if method == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return _ts_ml(train, future_idx, freq,
                      HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=42))
    raise ValueError(method)


def forecast_timeseries(df, cutoff_date, horizon, metric, methods, lookback_days=None, cutoff_hour=None):
    freq, steps, _ = parse_horizon(horizon)
    series = make_series(df, metric, freq)
    if series.empty:
        return pd.DataFrame()
    cutoff = _resolve_cutoff(cutoff_date, cutoff_hour)
    train = series[series.index <= cutoff]
    train = _apply_lookback(train, train.index, cutoff, lookback_days)
    if train.empty:
        return pd.DataFrame()
    fidx = _future_index(series, cutoff, freq, steps)
    out = pd.DataFrame({"timestamp": fidx})
    out["actual"] = series.reindex(fidx).to_numpy()
    methods = [m for m in methods if m in ALL_METHODS] or ["naive"]
    timings = {}
    for m in methods:
        t0 = time.perf_counter()
        out[m] = _ts_method(train, fidx, freq, m)
        timings[m] = time.perf_counter() - t0
    out["ensemble"] = out[methods].mean(axis=1)
    out.attrs["methods"] = methods
    out.attrs["timings"] = timings
    return out


# ---------------------------------------------------------------- per-session

SESSION_FEATURES_CAT = ["loai_tru", "loai_mat_bang", "mien"]
SESSION_FEATURES_NUM = ["hour", "is_weekend"]


def _soc_bin(s):
    return pd.cut(pd.to_numeric(s, errors="coerce"),
                  bins=[-1, 20, 40, 60, 80, 101],
                  labels=["0-20", "20-40", "40-60", "60-80", "80-100"]).astype("string")


def _session_design(train, test):
    cat = SESSION_FEATURES_CAT + ["soc_bin"]
    Xtr = pd.get_dummies(train[cat + SESSION_FEATURES_NUM], columns=cat, dummy_na=True)
    Xte = pd.get_dummies(test[cat + SESSION_FEATURES_NUM], columns=cat, dummy_na=True)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    return Xtr.to_numpy(dtype=float), Xte.to_numpy(dtype=float)


def _group_mean_backoff(train, test, target, group_cols, weights=None):
    tr = train.copy()
    tr["__w"] = 1.0 if weights is None else weights
    tr["__wy"] = tr["__w"] * tr[target].to_numpy(dtype=float)
    gmean = float(tr["__wy"].sum() / tr["__w"].sum())
    res = pd.Series(np.nan, index=test.index, dtype=float)
    for k in range(len(group_cols), 0, -1):
        cols = group_cols[:k]
        num = tr.groupby(cols, observed=True)["__wy"].sum()
        den = tr.groupby(cols, observed=True)["__w"].sum()
        prof = (num / den).rename("v").reset_index()
        need = res.isna()
        if not need.any():
            break
        merged = test.loc[need, cols].merge(prof, on=cols, how="left")
        res.loc[need] = merged["v"].to_numpy()
    return res.fillna(gmean).to_numpy()


def _session_method(train, test, target, method):
    gcols = ["loai_tru", "soc_bin", "loai_mat_bang"]
    if method == "naive":
        return np.full(len(test), float(train[target].mean()))
    if method == "seasonal_mean":
        return _group_mean_backoff(train, test, target, gcols)
    if method == "ewm_seasonal":
        age_days = (train["thoi_gian_bat_dau"].max() - train["thoi_gian_bat_dau"]).dt.total_seconds() / 86400.0
        w = 0.5 ** (age_days.to_numpy() / 30.0)
        return _group_mean_backoff(train, test, target, gcols, weights=w)
    if method in ("ridge", "rf", "hgb"):
        Xtr, Xte = _session_design(train, test)
        if method == "ridge":
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
        elif method == "rf":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=42)
        model.fit(Xtr, train[target].to_numpy(dtype=float))
        return np.clip(model.predict(Xte), 0, None)
    raise ValueError(method)


def forecast_sessions_level(df, cutoff_date, horizon, target, methods, lookback_days=None, cutoff_hour=None):
    if target not in df.columns or "thoi_gian_bat_dau" not in df.columns:
        return pd.DataFrame()
    s = df.dropna(subset=["thoi_gian_bat_dau", target]).copy()
    s = s[s[target] >= 0]
    if s.empty:
        return pd.DataFrame()
    s["soc_bin"] = _soc_bin(s.get("muc_pin_bat_dau"))
    for c in SESSION_FEATURES_CAT:
        if c not in s.columns:
            s[c] = "NA"
        s[c] = s[c].astype("string").fillna("NA")
    if "hour" not in s.columns:
        s["hour"] = s["thoi_gian_bat_dau"].dt.hour
    if "is_weekend" not in s.columns:
        s["is_weekend"] = (s["thoi_gian_bat_dau"].dt.weekday >= 5).astype(int)

    _, _, window = parse_horizon(horizon)
    cutoff = _resolve_cutoff(cutoff_date, cutoff_hour)
    window_end = cutoff + window
    train = s[s["thoi_gian_bat_dau"] <= cutoff]
    train = _apply_lookback(train, train["thoi_gian_bat_dau"], cutoff, lookback_days)
    test = s[(s["thoi_gian_bat_dau"] > cutoff) & (s["thoi_gian_bat_dau"] <= window_end)]
    if test.empty and len(train) >= 10:
        train = train.sort_values("thoi_gian_bat_dau")
        cut = int(len(train) * 0.8)
        train, test = train.iloc[:cut], train.iloc[cut:]
    if train.empty or test.empty:
        return pd.DataFrame()

    methods = [m for m in methods if m in ALL_METHODS] or ["naive"]
    out = test[["thoi_gian_bat_dau", "loai_tru", "muc_pin_bat_dau", target]].copy()
    out = out.rename(columns={target: "actual"}).reset_index(drop=True)
    test_idx = test.reset_index(drop=True)
    timings = {}
    for m in methods:
        t0 = time.perf_counter()
        out[m] = _session_method(train, test_idx, target, m)
        timings[m] = time.perf_counter() - t0
    out["ensemble"] = out[methods].mean(axis=1)
    out.attrs["methods"] = methods
    out.attrs["timings"] = timings
    return out


# ---------------------------------------------------------------- metrics

def error_text(actual, pred):
    a = pd.to_numeric(pd.Series(actual), errors="coerce")
    p = pd.to_numeric(pd.Series(pred), errors="coerce")
    mask = a.notna() & p.notna()
    if mask.sum() == 0:
        return "actual not available in window"
    err = (a[mask] - p[mask]).abs()
    mae = err.mean()
    denom = a[mask].abs()
    mape = (err[denom > 0] / denom[denom > 0]).mean() * 100 if (denom > 0).any() else np.nan
    return f"MAE={mae:.2f}" + (f" | MAPE={mape:.1f}%" if pd.notna(mape) else "")