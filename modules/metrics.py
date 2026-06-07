# \
from __future__ import annotations

import numpy as np
import pandas as pd


def kpi_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "sessions": "0",
            "energy": "0 kWh",
            "infrastructure": "0 / 0",
            "anomaly_rate": "0.00%",
        }

    total_sessions = len(df)
    total_energy = df["energy_kwh"].sum(skipna=True) if "energy_kwh" in df.columns else 0
    n_stations = df["ma_tram"].nunique() if "ma_tram" in df.columns else 0
    n_chargers = df["ma_tru"].nunique() if "ma_tru" in df.columns else 0
    anomaly_rate = df["is_anomaly"].mean() * 100 if "is_anomaly" in df.columns else 0

    if total_energy >= 1_000_000:
        energy_text = f"{total_energy / 1_000_000:.2f} GWh"
    elif total_energy >= 1_000:
        energy_text = f"{total_energy / 1_000:.2f} MWh"
    else:
        energy_text = f"{total_energy:,.0f} kWh"

    return {
        "sessions": f"{total_sessions:,.0f}",
        "energy": energy_text,
        "infrastructure": f"{n_stations:,.0f} / {n_chargers:,.0f}",
        "anomaly_rate": f"{anomaly_rate:.2f}%",
    }


def insight_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "peak_hours": "No data selected.",
            "hotspots": "No data selected.",
            "abnormal": "No data selected.",
            "smart_candidates": "No data selected.",
        }

    if "hour" in df.columns and df["hour"].notna().any():
        hourly = df.groupby("hour")["energy_kwh"].sum().sort_values(ascending=False)
        top_hours = hourly.head(3).index.astype(int).tolist()
        peak_hours = ", ".join([f"{h:02d}:00" for h in sorted(top_hours)])
    else:
        peak_hours = "Not available."

    if "tinh_thanh" in df.columns:
        province = df.groupby("tinh_thanh").size().sort_values(ascending=False).head(3)
        hotspots = ", ".join([str(x) for x in province.index])
    else:
        hotspots = "Not available."

    anomaly_rate = df["is_anomaly"].mean() * 100 if "is_anomaly" in df.columns else 0
    abnormal = f"{anomaly_rate:.2f}% of selected sessions are flagged as abnormal."

    opp = opportunity_by_province(df)
    if not opp.empty:
        high = opp.sort_values("priority_score", ascending=False).head(3)["tinh_thanh"].tolist()
        smart_candidates = ", ".join([str(x) for x in high])
    else:
        smart_candidates = "Not enough data."

    return {
        "peak_hours": peak_hours,
        "hotspots": hotspots,
        "abnormal": abnormal,
        "smart_candidates": smart_candidates,
    }


def opportunity_by_province(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "tinh_thanh" not in df.columns:
        return pd.DataFrame()

    d = df.copy()
    d["is_peak"] = d["hour"].isin([16, 17, 18, 19, 20, 21]) if "hour" in d.columns else False

    agg = d.groupby("tinh_thanh", dropna=False).agg(
        sessions=("tinh_thanh", "size"),
        energy_kwh=("energy_kwh", "sum"),
        peak_share=("is_peak", "mean"),
        flexibility=("scheduling_flexibility_raw", "median"),
        anomaly_rate=("is_anomaly", "mean"),
    ).reset_index()

    # Normalize 0-1 safely.
    for col in ["peak_share", "flexibility", "energy_kwh", "sessions"]:
        min_v = agg[col].min()
        max_v = agg[col].max()
        if max_v > min_v:
            agg[f"{col}_norm"] = (agg[col] - min_v) / (max_v - min_v)
        else:
            agg[f"{col}_norm"] = 0.5

    agg["priority_score"] = (
        0.45 * agg["peak_share_norm"]
        + 0.35 * agg["flexibility_norm"]
        + 0.20 * agg["energy_kwh_norm"]
    )

    return agg
