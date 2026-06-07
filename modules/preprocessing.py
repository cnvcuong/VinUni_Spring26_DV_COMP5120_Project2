# \
from __future__ import annotations

import numpy as np
import pandas as pd


DATETIME_COLUMNS = [
    "thoi_gian_bat_dau",
    "thoi_gian_dung_sac",
    "thoi_gian_ket_thuc",
    "thoi_gian_cap_nhat",
]

NUMERIC_COLUMNS = [
    "so_phut_qua_gio",
    "chi_so_dau",
    "chi_so_cuoi",
    "dien_nang_tieu_thu",
    "muc_pin_bat_dau",
    "muc_pin_ket_thuc",
    "dong",
    "dien_ap",
    "cong_suat",
    "vi_do",
    "kinh_do",
]


def _to_datetime(series: pd.Series) -> pd.Series:
    """Parse multiple date formats, prioritizing day-first Vietnamese exports."""
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if parsed.isna().mean() > 0.6:
        parsed_alt = pd.to_datetime(series, errors="coerce", dayfirst=False)
        if parsed_alt.notna().sum() > parsed.notna().sum():
            parsed = parsed_alt
    return parsed


def _clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )


def clean_and_engineer(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw charging logs and create dashboard features."""
    if raw.empty:
        return raw.copy()

    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in DATETIME_COLUMNS:
        if col in df.columns:
            df[col] = _to_datetime(df[col])

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [
        "ma_tram", "ma_tru", "ma_cong", "trang_thai", "li_do_dung_sac",
        "loai_tru", "ten_tram", "mien", "tinh_thanh", "loai_tram",
        "loai_mat_bang", "nam_thang"
    ]:
        if col in df.columns:
            df[col] = _clean_text(df[col])

    # Core feature names used by the app.
    if "dien_nang_tieu_thu" in df.columns:
        df["energy_kwh"] = df["dien_nang_tieu_thu"]
    else:
        df["energy_kwh"] = np.nan

    if "thoi_gian_bat_dau" in df.columns and "thoi_gian_ket_thuc" in df.columns:
        df["duration_minutes"] = (
            df["thoi_gian_ket_thuc"] - df["thoi_gian_bat_dau"]
        ).dt.total_seconds() / 60
    elif "thoi_gian_bat_dau" in df.columns and "thoi_gian_dung_sac" in df.columns:
        df["duration_minutes"] = (
            df["thoi_gian_dung_sac"] - df["thoi_gian_bat_dau"]
        ).dt.total_seconds() / 60
    else:
        df["duration_minutes"] = np.nan

    if "muc_pin_bat_dau" in df.columns and "muc_pin_ket_thuc" in df.columns:
        df["soc_delta"] = df["muc_pin_ket_thuc"] - df["muc_pin_bat_dau"]
        df["scheduling_flexibility_raw"] = 100 - df["muc_pin_bat_dau"]
    else:
        df["soc_delta"] = np.nan
        df["scheduling_flexibility_raw"] = np.nan

    df["avg_power_estimated_kw"] = np.where(
        df["duration_minutes"] > 0,
        df["energy_kwh"] / (df["duration_minutes"] / 60),
        np.nan,
    )

    if "thoi_gian_bat_dau" in df.columns:
        df["date"] = df["thoi_gian_bat_dau"].dt.date
        df["hour"] = df["thoi_gian_bat_dau"].dt.hour
        df["day_name"] = df["thoi_gian_bat_dau"].dt.day_name()
        df["weekday"] = df["thoi_gian_bat_dau"].dt.weekday
        df["is_weekend"] = df["weekday"].isin([5, 6])
        df["month"] = df["thoi_gian_bat_dau"].dt.to_period("M").astype(str)
    else:
        df["date"] = pd.NaT
        df["hour"] = np.nan
        df["day_name"] = pd.NA
        df["weekday"] = np.nan
        df["is_weekend"] = False
        df["month"] = pd.NA

    # Normalize location names for filtering.
    for col, default in [
        ("mien", "Unknown Region"),
        ("tinh_thanh", "Unknown Province"),
        ("loai_tru", "Unknown Charger"),
        ("loai_tram", "Unknown Station Type"),
        ("trang_thai", "Unknown Status"),
        ("ma_tram", "Unknown Station"),
        ("ten_tram", "Unknown Station Name"),
    ]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].astype("string").fillna(default)

    # Rule-based anomaly flags.
    anomaly_reasons = []
    anomaly_reasons.append(df["duration_minutes"].isna())
    anomaly_reasons.append(df["duration_minutes"] <= 0)
    anomaly_reasons.append(df["duration_minutes"] > 24 * 60)
    anomaly_reasons.append(df["energy_kwh"].isna())
    anomaly_reasons.append(df["energy_kwh"] <= 0)
    anomaly_reasons.append(df["energy_kwh"] > 500)
    anomaly_reasons.append(df["avg_power_estimated_kw"] > 400)

    if "muc_pin_bat_dau" in df.columns:
        anomaly_reasons.append((df["muc_pin_bat_dau"] < 0) | (df["muc_pin_bat_dau"] > 100))
    if "muc_pin_ket_thuc" in df.columns:
        anomaly_reasons.append((df["muc_pin_ket_thuc"] < 0) | (df["muc_pin_ket_thuc"] > 100))
    if "soc_delta" in df.columns:
        anomaly_reasons.append(df["soc_delta"] < -2)

    anomaly_matrix = np.column_stack([r.fillna(False).to_numpy() for r in anomaly_reasons])
    df["is_rule_anomaly"] = anomaly_matrix.any(axis=1)
    df["is_anomaly"] = df["is_rule_anomaly"]

    # Valid geo points.
    df["has_valid_geo"] = (
        df.get("vi_do", pd.Series(np.nan, index=df.index)).between(8, 24)
        & df.get("kinh_do", pd.Series(np.nan, index=df.index)).between(102, 110)
    )

    return df
