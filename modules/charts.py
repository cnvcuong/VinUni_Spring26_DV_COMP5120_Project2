from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .metrics import opportunity_by_province
from .forecasting import error_text, METHOD_LABELS

PLOTLY_TEMPLATE = "plotly_white"

# Palette aligned with www/styles.css
METHOD_COLORS = {
    "naive": "#d98a12",
    "seasonal_mean": "#6956b8",
    "ewm_seasonal": "#0fa3a3",
    "ridge": "#2b6fd6",
    "rf": "#19a974",
    "hgb": "#b5179e",
}
ACTUAL_COLOR = "#25345f"
ENSEMBLE_COLOR = "#d43f3a"


def empty_figure(message: str = "No data available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font=dict(size=18))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=360)
    return fig


# ============================================================
# Charts 1-3 and 5-6 (kept unchanged)
# ============================================================

def spatial_demand_map(df: pd.DataFrame) -> go.Figure:
    d = df[df.get("has_valid_geo", False)].copy()
    if d.empty:
        return empty_figure("No valid station coordinates in current selection")
    station = d.groupby(["ma_tram", "ten_tram", "tinh_thanh", "vi_do", "kinh_do"], dropna=False).agg(
        sessions=("ma_tram", "size"), energy_kwh=("energy_kwh", "sum"), anomaly_rate=("is_anomaly", "mean")
    ).reset_index()
    fig = px.scatter_mapbox(
        station, lat="vi_do", lon="kinh_do", size="sessions", color="energy_kwh", hover_name="ten_tram",
        hover_data={"ma_tram": True, "tinh_thanh": True, "sessions": ":,", "energy_kwh": ":,.1f", "anomaly_rate": ":.1%", "vi_do": False, "kinh_do": False},
        zoom=4.7, height=440, mapbox_style="open-street-map", color_continuous_scale="Blues"
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=0, r=0, t=10, b=0), coloraxis_colorbar_title="kWh")
    return fig


def temporal_load_profile(df: pd.DataFrame) -> go.Figure:
    if df.empty or "hour" not in df.columns:
        return empty_figure()
    hourly = df.groupby("hour", dropna=True).agg(sessions=("hour", "size"), energy_kwh=("energy_kwh", "sum")).reset_index()
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat = df.pivot_table(index="day_name", columns="hour", values="energy_kwh", aggfunc="sum", fill_value=0)
    heat = heat.reindex(weekdays).fillna(0).reindex(columns=list(range(24)), fill_value=0)
    fig = make_subplots(rows=2, cols=1, row_heights=[0.48, 0.52], vertical_spacing=0.18,
                        subplot_titles=("Average charging demand by hour", "Energy heatmap by weekday and hour"))
    fig.add_trace(go.Scatter(x=hourly["hour"], y=hourly["energy_kwh"], mode="lines+markers", name="Energy delivered"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale="Blues", colorbar=dict(title="kWh")), row=2, col=1)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=440, margin=dict(l=20, r=20, t=50, b=20))
    fig.update_xaxes(title_text="Hour of day", row=2, col=1)
    fig.update_yaxes(title_text="Energy (kWh)", row=1, col=1)
    return fig


def behavioral_scatter(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["duration_minutes", "energy_kwh"]).copy()
    d = d[(d["duration_minutes"] > 0) & (d["energy_kwh"] > 0)]
    if d.empty:
        return empty_figure()
    if len(d) > 50000:
        d = d.sample(50000, random_state=42)
    fig = px.scatter(
        d, x="duration_minutes", y="energy_kwh", color="loai_tru", symbol="is_anomaly", opacity=0.65,
        hover_data={"ma_tram": True, "ten_tram": True, "tinh_thanh": True, "duration_minutes": ":.1f", "energy_kwh": ":.2f", "avg_power_estimated_kw": ":.1f", "is_anomaly": True},
        labels={"duration_minutes": "Duration (minutes)", "energy_kwh": "Energy Delivered (kWh)", "loai_tru": "Charger Type", "is_anomaly": "Anomaly"},
        height=440, template=PLOTLY_TEMPLATE
    )
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), legend_title_text="Charger Type")
    return fig


def station_utilization_ranking(df: pd.DataFrame) -> go.Figure:
    if df.empty or "ma_tram" not in df.columns:
        return empty_figure()
    station = df.groupby(["ma_tram", "ten_tram", "tinh_thanh"], dropna=False).agg(
        sessions=("ma_tram", "size"), energy_kwh=("energy_kwh", "sum"), avg_duration=("duration_minutes", "mean"), anomaly_rate=("is_anomaly", "mean")
    ).reset_index()
    station["utilization_score"] = (station["sessions"].rank(pct=True) * 0.5 + station["energy_kwh"].rank(pct=True) * 0.4 + station["avg_duration"].rank(pct=True) * 0.1) * 100
    top = station.sort_values("utilization_score", ascending=False).head(12).sort_values("utilization_score")
    fig = px.bar(top, x="utilization_score", y="ten_tram", orientation="h", hover_data={"ma_tram": True, "tinh_thanh": True, "sessions": ":,", "energy_kwh": ":,.1f", "anomaly_rate": ":.1%"}, labels={"utilization_score": "Utilization Score", "ten_tram": "Station"}, height=400, template=PLOTLY_TEMPLATE)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
    return fig


def smart_charging_opportunity_matrix(df: pd.DataFrame) -> go.Figure:
    opp = opportunity_by_province(df)
    if opp.empty:
        return empty_figure()
    fig = px.scatter(
        opp, x="peak_share_norm", y="flexibility_norm", size="sessions", color="priority_score", hover_name="tinh_thanh",
        hover_data={"sessions": ":,", "energy_kwh": ":,.1f", "peak_share": ":.1%", "flexibility": ":.1f", "anomaly_rate": ":.1%", "priority_score": ":.2f"},
        labels={"peak_share_norm": "Peak Demand Intensity", "flexibility_norm": "Scheduling Flexibility", "priority_score": "Priority"},
        color_continuous_scale="RdYlGn_r", height=400, template=PLOTLY_TEMPLATE
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.6)
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.6)
    fig.add_annotation(x=0.75, y=0.92, text="Immediate Intervention", showarrow=False)
    fig.add_annotation(x=0.25, y=0.92, text="Flexible Opportunity", showarrow=False)
    fig.add_annotation(x=0.25, y=0.08, text="Low Priority", showarrow=False)
    fig.add_annotation(x=0.75, y=0.08, text="Demand Response Zone", showarrow=False)
    fig.update_xaxes(range=[-0.05, 1.05])
    fig.update_yaxes(range=[-0.05, 1.05])
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    return fig


# ============================================================
# Chart 4 REPLACEMENT for the (meaningless) status funnel.
# trang_thai / li_do_dung_sac are constant, so we show the
# charger-type mix instead: where sessions and energy concentrate.
# ============================================================

def charger_type_mix(df: pd.DataFrame) -> go.Figure:
    if df.empty or "loai_tru" not in df.columns:
        return empty_figure()
    g = df.groupby("loai_tru", dropna=False).agg(
        sessions=("loai_tru", "size"),
        energy_kwh=("energy_kwh", "sum"),
        avg_energy=("energy_kwh", "mean"),
    ).reset_index().sort_values("sessions", ascending=True)
    if g.empty:
        return empty_figure()
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.16,
        subplot_titles=("Sessions by charger type", "Total energy by charger type (kWh)"),
    )
    fig.add_trace(go.Bar(
        y=g["loai_tru"], x=g["sessions"], orientation="h", marker_color="#2b6fd6",
        hovertemplate="%{y}<br>Sessions: %{x:,}<extra></extra>",
        customdata=g[["avg_energy"]].to_numpy()), row=1, col=1)
    fig.add_trace(go.Bar(
        y=g["loai_tru"], x=g["energy_kwh"], orientation="h", marker_color="#19a974",
        hovertemplate="%{y}<br>Energy: %{x:,.1f} kWh<extra></extra>"), row=1, col=2)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=400, showlegend=False,
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ============================================================
# Charts 7-10: FORECASTING (4 methods + ensemble + actual)
# These consume tidy frames produced by modules.forecasting.
# ============================================================

def _methods_in(result: pd.DataFrame) -> list[str]:
    return result.attrs.get("methods", [c for c in METHOD_COLORS if c in result.columns])


def forecast_timeseries_chart(result: pd.DataFrame, y_title: str) -> go.Figure:
    if result is None or result.empty:
        return empty_figure("Select methods and press Predict")
    methods = _methods_in(result)
    timings = result.attrs.get("timings", {})
    actual = result["actual"]
    fig = go.Figure()
    # Actual observed values
    fig.add_trace(go.Scatter(
        x=result["timestamp"], y=actual, mode="lines+markers", name="Actual",
        line=dict(color=ACTUAL_COLOR, width=2),
        hovertemplate="%{x}<br>Actual: %{y:,.2f}<extra></extra>"))
    # Each selected method: label | MAE/MAPE | runtime
    for m in methods:
        name = f"{METHOD_LABELS.get(m, m)} | {error_text(actual, result[m])} | {timings.get(m, 0):.3f}s"
        fig.add_trace(go.Scatter(
            x=result["timestamp"], y=result[m], mode="lines", name=name,
            line=dict(color=METHOD_COLORS.get(m, "#888"), width=1.6),
            hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>"))
    # Ensemble (only meaningful when more than one method is selected)
    if len(methods) > 1:
        total = sum(timings.get(m, 0) for m in methods)
        name = f"Ensemble (mean) | {error_text(actual, result['ensemble'])} | {total:.3f}s"
        fig.add_trace(go.Scatter(
            x=result["timestamp"], y=result["ensemble"], mode="lines+markers", name=name,
            line=dict(color=ENSEMBLE_COLOR, width=3, dash="dash"),
            hovertemplate="%{x}<br>Ensemble: %{y:,.2f}<extra></extra>"))
    fig.update_layout(
        xaxis_title="Future time window", yaxis_title=y_title,
        template=PLOTLY_TEMPLATE, height=400, margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig


def forecast_session_chart(result: pd.DataFrame, value_label: str) -> go.Figure:
    if result is None or result.empty:
        return empty_figure("Select methods and press Forecast")
    methods = _methods_in(result)
    timings = result.attrs.get("timings", {})
    actual = pd.to_numeric(result["actual"], errors="coerce")
    fig = go.Figure()
    for m in methods:
        pred = pd.to_numeric(result[m], errors="coerce")
        name = f"{METHOD_LABELS.get(m, m)} | {error_text(actual, pred)} | {timings.get(m, 0):.3f}s"
        fig.add_trace(go.Scatter(
            x=actual, y=pred, mode="markers", name=name,
            marker=dict(color=METHOD_COLORS.get(m, "#888"), size=7, opacity=0.6),
            hovertemplate="Actual: %{x:,.2f}<br>Pred: %{y:,.2f}<extra></extra>"))
    if len(methods) > 1:
        ens = pd.to_numeric(result["ensemble"], errors="coerce")
        total = sum(timings.get(m, 0) for m in methods)
        name = f"Ensemble (mean) | {error_text(actual, ens)} | {total:.3f}s"
        fig.add_trace(go.Scatter(
            x=actual, y=ens, mode="markers", name=name,
            marker=dict(color=ENSEMBLE_COLOR, size=8, opacity=0.75, symbol="x"),
            hovertemplate="Actual: %{x:,.2f}<br>Ensemble: %{y:,.2f}<extra></extra>"))
    lo = float(np.nanmin(actual)) if actual.notna().any() else 0.0
    hi = float(np.nanmax(actual)) if actual.notna().any() else 1.0
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect (y=x)",
                             line=dict(color="gray", dash="dot"), hoverinfo="skip"))
    fig.update_layout(
        xaxis_title=f"Actual {value_label}", yaxis_title=f"Predicted {value_label}",
        template=PLOTLY_TEMPLATE, height=400, margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig