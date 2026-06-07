from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shinywidgets import output_widget, render_widget

from modules.data_loader import load_monthly_csvs
from modules.preprocessing import clean_and_engineer
from modules.ml import add_ml_anomaly_score
from modules.metrics import kpi_summary, insight_summary
from modules.forecasting import (
    forecast_timeseries,
    forecast_sessions_level,
    METHOD_LABELS,
    METHOD_FAMILIES,
)
from modules.charts import (
    behavioral_scatter,
    charger_type_mix,
    forecast_session_chart,
    forecast_timeseries_chart,
    smart_charging_opportunity_matrix,
    spatial_demand_map,
    station_utilization_ranking,
    temporal_load_profile,
)

DATA_DIR = Path(__file__).parent / "data"

# Horizon -> human label (resolution follows the agreed adaptive rule)
HORIZON_CHOICES = {
    "1H": "Next 1 hour (15-min steps)",
    "4H": "Next 4 hours (30-min steps)",
    "1D": "Next 1 day (hourly steps)",
    "1W": "Next 1 week (3-hour steps)",
    "1M": "Next 1 month (daily steps)",
}

# Build one checkbox group per non-empty method family (grouped UI).
_METHOD_DEFAULTS = {"stat": ["naive"], "ml": ["ridge"]}
_METHOD_GROUP_KEYS = [k for k, (_, m) in METHOD_FAMILIES.items() if m]


def _method_group_inputs():
    items = []
    for key, (label, methods) in METHOD_FAMILIES.items():
        if not methods:
            continue
        items.append(ui.input_checkbox_group(
            f"methods_{key}", label,
            choices={m: METHOD_LABELS[m] for m in methods},
            selected=_METHOD_DEFAULTS.get(key, []),
        ))
    return items


def _method_group_widths():
    n = max(1, len(_METHOD_GROUP_KEYS))
    return [max(2, 12 // n)] * n


def load_data_once() -> pd.DataFrame:
    raw = load_monthly_csvs(DATA_DIR)
    if raw.empty:
        return pd.DataFrame()
    df = clean_and_engineer(raw)
    df = add_ml_anomaly_score(df)
    return df


DATA = load_data_once()


def choices_from(df: pd.DataFrame, column: str, all_label: str = "All") -> list[str]:
    if df.empty or column not in df.columns:
        return [all_label]
    values = sorted([str(x) for x in df[column].dropna().unique()])
    return [all_label] + values


def date_range_from(df: pd.DataFrame) -> tuple[date, date]:
    if df.empty or "thoi_gian_bat_dau" not in df.columns or df["thoi_gian_bat_dau"].dropna().empty:
        today = date.today()
        return today, today
    return df["thoi_gian_bat_dau"].min().date(), df["thoi_gian_bat_dau"].max().date()


START_DATE, END_DATE = date_range_from(DATA)

app_ui = ui.page_fluid(
    ui.include_css("www/styles.css"),
    ui.div(
        {"class": "app-shell"},
        ui.div(
            {"class": "dashboard-header"},
            ui.h1("Vietnam EV Charging Visual Analytics Dashboard", class_="dashboard-title"),
            ui.div("From charging logs to smart charging opportunities", class_="dashboard-subtitle"),
        ),
        ui.div(
            {"class": "filter-card"},
            ui.div("Global Filters", class_="filter-title"),
            ui.layout_columns(
                ui.input_date_range("date_range", "Date Range", start=START_DATE, end=END_DATE, min=START_DATE, max=END_DATE, format="yyyy-mm-dd"),
                ui.input_selectize("region", "Region", choices=choices_from(DATA, "mien", "All Regions")),
                ui.input_selectize("province", "Province", choices=choices_from(DATA, "tinh_thanh", "All Provinces")),
                ui.input_selectize("station_name", "Station Name", choices=choices_from(DATA, "ten_tram", "All Stations")),
                ui.input_selectize("charger_type", "Charger Type", choices=choices_from(DATA, "loai_tru", "All Types")),
                ui.input_selectize("station_type", "Station Type", choices=choices_from(DATA, "loai_tram", "All Types")),
                ui.input_selectize("venue_type", "Venue Type", choices=choices_from(DATA, "loai_mat_bang", "All Venues")),
                ui.input_select("anomaly_mode", "Anomaly Toggle", choices={"all": "Show All", "normal": "Normal Only", "anomaly": "Anomalies Only"}, selected="all"),
                col_widths=(4, 2, 2, 4, 3, 3, 3, 3),
            ),
        ),
        ui.div(
            {"class": "kpi-grid"},
            ui.div({"class": "kpi-card kpi-blue"}, ui.div("Total Charging Sessions", class_="kpi-title"), ui.div(ui.output_text("kpi_sessions"), class_="kpi-value"), ui.div("Filtered charging sessions", class_="kpi-caption")),
            ui.div({"class": "kpi-card kpi-green"}, ui.div("Total Energy Delivered", class_="kpi-title"), ui.div(ui.output_text("kpi_energy"), class_="kpi-value"), ui.div("Energy consumed by selected sessions", class_="kpi-caption")),
            ui.div({"class": "kpi-card kpi-orange"}, ui.div("Infrastructure Coverage", class_="kpi-title"), ui.div(ui.output_text("kpi_infrastructure"), class_="kpi-value"), ui.div("Stations / Chargers", class_="kpi-caption")),
            ui.div({"class": "kpi-card kpi-red"}, ui.div("Anomaly Rate", class_="kpi-title"), ui.div(ui.output_text("kpi_anomaly"), class_="kpi-value"), ui.div("Rule-based + optional ML anomalies", class_="kpi-caption")),
        ),
        ui.layout_columns(
            ui.div({"class": "chart-card"}, ui.div("1. Spatial Demand Map", class_="chart-title"), ui.div("Bubble map by station location, sessions, and energy delivered.", class_="chart-subtitle"), output_widget("spatial_map")),
            ui.div({"class": "chart-card"}, ui.div("2. Temporal Load Profile", class_="chart-title"), ui.div("Hourly demand curve and weekday-hour heatmap.", class_="chart-subtitle"), output_widget("temporal_profile")),
            col_widths=(6, 6),
        ),
        ui.layout_columns(
            ui.div({"class": "chart-card"}, ui.div("3. Behavioral Analysis", class_="chart-title"), ui.div("Session duration vs. energy delivered, colored by charger type.", class_="chart-subtitle"), output_widget("behavior_scatter")),
            ui.div({"class": "chart-card"}, ui.div("4. Charger Type Mix", class_="chart-title"), ui.div("Where sessions and energy concentrate across charger types.", class_="chart-subtitle"), output_widget("charger_mix")),
            col_widths=(6, 6),
        ),
        ui.layout_columns(
            ui.div({"class": "chart-card"}, ui.div("5. Infrastructure Utilization Ranking", class_="chart-title"), ui.div("Top stations by utilization proxy based on sessions, energy, and duration.", class_="chart-subtitle"), output_widget("utilization_ranking")),
            ui.div({"class": "chart-card"}, ui.div("6. Smart Charging Opportunity Matrix", class_="chart-title"), ui.div("Priority zones based on peak demand intensity and scheduling flexibility.", class_="chart-subtitle"), output_widget("opportunity_matrix")),
            col_widths=(6, 6),
        ),
        ui.div(
            {"class": "filter-card"},
            ui.div("Forecast Settings", class_="filter-title"),
            ui.layout_columns(
                ui.input_date("forecast_cutoff", "Cutoff Date", value=END_DATE, min=START_DATE, max=END_DATE, format="yyyy-mm-dd"),
                ui.input_slider("cutoff_hour", "Cutoff Hour", min=0, max=23, value=23, step=1),
                ui.input_select("horizon_mode", "Horizon Mode", choices={"preset": "Preset", "custom": "Custom"}, selected="preset"),
                ui.input_select("lookback", "Training Window", choices={"all": "All history", "30": "Last 30 days", "60": "Last 60 days", "90": "Last 90 days"}, selected="all"),
                col_widths=(3, 3, 3, 3),
            ),
            ui.layout_columns(
                # Preset horizon shows only in Preset mode
                ui.panel_conditional(
                    "input.horizon_mode === 'preset'",
                    ui.input_select("forecast_horizon", "Preset Horizon", choices=HORIZON_CHOICES, selected="1D"),
                ),
                # Custom length/unit are enabled (shown) only in Custom mode
                ui.panel_conditional(
                    "input.horizon_mode === 'custom'",
                    ui.layout_columns(
                        ui.input_numeric("horizon_n", "Custom length", value=12, min=1, max=3650),
                        ui.input_select("horizon_unit", "Custom unit", choices={"hour": "hours", "day": "days", "week": "weeks", "month": "months"}, selected="hour"),
                        col_widths=(6, 6),
                    ),
                ),
                col_widths=(4, 8),
            ),
            ui.div("Methods (pick 1 or more), grouped by family", class_="filter-title"),
            ui.layout_columns(*_method_group_inputs(), col_widths=_method_group_widths()),
            ui.layout_columns(
                ui.input_action_button("predict_btn", "Forecast", class_="btn btn-primary"),
                ui.div("Custom length/unit apply only when Horizon Mode = Custom (step auto-selected). Forecasts follow ALL global filters including the Abnormal toggle; only the Date Range slider is ignored, since Cutoff Date + Cutoff Hour + Horizon define the window. Training Window limits how far back the models learn.", class_="chart-subtitle"),
                col_widths=(3, 9),
            ),
        ),
        ui.layout_columns(
            ui.div({"class": "chart-card"}, ui.div("7. Forecast: Number of EVs (sessions)", class_="chart-title"), ui.div("Predicted session count per time step, per method, plus ensemble mean and actual.", class_="chart-subtitle"), output_widget("forecast_sessions")),
            ui.div({"class": "chart-card"}, ui.div("8. Forecast: Total Energy Demand", class_="chart-title"), ui.div("Predicted aggregate energy (kWh) per time step, per method, plus ensemble and actual.", class_="chart-subtitle"), output_widget("forecast_energy")),
            col_widths=(6, 6),
        ),
        # ============================================================================
        # Charts 9 & 10 (PER-SESSION forecasts). This is a harder problem than #7/#8.
        # TO TEMPORARILY HIDE these two charts (without deleting), comment out:
        #   (a) this entire ui.layout_columns(...) block below, AND
        #   (b) in the server: the renders forecast_sess_energy / forecast_sess_duration
        #       (the two "sess_energy"/"sess_duration" lines in forecast_results may be
        #        left as-is or commented to save compute).
        # ============================================================================
        ui.layout_columns(
            ui.div({"class": "chart-card"}, ui.div("9. Forecast: Energy per Session", class_="chart-title"), ui.div("Per-session kWh predicted vs actual for sessions in the horizon window.", class_="chart-subtitle"), output_widget("forecast_sess_energy")),
            ui.div({"class": "chart-card"}, ui.div("10. Forecast: Charging Duration per Session", class_="chart-title"), ui.div("Per-session duration (minutes) predicted vs actual in the horizon window.", class_="chart-subtitle"), output_widget("forecast_sess_duration")),
            col_widths=(6, 6),
        ),
        # ============================ end charts 9 & 10 =============================
        ui.div(
            {"class": "insight-panel"},
            ui.div("Decision Support Insights", class_="chart-title"),
            ui.div(
                {"class": "insight-grid"},
                ui.div({"class": "insight-card"}, ui.div("Peak Hours", class_="insight-label"), ui.div(ui.output_text("insight_peak_hours"), class_="insight-text")),
                ui.div({"class": "insight-card"}, ui.div("Hotspot Provinces", class_="insight-label"), ui.div(ui.output_text("insight_hotspots"), class_="insight-text")),
                ui.div({"class": "insight-card"}, ui.div("Abnormal Patterns", class_="insight-label"), ui.div(ui.output_text("insight_abnormal"), class_="insight-text")),
                ui.div({"class": "insight-card"}, ui.div("Smart Charging Candidates", class_="insight-label"), ui.div(ui.output_text("insight_candidates"), class_="insight-text")),
            ),
        ),
    ),
    title="Vietnam EV Charging Analytics",
)


def apply_non_date_filters(df: pd.DataFrame, input: Inputs) -> pd.DataFrame:
    if df.empty:
        return df
    region = input.region()
    if region and region != "All Regions":
        df = df[df["mien"].astype(str) == region]
    province = input.province()
    if province and province != "All Provinces":
        df = df[df["tinh_thanh"].astype(str) == province]
    station_name = input.station_name()
    if station_name and station_name != "All Stations":
        df = df[df["ten_tram"].astype(str) == station_name]
    charger_type = input.charger_type()
    if charger_type and charger_type != "All Types":
        df = df[df["loai_tru"].astype(str) == charger_type]
    station_type = input.station_type()
    if station_type and station_type != "All Types":
        df = df[df["loai_tram"].astype(str) == station_type]
    venue_type = input.venue_type()
    if venue_type and venue_type != "All Venues":
        df = df[df["loai_mat_bang"].astype(str) == venue_type]
    return df


def server(input: Inputs, output: Outputs, session: Session):
    @reactive.Calc
    def forecast_base_data() -> pd.DataFrame:
        # Do not apply Date Range here (forecast validation needs records after cutoff),
        # but DO follow every other global filter, including the Abnormal toggle, so the
        # training data is controlled globally.
        df = apply_non_date_filters(DATA.copy(), input)
        if df.empty:
            return df
        anomaly_mode = input.anomaly_mode()
        if anomaly_mode == "normal":
            df = df[~df["is_anomaly"]]
        elif anomaly_mode == "anomaly":
            df = df[df["is_anomaly"]]
        return df

    @reactive.Calc
    def filtered_data() -> pd.DataFrame:
        df = apply_non_date_filters(DATA.copy(), input)
        if df.empty:
            return df
        dr = input.date_range()
        if dr and "thoi_gian_bat_dau" in df.columns:
            start, end = dr
            if start is not None:
                df = df[df["thoi_gian_bat_dau"].dt.date >= start]
            if end is not None:
                df = df[df["thoi_gian_bat_dau"].dt.date <= end]
        anomaly_mode = input.anomaly_mode()
        if anomaly_mode == "normal":
            df = df[~df["is_anomaly"]]
        elif anomaly_mode == "anomaly":
            df = df[df["is_anomaly"]]
        return df

    @reactive.Effect
    def _update_province_choices():
        df = DATA.copy()
        region = input.region()
        if not df.empty and region and region != "All Regions":
            df = df[df["mien"].astype(str) == region]
        ui.update_selectize("province", choices=choices_from(df, "tinh_thanh", "All Provinces"), selected="All Provinces", session=session)

    @reactive.Effect
    def _update_station_choices():
        df = DATA.copy()
        region = input.region()
        if not df.empty and region and region != "All Regions":
            df = df[df["mien"].astype(str) == region]
        province = input.province()
        if not df.empty and province and province != "All Provinces":
            df = df[df["tinh_thanh"].astype(str) == province]
        ui.update_selectize("station_name", choices=choices_from(df, "ten_tram", "All Stations"), selected="All Stations", session=session)

    @reactive.Calc
    def kpis() -> dict:
        return kpi_summary(filtered_data())

    @output
    @render.text
    def kpi_sessions():
        return kpis()["sessions"]

    @output
    @render.text
    def kpi_energy():
        return kpis()["energy"]

    @output
    @render.text
    def kpi_infrastructure():
        return kpis()["infrastructure"]

    @output
    @render.text
    def kpi_anomaly():
        return kpis()["anomaly_rate"]

    @output
    @render_widget
    def spatial_map():
        return spatial_demand_map(filtered_data())

    @output
    @render_widget
    def temporal_profile():
        return temporal_load_profile(filtered_data())

    @output
    @render_widget
    def behavior_scatter():
        return behavioral_scatter(filtered_data())

    @output
    @render_widget
    def charger_mix():
        return charger_type_mix(filtered_data())

    @output
    @render_widget
    def utilization_ranking():
        return station_utilization_ranking(filtered_data())

    @output
    @render_widget
    def opportunity_matrix():
        return smart_charging_opportunity_matrix(filtered_data())

    # ---- Forecasting: compute ONLY when the Forecast button is pressed ----
    @reactive.Calc
    def forecast_results():
        input.predict_btn()  # take a dependency on the button
        if input.predict_btn() == 0:
            return None
        with reactive.isolate():
            base = forecast_base_data()
            cutoff = input.forecast_cutoff()
            cutoff_hour = input.cutoff_hour()
            if input.horizon_mode() == "custom":
                horizon = f"CUSTOM:{int(input.horizon_n() or 1)}:{input.horizon_unit()}"
            else:
                horizon = input.forecast_horizon()
            # Gather selected methods from every family checkbox group
            methods = []
            for key in _METHOD_GROUP_KEYS:
                methods += list(input[f"methods_{key}"]())
            lb = input.lookback()
            lookback_days = None if lb == "all" else int(lb)
        if not methods:
            methods = ["naive"]
        return {
            "sessions": forecast_timeseries(base, cutoff, horizon, "sessions", methods, lookback_days, cutoff_hour),
            "energy": forecast_timeseries(base, cutoff, horizon, "energy", methods, lookback_days, cutoff_hour),
            "sess_energy": forecast_sessions_level(base, cutoff, horizon, "energy_kwh", methods, lookback_days, cutoff_hour),
            "sess_duration": forecast_sessions_level(base, cutoff, horizon, "duration_minutes", methods, lookback_days, cutoff_hour),
        }

    @output
    @render_widget
    def forecast_sessions():
        res = forecast_results()
        return forecast_timeseries_chart(res["sessions"] if res else None, "Number of Charging Sessions")

    @output
    @render_widget
    def forecast_energy():
        res = forecast_results()
        return forecast_timeseries_chart(res["energy"] if res else None, "Energy Demand (kWh)")

    @output
    @render_widget
    def forecast_sess_energy():
        res = forecast_results()
        return forecast_session_chart(res["sess_energy"] if res else None, "Energy per session (kWh)")

    @output
    @render_widget
    def forecast_sess_duration():
        res = forecast_results()
        return forecast_session_chart(res["sess_duration"] if res else None, "Duration (minutes)")

    @reactive.Calc
    def insights() -> dict:
        return insight_summary(filtered_data())

    @output
    @render.text
    def insight_peak_hours():
        return insights()["peak_hours"]

    @output
    @render.text
    def insight_hotspots():
        return insights()["hotspots"]

    @output
    @render.text
    def insight_abnormal():
        return insights()["abnormal"]

    @output
    @render.text
    def insight_candidates():
        return insights()["smart_candidates"]


app = App(app_ui, server)