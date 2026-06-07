# Data Directory

This directory is reserved for the EV charging session CSV files used by the dashboard.

The full dataset is **not included in this repository** because it contains private/proprietary operational charging-session records from Vgreen/VinFast. The data may include station-level operational information, charging-session timestamps, charger identifiers, location-related fields, and other business-sensitive attributes. Therefore, only the source code and documentation are shared publicly.

## Expected data format

To run the dashboard locally with the full dataset, place the monthly CSV files in this folder:

```text
data/
  2024-01.csv
  2024-02.csv
  2024-03.csv
  ...
````

The application automatically loads all `.csv` files in this directory and concatenates them into a single dataframe.

## Required columns

The dashboard expects charging-session records with columns such as:

* `thoi_gian_bat_dau`
* `thoi_gian_ket_thuc`
* `dien_nang_tieu_thu`
* `muc_pin_bat_dau`
* `muc_pin_ket_thuc`
* `ma_tram`
* `ten_tram`
* `ma_tru`
* `loai_tru`
* `loai_tram`
* `loai_mat_bang`
* `mien`
* `tinh_thanh`
* `vi_do`
* `kinh_do`

The preprocessing module will clean these fields and generate additional dashboard features such as:

* `energy_kwh`
* `duration_minutes`
* `soc_delta`
* `avg_power_estimated_kw`
* `hour`
* `weekday`
* `is_weekend`
* `has_valid_geo`
* `is_anomaly`
