# \
from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_csv_safely(path: Path) -> pd.DataFrame:
    """Read one CSV with common encodings used in Vietnamese operational exports."""
    encodings = ["utf-8-sig", "utf-8", "cp1258", "latin1"]
    last_error: Exception | None = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Cannot read {path}. Last error: {last_error}")


def load_monthly_csvs(data_dir: str | Path = "data") -> pd.DataFrame:
    """Load all monthly CSV files in the data folder and concatenate them."""
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.csv"))

    if not files:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for file in files:
        df = read_csv_safely(file)
        df["source_file"] = file.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)
