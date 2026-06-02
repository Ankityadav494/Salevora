"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import BASE_DIR, get_data_path


def load_sales_csv(path: Path | str | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path else get_data_path("processed_data_path")
    if not csv_path.exists():
        fallback = BASE_DIR / "data" / "processed" / "live_sales.csv"
        if fallback.exists():
            csv_path = fallback
        else:
            raw = BASE_DIR / "data" / "raw" / "sales_data.csv"
            if raw.exists():
                csv_path = raw
            else:
                raise FileNotFoundError("No sales data file found.")

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df.columns = df.columns.str.lower().str.strip()
    return df.sort_values("date").reset_index(drop=True)


def aggregate_weekly(df: pd.DataFrame, value_col: str = "sales") -> pd.DataFrame:
    work = df.copy()
    work["week"] = work["date"].dt.to_period("W").apply(lambda p: p.start_time)
    agg = {value_col: "sum"}
    if "revenue" in work.columns:
        agg["revenue"] = "sum"
    weekly = work.groupby("week", as_index=False).agg(agg)
    weekly = weekly.rename(columns={"week": "date"})
    return weekly.sort_values("date").reset_index(drop=True)
