"""Data cleaning and preprocessing."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLS = {"date", "sales"}


def validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLS - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")


def _fix_product_categories(df: pd.DataFrame) -> pd.DataFrame:
    """One stable category per product_id (mode of observed categories)."""
    if "product_id" not in df.columns:
        return df
    work = df.copy()
    work["product_id"] = work["product_id"].astype(str).str.strip()

    def _mode_cat(series: pd.Series) -> str:
        modes = series.mode()
        return str(modes.iloc[0]) if len(modes) else str(series.iloc[-1])

    cat_map = work.groupby("product_id")["category"].agg(_mode_cat)
    work["category"] = work["product_id"].map(cat_map).fillna(work["category"])
    return work


def _dedupe_product_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate date + product_id rows by summing numeric fields."""
    if "product_id" not in df.columns:
        return df

    spec: dict = {}
    if "sales" in df.columns:
        spec["sales"] = ("sales", "sum")
    if "revenue" in df.columns:
        spec["revenue"] = ("revenue", "sum")
    if "quantity" in df.columns:
        spec["quantity"] = ("quantity", "sum")
    if "price" in df.columns:
        spec["price"] = ("price", "mean")
    if "category" in df.columns:
        spec["category"] = ("category", "first")

    return (
        df.groupby(["date", "product_id"], as_index=False)
        .agg(**spec)
        .sort_values(["date", "product_id"])
        .reset_index(drop=True)
    )


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work.columns = work.columns.str.lower().str.strip()
    validate_schema(work)

    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"])
    work["sales"] = pd.to_numeric(work["sales"], errors="coerce").fillna(0)

    if "revenue" not in work.columns:
        work["revenue"] = work["sales"]
    else:
        work["revenue"] = pd.to_numeric(work["revenue"], errors="coerce").fillna(work["sales"])

    if "quantity" not in work.columns:
        work["quantity"] = work["sales"]
    else:
        work["quantity"] = pd.to_numeric(work["quantity"], errors="coerce").fillna(0)

    if "category" not in work.columns:
        work["category"] = "Uncategorised"

    if "product_id" in work.columns:
        work = _fix_product_categories(work)
        work = _dedupe_product_rows(work)

    # Drop outlier days (extreme daily totals)
    date_sales = work.groupby("date")["sales"].sum()
    q1, q3 = date_sales.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 3 * iqr
    lower = max(0, q1 - 3 * iqr)
    valid_dates = date_sales[(date_sales >= lower) & (date_sales <= upper)].index
    work = work[work["date"].isin(valid_dates)]

    return work.sort_values("date").reset_index(drop=True)


def aggregate_daily_company(df: pd.DataFrame) -> pd.DataFrame:
    """Company-level daily totals — use for revenue/sales dashboard forecasting."""
    agg: dict = {"sales": "sum", "revenue": "sum", "quantity": "sum"}
    daily = df.groupby("date", as_index=False).agg({k: agg[k] for k in agg if k in df.columns})
    return daily.sort_values("date").reset_index(drop=True)


def aggregate_daily_sku(df: pd.DataFrame) -> pd.DataFrame:
    """One row per SKU per day — use for quantity/inventory forecasting."""
    if "product_id" not in df.columns:
        raise ValueError("product_id required for SKU-level aggregation")

    spec: dict = {
        "quantity": ("quantity", "sum"),
        "sales": ("sales", "sum"),
        "revenue": ("revenue", "sum"),
        "category": ("category", "first"),
    }
    if "price" in df.columns:
        spec["price"] = ("price", "mean")

    out = (
        df.groupby(["date", "product_id"], as_index=False)
        .agg(**{k: v for k, v in spec.items() if k == "category" or k in df.columns})
    )
    return out.rename(columns={"product_id": "sku"}).sort_values(["date", "sku"]).reset_index(drop=True)


def train_test_split_ts(
    series: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.Series, pd.Series]:
    n = len(series)
    split = max(1, int(n * (1 - test_size)))
    return series.iloc[:split], series.iloc[split:]
