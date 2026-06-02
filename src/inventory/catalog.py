"""Build SKU catalog from uploaded sales CSV."""

from __future__ import annotations

import pandas as pd

from src.data.data_preprocessor import aggregate_daily_sku, clean_sales_data


def load_sales(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = df.columns.str.lower().str.strip()
    return clean_sales_data(df)


def sku_column(df: pd.DataFrame) -> str | None:
    for col in ("sku", "product_id", "product", "item_id"):
        if col in df.columns:
            return col
    return None


def quantity_column(df: pd.DataFrame) -> str:
    if "quantity" in df.columns:
        return "quantity"
    return "sales"


def build_catalog(sales_df: pd.DataFrame) -> pd.DataFrame:
    sku_col = sku_column(sales_df)
    if not sku_col:
        raise ValueError("Sales data must include product_id (or sku) for inventory forecasting.")

    sku_daily = aggregate_daily_sku(sales_df) if sku_col == "product_id" else sales_df
    if sku_col != "product_id":
        sku_daily = sku_daily.rename(columns={sku_col: "sku"})

    qty_col = quantity_column(sku_daily)
    sku_daily[qty_col] = pd.to_numeric(sku_daily[qty_col], errors="coerce").fillna(0)

    agg_spec = {
        "category": ("category", "first"),
        "total_qty": (qty_col, "sum"),
        "last_sale": ("date", "max"),
    }
    if "price" in sku_daily.columns:
        agg_spec["unit_cost"] = ("price", "mean")

    catalog = sku_daily.groupby("sku", as_index=False).agg(**agg_spec)
    catalog["name"] = catalog["sku"]
    if "unit_cost" not in catalog.columns:
        catalog["unit_cost"] = 0.0

    return catalog
