"""Demand estimation and per-SKU forecasting from sales history (quantity units)."""

from __future__ import annotations

import pandas as pd

from src.data.data_loader import aggregate_weekly
from src.data.data_preprocessor import aggregate_daily_sku
from src.inventory.catalog import quantity_column, sku_column
from src.models import ensemble_model


def _sku_series(sales_df: pd.DataFrame, sku: str) -> pd.Series:
    sku_col = sku_column(sales_df)
    qty_col = quantity_column(sales_df)
    if not sku_col:
        return pd.Series(dtype=float)

    if sku_col == "product_id":
        daily = aggregate_daily_sku(sales_df)
        rows = daily[daily["sku"] == sku]
    else:
        rows = sales_df[sales_df[sku_col] == sku].groupby("date")[qty_col].sum().reset_index()
        rows.columns = ["date", qty_col]

    if rows.empty:
        return pd.Series(dtype=float)

    col = qty_col if qty_col in rows.columns else "quantity"
    weekly = aggregate_weekly(rows.rename(columns={col: "quantity"}), "quantity")
    return weekly.set_index("date")["quantity"]


def daily_demand(sales_df: pd.DataFrame, sku: str, lookback_days: int = 30) -> float:
    sku_col = sku_column(sales_df)
    qty_col = quantity_column(sales_df)
    if not sku_col:
        return 0.0

    if sku_col == "product_id":
        daily = aggregate_daily_sku(sales_df)
        daily = daily[daily["sku"] == sku].groupby("date")[qty_col].sum()
    else:
        daily = sales_df[sales_df[sku_col] == sku].groupby("date")[qty_col].sum()

    if daily.empty:
        return 0.0

    cutoff = daily.index.max() - pd.Timedelta(days=lookback_days)
    recent = daily[daily.index >= cutoff]
    return float(recent.mean()) if len(recent) else float(daily.mean())


def forecast_weekly_units(
    sales_df: pd.DataFrame,
    sku: str,
    horizon_weeks: int = 4,
) -> list[float]:
    series = _sku_series(sales_df, sku)

    if len(series) < 4:
        avg_weekly = float(series.mean()) if len(series) else daily_demand(sales_df, sku) * 7
        return [max(0.0, avg_weekly)] * horizon_weeks

    if len(series) < 8:
        avg_weekly = float(series.tail(4).mean())
        return [max(0.0, avg_weekly)] * horizon_weeks

    preds = ensemble_model.fit_predict(series, horizon_weeks)
    values = preds["ensemble"] if isinstance(preds, dict) else preds
    return [max(0.0, float(v)) for v in values]
