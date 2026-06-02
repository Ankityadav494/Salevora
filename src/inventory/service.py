"""Compare sales-based demand forecasts with API inventory levels."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from pathlib import Path

from src.inventory import catalog, demand, store


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _status(stock: float, reorder_pt: float, days_left: float, forecast_shortfall: bool) -> str:
    if stock <= 0:
        return "stockout"
    if forecast_shortfall or days_left <= 3:
        return "critical"
    if stock <= reorder_pt or days_left <= 7:
        return "warning"
    return "ok"


def _abc_class(daily_demand: float, unit_cost: float) -> str:
    annual = daily_demand * 365 * unit_cost
    if annual >= 50000:
        return "A"
    if annual >= 15000:
        return "B"
    return "C"


def _eoq(daily_demand: float, unit_cost: float) -> float:
    return math.sqrt(max(1, 2 * daily_demand * 365 * 50 / (0.20 * max(0.01, unit_cost))))


def build_items(sales_path: str, horizon_weeks: int = 4) -> list[dict[str, Any]]:
    if not Path(sales_path).exists():
        return []

    sales_df = catalog.load_sales(sales_path)
    try:
        cat = catalog.build_catalog(sales_df)
    except ValueError:
        return []

    stock_map = store.load_all()
    items: list[dict[str, Any]] = []

    for _, row in cat.iterrows():
        sku = str(row["sku"])
        inv = stock_map.get(sku, store._default_record())
        stock = float(inv.get("stock", 0))
        unit_cost = float(inv.get("unit_cost") or row.get("unit_cost") or 0)
        category = inv.get("category") or row.get("category") or "Uncategorised"
        name = inv.get("name") or row["name"]
        lead_time = int(inv.get("lead_time") or 7)
        max_stock = float(inv.get("max_stock") or max(stock * 2, 100))

        daily = demand.daily_demand(sales_df, sku)
        weekly_forecast = demand.forecast_weekly_units(sales_df, sku, horizon_weeks)
        forecast_total = sum(weekly_forecast)
        forecast_daily = forecast_total / max(1, horizon_weeks * 7)

        reorder_pt = float(inv.get("reorder_pt") or 0)
        if reorder_pt <= 0:
            reorder_pt = round(forecast_daily * (lead_time + 3))

        days_left = stock / daily if daily > 0 else (9999 if stock > 0 else 0)
        forecast_shortfall = stock < forecast_total
        status = _status(stock, reorder_pt, days_left, forecast_shortfall)
        eoq = _eoq(daily or forecast_daily, unit_cost or 1)

        items.append({
            "sku": sku,
            "name": name,
            "category": category,
            "stock": round(stock),
            "reorder_pt": round(reorder_pt),
            "max_stock": round(max_stock),
            "cost": round(unit_cost, 2),
            "lead_time": lead_time,
            "daily_demand": round(daily or forecast_daily, 1),
            "forecast_weekly_units": [round(v, 1) for v in weekly_forecast],
            "forecast_total_units": round(forecast_total, 1),
            "forecast_horizon_weeks": horizon_weeks,
            "days_of_stock": round(min(days_left, 9999), 1),
            "eoq": round(eoq),
            "stock_value": round(stock * unit_cost, 2),
            "stock_pct": round(stock / max_stock * 100, 1) if max_stock else 0,
            "status": status,
            "abc_class": _abc_class(daily or forecast_daily, unit_cost or 1),
            "has_inventory_data": sku in stock_map,
            "forecast_shortfall": forecast_shortfall,
            "shortfall_units": round(max(0, forecast_total - stock), 1),
        })

    return sorted(items, key=lambda x: x["days_of_stock"])


def build_alerts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alerts = []
    for item in items:
        if item["status"] not in ("critical", "warning", "stockout"):
            continue

        order_qty = round(item["eoq"])
        if item["forecast_shortfall"]:
            order_qty = max(order_qty, round(item["shortfall_units"]))

        message_parts = []
        if not item["has_inventory_data"]:
            message_parts.append("We do not have a stock count for this item yet.")
        if item["forecast_shortfall"]:
            message_parts.append(
                f"Expected to sell about {item['forecast_total_units']} items in the next "
                f"{item['forecast_horizon_weeks']} weeks, but only {item['stock']} in stock."
            )
        if item["stock"] <= item["reorder_pt"]:
            message_parts.append(f"Stock is at or below your reorder level ({item['reorder_pt']}).")
        if item["days_of_stock"] <= item["lead_time"]:
            message_parts.append(
                f"About {item['days_of_stock']} days of stock left — supplier needs {item['lead_time']} days to deliver."
            )

        alerts.append({
            **item,
            "order_qty": order_qty,
            "order_cost": round(order_qty * item["cost"], 2),
            "alert_message": " ".join(message_parts) or "Consider ordering more soon.",
        })

    return sorted(alerts, key=lambda x: x["days_of_stock"])


def build_kpis(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "total_skus": 0,
            "total_value": 0,
            "critical": 0,
            "at_risk": 0,
            "stockouts": 0,
            "avg_days_stock": 0,
            "in_stock_pct": 0,
            "forecast_alerts": 0,
            "updated_at": _utc_now().isoformat().replace("+00:00", "Z"),
        }

    return {
        "total_skus": len(items),
        "total_value": round(sum(i["stock_value"] for i in items), 2),
        "critical": sum(1 for i in items if i["status"] == "critical"),
        "at_risk": sum(1 for i in items if i["status"] == "warning"),
        "stockouts": sum(1 for i in items if i["status"] == "stockout"),
        "avg_days_stock": round(
            sum(min(i["days_of_stock"], 999) for i in items) / len(items), 1
        ),
        "in_stock_pct": round(sum(1 for i in items if i["stock"] > 0) / len(items) * 100, 1),
        "forecast_alerts": sum(1 for i in items if i["forecast_shortfall"]),
        "updated_at": _utc_now().isoformat().replace("+00:00", "Z"),
    }


def build_forecast_detail(sales_path: str, sku: str, days: int = 7) -> dict[str, Any] | None:
    items = build_items(sales_path)
    item = next((i for i in items if i["sku"] == sku), None)
    if not item:
        return None

    sales_df = catalog.load_sales(sales_path)
    daily_avg = item["daily_demand"]
    running = float(item["stock"])
    now = _utc_now()
    day_rows = []

    for i in range(days):
        d = daily_avg * (0.9 + 0.2 * (i % 3) / 2)  # slight variation around forecast
        running = max(0, running - d)
        day_rows.append({
            "day": (now + timedelta(days=i + 1)).strftime("%a %d %b"),
            "demand": round(d, 1),
            "projected_stock": round(running),
        })

    return {
        "sku": sku,
        "name": item["name"],
        "category": item["category"],
        "current_stock": item["stock"],
        "daily_demand": item["daily_demand"],
        "forecast_total_units": item["forecast_total_units"],
        "forecast": day_rows,
        "stockout_day": next((d["day"] for d in day_rows if d["projected_stock"] == 0), None),
        "reorder_recommended": item["eoq"],
    }


def evaluate(sales_path: str) -> dict[str, Any]:
    items = build_items(sales_path)
    alerts = build_alerts(items)
    return {
        "items": items,
        "alerts": alerts,
        "kpis": build_kpis(items),
        "summary": {
            "skus_from_sales": len(items),
            "skus_with_inventory_api": sum(1 for i in items if i["has_inventory_data"]),
            "alert_count": len(alerts),
            "forecast_shortfall_count": sum(1 for i in items if i["forecast_shortfall"]),
        },
        "updated_at": _utc_now().isoformat().replace("+00:00", "Z"),
    }
