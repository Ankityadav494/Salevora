"""Persist inventory levels received from external API / webhooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.config import BASE_DIR

STOCK_FILE = BASE_DIR / "data" / "processed" / "inventory_stock.json"


def _default_record() -> dict[str, Any]:
    return {
        "stock": 0.0,
        "reorder_pt": 0.0,
        "lead_time": 7,
        "name": "",
        "category": "",
        "unit_cost": 0.0,
        "max_stock": 1000.0,
    }


def load_all() -> dict[str, dict[str, Any]]:
    if not STOCK_FILE.exists():
        return {}
    with open(STOCK_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, dict) else {}


def save_all(data: dict[str, dict[str, Any]]) -> None:
    STOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STOCK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get(sku: str) -> dict[str, Any]:
    return load_all().get(sku, _default_record())


def upsert(
    sku: str,
    stock: float,
    *,
    reorder_pt: float | None = None,
    lead_time: int | None = None,
    name: str | None = None,
    category: str | None = None,
    unit_cost: float | None = None,
    max_stock: float | None = None,
) -> dict[str, Any]:
    data = load_all()
    rec = {**_default_record(), **data.get(sku, {})}
    rec["stock"] = max(0.0, float(stock))
    if reorder_pt is not None:
        rec["reorder_pt"] = max(0.0, float(reorder_pt))
    if lead_time is not None:
        rec["lead_time"] = int(lead_time)
    if name:
        rec["name"] = name
    if category:
        rec["category"] = category
    if unit_cost is not None:
        rec["unit_cost"] = max(0.0, float(unit_cost))
    if max_stock is not None:
        rec["max_stock"] = max(1.0, float(max_stock))
    data[sku] = rec
    save_all(data)
    return rec


def sync_batch(updates: list[dict[str, Any]]) -> dict[str, Any]:
    updated, created = [], []
    for u in updates:
        sku = str(u["sku"]).strip()
        if not sku:
            continue
        existed = sku in load_all()
        upsert(
            sku,
            u["stock"],
            reorder_pt=u.get("reorder_pt"),
            lead_time=u.get("lead_time"),
            name=u.get("name"),
            category=u.get("category"),
            unit_cost=u.get("unit_cost"),
            max_stock=u.get("max_stock"),
        )
        (updated if existed else created).append(sku)
    return {"updated": updated, "created": created}


def list_stock() -> list[dict[str, Any]]:
    return [{"sku": k, **v} for k, v in load_all().items()]
