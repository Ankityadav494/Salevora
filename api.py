"""
Salevora Data Upload API
========================
FastAPI service to dynamically upload sales data (CSV / Excel) and
replace / append the live_sales.csv consumed by the Streamlit dashboard.

Endpoints
---------
GET  /             – Health-check
GET  /data/info    – Stats about the current live dataset
POST /data/upload  – Upload a CSV or Excel file (replace or append mode)
POST /data/reset   – Restore the original backup
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Literal

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ------------------------------------------------------------------ #
#  Paths
# ------------------------------------------------------------------ #
BASE_DIR      = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
LIVE_CSV      = PROCESSED_DIR / "live_sales.csv"
BACKUP_CSV    = PROCESSED_DIR / "live_sales_backup.csv"

REQUIRED_COLS = {"date", "sales"}          # minimum required columns
OPTIONAL_COLS = {"revenue", "category"}    # nice-to-have

# ------------------------------------------------------------------ #
#  App
# ------------------------------------------------------------------ #
app = FastAPI(
    title="Salevora Data API",
    description="Upload CSV / Excel sales data to update the live dashboard in real-time.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #




def _validate_schema(df: pd.DataFrame) -> None:
    """Ensure the dataframe has the minimum required columns."""
    missing = REQUIRED_COLS - set(df.columns.str.lower())
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required column(s): {', '.join(sorted(missing))}. "
                   f"Your file has: {', '.join(df.columns.tolist())}",
        )


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names, parse dates, fill defaults."""
    df.columns = df.columns.str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)

    if "revenue" not in df.columns:
        df["revenue"] = df["sales"]          # fallback: revenue = sales
    else:
        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(df["sales"])

    if "category" not in df.columns:
        df["category"] = "Uncategorised"

    return df.sort_values("date").reset_index(drop=True)


def _ensure_backup() -> None:
    """Create a one-time backup of the original live_sales.csv."""
    if LIVE_CSV.exists() and not BACKUP_CSV.exists():
        shutil.copy2(LIVE_CSV, BACKUP_CSV)


# ------------------------------------------------------------------ #
#  Routes
# ------------------------------------------------------------------ #

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "Salevora Data API"}


@app.get("/data/info", tags=["Data"])
def data_info():
    """Return basic statistics about the current live dataset."""
    if not LIVE_CSV.exists():
        raise HTTPException(status_code=404, detail="live_sales.csv not found.")

    df = pd.read_csv(LIVE_CSV, parse_dates=["date"])
    return {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "date_range": {
            "start": str(df["date"].min().date()),
            "end":   str(df["date"].max().date()),
        },
        "categories": sorted(df["category"].dropna().unique().tolist()) if "category" in df.columns else [],
        "total_revenue": float(round(df["revenue"].sum(), 2)) if "revenue" in df.columns else None,
    }


@app.post("/data/upload", tags=["Data"])
async def upload_data(
    request: Request,
    filename: str = Query(..., description="Name of the file being uploaded (e.g., data.csv)"),
    mode: Literal["replace", "append"] = Query(
        "replace",
        description="'replace' overwrites live data; 'append' merges and deduplicates by date+category.",
    ),
):
    """
    Upload a sales CSV or Excel file to update the live dashboard.
    Upload the file content in the raw request body.

    - **replace** – The uploaded data completely replaces `live_sales.csv`.
    - **append**  – Rows from the upload are merged into the existing dataset
                    (duplicate date+category rows are de-duplicated, keeping the new values).
    """
    _ensure_backup()

    content = await request.body()
    name = filename.lower()

    if name.endswith(".csv"):
        df_new = pd.read_csv(io.BytesIO(content))
    elif name.endswith((".xlsx", ".xls")):
        df_new = pd.read_excel(io.BytesIO(content))
    else:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload a .csv or .xlsx file.",
        )

    _validate_schema(df_new)
    df_new = _normalise(df_new)

    if mode == "replace":
        df_final = df_new

    else:  # append
        if not LIVE_CSV.exists():
            df_final = df_new
        else:
            df_existing = pd.read_csv(LIVE_CSV, parse_dates=["date"])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)

            # De-duplicate: keep the last (i.e., uploaded) value for same date+category
            dedup_cols = ["date", "category"] if "category" in df_combined.columns else ["date"]
            df_final = (
                df_combined
                .sort_values("date")
                .drop_duplicates(subset=dedup_cols, keep="last")
                .reset_index(drop=True)
            )

    # Save
    df_final.to_csv(LIVE_CSV, index=False)

    return JSONResponse(
        status_code=200,
        content={
            "status":      "success",
            "mode":        mode,
            "rows_saved":  len(df_final),
            "date_range": {
                "start": str(df_final["date"].min().date()),
                "end":   str(df_final["date"].max().date()),
            },
            "message": (
                "live_sales.csv has been updated. "
                "Refresh the Streamlit dashboard (clear cache) to see the new data."
            ),
        },
    )


@app.post("/data/reset", tags=["Data"])
def reset_data():
    """Restore live_sales.csv from the original backup."""
    if not BACKUP_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail="No backup found. Upload data first to auto-create one.",
        )
    shutil.copy2(BACKUP_CSV, LIVE_CSV)
    return {"status": "success", "message": "live_sales.csv restored from backup."}


@app.get("/data/download", tags=["Data"])
def download_data():
    """Download the current live_sales.csv as a JSON array of records."""
    if not LIVE_CSV.exists():
        raise HTTPException(status_code=404, detail="live_sales.csv not found.")
    df = pd.read_csv(LIVE_CSV, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return JSONResponse(
        content={
            "status": "success",
            "rows": int(len(df)),
            "data": df.head(1000).to_dict(orient="records"),   # capped at 1000 for performance
            "note": "Returns up to 1000 rows. Use /data/info for full stats.",
        }
    )


@app.get("/data/columns", tags=["Data"])
def get_columns():
    """Return the column names and dtypes of the current live dataset."""
    if not LIVE_CSV.exists():
        raise HTTPException(status_code=404, detail="live_sales.csv not found.")
    df = pd.read_csv(LIVE_CSV, nrows=0)
    return {
        "columns": df.columns.tolist(),
        "dtypes":  {col: str(dtype) for col, dtype in df.dtypes.items()},
        "required": list(REQUIRED_COLS),
        "optional": list(OPTIONAL_COLS),
    }


@app.get("/data/sample", tags=["Data"])
def get_sample(n: int = Query(10, ge=1, le=100, description="Number of sample rows (1-100)")):
    """Return the first N rows of the live dataset as a preview."""
    if not LIVE_CSV.exists():
        raise HTTPException(status_code=404, detail="live_sales.csv not found.")
    df = pd.read_csv(LIVE_CSV, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    sample = df.head(n)
    return {
        "rows_returned": int(len(sample)),
        "total_rows":    int(len(df)),
        "data": sample.to_dict(orient="records"),
    }


# ------------------------------------------------------------------ #
#  Real-Time Inventory Simulation Module
# ------------------------------------------------------------------ #
import random, math, asyncio
from datetime import datetime, timedelta
from fastapi import WebSocket, WebSocketDisconnect

_PRODUCTS = [
    {"sku":"ELEC-001","name":"Laptop Pro 15in","category":"Electronics","stock":45,"reorder_pt":20,"max_stock":100,"cost":899.99,"lead_time":7,"daily_demand":3.2},
    {"sku":"ELEC-002","name":"Wireless Headphones","category":"Electronics","stock":12,"reorder_pt":25,"max_stock":150,"cost":129.99,"lead_time":5,"daily_demand":8.5},
    {"sku":"ELEC-003","name":"USB-C Hub 7-Port","category":"Electronics","stock":88,"reorder_pt":30,"max_stock":200,"cost":49.99,"lead_time":3,"daily_demand":12.1},
    {"sku":"ELEC-004","name":"Mechanical Keyboard","category":"Electronics","stock":7,"reorder_pt":15,"max_stock":80,"cost":159.99,"lead_time":6,"daily_demand":4.3},
    {"sku":"CLTH-001","name":"Premium Hoodie XL","category":"Clothing","stock":234,"reorder_pt":50,"max_stock":500,"cost":39.99,"lead_time":14,"daily_demand":15.7},
    {"sku":"CLTH-002","name":"Running Shoes M10","category":"Clothing","stock":18,"reorder_pt":30,"max_stock":200,"cost":89.99,"lead_time":10,"daily_demand":9.2},
    {"sku":"CLTH-003","name":"Yoga Pants S/M","category":"Clothing","stock":67,"reorder_pt":40,"max_stock":300,"cost":29.99,"lead_time":12,"daily_demand":18.4},
    {"sku":"FOOD-001","name":"Protein Bar Box 24","category":"Food & Bev","stock":156,"reorder_pt":100,"max_stock":600,"cost":24.99,"lead_time":2,"daily_demand":45.3},
    {"sku":"FOOD-002","name":"Green Tea 100 Bags","category":"Food & Bev","stock":34,"reorder_pt":80,"max_stock":400,"cost":12.99,"lead_time":3,"daily_demand":28.6},
    {"sku":"FOOD-003","name":"Whey Protein 2kg","category":"Food & Bev","stock":9,"reorder_pt":25,"max_stock":150,"cost":59.99,"lead_time":4,"daily_demand":6.8},
    {"sku":"SPRT-001","name":"Dumbbell Set 20kg","category":"Sports","stock":23,"reorder_pt":10,"max_stock":60,"cost":79.99,"lead_time":8,"daily_demand":2.1},
    {"sku":"SPRT-002","name":"Yoga Mat Premium","category":"Sports","stock":5,"reorder_pt":20,"max_stock":100,"cost":34.99,"lead_time":5,"daily_demand":7.3},
    {"sku":"SPRT-003","name":"Resistance Bands Set","category":"Sports","stock":112,"reorder_pt":30,"max_stock":250,"cost":19.99,"lead_time":4,"daily_demand":11.2},
    {"sku":"HOME-001","name":"Air Purifier Room","category":"Home & Garden","stock":31,"reorder_pt":15,"max_stock":80,"cost":199.99,"lead_time":9,"daily_demand":4.7},
    {"sku":"HOME-002","name":"Smart LED Strip 5m","category":"Home & Garden","stock":67,"reorder_pt":25,"max_stock":200,"cost":29.99,"lead_time":6,"daily_demand":8.9},
    {"sku":"HOME-003","name":"Indoor Plant Pot Set","category":"Home & Garden","stock":2,"reorder_pt":20,"max_stock":120,"cost":24.99,"lead_time":7,"daily_demand":5.4},
]
_sim_stock = {p["sku"]: float(p["stock"]) for p in _PRODUCTS}
_last_tick = datetime.utcnow()

def _tick():
    global _last_tick
    now = datetime.utcnow()
    h = (now - _last_tick).total_seconds() / 3600.0
    if h < 0.001: return
    for p in _PRODUCTS:
        sold = p["daily_demand"] / 24.0 * h * (0.75 + random.random() * 0.5)
        _sim_stock[p["sku"]] = max(0.0, _sim_stock[p["sku"]] - sold)
    _last_tick = now

def _eoq(p):
    return math.sqrt(max(1, 2 * p["daily_demand"] * 365 * 50 / (0.20 * max(0.01, p["cost"]))))

def _status(s, p):
    if s <= 0: return "stockout"
    if s <= p["reorder_pt"] * 0.5: return "critical"
    if s <= p["reorder_pt"]: return "warning"
    return "ok"

def _abc(p):
    v = p["daily_demand"] * 365 * p["cost"]
    return "A" if v >= 50000 else ("B" if v >= 15000 else "C")

def _build(p):
    s = _sim_stock[p["sku"]]
    days = s / p["daily_demand"] if p["daily_demand"] > 0 else 9999
    return {"sku": p["sku"], "name": p["name"], "category": p["category"],
            "stock": round(s), "reorder_pt": p["reorder_pt"], "max_stock": p["max_stock"],
            "cost": p["cost"], "lead_time": p["lead_time"], "daily_demand": round(p["daily_demand"], 1),
            "days_of_stock": round(days, 1), "eoq": round(_eoq(p)),
            "stock_value": round(s * p["cost"], 2), "stock_pct": round(s / p["max_stock"] * 100, 1),
            "status": _status(s, p), "abc_class": _abc(p)}

@app.get("/api/inventory/live", tags=["Inventory"])
def inv_live():
    _tick(); items = sorted([_build(p) for p in _PRODUCTS], key=lambda x: x["days_of_stock"])
    return {"items": items, "updated_at": datetime.utcnow().isoformat() + "Z"}

@app.get("/api/inventory/kpis", tags=["Inventory"])
def inv_kpis():
    _tick(); items = [_build(p) for p in _PRODUCTS]
    total_val = sum(i["stock_value"] for i in items)
    avg_days = sum(min(i["days_of_stock"], 999) for i in items) / len(items)
    return {"total_skus": len(_PRODUCTS), "total_value": round(total_val, 2),
            "critical": sum(1 for i in items if i["status"] == "critical"),
            "at_risk": sum(1 for i in items if i["status"] == "warning"),
            "stockouts": sum(1 for i in items if i["status"] == "stockout"),
            "avg_days_stock": round(avg_days, 1),
            "in_stock_pct": round(sum(1 for i in items if i["stock"] > 0) / len(items) * 100, 1),
            "updated_at": datetime.utcnow().isoformat() + "Z"}

@app.get("/api/inventory/alerts", tags=["Inventory"])
def inv_alerts():
    _tick()
    alerts = []
    for p in _PRODUCTS:
        item = _build(p)
        if item["status"] in ("critical", "warning", "stockout"):
            item["order_qty"] = round(_eoq(p)); item["order_cost"] = round(_eoq(p) * p["cost"], 2)
            alerts.append(item)
    return {"alerts": sorted(alerts, key=lambda x: x["days_of_stock"]), "count": len(alerts), "updated_at": datetime.utcnow().isoformat() + "Z"}

@app.get("/api/inventory/forecast", tags=["Inventory"])
def inv_forecast(sku: str = None):
    _tick(); results = []
    for p in _PRODUCTS:
        if sku and p["sku"] != sku: continue
        running = _sim_stock[p["sku"]]; days = []
        for i in range(7):
            d = p["daily_demand"] * (0.80 + random.random() * 0.40)
            running = max(0, running - d)
            days.append({"day": (datetime.utcnow() + timedelta(days=i+1)).strftime("%a %d %b"), "demand": round(d, 1), "projected_stock": round(running)})
        results.append({"sku": p["sku"], "name": p["name"], "category": p["category"],
                        "current_stock": round(_sim_stock[p["sku"]]), "daily_demand": p["daily_demand"],
                        "forecast": days, "stockout_day": next((d["day"] for d in days if d["projected_stock"] == 0), None),
                        "reorder_recommended": round(_eoq(p))})
    return {"forecasts": results, "updated_at": datetime.utcnow().isoformat() + "Z"}

@app.get("/api/inventory/abc", tags=["Inventory"])
def inv_abc():
    _tick()
    items = sorted([{"sku": p["sku"], "name": p["name"], "category": p["category"],
                     "abc_class": _abc(p), "annual_value": round(p["daily_demand"] * 365 * p["cost"], 2),
                     "daily_demand": p["daily_demand"], "cost": p["cost"]} for p in _PRODUCTS], key=lambda x: -x["annual_value"])
    total = sum(i["annual_value"] for i in items); cum = 0
    for i in items: cum += i["annual_value"]; i["cumulative_pct"] = round(cum / total * 100, 1)
    return {"items": items, "total_annual_value": round(total, 2)}

@app.post("/api/inventory/restock/{sku}", tags=["Inventory"])
def inv_restock(sku: str, qty: float = 0):
    p = next((x for x in _PRODUCTS if x["sku"] == sku), None)
    if not p: raise HTTPException(status_code=404, detail=f"SKU {sku} not found.")
    qty = qty if qty > 0 else _eoq(p)
    _sim_stock[sku] = min(p["max_stock"], _sim_stock[sku] + qty)
    return {"sku": sku, "name": p["name"], "qty_added": round(qty), "new_stock": round(_sim_stock[sku])}

from pydantic import BaseModel
from typing import List

class StockUpdate(BaseModel):
    sku: str
    stock: float

@app.post("/api/inventory/sync", tags=["Inventory"])
def inv_sync(updates: List[StockUpdate]):
    """Webhook to receive live data from POS/ERP (Shopify, SAP, etc.)"""
    updated_count = 0
    not_found = []
    
    for u in updates:
        if u.sku in _sim_stock:
            # Overwrite the simulation with TRUE live data from external system
            _sim_stock[u.sku] = max(0.0, float(u.stock))
            updated_count += 1
        else:
            not_found.append(u.sku)
            
    return {
        "status": "success",
        "message": f"Successfully synced {updated_count} SKUs.",
        "skipped_unknown_skus": not_found,
        "new_levels": {u.sku: _sim_stock.get(u.sku) for u in updates if u.sku in _sim_stock}
    }

class _WSMgr:
    def __init__(self): self.active = []
    async def connect(self, ws): await ws.accept(); self.active.append(ws)
    def disconnect(self, ws): self.active = [w for w in self.active if w is not ws]
    async def broadcast(self, data):
        dead = []
        for ws in self.active:
            try: await ws.send_json(data)
            except: dead.append(ws)
        for ws in dead: self.disconnect(ws)

_wsmgr = _WSMgr()

@app.websocket("/ws/inventory")
async def inv_ws(websocket: WebSocket):
    await _wsmgr.connect(websocket)
    try:
        while True:
            _tick(); items = [_build(p) for p in _PRODUCTS]
            await websocket.send_json({"type": "snapshot", "items": items,
                "alert_count": sum(1 for i in items if i["status"] in ("critical","warning","stockout")),
                "updated_at": datetime.utcnow().isoformat() + "Z"})
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        _wsmgr.disconnect(websocket)

# ------------------------------------------------------------------ #
#  Entry-point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

