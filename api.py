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
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ------------------------------------------------------------------ #
#  Paths
# ------------------------------------------------------------------ #
BASE_DIR      = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env", override=True)
WEBSITE_DIR   = BASE_DIR / "website"
ASSET_VERSION = "12"  # bump when CSS/JS changes to bust browser cache
PROCESSED_DIR = BASE_DIR / "data" / "processed"
LIVE_CSV      = PROCESSED_DIR / "live_sales.csv"
BACKUP_CSV    = PROCESSED_DIR / "live_sales_backup.csv"

REQUIRED_COLS = {"date", "sales"}          # minimum required columns
OPTIONAL_COLS = {"revenue", "category"}    # nice-to-have

from src.auth.database import get_db_info, init_db
from src.auth.deps import get_current_user, get_optional_user
from src.auth.router import router as auth_router
from src.alerts.router import router as alerts_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    db = get_db_info()
    print(f"  Database: {db['backend']}" + (f" ({db.get('database', '')})" if db["backend"] == "mongodb" else ""))
    from src.alerts import brevo
    if brevo.is_configured():
        s = brevo.sender()
        print(f"  Email (Brevo): ready — sender {s['email']}")
    else:
        print("  Email (Brevo): NOT configured — set BREVO_* in .env for OTP & alerts")
    yield
    if db["backend"] == "mongodb":
        from src.db.mongo import close_client
        close_client()


# ------------------------------------------------------------------ #
#  App
# ------------------------------------------------------------------ #
app = FastAPI(
    title="Salevora Data API",
    description="Upload CSV / Excel sales data to update the live dashboard in real-time.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(alerts_router)


@app.middleware("http")
async def disable_frontend_cache(request: Request, call_next):
    """Always serve fresh HTML/CSS/JS — avoids stale purple theme in browser cache."""
    response = await call_next(request)
    path = request.url.path.lower()
    if path == "/" or path.endswith((".html", ".css", ".js")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


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

    if "product_id" in df.columns:
        df["product_id"] = df["product_id"].astype(str).str.strip()
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)

    return df.sort_values("date").reset_index(drop=True)


def _ensure_backup() -> None:
    """Create a one-time backup of the original live_sales.csv."""
    if LIVE_CSV.exists() and not BACKUP_CSV.exists():
        shutil.copy2(LIVE_CSV, BACKUP_CSV)


# ------------------------------------------------------------------ #
#  Routes
# ------------------------------------------------------------------ #

@app.get("/api/health", tags=["Health"])
def health_check():
    db = get_db_info()
    return {
        "status": "ok" if db.get("connected") else "degraded",
        "service": "Salevora Data API",
        "version": "1.0.0",
        "database": db,
    }


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
    user: dict = Depends(get_current_user),
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

    else:  
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

    inv_summary = None
    alert_result = None
    try:
        from src.inventory.service import evaluate
        inv_summary = evaluate(str(LIVE_CSV))["summary"]
    except Exception:
        pass

    try:
        from src.alerts.service import auto_send_for_user
        alert_result = auto_send_for_user(user)
    except Exception:
        pass

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
            "inventory":   inv_summary,
            "alert_email": alert_result,
            "message": "Your sales file was saved and your stock check was updated.",
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


# ------------------------------------------------------------------ #
#  ML Forecasting (ARIMA + Prophet + Neural Net Ensemble)
# ------------------------------------------------------------------ #

@app.get("/api/forecast", tags=["Forecast"])
def get_forecast(
    horizon: int = Query(12, ge=1, le=52, description="Forecast horizon in weeks"),
    model: str = Query("ensemble", description="arima | prophet | lstm | ensemble"),
):
    """Run sales forecast on the live dataset using Python ML models."""
    try:
        from src.prediction.predictor import run_forecast
        return run_forecast(horizon=horizon, model=model, data_path=str(LIVE_CSV))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No sales data found. Upload a CSV first.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}")


@app.get("/api/analytics/summary", tags=["Analytics"])
def analytics_summary():
    """Return aggregated analytics from the live dataset."""
    try:
        from src.prediction.predictor import analytics_summary as _summary
        return _summary(data_path=str(LIVE_CSV))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No sales data found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/models/status", tags=["Forecast"])
def models_status():
    """Return training artifact info if available."""
    artifact_path = BASE_DIR / "models" / "forecast_artifact.joblib"
    if not artifact_path.exists():
        return {
            "trained": False,
            "message": "Run `python src/training/train.py` to train and evaluate models.",
            "available_models": ["arima", "prophet", "lstm", "ensemble"],
        }
    import joblib
    artifact = joblib.load(artifact_path)
    return {"trained": True, **artifact}


# ------------------------------------------------------------------ #
#  Inventory Intelligence (sales CSV + inventory API)
# ------------------------------------------------------------------ #
import asyncio
from datetime import timezone
from typing import List, Optional

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from src.inventory import store as inv_store
from src.inventory.service import (
    build_alerts,
    build_forecast_detail,
    build_items,
    build_kpis,
    evaluate,
)


def _utc_now():
    from datetime import datetime
    return datetime.now(timezone.utc)


def _require_sales():
    if not LIVE_CSV.exists():
        raise HTTPException(status_code=404, detail="No sales data. Upload a CSV with product_id first.")


@app.get("/api/app/status", tags=["Health"])
def app_status(user: dict | None = Depends(get_optional_user)):
    """Single bootstrap endpoint for the frontend."""
    from src.alerts import brevo

    payload: dict = {
        "online": True,
        "version": "1.0.0",
        "email_ready": brevo.is_configured(),
        "database": get_db_info(),
        "sales": None,
        "inventory": None,
    }
    if LIVE_CSV.exists():
        df = pd.read_csv(LIVE_CSV, parse_dates=["date"])
        payload["sales"] = {
            "rows": int(len(df)),
            "date_start": str(df["date"].min().date()),
            "date_end": str(df["date"].max().date()),
        }
        try:
            items = build_items(str(LIVE_CSV))
            alerts = build_alerts(items)
            payload["inventory"] = {
                "products": len(items),
                "alerts": len(alerts),
                "critical": sum(1 for a in alerts if a.get("status") == "critical"),
            }
        except Exception:
            pass
    if user:
        payload["user"] = {
            "name": user["name"],
            "email": user["email"],
            "alerts_enabled": user.get("alerts_enabled", True),
        }
    return payload


class StockUpdate(BaseModel):
    sku: str
    stock: float
    reorder_pt: Optional[float] = None
    lead_time: Optional[int] = Field(default=7, ge=1, le=90)
    name: Optional[str] = None
    category: Optional[str] = None
    unit_cost: Optional[float] = None
    max_stock: Optional[float] = None


@app.get("/api/inventory/stock", tags=["Inventory"])
def inv_get_stock():
    """Return inventory levels synced from external API."""
    return {"items": inv_store.list_stock(), "count": len(inv_store.load_all())}


@app.post("/api/inventory/sync", tags=["Inventory"])
def inv_sync(
    updates: List[StockUpdate],
    user: dict | None = Depends(get_optional_user),
):
    """
    Receive live stock from POS / ERP / Shopify / WooCommerce webhook.
    Demand is predicted separately from uploaded sales CSV.
    """
    payload = [u.model_dump(exclude_none=True) for u in updates]
    result = inv_store.sync_batch(payload)
    _require_sales()
    evaluation = evaluate(str(LIVE_CSV))
    alert_result = None
    try:
        from src.alerts.service import auto_send_for_user
        alert_result = auto_send_for_user(user)
    except Exception:
        pass
    return {
        "status": "success",
        "message": f"Updated stock for {len(result['updated']) + len(result['created'])} product(s).",
        "updated": result["updated"],
        "created": result["created"],
        "alerts": evaluation["alerts"],
        "alert_count": len(evaluation["alerts"]),
        "alert_email": alert_result,
    }


@app.post("/api/inventory/evaluate", tags=["Inventory"])
def inv_evaluate(user: dict = Depends(get_current_user)):
    """Re-run demand forecast from sales CSV and compare against API inventory."""
    _require_sales()
    return evaluate(str(LIVE_CSV))


@app.get("/api/inventory/live", tags=["Inventory"])
def inv_live():
    _require_sales()
    items = build_items(str(LIVE_CSV))
    return {"items": items, "updated_at": _utc_now().isoformat().replace("+00:00", "Z")}


@app.get("/api/inventory/kpis", tags=["Inventory"])
def inv_kpis():
    _require_sales()
    items = build_items(str(LIVE_CSV))
    return build_kpis(items)


@app.get("/api/inventory/alerts", tags=["Inventory"])
def inv_alerts():
    _require_sales()
    items = build_items(str(LIVE_CSV))
    alerts = build_alerts(items)
    return {
        "alerts": alerts,
        "count": len(alerts),
        "updated_at": _utc_now().isoformat().replace("+00:00", "Z"),
    }


@app.get("/api/inventory/forecast", tags=["Inventory"])
def inv_forecast(sku: str = None):
    _require_sales()
    path = str(LIVE_CSV)
    if sku:
        detail = build_forecast_detail(path, sku)
        if not detail:
            raise HTTPException(status_code=404, detail=f"SKU {sku} not found in sales data.")
        return {"forecasts": [detail], "updated_at": _utc_now().isoformat().replace("+00:00", "Z")}

    items = build_items(path)
    forecasts = []
    for item in items[:20]:
        detail = build_forecast_detail(path, item["sku"])
        if detail:
            forecasts.append(detail)
    return {"forecasts": forecasts, "updated_at": _utc_now().isoformat().replace("+00:00", "Z")}


@app.get("/api/inventory/abc", tags=["Inventory"])
def inv_abc():
    _require_sales()
    items = build_items(str(LIVE_CSV))
    ranked = sorted(
        [{
            "sku": i["sku"],
            "name": i["name"],
            "category": i["category"],
            "abc_class": i["abc_class"],
            "annual_value": round(i["daily_demand"] * 365 * i["cost"], 2),
            "daily_demand": i["daily_demand"],
            "cost": i["cost"],
        } for i in items],
        key=lambda x: -x["annual_value"],
    )
    total = sum(i["annual_value"] for i in ranked) or 1
    cum = 0
    for i in ranked:
        cum += i["annual_value"]
        i["cumulative_pct"] = round(cum / total * 100, 1)
    return {"items": ranked, "total_annual_value": round(total, 2)}


@app.post("/api/inventory/restock/{sku}", tags=["Inventory"])
def inv_restock(sku: str, qty: float = 0, user: dict = Depends(get_current_user)):
    _require_sales()
    items = build_items(str(LIVE_CSV))
    item = next((i for i in items if i["sku"] == sku), None)
    if not item:
        raise HTTPException(status_code=404, detail=f"SKU {sku} not found in sales catalog.")

    qty = qty if qty > 0 else item["eoq"]
    rec = inv_store.get(sku)
    new_stock = float(rec.get("stock", 0)) + qty
    inv_store.upsert(
        sku, new_stock,
        name=item["name"],
        category=item["category"],
        unit_cost=item["cost"],
        reorder_pt=item["reorder_pt"],
        lead_time=item["lead_time"],
        max_stock=item["max_stock"],
    )
    return {"sku": sku, "name": item["name"], "qty_added": round(qty), "new_stock": round(new_stock)}


class _WSMgr:
    def __init__(self):
        self.active = []

    async def connect(self, ws):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws):
        self.active = [w for w in self.active if w is not ws]


_wsmgr = _WSMgr()


@app.websocket("/ws/inventory")
async def inv_ws(websocket: WebSocket):
    await _wsmgr.connect(websocket)
    try:
        while True:
            if LIVE_CSV.exists():
                items = build_items(str(LIVE_CSV))
                alerts = build_alerts(items)
                await websocket.send_json({
                    "type": "snapshot",
                    "items": items,
                    "alert_count": len(alerts),
                    "updated_at": _utc_now().isoformat().replace("+00:00", "Z"),
                })
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        _wsmgr.disconnect(websocket)

# ------------------------------------------------------------------ #
#  Static frontend (HTML / CSS / JS) — must be registered last
# ------------------------------------------------------------------ #
if WEBSITE_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEBSITE_DIR), html=True), name="website")

# ------------------------------------------------------------------ #
#  Entry-point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import os
    import socket

    port = int(os.getenv("PORT", "8000"))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            print()
            print(f"  Port {port} is already in use.")
            print("  Stop the other Salevora server first, then run python api.py again.")
            print("  PowerShell:  Get-Process python* | Stop-Process -Force")
            print()
            raise SystemExit(1)

    # Bind all interfaces so LAN devices can connect; browser must use localhost.
    print()
    print("  Salevora")
    print("  --------")
    print(f"  Open in your browser:  http://localhost:{port}")
    print(f"  API docs:              http://localhost:{port}/docs")
    print("  (Do not use 0.0.0.0 in the browser — that address will not load.)")
    print()
    # reload=False avoids orphan worker processes when the app is started more than once.
    reload = os.getenv("DEV_RELOAD", "").lower() in ("1", "true", "yes")
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=reload)

