"""
FastAPI REST API for Real-Time Sales Forecasting
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
from datetime import datetime

# -------------------------------------------------
# Initialize FastAPI App
# -------------------------------------------------
app = FastAPI(
    title="Salevora – Real-Time Sales Forecasting API",
    description="API to ingest live Walmart sales data",
    version="1.0"
)

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "live_sales.csv"

# -------------------------------------------------
# Request Model
# -------------------------------------------------
class SalesInput(BaseModel):
    date: str
    Store: int
    Dept: int
    sales: float
    IsHoliday: bool

# -------------------------------------------------
# Endpoints
# -------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Salevora – Real-Time Sales Forecasting API",
        "endpoints": {
            "/update-sales": "POST new sales data",
            "/health": "API health check"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/update-sales")
def update_sales(data: SalesInput):
    """
    Receive new sales data and append it to live dataset
    """
    try:
        # Load existing live data
        df = pd.read_csv(DATA_FILE)

        # Create new row
        new_row = pd.DataFrame([{
            "date": data.date,
            "Store": data.Store,
            "Dept": data.Dept,
            "sales": data.sales,
            "IsHoliday": data.IsHoliday
        }])

        # Append and save
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)

        return {
            "message": "Sales data updated successfully",
            "data_added": new_row.to_dict(orient="records")[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
