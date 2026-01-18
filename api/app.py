"""
FastAPI REST API for Sales Forecasting
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config

# Initialize FastAPI app
config = load_config()
app = FastAPI(
    title=config['api']['title'],
    version=config['api']['version'],
    description="REST API for Sales Forecasting and Demand Prediction"
)


class ForecastRequest(BaseModel):
    """Request model for forecasting"""
    periods: int = 30
    model_type: Optional[str] = "prophet"


class ForecastResponse(BaseModel):
    """Response model for forecasting"""
    predictions: list
    dates: list
    model_type: str
    timestamp: str


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Sales Forecasting API",
        "version": config['api']['version'],
        "endpoints": {
            "/health": "Health check",
            "/forecast": "Generate sales forecast",
            "/models": "List available models"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": ["prophet", "lstm", "ensemble"],
        "default": "prophet"
    }


@app.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """
    Generate sales forecast
    
    Args:
        request: Forecast request with periods and model type
        
    Returns:
        ForecastResponse: Forecast predictions
    """
    try:
        # Placeholder for actual model prediction
        # In production, load trained model and make predictions
        
        # Generate sample predictions (replace with actual model)
        import numpy as np
        predictions = np.random.uniform(100, 500, request.periods).tolist()
        
        # Generate dates
        dates = pd.date_range(
            start=datetime.now(),
            periods=request.periods,
            freq='D'
        ).strftime('%Y-%m-%d').tolist()
        
        return ForecastResponse(
            predictions=predictions,
            dates=dates,
            model_type=request.model_type,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config['api']['host'],
        port=config['api']['port']
    )
