"""
Weighted Ensemble Forecasting
Combines ARIMA, Prophet, and LSTM using inverse-RMSE weighting
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ======================================================
# PROJECT ROOT
# ======================================================
BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "models"              # trained models
SRC_MODELS_DIR = BASE_DIR / "src" / "models" # metrics.json
DATA_DIR = BASE_DIR / "data" / "processed"   # processed data


# ======================================================
# LOAD METRICS
# ======================================================
def load_metrics():
    metrics_path = SRC_MODELS_DIR / "metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"❌ metrics.json not found at {metrics_path}")

    with open(metrics_path, "r") as f:
        return json.load(f)

# ======================================================
# ARIMA FORECAST
# ======================================================
def arima_forecast(steps):
    model_path = MODEL_DIR / "arima_model.pkl"
    model = ARIMAResults.load(model_path)
    return model.forecast(steps=steps).values

# ======================================================
# PROPHET FORECAST
# ======================================================
def prophet_forecast(steps):
    prophet_path = MODEL_DIR / "prophet_model.csv"
    df = pd.read_csv(prophet_path)
    return df["yhat"].tail(steps).values

# ======================================================
# LSTM FORECAST
# ======================================================
def lstm_forecast(steps, sales_series):
    values = sales_series.values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    window = 5
    X, y = [], []

    for i in range(len(scaled) - window):
        X.append(scaled[i:i + window])
        y.append(scaled[i + window])

    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y)

    model = Sequential([
        LSTM(32, input_shape=(window, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=15, batch_size=8, verbose=0)

    last_seq = scaled[-window:]
    preds = []

    for _ in range(steps):
        pred = model.predict(last_seq.reshape(1, window, 1), verbose=0)
        preds.append(pred[0][0])
        last_seq = np.append(last_seq[1:], pred, axis=0)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    return preds.flatten()

# ======================================================
# ENSEMBLE FORECAST
# ======================================================
def ensemble_forecast(steps=12):
    # Load processed sales data
    data_path = BASE_DIR / "data" / "processed" / "processed_sales_data.csv"
    df = pd.read_csv(data_path)

    df["date"] = pd.to_datetime(df["date"])
    df = df.groupby("date", as_index=False)["sales"].sum()
    df = df.set_index("date").sort_index()

    sales = df["sales"]
    print(f"📊 Total data points available: {len(sales)}")

    # Load RMSE metrics
    metrics = load_metrics()

    # Inverse-RMSE weights
    inv = {
        "arima": 1 / metrics["arima_rmse"],
        "prophet": 1 / metrics["prophet_rmse"],
        "lstm": 1 / metrics["lstm_rmse"]
    }

    total = sum(inv.values())
    weights = {k: v / total for k, v in inv.items()}
    print("🔢 Model Weights (inverse RMSE):", weights)

    # Individual forecasts
    f_arima = arima_forecast(steps)
    f_prophet = prophet_forecast(steps)
    f_lstm = lstm_forecast(steps, sales)

    # Weighted ensemble
    ensemble = (
        weights["arima"] * f_arima +
        weights["prophet"] * f_prophet +
        weights["lstm"] * f_lstm
    )

    future_dates = pd.date_range(
        start=sales.index[-1],
        periods=steps + 1,
        freq="W"
    )[1:]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Ensemble Forecast": ensemble
    })

    print("✅ Weighted Ensemble Forecast Generated Successfully")
    return forecast_df

# ======================================================
# RUN AS SCRIPT
# ======================================================
if __name__ == "__main__":
    df = ensemble_forecast(steps=12)
    print(df)
