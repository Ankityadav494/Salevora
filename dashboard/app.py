"""
Interactive Sales Forecasting Dashboard
Models: ARIMA | Prophet | LSTM | Weighted Ensemble
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Ensemble import
from src.models.ensemble_forecast import ensemble_forecast

# ---------------- PATH SETUP ----------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
st.title("📈 Sales Forecasting & Demand Prediction Dashboard")

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_DIR / "processed_sales_data.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.groupby("date", as_index=False)["sales"].sum()
df = df.set_index("date").sort_index()

sales = df["sales"]

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")

model_choice = st.sidebar.selectbox(
    "Select Forecasting Model",
    ["ARIMA", "Prophet", "LSTM", "Ensemble (Weighted)"]
)

forecast_steps = st.sidebar.slider(
    "Forecast Horizon (weeks)",
    min_value=4,
    max_value=24,
    value=12
)

# ---------------- HISTORICAL DATA ----------------
st.subheader("📊 Historical Sales")
st.line_chart(sales)

# ---------------- MODEL FORECASTS ----------------
if model_choice == "ARIMA":
    model = ARIMAResults.load(MODEL_DIR / "arima_model.pkl")
    forecast = model.forecast(steps=forecast_steps).values

elif model_choice == "Prophet":
    prophet_df = pd.read_csv(MODEL_DIR / "prophet_model.csv")
    forecast = prophet_df["yhat"].tail(forecast_steps).values

elif model_choice == "LSTM":
    values = sales.values.reshape(-1, 1)
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
    for _ in range(forecast_steps):
        pred = model.predict(last_seq.reshape(1, window, 1), verbose=0)
        preds.append(pred[0][0])
        last_seq = np.append(last_seq[1:], pred, axis=0)

    forecast = scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)
    ).flatten()

else:  # ENSEMBLE
    ensemble_df = ensemble_forecast(steps=forecast_steps)
    forecast = ensemble_df["Ensemble Forecast"].values

# ---------------- FUTURE DATES ----------------
future_dates = pd.date_range(
    start=sales.index[-1],
    periods=forecast_steps + 1,
    freq="W"
)[1:]

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecasted Sales": forecast
})

# ---------------- OUTPUT ----------------
st.subheader(f"🔮 {model_choice} Forecast")

st.dataframe(forecast_df)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(sales.index, sales, label="Historical Sales")
ax.plot(
    forecast_df["Date"],
    forecast_df["Forecasted Sales"],
    linestyle="--",
    marker="o",
    label=model_choice
)
ax.set_title(f"Future Forecast using {model_choice}")
ax.legend()
st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    " **ARIMA | Prophet | LSTM | Weighted Ensemble – Final Year Data Science Project**"
)
