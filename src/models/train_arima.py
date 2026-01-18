"""
ARIMA model training script for Sales Forecasting Project
Includes safe fallback for small or sparse datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


def train_arima():
    # Paths
    data_path = Path("data/processed/processed_sales_data.csv")
    model_path = Path("models/arima_model.pkl")

    # Load processed data
    df = pd.read_csv(data_path)

    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    # Aggregate duplicate dates (Walmart has multiple stores/depts per date)
    df = df.groupby("date", as_index=False)["sales"].sum()

    # Set datetime index (DO NOT force frequency)
    df = df.set_index("date").sort_index()

    sales = df["sales"]

    print(f"📊 Total data points available: {len(sales)}")

    # ---------- SAFETY CHECK ----------
    if len(sales) == 0:
        raise ValueError("❌ No data available after preprocessing. Check input CSV.")

    # ---------- VERY SMALL DATA → BASELINE ----------
    if len(sales) < 8:
        print("⚠️ Data too small for ARIMA")
        print("➡️ Using Naive Forecast (Last Value Baseline)")

        last_value = sales.iloc[-1]
        print(f"📈 Last observed sales value: {last_value:.2f}")
        print("✅ Baseline model completed successfully")
        return

    # ---------- ARIMA TRAINING ----------
    split = int(len(sales) * 0.8)
    train, test = sales.iloc[:split], sales.iloc[split:]

    # Choose ARIMA order safely
    order = (1, 1, 0) if len(train) < 12 else (1, 1, 1)

    print(f"🧠 Training ARIMA model with order={order}")

    model = ARIMA(
        train,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit()

    # Evaluation
    if len(test) > 0:
        forecast = model_fit.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test, forecast))
        print(f"📉 RMSE: {rmse:.2f}")
    else:
        print("⚠️ No test data available — skipping RMSE calculation")

    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_fit.save(model_path)

    print("✅ ARIMA model trained successfully")
    print(f"💾 Model saved at: {model_path}")


if __name__ == "__main__":
    train_arima()
