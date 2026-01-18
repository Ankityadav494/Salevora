"""
ARIMA model evaluation script
Evaluates model performance using RMSE
and plots Actual vs Predicted sales
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import mean_squared_error


def evaluate_arima():
    # Paths
    data_path = Path("data/processed/processed_sales_data.csv")
    model_path = Path("models/arima_model.pkl")

    # Load processed data
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    # Aggregate by date (same as training)
    df = df.groupby("date", as_index=False)["sales"].sum()
    df = df.set_index("date").sort_index()

    sales = df["sales"]

    # Train-test split (80-20)
    split = int(len(sales) * 0.8)
    train, test = sales.iloc[:split], sales.iloc[split:]

    # Load trained model
    model_fit = ARIMAResults.load(model_path)

    # Forecast
    forecast = model_fit.forecast(steps=len(test))

    # RMSE
    rmse = np.sqrt(mean_squared_error(test, forecast))
    print(f"📉 RMSE: {rmse:.2f}")

    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train, label="Train")
    plt.plot(test.index, test, label="Actual", color="black")
    plt.plot(test.index, forecast, label="Predicted", linestyle="--")
    plt.title("ARIMA Model – Actual vs Predicted Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_arima()
