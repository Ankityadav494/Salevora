"""
LSTM model training for Sales Forecasting Project
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def create_sequences(data, window_size=5):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def train_lstm():
    data_path = Path("data/processed/processed_sales_data.csv")

    # Load data
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    # Aggregate by date
    df = df.groupby("date", as_index=False)["sales"].sum()
    df = df.sort_values("date")

    sales = df["sales"].values.reshape(-1, 1)

    if len(sales) < 20:
        print("⚠️ Not enough data for LSTM — skipping LSTM training")
        return

    # Scale data
    scaler = MinMaxScaler()
    sales_scaled = scaler.fit_transform(sales)

    # Create sequences
    X, y = create_sequences(sales_scaled, window_size=5)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train-test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential([
        LSTM(32, activation="tanh", input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=1)

    # Predict
    y_pred = model.predict(X_test)

    # Inverse scaling
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

    print("✅ LSTM model trained successfully")
    print(f"📉 LSTM RMSE: {rmse:.2f}")


if __name__ == "__main__":
    train_lstm()
