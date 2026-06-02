# Salevora — Real-Time Sales Forecasting & Demand Prediction

A full-stack sales intelligence platform with **Python ML backend** (FastAPI) and **HTML/CSS/JS frontend**. Predict sales using ARIMA, Prophet, and neural network models in an ensemble pipeline, with live inventory tracking and interactive dashboards.

## Features

- **Real-Time Forecasting** — Ensemble of ARIMA, Prophet, and neural network models
- **Interactive Dashboard** — Upload CSV/Excel, view KPIs, trends, and forecasts
- **Live Inventory** — Real-time stock monitoring with WebSocket updates
- **Data Pipeline** — Automated preprocessing and validation
- **REST API** — Upload data, run forecasts, inventory alerts, analytics
- **Model Training** — Train and evaluate all models with hold-out metrics

## Project Structure

```
Salevora/
├── api.py                      # FastAPI backend (serves API + frontend)
├── config.yaml                 # ML and app configuration
├── requirements.txt
├── data/
│   ├── raw/                    # Raw datasets
│   └── processed/              # live_sales.csv (active dataset)
├── src/
│   ├── data/                   # Loading and preprocessing
│   ├── models/                 # ARIMA, Prophet, LSTM/MLP, Ensemble
│   ├── training/train.py       # Model training pipeline
│   ├── prediction/predictor.py # Forecast API logic
│   └── utils/                  # Config + metrics
├── website/                    # Frontend (HTML / CSS / JS)
│   ├── index.html              # Sales dashboard
│   ├── app.js
│   ├── style.css
│   ├── inventory.html          # Live inventory page
│   ├── inventory.js
│   └── inventory.css
├── models/                     # Saved training artifacts
└── logs/                       # Training logs
```

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone and enter project
cd Salevora

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Start the Python backend (serves API + frontend)

```bash
python api.py
```

Open **http://localhost:8000** in your browser.

### 2. Sign in

Create an account or sign in with your email. A verification code will be sent to your inbox.

### 3. Upload sales data

Drag & drop a CSV or Excel file with at least `date` and `sales` columns.  
Click **Run Forecast & Predictions** — data syncs to the Python backend and ML models generate forecasts.

### 4. Train models (optional)

```bash
python src/training/train.py
```

### 5. API docs

Visit **http://localhost:8000/docs** for interactive Swagger documentation.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/forecast` | Run ML forecast (`?horizon=12&model=ensemble`) |
| GET | `/api/analytics/summary` | Dataset analytics |
| GET | `/api/models/status` | Training artifact status |
| GET | `/data/info` | Live dataset stats |
| POST | `/data/upload` | Upload CSV/Excel |
| POST | `/data/reset` | Restore from backup |
| GET | `/api/inventory/live` | Live inventory levels |
| WS | `/ws/inventory` | Real-time inventory WebSocket |

## Technology Stack

**Backend:** Python, FastAPI, Pandas, scikit-learn, statsmodels, Prophet  
**Frontend:** HTML5, CSS3, Vanilla JavaScript, Plotly.js  
**Models:** ARIMA, Prophet, MLP Neural Network, Weighted Ensemble

## License

MIT
