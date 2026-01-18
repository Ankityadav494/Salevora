"""
Main training script for sales forecasting models
"""
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import load_data
from src.data.data_preprocessor import preprocess_sales_data
from src.data.feature_engineering import create_all_features
from src.models.prophet_model import ProphetModel
from src.models.train_lstm import LSTMModel
from src.utils.config import load_config, ensure_directories
from src.utils.metrics import calculate_all_metrics, print_metrics
from sklearn.model_selection import train_test_split


def train_prophet_model(train_df, test_df, config):
    """Train Prophet model"""
    print("\n" + "="*50)
    print("Training Prophet Model")
    print("="*50)
    
    # Prepare data
    prophet_train = train_df.reset_index()
    prophet_train['ds'] = pd.to_datetime(prophet_train.index)
    prophet_train['y'] = prophet_train['sales']
    
    # Initialize and train model
    prophet_config = config['models']['prophet']
    model = ProphetModel(
        yearly_seasonality=prophet_config['yearly_seasonality'],
        weekly_seasonality=prophet_config['weekly_seasonality'],
        daily_seasonality=prophet_config['daily_seasonality'],
        seasonality_mode=prophet_config['seasonality_mode']
    )
    
    model.train(prophet_train, date_column='ds', target_column='y')
    
    # Make predictions on test set
    test_periods = len(test_df)
    forecast = model.predict(periods=test_periods)
    
    # Get predictions for test period
    predictions = forecast['yhat'].tail(test_periods).values
    
    # Calculate metrics
    metrics = calculate_all_metrics(test_df['sales'].values, predictions)
    print_metrics(metrics)
    
    # Save model
    if config['training']['save_model']:
        model_path = Path(config['training']['model_save_path']) / 'prophet_model.pkl'
        model.save(str(model_path))
    
    return model, metrics


def train_lstm_model(train_df, test_df, config):
    """Train LSTM model"""
    print("\n" + "="*50)
    print("Training LSTM Model")
    print("="*50)
    
    # Initialize and train model
    lstm_config = config['models']['lstm']
    model = LSTMModel(
        sequence_length=lstm_config['sequence_length'],
        units=lstm_config['units'],
        dropout=lstm_config['dropout'],
        epochs=lstm_config['epochs'],
        batch_size=lstm_config['batch_size'],
        learning_rate=lstm_config['learning_rate']
    )
    
    model.train(train_df, target_column='sales')
    
    # Make predictions
    predictions = model.predict(train_df, target_column='sales', periods=len(test_df))
    
    # Calculate metrics
    metrics = calculate_all_metrics(test_df['sales'].values, predictions)
    print_metrics(metrics)
    
    # Save model
    if config['training']['save_model']:
        model_path = Path(config['training']['model_save_path']) / 'lstm_model.h5'
        model.model.save(str(model_path))
    
    return model, metrics


def main():
    """Main training function"""
    print("="*50)
    print("Sales Forecasting Model Training")
    print("="*50)
    
    # Load configuration
    config = load_config()
    ensure_directories()
    
    # Load data
    data_path = config['data']['raw_data_path']
    print(f"\nLoading data from {data_path}...")
    
    if not Path(data_path).exists():
        print(f"Data file not found. Please generate sample data first.")
        print("Run: python src/data/generate_sample_data.py")
        return
    
    df = load_data(data_path)
    
    # Preprocess data
    print("\nPreprocessing data...")
    df = preprocess_sales_data(df, date_column='date', target_column='sales')
    
    # For simplicity, use first product's data
    if 'product_id' in df.columns:
        df = df[df['product_id'] == df['product_id'].iloc[0]]
        df = df[['sales']]  # Keep only sales column
    
    # Split data
    test_size = config['data']['test_size']
    train_size = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"\nTrain size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Train models
    results = {}
    
    if config['models']['prophet']['enabled']:
        prophet_model, prophet_metrics = train_prophet_model(train_df, test_df, config)
        results['prophet'] = {'model': prophet_model, 'metrics': prophet_metrics}
    
    if config['models']['lstm']['enabled']:
        lstm_model, lstm_metrics = train_lstm_model(train_df, test_df, config)
        results['lstm'] = {'model': lstm_model, 'metrics': lstm_metrics}
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    
    return results


if __name__ == "__main__":
    main()
