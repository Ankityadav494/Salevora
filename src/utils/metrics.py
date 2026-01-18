"""
Evaluation metrics for forecasting models
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        float: MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        float: RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        float: MAPE value (in percentage)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_r2_score(y_true, y_pred):
    """
    Calculate R-squared score
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        float: R² score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def calculate_all_metrics(y_true, y_pred):
    """
    Calculate all evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'MAE': calculate_mae(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'R2': calculate_r2_score(y_true, y_pred)
    }
    
    return metrics


def print_metrics(metrics):
    """
    Print metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    for metric_name, metric_value in metrics.items():
        if metric_name == 'MAPE':
            print(f"{metric_name:15s}: {metric_value:.2f}%")
        elif metric_name == 'R2':
            print(f"{metric_name:15s}: {metric_value:.4f}")
        else:
            print(f"{metric_name:15s}: {metric_value:.2f}")
    print("="*50 + "\n")
