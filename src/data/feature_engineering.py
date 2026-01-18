"""
Feature engineering utilities for time series forecasting
"""
import pandas as pd
import numpy as np
from typing import List, Optional


def create_time_features(df: pd.DataFrame, date_column: Optional[str] = None) -> pd.DataFrame:
    """
    Create time-based features from datetime
    
    Args:
        df: Input dataframe
        date_column: Name of date column (if not index)
        
    Returns:
        pd.DataFrame: Dataframe with time features added
    """
    df = df.copy()
    
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column])
        datetime_index = df[date_column]
    else:
        datetime_index = df.index
    
    # Extract time features
    df['year'] = datetime_index.year
    df['month'] = datetime_index.month
    df['day'] = datetime_index.day
    df['day_of_week'] = datetime_index.dayofweek
    df['day_of_year'] = datetime_index.dayofyear
    df['week'] = datetime_index.isocalendar().week
    df['quarter'] = datetime_index.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding for periodic features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df


def create_lag_features(df: pd.DataFrame, column: str, lags: List[int]) -> pd.DataFrame:
    """
    Create lag features for a specific column
    
    Args:
        df: Input dataframe
        column: Column name to create lags for
        lags: List of lag periods (e.g., [1, 7, 30])
        
    Returns:
        pd.DataFrame: Dataframe with lag features added
    """
    df = df.copy()
    
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame, column: str, windows: List[int]) -> pd.DataFrame:
    """
    Create rolling statistics features
    
    Args:
        df: Input dataframe
        column: Column name to create rolling features for
        windows: List of window sizes (e.g., [7, 30, 90])
        
    Returns:
        pd.DataFrame: Dataframe with rolling features added
    """
    df = df.copy()
    
    for window in windows:
        df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()
        df[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window).std()
        df[f'{column}_rolling_min_{window}'] = df[column].rolling(window=window).min()
        df[f'{column}_rolling_max_{window}'] = df[column].rolling(window=window).max()
    
    return df


def create_expanding_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Create expanding window features
    
    Args:
        df: Input dataframe
        column: Column name to create expanding features for
        
    Returns:
        pd.DataFrame: Dataframe with expanding features added
    """
    df = df.copy()
    
    df[f'{column}_expanding_mean'] = df[column].expanding().mean()
    df[f'{column}_expanding_std'] = df[column].expanding().std()
    
    return df


def create_all_features(df: pd.DataFrame, target_column: str, 
                       lags: List[int] = [1, 7, 30],
                       rolling_windows: List[int] = [7, 30, 90]) -> pd.DataFrame:
    """
    Create all features for time series forecasting
    
    Args:
        df: Input dataframe
        target_column: Name of target column
        lags: List of lag periods
        rolling_windows: List of rolling window sizes
        
    Returns:
        pd.DataFrame: Dataframe with all features
    """
    df = df.copy()
    
    # Time features
    df = create_time_features(df)
    
    # Lag features
    df = create_lag_features(df, target_column, lags)
    
    # Rolling features
    df = create_rolling_features(df, target_column, rolling_windows)
    
    # Expanding features
    df = create_expanding_features(df, target_column)
    
    # Drop rows with NaN values created by lag/rolling features
    df = df.dropna()
    
    return df
