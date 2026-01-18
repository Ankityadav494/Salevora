"""
Data preprocessing utilities
"""
import pandas as pd
import numpy as np
from typing import Optional


def handle_missing_values(df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
    """
    Handle missing values in the dataframe
    
    Args:
        df: Input dataframe
        method: Method to handle missing values ('forward_fill', 'backward_fill', 'mean', 'median', 'drop')
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    df = df.copy()
    
    if method == 'forward_fill':
        df = df.fillna(method='ffill')
    elif method == 'backward_fill':
        df = df.fillna(method='bfill')
    elif method == 'mean':
        df = df.fillna(df.mean())
    elif method == 'median':
        df = df.fillna(df.median())
    elif method == 'drop':
        df = df.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return df


def remove_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
    """
    Remove outliers from a specific column
    
    Args:
        df: Input dataframe
        column: Column name to remove outliers from
        method: Method to detect outliers ('iqr', 'zscore')
        
    Returns:
        pd.DataFrame: Dataframe with outliers removed
    """
    df = df.copy()
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        df = df[z_scores < 3]
    
    return df


def convert_to_datetime(df: pd.DataFrame, column: str, format: Optional[str] = None) -> pd.DataFrame:
    """
    Convert column to datetime format
    
    Args:
        df: Input dataframe
        column: Column name to convert
        format: Optional datetime format string
        
    Returns:
        pd.DataFrame: Dataframe with converted datetime column
    """
    df = df.copy()
    
    if format:
        df[column] = pd.to_datetime(df[column], format=format)
    else:
        df[column] = pd.to_datetime(df[column])
    
    return df


def set_datetime_index(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Set datetime column as index
    
    Args:
        df: Input dataframe
        column: Datetime column name
        
    Returns:
        pd.DataFrame: Dataframe with datetime index
    """
    df = df.copy()
    df = convert_to_datetime(df, column)
    df = df.set_index(column)
    df = df.sort_index()
    
    return df


def preprocess_sales_data(df: pd.DataFrame, date_column: str = 'date', 
                         target_column: str = 'sales') -> pd.DataFrame:
    """
    Complete preprocessing pipeline for sales data
    
    Args:
        df: Input dataframe
        date_column: Name of the date column
        target_column: Name of the target column (sales)
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    df = df.copy()
    
    # Convert to datetime
    df = convert_to_datetime(df, date_column)
    
    # Set datetime as index
    df = set_datetime_index(df, date_column)
    
    # Handle missing values
    df = handle_missing_values(df, method='forward_fill')
    
    # Remove outliers from target column
    if target_column in df.columns:
        df = remove_outliers(df, target_column, method='iqr')
    
    return df
