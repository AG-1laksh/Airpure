"""
Data Preprocessing Module
Handles data cleaning, scaling, and preparation
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from typing import Tuple, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    POLLUTION_FEATURES, METEOROLOGICAL_FEATURES, TARGET_VARIABLE,
    PROCESSED_DATA_DIR, TEST_SIZE, RANDOM_STATE, get_city_processed_dir
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame, 
                    scaler_type: str = "standard",
                    handle_missing: str = "ffill",
                    remove_outliers: bool = True) -> pd.DataFrame:
    """
    Preprocess air quality data
    
    Args:
        df: Raw dataframe
        scaler_type: Type of scaler ('standard' or 'minmax')
        handle_missing: Method to handle missing values ('ffill', 'drop', 'interpolate')
        remove_outliers: Whether to remove outliers
        
    Returns:
        Preprocessed dataframe
    """
    logger.info("Starting data preprocessing...")
    df_clean = df.copy()
    
    # 1. Handle missing values
    logger.info(f"Missing values before: {df_clean.isnull().sum().sum()}")
    df_clean = handle_missing_values(df_clean, method=handle_missing)
    logger.info(f"Missing values after: {df_clean.isnull().sum().sum()}")
    
    # 2. Remove duplicates
    before_dup = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    logger.info(f"Removed {before_dup - len(df_clean)} duplicate rows")
    
    # 3. Sort by date
    if 'Date' in df_clean.columns:
        df_clean = df_clean.sort_values('Date').reset_index(drop=True)
    
    # 4. Handle outliers
    if remove_outliers:
        df_clean = remove_outliers_iqr(df_clean)
    
    # 5. Feature scaling (exclude Date and City columns)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if 'Date' in numeric_cols:
        numeric_cols.remove('Date')
    
    logger.info(f"Scaling {len(numeric_cols)} numeric features using {scaler_type}")
    
    # Note: We'll save the scaler for later use in predictions
    # For now, we'll scale but can also return unscaled version
    
    logger.info("Preprocessing completed")
    return df_clean


def handle_missing_values(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Handle missing values in the dataset
    
    Args:
        df: Input dataframe
        method: Method to handle missing values
        
    Returns:
        DataFrame with missing values handled
    """
    df_filled = df.copy()
    
    if method == "ffill":
        # Forward fill - use previous value
        df_filled = df_filled.fillna(method='ffill')
        # Backward fill for any remaining
        df_filled = df_filled.fillna(method='bfill')
    
    elif method == "interpolate":
        # Linear interpolation
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        df_filled[numeric_cols] = df_filled[numeric_cols].interpolate(method='linear')
        df_filled = df_filled.fillna(method='bfill')
    
    elif method == "drop":
        # Drop rows with missing values
        df_filled = df_filled.dropna()
    
    else:
        logger.warning(f"Unknown method {method}, using forward fill")
        df_filled = df_filled.fillna(method='ffill').fillna(method='bfill')
    
    return df_filled


def remove_outliers_iqr(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers using IQR method
    
    Args:
        df: Input dataframe
        multiplier: IQR multiplier for outlier detection
        
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    # Exclude Date column if it's numeric
    numeric_cols = [col for col in numeric_cols if col not in ['Date']]
    
    initial_len = len(df_clean)
    
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Filter outliers
        df_clean = df_clean[
            (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        ]
    
    removed = initial_len - len(df_clean)
    logger.info(f"Removed {removed} outlier rows ({removed/initial_len*100:.2f}%)")
    
    return df_clean.reset_index(drop=True)


def scale_features(df: pd.DataFrame, 
                   scaler_type: str = "standard",
                   exclude_cols: list = None) -> Tuple[pd.DataFrame, object]:
    """
    Scale numeric features
    
    Args:
        df: Input dataframe
        scaler_type: Type of scaler
        exclude_cols: Columns to exclude from scaling
        
    Returns:
        Scaled dataframe and fitted scaler
    """
    if exclude_cols is None:
        exclude_cols = ['Date', 'City']
    
    df_scaled = df.copy()
    
    # Get numeric columns
    numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
    
    # Select scaler
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Fit and transform
    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
    
    logger.info(f"Scaled {len(cols_to_scale)} features using {scaler_type} scaler")
    
    return df_scaled, scaler


def create_sequences(data: np.ndarray, time_steps: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time-series prediction (for LSTM)
    
    Args:
        data: Input data array
        time_steps: Number of time steps to look back
        
    Returns:
        X (sequences) and y (targets)
    """
    X, y = [], []
    
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :])
        y.append(data[i + time_steps, -1])  # Assuming target is last column
    
    return np.array(X), np.array(y)


def prepare_train_test_split(df: pd.DataFrame, 
                             test_size: float = TEST_SIZE,
                             target_col: str = TARGET_VARIABLE) -> Tuple:
    """
    Split data into train and test sets
    
    Args:
        df: Input dataframe
        test_size: Proportion of test data
        target_col: Target variable column name
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Separate features and target
    feature_cols = [col for col in df.columns 
                   if col not in [target_col, 'Date', 'City']]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Split (time-series aware - no shuffling)
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def save_processed_data(df: pd.DataFrame, city: str, suffix: str = "processed"):
    """
    Save processed data to CSV
    
    Args:
        df: Processed dataframe
        city: City name
        suffix: File suffix
    """
    output_path = PROCESSED_DATA_DIR / f"{city}_{suffix}.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of the dataset
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "n_samples": len(df),
        "n_features": len(df.columns),
        "date_range": None,
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().to_dict()
    }
    
    if 'Date' in df.columns:
        summary["date_range"] = {
            "start": df['Date'].min(),
            "end": df['Date'].max()
        }
    
    return summary


if __name__ == "__main__":
    # Test preprocessing
    from data_loader import load_data
    
    print("Testing preprocessing module...")
    
    # Load sample data
    df = load_data("Delhi")
    print(f"Original data shape: {df.shape}")
    
    # Preprocess
    df_clean = preprocess_data(df)
    print(f"Cleaned data shape: {df_clean.shape}")
    
    # Get summary
    summary = get_data_summary(df_clean)
    print(f"\nData summary:")
    print(f"Samples: {summary['n_samples']}")
    print(f"Features: {summary['n_features']}")
    
    # Prepare train/test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(df_clean)
    print(f"\nTrain/Test split:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
