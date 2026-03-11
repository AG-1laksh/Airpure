"""
Feature Engineering Module
Creates lag features and additional features for time-series prediction
"""
import pandas as pd
import numpy as np
from typing import List
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import POLLUTION_FEATURES, METEOROLOGICAL_FEATURES, TARGET_VARIABLE, LAG_DAYS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_lag_features(df: pd.DataFrame, 
                       columns: List[str], 
                       lag_days: int = LAG_DAYS) -> pd.DataFrame:
    """
    Create lag features for time-series prediction
    
    Args:
        df: Input dataframe
        columns: Columns to create lag features for
        lag_days: Number of days to look back
        
    Returns:
        DataFrame with lag features
    """
    logger.info(f"Creating {lag_days}-day lag features for {len(columns)} variables")
    
    df_lagged = df.copy()
    
    for col in columns:
        if col in df.columns:
            for lag in range(1, lag_days + 1):
                df_lagged[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Drop rows with NaN values created by lagging
    initial_len = len(df_lagged)
    df_lagged = df_lagged.dropna()
    logger.info(f"Dropped {initial_len - len(df_lagged)} rows due to lag creation")
    
    return df_lagged.reset_index(drop=True)


def create_rolling_features(df: pd.DataFrame,
                            columns: List[str],
                            windows: List[int] = [3, 7, 14],
                            shift: int = 0) -> pd.DataFrame:
    """
    Create rolling window features (mean, std, min, max)
    
    Args:
        df: Input dataframe
        columns: Columns to create rolling features for
        windows: Window sizes for rolling calculations
        shift: Shift series before rolling (use 1 to avoid current-day leakage)
        
    Returns:
        DataFrame with rolling features
    """
    logger.info(f"Creating rolling features with windows: {windows}")
    
    df_rolling = df.copy()
    
    for col in columns:
        if col in df.columns:
            for window in windows:
                series = df[col].shift(shift) if shift else df[col]
                df_rolling[f'{col}_rolling_mean_{window}'] = series.rolling(window=window).mean()
                df_rolling[f'{col}_rolling_std_{window}'] = series.rolling(window=window).std()
                df_rolling[f'{col}_rolling_min_{window}'] = series.rolling(window=window).min()
                df_rolling[f'{col}_rolling_max_{window}'] = series.rolling(window=window).max()
    
    # Drop NaN values
    df_rolling = df_rolling.dropna().reset_index(drop=True)
    
    return df_rolling


def create_temporal_features(df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
    """
    Create temporal features from date column
    
    Args:
        df: Input dataframe
        date_col: Name of date column
        
    Returns:
        DataFrame with temporal features
    """
    logger.info("Creating temporal features")
    
    df_temporal = df.copy()
    
    if date_col in df.columns:
        df_temporal['Year'] = df_temporal[date_col].dt.year
        df_temporal['Month'] = df_temporal[date_col].dt.month
        df_temporal['Day'] = df_temporal[date_col].dt.day
        df_temporal['DayOfWeek'] = df_temporal[date_col].dt.dayofweek
        df_temporal['DayOfYear'] = df_temporal[date_col].dt.dayofyear
        df_temporal['Week'] = df_temporal[date_col].dt.isocalendar().week.astype(int)
        df_temporal['Quarter'] = df_temporal[date_col].dt.quarter
        
        # Cyclical encoding for month and day
        df_temporal['Month_sin'] = np.sin(2 * np.pi * df_temporal['Month'] / 12)
        df_temporal['Month_cos'] = np.cos(2 * np.pi * df_temporal['Month'] / 12)
        df_temporal['Day_sin'] = np.sin(2 * np.pi * df_temporal['Day'] / 31)
        df_temporal['Day_cos'] = np.cos(2 * np.pi * df_temporal['Day'] / 31)
        
        logger.info("Created temporal features: Year, Month, Day, DayOfWeek, etc.")
    
    return df_temporal


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between pollution and meteorological variables
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with interaction features
    """
    logger.info("Creating interaction features")
    
    df_interact = df.copy()
    
    # PM2.5 interactions
    if 'PM2.5' in df.columns and 'Temperature' in df.columns:
        df_interact['PM25_Temp_interaction'] = df['PM2.5'] * df['Temperature']
    
    if 'PM2.5' in df.columns and 'Humidity' in df.columns:
        df_interact['PM25_Humidity_interaction'] = df['PM2.5'] * df['Humidity']
    
    if 'PM2.5' in df.columns and 'Wind_Speed' in df.columns:
        df_interact['PM25_Wind_interaction'] = df['PM2.5'] / (df['Wind_Speed'] + 1)
    
    # NO2 interactions
    if 'NO2' in df.columns and 'Temperature' in df.columns:
        df_interact['NO2_Temp_interaction'] = df['NO2'] * df['Temperature']
    
    return df_interact


def create_pollution_index_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create composite pollution index features
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with pollution index features
    """
    logger.info("Creating pollution index features")
    
    df_index = df.copy()
    
    # Average pollution level (normalized)
    pollutants = []
    for pollutant in POLLUTION_FEATURES:
        if pollutant in df.columns:
            pollutants.append(pollutant)
    
    if pollutants:
        df_index['Avg_Pollution'] = df[pollutants].mean(axis=1)
        df_index['Max_Pollution'] = df[pollutants].max(axis=1)
        df_index['Min_Pollution'] = df[pollutants].min(axis=1)
        df_index['Pollution_Range'] = df_index['Max_Pollution'] - df_index['Min_Pollution']
    
    return df_index


def engineer_features(df: pd.DataFrame,
                     include_lag: bool = True,
                     include_rolling: bool = True,
                     include_temporal: bool = True,
                     include_interaction: bool = True,
                     include_pollution_index: bool = False,
                     lag_days: int = LAG_DAYS) -> pd.DataFrame:
    """
    Complete feature engineering pipeline
    
    Args:
        df: Input dataframe
        include_lag: Include lag features
        include_rolling: Include rolling features
        include_temporal: Include temporal features
        include_interaction: Include interaction features
        lag_days: Number of lag days
        
    Returns:
        DataFrame with engineered features
    """
    logger.info("Starting feature engineering pipeline")
    
    df_engineered = df.copy()
    
    # Temporal features
    if include_temporal and 'Date' in df.columns:
        df_engineered = create_temporal_features(df_engineered)
    
    # Interaction features
    if include_interaction:
        df_engineered = create_interaction_features(df_engineered)
    
    # Pollution index features (disabled by default to avoid target leakage)
    if include_pollution_index:
        df_engineered = create_pollution_index_features(df_engineered)
    
    # Rolling features (before lag to avoid excessive features)
    if include_rolling:
        # Rolling features for pollutants/meteorology (current-day is acceptable)
        feature_cols = list(dict.fromkeys(POLLUTION_FEATURES + METEOROLOGICAL_FEATURES))
        available_cols = [col for col in feature_cols if col in df_engineered.columns]
        if available_cols:
            df_engineered = create_rolling_features(df_engineered, available_cols, windows=[3, 7], shift=0)

        # Rolling features for target (use only past values to avoid leakage)
        if TARGET_VARIABLE in df_engineered.columns:
            df_engineered = create_rolling_features(
                df_engineered,
                [TARGET_VARIABLE],
                windows=[3, 7],
                shift=1
            )
    
    # Lag features (do this last)
    if include_lag:
        feature_cols = list(dict.fromkeys([TARGET_VARIABLE] + POLLUTION_FEATURES + METEOROLOGICAL_FEATURES))
        available_cols = [col for col in feature_cols if col in df_engineered.columns]
        if available_cols:
            df_engineered = create_lag_features(df_engineered, available_cols, lag_days)
    
    logger.info(f"Feature engineering complete. Final shape: {df_engineered.shape}")
    
    return df_engineered


def select_features_for_prediction(df: pd.DataFrame,
                                   target_col: str = 'AQI',
                                   exclude_cols: List[str] = None) -> pd.DataFrame:
    """
    Select relevant features for prediction
    
    Args:
        df: Input dataframe with engineered features
        target_col: Target variable
        exclude_cols: Additional columns to exclude
        
    Returns:
        DataFrame with selected features
    """
    if exclude_cols is None:
        exclude_cols = ['Date', 'City', 'Year']
    
    # Keep target and all features except specified exclusions
    cols_to_keep = [col for col in df.columns if col not in exclude_cols or col == target_col]
    
    return df[cols_to_keep]


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import load_data
    from preprocessing import preprocess_data
    
    print("Testing feature engineering module...")
    
    # Load and preprocess data
    df = load_data("Delhi")
    df_clean = preprocess_data(df, remove_outliers=False)
    
    print(f"Original shape: {df_clean.shape}")
    
    # Engineer features
    df_engineered = engineer_features(df_clean, lag_days=7)
    print(f"Engineered shape: {df_engineered.shape}")
    print(f"New features: {df_engineered.shape[1] - df_clean.shape[1]}")
    
    print(f"\nSample columns: {list(df_engineered.columns[:20])}")
