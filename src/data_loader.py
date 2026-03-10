"""
Data Loading Module
Handles loading air quality data from various sources
"""
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from typing import Optional, List
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR, CITIES, get_city_raw_dir, get_city_processed_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_delhi_excel_data() -> Optional[pd.DataFrame]:
    """
    Load Delhi AQI data from year-wise Excel files.

    The files are in wide format (rows = days 1-31, columns = months),
    one file per year. This function melts them into a long-format
    time-series DataFrame.

    Returns:
        DataFrame with columns [Date, AQI, City], sorted by Date,
        or None if no Excel files are found.
    """
    city_raw_dir = get_city_raw_dir("Delhi")
    excel_files = sorted(city_raw_dir.glob("AQI_daily_city_level_delhi_*.xlsx"))

    if not excel_files:
        logger.warning("No Delhi Excel files found.")
        return None

    months = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    month_map = {m: i + 1 for i, m in enumerate(months)}

    all_dfs = []
    for fp in excel_files:
        try:
            year = int(fp.stem.split('_')[-1])
        except ValueError:
            logger.warning(f"Cannot parse year from {fp.name}, skipping.")
            continue

        df = pd.read_excel(fp)

        # Handle both 'Date' (2018-2020) and 'Day' (2021+) column names
        day_col = 'Date' if 'Date' in df.columns else 'Day'

        # Keep only numeric day rows (1-31); drop summary rows like "Good", "SP", etc.
        df = df[pd.to_numeric(df[day_col], errors='coerce').notna()].copy()
        df[day_col] = df[day_col].astype(int)
        df = df[df[day_col].between(1, 31)]

        present_months = [m for m in months if m in df.columns]

        df_melted = df.melt(
            id_vars=day_col,
            value_vars=present_months,
            var_name='Month',
            value_name='AQI'
        )
        df_melted = df_melted.rename(columns={day_col: 'Day'})
        df_melted = df_melted.dropna(subset=['AQI'])
        df_melted['Month_num'] = df_melted['Month'].map(month_map)

        # Build date strings; coerce invalid dates (e.g. Feb 30) to NaT
        date_str = (
            str(year) + '-' +
            df_melted['Month_num'].astype(str).str.zfill(2) + '-' +
            df_melted['Day'].astype(int).astype(str).str.zfill(2)
        )
        df_melted['Date'] = pd.to_datetime(date_str, errors='coerce')
        df_melted = df_melted.dropna(subset=['Date'])

        all_dfs.append(df_melted[['Date', 'AQI']].copy())
        logger.info(f"Loaded {len(df_melted)} records from {fp.name}")

    if not all_dfs:
        return None

    combined = (
        pd.concat(all_dfs, ignore_index=True)
        .sort_values('Date')
        .reset_index(drop=True)
    )
    combined['City'] = 'Delhi'
    combined['AQI'] = combined['AQI'].round(0).astype(int)

    logger.info(
        f"Delhi Excel data: {len(combined)} records "
        f"({combined['Date'].min().date()} → {combined['Date'].max().date()})"
    )

    # Save merged CSV to processed folder for fast re-use
    processed_dir = get_city_processed_dir("Delhi")
    processed_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = processed_dir / "Delhi_AQI_merged.csv"
    combined.to_csv(merged_csv, index=False)
    logger.info(f"Merged CSV saved to {merged_csv}")

    return combined


def load_delhi_kaggle_data(source_file: str = "final_dataset") -> Optional[pd.DataFrame]:
    """
    Load Delhi air quality data from the Kaggle dataset
    (kunshbhatia/delhi-air-quality-dataset).

    The raw files have separate Day/Month/Year integer columns and use
    'Ozone' instead of 'O3'.  This function normalises them into the
    standard project format.

    Args:
        source_file: Base filename without extension.
                     One of 'final_dataset', 'Cleaned_NSUT', 'FINAL_ITO_DATA'.

    Returns:
        Normalised DataFrame or None if the file is not found.
    """
    city_raw_dir = get_city_raw_dir("Delhi")

    # Try CSV first, then XLSX
    fp = None
    for ext in [".csv", ".xlsx"]:
        candidate = city_raw_dir / f"{source_file}{ext}"
        if candidate.exists():
            fp = candidate
            break

    if fp is None:
        logger.warning(f"Kaggle dataset '{source_file}' not found in {city_raw_dir}")
        return None

    logger.info(f"Loading Delhi Kaggle dataset from {fp}")
    df = pd.read_csv(fp) if fp.suffix == ".csv" else pd.read_excel(fp)

    # Reconstruct a proper datetime from the separate Day / Month / Year columns.
    # The source 'Date' column contains day-of-month integers (1-31).
    df["Date"] = pd.to_datetime(
        {"year": df["Year"], "month": df["Month"], "day": df["Date"]},
        errors="coerce",
    )
    df = df.dropna(subset=["Date"])

    # Rename 'Ozone' → 'O3' to match the project's POLLUTION_FEATURES config.
    if "Ozone" in df.columns:
        df = df.rename(columns={"Ozone": "O3"})

    # Drop columns that are now redundant (Date/temporal features are
    # reconstructed from the proper Date by feature_engineering).
    df = df.drop(columns=["Month", "Year", "Days"], errors="ignore")

    df["City"] = "Delhi"
    df = df.sort_values("Date").reset_index(drop=True)

    logger.info(
        f"Loaded {len(df)} Delhi Kaggle records "
        f"({df['Date'].min().date()} \u2192 {df['Date'].max().date()})"
    )
    return df


def load_data(city: str, file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load air quality data for a specific city.

    For Delhi, tries (in order):
      1. Kaggle dataset files in data/Delhi/raw/ (final_dataset.csv preferred)
      2. Year-wise Excel files (AQI_daily_city_level_delhi_*.xlsx)
      3. Generic CSV at data/<City>/raw/<City>_air_quality.csv
      4. Synthetic sample data

    Args:
        city: City name (Delhi, Mumbai, Chennai, Bangalore)
        file_path: Optional custom file path (overrides auto-detection)

    Returns:
        DataFrame with air quality data
    """
    if city not in CITIES:
        logger.warning(f"City {city} not in predefined list. Proceeding anyway.")

    if file_path is None:
        if city == "Delhi":
            # 1. Try Kaggle dataset files (preferred: more parameters)
            for kaggle_name in ["final_dataset", "Cleaned_NSUT", "FINAL_ITO_DATA"]:
                df = load_delhi_kaggle_data(source_file=kaggle_name)
                if df is not None:
                    return df

            # 2. Fall back to year-wise Excel files
            city_raw_dir = get_city_raw_dir(city)
            excel_files = list(city_raw_dir.glob("AQI_daily_city_level_delhi_*.xlsx"))
            if excel_files:
                logger.info(f"Found {len(excel_files)} Delhi Excel file(s). Loading...")
                df_excel = load_delhi_excel_data()
                if df_excel is not None:
                    return df_excel

        # 3. Generic path: city folder CSV → legacy CSV
        city_path = get_city_raw_dir(city) / f"{city}_air_quality.csv"
        legacy_path = RAW_DATA_DIR / f"{city}_air_quality.csv"
        file_path = city_path if city_path.exists() else legacy_path

    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        elif 'date' in df.columns:
            df.rename(columns={'date': 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])

        logger.info(f"Loaded {len(df)} records for {city}")
        return df

    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}. Generating sample data...")
        return create_sample_data(city)
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def create_sample_data(city: str, n_samples: int = 1000) -> pd.DataFrame:
    """
    Create sample air quality data for demonstration
    
    Args:
        city: City name
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with synthetic data
    """
    logger.info(f"Generating {n_samples} sample records for {city}")
    
    np.random.seed(42)
    
    # Generate date range
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_samples, freq='D')
    
    # Generate pollution data with realistic patterns
    # Adding seasonal and trend components
    trend = np.linspace(100, 150, n_samples)
    seasonal = 30 * np.sin(np.arange(n_samples) * 2 * np.pi / 365)
    noise = np.random.normal(0, 20, n_samples)
    
    pm25 = np.clip(trend + seasonal + noise, 0, 500)
    pm10 = pm25 * 1.5 + np.random.normal(0, 10, n_samples)
    no2 = np.clip(40 + np.random.normal(0, 15, n_samples), 0, 200)
    so2 = np.clip(15 + np.random.normal(0, 5, n_samples), 0, 80)
    co = np.clip(1.2 + np.random.normal(0, 0.3, n_samples), 0, 10)
    o3 = np.clip(50 + np.random.normal(0, 20, n_samples), 0, 200)
    
    # Generate meteorological data
    temp = 25 + 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 365) + np.random.normal(0, 3, n_samples)
    humidity = np.clip(60 + np.random.normal(0, 15, n_samples), 20, 100)
    wind_speed = np.clip(np.abs(np.random.normal(10, 5, n_samples)), 0, 50)
    rainfall = np.clip(np.random.exponential(2, n_samples), 0, 100)
    
    # Calculate AQI (simplified - based primarily on PM2.5)
    aqi = calculate_aqi_from_pm25(pm25)
    
    df = pd.DataFrame({
        'Date': dates,
        'City': city,
        'PM2.5': pm25,
        'PM10': pm10,
        'NO2': no2,
        'SO2': so2,
        'CO': co,
        'O3': o3,
        'Temperature': temp,
        'Humidity': humidity,
        'Wind_Speed': wind_speed,
        'Rainfall': rainfall,
        'AQI': aqi
    })
    
    # Save sample data to city-specific folder
    city_raw_dir = get_city_raw_dir(city)
    city_raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = city_raw_dir / f"{city}_air_quality.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Sample data saved to {output_path}")
    
    return df


def calculate_aqi_from_pm25(pm25: np.ndarray) -> np.ndarray:
    """
    Calculate AQI from PM2.5 values (simplified Indian standard)
    
    Args:
        pm25: PM2.5 concentration values
        
    Returns:
        AQI values
    """
    # Simplified AQI calculation based on PM2.5
    # This is a rough approximation
    aqi = np.zeros_like(pm25)
    
    # Define breakpoints (PM2.5, AQI)
    breakpoints = [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500)
    ]
    
    for pm_low, pm_high, aqi_low, aqi_high in breakpoints:
        mask = (pm25 >= pm_low) & (pm25 <= pm_high)
        aqi[mask] = ((aqi_high - aqi_low) / (pm_high - pm_low)) * (pm25[mask] - pm_low) + aqi_low
    
    # Handle values above 500
    aqi[pm25 > 500] = 500
    
    return aqi


def download_from_openaq(city: str, parameter: str = "pm25", 
                         limit: int = 10000) -> pd.DataFrame:
    """
    Download data from OpenAQ API
    
    Args:
        city: City name
        parameter: Pollutant parameter
        limit: Maximum number of records
        
    Returns:
        DataFrame with downloaded data
    """
    logger.info(f"Downloading {parameter} data for {city} from OpenAQ...")
    
    url = "https://api.openaq.org/v2/measurements"
    params = {
        "city": city,
        "parameter": parameter,
        "limit": limit,
        "order_by": "datetime",
        "sort": "desc"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = data.get('results', [])
        
        if not results:
            logger.warning(f"No data returned for {city}")
            return pd.DataFrame()
        
        # Parse results
        records = []
        for item in results:
            records.append({
                'Date': item['date']['utc'],
                'City': item['city'],
                'Parameter': item['parameter'],
                'Value': item['value'],
                'Unit': item['unit']
            })
        
        df = pd.DataFrame(records)
        logger.info(f"Downloaded {len(df)} records")
        
        return df
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading from OpenAQ: {str(e)}")
        return pd.DataFrame()


def download_data(city: str, source: str = "sample") -> pd.DataFrame:
    """
    Download air quality data from specified source
    
    Args:
        city: City name
        source: Data source ('sample', 'openaq', 'kaggle')
        
    Returns:
        DataFrame with air quality data
    """
    if source == "sample":
        return create_sample_data(city)
    elif source == "openaq":
        return download_from_openaq(city)
    else:
        logger.warning(f"Source {source} not implemented. Using sample data.")
        return create_sample_data(city)


def load_multiple_cities(cities: List[str]) -> pd.DataFrame:
    """
    Load data for multiple cities and combine
    
    Args:
        cities: List of city names
        
    Returns:
        Combined DataFrame
    """
    dfs = []
    for city in cities:
        try:
            df = load_data(city)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not load data for {city}: {str(e)}")
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined data: {len(combined)} total records")
        return combined
    else:
        raise ValueError("No data could be loaded for any city")


if __name__ == "__main__":
    # Test the module
    print("Testing data loader...")
    
    # Create sample data for Delhi
    df = load_data("Delhi")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nSummary statistics:\n{df.describe()}")
