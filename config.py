"""
Configuration file for Air Pollution Prediction System
"""
import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"           # legacy fallback
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # legacy fallback

# City-specific data directories (preferred)
def get_city_raw_dir(city: str):
    return DATA_DIR / city / "raw"

def get_city_processed_dir(city: str):
    return DATA_DIR / city / "processed"

# Model Directory
MODELS_DIR = BASE_DIR / "models"

# Results Directory
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
                  FIGURES_DIR, TABLES_DIR, PREDICTIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Target Cities
CITIES = ["Delhi", "Mumbai", "Chennai", "Bangalore"]

# Features Configuration
POLLUTION_FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
METEOROLOGICAL_FEATURES = ["Temperature", "Humidity", "Wind_Speed", "Rainfall"]
TARGET_VARIABLE = "AQI"

# All features
ALL_FEATURES = POLLUTION_FEATURES + METEOROLOGICAL_FEATURES

# Time-series Configuration
LAG_DAYS = 7  # Number of previous days to use for prediction
PREDICTION_HORIZONS = [1, 3, 7]  # Days ahead to predict

# LSTM Configuration
LSTM_CONFIG = {
    "time_steps": 7,
    "lstm_units": 64,
    "dropout_rate": 0.2,
    "epochs": 100,
    "batch_size": 32,
    "validation_split": 0.2
}

# Model Configuration
ML_MODELS = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
    "Support Vector Machine",
    "Gradient Boosting",
    "XGBoost"
]

# Train-Test Split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Evaluation Metrics
METRICS = ["RMSE", "MAE", "R2"]

# Visualization Settings
PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_SIZE = (12, 6)
DPI = 300

# AQI Categories (India Standard)
AQI_CATEGORIES = {
    "Good": (0, 50),
    "Satisfactory": (51, 100),
    "Moderate": (101, 200),
    "Poor": (201, 300),
    "Very Poor": (301, 400),
    "Severe": (401, 500)
}

# Color scheme for AQI categories
AQI_COLORS = {
    "Good": "#00E400",
    "Satisfactory": "#FFFF00",
    "Moderate": "#FF7E00",
    "Poor": "#FF0000",
    "Very Poor": "#8F3F97",
    "Severe": "#7E0023"
}

# Data Sources
DATA_SOURCES = {
    "OpenAQ": "https://openaq.org/",
    "CPCB": "https://cpcb.nic.in/",
    "Kaggle": "https://www.kaggle.com/datasets"
}

# API Configuration (if using real-time data)
API_CONFIG = {
    "openaq_api": "https://api.openaq.org/v2/",
    "request_timeout": 30
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    "title": "Air Pollution Prediction Dashboard",
    "page_icon": "🌍",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Research Paper Sections
PAPER_SECTIONS = [
    "Abstract",
    "Introduction",
    "Literature Review",
    "Dataset Description",
    "Methodology",
    "Machine Learning Models",
    "LSTM Deep Learning Model",
    "Experimental Results",
    "Feature Importance Analysis",
    "Discussion",
    "Conclusion",
    "Future Work"
]

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
