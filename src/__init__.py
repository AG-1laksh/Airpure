"""
Air Pollution Prediction System - Source Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .data_loader import load_data, download_data
from .preprocessing import preprocess_data, create_sequences
from .feature_engineering import create_lag_features, engineer_features
from .ml_models import train_ml_models, get_ml_model
from .lstm_model import build_lstm_model, train_lstm
from .ensemble import create_ensemble, predict_ensemble
from .evaluation import evaluate_model, compare_models
from .explainability import calculate_shap_values, plot_shap_summary
from .visualization import (
    plot_time_series,
    plot_correlation_matrix,
    plot_feature_importance,
    plot_predictions,
    plot_model_comparison,
    plot_aqi_distribution,
    plot_residuals,
    plot_learning_curve,
    plot_seasonal_analysis,
    plot_yearly_trend,
)

__all__ = [
    "load_data",
    "download_data",
    "preprocess_data",
    "create_sequences",
    "create_lag_features",
    "engineer_features",
    "train_ml_models",
    "get_ml_model",
    "build_lstm_model",
    "train_lstm",
    "create_ensemble",
    "predict_ensemble",
    "evaluate_model",
    "compare_models",
    "calculate_shap_values",
    "plot_shap_summary",
    "plot_time_series",
    "plot_correlation_matrix",
    "plot_feature_importance",
    "plot_predictions",
    "plot_model_comparison",
    "plot_aqi_distribution",
    "plot_residuals",
    "plot_learning_curve",
    "plot_seasonal_analysis",
    "plot_yearly_trend",
]
