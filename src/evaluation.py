"""
Model Evaluation Module
Evaluates and compares different models
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from pathlib import Path
import logging
from typing import Dict, List, Any

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import TABLES_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict:
    """
    Evaluate a single model's predictions
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (handle zero values)
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred)
    except:
        mape = np.nan
    
    metrics = {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MSE": mse,
        "MAPE": mape
    }
    
    logger.info(f"\n{model_name} Evaluation:")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  R² Score: {r2:.4f}")
    
    return metrics


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple models and create comparison table
    
    Args:
        results: Dictionary with model results
        Format: {model_name: {"y_true": array, "y_pred": array}}
        
    Returns:
        DataFrame with comparison metrics
    """
    logger.info("Comparing models...")
    
    comparison_data = []
    
    for model_name, data in results.items():
        if data.get('y_true') is not None and data.get('y_pred') is not None:
            metrics = evaluate_model(data['y_true'], data['y_pred'], model_name)
            comparison_data.append(metrics)
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by RMSE (lower is better)
    comparison_df = comparison_df.sort_values('RMSE', ascending=True)
    
    logger.info("\n=== Model Comparison ===")
    logger.info(f"\n{comparison_df.to_string()}")
    
    # Save to CSV
    output_path = TABLES_DIR / "model_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    logger.info(f"\nComparison saved to {output_path}")
    
    return comparison_df


def get_best_model(comparison_df: pd.DataFrame, metric: str = "RMSE") -> str:
    """
    Get the best performing model
    
    Args:
        comparison_df: Model comparison dataframe
        metric: Metric to use for comparison
        
    Returns:
        Name of best model
    """
    if metric in ["RMSE", "MAE", "MSE", "MAPE"]:
        # Lower is better
        best_model = comparison_df.loc[comparison_df[metric].idxmin(), 'Model']
    elif metric in ["R2"]:
        # Higher is better
        best_model = comparison_df.loc[comparison_df[metric].idxmax(), 'Model']
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    logger.info(f"\nBest model based on {metric}: {best_model}")
    
    return best_model


def calculate_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate prediction residuals
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Residuals array
    """
    return y_true - y_pred


def analyze_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Analyze prediction residuals
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with residual statistics
    """
    residuals = calculate_residuals(y_true, y_pred)
    
    analysis = {
        "mean": np.mean(residuals),
        "std": np.std(residuals),
        "min": np.min(residuals),
        "max": np.max(residuals),
        "median": np.median(residuals),
        "q25": np.percentile(residuals, 25),
        "q75": np.percentile(residuals, 75)
    }
    
    logger.info("\nResidual Analysis:")
    for key, value in analysis.items():
        logger.info(f"  {key}: {value:.4f}")
    
    return analysis


def calculate_accuracy_by_aqi_category(y_true: np.ndarray, 
                                       y_pred: np.ndarray) -> pd.DataFrame:
    """
    Calculate prediction accuracy for different AQI categories
    
    Args:
        y_true: True AQI values
        y_pred: Predicted AQI values
        
    Returns:
        DataFrame with accuracy by category
    """
    # Define AQI categories
    categories = [
        ("Good", 0, 50),
        ("Satisfactory", 51, 100),
        ("Moderate", 101, 200),
        ("Poor", 201, 300),
        ("Very Poor", 301, 400),
        ("Severe", 401, 500)
    ]
    
    results = []
    
    for cat_name, low, high in categories:
        # Find samples in this category
        mask = (y_true >= low) & (y_true <= high)
        
        if np.sum(mask) > 0:
            cat_true = y_true[mask]
            cat_pred = y_pred[mask]
            
            rmse = np.sqrt(mean_squared_error(cat_true, cat_pred))
            mae = mean_absolute_error(cat_true, cat_pred)
            
            results.append({
                "Category": cat_name,
                "Range": f"{low}-{high}",
                "Count": np.sum(mask),
                "RMSE": rmse,
                "MAE": mae
            })
    
    df = pd.DataFrame(results)
    
    logger.info("\n=== Accuracy by AQI Category ===")
    logger.info(f"\n{df.to_string()}")
    
    return df


def cross_validate_model(model: Any,
                        X: np.ndarray,
                        y: np.ndarray,
                        cv: int = 5) -> Dict:
    """
    Perform cross-validation on a model using time-series aware splits
    
    Args:
        model: Model to validate
        X: Features
        y: Target
        cv: Number of folds
        
    Returns:
        Dictionary with CV results
    """
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    
    logger.info(f"Performing {cv}-fold time-series cross-validation...")
    
    # TimeSeriesSplit ensures no future data leaks into earlier folds
    tscv = TimeSeriesSplit(n_splits=cv)
    
    # Calculate scores for different metrics
    rmse_scores = np.sqrt(-cross_val_score(
        model, X, y, cv=tscv, scoring='neg_mean_squared_error'
    ))
    mae_scores = -cross_val_score(
        model, X, y, cv=tscv, scoring='neg_mean_absolute_error'
    )
    r2_scores = cross_val_score(
        model, X, y, cv=tscv, scoring='r2'
    )
    
    results = {
        "RMSE": {
            "mean": rmse_scores.mean(),
            "std": rmse_scores.std(),
            "scores": rmse_scores
        },
        "MAE": {
            "mean": mae_scores.mean(),
            "std": mae_scores.std(),
            "scores": mae_scores
        },
        "R2": {
            "mean": r2_scores.mean(),
            "std": r2_scores.std(),
            "scores": r2_scores
        }
    }
    
    logger.info("\nCross-Validation Results:")
    logger.info(f"  RMSE: {results['RMSE']['mean']:.4f} (+/- {results['RMSE']['std']:.4f})")
    logger.info(f"  MAE: {results['MAE']['mean']:.4f} (+/- {results['MAE']['std']:.4f})")
    logger.info(f"  R²: {results['R2']['mean']:.4f} (+/- {results['R2']['std']:.4f})")
    
    return results


def save_evaluation_results(comparison_df: pd.DataFrame,
                           filename: str = "evaluation_results.csv"):
    """
    Save evaluation results to file
    
    Args:
        comparison_df: Comparison dataframe
        filename: Output filename
    """
    output_path = TABLES_DIR / filename
    comparison_df.to_csv(output_path, index=False)
    logger.info(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    # Test evaluation module
    print("Testing evaluation module...")
    
    # Generate sample predictions
    np.random.seed(42)
    y_true = np.random.rand(100) * 200 + 50  # AQI values 50-250
    y_pred1 = y_true + np.random.randn(100) * 20  # Model 1
    y_pred2 = y_true + np.random.randn(100) * 15  # Model 2 (better)
    
    # Evaluate individual models
    metrics1 = evaluate_model(y_true, y_pred1, "Model 1")
    metrics2 = evaluate_model(y_true, y_pred2, "Model 2")
    
    # Compare models
    results = {
        "Model 1": {"y_true": y_true, "y_pred": y_pred1},
        "Model 2": {"y_true": y_true, "y_pred": y_pred2}
    }
    
    comparison = compare_models(results)
    best = get_best_model(comparison)
    
    print(f"\nBest model: {best}")
