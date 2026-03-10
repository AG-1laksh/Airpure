"""
Ensemble Model Module
Combines predictions from multiple models
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import joblib
from pathlib import Path
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def average_ensemble(predictions: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
    """
    Create ensemble using weighted average
    
    Args:
        predictions: List of prediction arrays from different models
        weights: Optional weights for each model
        
    Returns:
        Ensemble predictions
    """
    if weights is None:
        # Equal weights
        weights = [1.0 / len(predictions)] * len(predictions)
    
    # Normalize weights
    weights = np.array(weights) / np.sum(weights)
    
    # Weighted average
    ensemble_pred = np.zeros_like(predictions[0])
    for pred, weight in zip(predictions, weights):
        ensemble_pred += weight * pred
    
    logger.info(f"Created weighted average ensemble with weights: {weights}")
    
    return ensemble_pred


def median_ensemble(predictions: List[np.ndarray]) -> np.ndarray:
    """
    Create ensemble using median
    
    Args:
        predictions: List of prediction arrays
        
    Returns:
        Ensemble predictions
    """
    stacked = np.stack(predictions, axis=1)
    ensemble_pred = np.median(stacked, axis=1)
    
    logger.info("Created median ensemble")
    
    return ensemble_pred


def voting_ensemble(predictions: List[np.ndarray], method: str = 'mean') -> np.ndarray:
    """
    Create voting ensemble
    
    Args:
        predictions: List of prediction arrays
        method: Voting method ('mean', 'median', 'min', 'max')
        
    Returns:
        Ensemble predictions
    """
    stacked = np.stack(predictions, axis=1)
    
    if method == 'mean':
        ensemble_pred = np.mean(stacked, axis=1)
    elif method == 'median':
        ensemble_pred = np.median(stacked, axis=1)
    elif method == 'min':
        ensemble_pred = np.min(stacked, axis=1)
    elif method == 'max':
        ensemble_pred = np.max(stacked, axis=1)
    else:
        raise ValueError(f"Unknown voting method: {method}")
    
    logger.info(f"Created {method} voting ensemble")
    
    return ensemble_pred


def stacking_ensemble(X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_test: np.ndarray,
                     base_models: List[Any],
                     meta_model: Any = None) -> np.ndarray:
    """
    Create stacking ensemble with meta-learner.
    Uses out-of-fold (OOF) predictions via TimeSeriesSplit to prevent
    training-data leakage into the meta-learner.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        base_models: List of base models
        meta_model: Meta-learner model
        
    Returns:
        Ensemble predictions
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit
    
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    logger.info(f"Creating stacking ensemble with {len(base_models)} base models (OOF, {n_splits} splits)")
    
    # OOF meta-features for training the meta-learner (no leakage)
    oof_meta_features = np.zeros((len(y_train), len(base_models)))
    # Test predictions accumulated over folds then averaged
    test_fold_preds = np.zeros((len(X_test), len(base_models), n_splits))
    
    for fold_i, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        for model_i, model in enumerate(base_models):
            model.fit(X_train[tr_idx], y_train[tr_idx])
            oof_meta_features[val_idx, model_i] = model.predict(X_train[val_idx])
            test_fold_preds[:, model_i, fold_i] = model.predict(X_test)
    
    X_train_meta = oof_meta_features
    X_test_meta = test_fold_preds.mean(axis=2)  # average test preds over folds
    
    # Train meta-model
    if meta_model is None:
        meta_model = Ridge(alpha=1.0)
    
    logger.info("Training meta-learner on OOF predictions")
    meta_model.fit(X_train_meta, y_train)
    
    # Final predictions
    ensemble_pred = meta_model.predict(X_test_meta)
    
    logger.info("Stacking ensemble complete")
    
    return ensemble_pred


def create_ensemble(model_predictions: Dict[str, np.ndarray],
                   method: str = 'weighted_average',
                   weights: Dict[str, float] = None) -> np.ndarray:
    """
    Create ensemble from model predictions
    
    Args:
        model_predictions: Dictionary of {model_name: predictions}
        method: Ensemble method
        weights: Optional weights dictionary
        
    Returns:
        Ensemble predictions
    """
    predictions_list = list(model_predictions.values())
    model_names = list(model_predictions.keys())
    
    logger.info(f"Creating ensemble from {len(model_names)} models: {model_names}")
    
    if method == 'weighted_average':
        if weights is not None:
            weight_values = [weights.get(name, 1.0) for name in model_names]
        else:
            weight_values = None
        
        ensemble_pred = average_ensemble(predictions_list, weight_values)
    
    elif method == 'median':
        ensemble_pred = median_ensemble(predictions_list)
    
    elif method in ['mean', 'min', 'max']:
        ensemble_pred = voting_ensemble(predictions_list, method)
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    return ensemble_pred


def optimize_ensemble_weights(model_predictions: Dict[str, np.ndarray],
                              y_true: np.ndarray,
                              method: str = 'minimize_rmse') -> Dict[str, float]:
    """
    Optimize ensemble weights to minimize prediction error
    
    Args:
        model_predictions: Dictionary of model predictions
        y_true: True target values
        method: Optimization method
        
    Returns:
        Optimized weights dictionary
    """
    from scipy.optimize import minimize
    from sklearn.metrics import mean_squared_error
    
    model_names = list(model_predictions.keys())
    predictions_array = np.column_stack(list(model_predictions.values()))
    
    def objective(weights):
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        
        # Calculate weighted prediction
        weighted_pred = np.dot(predictions_array, weights)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_true, weighted_pred))
        
        return rmse
    
    # Initial weights (equal)
    n_models = len(model_names)
    initial_weights = np.ones(n_models) / n_models
    
    # Constraints: weights should be non-negative and sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_models)]
    
    # Optimize
    result = minimize(objective, initial_weights, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x / np.sum(result.x)  # Normalize
    
    # Create weights dictionary
    weights_dict = dict(zip(model_names, optimal_weights))
    
    logger.info("Optimized ensemble weights:")
    for name, weight in weights_dict.items():
        logger.info(f"  {name}: {weight:.4f}")
    
    return weights_dict


def predict_ensemble(models: Dict[str, Any],
                    X: np.ndarray,
                    method: str = 'weighted_average',
                    weights: Dict[str, float] = None) -> np.ndarray:
    """
    Generate ensemble predictions from trained models
    
    Args:
        models: Dictionary of trained models
        X: Features to predict
        method: Ensemble method
        weights: Optional weights
        
    Returns:
        Ensemble predictions
    """
    # Get predictions from each model
    predictions = {}
    
    for name, model in models.items():
        try:
            pred = model.predict(X)
            if len(pred.shape) > 1:
                pred = pred.flatten()
            predictions[name] = pred
        except Exception as e:
            logger.warning(f"Could not get predictions from {name}: {str(e)}")
    
    # Create ensemble
    if predictions:
        ensemble_pred = create_ensemble(predictions, method, weights)
        return ensemble_pred
    else:
        raise ValueError("No predictions available for ensemble")


def save_ensemble_weights(weights: Dict[str, float], filename: str = "ensemble_weights.pkl"):
    """
    Save ensemble weights to file
    
    Args:
        weights: Weights dictionary
        filename: Output filename
    """
    output_path = MODELS_DIR / filename
    joblib.dump(weights, output_path)
    logger.info(f"Ensemble weights saved to {output_path}")


def load_ensemble_weights(filename: str = "ensemble_weights.pkl") -> Dict[str, float]:
    """
    Load ensemble weights from file
    
    Args:
        filename: Input filename
        
    Returns:
        Weights dictionary
    """
    input_path = MODELS_DIR / filename
    
    if not input_path.exists():
        logger.warning(f"Weights file not found: {input_path}")
        return {}
    
    weights = joblib.load(input_path)
    logger.info(f"Loaded ensemble weights from {input_path}")
    
    return weights


if __name__ == "__main__":
    # Test ensemble module
    print("Testing ensemble module...")
    
    # Generate sample predictions
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.rand(n_samples) * 200 + 50
    
    # Simulate predictions from different models
    pred1 = y_true + np.random.randn(n_samples) * 20
    pred2 = y_true + np.random.randn(n_samples) * 15
    pred3 = y_true + np.random.randn(n_samples) * 25
    
    predictions = {
        "Model 1": pred1,
        "Model 2": pred2,
        "Model 3": pred3
    }
    
    # Test different ensemble methods
    print("\n=== Testing Ensemble Methods ===")
    
    # Average ensemble
    ensemble_avg = create_ensemble(predictions, method='mean')
    print(f"Average ensemble shape: {ensemble_avg.shape}")
    
    # Median ensemble
    ensemble_median = create_ensemble(predictions, method='median')
    print(f"Median ensemble shape: {ensemble_median.shape}")
    
    # Optimize weights
    optimal_weights = optimize_ensemble_weights(predictions, y_true)
    print(f"\nOptimal weights: {optimal_weights}")
    
    # Weighted ensemble with optimal weights
    ensemble_weighted = create_ensemble(predictions, method='weighted_average', 
                                       weights=optimal_weights)
    
    from sklearn.metrics import mean_squared_error
    rmse_avg = np.sqrt(mean_squared_error(y_true, ensemble_avg))
    rmse_weighted = np.sqrt(mean_squared_error(y_true, ensemble_weighted))
    
    print(f"\nAverage ensemble RMSE: {rmse_avg:.4f}")
    print(f"Weighted ensemble RMSE: {rmse_weighted:.4f}")
