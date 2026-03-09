"""
Machine Learning Models Module
Implements traditional ML models for AQI prediction
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import logging
from typing import Dict, Tuple, Any

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR, RANDOM_STATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_ml_model(model_name: str, **kwargs) -> Any:
    """
    Get ML model by name
    
    Args:
        model_name: Name of the model
        **kwargs: Additional parameters for the model
        
    Returns:
        Initialized model
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            random_state=kwargs.get('random_state', RANDOM_STATE),
            max_depth=kwargs.get('max_depth', 10)
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            random_state=kwargs.get('random_state', RANDOM_STATE),
            max_depth=kwargs.get('max_depth', 15),
            n_jobs=-1
        ),
        "Support Vector Machine": Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 100.0),
                epsilon=kwargs.get('epsilon', 0.1)
            ))
        ]),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=kwargs.get('random_state', RANDOM_STATE),
            max_depth=kwargs.get('max_depth', 5)
        ),
        "XGBoost": XGBRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=kwargs.get('random_state', RANDOM_STATE),
            max_depth=kwargs.get('max_depth', 6)
        )
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")
    
    return models[model_name]


def train_single_model(model_name: str,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_test: np.ndarray = None,
                       y_test: np.ndarray = None,
                       save_model: bool = True,
                       **kwargs) -> Tuple[Any, Dict]:
    """
    Train a single ML model
    
    Args:
        model_name: Name of the model
        X_train: Training features
        y_train: Training target
        X_test: Test features (optional)
        y_test: Test target (optional)
        save_model: Whether to save the trained model
        **kwargs: Additional model parameters
        
    Returns:
        Trained model and training info
    """
    logger.info(f"Training {model_name}...")
    
    # Get model
    model = get_ml_model(model_name, **kwargs)
    
    # Train
    model.fit(X_train, y_train)
    
    # Get training score
    train_score = model.score(X_train, y_train)
    logger.info(f"{model_name} - Training R² Score: {train_score:.4f}")
    
    info = {
        "model_name": model_name,
        "train_score": train_score,
        "n_features": X_train.shape[1],
        "n_samples": X_train.shape[0]
    }
    
    # Test score if available
    if X_test is not None and y_test is not None:
        test_score = model.score(X_test, y_test)
        info["test_score"] = test_score
        logger.info(f"{model_name} - Test R² Score: {test_score:.4f}")
    
    # Save model
    if save_model:
        model_path = MODELS_DIR / f"{model_name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        info["model_path"] = str(model_path)
    
    return model, info


def train_ml_models(X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_test: np.ndarray = None,
                    y_test: np.ndarray = None,
                    models_to_train: list = None,
                    save_models: bool = True) -> Dict:
    """
    Train multiple ML models
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models_to_train: List of model names to train
        save_models: Whether to save trained models
        
    Returns:
        Dictionary of trained models and their info
    """
    if models_to_train is None:
        models_to_train = [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "Support Vector Machine",
            "Gradient Boosting",
            "XGBoost"
        ]
    
    logger.info(f"Training {len(models_to_train)} models...")
    
    results = {}
    
    for model_name in models_to_train:
        try:
            model, info = train_single_model(
                model_name, X_train, y_train, X_test, y_test, save_models
            )
            results[model_name] = {
                "model": model,
                "info": info
            }
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            results[model_name] = {
                "model": None,
                "info": {"error": str(e)}
            }
    
    logger.info("All models trained successfully")
    
    return results


def predict_with_model(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Make predictions with a trained model
    
    Args:
        model: Trained model
        X: Features
        
    Returns:
        Predictions
    """
    return model.predict(X)


def load_saved_model(model_name: str) -> Any:
    """
    Load a saved model from disk
    
    Args:
        model_name: Name of the model
        
    Returns:
        Loaded model
    """
    model_path = MODELS_DIR / f"{model_name.replace(' ', '_').lower()}_model.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    return model


def get_feature_importance(model: Any, feature_names: list = None) -> pd.DataFrame:
    """
    Get feature importance from tree-based models
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    else:
        logger.warning(f"Model {type(model).__name__} does not have feature_importances_")
        return pd.DataFrame()


def hyperparameter_tuning(model_name: str,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         param_grid: dict = None) -> Tuple[Any, Dict]:
    """
    Perform hyperparameter tuning using grid search
    
    Args:
        model_name: Name of the model
        X_train: Training features
        y_train: Training target
        param_grid: Parameter grid for tuning
        
    Returns:
        Best model and best parameters
    """
    from sklearn.model_selection import GridSearchCV
    
    logger.info(f"Performing hyperparameter tuning for {model_name}...")
    
    # Default parameter grids
    if param_grid is None:
        default_grids = {
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5]
            },
            "XGBoost": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 6, 9]
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7]
            }
        }
        param_grid = default_grids.get(model_name, {})
    
    if not param_grid:
        logger.warning(f"No parameter grid for {model_name}, skipping tuning")
        return train_single_model(model_name, X_train, y_train)
    
    # Get base model
    base_model = get_ml_model(model_name)
    
    # Grid search
    grid_search = GridSearchCV(
        base_model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


if __name__ == "__main__":
    # Test ML models
    from data_loader import load_data
    from preprocessing import preprocess_data, prepare_train_test_split
    
    print("Testing ML models module...")
    
    # Load and prepare data
    df = load_data("Delhi")
    df_clean = preprocess_data(df, remove_outliers=False)
    
    X_train, X_test, y_train, y_test = prepare_train_test_split(df_clean)
    
    print(f"Training data shape: {X_train.shape}")
    
    # Train models
    results = train_ml_models(X_train, y_train, X_test, y_test)
    
    print("\n=== Model Results ===")
    for model_name, result in results.items():
        if result['info'].get('error'):
            print(f"{model_name}: ERROR - {result['info']['error']}")
        else:
            print(f"{model_name}:")
            print(f"  Train R²: {result['info']['train_score']:.4f}")
            if 'test_score' in result['info']:
                print(f"  Test R²: {result['info']['test_score']:.4f}")
