"""
Explainability Module
Implements SHAP (SHapley Additive exPlanations) for model interpretability
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import logging
from typing import Any, Dict, List, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import FIGURES_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _detect_model_type(model: Any) -> str:
    """
    Auto-detect the appropriate SHAP explainer type from the model class.
    Returns 'tree', 'linear', or 'kernel'.
    """
    class_name = type(model).__name__.lower()
    # Check for sklearn Pipeline wrapping a tree/linear model
    if hasattr(model, 'steps'):
        inner = model.steps[-1][1]
        return _detect_model_type(inner)
    tree_models = {'randomforest', 'gradientboosting', 'decisiontree',
                   'extratrees', 'xgb', 'lgbm', 'catboost'}
    linear_models = {'linearregression', 'ridge', 'lasso', 'elasticnet',
                     'logisticregression', 'sgd'}
    if any(t in class_name for t in tree_models):
        return 'tree'
    if any(t in class_name for t in linear_models):
        return 'linear'
    return 'kernel'


def calculate_shap_values(model: Any,
                         X: np.ndarray,
                         feature_names: List[str] = None,
                         model_type: str = 'auto') -> shap.Explanation:
    """
    Calculate SHAP values for model explanations
    
    Args:
        model: Trained model
        X: Feature data
        feature_names: List of feature names
        model_type: Type of model ('tree', 'linear', 'kernel', 'deep')
        
    Returns:
        SHAP explanation object
    """
    logger.info(f"Calculating SHAP values for {model_type} model...")
    
    # Auto-detect if not specified
    if model_type == 'auto':
        model_type = _detect_model_type(model)
        logger.info(f"Auto-detected SHAP explainer type: {model_type}")
    
    # Select appropriate explainer based on model type
    if model_type == 'tree':
        # For tree-based models (Random Forest, XGBoost, etc.)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    
    elif model_type == 'linear':
        # For linear models
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)
    
    elif model_type == 'kernel':
        # Model-agnostic kernel explainer (slower but works for any model)
        # Use a background sample for faster computation
        background = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X)
    
    elif model_type == 'deep':
        # For deep learning models
        background = X[:100]  # Use subset as background
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X)
    
    else:
        # Default to kernel explainer
        logger.warning(f"Unknown model type {model_type}, using kernel explainer")
        background = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X)
    
    logger.info(f"SHAP values calculated. Shape: {np.array(shap_values).shape}")
    
    return shap_values, explainer


def plot_shap_summary(shap_values: np.ndarray,
                     X: np.ndarray,
                     feature_names: List[str] = None,
                     title: str = "SHAP Summary Plot",
                     save_path: str = None):
    """
    Create SHAP summary plot showing feature importance
    
    Args:
        shap_values: SHAP values
        X: Feature data
        feature_names: List of feature names
        title: Plot title
        save_path: Path to save figure
    """
    logger.info("Creating SHAP summary plot...")
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(FIGURES_DIR / save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved SHAP summary plot to {save_path}")
    
    plt.show()
    plt.close()


def plot_shap_bar(shap_values: np.ndarray,
                 feature_names: List[str] = None,
                 title: str = "SHAP Feature Importance",
                 max_display: int = 15,
                 save_path: str = None):
    """
    Create SHAP bar plot showing mean absolute SHAP values
    
    Args:
        shap_values: SHAP values
        feature_names: List of feature names
        title: Plot title
        max_display: Maximum number of features to display
        save_path: Path to save figure
    """
    logger.info("Creating SHAP bar plot...")
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, features=None, feature_names=feature_names,
                     plot_type="bar", max_display=max_display, show=False)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(FIGURES_DIR / save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved SHAP bar plot to {save_path}")
    
    plt.show()
    plt.close()


def plot_shap_waterfall(shap_values: np.ndarray,
                       expected_value: float,
                       X: np.ndarray,
                       feature_names: List[str] = None,
                       sample_idx: int = 0,
                       title: str = "SHAP Waterfall Plot",
                       save_path: str = None):
    """
    Create SHAP waterfall plot for a single prediction
    
    Args:
        shap_values: SHAP values
        expected_value: Expected (base) value
        X: Feature data
        feature_names: List of feature names
        sample_idx: Index of sample to explain
        title: Plot title
        save_path: Path to save figure
    """
    logger.info(f"Creating SHAP waterfall plot for sample {sample_idx}...")
    
    # Create explanation object for the sample
    if isinstance(shap_values, shap.Explanation):
        explanation = shap_values[sample_idx]
    else:
        explanation = shap.Explanation(
            values=shap_values[sample_idx],
            base_values=expected_value,
            data=X[sample_idx] if X is not None else None,
            feature_names=feature_names
        )
    
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, show=False)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(FIGURES_DIR / save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved SHAP waterfall plot to {save_path}")
    
    plt.show()
    plt.close()


def plot_shap_dependence(shap_values: np.ndarray,
                        X: np.ndarray,
                        feature_name: str,
                        feature_names: List[str] = None,
                        interaction_feature: str = None,
                        title: str = None,
                        save_path: str = None):
    """
    Create SHAP dependence plot showing effect of a single feature
    
    Args:
        shap_values: SHAP values
        X: Feature data
        feature_name: Feature to plot
        feature_names: List of all feature names
        interaction_feature: Feature to use for coloring
        title: Plot title
        save_path: Path to save figure
    """
    if title is None:
        title = f"SHAP Dependence Plot - {feature_name}"
    
    logger.info(f"Creating SHAP dependence plot for {feature_name}...")
    
    # Get feature index
    if feature_names is not None and feature_name in feature_names:
        feature_idx = feature_names.index(feature_name)
    else:
        feature_idx = int(feature_name) if isinstance(feature_name, str) and feature_name.isdigit() else 0
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature_idx, shap_values, X,
        feature_names=feature_names,
        interaction_index=interaction_feature,
        show=False
    )
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(FIGURES_DIR / save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved SHAP dependence plot to {save_path}")
    
    plt.show()
    plt.close()


def plot_shap_force(shap_values: np.ndarray,
                   expected_value: float,
                   X: np.ndarray,
                   feature_names: List[str] = None,
                   sample_idx: int = 0,
                   save_path: str = None):
    """
    Create SHAP force plot for a single prediction
    
    Args:
        shap_values: SHAP values
        expected_value: Expected (base) value
        X: Feature data  
        feature_names: List of feature names
        sample_idx: Index of sample to explain
        save_path: Path to save figure
    """
    logger.info(f"Creating SHAP force plot for sample {sample_idx}...")
    
    # Create force plot
    force_plot = shap.force_plot(
        expected_value,
        shap_values[sample_idx],
        X[sample_idx] if X is not None else None,
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    
    if save_path:
        plt.savefig(FIGURES_DIR / save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved SHAP force plot to {save_path}")
    
    plt.show()
    plt.close()


def get_feature_importance_from_shap(shap_values: np.ndarray,
                                     feature_names: List[str] = None) -> pd.DataFrame:
    """
    Extract feature importance from SHAP values
    
    Args:
        shap_values: SHAP values
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(mean_abs_shap))]
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Importance': mean_abs_shap
    }).sort_values('SHAP_Importance', ascending=False)
    
    logger.info("\nTop 10 Features by SHAP Importance:")
    logger.info(f"\n{importance_df.head(10).to_string()}")
    
    return importance_df


def explain_prediction(model: Any,
                      X: np.ndarray,
                      sample_idx: int,
                      feature_names: List[str] = None,
                      model_type: str = 'tree') -> Dict:
    """
    Comprehensive explanation for a single prediction
    
    Args:
        model: Trained model
        X: Feature data
        sample_idx: Index of sample to explain
        feature_names: List of feature names
        model_type: Type of model
        
    Returns:
        Dictionary with explanation details
    """
    logger.info(f"Explaining prediction for sample {sample_idx}...")
    
    # Calculate SHAP values
    shap_values, explainer = calculate_shap_values(model, X[sample_idx:sample_idx+1], 
                                                   feature_names, model_type)
    
    # Get prediction
    prediction = model.predict(X[sample_idx:sample_idx+1])[0]
    
    # Get expected value
    if hasattr(explainer, 'expected_value'):
        expected_value = explainer.expected_value
    else:
        expected_value = model.predict(X).mean()
    
    explanation = {
        'sample_idx': sample_idx,
        'prediction': prediction,
        'expected_value': expected_value,
        'shap_values': shap_values[0] if isinstance(shap_values, np.ndarray) else shap_values,
        'feature_values': X[sample_idx]
    }
    
    # Print explanation
    logger.info(f"\nPrediction: {prediction:.2f}")
    logger.info(f"Expected value: {expected_value:.2f}")
    logger.info(f"Feature contributions:")
    
    if feature_names:
        contributions = list(zip(feature_names, explanation['shap_values'], explanation['feature_values']))
        contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
        
        for feat, shap_val, feat_val in contributions[:10]:
            logger.info(f"  {feat} = {feat_val:.2f} -> SHAP: {shap_val:+.2f}")
    
    return explanation


if __name__ == "__main__":
    # Test explainability module
    from data_loader import load_data
    from preprocessing import preprocess_data, prepare_train_test_split
    from ml_models import train_single_model
    
    print("Testing explainability module...")
    
    # Load and prepare data
    df = load_data("Delhi")
    df_clean = preprocess_data(df, remove_outliers=False)
    X_train, X_test, y_train, y_test = prepare_train_test_split(df_clean)
    
    # Get feature names
    feature_cols = [col for col in df_clean.columns if col not in ['AQI', 'Date', 'City']]
    
    # Train a Random Forest model
    print("Training Random Forest model...")
    model, _ = train_single_model("Random Forest", X_train, y_train, save_model=False)
    
    # Calculate SHAP values
    print("Calculating SHAP values...")
    shap_values, explainer = calculate_shap_values(model, X_test[:100], feature_cols, model_type='tree')
    
    # Plot SHAP summary
    plot_shap_summary(shap_values, X_test[:100], feature_cols, 
                     save_path='test_shap_summary.png')
    
    # Get feature importance
    importance_df = get_feature_importance_from_shap(shap_values, feature_cols)
    print(f"\nTop 5 important features:\n{importance_df.head()}")
    
    print("\nExplainability tests complete!")
