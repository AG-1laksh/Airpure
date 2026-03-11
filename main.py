"""
Main execution script for Air Pollution Prediction System
"""
import argparse
import logging
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import modules
from src.data_loader import load_data, create_sample_data
from src.preprocessing import preprocess_data, prepare_train_test_split, scale_features
from src.feature_engineering import engineer_features
from src.ml_models import train_ml_models
from src.lstm_model import train_lstm, create_lstm_sequences
from src.ensemble import create_ensemble, optimize_ensemble_weights
from src.evaluation import compare_models, evaluate_model
from src.explainability import calculate_shap_values, plot_shap_summary, get_feature_importance_from_shap
from src.visualization import (
    plot_time_series, plot_correlation_matrix, plot_predictions,
    plot_model_comparison, plot_aqi_distribution,
    plot_seasonal_analysis, plot_yearly_trend
)
from config import CITIES, TARGET_VARIABLE, LSTM_CONFIG, PROCESSED_DATA_DIR, get_city_processed_dir, MODELS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('airpure.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def run_data_collection(city: str):
    """Run data collection phase"""
    logger.info(f"=== DATA COLLECTION: {city} ===")
    
    # Load or create data
    df = load_data(city)
    logger.info(f"Loaded {len(df)} records")
    
    return df


def run_preprocessing(df, city: str):
    """Run preprocessing phase"""
    logger.info("=== DATA PREPROCESSING ===")
    
    # Preprocess data
    df_clean = preprocess_data(df, remove_outliers=True)
    
    # Create visualizations
    logger.info("Creating exploratory visualizations...")
    plot_time_series(df_clean, [col for col in ['PM2.5', 'AQI'] if col in df_clean.columns],
                     save_path=f'{city}_timeseries.png')
    plot_correlation_matrix(df_clean, save_path=f'{city}_correlation.png')
    plot_aqi_distribution(df_clean['AQI'].values, save_path=f'{city}_aqi_dist.png')
    plot_seasonal_analysis(df_clean, save_path=f'{city}_seasonal.png')
    plot_yearly_trend(df_clean, save_path=f'{city}_yearly_trend.png')

    return df_clean


def run_feature_engineering(df, city: str):
    """Run feature engineering phase"""
    logger.info("=== FEATURE ENGINEERING ===")
    
    # Engineer features
    df_engineered = engineer_features(df, include_lag=True, lag_days=7)
    logger.info(f"Engineered features. Shape: {df_engineered.shape}")
    
    # Save so subsequent modes (train, lstm …) can load without re-running
    # prefer city-specific dir, fall back to legacy
    city_proc_dir = get_city_processed_dir(city)
    city_proc_dir.mkdir(parents=True, exist_ok=True)
    city_path = city_proc_dir / f"{city}_processed.csv"
    df_engineered.to_csv(city_path, index=False)
    # also write to legacy path so --mode train always finds it
    legacy_path = PROCESSED_DATA_DIR / f"{city}_processed.csv"
    df_engineered.to_csv(legacy_path, index=False)
    logger.info(f"Saved processed data to {legacy_path}")
    
    return df_engineered


def run_ml_training(df, city: str):
    """Run ML model training"""
    logger.info("=== TRAINING ML MODELS ===")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_train_test_split(df, target_col=TARGET_VARIABLE)
    
    # Scale features (required for SVR and Linear Regression correctness)
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    scaler_path = MODELS_DIR / f"{city}_ml_scaler.pkl"
    joblib.dump(scaler_X, scaler_path)
    logger.info(f"Saved ML feature scaler to {scaler_path}")
    
    # Train traditional ML models
    ml_results = train_ml_models(X_train, y_train, X_test, y_test, save_models=True, city=city)
    
    # Collect predictions for comparison
    comparison_data = {}
    for model_name, result in ml_results.items():
        if result['model'] is not None:
            y_pred = result['model'].predict(X_test)
            comparison_data[model_name] = {
                'y_true': y_test,
                'y_pred': y_pred
            }
    
    # Compare models
    comparison_df = compare_models(comparison_data)
    
    # Plot comparison
    plot_model_comparison(comparison_df, metric='RMSE', save_path=f'{city}_model_comparison.png')
    
    return ml_results, comparison_df, X_train, X_test, y_train, y_test


def run_lstm_training(df, city: str):
    """Run LSTM training"""
    logger.info("=== TRAINING LSTM MODEL ===")
    
    # Prepare data
    feature_cols = [col for col in df.columns if col not in [TARGET_VARIABLE, 'Date', 'City']]
    X = df[feature_cols].values
    y = df[TARGET_VARIABLE].values
    
    # Scale data
    from sklearn.preprocessing import MinMaxScaler
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Split data
    split_idx = int(len(X_scaled) * 0.8)
    X_train = X_scaled[:split_idx]
    X_test = X_scaled[split_idx:]
    y_train = y_scaled[:split_idx]
    y_test = y_scaled[split_idx:]
    
    # Train LSTM
    lstm_model, lstm_info = train_lstm(
        X_train, y_train, X_test, y_test,
        time_steps=LSTM_CONFIG['time_steps'],
        lstm_config=LSTM_CONFIG,
        save_model=True,
        model_name=f'{city}_lstm_model'
    )
    
    # Save scalers alongside the model for correct inference-time inverse-transform
    joblib.dump(scaler_X, MODELS_DIR / f"{city}_lstm_scaler_X.pkl")
    joblib.dump(scaler_y, MODELS_DIR / f"{city}_lstm_scaler_y.pkl")
    logger.info(f"Saved LSTM scalers to {MODELS_DIR}")
    
    return lstm_model, lstm_info, scaler_X, scaler_y


def run_explainability(model, X_test, feature_names, city: str):
    """Run explainability analysis"""
    logger.info("=== EXPLAINABILITY ANALYSIS ===")
    
    # Calculate SHAP values
    shap_values, explainer = calculate_shap_values(model, X_test[:200], feature_names, model_type='auto')
    
    # Plot SHAP summary
    plot_shap_summary(shap_values, X_test[:200], feature_names, 
                     save_path=f'{city}_shap_summary.png')
    
    # Get feature importance
    importance_df = get_feature_importance_from_shap(shap_values, feature_names)
    
    return importance_df


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Air Pollution Prediction System')
    parser.add_argument('--city', type=str, default='Delhi', choices=CITIES,
                       help='City to analyze')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'preprocess', 'train', 'lstm', 'evaluate', 'explain'],
                       help='Execution mode')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("AIR POLLUTION PREDICTION SYSTEM")
    logger.info("="*60)
    logger.info(f"City: {args.city}")
    logger.info(f"Mode: {args.mode}")
    logger.info("="*60)
    
    try:
        # Data collection
        df = run_data_collection(args.city)
        
        if args.mode in ['all', 'preprocess']:
            # Preprocessing
            df_clean = run_preprocessing(df, args.city)
            
            # Feature engineering
            df_final = run_feature_engineering(df_clean, args.city)
        else:
            # Load preprocessed data — fall back to running it if the file doesn't exist
            legacy_path = PROCESSED_DATA_DIR / f"{args.city}_processed.csv"
            if legacy_path.exists():
                df_final = pd.read_csv(legacy_path)
                logger.info(f"Loaded preprocessed data from {legacy_path}")
            else:
                logger.warning("Preprocessed CSV not found — running preprocessing + feature engineering now.")
                df_clean = run_preprocessing(df, args.city)
                df_final = run_feature_engineering(df_clean, args.city)
        
        if args.mode in ['all', 'train']:
            # Train ML models
            ml_results, comparison_df, X_train, X_test, y_train, y_test = run_ml_training(df_final, args.city)
            
            # Get best model
            best_model_name = comparison_df.iloc[0]['Model']
            best_model = ml_results[best_model_name]['model']
            
            # Feature names
            feature_cols = [col for col in df_final.columns if col not in [TARGET_VARIABLE, 'Date', 'City']]
            
            # Evaluate best model
            y_pred = best_model.predict(X_test)
            plot_predictions(y_test, y_pred, title=f"Best Model: {best_model_name}",
                           save_path=f'{args.city}_best_predictions.png')
        
        if args.mode in ['all', 'lstm']:
            # Train LSTM
            lstm_model, lstm_info, lstm_scaler_X, lstm_scaler_y = run_lstm_training(df_final, args.city)
        
        if args.mode in ['all', 'explain']:
            # Explainability analysis
            feature_cols = [col for col in df_final.columns if col not in [TARGET_VARIABLE, 'Date', 'City']]
            X_train, X_test, y_train, y_test = prepare_train_test_split(df_final)
            
            # Load best model (Random Forest for tree-based SHAP)
            from src.ml_models import load_saved_model
            try:
                model = load_saved_model("Random Forest", city=args.city)
                importance_df = run_explainability(model, X_test, feature_cols, args.city)
            except:
                logger.warning("Could not load saved model for explainability")
        
        logger.info("="*60)
        logger.info("EXECUTION COMPLETE!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
