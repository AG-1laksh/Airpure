"""
Prediction script for Air Pollution Prediction System
Run: python predict.py
"""
import sys, joblib, numpy as np
sys.path.insert(0, 'D:/LAKSHYA/Desktop/Airpure')

from src.data_loader import load_data
from src.preprocessing import preprocess_data, prepare_train_test_split
from src.feature_engineering import engineer_features
from src.ml_models import load_saved_model
from src.lstm_model import load_lstm_model, predict_lstm
from src.evaluation import evaluate_model

# --- Prepare data ---
print("Loading and preparing data...")
df = load_data('Delhi')
df_clean = preprocess_data(df, remove_outliers=True)
df_feat = engineer_features(df_clean, include_lag=True, lag_days=7)

X_train, X_test, y_train, y_test = prepare_train_test_split(df_feat, target_col='AQI')
print(f"Test samples: {len(X_test)}")

# --- Load scaler saved during training ---
scaler = joblib.load('models/Delhi_ml_scaler.pkl')
X_test_scaled = scaler.transform(X_test)

# --- Predict with each available model ---
model_names = ['Random Forest', 'XGBoost', 'Gradient Boosting',
               'Linear Regression', 'Decision Tree', 'Support Vector Machine']

print("\n=== Predictions (first 10 samples) ===")
print(f"{'Model':<30} {'Predictions':}")
print("-" * 70)

for name in model_names:
    try:
        model = load_saved_model(name)
        y_pred = model.predict(X_test_scaled)
        metrics = evaluate_model(y_test, y_pred, model_name=name)
        print(f"\n{name}")
        print(f"  Predictions : {y_pred[:10].round(1)}")
        print(f"  RMSE={metrics['RMSE']:.2f}  MAE={metrics['MAE']:.2f}  R2={metrics['R2']:.3f}")
        print("  Actual vs Predicted (first 10):")
        for actual, pred in zip(y_test[:10], y_pred[:10]):
            print(f"    Actual: {actual:6.1f}  →  Predicted: {pred:6.1f}  (error: {abs(actual-pred):.1f})")
    except FileNotFoundError:
        print(f"{name:<30} [not trained yet — run: python main.py --city Delhi --mode train]")

# --- LSTM Prediction ---
print("\n=== LSTM Prediction ===")
try:
    scaler_X = joblib.load('models/Delhi_lstm_scaler_X.pkl')
    scaler_y = joblib.load('models/Delhi_lstm_scaler_y.pkl')

    X_test_sc = scaler_X.transform(X_test)

    lstm_model = load_lstm_model('Delhi_lstm_model')
    y_pred_scaled = predict_lstm(lstm_model, X_test_sc, time_steps=7)

    # Inverse-transform back to AQI units
    y_pred_aqi = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Align y_test — LSTM drops first `time_steps` rows
    y_test_aligned = y_test[7:]
    metrics = evaluate_model(y_test_aligned, y_pred_aqi, model_name='LSTM')

    print(f"  Predictions : {y_pred_aqi[:10].round(1)}")
    print(f"  RMSE={metrics['RMSE']:.2f}  MAE={metrics['MAE']:.2f}  R2={metrics['R2']:.3f}")

except FileNotFoundError:
    print("  [not trained yet — run: python main.py --city Delhi --mode lstm]")
