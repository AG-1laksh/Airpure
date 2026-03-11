"""
Prediction script for Air Pollution Prediction System
Run: python predict.py --city Mumbai
"""
import os
import sys
import argparse
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.ml_models import load_saved_model, get_feature_importance
from src.lstm_model import load_lstm_model, predict_lstm, create_lstm_sequences
from src.evaluation import evaluate_model
from src.visualization import plot_predictions, plot_model_comparison, plot_residuals, plot_feature_importance
from src.explainability import calculate_shap_values, plot_shap_summary, get_feature_importance_from_shap
from config import MODELS_DIR, TABLES_DIR, PREDICTIONS_DIR, FIGURES_DIR, RANDOM_STATE, LSTM_CONFIG


def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def time_series_split(df: pd.DataFrame, test_size: float = 0.2, date_col: str = "Date"):
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    return train_df, test_df


def save_predictions_csv(city: str, model_name: str, dates, y_true, y_pred):
    out = pd.DataFrame({
        "Date": dates,
        "Actual_AQI": y_true,
        "Predicted_AQI": y_pred,
        "Model": model_name
    })
    filename = f"{city}_{model_name.lower().replace(' ', '_')}_predictions.csv"
    out_path = PREDICTIONS_DIR / filename
    out.to_csv(out_path, index=False)
    return out_path


parser = argparse.ArgumentParser(description="Predict AQI for a city")
parser.add_argument("--city", type=str, default="Delhi", help="City name (e.g., Mumbai)")
parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction (chronological)")
args = parser.parse_args()

city = args.city
set_seeds(RANDOM_STATE)

# --- Prepare data ---
print("Loading and preparing data...")
df = load_data(city)
df_clean = preprocess_data(df, remove_outliers=True)
df_feat = engineer_features(df_clean, include_lag=True, include_rolling=True, lag_days=7)

if "Date" not in df_feat.columns:
    raise ValueError("Date column is required for time-series split.")

train_df, test_df = time_series_split(df_feat, test_size=args.test_size, date_col="Date")
feature_cols = [col for col in df_feat.columns if col not in ["AQI", "Date", "City"]]

X_train = train_df[feature_cols].values
y_train = train_df["AQI"].values
X_test = test_df[feature_cols].values
y_test = test_df["AQI"].values
test_dates = test_df["Date"].values

print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# --- Predict with each available model ---
model_names = [
    "Random Forest",
    "XGBoost",
    "Gradient Boosting",
    "Linear Regression",
    "Decision Tree",
    "Support Vector Machine"
]

scaling_models = {"Linear Regression", "Support Vector Machine"}
comparison_rows = []
predictions_by_model = {}

print("\n=== Predictions (first 10 samples) ===")
print(f"{'Model':<30} {'Predictions':}")
print("-" * 70)

for name in model_names:
    try:
        model = load_saved_model(name, city=city)

        if name in scaling_models and not (hasattr(model, "named_steps") and "scaler" in model.named_steps):
            scaler = StandardScaler()
            X_train_used = scaler.fit_transform(X_train)
            X_test_used = scaler.transform(X_test)
        else:
            X_train_used = X_train
            X_test_used = X_test

        y_pred = model.predict(X_test_used)
        metrics = evaluate_model(y_test, y_pred, model_name=name)
        predictions_by_model[name] = y_pred

        comparison_rows.append({
            "Model": name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "R2": metrics["R2"]
        })

        save_predictions_csv(city, name, test_dates, y_test, y_pred)

        print(f"\n{name}")
        print(f"  Predictions : {np.asarray(y_pred[:10]).round(1)}")
        print(f"  RMSE={metrics['RMSE']:.2f}  MAE={metrics['MAE']:.2f}  R2={metrics['R2']:.3f}")
        print("  Actual vs Predicted (first 10):")
        for actual, pred in zip(y_test[:10], y_pred[:10]):
            print(f"    Actual: {actual:6.1f}  →  Predicted: {pred:6.1f}  (error: {abs(actual-pred):.1f})")
    except FileNotFoundError:
        print(f"{name:<30} [not trained yet — run: python main.py --city {city} --mode train]")

comparison_df = pd.DataFrame(comparison_rows).sort_values("RMSE").reset_index(drop=True)
comparison_path = TABLES_DIR / f"{city}_model_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)

print("\n=== Model Comparison (sorted by RMSE) ===")
print(comparison_df.to_string(index=False))
print(f"\nSaved comparison table → {comparison_path}")

if not comparison_df.empty:
    best_model_name = comparison_df.iloc[0]["Model"]
    best_pred = predictions_by_model.get(best_model_name)
    print(f"\nTop model: {best_model_name}")

    if best_pred is not None:
        plot_predictions(y_test, best_pred,
                         title=f"{city} | {best_model_name} | Actual vs Predicted",
                         save_path=f"{city}_best_predictions.png")
        plot_residuals(y_test, best_pred,
                       title=f"{city} | {best_model_name} | Residuals",
                       save_path=f"{city}_best_residuals.png")

    plot_model_comparison(comparison_df, save_path=f"{city}_model_comparison.png")

# --- Feature importance (tree models) ---
tree_models = ["Random Forest", "XGBoost", "Gradient Boosting", "Decision Tree"]
tree_model_name = next((m for m in comparison_df["Model"] if m in tree_models), None) if not comparison_df.empty else None

if tree_model_name:
    try:
        tree_model = load_saved_model(tree_model_name, city=city)
        importance_df = get_feature_importance(tree_model, feature_cols)
        if not importance_df.empty:
            plot_feature_importance(importance_df, top_n=15,
                                    title=f"{city} | {tree_model_name} Feature Importance",
                                    save_path=f"{city}_{tree_model_name.lower().replace(' ', '_')}_feature_importance.png")
    except Exception:
        pass

# --- SHAP Explainability ---
try:
    import shap
    if tree_model_name:
        tree_model = load_saved_model(tree_model_name, city=city)
        X_sample = pd.DataFrame(X_test[:200], columns=feature_cols)
        shap_values, explainer = calculate_shap_values(tree_model, X_sample, feature_cols, model_type='auto')
        plot_shap_summary(shap_values, X_sample, feature_cols,
                          save_path=f"{city}_{tree_model_name.lower().replace(' ', '_')}_shap_summary.png")

        shap_importance = get_feature_importance_from_shap(shap_values, feature_cols)
        if not shap_importance.empty:
            plot_feature_importance(shap_importance, top_n=15,
                                    title=f"{city} | SHAP Feature Importance",
                                    save_path=f"{city}_{tree_model_name.lower().replace(' ', '_')}_shap_importance.png")

        top_features = shap_importance.head(3)["Feature"].tolist() if not shap_importance.empty else feature_cols[:3]
        for feat in top_features:
            plt.figure()
            shap.dependence_plot(feat, shap_values, X_sample, feature_names=feature_cols, show=False)
            dep_path = FIGURES_DIR / f"{city}_{tree_model_name.lower().replace(' ', '_')}_shap_dependence_{feat}.png"
            plt.savefig(dep_path, bbox_inches='tight')
            plt.close()
except Exception:
    print("SHAP not available or failed to run. Install shap for explainability plots.")

# --- LSTM Prediction ---
print("\n=== LSTM Prediction ===")
try:
    time_steps = LSTM_CONFIG.get("time_steps", 7)
    scaler_X = joblib.load(MODELS_DIR / f"{city}_lstm_scaler_X.pkl")
    scaler_y = joblib.load(MODELS_DIR / f"{city}_lstm_scaler_y.pkl")

    X_test_sc = scaler_X.transform(X_test)
    X_test_seq, y_test_seq = create_lstm_sequences(X_test_sc, y_test, time_steps)

    lstm_model = load_lstm_model(f"{city}_lstm_model")
    y_pred_scaled = predict_lstm(lstm_model, X_test_sc, time_steps=time_steps)
    y_pred_aqi = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    if len(y_test_seq) > 0:
        lstm_dates = test_dates[time_steps:]
        metrics = evaluate_model(y_test_seq, y_pred_aqi, model_name='LSTM')

        save_predictions_csv(city, "LSTM", lstm_dates, y_test_seq, y_pred_aqi)

        print(f"  Predictions : {y_pred_aqi[:10].round(1)}")
        print(f"  RMSE={metrics['RMSE']:.2f}  MAE={metrics['MAE']:.2f}  R2={metrics['R2']:.3f}")
    else:
        print("  Not enough samples for LSTM sequence evaluation.")

    # --- Future AQI Forecasting ---
    X_all = df_feat[feature_cols].values
    X_all_scaled = scaler_X.transform(X_all)
    last_sequence = X_all_scaled[-time_steps:]

    forecast_days = [1, 3, 7]
    max_days = max(forecast_days)
    forecasts_scaled = []
    seq = last_sequence.copy()
    last_row = seq[-1].copy()

    for _ in range(max_days):
        next_pred_scaled = lstm_model.predict(seq.reshape(1, time_steps, -1), verbose=0)[0, 0]
        forecasts_scaled.append(next_pred_scaled)
        seq = np.roll(seq, -1, axis=0)
        seq[-1] = last_row  # keep last observed features (naive persistence)

    forecasts = scaler_y.inverse_transform(np.array(forecasts_scaled).reshape(-1, 1)).flatten()
    forecast_rows = []
    for d in forecast_days:
        forecast_rows.append({"Horizon_Days": d, "Predicted_AQI": forecasts[d - 1]})

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_path = PREDICTIONS_DIR / f"{city}_lstm_forecast.csv"
    forecast_df.to_csv(forecast_path, index=False)
    print("\n=== Future AQI Forecasts (LSTM) ===")
    print(forecast_df.to_string(index=False))
    print(f"Saved forecasts → {forecast_path}")

except FileNotFoundError:
    print(f"  [not trained yet — run: python main.py --city {city} --mode lstm]")
