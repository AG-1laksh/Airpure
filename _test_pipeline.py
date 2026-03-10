"""Full pipeline diagnostic test"""
import warnings
warnings.filterwarnings('ignore')
import traceback

steps = {}

# ── 1. Data Loading ──────────────────────────────────────────────────────────
try:
    from src.data_loader import load_data
    df = load_data('Delhi')
    steps['1_load'] = f"OK  shape={df.shape}  cols={df.columns.tolist()}"
except Exception as e:
    steps['1_load'] = f"FAIL  {e}"
    traceback.print_exc()

# ── 2. Preprocessing ─────────────────────────────────────────────────────────
try:
    from src.preprocessing import preprocess_data
    df_clean = preprocess_data(df, remove_outliers=True)
    steps['2_preprocess'] = f"OK  shape={df_clean.shape}"
except Exception as e:
    steps['2_preprocess'] = f"FAIL  {e}"
    traceback.print_exc()

# ── 3. Feature Engineering ───────────────────────────────────────────────────
try:
    from src.feature_engineering import engineer_features
    df_feat = engineer_features(df_clean, include_lag=True, lag_days=7)
    steps['3_features'] = f"OK  shape={df_feat.shape}  cols={df_feat.columns.tolist()}"
except Exception as e:
    steps['3_features'] = f"FAIL  {e}"
    traceback.print_exc()

# ── 4. Train/Test Split ──────────────────────────────────────────────────────
try:
    from src.preprocessing import prepare_train_test_split
    X_train, X_test, y_train, y_test = prepare_train_test_split(df_feat, target_col='AQI')
    steps['4_split'] = f"OK  X_train={X_train.shape}  X_test={X_test.shape}"
except Exception as e:
    steps['4_split'] = f"FAIL  {e}"
    traceback.print_exc()

# ── 5. ML Models ─────────────────────────────────────────────────────────────
try:
    from src.ml_models import train_ml_models
    results = train_ml_models(X_train, y_train, X_test, y_test, save_models=True)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score as _r2
    model_lines = []
    for name, r in results.items():
        if r.get('model') is not None:
            y_pred = r['model'].predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2v = _r2(y_test, y_pred)
            model_lines.append(f"  {name:25s} RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2v:.3f}")
        else:
            model_lines.append(f"  {name:25s} FAILED")
    steps['5_ml_models'] = "OK\n" + "\n".join(model_lines)
except Exception as e:
    steps['5_ml_models'] = f"FAIL  {e}"
    traceback.print_exc()

# ── 6. LSTM ──────────────────────────────────────────────────────────────────
try:
    from src.lstm_model import train_lstm
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    feature_cols = [c for c in df_feat.columns if c not in ['AQI', 'Date', 'City']]
    X = df_feat[feature_cols].values
    y = df_feat['AQI'].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_sc = scaler_X.fit_transform(X)
    y_sc = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    split = int(len(X_sc) * 0.8)
    lstm_model, lstm_info = train_lstm(
        X_sc[:split], y_sc[:split],
        X_sc[split:], y_sc[split:],
        time_steps=30,
        lstm_config={'lstm_units': 64, 'dropout_rate': 0.2, 'num_lstm_layers': 2,
                     'epochs': 5, 'batch_size': 32},
        save_model=True, model_name='Delhi_lstm_model'
    )
    history = lstm_info.get('history', {})
    val_loss_list = history.get('val_loss', history.get('loss', ['N/A']))
    final_val_loss = val_loss_list[-1] if val_loss_list else 'N/A'
    steps['6_lstm'] = f"OK  final_val_loss={final_val_loss}"
except Exception as e:
    steps['6_lstm'] = f"FAIL  {e}"
    traceback.print_exc()

# ── 7. Ensemble ──────────────────────────────────────────────────────────────
try:
    from src.ensemble import create_ensemble
    from src.ml_models import train_ml_models
    preds = {}
    for name, r in results.items():
        if r.get('model'):
            preds[name] = r['model'].predict(X_test)
    ens_pred = create_ensemble(preds, method='weighted_average')
    steps['7_ensemble'] = f"OK  ensemble_preds_shape={ens_pred.shape}"
except Exception as e:
    steps['7_ensemble'] = f"FAIL  {e}"
    traceback.print_exc()

# ── 8. Evaluation ────────────────────────────────────────────────────────────
try:
    from src.evaluation import evaluate_model, compare_models
    comparison_data = {name: {'y_true': y_test, 'y_pred': r['model'].predict(X_test)}
                       for name, r in results.items() if r.get('model')}
    cmp = compare_models(comparison_data)
    steps['8_evaluation'] = f"OK\n{cmp.to_string(index=False)}"
except Exception as e:
    steps['8_evaluation'] = f"FAIL  {e}"
    traceback.print_exc()

# ── 9. Visualization ─────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    from src.visualization import plot_time_series, plot_aqi_distribution, plot_model_comparison
    plot_time_series(df_clean, ['AQI'], save_path='_test_timeseries.png')
    plot_aqi_distribution(df_clean['AQI'].values, save_path='_test_aqi_dist.png')
    steps['9_visualization'] = "OK  plots saved"
except Exception as e:
    steps['9_visualization'] = f"FAIL  {e}"
    traceback.print_exc()

# ── 10. SHAP Explainability ──────────────────────────────────────────────────
try:
    from src.explainability import calculate_shap_values, plot_shap_summary
    best_name = [n for n,r in results.items() if r.get('model')][0]
    best_model = results[best_name]['model']
    feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else [f'f{i}' for i in range(X_train.shape[1])]
    import numpy as np
    shap_vals, explainer = calculate_shap_values(best_model, X_test[:100], feature_names, model_type='auto')
    steps['10_shap'] = f"OK  shap_shape={shap_vals.shape}"
except Exception as e:
    steps['10_shap'] = f"FAIL  {e}"
    traceback.print_exc()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PIPELINE DIAGNOSTIC REPORT")
print("="*60)
for step, status in steps.items():
    icon = "[OK]" if status.startswith("OK") else "[FAIL]"
    print(f"\n{icon} {step}")
    print(f"    {status}")
print("\n" + "="*60)

import os
for f in ['_test_timeseries.png', '_test_aqi_dist.png']:
    if os.path.exists(f):
        os.remove(f)
