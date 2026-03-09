"""
LSTM Deep Learning Model Module
Implements LSTM neural network for time-series AQI prediction
"""
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path
import logging
from typing import Tuple, Dict

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import LSTM_CONFIG, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_lstm_sequences(X: np.ndarray, y: np.ndarray, time_steps: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM input
    
    Args:
        X: Feature array
        y: Target array
        time_steps: Number of time steps to look back
        
    Returns:
        X_sequences (samples, time_steps, features), y_sequences
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps), :])
        y_seq.append(y[i + time_steps])
    
    return np.array(X_seq), np.array(y_seq)


def build_lstm_model(input_shape: Tuple[int, int],
                     lstm_units: int = 64,
                     dropout_rate: float = 0.2,
                     num_lstm_layers: int = 2,
                     learning_rate: float = 0.001) -> Sequential:
    """
    Build LSTM model architecture
    
    Args:
        input_shape: (time_steps, n_features)
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
        num_lstm_layers: Number of LSTM layers
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled LSTM model
    """
    logger.info(f"Building LSTM model with input shape: {input_shape}")
    
    model = Sequential()
    
    # First LSTM layer
    if num_lstm_layers > 1:
        model.add(LSTM(lstm_units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        
        # Additional LSTM layers
        for i in range(num_lstm_layers - 2):
            model.add(LSTM(lstm_units, return_sequences=True))
            model.add(Dropout(dropout_rate))
        
        # Last LSTM layer
        model.add(LSTM(lstm_units, return_sequences=False))
        model.add(Dropout(dropout_rate))
    else:
        # Single LSTM layer
        model.add(LSTM(lstm_units, return_sequences=False, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
    
    # Dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    logger.info("LSTM model architecture:")
    model.summary(print_fn=logger.info)
    
    return model


def train_lstm(X_train: np.ndarray,
               y_train: np.ndarray,
               X_val: np.ndarray = None,
               y_val: np.ndarray = None,
               time_steps: int = None,
               lstm_config: dict = None,
               save_model: bool = True,
               model_name: str = "lstm_model") -> Tuple[Sequential, Dict]:
    """
    Train LSTM model
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        time_steps: Number of time steps
        lstm_config: LSTM configuration dictionary
        save_model: Whether to save the model
        model_name: Name for saving the model
        
    Returns:
        Trained model and training history
    """
    # Use default config if not provided
    if lstm_config is None:
        lstm_config = LSTM_CONFIG.copy()
    
    if time_steps is None:
        time_steps = lstm_config.get('time_steps', 7)
    
    logger.info(f"Preparing LSTM sequences with {time_steps} time steps...")
    
    # Create sequences
    X_train_seq, y_train_seq = create_lstm_sequences(X_train, y_train, time_steps)
    
    logger.info(f"LSTM training sequences shape: {X_train_seq.shape}")
    
    # Validation sequences
    validation_data = None
    if X_val is not None and y_val is not None:
        X_val_seq, y_val_seq = create_lstm_sequences(X_val, y_val, time_steps)
        validation_data = (X_val_seq, y_val_seq)
        logger.info(f"LSTM validation sequences shape: {X_val_seq.shape}")
    
    # Build model
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    model = build_lstm_model(
        input_shape=input_shape,
        lstm_units=lstm_config.get('lstm_units', 64),
        dropout_rate=lstm_config.get('dropout_rate', 0.2),
        num_lstm_layers=lstm_config.get('num_lstm_layers', 2),
        learning_rate=lstm_config.get('learning_rate', 0.001)
    )
    
    # Callbacks
    callbacks = []
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss' if validation_data else 'loss',
        patience=10,
        restore_best_weights=True
    )
    callbacks.append(early_stop)
    
    # Model checkpoint
    if save_model:
        checkpoint_path = MODELS_DIR / f"{model_name}_best.keras"
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss' if validation_data else 'loss',
            save_best_only=True
        )
        callbacks.append(checkpoint)
    
    # Train model
    logger.info("Training LSTM model...")
    
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=lstm_config.get('epochs', 100),
        batch_size=lstm_config.get('batch_size', 32),
        validation_data=validation_data,
        validation_split=lstm_config.get('validation_split', 0.2) if validation_data is None else 0,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    if save_model:
        final_model_path = MODELS_DIR / f"{model_name}.keras"
        model.save(final_model_path)
        logger.info(f"LSTM model saved to {final_model_path}")
    
    # Training info
    info = {
        "model_name": model_name,
        "time_steps": time_steps,
        "n_features": X_train_seq.shape[2],
        "n_samples": X_train_seq.shape[0],
        "history": history.history
    }
    
    return model, info


def predict_lstm(model: Sequential,
                X: np.ndarray,
                time_steps: int = 7) -> np.ndarray:
    """
    Make predictions with LSTM model
    
    Args:
        model: Trained LSTM model
        X: Feature array
        time_steps: Number of time steps
        
    Returns:
        Predictions
    """
    # Create sequences
    X_seq = []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps), :])
    
    X_seq = np.array(X_seq)
    
    # Predict
    predictions = model.predict(X_seq)
    
    return predictions.flatten()


def load_lstm_model(model_name: str = "lstm_model") -> Sequential:
    """
    Load saved LSTM model
    
    Args:
        model_name: Name of the model
        
    Returns:
        Loaded LSTM model
    """
    model_path = MODELS_DIR / f"{model_name}.keras"
    # Fall back to legacy .h5 if .keras not found
    if not model_path.exists():
        model_path = MODELS_DIR / f"{model_name}.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = keras.models.load_model(model_path)
    logger.info(f"Loaded LSTM model from {model_path}")
    
    return model


def predict_future_aqi(model: Sequential,
                      last_sequence: np.ndarray,
                      n_days: int = 7,
                      scaler: MinMaxScaler = None) -> np.ndarray:
    """
    Predict future AQI values
    
    Args:
        model: Trained LSTM model
        last_sequence: Last sequence of features (time_steps, n_features)
        n_days: Number of days to predict ahead
        scaler: Scaler used for data normalization (optional)
        
    Returns:
        Array of future AQI predictions
    """
    logger.info(f"Predicting {n_days} days ahead...")
    
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_days):
        # Reshape for prediction
        input_seq = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
        
        # Predict next value
        next_pred = model.predict(input_seq, verbose=0)[0, 0]
        predictions.append(next_pred)
        
        # Update sequence (sliding window)
        # Note: This is simplified - in practice, you'd need to update all features
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, -1] = next_pred  # Update AQI in last position
    
    predictions = np.array(predictions)
    
    # Inverse transform if scaler provided
    if scaler is not None:
        # Assuming AQI is the last feature
        predictions = scaler.inverse_transform(
            np.concatenate([np.zeros((len(predictions), scaler.scale_.shape[0]-1)), 
                          predictions.reshape(-1, 1)], axis=1)
        )[:, -1]
    
    return predictions


def evaluate_lstm(model: Sequential,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 time_steps: int = 7) -> Dict:
    """
    Evaluate LSTM model performance
    
    Args:
        model: Trained LSTM model
        X_test: Test features
        y_test: Test target
        time_steps: Number of time steps
        
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Create sequences
    X_test_seq, y_test_seq = create_lstm_sequences(X_test, y_test, time_steps)
    
    # Predict
    y_pred = model.predict(X_test_seq).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_seq, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_seq, y_pred)
    r2 = r2_score(y_test_seq, y_pred)
    
    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }
    
    logger.info("LSTM Evaluation Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return metrics


if __name__ == "__main__":
    # Test LSTM model
    from data_loader import load_data
    from preprocessing import preprocess_data, scale_features
    
    print("Testing LSTM model module...")
    
    # Load and prepare data
    df = load_data("Delhi")
    df_clean = preprocess_data(df, remove_outliers=False)
    
    # Scale features
    df_scaled, scaler = scale_features(df_clean, scaler_type="minmax")
    
    # Prepare data
    feature_cols = [col for col in df_scaled.columns if col not in ['Date', 'City', 'AQI']]
    X = df_scaled[feature_cols].values
    y = df_scaled['AQI'].values
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training data: {X_train.shape}")
    
    # Train LSTM
    model, info = train_lstm(X_train, y_train, X_test, y_test, time_steps=7)
    
    print("\nLSTM model trained successfully!")
    print(f"Training samples: {info['n_samples']}")
