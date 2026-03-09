# Getting Started Guide

## Air Pollution Prediction System - Quick Start

This guide will help you set up and run the air pollution prediction system.

---

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git
- 4GB+ RAM recommended
- Windows/Linux/MacOS

---

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/AG-1laksh/Airpure.git
cd Airpure
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter installation errors with TensorFlow, try:
```bash
pip install tensorflow-cpu  # For CPU-only version
```

### 4. Verify Installation

```bash
python -c "import tensorflow, sklearn, pandas, numpy; print('All packages installed successfully!')"
```

---

## Quick Start

### Option 1: Run Complete Pipeline

Execute the main script to run the entire pipeline:

```bash
python main.py --city Delhi --mode all
```

This will:
- Load/create sample data
- Preprocess and engineer features
- Train all ML models
- Train LSTM model
- Generate visualizations
- Save results and models

**Options:**
- `--city`: Select city (Delhi, Mumbai, Chennai, Bangalore)
- `--mode`: Execution mode
  - `all`: Run complete pipeline
  - `preprocess`: Only preprocessing
  - `train`: Only ML training
  - `lstm`: Only LSTM training
  - `evaluate`: Only evaluation
  - `explain`: Only explainability analysis

### Option 2: Interactive Dashboard

Launch the Streamlit dashboard for interactive exploration:

```bash
streamlit run dashboard/app.py
```

Then open your browser to: `http://localhost:8501`

### Option 3: Jupyter Notebooks

Start Jupyter and explore the notebooks:

```bash
jupyter notebook
```

Navigate to `notebooks/` directory and open any notebook.

---

## Project Structure Overview

```
Airpure/
├── data/                    # Dataset storage
│   ├── raw/                 # Raw datasets
│   ├── processed/           # Cleaned datasets
│   └── README.md           # Data documentation
│
├── src/                     # Source code modules
│   ├── data_loader.py      # Data loading functions
│   ├── preprocessing.py    # Data preprocessing
│   ├── feature_engineering.py # Feature creation
│   ├── ml_models.py        # Traditional ML models
│   ├── lstm_model.py       # LSTM deep learning
│   ├── ensemble.py         # Ensemble methods
│   ├── evaluation.py       # Model evaluation
│   ├── explainability.py   # SHAP analysis
│   └── visualization.py    # Plotting functions
│
├── models/                  # Saved trained models
│   └── *.pkl, *.h5         # Model files
│
├── results/                 # Output results
│   ├── figures/            # Generated plots
│   ├── tables/             # Result tables
│   └── predictions/        # Prediction outputs
│
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_results_analysis.ipynb
│
├── dashboard/              # Streamlit dashboard
│   └── app.py
│
├── research_paper/         # Research documentation
│   └── paper_outline.md
│
├── config.py               # Configuration settings
├── main.py                 # Main execution script
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

---

## Step-by-Step Workflow

### Step 1: Data Preparation

The system automatically creates sample data if no dataset is found. To use your own data:

1. Place CSV file in `data/raw/` directory
2. Name it: `{CityName}_air_quality.csv`
3. Ensure it has these columns:
   - Date, City, PM2.5, PM10, NO2, SO2, CO, O3
   - Temperature, Humidity, Wind_Speed, Rainfall, AQI

### Step 2: Data Preprocessing

```python
from src.data_loader import load_data
from src.preprocessing import preprocess_data

# Load data
df = load_data("Delhi")

# Preprocess
df_clean = preprocess_data(df, remove_outliers=True)
```

### Step 3: Feature Engineering

```python
from src.feature_engineering import engineer_features

# Create lag features and rolling statistics
df_features = engineer_features(df_clean, lag_days=7)
```

### Step 4: Train ML Models

```python
from src.ml_models import train_ml_models
from src.preprocessing import prepare_train_test_split

# Prepare data
X_train, X_test, y_train, y_test = prepare_train_test_split(df_features)

# Train models
results = train_ml_models(X_train, y_train, X_test, y_test)
```

### Step 5: Train LSTM Model

```python
from src.lstm_model import train_lstm

# Train LSTM
lstm_model, info = train_lstm(X_train, y_train, X_test, y_test)
```

### Step 6: Evaluate and Compare

```python
from src.evaluation import compare_models, evaluate_model

# Compare all models
comparison_df = compare_models(results_dict)
print(comparison_df)
```

### Step 7: Explainability Analysis

```python
from src.explainability import calculate_shap_values, plot_shap_summary

# Calculate SHAP values
shap_values, explainer = calculate_shap_values(
    model, X_test, feature_names, model_type='tree'
)

# Visualize
plot_shap_summary(shap_values, X_test, feature_names)
```

---

## Example Usage Scripts

### Example 1: Quick Prediction

```python
import numpy as np
from src.ml_models import load_saved_model

# Load trained model
model = load_saved_model("Random Forest")

# Create sample input
input_data = np.array([[
    100,  # PM2.5
    150,  # PM10
    40,   # NO2
    15,   # SO2
    1.2,  # CO
    50,   # O3
    25,   # Temperature
    60,   # Humidity
    10,   # Wind Speed
    0     # Rainfall
]])

# Predict AQI
predicted_aqi = model.predict(input_data)
print(f"Predicted AQI: {predicted_aqi[0]:.2f}")
```

### Example 2: Batch Predictions

```python
from src.data_loader import load_data
from src.preprocessing import preprocess_data

# Load data
df = load_data("Delhi")
df_clean = preprocess_data(df)

# Prepare features
feature_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
                'Temperature', 'Humidity', 'Wind_Speed', 'Rainfall']
X = df_clean[feature_cols].values

# Batch predict
predictions = model.predict(X)

# Add to dataframe
df_clean['Predicted_AQI'] = predictions
```

---

## Configuration

Edit `config.py` to customize:

```python
# Target cities
CITIES = ["Delhi", "Mumbai", "Chennai", "Bangalore"]

# Features
POLLUTION_FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
METEOROLOGICAL_FEATURES = ["Temperature", "Humidity", "Wind_Speed", "Rainfall"]

# LSTM configuration
LSTM_CONFIG = {
    "time_steps": 7,
    "lstm_units": 64,
    "dropout_rate": 0.2,
    "epochs": 100,
    "batch_size": 32
}

# Model selection
ML_MODELS = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
    "Support Vector Machine",
    "Gradient Boosting",
    "XGBoost"
]
```

---

## Troubleshooting

### Common Issues

**1. TensorFlow Installation Error**
```bash
# Try CPU version
pip install tensorflow-cpu

# Or specific version
pip install tensorflow==2.13.0
```

**2. Memory Error During LSTM Training**
```python
# Reduce batch size in config.py
LSTM_CONFIG["batch_size"] = 16  # instead of 32
```

**3. Dependencies Conflict**
```bash
# Create fresh environment
python -m venv venv_new
# Activate and reinstall
pip install -r requirements.txt
```

**4. Dashboard Not Loading**
```bash
# Check if port is in use
# Try different port
streamlit run dashboard/app.py --server.port 8502
```

**5. Missing Data Errors**
```python
# System automatically creates sample data
# Just run the script
python main.py --city Delhi
```

---

## Performance Optimization

### For Faster Training

1. **Use GPU for LSTM:**
   ```bash
   pip install tensorflow-gpu  # If you have CUDA-compatible GPU
   ```

2. **Reduce Data Size:**
   ```python
   # In preprocessing.py
   df = df.sample(frac=0.5)  # Use 50% of data
   ```

3. **Parallel Processing:**
   ```python
   # Already enabled in Random Forest
   RandomForestRegressor(n_jobs=-1)  # Uses all CPU cores
   ```

### For Better Predictions

1. **Increase Lag Days:**
   ```python
   engineer_features(df, lag_days=14)  # Use 14 days instead of 7
   ```

2. **Hyperparameter Tuning:**
   ```python
   from src.ml_models import hyperparameter_tuning
   best_model, best_params = hyperparameter_tuning("XGBoost", X_train, y_train)
   ```

3. **Ensemble Optimization:**
   ```python
   from src.ensemble import optimize_ensemble_weights
   optimal_weights = optimize_ensemble_weights(predictions, y_true)
   ```

---

## Data Sources

### Recommended Datasets

1. **Kaggle - Air Quality Data in India**
   - URL: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
   - Download and place in `data/raw/`

2. **OpenAQ API**
   ```python
   from src.data_loader import download_from_openaq
   df = download_from_openaq("Delhi", parameter="pm25")
   ```

3. **CPCB India**
   - URL: https://cpcb.nic.in/
   - Manual download from website

---

## Next Steps

### For Beginners:
1. Run `python main.py --city Delhi --mode all`
2. Explore the generated plots in `results/figures/`
3. Check the dashboard: `streamlit run dashboard/app.py`
4. Read the research paper outline

### For Researchers:
1. Customize feature engineering
2. Add new models to `ml_models.py`
3. Experiment with LSTM architectures
4. Conduct hyperparameter tuning
5. Extend to multiple cities

### For Developers:
1. Integrate with real-time data APIs
2. Deploy dashboard to cloud (Heroku/AWS)
3. Create REST API for predictions
4. Build mobile application
5. Add database integration

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## Support

- **GitHub Issues:** https://github.com/AG-1laksh/Airpure/issues
- **Documentation:** See `research_paper/paper_outline.md`
- **Notebooks:** Check `notebooks/` for examples

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

- OpenAQ for open air quality data
- TensorFlow and scikit-learn communities
- Air quality research community

---

## Citation

If you use this project in your research, please cite:

```
@misc{airpure2026,
  author = {Your Name},
  title = {Air Pollution Prediction using Machine Learning and Deep Learning},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/AG-1laksh/Airpure}
}
```

---

**Ready to start? Run:**

```bash
python main.py --city Delhi --mode all
```

Happy predicting! 🌍✨
