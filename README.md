# Air Pollution Detection and Prediction System

## Detection and Prediction of Air Pollution Levels Using Machine Learning and Deep Learning Models

### Project Overview
This research project implements an end-to-end machine learning pipeline for predicting Air Quality Index (AQI) using historical air pollution and meteorological data. The system combines traditional machine learning models, deep learning (LSTM), ensemble methods, and explainable AI techniques.

### Features
- 🌍 Multi-city air quality analysis
- 🤖 Multiple ML models (Linear Regression, Decision Tree, Random Forest, SVM, Gradient Boosting, XGBoost)
- 🧠 LSTM deep learning for time-series prediction
- 🔄 Ensemble/hybrid models
- 📊 Comprehensive visualizations
- 🔍 Feature importance analysis
- 💡 Explainable AI using SHAP
- 🔮 Future AQI prediction (1-day, 3-day, 7-day ahead)
- 📱 Interactive Streamlit dashboard


### Project Structure
```
Airpure/
├── data/                     # Dataset storage
│   ├── raw/                  # Raw downloaded data
│   ├── processed/            # Cleaned and preprocessed data
│   └── README.md
├── notebooks/                # Jupyter notebooks for analysis
├── src/                      # Source code modules
├── models/                   # Saved trained models
├── results/                  # Outputs and visualizations
├── dashboard/                # Streamlit dashboard
├── research_paper/           # Research paper content
├── requirements.txt
├── config.py
├── main.py
└── README.md
```

### Installation

```bash
# Clone the repository
git clone https://github.com/AG-1laksh/Airpure.git
cd Airpure

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Data Collection and Preprocessing
```bash
python main.py --mode preprocess --city Delhi
```

#### 2. Train All Models
```bash
python main.py --mode train --city Delhi
```

#### 3. Evaluate Models
```bash
python main.py --mode evaluate
```

#### 4. Generate Predictions
```bash
python main.py --mode predict --days 7
```

#### 5. Run Dashboard
```bash
streamlit run dashboard/app.py
```

### Dataset Sources
- **OpenAQ**: https://openaq.org/
- **CPCB India**: https://cpcb.nic.in/
- **Kaggle**: Air Quality datasets

### Target Cities
- Delhi
- Mumbai
- Chennai
- Bangalore

### Models Implemented

#### Traditional ML Models
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Support Vector Machine (SVR)
5. Gradient Boosting Regressor
6. XGBoost Regressor

#### Deep Learning
- LSTM (Long Short-Term Memory) for time-series prediction

#### Ensemble
- Hybrid models combining multiple predictions

### Evaluation Metrics
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

### Research Contributions
- Integration of pollution and meteorological data
- Comprehensive model comparison
- LSTM-based temporal pattern analysis
- Hybrid ensemble modeling
- Feature importance and explainability analysis
- Multi-horizon future predictions
- Interactive visualization dashboard

### Requirements
- Python 3.8+
- TensorFlow/Keras
- Scikit-learn
- XGBoost
- SHAP
- Pandas, NumPy
- Matplotlib, Seaborn
- Streamlit


```

### Contact
GitHub: https://github.com/AG-1laksh/Airpure
