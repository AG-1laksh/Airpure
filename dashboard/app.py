"""
Streamlit Dashboard for Air Pollution Prediction System
Run with: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_data, create_sample_data
from src.preprocessing import preprocess_data
from src.ml_models import get_ml_model
from src.visualization import plot_time_series, plot_correlation_matrix
from config import CITIES, POLLUTION_FEATURES, METEOROLOGICAL_FEATURES, AQI_CATEGORIES, AQI_COLORS

# Page configuration
st.set_page_config(
    page_title="Air Pollution Prediction Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .aqi-good { background-color: #00E400; }
    .aqi-satisfactory { background-color: #FFFF00; }
    .aqi-moderate { background-color: #FF7E00; }
    .aqi-poor { background-color: #FF0000; color: white; }
    .aqi-very-poor { background-color: #8F3F97; color: white; }
    .aqi-severe { background-color: #7E0023; color: white; }
</style>
""", unsafe_allow_html=True)


def get_aqi_category(aqi_value):
    """Get AQI category and color"""
    for category, (low, high) in AQI_CATEGORIES.items():
        if low <= aqi_value <= high:
            return category, AQI_COLORS[category]
    return "Severe", AQI_COLORS["Severe"]


def main():
    # Header
    st.markdown('<div class="main-header">🌍 Air Pollution Prediction Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # City selection
    selected_city = st.sidebar.selectbox("Select City", CITIES)
    
    # Mode selection
    mode = st.sidebar.radio("Mode", ["📊 Explore Data", "🔮 Predict AQI", "📈 Model Comparison"])
    
    # Load data
    with st.spinner(f"Loading data for {selected_city}..."):
        try:
            df = load_data(selected_city)
        except:
            st.warning(f"No data found for {selected_city}. Creating sample data...")
            df = create_sample_data(selected_city)
        
        df_clean = preprocess_data(df, remove_outliers=False)
    
    # Main content based on mode
    if mode == "📊 Explore Data":
        show_data_exploration(df_clean, selected_city)
    
    elif mode == "🔮 Predict AQI":
        show_prediction_interface(df_clean, selected_city)
    
    elif mode == "📈 Model Comparison":
        show_model_comparison(df_clean, selected_city)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About**
    
    This dashboard provides air quality analysis and AQI prediction for major Indian cities.
    
    **Project:** Air Pollution Prediction using ML/DL
    
    **GitHub:** [Airpure](https://github.com/AG-1laksh/Airpure)
    """)


def show_data_exploration(df, city):
    """Show data exploration interface"""
    st.markdown(f'<div class="sub-header">📊 Data Exploration - {city}</div>', 
                unsafe_allow_html=True)

    # ── Only work with columns that actually exist ──────────────────────────
    avail_pollution = [c for c in POLLUTION_FEATURES if c in df.columns]
    avail_meteo     = [c for c in METEOROLOGICAL_FEATURES if c in df.columns]
    avail_all       = avail_pollution + avail_meteo + (['AQI'] if 'AQI' in df.columns else [])
    numeric_cols    = df.select_dtypes(include='number').columns.tolist()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    latest_aqi = df['AQI'].iloc[-1]
    avg_aqi = df['AQI'].mean()
    max_aqi = df['AQI'].max()
    min_aqi = df['AQI'].min()
    
    category, color = get_aqi_category(latest_aqi)
    
    with col1:
        st.metric("Current AQI", f"{latest_aqi:.0f}", delta=f"{latest_aqi - avg_aqi:.0f} from avg")
        st.markdown(f'<div style="background-color:{color}; padding:0.5rem; border-radius:0.3rem; text-align:center; font-weight:bold;">{category}</div>', 
                   unsafe_allow_html=True)
    
    with col2:
        st.metric("Average AQI", f"{avg_aqi:.0f}")
    
    with col3:
        st.metric("Maximum AQI", f"{max_aqi:.0f}")
    
    with col4:
        st.metric("Minimum AQI", f"{min_aqi:.0f}")
    
    st.markdown("---")
    
    # Data overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Dataset Overview")
        st.write(f"**Total Records:** {len(df)}")
        st.write(f"**Date Range:** {df['Date'].min().date()} to {df['Date'].max().date()}")
        st.write(f"**Features:** {len(df.columns)}")
        
        # Show sample data
        st.dataframe(df.head(10))
    
    with col2:
        st.subheader("📊 Statistical Summary")
        summary_cols = avail_all if avail_all else numeric_cols
        st.dataframe(df[summary_cols].describe())
    
    # Time series plots
    st.markdown("---")
    st.subheader("📈 Pollution Trends Over Time")
    
    selected_features = st.multiselect(
        "Select features to plot",
        avail_all if avail_all else numeric_cols,
        default=[c for c in ['AQI', 'PM2.5'] if c in df.columns]
    )
    
    if selected_features:
        existing = [f for f in selected_features if f in df.columns]
        if not existing:
            st.warning("None of the selected features exist in this dataset.")
        else:
            fig, axes = plt.subplots(len(existing), 1, figsize=(14, 3*len(existing)))
            if len(existing) == 1:
                axes = [axes]
            for i, feature in enumerate(existing):
                axes[i].plot(df['Date'], df[feature], linewidth=1.5, color='#2E86AB')
                axes[i].set_title(f'{feature} Over Time', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Date')
                axes[i].set_ylabel(feature)
                axes[i].grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
    
    # Correlation heatmap
    st.markdown("---")
    st.subheader("🔥 Feature Correlation Heatmap")

    corr_features = [c for c in POLLUTION_FEATURES + METEOROLOGICAL_FEATURES + ['AQI']
                     if c in df.columns]
    if len(corr_features) < 2:
        corr_features = numeric_cols   # fall back to whatever numbers we have

    if len(corr_features) >= 2:
        corr = df[corr_features].corr()
        fig, ax = plt.subplots(figsize=(max(6, len(corr_features)), max(5, len(corr_features)-1)))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0,
                    square=True, linewidths=1, ax=ax)
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns to show a correlation matrix.")


def show_prediction_interface(df, city):
    """Show AQI prediction interface"""
    st.markdown(f'<div class="sub-header">🔮 AQI Prediction - {city}</div>', 
                unsafe_allow_html=True)
    
    st.write("Adjust the pollution and meteorological parameters to predict AQI:")
    
    # Input features
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏭 Pollution Parameters")
        pm25 = st.slider("PM2.5 (μg/m³)", 0, 500, 100)
        pm10 = st.slider("PM10 (μg/m³)", 0, 500, 150)
        no2 = st.slider("NO2 (μg/m³)", 0, 200, 40)
        so2 = st.slider("SO2 (μg/m³)", 0, 80, 15)
        co = st.slider("CO (mg/m³)", 0.0, 10.0, 1.2)
        o3 = st.slider("O3 (μg/m³)", 0, 200, 50)
    
    with col2:
        st.subheader("🌤️ Meteorological Parameters")
        temp = st.slider("Temperature (°C)", -10, 50, 25)
        humidity = st.slider("Humidity (%)", 0, 100, 60)
        wind_speed = st.slider("Wind Speed (km/h)", 0, 50, 10)
        rainfall = st.slider("Rainfall (mm)", 0, 100, 0)
    
    # Create input array
    input_features = np.array([[pm25, pm10, no2, so2, co, o3, temp, humidity, wind_speed, rainfall]])
    
    if st.button("🔮 Predict AQI", type="primary"):
        with st.spinner("Predicting..."):
            try:
                # Use a simple prediction based on PM2.5 (simplified)
                # In production, load trained model
                predicted_aqi = pm25 * 1.5 + pm10 * 0.5 + no2 * 0.3
                predicted_aqi = min(predicted_aqi, 500)
                
                category, color = get_aqi_category(predicted_aqi)
                
                st.success("Prediction Complete!")
                
                # Display prediction
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    st.markdown(f"""
                    <div style="background-color:{color}; padding:2rem; border-radius:1rem; text-align:center;">
                        <h1 style="color:{'white' if predicted_aqi > 200 else 'black'}; margin:0;">
                            Predicted AQI: {predicted_aqi:.0f}
                        </h1>
                        <h2 style="color:{'white' if predicted_aqi > 200 else 'black'}; margin-top:1rem;">
                            {category}
                        </h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Health recommendations
                st.markdown("---")
                st.subheader("💡 Health Recommendations")
                
                if predicted_aqi <= 50:
                    st.info("✅ Air quality is good. Perfect for outdoor activities!")
                elif predicted_aqi <= 100:
                    st.info("👍 Air quality is satisfactory. Enjoy outdoor activities!")
                elif predicted_aqi <= 200:
                    st.warning("⚠️ Sensitive individuals should limit prolonged outdoor exertion.")
                elif predicted_aqi <= 300:
                    st.warning("⚠️ Everyone should reduce prolonged outdoor exertion.")
                elif predicted_aqi <= 400:
                    st.error("🚨 Avoid outdoor activities. Health alert!")
                else:
                    st.error("🚨 SEVERE! Stay indoors and use air purifiers!")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")


def show_model_comparison(df, city):
    """Show model comparison interface"""
    st.markdown(f'<div class="sub-header">📈 Model Comparison - {city}</div>', 
                unsafe_allow_html=True)
    
    st.info("This section shows comparison of different ML models for AQI prediction.")
    
    # Simulated model results
    model_results = pd.DataFrame({
        'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 
                 'XGBoost', 'Gradient Boosting', 'LSTM'],
        'RMSE': [28.5, 22.3, 15.7, 14.2, 15.1, 12.8],
        'MAE': [21.3, 17.2, 11.9, 10.5, 11.2, 9.7],
        'R² Score': [0.78, 0.84, 0.91, 0.93, 0.92, 0.95]
    })
    
    # Display table
    st.subheader("📊 Model Performance Metrics")
    st.dataframe(model_results.style.highlight_min(subset=['RMSE', 'MAE'], color='lightgreen')
                                    .highlight_max(subset=['R² Score'], color='lightgreen'))
    
    # Best model
    best_model = model_results.loc[model_results['RMSE'].idxmin(), 'Model']
    st.success(f"🏆 Best Model: **{best_model}** (Lowest RMSE)")
    
    # Comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(model_results['Model'], model_results['RMSE'], color='#2E86AB')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('Model Comparison - RMSE (Lower is Better)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(model_results['Model'], model_results['R² Score'], color='#A23B72')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Model Comparison - R² Score (Higher is Better)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)


if __name__ == "__main__":
    main()
