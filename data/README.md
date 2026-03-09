# Dataset Information

## Air Quality Dataset

### Data Sources

#### 1. OpenAQ
- **URL**: https://openaq.org/
- **Description**: Global air quality data from real-time monitoring stations
- **Coverage**: Multiple cities worldwide
- **Format**: JSON/CSV
- **Update Frequency**: Real-time

#### 2. Central Pollution Control Board (CPCB) India
- **URL**: https://cpcb.nic.in/
- **Description**: Official air quality data for Indian cities
- **Coverage**: Major cities in India
- **Format**: CSV/Excel
- **Update Frequency**: Daily

#### 3. Kaggle Datasets
- **URL**: https://www.kaggle.com/datasets
- **Recommended Datasets**:
  - Air Quality Data in India (2015-2020)
  - City Air Quality Dataset
  - Real-time Air Quality Index

### Target Cities
1. **Delhi** - National Capital Region
2. **Mumbai** - Financial Capital
3. **Chennai** - South India Metro
4. **Bangalore** - IT Capital

### Features Description

#### Pollution Features
| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| PM2.5 | Particulate Matter < 2.5 μm | μg/m³ | 0-500+ |
| PM10 | Particulate Matter < 10 μm | μg/m³ | 0-500+ |
| NO2 | Nitrogen Dioxide | μg/m³ | 0-200+ |
| SO2 | Sulfur Dioxide | μg/m³ | 0-80+ |
| CO | Carbon Monoxide | mg/m³ | 0-30+ |
| O3 | Ozone | μg/m³ | 0-200+ |

#### Meteorological Features
| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| Temperature | Ambient Temperature | °C | -10 to 50 |
| Humidity | Relative Humidity | % | 0-100 |
| Wind Speed | Wind Speed | km/h | 0-100 |
| Rainfall | Precipitation | mm | 0-500 |

#### Target Variable
- **AQI (Air Quality Index)**: Composite index (0-500+)

### AQI Categories (India Standard)
| Category | AQI Range | Health Impact | Color Code |
|----------|-----------|---------------|------------|
| Good | 0-50 | Minimal impact | Green |
| Satisfactory | 51-100 | Minor breathing discomfort | Yellow |
| Moderate | 101-200 | Breathing discomfort to sensitive people | Orange |
| Poor | 201-300 | Breathing discomfort to most people | Red |
| Very Poor | 301-400 | Respiratory illness on prolonged exposure | Purple |
| Severe | 401-500 | Affects healthy people, serious impacts | Maroon |

### Data Collection Instructions

#### Option 1: Use Kaggle Dataset (Recommended for Research)
1. Download from: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
2. Place CSV files in `data/raw/` directory
3. Run preprocessing script

#### Option 2: Use OpenAQ API
```python
import requests

url = "https://api.openaq.org/v2/measurements"
params = {
    "city": "Delhi",
    "parameter": "pm25",
    "limit": 10000
}
response = requests.get(url, params=params)
data = response.json()
```

#### Option 3: Manual Download from CPCB
1. Visit https://cpcb.nic.in/
2. Navigate to "Air Quality Data"
3. Select city and date range
4. Download CSV files

### Dataset Structure

Expected CSV format:
```
Date,City,PM2.5,PM10,NO2,SO2,CO,O3,Temperature,Humidity,Wind_Speed,Rainfall,AQI
2023-01-01,Delhi,150,200,45,20,1.5,30,15,65,5,2,175
...
```

### Data Quality Notes
- Missing values: Handle using forward fill or interpolation
- Outliers: Values beyond typical ranges should be investigated
- Temporal gaps: Ensure continuous time series or fill gaps
- Consistency: Verify units and scaling across sources

### Sample Data
A sample dataset is provided in `data/raw/sample_air_quality.csv` for testing purposes.

### Data Preprocessing Steps
1. Load raw data
2. Handle missing values
3. Remove duplicates
4. Detect and handle outliers
5. Feature scaling/normalization
6. Create lag features
7. Split into train/test sets
8. Save processed data

### Usage
```python
from src.data_loader import load_data
from src.preprocessing import preprocess_data

# Load raw data
df = load_data(city="Delhi")

# Preprocess
df_clean = preprocess_data(df)
```

### License
Data usage subject to respective source licenses. Ensure compliance with terms of service.
