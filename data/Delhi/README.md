# Delhi Air Quality Dataset

Place your Delhi air quality dataset files in this folder.

## Folder Structure

```
Delhi/
├── raw/          ← Place your original/downloaded CSV files here
├── processed/    ← Cleaned and preprocessed files are saved here
└── README.md
```

## Expected File Format

Place your CSV file in `raw/` and name it **`Delhi_air_quality.csv`**.

The file should contain the following columns:

| Column | Description | Unit |
|--------|-------------|------|
| Date | Date of measurement | YYYY-MM-DD |
| City | City name (Delhi) | - |
| PM2.5 | Particulate Matter < 2.5 μm | μg/m³ |
| PM10 | Particulate Matter < 10 μm | μg/m³ |
| NO2 | Nitrogen Dioxide | μg/m³ |
| SO2 | Sulfur Dioxide | μg/m³ |
| CO | Carbon Monoxide | mg/m³ |
| O3 | Ozone | μg/m³ |
| Temperature | Ambient Temperature | °C |
| Humidity | Relative Humidity | % |
| Wind_Speed | Wind Speed | km/h |
| Rainfall | Precipitation | mm |
| AQI | Air Quality Index | 0–500 |

## Accepted Data Sources

- **CPCB India** — https://cpcb.nic.in/
- **OpenAQ** — https://openaq.org/
- **Kaggle** — https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
- Any CSV exported from a Delhi air quality monitoring station

## Usage

Once you place your dataset in `raw/`, load it in code as:

```python
from src.data_loader import load_data

df = load_data("Delhi", file_path="data/Delhi/raw/Delhi_air_quality.csv")
```

Or run the full pipeline:

```bash
python main.py --city Delhi --mode all
```

Processed output will be saved automatically to `data/Delhi/processed/`.
