# PulmoPulse — Air-Quality & Respiratory-Risk Forecasting

PulmoPulse transforms ordinary weather forecasts into forward-looking insights on **air-pollution levels** and the **incidence of seven common respiratory diseases**.

---

## Why this project exists

- **Early warning for public health.** By forecasting pollutant loads and likely respiratory-case counts a week or two ahead, health authorities and citizens can take proactive measures.
- **Data-driven policy.** Transparent, reproducible predictions support evidence-based interventions in air-quality management.
- **Open science.** All code, models, and feature-engineering steps are openly shared to foster collaboration and further research.

---

## How it works (conceptual)

1. **Input**  
   Hourly or daily weather data for the next 7–14 days—either uploaded as a CSV or fetched from a weather API.

2. **Feature engineering**  
   Climate variables are cleaned and enriched (diurnal temperature range, humidity–temperature index, rolling precipitation metrics, wind variability, etc.).

3. **Inference**  
   Two CatBoost regressors—registered in an MLflow model registry—generate:  
   - Daily concentrations for AQI components (CO, NO, NO₂, O₃, SO₂, PM₂.₅, PM₁₀, NH₃).  
   - Daily case counts for asthma, bronchitis, COPD, influenza, pneumonia, tuberculosis, and URTI.

4. **Visualization & insight**  
   A Streamlit UI displays interactive charts and downloadable tables for rapid situational awareness.

---

## Key features

| Category | Highlights |
|----------|------------|
| **End-user app** | Streamlit web UI, Altair charts, single-click CSV download |
| **Modeling** | CatBoost multi-output regressors, MLflow registry integration |
| **Data pipeline** | Robust climate feature-engineering (`feature_engineering.py`), cache-efficient inference |
| **Dev experience** | Pre-configured VS Code dev-container, MIT license, minimal dependencies |

---

## Repository at a glance

```
├── app.py                  # Streamlit interface & inference workflow
├── feature_engineering.py  # Climate feature-engineering pipeline
├── requirements.txt        # Lightweight Python dependency list
├── pulmo_icon.png          # Favicon / logo
├── .streamlit/             # UI theme settings
└── .devcontainer/          # VS Code-based container config
```

---

## Acknowledgements

- [Global Partnership for Sustainable Development Data (GPSDD)](https://www.data4sdgs.org/)
- [Visual Crossing](https://www.visualcrossing.com/) – weather API  
- Early reviewers from the GPSDD CAN Fellows Network
