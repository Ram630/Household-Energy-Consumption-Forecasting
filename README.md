# ⚡ Household Energy Consumption Forecasting (ML Pipeline)

An end-to-end Machine Learning pipeline that forecasts daily household electricity consumption using historical appliance logs, autoregressive lag features, and calendar/seasonal patterns.

## 📌 Project Overview
Predicting daily household electricity demand enables utility providers to optimize load balance and allows smart home systems to manage energy efficiency. This project utilizes hourly appliance usage logs across 20 households to construct daily autoregressive features and train a high-accuracy Random Forest regressor.

## 📊 Dataset
- **Source:** Kaggle ([Appliance wise Hourly Electricity Consumption](https://www.kaggle.com/datasets/frank451995/appliance-wise-hourly-electricity-consumption/data))
- **Records:** ~175,000 hourly readings resampled into 16,880 daily household entries.
- **Key Attributes:** Household ID, Appliance usage (AC, Fridge, TV, Fans, Lights, Washing Machine), Season, Festival indicators, Timestamp.

## 🛠️ Feature Engineering & Methodology
- **Resampling:** Aggregated hourly appliance-level readings into daily total consumption (`total_kwh`).
- **Lag Features:** 1-day prior consumption (`kwh_lag_1`), 7-day prior consumption (`kwh_lag_7`).
- **Rolling Statistics:** 7-day moving average consumption (`kwh_roll_mean_7`).
- **Encoding:** One-Hot Encoding applied to `house_id`, `season`, and `festival` parameters.
- **Model:** Random Forest Regressor (100 estimators).

## 📈 Model Performance
- **R² Score:** `0.984` (Explains 98.4% of consumption variance)
- **Mean Absolute Error (MAE):** `~1.03 kWh/day`
- **Root Mean Squared Error (RMSE):** `~1.37 kWh/day`

## 🚀 Getting Started
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/household-energy-consumption-forecasting.git](https://github.com/YOUR_USERNAME/household-energy-consumption-forecasting.git)

# Install dependencies
pip install -r requirements.txt

# Run the pipeline script
python energy_forecasting.py
