import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
# Load raw hourly appliance consumption dataset
raw_df = pd.read_csv('appliance_usage_dataset.csv')

# Ensure timestamp is formatted properly
raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])

# Compute total hourly consumption across all appliances
appliance_cols = ['ac', 'fridge', 'lights', 'fans', 'washing_machine', 'tv']
raw_df['total_kwh'] = raw_df[appliance_cols].sum(axis=1)

# Handle categorical missing values (normal non-holiday days)
raw_df['festival'] = raw_df['festival'].fillna('None')
raw_df['season'] = raw_df['season'].fillna('Unknown')

# Aggregate hourly data into daily household totals
daily_df = raw_df.groupby(['house_id', pd.Grouper(key='timestamp', freq='D')]).agg({
    'ac': 'sum',
    'fridge': 'sum',
    'lights': 'sum',
    'fans': 'sum',
    'washing_machine': 'sum',
    'tv': 'sum',
    'total_kwh': 'sum',
    'season': 'first',
    'festival': 'first'
}).reset_index()

# Sort chronologically per household
daily_df = daily_df.sort_values(by=['house_id', 'timestamp']).reset_index(drop=True)

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
# Autoregressive Lag Features
daily_df['kwh_lag_1'] = daily_df.groupby('house_id')['total_kwh'].shift(1)
daily_df['kwh_lag_7'] = daily_df.groupby('house_id')['total_kwh'].shift(7)
daily_df['kwh_roll_mean_7'] = daily_df.groupby('house_id')['total_kwh'].shift(1).rolling(window=7).mean()

# Temporal/Calendar Features
daily_df['day_of_week'] = daily_df['timestamp'].dt.dayofweek
daily_df['day_of_month'] = daily_df['timestamp'].dt.day
daily_df['month'] = daily_df['timestamp'].dt.month

# Categorical One-Hot Encoding
encoded_df = pd.get_dummies(daily_df, columns=['house_id', 'season', 'festival'], drop_first=True)

# Clean rows affected by initial lag shifts
lag_cols = ['kwh_lag_1', 'kwh_lag_7', 'kwh_roll_mean_7']
clean_df = encoded_df.dropna(subset=lag_cols).reset_index(drop=True)

# ==========================================
# 3. MODEL TRAINING & EVALUATION
# ==========================================
ignore_cols = ['timestamp', 'total_kwh', 'ac', 'fridge', 'lights', 'fans', 'washing_machine', 'tv']
features = [col for col in clean_df.columns if col not in ignore_cols]
target = 'total_kwh'

X = clean_df[features]
y = clean_df[target]

# 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation Metrics
predictions = model.predict(X_test)
print("--- MODEL PERFORMANCE METRICS ---")
print(f"R² Score: {r2_score(y_test, predictions):.3f}")
print(f"MAE:      {mean_absolute_error(y_test, predictions):.3f} kWh")
print(f"RMSE:     {root_mean_squared_error(y_test, predictions):.3f} kWh")

# ==========================================
# 4. SAVE MODEL ARTIFACT
# ==========================================
joblib.dump(model, 'random_forest_energy_model.pkl')
print("\nModel artifact successfully saved to 'random_forest_energy_model.pkl'")