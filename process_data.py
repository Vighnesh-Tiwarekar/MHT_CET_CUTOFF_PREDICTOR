import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================
FILE_PATH = 'Cutoff_Seat_Combo.csv'
df = pd.read_csv(FILE_PATH)

# Population data provided by you
population_map = {
    2020: 174679,
    2021: 161752,
    2022: 232964,
    2023: 313730,
    2024: 295577,
    2025: 310000, # Actual/Estimate for 2025
    2026: 335000  # Predicted for 2026
}

# Cleanup: Remove NaNs in target columns
df = df.dropna(subset=['Percentile', 'Merit No'])

# ==========================================
# 2. ADVANCED FEATURE ENGINEERING (The "Fixes")
# ==========================================

# A. Population Normalization
df['total_population'] = df['Year'].map(population_map)
df['relative_rank'] = df['Merit No'] / df['total_population']

# B. Branch Trend Feature (How much a branch is growing in popularity)
branch_trends = df.groupby(['Year', 'Branch Name'])['Percentile'].mean().reset_index()
branch_trends.rename(columns={'Percentile': 'avg_branch_percentile'}, inplace=True)
df = df.merge(branch_trends, on=['Year', 'Branch Name'], how='left')

# C. College Reputation Score
college_rank = df[df['Category'].str.contains('OPEN', na=False)].groupby(['Year', 'College Code'])['Percentile'].mean().reset_index()
college_rank.rename(columns={'Percentile': 'college_reputation'}, inplace=True)
df = df.merge(college_rank, on=['Year', 'College Code'], how='left')
df['college_reputation'] = df['college_reputation'].ffill().bfill()

# D. THE KEY FIX: Lag Feature (What was the cutoff LAST year for this exact seat?)
df_lag = df[['Year', 'College Code', 'Branch Code', 'Category', 'Quota', 'Percentile']].copy()
df_lag['Year'] = df_lag['Year'] + 1 
df_lag.rename(columns={'Percentile': 'prev_year_cutoff'}, inplace=True)
df = df.merge(df_lag, on=['Year', 'College Code', 'Branch Code', 'Category', 'Quota'], how='left')

# Fill missing lag data (for new branches) with the college average
df['prev_year_cutoff'] = df['prev_year_cutoff'].fillna(df['college_reputation'])

# E. Supply-Demand Ratio
df['seat_to_pop_ratio'] = df['Seat_Intake'] / df['total_population']

# ==========================================
# 3. ENCODING & MODEL PREP
# ==========================================
le_dict = {}
for col in ['College Code', 'Branch Code', 'Quota', 'Category']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

features = [
    'Year', 'College Code', 'Branch Code', 'Quota', 'Category', 
    'Seat_Intake', 'total_population', 'relative_rank', 
    'avg_branch_percentile', 'college_reputation', 'prev_year_cutoff', 'seat_to_pop_ratio'
]

# ==========================================
# 4. TRAINING (2020-2024) & VALIDATION (2025)
# ==========================================
train_df = df[df['Year'] < 2025].copy()
val_df = df[df['Year'] == 2025].copy()

model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1
)

model.fit(train_df[features], train_df['Percentile'])

# Predict 2025
val_df['predicted_percentile'] = model.predict(val_df[features])

# Accuracy Report
mae = mean_absolute_error(val_df['Percentile'], val_df['predicted_percentile'])
r2 = r2_score(val_df['Percentile'], val_df['predicted_percentile'])

print(f"\n--- REFINED 2025 REPORT ---")
print(f"New MAE: {mae:.4f} (Should be lower than 5.0!)")
print(f"New R2:  {r2:.4f}")

# ==========================================
# 5. PREDICTING 2026 (THE NEWS INJECTION)
# ==========================================
# Create a 2026 dataset based on 2025 structure
df_2026 = val_df.copy()
df_2026['Year'] = 2026
df_2026['total_population'] = 335000 
df_2026['prev_year_cutoff'] = val_df['Percentile'] # 2025 actual becomes 2026's lag

# --- INJECT YOUR NEWS HERE ---
# Example: If College 1002 doubles seats in 2026
# df_2026.loc[df_2026['College Code'] == le_dict['College Code'].transform(['1002'])[0], 'Seat_Intake'] = 120

# Re-calculate ratio for 2026
df_2026['seat_to_pop_ratio'] = df_2026['Seat_Intake'] / df_2026['total_population']

# FINAL PREDICTION
df_2026['predicted_2026_cutoff'] = model.predict(df_2026[features])

# Decode for readability
for col in ['College Code', 'Branch Code', 'Quota', 'Category']:
    df_2026[col] = le_dict[col].inverse_transform(df_2026[col])

df_2026.to_csv('Final_2026_Predictions.csv', index=False)
print("\nSuccess! Final_2026_Predictions.csv generated.")