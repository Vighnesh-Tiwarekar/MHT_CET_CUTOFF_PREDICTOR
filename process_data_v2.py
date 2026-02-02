import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. DATA LOADING & POPULATION MAP
# ==========================================
FILE_PATH = 'Cutoff_Seat_Combo.csv'
df = pd.read_csv(FILE_PATH)

population_map = {
    2020: 174679, 2021: 161752, 2022: 232964, 
    2023: 313730, 2024: 295577, 2025: 310000, 
    2026: 335000  
}

# Cleanup: Essential for XGBoost
df = df.dropna(subset=['Percentile', 'Merit No'])

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
df['total_population'] = df['Year'].map(population_map)
df['relative_rank'] = df['Merit No'] / df['total_population']

# Branch Trend
branch_trends = df.groupby(['Year', 'Branch Name'])['Percentile'].mean().reset_index()
branch_trends.rename(columns={'Percentile': 'avg_branch_percentile'}, inplace=True)
df = df.merge(branch_trends, on=['Year', 'Branch Name'], how='left')

# College Reputation
college_rank = df[df['Category'].str.contains('OPEN', na=False)].groupby(['Year', 'College Code'])['Percentile'].mean().reset_index()
college_rank.rename(columns={'Percentile': 'college_reputation'}, inplace=True)
df = df.merge(college_rank, on=['Year', 'College Code'], how='left')
df['college_reputation'] = df['college_reputation'].ffill().bfill()

# Lag Feature (Last year's cutoff)
df_lag = df[['Year', 'College Code', 'Branch Code', 'Category', 'Quota', 'Percentile']].copy()
df_lag['Year'] = df_lag['Year'] + 1 
df_lag.rename(columns={'Percentile': 'prev_year_cutoff'}, inplace=True)
df = df.merge(df_lag, on=['Year', 'College Code', 'Branch Code', 'Category', 'Quota'], how='left')
df['prev_year_cutoff'] = df['prev_year_cutoff'].fillna(df['college_reputation'])

# Supply-Demand Ratio
df['seat_to_pop_ratio'] = df['Seat_Intake'] / df['total_population']

# ==========================================
# 3. ENCODING & MODEL PREP
# ==========================================
le_dict = {}
original_cols = ['College Code', 'Branch Code', 'Quota', 'Category']
for col in original_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

features = [
    'Year', 'College Code', 'Branch Code', 'Quota', 'Category', 
    'Seat_Intake', 'total_population', 'relative_rank', 
    'avg_branch_percentile', 'college_reputation', 'prev_year_cutoff', 'seat_to_pop_ratio'
]

# ==========================================
# 4. TRAINING & 2025 VALIDATION
# ==========================================
train_df = df[df['Year'] < 2025].copy()
val_df = df[df['Year'] == 2025].copy()

model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(train_df[features], train_df['Percentile'])

# Predict 2025
val_df['predicted_percentile'] = model.predict(val_df[features])

# --- ACCURACY CALCULATION ---
mae_2025 = mean_absolute_error(val_df['Percentile'], val_df['predicted_percentile'])
r2_2025 = r2_score(val_df['Percentile'], val_df['predicted_percentile'])

print("\n" + "="*40)
print("       2025 VALIDATION REPORT")
print("="*40)
print(f"Mean Absolute Error (MAE): {mae_2025:.4f}")
print(f"Accuracy Score (R2):      {r2_2025:.4f}")
print("="*40)

# Decode 2025 for inspection and save
val_export = val_df.copy()
for col in original_cols:
    val_export[col] = le_dict[col].inverse_transform(val_export[col])

val_export.to_csv('Validation_2025_Results.csv', index=False)
print("File Saved: Validation_2025_Results.csv")

# ==========================================
# 5. PREDICTING 2026
# ==========================================
df_2026 = val_df.copy()
df_2026['Year'] = 2026
df_2026['total_population'] = 335000 
df_2026['prev_year_cutoff'] = val_df['Percentile'] # Use 2025 actuals for 2026 history

# Update Ratios
df_2026['seat_to_pop_ratio'] = df_2026['Seat_Intake'] / df_2026['total_population']

# Prediction
df_2026['predicted_2026_cutoff'] = model.predict(df_2026[features])

# Decode 2026 and save
for col in original_cols:
    df_2026[col] = le_dict[col].inverse_transform(df_2026[col])

df_2026.to_csv('Final_2026_Predictions.csv', index=False)
print("File Saved: Final_2026_Predictions.csv")