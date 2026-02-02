import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. LOAD THE PREDICTED DATA
file_path = '2025_Prediction_Test.csv'
df = pd.read_csv(file_path)

# 2. DATA CLEANING
# Ensure we only compare rows where we have both values
df = df.dropna(subset=['Percentile', 'predicted_2025_percentile'])

# 3. CALCULATE METRICS
actual = df['Percentile']
predicted = df['predicted_2025_percentile']

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

# Calculate error and error percentage
df['absolute_error'] = (df['Percentile'] - df['predicted_2025_percentile']).abs()

print("-" * 30)
print("   MODEL ACCURACY SUMMARY")
print("-" * 30)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-Squared (Accuracy Score): {r2:.4f}")
print("-" * 30)

# 4. TOP 10 BRANCHES WITH HIGHEST ERROR
# This helps identify if a specific branch (like AI/DS) had a "shock" shift
print("\n[!] Top 10 Branches with Highest Prediction Error:")
branch_err = df.groupby('Branch Code')['absolute_error'].mean().sort_values(ascending=False).head(10)
print(branch_err)

# 5. CATEGORY-WISE PERFORMANCE
print("\n[!] Accuracy by Category:")
cat_err = df.groupby('Category')['absolute_error'].mean().sort_values()
print(cat_err)

# 6. IDENTIFY OUTLIERS (The "Shock" Colleges)
# These are rows where the model was off by more than 5 percentile points
outliers = df[df['absolute_error'] > 5]
print(f"\n[!] Number of extreme outliers (>5% off): {len(outliers)}")

# 7. SAVE DETAILED ERROR REPORT
df.to_csv('detailed_accuracy_report.csv', index=False)
print("\nDone! Detailed report saved to 'detailed_accuracy_report.csv'")