"""
ML Assignment 1 - Part 2: Subterranean Thermal Reservoir Mapping (var2)
Roll Number: BT2024208

Goal: Evaluate Linear Regression, Ridge, Lasso, and ElasticNet across
      polynomial degrees (1 to 10) using 5-fold Cross-Validation.
      Select the best model, predict Thermal Anomaly Score, and save to BT2024208_pred_var2.csv.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────────────────
TRAIN_PATH  = r"Data\BT2024208_train_var2.csv"
TEST_PATH   = r"Data\BT2024208_test_var2.csv"
OUTPUT_PATH = r"Data\BT2024208_pred_var2.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

print("============================================================", flush=True)
print("ML Assignment 1 - Part 2: Thermal Reservoir Mapping (var2)", flush=True)
print("Model Comparison: OLS Linear, Ridge, Lasso, ElasticNet (Degrees 1-10)", flush=True)
print("Roll No: BT2024208", flush=True)
print("============================================================", flush=True)
print(f"Training samples : {len(train_df)}", flush=True)
print(f"Test samples     : {len(test_df)}", flush=True)

# ──────────────────────────────────────────────────────────
# 2. Feature Selection (all 3 features: x1, x2, x3)
# ──────────────────────────────────────────────────────────
FEATURES = ['x1', 'x2', 'x3']
TARGET   = 'y'

X_train = train_df[FEATURES].values
y_train = train_df[TARGET].values
X_test  = test_df[FEATURES].values

print(f"Features used    : {FEATURES}", flush=True)

# ──────────────────────────────────────────────────────────
# 3. Define Candidate Models across Degrees 1-10
# ──────────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)
MAX_DEGREE = 20

results = []

def get_candidate_models():
    models = {
        'Linear Regression':            LinearRegression(),
        'Ridge (alpha=0.1)':            Ridge(alpha=0.1),
        'Ridge (alpha=1.0)':            Ridge(alpha=1.0),
        'Ridge (alpha=10.0)':           Ridge(alpha=10.0),
        'Lasso (alpha=0.001)':          Lasso(alpha=0.001, max_iter=10000, tol=0.01, random_state=42),
        'Lasso (alpha=0.01)':           Lasso(alpha=0.01,  max_iter=10000, tol=0.01, random_state=42),
        'Lasso (alpha=0.1)':            Lasso(alpha=0.1,   max_iter=10000, tol=0.01, random_state=42),
        'ElasticNet (a=0.001, l1=0.5)': ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, tol=0.01, random_state=42),
        'ElasticNet (a=0.01, l1=0.5)':  ElasticNet(alpha=0.01,  l1_ratio=0.5, max_iter=10000, tol=0.01, random_state=42),
        'ElasticNet (a=0.01, l1=0.7)':  ElasticNet(alpha=0.01,  l1_ratio=0.7, max_iter=10000, tol=0.01, random_state=42),
    }
    return models

print(flush=True)
print(f"{'Degree':>6} | {'Model Candidate':>30} | {'CV MSE':>10} | {'CV R2':>8}", flush=True)
print("-" * 65, flush=True)

for deg in range(1, MAX_DEGREE + 1):
    candidates = get_candidate_models()
    for name, model_inst in candidates.items():
        pipe = Pipeline([
            ('poly',   PolynomialFeatures(degree=deg, include_bias=True)),
            ('scaler', StandardScaler()),
            ('model',  model_inst)
        ])

        cv_res = cross_validate(pipe, X_train, y_train, cv=kf,
                                scoring={'mse': 'neg_mean_squared_error', 'r2': 'r2'},
                                n_jobs=-1)

        mse_val = -cv_res['test_mse'].mean()
        r2_val  =  cv_res['test_r2'].mean()

        results.append({
            'degree':            deg,
            'model_name':        name,
            'pipeline_template': pipe,
            'cv_mse':            mse_val,
            'cv_r2':             r2_val
        })
        print(f"  {deg:4d} | {name:>30} | {mse_val:10.4f} | {r2_val:8.4f}", flush=True)

# ──────────────────────────────────────────────────────────
# 4. Select Best Model with 1% Relative Tolerance (Parsimony)
# ──────────────────────────────────────────────────────────
# Complexity rank: lower = simpler (for tie-breaking within same degree)
COMPLEXITY_RANK = {
    'Linear Regression':            0,
    'Ridge (alpha=0.1)':            1,
    'Ridge (alpha=1.0)':            2,
    'Ridge (alpha=10.0)':           3,
    'Lasso (alpha=0.001)':          4,
    'Lasso (alpha=0.01)':           5,
    'Lasso (alpha=0.1)':            6,
    'ElasticNet (a=0.001, l1=0.5)': 7,
    'ElasticNet (a=0.01, l1=0.5)':  8,
    'ElasticNet (a=0.01, l1=0.7)':  9,
}
TOLERANCE = 0.01   # 1% relative tolerance

results_df = pd.DataFrame(results)
results_df['complexity_rank'] = results_df['model_name'].map(COMPLEXITY_RANK)

# Save metrics to CSV for plotting
metrics_csv_path = r"Data\metrics_part2.csv"
results_df[['degree', 'model_name', 'cv_mse', 'cv_r2']].to_csv(metrics_csv_path, index=False)
print(f"Saved cross-validation metrics to: {metrics_csv_path}", flush=True)

# Raw global best (lowest CV MSE)
best_idx  = results_df['cv_mse'].idxmin()
best_row  = results_df.loc[best_idx]
RAW_BEST_MSE = best_row['cv_mse']

# Threshold: within 1% of the raw best MSE
threshold = RAW_BEST_MSE * (1 + TOLERANCE)

# Among all candidates within threshold, pick lowest degree -> then lowest complexity
within_tol = results_df[results_df['cv_mse'] <= threshold].copy()
within_tol_sorted = within_tol.sort_values(['degree', 'complexity_rank'])
selected_row = within_tol_sorted.iloc[0]

BEST_DEGREE   = int(selected_row['degree'])
BEST_NAME     = selected_row['model_name']
BEST_TEMPLATE = selected_row['pipeline_template']

print(flush=True)
print("=" * 65, flush=True)
print("  Raw global best (lowest CV MSE):", flush=True)
print(f"     Model     : {best_row['model_name']}", flush=True)
print(f"     Degree    : {int(best_row['degree'])}", flush=True)
print(f"     CV MSE    : {RAW_BEST_MSE:.4f}", flush=True)
print(f"     CV R2     : {best_row['cv_r2']:.4f}", flush=True)
print(flush=True)
print(f"  1% Tolerance threshold : {threshold:.4f}", flush=True)
print(f"  Candidates within tolerance : {len(within_tol)}", flush=True)
print(flush=True)
print("  SELECTED MODEL (simplest within 1% tolerance):", flush=True)
print(f"     Model     : {BEST_NAME}", flush=True)
print(f"     Degree    : {BEST_DEGREE}", flush=True)
print(f"     CV MSE    : {selected_row['cv_mse']:.4f}", flush=True)
print(f"     CV R2     : {selected_row['cv_r2']:.4f}", flush=True)
print("=" * 65, flush=True)

# ──────────────────────────────────────────────────────────
# 5. Train Final Pipeline on full training set
# ──────────────────────────────────────────────────────────
final_pipeline = BEST_TEMPLATE
final_pipeline.fit(X_train, y_train)

y_train_pred = final_pipeline.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2  = r2_score(y_train, y_train_pred)

print(flush=True)
print(f"Final Trained Model ({BEST_NAME}, degree={BEST_DEGREE}) - Training Metrics:", flush=True)
print(f"  MSE  = {train_mse:.6f}", flush=True)
print(f"  R2   = {train_r2:.6f}", flush=True)

# ──────────────────────────────────────────────────────────
# 6. Predict on Test Set & Save
# ──────────────────────────────────────────────────────────
y_test_pred = final_pipeline.predict(X_test)

print(flush=True)
print(f"Test Predictions Summary:", flush=True)
print(f"  Count : {len(y_test_pred)}", flush=True)
print(f"  Min   : {y_test_pred.min():.4f}", flush=True)
print(f"  Max   : {y_test_pred.max():.4f}", flush=True)
print(f"  Mean  : {y_test_pred.mean():.4f}", flush=True)
print(f"  Std   : {y_test_pred.std():.4f}", flush=True)

pred_df = pd.DataFrame({'y': y_test_pred})
pred_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nPredictions saved to: {OUTPUT_PATH}", flush=True)

print(flush=True)
print("============================================================", flush=True)
print("DONE", flush=True)
print("============================================================", flush=True)
