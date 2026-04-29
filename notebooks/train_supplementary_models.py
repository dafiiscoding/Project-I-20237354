"""
train_supplementary_models.py
=============================
Tai huan luyen 3 model phu (LR, DT, RF) voi best hyperparameters tu notebook 03,
luu thanh .pkl de analysis_addendum.py co the load va chay DeLong + calibration.

Best model XGBoost da duoc luu san o models/best_model.pkl.

Chay: python notebooks/train_supplementary_models.py
"""

import io
import sys
from pathlib import Path

# Force UTF-8 stdout cho Windows console (cp1252 default)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import roc_auc_score

SPLITS_DIR = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

FEATURES = [
    'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents', 'TotalDelinquencyScore', 'FinancialStressIndex',
    'DebtToIncomeRatio', 'DelinquencyTrend',
]
TARGET = 'SeriousDlqin2yrs'

print("Loading splits...")
train_df = pd.read_csv(SPLITS_DIR / "train.csv")
test_df  = pd.read_csv(SPLITS_DIR / "test.csv")
X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]
print(f"  Train: {X_train.shape}, positive rate: {y_train.mean()*100:.2f}%")
print(f"  Test:  {X_test.shape},  positive rate: {y_test.mean()*100:.2f}%")


# ─── Logistic Regression: L1, C=0.001, RobustScaler ────────────────────────────
print("\n[1/3] Training Logistic Regression (L1, C=0.001)...")
lr_pipe = SkPipeline([
    ('scaler', RobustScaler()),
    ('clf', LogisticRegression(
        penalty='elasticnet',
        l1_ratio=1.0,
        C=0.001,
        solver='saga',
        class_weight='balanced',
        max_iter=5000,
        random_state=RANDOM_STATE,
    )),
])
lr_pipe.fit(X_train, y_train)
auc_lr = roc_auc_score(y_test, lr_pipe.predict_proba(X_test)[:, 1])
joblib.dump(lr_pipe, MODELS_DIR / "model_lr.pkl")
print(f"  Test AUC: {auc_lr:.4f}  →  saved models/model_lr.pkl")


# ─── Decision Tree: max_depth=10, min_samples_leaf=200, max_features=0.5 ───────
print("\n[2/3] Training Decision Tree (max_depth=10, min_samples_leaf=200)...")
dt = DecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    min_samples_leaf=200,
    max_features=0.5,
    class_weight='balanced',
    random_state=RANDOM_STATE,
)
dt.fit(X_train, y_train)
auc_dt = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])
joblib.dump(dt, MODELS_DIR / "model_dt.pkl")
print(f"  Test AUC: {auc_dt:.4f}  →  saved models/model_dt.pkl")


# ─── Random Forest: n_estimators=200, max_depth=10, max_features=0.3 ───────────
print("\n[3/3] Training Random Forest (n_estimators=200, max_depth=10, max_features=0.3)...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    max_features=0.3,
    class_weight='balanced',
    oob_score=True,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
rf.fit(X_train, y_train)
auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
joblib.dump(rf, MODELS_DIR / "model_rf.pkl")
print(f"  Test AUC: {auc_rf:.4f}  |  OOB: {rf.oob_score_:.4f}  →  saved models/model_rf.pkl")

print("\nDone. Tiếp theo chạy: python notebooks/analysis_addendum.py")
