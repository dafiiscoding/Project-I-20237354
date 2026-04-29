"""
models.py — Các wrapper huấn luyện cho Loan Default Prediction

Cung cấp hàm huấn luyện chuẩn hóa cho 4 mô hình: Logistic Regression,
Decision Tree, Random Forest, XGBoost. Mỗi wrapper trả về mô hình đã
khớp (fitted model) cùng metadata của quá trình huấn luyện.

Nguyên tắc thiết kế:
- Mọi mô hình đều hỗ trợ Stratified K-Fold CV
- Dữ liệu mất cân bằng (imbalanced data) được xử lý nhất quán
  (class_weight hoặc scale_pos_weight)
- Random state cố định để đảm bảo tái lập kết quả (reproducibility)
"""

import time
import warnings
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, cross_val_score
)
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline as SkPipeline
import xgboost as xgb

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
MODELS_DIR = Path(__file__).parent.parent / "models"


# ─── Logistic Regression ──────────────────────────────────────────────────────

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    n_iter: int = 12,
    random_state: int = RANDOM_STATE,
) -> Tuple[Any, Dict]:
    """
    Huấn luyện Logistic Regression với RobustScaler và chính quy hóa L1/L2.

    Cơ sở toán học:
        P(y=1|x) = σ(xᵀβ) = 1 / (1 + exp(-xᵀβ))
        Hàm mất mát (MLE): L(β) = -Σ [yᵢ log σ(xᵢᵀβ) + (1-yᵢ) log(1-σ(xᵢᵀβ))]
        Phạt L2: L_reg = L(β) + (1/2C) ||β||²
        Phạt L1: L_reg = L(β) + (1/C) ||β||₁

    Tìm kiếm siêu tham số (hyperparameter search):
        - C ∈ [0.001, 10]: nghịch đảo cường độ chính quy hóa
        - penalty ∈ ['l1', 'l2']
        - solver: saga (hỗ trợ cả L1 và L2 cho tập dữ liệu lớn)

    Parameters
    ----------
    X_train : pd.DataFrame
        Đặc trưng huấn luyện (thô, chưa chuẩn hóa — pipeline sẽ tự scale).
    y_train : pd.Series
        Nhãn nhị phân (0/1).
    cv : int
        Số fold trong Stratified CV.
    n_iter : int
        Số vòng lặp RandomizedSearchCV.
    random_state : int

    Returns
    -------
    pipeline : SkPipeline
        Pipeline đã khớp (RobustScaler + LogisticRegression).
    meta : dict
        Metadata huấn luyện: best_params, cv_auc_mean, cv_auc_std, train_time_sec.
    """
    t0 = time.time()

    pipeline = SkPipeline([
        ('scaler', RobustScaler()),
        ('clf', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state,
            solver='saga',
        )),
    ])

    param_dist = {
        'clf__C': np.logspace(-3, 1, 100),
        'clf__penalty': ['l1', 'l2'],
    }

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    cv_scores = cross_val_score(best, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)

    meta = {
        'best_params': search.best_params_,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'train_time_sec': time.time() - t0,
    }
    return best, meta


# ─── Decision Tree ────────────────────────────────────────────────────────────

def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    n_iter: int = 20,
    random_state: int = RANDOM_STATE,
) -> Tuple[Any, Dict]:
    """
    Huấn luyện Decision Tree CART với cắt tỉa (pruning) qua max_depth/min_samples.

    Cơ sở toán học:
        Gini(t) = 1 - Σ p(c|t)²
        Gain(t, A) = Gini(t) - Σ |tᵢ|/|t| * Gini(tᵢ)
        Điểm phân chia chọn thuộc tính A tối đa hóa Gain(t, A)

    Tìm kiếm siêu tham số:
        - max_depth ∈ [3, 20]: kiểm soát độ sâu cây
        - min_samples_leaf ∈ [10, 500]: cắt tỉa chống overfitting
        - min_samples_split ∈ [20, 500]
        - criterion ∈ ['gini', 'entropy']

    Returns
    -------
    model : DecisionTreeClassifier
    meta : dict
    """
    t0 = time.time()

    param_dist = {
        'max_depth': list(range(3, 21)),
        'min_samples_leaf': list(range(10, 500, 10)),
        'min_samples_split': list(range(20, 500, 10)),
        'criterion': ['gini', 'entropy'],
    }

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    base = DecisionTreeClassifier(class_weight='balanced', random_state=random_state)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    cv_scores = cross_val_score(best, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)

    meta = {
        'best_params': search.best_params_,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'train_time_sec': time.time() - t0,
    }
    return best, meta


# ─── Random Forest ────────────────────────────────────────────────────────────

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    n_iter: int = 15,
    random_state: int = RANDOM_STATE,
) -> Tuple[Any, Dict]:
    """
    Huấn luyện Random Forest (Bagging + lấy mẫu ngẫu nhiên đặc trưng).

    Cơ sở toán học:
        Tập hợp mô hình: F(x) = (1/B) Σᵦ fᵦ(x)  (biểu quyết đa số / trung bình xác suất)
        Giảm phương sai: Var(F) = ρσ² + (1-ρ)σ²/B
          - ρ = tương quan từng cặp cây (pairwise correlation)
          - Lấy mẫu ngẫu nhiên đặc trưng làm giảm ρ → giảm phương sai tổng thể
        OOB error: mỗi cây chỉ dùng ~63.2% mẫu → ~36.8% mẫu OOB
          → ước lượng không thiên lệch lỗi test mà không cần validation set riêng

    Tìm kiếm siêu tham số:
        - n_estimators ∈ [100, 500]
        - max_depth ∈ [5, 25] + None
        - max_features ∈ ['sqrt', 'log2', 0.3, 0.5]
        - min_samples_leaf ∈ [1, 50]

    Returns
    -------
    model : RandomForestClassifier (với oob_score=True)
    meta : dict
    """
    t0 = time.time()

    param_dist = {
        'n_estimators': [100, 150, 200, 300, 400, 500],
        'max_depth': list(range(5, 26)) + [None],
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
        'min_samples_leaf': list(range(1, 51)),
    }

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    base = RandomForestClassifier(
        class_weight='balanced',
        oob_score=True,
        n_jobs=-1,
        random_state=random_state,
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    cv_scores = cross_val_score(best, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)

    meta = {
        'best_params': search.best_params_,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'oob_score': best.oob_score_,
        'train_time_sec': time.time() - t0,
    }
    return best, meta


# ─── XGBoost ──────────────────────────────────────────────────────────────────

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    n_iter: int = 15,
    random_state: int = RANDOM_STATE,
) -> Tuple[Any, Dict]:
    """
    Huấn luyện XGBoost với scale_pos_weight cho dữ liệu mất cân bằng lớp.

    Cơ sở toán học:
        Mô hình cộng dần: F_t(x) = F_{t-1}(x) + η·f_t(x)
        Hàm mục tiêu: Σᵢ l(yᵢ, ŷᵢ) + Ω(f_t)
          l = log-loss cho phân loại nhị phân
          Ω(f) = γT + (λ/2)||w||²  (T = số lá, w = trọng số lá)
        Khai triển Taylor bậc 2 → nghiệm dạng đóng cho trọng số lá:
          w*_j = - Σ gᵢ / (Σ hᵢ + λ)
          gain = (1/2)[G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ

    scale_pos_weight = n_negative / n_positive (13.96 cho dataset này)
    → Tương đương gán trọng số mỗi mẫu dương gấp 13.96× mẫu âm trong gradient hàm mất mát

    Tìm kiếm siêu tham số:
        - n_estimators ∈ [100, 500]
        - max_depth ∈ [3, 8]
        - learning_rate ∈ [0.01, 0.3]
        - subsample ∈ [0.6, 1.0]
        - colsample_bytree ∈ [0.6, 1.0]
        - reg_lambda (L2) ∈ [0.1, 10]
        - reg_alpha (L1) ∈ [0, 5]

    Returns
    -------
    model : xgb.XGBClassifier
    meta : dict
    """
    t0 = time.time()

    spw = (y_train == 0).sum() / (y_train == 1).sum()

    param_dist = {
        'n_estimators': [100, 150, 200, 300, 400, 500],
        'max_depth': list(range(3, 9)),
        'learning_rate': np.logspace(-2, np.log10(0.3), 50),
        'subsample': np.arange(0.6, 1.01, 0.05),
        'colsample_bytree': np.arange(0.6, 1.01, 0.05),
        'reg_lambda': np.logspace(-1, 1, 30),
        'reg_alpha': np.linspace(0, 5, 20),
    }

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    base = xgb.XGBClassifier(
        scale_pos_weight=spw,
        tree_method='hist',
        eval_metric='auc',
        random_state=random_state,
        n_jobs=-1,
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    cv_scores = cross_val_score(best, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)

    meta = {
        'best_params': search.best_params_,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'scale_pos_weight': spw,
        'train_time_sec': time.time() - t0,
    }
    return best, meta


# ─── Lưu / Tải mô hình ───────────────────────────────────────────────────────

def save_model(model: Any, name: str) -> Path:
    """
    Lưu mô hình vào models/{name}.pkl.

    Parameters
    ----------
    model : mô hình sklearn/xgb đã khớp (fitted)
    name : str
        Tên file (không có phần mở rộng). Ví dụ: 'best_model', 'model_lr'.

    Returns
    -------
    Path
        Đường dẫn file đã lưu.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    return path


def load_model(name: str) -> Any:
    """
    Tải mô hình từ models/{name}.pkl.

    Parameters
    ----------
    name : str
        'best_model', 'model_lr', 'model_dt', 'model_rf', 'model_xgb'.
    """
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model không tồn tại: {path}")
    return joblib.load(path)
