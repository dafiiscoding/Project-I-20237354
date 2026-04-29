"""
preprocessing.py — sklearn Pipeline cho Loan Default Prediction

Hai pipeline tiền xử lý:
- pipeline_lr   : KNNImputer + RobustScaler  (cho Logistic Regression)
- pipeline_tree : KNNImputer only            (cho Random Forest, XGBoost)

Mọi quyết định thiết kế đều có lý do ghi trong REASONING_LOG.md (R03, R05, R10).
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import joblib

from src.features import engineer_all_features


# ─── Hằng số ─────────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent.parent / "models"

# Ngưỡng capping (phân vị 99) — lấy từ EDA trên tập train
# Đặt cứng (hard-coded) để đảm bảo nhất quán giữa train/val/test
CAP_THRESHOLDS = {
    'RevolvingUtilizationOfUnsecuredLines': None,   # fit từ training data
    'MonthlyIncome': None,
    'DebtRatio': None,
    'NumberOfTimes90DaysLate': None,
    'NumberOfTime30-59DaysPastDueNotWorse': None,
    'NumberOfTime60-89DaysPastDueNotWorse': None,
    'NumberRealEstateLoansOrLines': None,
    'NumberOfOpenCreditLinesAndLoans': None,
}

KNN_NEIGHBORS = 5  # k trong KNN Imputer — đánh đổi: bias vs variance
RANDOM_STATE = 42


# ─── Làm sạch dữ liệu ────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame, cap_quantile: float = 0.99,
               cap_thresholds: dict = None) -> pd.DataFrame:
    """
    Làm sạch dữ liệu: xóa age==0, capping outliers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame thô sau khi load từ cs-training.csv.
    cap_quantile : float
        Phân vị dùng để capping (mặc định 0.99 = phân vị 99).
    cap_thresholds : dict, optional
        Ngưỡng đã tính trước {col: value}. Nếu None, tính từ df.
        Dùng ngưỡng tính trước khi transform val/test để tránh data leakage.

    Returns
    -------
    pd.DataFrame
        DataFrame sau làm sạch (số hàng giảm nếu age==0 bị xóa).
    dict
        Ngưỡng đã dùng {col: value} — lưu lại để áp dụng trên val/test.
    """
    df_out = df.copy()

    # Xóa age==0: lỗi dữ liệu, không thể impute giá trị có ý nghĩa
    n_before = len(df_out)
    df_out = df_out[df_out['age'] > 0].copy()
    n_removed = n_before - len(df_out)

    # Capping tại phân vị 99
    cols_to_cap = [
        'RevolvingUtilizationOfUnsecuredLines', 'MonthlyIncome',
        'DebtRatio', 'NumberOfTimes90DaysLate',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberRealEstateLoansOrLines', 'NumberOfOpenCreditLinesAndLoans',
    ]

    if cap_thresholds is None:
        # Tính ngưỡng từ dữ liệu hiện tại (chỉ dùng trên tập train)
        cap_thresholds = {col: df_out[col].quantile(cap_quantile)
                         for col in cols_to_cap}

    for col, threshold in cap_thresholds.items():
        if col in df_out.columns:
            df_out[col] = df_out[col].clip(upper=threshold)

    return df_out, cap_thresholds


# ─── Điền giá trị khuyết ─────────────────────────────────────────────────────

def impute_missing(df: pd.DataFrame, knn_imputer=None,
                   median_dependents: float = None):
    """
    Điền giá trị khuyết (impute missing values):
    - MonthlyIncome (19.82%): KNN Imputer k=5 (căn cứ bằng chứng MAR, R03)
    - NumberOfDependents (2.62%): Trung vị (gần MCAR, R04)

    Parameters
    ----------
    df : pd.DataFrame
    knn_imputer : KNNImputer hoặc None
        Imputer đã khớp. None = khớp mới trên df (chỉ dùng trên tập train).
    median_dependents : float hoặc None
        Trung vị đã tính trước. None = tính từ df.

    Returns
    -------
    pd.DataFrame, KNNImputer, float
    """
    df_out = df.copy()
    numeric_cols = [c for c in df_out.columns
                    if df_out[c].dtype in ['float64', 'int64']
                    and c != 'SeriousDlqin2yrs']

    if knn_imputer is None:
        knn_imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS, metric='nan_euclidean')
        df_out[numeric_cols] = knn_imputer.fit_transform(df_out[numeric_cols])
    else:
        df_out[numeric_cols] = knn_imputer.transform(df_out[numeric_cols])

    # NumberOfDependents: trung vị (dự phòng nếu KNN không điền đủ)
    if median_dependents is None:
        median_dependents = df['NumberOfDependents'].median()
    df_out['NumberOfDependents'] = df_out['NumberOfDependents'].fillna(median_dependents)

    return df_out, knn_imputer, median_dependents


# ─── Xây dựng Pipeline ───────────────────────────────────────────────────────

def build_lr_pipeline() -> Pipeline:
    """
    Pipeline cho Logistic Regression: điền giá trị khuyết + RobustScaler.

    RobustScaler (trung vị/IQR) thay vì StandardScaler (trung bình/std) vì:
    - Dữ liệu vẫn còn lệch phải (right-skewed) sau capping
    - RobustScaler bền vững với outlier còn sót
    - Tham chiếu: REASONING_LOG.md R10
    """
    return Pipeline([
        ('imputer', KNNImputer(n_neighbors=KNN_NEIGHBORS, metric='nan_euclidean')),
        ('scaler', RobustScaler()),
    ])


def build_tree_pipeline() -> Pipeline:
    """
    Pipeline cho các mô hình dạng cây (RF, XGBoost): chỉ điền giá trị khuyết.

    Mô hình dạng cây không cần chuẩn hóa (phân chia dựa trên ngưỡng giá trị,
    không cần gradient ổn định hay độ đo khoảng cách).
    """
    return Pipeline([
        ('imputer', KNNImputer(n_neighbors=KNN_NEIGHBORS, metric='nan_euclidean')),
    ])


# ─── Lưu / Tải Pipeline ──────────────────────────────────────────────────────

def load_pipeline(name: str = 'pipeline') -> Pipeline:
    """
    Tải pipeline đã lưu từ thư mục models/.

    Parameters
    ----------
    name : str
        'pipeline' (LR), 'pipeline_lr', hoặc 'pipeline_tree'.
    """
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline không tồn tại: {path}. Chạy 02_Preprocessing.ipynb trước."
        )
    return joblib.load(path)
