"""Tiện ích dùng chung cho Streamlit app."""

import io
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from pathlib import Path
from typing import Tuple, List

# ─── Hằng số ────────────────────────────────────────────────────────────────

THRESHOLD = 0.625

REQUIRED_COLS = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents',
]

FEATURES = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents',
    'TotalDelinquencyScore',
    'FinancialStressIndex',
    'AbsoluteMonthlyDebt',
    'DelinquencyTrend',
]

FEATURE_LABELS = {
    'RevolvingUtilizationOfUnsecuredLines': 'Tỷ lệ sử dụng hạn mức tín dụng',
    'age': 'Tuổi',
    'NumberOfTime30-59DaysPastDueNotWorse': 'Số lần trễ 30-59 ngày',
    'DebtRatio': 'Tỷ lệ nợ/thu nhập',
    'MonthlyIncome': 'Thu nhập hàng tháng',
    'NumberOfOpenCreditLinesAndLoans': 'Số tài khoản tín dụng đang mở',
    'NumberOfTimes90DaysLate': 'Số lần trễ >90 ngày',
    'NumberRealEstateLoansOrLines': 'Số khoản vay bất động sản',
    'NumberOfTime60-89DaysPastDueNotWorse': 'Số lần trễ 60-89 ngày',
    'NumberOfDependents': 'Số người phụ thuộc',
    'TotalDelinquencyScore': 'Điểm lịch sử trả nợ tổng hợp',
    'FinancialStressIndex': 'Chỉ số stress tài chính',
    'AbsoluteMonthlyDebt': 'Tổng nợ tuyệt đối (USD/tháng)',
    'DelinquencyTrend': 'Xu hướng cải thiện nợ',
}

RISK_COLORS = {
    'LOW': '#2ecc71',
    'MEDIUM': '#f39c12',
    'HIGH': '#e67e22',
    'VERY_HIGH': '#e74c3c',
}

RISK_LABELS_VI = {
    'LOW': 'THẤP',
    'MEDIUM': 'TRUNG BÌNH',
    'HIGH': 'CAO',
    'VERY_HIGH': 'RẤT CAO',
}

TARGET_COL = 'SeriousDlqin2yrs'


# ─── Load model ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Tải mô hình XGBoost từ file pkl."""
    model_path = Path('models/best_model.pkl')
    if not model_path.exists():
        st.error(f"Không tìm thấy file model: {model_path}")
        st.stop()
    return joblib.load(model_path)


@st.cache_resource
def load_explainer(_model):
    """Tải SHAP TreeExplainer (cached). Tiền tố _ để Streamlit không hash model."""
    import shap
    return shap.TreeExplainer(_model)


# ─── Risk tier ──────────────────────────────────────────────────────────────

def get_risk_tier(prob: float) -> Tuple[str, str, str]:
    """Phân loại risk tier. Trả về (tier_key, label_vi, color)."""
    if prob < 0.10:
        return 'LOW', f"🟢 Rủi ro {RISK_LABELS_VI['LOW']}", RISK_COLORS['LOW']
    elif prob < 0.30:
        return 'MEDIUM', f"🟡 Rủi ro {RISK_LABELS_VI['MEDIUM']}", RISK_COLORS['MEDIUM']
    elif prob < THRESHOLD:
        return 'HIGH', f"🟠 Rủi ro {RISK_LABELS_VI['HIGH']}", RISK_COLORS['HIGH']
    else:
        return 'VERY_HIGH', f"🔴 Rủi ro {RISK_LABELS_VI['VERY_HIGH']}", RISK_COLORS['VERY_HIGH']


# ─── Validation ─────────────────────────────────────────────────────────────

def validate_batch_input(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Kiểm tra DataFrame upload có đủ cột bắt buộc không.
    Trả về (ok, list_errors).
    """
    errors = []
    if len(df) == 0:
        errors.append("File không có dữ liệu (0 dòng).")
        return False, errors

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        errors.append(f"Thiếu {len(missing)} cột bắt buộc: {', '.join(missing)}")

    if len(df) > 500_000:
        errors.append("File quá lớn (> 500,000 dòng). Vui lòng chia nhỏ file.")

    return len(errors) == 0, errors


# ─── Feature engineering ────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính 4 engineered features và trả về DataFrame với đúng thứ tự cột.
    Xử lý NaN: impute median cho MonthlyIncome và NumberOfDependents.
    Loại bỏ dòng có age == 0.
    """
    df = df.copy()

    # Drop unnamed index col nếu có
    df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')], errors='ignore')

    # Coerce dtypes
    for col in REQUIRED_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop age == 0
    n_before = len(df)
    df = df[df['age'] != 0]
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        st.warning(f"Đã bỏ {n_dropped} dòng có tuổi = 0.")

    # Impute median cho MonthlyIncome, NumberOfDependents
    for col in ['MonthlyIncome', 'NumberOfDependents']:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            n_nan = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            st.warning(f"Đã điền {n_nan} giá trị thiếu trong cột '{col}' bằng trung vị ({median_val:.1f}).")

    # Check remaining NaN
    nan_cols = [c for c in REQUIRED_COLS if c in df.columns and df[c].isna().any()]
    if nan_cols:
        st.error(f"Vẫn còn giá trị thiếu sau xử lý trong cột: {', '.join(nan_cols)}. Vui lòng kiểm tra dữ liệu đầu vào.")
        st.stop()

    # Tính engineered features
    df['TotalDelinquencyScore'] = (
        df['NumberOfTimes90DaysLate'] * 3
        + df['NumberOfTime60-89DaysPastDueNotWorse'] * 2
        + df['NumberOfTime30-59DaysPastDueNotWorse'] * 1
    )
    df['FinancialStressIndex'] = df['RevolvingUtilizationOfUnsecuredLines'] * df['TotalDelinquencyScore']
    df['AbsoluteMonthlyDebt'] = df['DebtRatio'] * df['MonthlyIncome']
    df['DelinquencyTrend'] = df['NumberOfTime30-59DaysPastDueNotWorse'] - df['NumberOfTimes90DaysLate']

    return df[FEATURES]


# ─── Batch predict ───────────────────────────────────────────────────────────

def batch_predict(model, X: pd.DataFrame, chunk_size: int = 5000) -> np.ndarray:
    """
    Predict xác suất vỡ nợ theo lô. Dùng progress bar nếu > 10k dòng.
    """
    n = len(X)
    if n <= chunk_size:
        return model.predict_proba(X)[:, 1]

    probs = []
    progress = st.progress(0, text="Đang tính toán...")
    for i in range(0, n, chunk_size):
        chunk = X.iloc[i:i + chunk_size]
        probs.append(model.predict_proba(chunk)[:, 1])
        progress.progress(min((i + chunk_size) / n, 1.0), text=f"Đang xử lý {min(i + chunk_size, n):,}/{n:,} dòng...")
    progress.empty()
    return np.concatenate(probs)


# ─── CSV template ────────────────────────────────────────────────────────────

def make_template_csv() -> bytes:
    """Tạo CSV mẫu với 5 dòng giả để user biết định dạng yêu cầu."""
    sample = pd.DataFrame({
        'RevolvingUtilizationOfUnsecuredLines': [0.3, 0.8, 0.1, 0.5, 0.95],
        'age': [45, 30, 55, 40, 25],
        'NumberOfTime30-59DaysPastDueNotWorse': [0, 2, 0, 1, 3],
        'DebtRatio': [0.35, 0.6, 0.2, 0.45, 0.9],
        'MonthlyIncome': [5000, 3000, 8000, 4500, 2000],
        'NumberOfOpenCreditLinesAndLoans': [5, 8, 3, 6, 10],
        'NumberOfTimes90DaysLate': [0, 1, 0, 0, 2],
        'NumberRealEstateLoansOrLines': [1, 0, 2, 1, 0],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 1, 0, 0, 1],
        'NumberOfDependents': [0, 2, 1, 3, 0],
    })
    return sample.to_csv(index=False).encode('utf-8')
