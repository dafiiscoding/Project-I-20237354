"""
features.py — Feature engineering functions cho Loan Default Prediction

Mục đích: Tạo 4 composite features có cơ sở từ credit risk theory và FICO methodology.
Mọi feature mới đều có lý do tài chính cụ thể — không phải arbitrary transformations.

Tham khảo:
- FICO score methodology (delinquency severity tiers)
- Basel III credit risk framework
- Standard credit analyst practice (DTI calculation)
"""

import numpy as np
import pandas as pd


# ─── Hằng số ─────────────────────────────────────────────────────────────────

# Trọng số mức độ nghiêm trọng trả nợ trễ — theo thứ bậc trong FICO score
# 90+ ngày: vi phạm nặng nhất, 60-89: nghiêm trọng, 30-59: nhẹ
DELINQUENCY_WEIGHT_90PLUS  = 3  # Vi phạm nặng: lưu trong credit report 7 năm
DELINQUENCY_WEIGHT_60TO89  = 2  # Vi phạm nghiêm trọng: ngưỡng chuyển công ty thu nợ
DELINQUENCY_WEIGHT_30TO59  = 1  # Vi phạm nhẹ: tín hiệu đầu tiên của việc trễ thanh toán


# ─── Hàm Tạo Đặc trưng (Feature Engineering) ────────────────────────────────

def compute_total_delinquency_score(df: pd.DataFrame) -> pd.Series:
    """
    Tổng hợp 3 delinquency features thành 1 weighted score.

    TotalDelinquencyScore = 3×(90+) + 2×(60-89) + 1×(30-59)

    Lý do trọng số:
    - Phản ánh severity hierarchy trong FICO scoring methodology
    - Trễ 90+ ngày tăng default rate 6-7× baseline (từ EDA: 35-40% vs 6.68%)
    - Trễ 30-59 ngày tăng default rate 3× baseline (18-22%)
    - Trọng số 3:2:1 là linear approximation của escalation này

    Tác dụng phụ: giải quyết multicollinearity giữa 3 delinquency features
    (thay vì đưa 3 correlated features vào Logistic Regression riêng lẻ)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chứa 3 delinquency columns.

    Returns
    -------
    pd.Series
        Weighted delinquency score (integer ≥ 0).
    """
    required = [
        'NumberOfTimes90DaysLate',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfTime30-59DaysPastDueNotWorse',
    ]
    _check_columns(df, required)

    return (
        df['NumberOfTimes90DaysLate'] * DELINQUENCY_WEIGHT_90PLUS +
        df['NumberOfTime60-89DaysPastDueNotWorse'] * DELINQUENCY_WEIGHT_60TO89 +
        df['NumberOfTime30-59DaysPastDueNotWorse'] * DELINQUENCY_WEIGHT_30TO59
    )


def compute_financial_stress_index(df: pd.DataFrame) -> pd.Series:
    """
    Index kết hợp credit utilization cao VÀ delinquency history.

    FinancialStressIndex = RevolvingUtilization × TotalDelinquencyScore

    Lý do dùng phép nhân (interaction term):
    - Phép nhân capture "joint stress": chỉ cao khi CẢ HAI đều cao
    - Người maxed out thẻ nhưng trả đúng hạn → FSI ≈ 0 (ít risk hơn)
    - Người có lịch sử trễ nhưng không dùng thẻ → FSI = 0
    - Người vừa maxed out VÀ có delinquency → FSI cao = worst profile
    - Phù hợp với behavioral finance: utilization cao + delinquency = cycle nợ

    Parameters
    ----------
    df : pd.DataFrame
        Phải có RevolvingUtilizationOfUnsecuredLines và
        TotalDelinquencyScore (đã tính trước).

    Returns
    -------
    pd.Series
        FSI ≥ 0 (0 khi không có delinquency).
    """
    _check_columns(df, ['RevolvingUtilizationOfUnsecuredLines', 'TotalDelinquencyScore'])

    return (
        df['RevolvingUtilizationOfUnsecuredLines'] *
        df['TotalDelinquencyScore']
    )


def compute_debt_to_income_ratio(df: pd.DataFrame) -> pd.Series:
    """
    DTI tuyệt đối (absolute debt amount USD/month).

    DebtToIncomeRatio = DebtRatio × MonthlyIncome

    Lý do cần feature này:
    - DebtRatio raw là dimensionless ratio (nợ/income), không có đơn vị USD
    - Cùng DebtRatio=0.5 với income=$2k (nợ $1k/tháng) vs income=$20k (nợ $10k/tháng)
      là 2 situations tài chính hoàn toàn khác nhau
    - DebtToIncomeRatio = số USD nợ thực tế mỗi tháng → meaningful cho credit analysts
    - Chỉ có ý nghĩa sau khi MonthlyIncome đã được imputed (không có missing)

    Parameters
    ----------
    df : pd.DataFrame
        Phải có DebtRatio và MonthlyIncome (không có NaN).

    Returns
    -------
    pd.Series
        Monthly debt amount (USD). Có thể lớn với người income cao, nợ nhiều.
    """
    _check_columns(df, ['DebtRatio', 'MonthlyIncome'])

    return df['DebtRatio'] * df['MonthlyIncome']


def compute_delinquency_trend(df: pd.DataFrame) -> pd.Series:
    """
    Indicator về chiều hướng (trajectory) của financial health.

    DelinquencyTrend = NumberOfTime30-59DaysPastDueNotWorse - NumberOfTimes90DaysLate

    Diễn giải:
    - Âm (nhiều 90+, ít 30-59): đang XẤU DẦN — pattern escalation
    - Dương (nhiều 30-59, ít 90+): pattern CẢI THIỆN hoặc "chỉ miss nhẹ"
    - ≈ 0: không có delinquency, hoặc pattern flat

    Lưu ý: Feature này yếu (Spearman ρ ≈ 0.07 từ EDA) nhưng capture
    trend information mà các features riêng lẻ không có.

    Parameters
    ----------
    df : pd.DataFrame
        Phải có NumberOfTime30-59DaysPastDueNotWorse và NumberOfTimes90DaysLate.

    Returns
    -------
    pd.Series
        Integer, có thể âm.
    """
    _check_columns(df, [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate',
    ])

    return (
        df['NumberOfTime30-59DaysPastDueNotWorse'] -
        df['NumberOfTimes90DaysLate']
    )


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply tất cả 4 feature engineering steps theo đúng thứ tự dependency.

    Thứ tự quan trọng:
    1. TotalDelinquencyScore (cần trước FSI)
    2. FinancialStressIndex (cần TotalDelinquencyScore)
    3. DebtToIncomeRatio (cần MonthlyIncome đã imputed)
    4. DelinquencyTrend (independent)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã qua cleaning và imputation (không có NaN).

    Returns
    -------
    pd.DataFrame
        DataFrame gốc + 4 engineered features.
    """
    df_out = df.copy()

    df_out['TotalDelinquencyScore'] = compute_total_delinquency_score(df_out)
    df_out['FinancialStressIndex']  = compute_financial_stress_index(df_out)
    df_out['DebtToIncomeRatio']     = compute_debt_to_income_ratio(df_out)
    df_out['DelinquencyTrend']      = compute_delinquency_trend(df_out)

    return df_out


# ─── Hàm nội bộ ──────────────────────────────────────────────────────────────

def _check_columns(df: pd.DataFrame, required: list) -> None:
    """Raise ValueError nếu thiếu columns cần thiết."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame thiếu columns: {missing}")
