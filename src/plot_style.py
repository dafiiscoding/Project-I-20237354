"""
Vietnamese plotting style setup for matplotlib + seaborn.
Centralizes font configuration, DPI settings, and Vietnamese label mappings.
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import Dict

# Font configuration cho Vietnamese characters
def setup_vietnamese_style(dpi: int = 150) -> None:
    """
    Setup matplotlib style cho tiếng Việt.

    Args:
        dpi: DPI cho figures (150 cho báo cáo, 110 cho slide)
    """
    # Detect platform và set font accordingly
    try:
        # Windows: Segoe UI, Tahoma có glyph Việt
        plt.rcParams['font.family'] = 'Segoe UI'
    except:
        try:
            plt.rcParams['font.family'] = 'DejaVu Sans'
        except:
            plt.rcParams['font.family'] = 'sans-serif'

    # Style settings
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5

    # Font sizes
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

    # Seaborn style
    sns.set_palette("husl")

    # Test glyph rendering
    test_text = "Vỡ nợ — Đặc trưng — Kiểm định"
    try:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, test_text, ha='center', va='center', fontsize=14)
        ax.axis('off')
        # Nếu render được thì close và return, không save
        plt.close(fig)
        return
    except Exception as e:
        print(f"⚠️ Font test failed: {e}. Trying fallback font...")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.close('all')


# Vietnamese labels mapping (English → Việt)
LABELS_VI: Dict[str, str] = {
    # Metrics
    'False Positive Rate': 'Tỷ lệ dương tính giả (1-Specificity)',
    'True Positive Rate': 'Tỷ lệ dương tính thật (Recall)',
    'Precision': 'Độ chính xác (Precision)',
    'Recall': 'Nhạy cảm (Recall)',
    'F1-Score': 'Điểm F1',
    'F2-Score': 'Điểm F2',
    'AUC': 'Diện tích dưới đường cong (AUC)',
    'ROC Curve': 'Đường cong ROC',
    'PR Curve': 'Đường cong Precision-Recall',
    'Confusion Matrix': 'Ma trận nhầm lẫn',

    # Predictions
    'Predicted': 'Dự báo',
    'No Default': 'Không vỡ nợ',
    'Default': 'Vỡ nợ',
    'True Label': 'Nhãn thực tế',
    'Predicted Label': 'Nhãn dự báo',

    # Statistical terms
    'Mean predicted probability': 'Xác suất dự báo trung bình',
    'Fraction of positives': 'Tỷ lệ vỡ nợ thực tế',
    'Reliability Diagram': 'Biểu đồ độ tin cậy',
    'Calibration Quality': 'Chất lượng hiệu chỉnh xác suất',
    'Brier Score': 'Brier Score',
    'Expected Calibration Error': 'Sai số hiệu chỉnh mong đợi (ECE)',

    # Model names (keep English)
    'Logistic Regression': 'Logistic Regression',
    'Decision Tree': 'Decision Tree',
    'Random Forest': 'Random Forest',
    'XGBoost': 'XGBoost',

    # Plot titles
    'Model Comparison': 'So sánh các mô hình',
    'ROC Curves — Model Comparison': 'So sánh đường cong ROC giữa các mô hình',
    'Precision-Recall Curves — Model Comparison': 'So sánh đường cong Precision-Recall giữa các mô hình',
    'Feature Importance': 'Tầm quan trọng đặc trưng',
    'Learning Curves': 'Đường cong học tập',
    'Confusion Matrix': 'Ma trận nhầm lẫn',

    # Axes labels
    'Number of Training Samples': 'Số lượng mẫu huấn luyện',
    'Score': 'Điểm số',
    'Probability': 'Xác suất',
    'Age': 'Tuổi',
    'Income': 'Thu nhập',
    'Debt Ratio': 'Tỷ lệ nợ',
    'Count': 'Số lượng',

    # Feature names
    'RevolvingUtilizationOfUnsecuredLines': 'Tỷ lệ sử dụng hạn mức tín dụng',
    'Age': 'Tuổi',
    'NumberOfTimes30-59DaysPastDueNotWorse': 'Số lần trễ 30-59 ngày',
    'DebtRatio': 'Tỷ lệ nợ',
    'MonthlyIncome': 'Thu nhập hàng tháng',
    'NumberOfOpenCreditLinesAndLoans': 'Số dòng tín dụng mở',
    'NumberOfTimes90DaysPastDue': 'Số lần trễ 90+ ngày',
    'NumberRealEstateLoansOrLines': 'Số khoản vay bất động sản',
    'NumberOfTimes60-89DaysPastDueNotWorse': 'Số lần trễ 60-89 ngày',
    'NumberOfDependents': 'Số người phụ thuộc',

    # Feature engineered
    'TotalDelinquencyScore': 'Điểm vỡ nợ tổng hợp',
    'FinancialStressIndex': 'Chỉ số áp lực tài chính',
    'AbsoluteMonthlyDebt': 'Khoản nợ hàng tháng tuyệt đối',
    'DelinquencyTrend': 'Xu hướng vỡ nợ',
}

# Model palette (colors for LR, DT, RF, XGB)
PALETTE_MODELS = {
    'Logistic Regression': '#1f77b4',  # blue
    'Decision Tree': '#ff7f0e',        # orange
    'Random Forest': '#2ca02c',        # green
    'XGBoost': '#d62728',              # red
}

# Risk tier colors
RISK_COLORS = {
    'LOW': '#2ecc71',           # green
    'MEDIUM': '#f39c12',        # orange
    'HIGH': '#e74c3c',          # red
    'VERY_HIGH': '#c0392b',     # dark red
}


def translate_label(text: str, default: str = None) -> str:
    """
    Translate English label to Vietnamese if available.

    Args:
        text: English label to translate
        default: Default if not found (use original text if None)

    Returns:
        Vietnamese label or original text if no mapping exists
    """
    return LABELS_VI.get(text, default if default is not None else text)
