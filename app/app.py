"""
Bảng điều khiển Streamlit: Dự báo Rủi ro Vỡ nợ Tín dụng
=========================================================
Ứng dụng dự báo xác suất vỡ nợ tín dụng bằng mô hình XGBoost.
- Tải mô hình từ models/best_model.pkl
- 10 đặc trưng đầu vào do người dùng nhập
- 4 đặc trưng kỹ thuật (engineered features) được tính tự động
- Trực quan hóa SHAP waterfall
- Phân loại mức độ rủi ro (risk tier)

Chạy: streamlit run app/app.py (từ thư mục gốc project)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CẤU HÌNH & HẰNG SỐ
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Feature names (order PHẢI đúng)
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
    'DebtToIncomeRatio',
    'DelinquencyTrend',
]

# Feature labels cho tiếng Việt
FEATURE_LABELS = {
    'RevolvingUtilizationOfUnsecuredLines': 'Tỷ lệ sử dụng hạn mức tín dụng',
    'age': 'Tuổi',
    'NumberOfTime30-59DaysPastDueNotWorse': 'Số lần trễ 30–59 ngày',
    'DebtRatio': 'Tỷ lệ nợ/thu nhập',
    'MonthlyIncome': 'Thu nhập hàng tháng',
    'NumberOfOpenCreditLinesAndLoans': 'Số tài khoản tín dụng đang mở',
    'NumberOfTimes90DaysLate': 'Số lần trễ >90 ngày',
    'NumberRealEstateLoansOrLines': 'Số khoản vay bất động sản',
    'NumberOfTime60-89DaysPastDueNotWorse': 'Số lần trễ 60–89 ngày',
    'NumberOfDependents': 'Số người phụ thuộc',
    'TotalDelinquencyScore': 'Điểm lịch sử trả nợ tổng hợp',
    'FinancialStressIndex': 'Chỉ số stress tài chính',
    'DebtToIncomeRatio': 'Tổng nợ tuyệt đối',
    'DelinquencyTrend': 'Xu hướng cải thiện nợ',
}

# Deployment threshold
THRESHOLD = 0.625

# Risk tier colors
RISK_COLORS = {
    'LOW': '#2ecc71',
    'MEDIUM': '#f39c12',
    'HIGH': '#e67e22',
    'VERY_HIGH': '#e74c3c',
}

RISK_LABELS = {
    'LOW': '🟢 LOW RISK',
    'MEDIUM': '🟡 MEDIUM RISK',
    'HIGH': '🟠 HIGH RISK',
    'VERY_HIGH': '🔴 VERY HIGH RISK',
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HÀM CACHE (CACHED FUNCTIONS)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Tải mô hình XGBoost từ file pickle"""
    model_path = Path('models/best_model.pkl')
    if not model_path.exists():
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    return joblib.load(model_path)


@st.cache_resource
def load_explainer(_model):
    """Tải SHAP TreeExplainer (có cache).

    Dùng tiền tố `_` để Streamlit không cố hash XGBoost model khi cache.
    """
    return shap.TreeExplainer(_model)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HÀM TIỆN ÍCH
# ═══════════════════════════════════════════════════════════════════════════════

def get_risk_tier(prob):
    """Phân loại risk tier dựa trên xác suất"""
    if prob < 0.10:
        return 'LOW', RISK_LABELS['LOW'], RISK_COLORS['LOW']
    elif prob < 0.30:
        return 'MEDIUM', RISK_LABELS['MEDIUM'], RISK_COLORS['MEDIUM']
    elif prob < THRESHOLD:
        return 'HIGH', RISK_LABELS['HIGH'], RISK_COLORS['HIGH']
    else:
        return 'VERY_HIGH', RISK_LABELS['VERY_HIGH'], RISK_COLORS['VERY_HIGH']


def compute_engineered_features(
    revolving_util, n30_59, n60_89, n90,
    debt_ratio, monthly_income
):
    """Tính 4 engineered features từ raw features"""
    total_delinq = n90 * 3 + n60_89 * 2 + n30_59 * 1
    financial_stress = revolving_util * total_delinq
    dti = debt_ratio * monthly_income
    delinq_trend = n30_59 - n90
    
    return total_delinq, financial_stress, dti, delinq_trend


def build_prediction_dataframe(
    revolving_util, age, n30_59, debt_ratio, monthly_income,
    n_open, n90, n_realestate, n60_89, n_dependents
):
    """Xây dựng DataFrame với đúng thứ tự features"""
    total_delinq, financial_stress, dti, delinq_trend = compute_engineered_features(
        revolving_util, n30_59, n60_89, n90, debt_ratio, monthly_income
    )
    
    data = {
        'RevolvingUtilizationOfUnsecuredLines': revolving_util,
        'age': age,
        'NumberOfTime30-59DaysPastDueNotWorse': n30_59,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': monthly_income,
        'NumberOfOpenCreditLinesAndLoans': n_open,
        'NumberOfTimes90DaysLate': n90,
        'NumberRealEstateLoansOrLines': n_realestate,
        'NumberOfTime60-89DaysPastDueNotWorse': n60_89,
        'NumberOfDependents': n_dependents,
        'TotalDelinquencyScore': total_delinq,
        'FinancialStressIndex': financial_stress,
        'DebtToIncomeRatio': dti,
        'DelinquencyTrend': delinq_trend,
    }
    
    return pd.DataFrame([data])[FEATURES]


def plot_shap_waterfall(model, X_df, explanation, prob):
    """Vẽ biểu đồ SHAP waterfall (dạng bar chart thủ công)"""
    shap_vals = explanation.values[0]  # shape (14,)
    base_val = explanation.base_values[0]

    # Top 10 features theo |SHAP|
    order = np.argsort(np.abs(shap_vals))[::-1][:10]
    top_feat = [FEATURES[i] for i in order]
    top_shap = shap_vals[order]
    top_fval = [f"{X_df.iloc[0, i]:.3g}" for i in order]
    
    # Tạo figure
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in top_shap]
    y_pos = np.arange(len(top_feat))[::-1]

    ax.barh(y_pos, top_shap, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax.set_yticks(y_pos)

    # Định dạng nhãn: tên feature = giá trị
    labels = [f"{FEATURE_LABELS.get(f, f)[:26]} = {v}"
              for f, v in zip(top_feat, top_fval)]
    ax.set_yticklabels(labels, fontsize=9)
    
    ax.axvline(0, color='black', linewidth=1, linestyle='-')
    ax.set_xlabel('SHAP value  (red = ↑ risk, green = ↓ risk)', fontsize=10, fontweight='bold')
    
    base_prob = 1 / (1 + np.exp(-base_val)) if base_val is not None else 0.067
    title = f'Why Default? | Base prob: {base_prob:.1%} → Final: {prob:.1%}'
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
    
    # Chú giải
    pos_patch = mpatches.Patch(color='#e74c3c', label='↑ Increase default risk')
    neg_patch = mpatches.Patch(color='#2ecc71', label='↓ Decrease default risk')
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9, loc='lower right')
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ỨNG DỤNG CHÍNH
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Tiêu đề
    st.title("🏦 Loan Default Risk Predictor")
    st.caption("Model: XGBoost | AUC=0.8714 | Deployment Threshold=0.625")

    # Tải mô hình một lần
    model = load_model()

    # Bố cục chính: 2 cột
    col_input, col_output = st.columns([1, 1.2], gap="large")

    # ━━━ CỘT NHẬP LIỆU ━━━
    with col_input:
        st.subheader("📋 Customer Information")
        
        # Form nhập liệu (2 cột con để gọn bố cục)
        col_a, col_b = st.columns(2, gap="small")
        
        with col_a:
            revolving_util = st.number_input(
                "Revolving Utilization",
                min_value=0.0, max_value=1.0, value=0.3, step=0.01,
                help="Ratio of unsecured credit used (0-1)"
            )
            age = st.number_input(
                "Age (years)",
                min_value=18, max_value=100, value=45, step=1
            )
            n30_59 = st.number_input(
                "Times 30-59 days late",
                min_value=0, max_value=20, value=0, step=1
            )
            debt_ratio = st.number_input(
                "Debt Ratio",
                min_value=0.0, max_value=5.0, value=0.35, step=0.01,
                help="Total debt / monthly income"
            )
            monthly_income = st.number_input(
                "Monthly Income ($)",
                min_value=0, max_value=50000, value=5000, step=100
            )
        
        with col_b:
            n_open = st.number_input(
                "Open credit lines",
                min_value=0, max_value=50, value=5, step=1
            )
            n90 = st.number_input(
                "Times >90 days late",
                min_value=0, max_value=20, value=0, step=1
            )
            n_realestate = st.number_input(
                "Real estate loans",
                min_value=0, max_value=20, value=1, step=1
            )
            n60_89 = st.number_input(
                "Times 60-89 days late",
                min_value=0, max_value=20, value=0, step=1
            )
            n_dependents = st.number_input(
                "Number of dependents",
                min_value=0, max_value=20, value=0, step=1
            )
        
        st.markdown("---")

        # Nút dự báo
        predict_btn = st.button(
            "🔍 Predict Risk",
            use_container_width=True,
            type="primary",
            key="predict_button"
        )
    
    # ━━━ CỘT KẾT QUẢ ━━━
    with col_output:
        if predict_btn:
            # Xây dựng DataFrame đầu vào
            X_df = build_prediction_dataframe(
                revolving_util, age, n30_59, debt_ratio, monthly_income,
                n_open, n90, n_realestate, n60_89, n_dependents
            )
            
            # Dự báo
            prob = model.predict_proba(X_df)[0, 1]
            risk_key, risk_label, risk_color = get_risk_tier(prob)

            # Quyết định
            decision = "🔴 REJECT" if prob >= THRESHOLD else "✅ APPROVE"
            decision_color = "#e74c3c" if prob >= THRESHOLD else "#2ecc71"

            # Hiển thị mức độ rủi ro (lớn, căn giữa)
            st.markdown(
                f"<div style='text-align: center; "
                f"padding: 20px; border-radius: 8px; "
                f"background-color: {risk_color}; color: white;'>"
                f"<h2 style='margin: 0;'>{risk_label}</h2>"
                f"<p style='margin: 8px 0 0 0; font-size: 18px;'>P(Default) = <b>{prob:.1%}</b></p>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            st.markdown("")

            # Hàng chỉ số đánh giá
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Default Probability", f"{prob:.1%}")
            with m_col2:
                st.metric("Decision (t=0.625)", decision, delta=None)
            with m_col3:
                st.metric("Score", f"{prob:.3f}")
            
            st.markdown("---")

            # Top 3 yếu tố rủi ro
            st.subheader("🔑 Top Risk Factors")

            # Tải explainer và lấy giá trị SHAP
            explainer = load_explainer(model)
            explanation = explainer(X_df)
            shap_vals = explanation.values[0]

            # Lấy top 3
            top3_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
            
            for rank, idx in enumerate(top3_idx, 1):
                feat_name = FEATURES[idx]
                feat_label = FEATURE_LABELS.get(feat_name, feat_name)
                feat_val = X_df.iloc[0, idx]
                shap_val = shap_vals[idx]
                direction = "↑ increases" if shap_val > 0 else "↓ decreases"
                
                st.write(
                    f"**{rank}. {feat_label}** = `{feat_val:.3g}` "
                    f"({direction} default risk)"
                )
            
            st.markdown("---")

            # SHAP Waterfall
            st.subheader("📊 Detailed Explanation (SHAP)")
            st.markdown(
                "Tại sao mô hình đưa ra dự báo này? "
                "Đóng góp của từng feature (đỏ = tăng rủi ro, xanh = giảm rủi ro):"
            )

            fig = plot_shap_waterfall(model, X_df, explanation, prob)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)  # QUAN TRỌNG: tránh memory leak
            
        else:
            st.info("👈 Nhập thông tin khách hàng và nhấn **Predict Risk** để bắt đầu")
            st.markdown("---")
            st.subheader("📖 Giới thiệu ứng dụng")
            st.markdown("""
            - **Mô hình**: XGBoost huấn luyện trên 150.000 hồ sơ vay
            - **Đặc trưng**: 10 đặc trưng gốc + 4 đặc trưng kỹ thuật (tự động tính)
            - **Chỉ số**: AUC-ROC = 0,8714 (tập kiểm tra)
            - **Ngưỡng**: 0,625 — duyệt nếu P(vỡ nợ) < 0,625
            - **Top features**: Tỷ lệ sử dụng tín dụng, lịch sử trễ hạn, tuổi, tỷ lệ nợ, thu nhập
            """)


if __name__ == "__main__":
    main()
