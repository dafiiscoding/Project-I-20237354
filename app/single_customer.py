"""Tab 1: Dự báo khách hàng đơn lẻ."""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from .utils import FEATURES, FEATURE_LABELS, THRESHOLD, load_model, load_explainer, get_risk_tier
    from .viz import plot_shap_waterfall
except ImportError:
    from utils import FEATURES, FEATURE_LABELS, THRESHOLD, load_model, load_explainer, get_risk_tier
    from viz import plot_shap_waterfall


def _build_dataframe(
    revolving_util, age, n30_59, debt_ratio, monthly_income,
    n_open, n90, n_realestate, n60_89, n_dependents
) -> pd.DataFrame:
    """Xây dựng DataFrame 1 dòng với đầy đủ features."""
    total_delinq = n90 * 3 + n60_89 * 2 + n30_59 * 1
    financial_stress = revolving_util * total_delinq
    dti = debt_ratio * monthly_income
    delinq_trend = n30_59 - n90

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
        'AbsoluteMonthlyDebt': dti,
        'DelinquencyTrend': delinq_trend,
    }
    return pd.DataFrame([data])[FEATURES]


def render(model) -> None:
    """Render Tab 1: nhập liệu + dự báo đơn lẻ."""
    col_input, col_output = st.columns([1, 1.2], gap="large")

    # ── Cột nhập liệu ──
    with col_input:
        st.subheader("📋 Thông tin khách hàng")
        col_a, col_b = st.columns(2, gap="small")

        with col_a:
            revolving_util = st.number_input(
                "Tỷ lệ sử dụng hạn mức",
                min_value=0.0, max_value=10.0, value=0.3, step=0.01,
                help="Tỷ lệ tín dụng không đảm bảo đã sử dụng (0–1 bình thường; >1 vượt hạn mức)"
            )
            age = st.number_input("Tuổi", min_value=18, max_value=100, value=45, step=1)
            n30_59 = st.number_input("Số lần trễ 30-59 ngày", min_value=0, max_value=20, value=0, step=1)
            debt_ratio = st.number_input(
                "Tỷ lệ nợ/thu nhập",
                min_value=0.0, max_value=5.0, value=0.35, step=0.01,
                help="Tổng nợ / thu nhập hàng tháng"
            )
            monthly_income = st.number_input("Thu nhập hàng tháng ($)", min_value=0, max_value=50000, value=5000, step=100)

        with col_b:
            n_open = st.number_input("Số dòng tín dụng đang mở", min_value=0, max_value=50, value=5, step=1)
            n90 = st.number_input("Số lần trễ >90 ngày", min_value=0, max_value=20, value=0, step=1)
            n_realestate = st.number_input("Số khoản vay bất động sản", min_value=0, max_value=20, value=1, step=1)
            n60_89 = st.number_input("Số lần trễ 60-89 ngày", min_value=0, max_value=20, value=0, step=1)
            n_dependents = st.number_input("Số người phụ thuộc", min_value=0, max_value=20, value=0, step=1)

        st.markdown("---")
        predict_btn = st.button("🔍 Dự báo rủi ro", use_container_width=True, type="primary")

    # ── Cột kết quả ──
    with col_output:
        if predict_btn:
            X_df = _build_dataframe(
                revolving_util, age, n30_59, debt_ratio, monthly_income,
                n_open, n90, n_realestate, n60_89, n_dependents
            )
            prob = model.predict_proba(X_df)[0, 1]
            risk_key, risk_label, risk_color = get_risk_tier(prob)
            decision = "🔴 TỪ CHỐI" if prob >= THRESHOLD else "✅ DUYỆT"

            # Banner rủi ro
            st.markdown(
                f"<div style='text-align:center; padding:20px; border-radius:8px; "
                f"background:{risk_color}; color:white;'>"
                f"<h2 style='margin:0'>{risk_label}</h2>"
                f"<p style='margin:8px 0 0; font-size:18px;'>P(Vỡ nợ) = <b>{prob:.1%}</b></p>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.markdown("")

            m1, m2, m3 = st.columns(3)
            m1.metric("Xác suất vỡ nợ", f"{prob:.1%}")
            m2.metric("Quyết định (t=0.625)", decision)
            m3.metric("Điểm số", f"{prob:.3f}")

            st.markdown("---")
            st.subheader("🔑 Top 3 yếu tố rủi ro chính")

            explainer = load_explainer(model)
            explanation = explainer(X_df)
            shap_vals = explanation.values[0]
            top3_idx = np.argsort(np.abs(shap_vals))[::-1][:3]

            for rank, idx in enumerate(top3_idx, 1):
                feat_name = FEATURES[idx]
                feat_label = FEATURE_LABELS.get(feat_name, feat_name)
                feat_val = X_df.iloc[0, idx]
                direction = "tăng" if shap_vals[idx] > 0 else "giảm"
                st.write(f"**{rank}. {feat_label}** = `{feat_val:.3g}` ({direction} rủi ro vỡ nợ)")

            st.markdown("---")
            st.subheader("📊 Giải thích chi tiết (SHAP)")
            st.caption("Đỏ = tăng rủi ro vỡ nợ, Xanh = giảm rủi ro vỡ nợ")

            fig = plot_shap_waterfall(X_df, explanation, prob)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        else:
            st.info("👈 Nhập thông tin khách hàng rồi nhấn **Dự báo rủi ro** để bắt đầu")
            st.markdown("---")
            st.subheader("📖 Giới thiệu ứng dụng")
            st.markdown("""
            - **Mô hình**: XGBoost huấn luyện trên 104.999 hồ sơ (train split của bộ dữ liệu 149.999 hồ sơ)
            - **Đặc trưng**: 10 đặc trưng gốc + 4 đặc trưng kỹ thuật (tự động tính)
            - **Chỉ số**: AUC-ROC = 0,8714 (tập kiểm tra)
            - **Ngưỡng**: 0,625 — duyệt nếu P(vỡ nợ) < 0,625
            - **Top features**: Tỷ lệ sử dụng tín dụng, lịch sử trễ hạn, tuổi
            """)
