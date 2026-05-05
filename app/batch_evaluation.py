"""Tab 2: Đánh giá theo lô (CSV upload)."""

import sys
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from .utils import (FEATURES, FEATURE_LABELS, REQUIRED_COLS, TARGET_COL, THRESHOLD,
                        RISK_COLORS, RISK_LABELS_VI, validate_batch_input, prepare_features,
                        batch_predict, get_risk_tier, make_template_csv)
    from .viz import (plot_risk_tier_donut, plot_prob_histogram, plot_segment_analysis,
                      plot_income_segment, plot_shap_global,
                      plot_roc_pr, plot_lift_gain, plot_calibration_batch, compute_ks_stat)
except ImportError:
    from utils import (FEATURES, FEATURE_LABELS, REQUIRED_COLS, TARGET_COL, THRESHOLD,
                       RISK_COLORS, RISK_LABELS_VI, validate_batch_input, prepare_features,
                       batch_predict, get_risk_tier, make_template_csv)
    from viz import (plot_risk_tier_donut, plot_prob_histogram, plot_segment_analysis,
                     plot_income_segment, plot_shap_global,
                     plot_roc_pr, plot_lift_gain, plot_calibration_batch, compute_ks_stat)


def render(model) -> None:
    """Render Tab 2: CSV upload + batch evaluation + export."""

    st.subheader("📊 Đánh giá rủi ro theo lô (CSV)")
    st.caption("Tải lên file CSV chứa danh sách khách hàng để đánh giá hàng loạt.")

    # ── Step 1: Upload ──
    col_upload, col_template = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Tải file CSV (tối đa 500,000 dòng)",
            type=["csv"],
            help=f"File cần chứa đủ {len(REQUIRED_COLS)} cột raw. Tải CSV mẫu bên phải nếu chưa có."
        )
    with col_template:
        st.markdown("&nbsp;")
        st.download_button(
            label="📥 Tải CSV mẫu",
            data=make_template_csv(),
            file_name="mau_danh_sach_khachhang.csv",
            mime="text/csv",
        )

    if not uploaded:
        st.markdown("---")
        _show_column_guide()
        return

    # ── Step 2: Load & Validate ──
    try:
        for enc in ['utf-8', 'latin-1']:
            try:
                df_raw = pd.read_csv(io.StringIO(uploaded.getvalue().decode(enc)))
                break
            except UnicodeDecodeError:
                continue
        else:
            st.error("Không đọc được file. Hãy đảm bảo file được lưu dưới encoding UTF-8 hoặc Latin-1.")
            return
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        return

    ok, errors = validate_batch_input(df_raw)
    if not ok:
        for err in errors:
            st.error(f"❌ {err}")
        st.info("Vui lòng tải CSV mẫu ở trên để xem đúng định dạng.")
        return

    has_target = TARGET_COL in df_raw.columns
    n_rows = len(df_raw)
    st.success(f"✅ Tải thành công: **{n_rows:,} dòng**, {len(df_raw.columns)} cột{' (có cột mục tiêu — sẽ hiển thị kết quả benchmark)' if has_target else ''}.")

    with st.expander("👀 Xem trước dữ liệu (10 dòng đầu)"):
        st.dataframe(df_raw.head(10), use_container_width=True)
        nan_info = df_raw[REQUIRED_COLS].isna().sum()
        nan_info = nan_info[nan_info > 0]
        if not nan_info.empty:
            st.warning("Các cột có giá trị thiếu:\n" + "\n".join(f"- **{c}**: {v} giá trị" for c, v in nan_info.items()))

    # ── Step 3: Run prediction ──
    if st.button("🚀 Đánh giá toàn bộ", type="primary", use_container_width=False):
        with st.spinner("Đang chuẩn bị đặc trưng..."):
            X_feat = prepare_features(df_raw)

        with st.spinner("Đang dự báo..."):
            probs = batch_predict(model, X_feat)

        df_out = df_raw.copy()
        df_out['probability'] = probs
        df_out['risk_tier'] = [get_risk_tier(p)[0] for p in probs]
        df_out['decision'] = np.where(probs >= THRESHOLD, 'REJECT', 'APPROVE')

        st.session_state['batch_result'] = df_out
        st.session_state['batch_raw'] = df_raw
        st.session_state['batch_feat'] = X_feat
        st.session_state['has_target'] = has_target
        st.rerun()

    # ── Step 4: Show results ──
    if 'batch_result' in st.session_state:
        df_out = st.session_state['batch_result']
        df_raw_cached = st.session_state.get('batch_raw', df_raw)
        X_feat_cached = st.session_state.get('batch_feat', None)
        has_target_cached = st.session_state.get('has_target', False)

        st.markdown("---")
        st.markdown("### 📈 Kết quả đánh giá")

        # A. Tổng quan + đánh giá portfolio
        _render_overview(df_out)
        _render_portfolio_assessment(df_out)

        st.markdown("---")

        # B + C: Donut + Histogram
        col_b, col_c = st.columns(2)
        with col_b:
            st.plotly_chart(plot_risk_tier_donut(df_out), use_container_width=True)
            tier_table = df_out['risk_tier'].value_counts().rename({
                'LOW': 'Thấp', 'MEDIUM': 'Trung bình', 'HIGH': 'Cao', 'VERY_HIGH': 'Rất cao'
            })
            st.dataframe(
                tier_table.rename_axis("Mức rủi ro").reset_index(name="Số lượng"),
                use_container_width=True, hide_index=True
            )
        with col_c:
            st.plotly_chart(plot_prob_histogram(df_out), use_container_width=True)

        st.markdown("---")

        # D. Top 10 rủi ro cao nhất
        st.markdown("#### Danh sách 10 khách hàng rủi ro cao nhất")
        top10 = df_out.nlargest(10, 'probability')[
            [c for c in REQUIRED_COLS if c in df_out.columns] + ['probability', 'risk_tier', 'decision']
        ].copy()
        top10['probability'] = top10['probability'].apply(lambda x: f"{x:.1%}")
        top10['risk_tier'] = top10['risk_tier'].map({
            'LOW': '🟢 Thấp', 'MEDIUM': '🟡 TB', 'HIGH': '🟠 Cao', 'VERY_HIGH': '🔴 Rất cao'
        })
        top10['decision'] = top10['decision'].map({'APPROVE': '✅ Duyệt', 'REJECT': '❌ Từ chối'})
        st.dataframe(top10, use_container_width=True, hide_index=True)

        st.markdown("---")

        # E. Phân tích segment
        st.markdown("#### Phân tích rủi ro theo phân khúc")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.plotly_chart(plot_segment_analysis(df_raw_cached, df_out), use_container_width=True)
        with col_e2:
            fig_inc = plot_income_segment(df_raw_cached, df_out)
            if fig_inc:
                st.plotly_chart(fig_inc, use_container_width=True)

        st.markdown("---")

        # F. SHAP global (sample 500)
        if X_feat_cached is not None:
            st.markdown("#### Tầm quan trọng đặc trưng toàn cục (SHAP)")
            st.caption(f"Tính trên mẫu ngẫu nhiên 500 khách hàng từ tập upload.")
            n_sample = min(500, len(X_feat_cached))
            X_sample = X_feat_cached.sample(n=n_sample, random_state=42)
            with st.spinner("Đang tính SHAP..."):
                fig_shap = plot_shap_global(model, X_sample)
            st.pyplot(fig_shap, use_container_width=True)
            plt.close(fig_shap)

        # G. Benchmark (chỉ khi có target)
        if has_target_cached and TARGET_COL in df_raw_cached.columns:
            st.markdown("---")
            st.markdown("#### Đánh giá hiệu năng mô hình (Benchmark)")
            _render_benchmark(model, X_feat_cached, df_raw_cached[TARGET_COL])

        st.markdown("---")

        # H. Download
        st.markdown("#### Tải kết quả")
        df_download = df_out.copy()
        df_download['probability'] = df_download['probability'].round(4)
        csv_bytes = df_download.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Tải CSV kết quả",
            data=csv_bytes,
            file_name="ket_qua_danh_gia_rui_ro.csv",
            mime="text/csv",
        )
        st.caption(f"File kết quả gồm {len(df_download):,} dòng với 3 cột bổ sung: `probability`, `risk_tier`, `decision`.")

        if st.button("🗑️ Xóa kết quả hiện tại", key="clear_results"):
            for k in ['batch_result', 'batch_raw', 'batch_feat', 'has_target']:
                st.session_state.pop(k, None)
            st.rerun()


def _render_overview(df_out: pd.DataFrame) -> None:
    """Hiển thị 4 metric tổng quan."""
    n = len(df_out)
    pct_default = (df_out['probability'] >= THRESHOLD).mean()
    avg_prob = df_out['probability'].mean()
    pct_reject = (df_out['decision'] == 'REJECT').mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng số khách hàng", f"{n:,}")
    col2.metric("Tỷ lệ dự báo vỡ nợ", f"{pct_default:.1%}")
    col3.metric("Xác suất vỡ nợ TB", f"{avg_prob:.1%}")
    col4.metric("Tỷ lệ bị từ chối", f"{pct_reject:.1%}")


def _render_benchmark(model, X_feat: pd.DataFrame, y_true: pd.Series) -> None:
    """Khi CSV có cột target — đánh giá toàn diện hiệu năng mô hình trên tập đó."""
    import numpy as np

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
        from evaluation import evaluate_model, plot_confusion_matrix

        y_arr = y_true.values if hasattr(y_true, 'values') else np.asarray(y_true)
        probs = model.predict_proba(X_feat)[:, 1]

        # 4 KPI chính
        metrics = evaluate_model(model, X_feat, y_arr, threshold=THRESHOLD, name='XGBoost')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AUC-ROC", f"{metrics['AUC-ROC']:.4f}",
                    help="Khả năng phân biệt — 0,5 = ngẫu nhiên, 1 = hoàn hảo")
        col2.metric("Recall", f"{metrics['Recall']:.3f}",
                    help="Tỷ lệ phát hiện đúng người thực sự vỡ nợ")
        col3.metric("Precision", f"{metrics['Precision']:.3f}",
                    help="Trong số bị từ chối, bao nhiêu thực sự vỡ nợ")
        col4.metric("F2-Score", f"{metrics['F2']:.3f}",
                    help="Trung bình điều hòa Recall (gấp đôi) và Precision")

        # KS statistic
        ks = compute_ks_stat(y_arr, probs)
        col5, col6 = st.columns(2)
        col5.metric("KS Statistic", f"{ks['ks']:.3f}",
                    help="Khoảng cách CDF giữa 2 lớp; >0,3 = mô hình credit risk tốt")
        col6.metric("Ngưỡng KS-optimal", f"{ks['threshold_at_ks']:.3f}",
                    help="Ngưỡng tại đó CDF khác nhau nhiều nhất")

        # Reference benchmark từ test set (báo cáo chính)
        delta_auc = metrics['AUC-ROC'] - 0.8714
        if abs(delta_auc) < 0.01:
            st.success(
                f"Hiệu năng AUC trên tập của bạn = {metrics['AUC-ROC']:.4f} — **gần như đồng nhất** "
                f"với tập kiểm tra gốc (0,8714, Δ={delta_auc:+.4f}). Mô hình vẫn ổn định trên dữ liệu mới."
            )
        elif delta_auc < -0.01:
            st.warning(
                f"AUC = {metrics['AUC-ROC']:.4f} thấp hơn benchmark gốc 0,8714 "
                f"(Δ={delta_auc:+.4f}). Cân nhắc data drift hoặc khác biệt phân phối."
            )
        else:
            st.info(
                f"AUC = {metrics['AUC-ROC']:.4f} cao hơn benchmark gốc 0,8714 "
                f"(Δ={delta_auc:+.4f}). Tập của bạn có thể dễ phân biệt hơn tập test."
            )

        # ROC + PR
        st.markdown("##### Đường cong ROC và Precision-Recall")
        fig_rocpr = plot_roc_pr(y_arr, probs)
        st.plotly_chart(fig_rocpr, use_container_width=True)

        # Lift + Gain
        st.markdown("##### Phân tích Lift và Cumulative Gain (theo decile rủi ro)")
        st.caption(
            "Decile 1 = 10% hồ sơ rủi ro nhất theo mô hình. Lift cao ở decile đầu = mô hình "
            "tập trung tín hiệu vỡ nợ vào nhóm top đầu — cốt lõi của credit scoring."
        )
        fig_lg = plot_lift_gain(y_arr, probs, n_deciles=10)
        st.plotly_chart(fig_lg, use_container_width=True)

        # Calibration
        st.markdown("##### Hiệu chỉnh xác suất (Reliability Diagram)")
        st.caption(
            "Đường cong gần đường chéo = xác suất do mô hình đưa ra phản ánh đúng tỷ lệ vỡ nợ thực tế. "
            "Lệch phía trên = mô hình under-estimate; lệch dưới = over-estimate."
        )
        fig_cal = plot_calibration_batch(y_arr, probs, n_bins=10)
        st.plotly_chart(fig_cal, use_container_width=True)

        # Confusion matrix
        st.markdown(f"##### Ma trận nhầm lẫn (ngưỡng = {THRESHOLD})")
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_confusion_matrix(model, X_feat, y_arr, threshold=THRESHOLD,
                              title=f'Ma trận nhầm lẫn (Ngưỡng={THRESHOLD})', ax=ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    except Exception as e:
        st.warning(f"Không thể tính benchmark: {e}")
        import traceback
        with st.expander("Chi tiết lỗi"):
            st.code(traceback.format_exc())


def _render_portfolio_assessment(df_out: pd.DataFrame) -> None:
    """Đánh giá định tính tổng thể danh mục — nhận xét bằng ngôn ngữ kinh doanh."""
    import numpy as np

    n = len(df_out)
    avg_p = df_out['probability'].mean()
    pct_reject = (df_out['decision'] == 'REJECT').mean()
    tier = df_out['risk_tier'].value_counts(normalize=True)
    pct_low = tier.get('LOW', 0)
    pct_high_plus = tier.get('HIGH', 0) + tier.get('VERY_HIGH', 0)

    # So sánh với prevalence dataset gốc (6,68%) để diễn giải
    baseline = 0.0668
    delta_vs_baseline = avg_p - baseline

    # Quyết định nhãn portfolio
    if avg_p < 0.08 and pct_reject < 0.05:
        verdict = ("🟢 **Danh mục chất lượng cao**", "#27ae60",
                   f"Xác suất vỡ nợ trung bình {avg_p:.1%} thấp hơn baseline thị trường 6,68%. "
                   f"Chỉ {pct_reject:.1%} hồ sơ bị từ chối — phần lớn khách hàng đáp ứng tiêu chí cấp tín dụng.")
    elif avg_p < 0.15 and pct_reject < 0.15:
        verdict = ("🟡 **Danh mục mức trung bình**", "#f39c12",
                   f"Xác suất vỡ nợ trung bình {avg_p:.1%}, tỷ lệ từ chối {pct_reject:.1%}. "
                   f"Phù hợp với baseline, cần theo dõi nhóm HIGH ({tier.get('HIGH', 0):.1%}) sát sao.")
    elif avg_p < 0.30:
        verdict = ("🟠 **Danh mục có dấu hiệu rủi ro**", "#e67e22",
                   f"Xác suất vỡ nợ trung bình {avg_p:.1%} cao hơn baseline {delta_vs_baseline*100:+.1f} điểm phần trăm. "
                   f"Tỷ lệ từ chối {pct_reject:.1%} — cần review tiêu chí tiền sàng lọc.")
    else:
        verdict = ("🔴 **Danh mục rủi ro cao**", "#c0392b",
                   f"Xác suất vỡ nợ trung bình {avg_p:.1%} — cao gấp {avg_p/baseline:.1f}× baseline. "
                   f"{pct_reject:.1%} hồ sơ bị từ chối. Khuyến nghị: thắt chặt tiêu chí, áp lãi suất phòng rủi ro.")

    label, color, body = verdict
    st.markdown(
        f"""
        <div style="background-color:{color}15; border-left:6px solid {color};
                    padding:14px 18px; border-radius:6px; margin-top:10px;">
            <div style="font-size:17px; font-weight:600; margin-bottom:6px;">{label}</div>
            <div style="font-size:14px; color:#34495e; line-height:1.5;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mini recommendation table
    cols = st.columns(3)
    cols[0].markdown(
        f"**🟢 Hồ sơ an toàn (LOW)**\n\n{pct_low:.1%} ({int(pct_low * n):,} người)\n\n_Có thể ưu tiên duyệt nhanh, lãi suất ưu đãi._"
    )
    cols[1].markdown(
        f"**🟠 Hồ sơ cần thẩm định kỹ (HIGH)**\n\n{tier.get('HIGH', 0):.1%} ({int(tier.get('HIGH', 0) * n):,} người)\n\n_Yêu cầu xác minh thu nhập + tài sản đảm bảo bổ sung._"
    )
    cols[2].markdown(
        f"**🔴 Hồ sơ từ chối (VERY HIGH)**\n\n{tier.get('VERY_HIGH', 0):.1%} ({int(tier.get('VERY_HIGH', 0) * n):,} người)\n\n_Vượt ngưỡng triển khai 0,625 — không cấp tín dụng._"
    )


def _show_column_guide() -> None:
    """Hiển thị hướng dẫn các cột cần thiết."""
    st.markdown("**10 cột bắt buộc trong file CSV:**")
    col_desc = {
        'RevolvingUtilizationOfUnsecuredLines': 'Tỷ lệ sử dụng hạn mức tín dụng (số thực, 0-10)',
        'age': 'Tuổi (số nguyên, 18-100)',
        'NumberOfTime30-59DaysPastDueNotWorse': 'Số lần trễ 30-59 ngày (số nguyên)',
        'DebtRatio': 'Tỷ lệ nợ/thu nhập (số thực)',
        'MonthlyIncome': 'Thu nhập hàng tháng (số nguyên, có thể để trống)',
        'NumberOfOpenCreditLinesAndLoans': 'Số dòng tín dụng đang mở (số nguyên)',
        'NumberOfTimes90DaysLate': 'Số lần trễ >90 ngày (số nguyên)',
        'NumberRealEstateLoansOrLines': 'Số khoản vay bất động sản (số nguyên)',
        'NumberOfTime60-89DaysPastDueNotWorse': 'Số lần trễ 60-89 ngày (số nguyên)',
        'NumberOfDependents': 'Số người phụ thuộc (số nguyên, có thể để trống)',
    }
    data = [{'Tên cột': k, 'Mô tả': v} for k, v in col_desc.items()]
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    st.caption("Cột tùy chọn: `SeriousDlqin2yrs` (0/1) — nếu có sẽ tính thêm chỉ số benchmark.")
