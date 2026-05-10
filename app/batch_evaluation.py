"""Tab 2: Đánh giá theo lô bằng CSV/XLSX."""

import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from .utils import (FEATURES, FEATURE_LABELS, REQUIRED_COLS, TARGET_COL, THRESHOLD,
                        RISK_COLORS, RISK_LABELS_VI, validate_batch_input, prepare_features,
                        batch_predict, get_risk_tier, make_template_csv, read_uploaded_table)
    from .viz import (plot_risk_tier_donut, plot_prob_histogram, plot_segment_analysis,
                      plot_income_segment, plot_shap_global,
                      plot_roc_curve, plot_pr_curve, plot_lift_gain, plot_calibration_batch,
                      compute_ks_stat, plot_threshold_tradeoff)
except ImportError:
    from utils import (FEATURES, FEATURE_LABELS, REQUIRED_COLS, TARGET_COL, THRESHOLD,
                       RISK_COLORS, RISK_LABELS_VI, validate_batch_input, prepare_features,
                       batch_predict, get_risk_tier, make_template_csv, read_uploaded_table)
    from viz import (plot_risk_tier_donut, plot_prob_histogram, plot_segment_analysis,
                     plot_income_segment, plot_shap_global,
                     plot_roc_curve, plot_pr_curve, plot_lift_gain, plot_calibration_batch,
                     compute_ks_stat, plot_threshold_tradeoff)


def _get_demo_sources() -> list[tuple[str, Path]]:
    """Trả về nguồn dữ liệu đề xuất, luôn có sample đi kèm app khi deploy."""
    root_dir = Path(__file__).parent.parent
    raw_dir = root_dir / 'data' / 'raw'
    sample_dir = Path(__file__).parent / 'sample_data'
    batch_full = raw_dir / 'batch_demo_from_training_5000.csv'
    batch_sample = sample_dir / 'batch_demo_from_training_5000.csv'
    kaggle_sample = sample_dir / 'kaggle_test_sample.csv'
    kaggle_full = raw_dir / 'cs-test.csv'

    sources = []
    if batch_full.exists():
        sources.append(("Demo hậu kiểm 5.000 dòng từ tập huấn luyện (có nhãn thật)", batch_full))
    elif batch_sample.exists():
        sources.append(("Demo hậu kiểm 5.000 dòng từ tập huấn luyện (có nhãn thật)", batch_sample))

    if kaggle_sample.exists():
        sources.append(("Demo dự đoán mẫu Kaggle (không có nhãn thật)", kaggle_sample))
    if kaggle_full.exists():
        sources.append(("Tập cs-test Kaggle đầy đủ trên máy (không có nhãn thật)", kaggle_full))
    return sources


def render(model) -> None:
    """Hiển thị Tab 2: tải dữ liệu, dự báo theo lô, hậu kiểm và xuất kết quả."""

    st.subheader("📊 Đánh giá rủi ro theo lô (CSV/XLSX)")
    st.caption("Tải lên file riêng hoặc chọn dữ liệu đề xuất để thử nhanh chức năng đánh giá hàng loạt.")

    # ── Step 1: Upload ──
    demo_sources = _get_demo_sources()
    source_options = [label for label, _ in demo_sources] + ["Upload file mới"]
    source_choice = st.radio(
        "Chọn dữ liệu để đánh giá",
        options=source_options,
        index=0 if demo_sources else len(source_options) - 1,
        horizontal=True,
        help="Demo 5.000 dòng có nhãn thật dùng để hậu kiểm; Kaggle test không có nhãn thật nên chỉ dùng để dự đoán.",
    )
    if source_choice.startswith("Demo hậu kiểm"):
        st.info("Dữ liệu demo 5.000 dòng được tách từ tập huấn luyện gốc, có nhãn thật 0/1 để mô phỏng hậu kiểm AUC/Recall/Precision/F2.")
    elif "Kaggle" in source_choice:
        st.info("Dữ liệu Kaggle test không có nhãn thật, ứng dụng chỉ dự đoán xác suất và không tính hậu kiểm.")

    col_upload, col_template = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Tải file dữ liệu (CSV hoặc XLSX, tối đa 500.000 dòng)",
            type=["csv", "xlsx"],
            disabled=source_choice != "Upload file mới",
            help=f"File cần chứa đủ {len(REQUIRED_COLS)} cột dữ liệu gốc. Tải CSV mẫu bên phải nếu chưa có."
        )
    with col_template:
        st.markdown("&nbsp;")
        st.download_button(
            label="📥 Tải CSV mẫu",
            data=make_template_csv(),
            file_name="mau_danh_sach_khachhang.csv",
            mime="text/csv",
        )

    if source_choice == "Upload file mới" and not uploaded:
        st.markdown("---")
        _show_column_guide()
        return

    # ── Step 2: Load & Validate ──
    try:
        if source_choice == "Upload file mới":
            df_raw = read_uploaded_table(uploaded)
            source_note = getattr(uploaded, 'name', 'file tải lên')
        else:
            local_path = dict(demo_sources)[source_choice]
            df_raw = pd.read_csv(local_path)
            source_note = str(local_path)
    except ValueError as e:
        st.error(str(e))
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
    st.success(f"✅ Tải thành công: **{n_rows:,} dòng**, {len(df_raw.columns)} cột từ `{source_note}`{' (có cột nhãn thật — sẽ hiển thị kết quả hậu kiểm)' if has_target else ''}.")

    decision_threshold = st.slider(
        "Ngưỡng quyết định từ chối (dùng cho tập file này)",
        min_value=0.10,
        max_value=0.90,
        value=float(st.session_state.get('batch_threshold', THRESHOLD)),
        step=0.005,
        help="Khách hàng có xác suất vỡ nợ >= ngưỡng sẽ bị xếp vào nhóm từ chối.",
    )

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

        df_out = df_raw.loc[X_feat.index].copy()
        df_out['probability'] = probs
        df_out['risk_tier'] = [get_risk_tier(p, threshold=decision_threshold)[0] for p in probs]
        df_out['decision'] = np.where(probs >= decision_threshold, 'REJECT', 'APPROVE')
        df_out['decision_threshold'] = decision_threshold

        st.session_state['batch_result'] = df_out
        st.session_state['batch_raw'] = df_raw
        st.session_state['batch_feat'] = X_feat
        st.session_state['has_target'] = has_target
        st.session_state['batch_threshold'] = decision_threshold
        st.rerun()

    # ── Step 4: Show results ──
    if 'batch_result' in st.session_state:
        df_out = st.session_state['batch_result']
        df_raw_cached = st.session_state.get('batch_raw', df_raw)
        X_feat_cached = st.session_state.get('batch_feat', None)
        has_target_cached = st.session_state.get('has_target', False)
        threshold_cached = float(st.session_state.get('batch_threshold', THRESHOLD))

        st.markdown("---")
        st.markdown("### 📈 Kết quả đánh giá")
        st.caption(f"Ngưỡng quyết định đang dùng cho tập này: **{threshold_cached:.3f}**")

        # A. Tổng quan + đánh giá portfolio
        _render_overview(df_out, threshold_cached)
        _render_portfolio_assessment(df_out, threshold_cached)

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
        _render_segment_summary_table(df_raw_cached, df_out)

        st.markdown("---")

        # F. SHAP global (sample 500)
        if X_feat_cached is not None:
            st.markdown("#### Tầm quan trọng đặc trưng toàn cục (SHAP)")
            st.caption(f"Tính trên mẫu ngẫu nhiên 500 khách hàng từ tập tải lên.")
            n_sample = min(500, len(X_feat_cached))
            X_sample = X_feat_cached.sample(n=n_sample, random_state=42)
            with st.spinner("Đang tính SHAP..."):
                fig_shap = plot_shap_global(model, X_sample)
            st.pyplot(fig_shap, use_container_width=True)
            plt.close(fig_shap)

        # G. Hậu kiểm (chỉ khi có nhãn thật)
        if has_target_cached and TARGET_COL in df_raw_cached.columns:
            st.markdown("---")
            st.markdown("#### Đánh giá hiệu năng mô hình (hậu kiểm)")
            _render_benchmark(model, X_feat_cached, df_raw_cached[TARGET_COL], threshold_cached)

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
        st.caption(f"File kết quả gồm {len(df_download):,} dòng với 4 cột bổ sung: `probability`, `risk_tier`, `decision`, `decision_threshold`.")

        if st.button("🗑️ Xóa kết quả hiện tại", key="clear_results"):
            for k in ['batch_result', 'batch_raw', 'batch_feat', 'has_target']:
                st.session_state.pop(k, None)
            st.rerun()


def _render_overview(df_out: pd.DataFrame, threshold: float) -> None:
    """Hiển thị 4 metric tổng quan."""
    n = len(df_out)
    pct_default = (df_out['probability'] >= threshold).mean()
    avg_prob = df_out['probability'].mean()
    pct_reject = (df_out['decision'] == 'REJECT').mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng số khách hàng", f"{n:,}")
    col2.metric("Tỷ lệ dự báo vỡ nợ", f"{pct_default:.1%}")
    col3.metric("Xác suất vỡ nợ TB", f"{avg_prob:.1%}")
    col4.metric(f"Tỷ lệ bị từ chối (t={threshold:.3f})", f"{pct_reject:.1%}")


def _render_benchmark(model, X_feat: pd.DataFrame, y_true: pd.Series, threshold: float) -> None:
    """Khi CSV có cột nhãn thật, đánh giá toàn diện hiệu năng mô hình trên tập đó."""
    import numpy as np

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
        from evaluation import evaluate_model, plot_confusion_matrix

        y_series = pd.to_numeric(pd.Series(y_true).reindex(X_feat.index), errors='coerce')
        valid_mask = y_series.notna() & y_series.isin([0, 1])
        n_invalid = int((~valid_mask).sum())
        if n_invalid > 0:
            st.info(
                f"Hậu kiểm bỏ qua {n_invalid:,} dòng có `{TARGET_COL}` trống hoặc không phải 0/1. "
                "Dự đoán vẫn được giữ cho toàn bộ các dòng hợp lệ về đặc trưng."
            )

        X_eval = X_feat.loc[valid_mask].copy()
        y_arr = y_series.loc[valid_mask].astype(int).to_numpy()

        if len(y_arr) == 0:
            st.warning(f"Không thể tính hậu kiểm: cột `{TARGET_COL}` không có dòng nhãn thật 0/1 hợp lệ.")
            return
        if len(np.unique(y_arr)) < 2:
            st.warning(
                f"Không thể tính AUC/ROC/PR vì `{TARGET_COL}` chỉ có một lớp trong các dòng hợp lệ. "
                "Cần có cả 0 và 1 để đánh giá chất lượng phân biệt."
            )
            return

        probs = model.predict_proba(X_eval)[:, 1]

        # 4 KPI chính
        metrics = evaluate_model(model, X_eval, y_arr, threshold=threshold, name='XGBoost')
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
        col6.metric("Ngưỡng KS tối ưu", f"{ks['threshold_at_ks']:.3f}",
                    help="Ngưỡng tại đó CDF khác nhau nhiều nhất")

        # Reference benchmark từ test set (báo cáo chính)
        _render_business_impact(y_arr, probs, threshold)

        delta_auc = metrics['AUC-ROC'] - 0.8714
        if abs(delta_auc) < 0.01:
            st.success(
                f"Hiệu năng AUC trên tập của bạn = {metrics['AUC-ROC']:.4f} — **gần như đồng nhất** "
                f"với tập kiểm tra gốc (0,8714, Δ={delta_auc:+.4f}). Mô hình vẫn ổn định trên dữ liệu mới."
            )
        elif delta_auc < -0.01:
            st.warning(
                f"AUC = {metrics['AUC-ROC']:.4f} thấp hơn mốc kiểm tra gốc 0,8714 "
                f"(Δ={delta_auc:+.4f}). Cân nhắc data drift hoặc khác biệt phân phối."
            )
        else:
            st.info(
                f"AUC = {metrics['AUC-ROC']:.4f} cao hơn benchmark gốc 0,8714 "
                f"(Δ={delta_auc:+.4f}). Tập của bạn có thể dễ phân biệt hơn tập test."
            )

        # ROC + PR
        st.markdown("##### ROC và Precision-Recall (tách riêng để dễ đọc)")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.plotly_chart(plot_roc_curve(y_arr, probs), use_container_width=True)
        with col_r2:
            st.plotly_chart(plot_pr_curve(y_arr, probs), use_container_width=True)

        # Lift + Gain
        st.markdown("##### Phân tích Lift và Cumulative Gain (theo nhóm 10% rủi ro)")
        st.caption(
            "Nhóm 1 = 10% hồ sơ rủi ro nhất theo mô hình. Lift cao ở nhóm đầu nghĩa là mô hình "
            "tập trung tín hiệu vỡ nợ vào nhóm rủi ro cao — cốt lõi của chấm điểm tín dụng."
        )
        fig_lg = plot_lift_gain(y_arr, probs, n_deciles=10)
        st.plotly_chart(fig_lg, use_container_width=True)

        st.markdown("##### Trade-off theo ngưỡng")
        st.caption("Giúp chọn ngưỡng phù hợp giữa Recall, Precision, F2 và tỷ lệ từ chối.")
        st.plotly_chart(plot_threshold_tradeoff(y_arr, probs), use_container_width=True)

        # Calibration
        st.markdown("##### Hiệu chỉnh xác suất (Reliability Diagram)")
        st.caption(
            "Đường cong gần đường chéo = xác suất do mô hình đưa ra phản ánh đúng tỷ lệ vỡ nợ thực tế. "
            "Lệch phía trên = mô hình under-estimate; lệch dưới = over-estimate."
        )
        fig_cal = plot_calibration_batch(y_arr, probs, n_bins=10)
        st.plotly_chart(fig_cal, use_container_width=True)

        # Confusion matrix
        st.markdown(f"##### Ma trận nhầm lẫn (ngưỡng = {threshold:.3f})")
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_confusion_matrix(model, X_eval, y_arr, threshold=threshold,
                              title=f'Ma trận nhầm lẫn (Ngưỡng={threshold:.3f})', ax=ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    except Exception as e:
        st.warning(f"Không thể tính hậu kiểm: {e}")
        import traceback
        with st.expander("Chi tiết lỗi"):
            st.code(traceback.format_exc())


def _render_business_impact(y_arr: np.ndarray, probs: np.ndarray, threshold: float) -> None:
    """Quy đổi kết quả hậu kiểm thành chi phí/lợi ích kinh doanh ước tính."""
    st.markdown("##### Tác động kinh doanh ước tính")
    st.caption(
        "Đây là mô phỏng chi phí rủi ro để giải thích mô hình giúp giảm tổn thất như thế nào "
        "trên lô dữ liệu có nhãn thật, không phải lợi nhuận kế toán chính thức."
    )

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        fn_cost = st.number_input(
            "Tổn thất khi bỏ sót 1 hồ sơ vỡ nợ - FN (USD)",
            min_value=0,
            value=11250,
            step=250,
            key="batch_fn_cost",
        )
    with col_cfg2:
        fp_cost = st.number_input(
            "Chi phí cơ hội khi từ chối nhầm 1 hồ sơ tốt - FP (USD)",
            min_value=0,
            value=500,
            step=50,
            key="batch_fp_cost",
        )

    y_arr = np.asarray(y_arr).astype(int)
    pred = (probs >= threshold).astype(int)

    tp = int(((pred == 1) & (y_arr == 1)).sum())
    fp = int(((pred == 1) & (y_arr == 0)).sum())
    tn = int(((pred == 0) & (y_arr == 0)).sum())
    fn = int(((pred == 0) & (y_arr == 1)).sum())

    n = len(y_arr)
    n_bad = int((y_arr == 1).sum())
    rejected = int(pred.sum())
    approved = n - rejected

    approve_all_cost = n_bad * fn_cost
    current_fn_cost = fn * fn_cost
    current_fp_cost = fp * fp_cost
    current_cost = current_fn_cost + current_fp_cost
    avoided_loss = tp * fn_cost
    savings_vs_approve_all = approve_all_cost - current_cost

    thresholds = np.arange(0.01, 0.991, 0.005)
    costs = []
    reject_rates = []
    for t in thresholds:
        pred_t = (probs >= t).astype(int)
        fp_t = int(((pred_t == 1) & (y_arr == 0)).sum())
        fn_t = int(((pred_t == 0) & (y_arr == 1)).sum())
        costs.append(fn_t * fn_cost + fp_t * fp_cost)
        reject_rates.append(float(pred_t.mean()))

    best_idx = int(np.argmin(costs))
    best_threshold = float(thresholds[best_idx])
    best_cost = float(costs[best_idx])
    best_reject_rate = float(reject_rates[best_idx])

    def money(value: float) -> str:
        return f"{value:,.0f} USD"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Chi phí tại ngưỡng hiện tại", money(current_cost))
    col2.metric("Tiết kiệm so với duyệt tất cả", money(savings_vs_approve_all))
    col3.metric("Tổn thất tránh được", money(avoided_loss))
    col4.metric("Chi phí từ chối nhầm", money(current_fp_cost))

    st.dataframe(
        pd.DataFrame([
                {"Nhóm": "Thực tế vỡ nợ", "Số hồ sơ": n_bad, "Diễn giải": "Nhãn thật = 1 trong file tải lên"},
            {"Nhóm": "Từ chối đúng (TP)", "Số hồ sơ": tp, "Diễn giải": "Chặn được hồ sơ thực sự vỡ nợ"},
            {"Nhóm": "Từ chối nhầm (FP)", "Số hồ sơ": fp, "Diễn giải": "Hồ sơ tốt nhưng bị từ chối"},
            {"Nhóm": "Bỏ sót nợ xấu (FN)", "Số hồ sơ": fn, "Diễn giải": "Hồ sơ vỡ nợ nhưng vẫn được duyệt"},
            {"Nhóm": "Duyệt đúng (TN)", "Số hồ sơ": tn, "Diễn giải": "Hồ sơ tốt được duyệt"},
            {"Nhóm": "Tổng bị từ chối", "Số hồ sơ": rejected, "Diễn giải": f"{rejected / n:.1%} toàn bộ lô"},
            {"Nhóm": "Tổng được duyệt", "Số hồ sơ": approved, "Diễn giải": f"{approved / n:.1%} toàn bộ lô"},
        ]),
        use_container_width=True,
        hide_index=True,
    )

    delta_best = current_cost - best_cost
    if delta_best > 0:
        st.info(
            f"Với giả định chi phí hiện tại, ngưỡng tối ưu theo chi phí trên lô này khoảng "
            f"**{best_threshold:.3f}**; chi phí ước tính **{money(best_cost)}**, thấp hơn ngưỡng đang dùng "
            f"**{money(delta_best)}**. Tỷ lệ từ chối tại ngưỡng đó là **{best_reject_rate:.1%}**."
        )
    else:
        st.success(
            f"Ngưỡng hiện tại đang gần tối ưu theo giả định chi phí này. Ngưỡng tốt nhất quét được là "
            f"**{best_threshold:.3f}** với chi phí **{money(best_cost)}**."
        )


def _render_portfolio_assessment(df_out: pd.DataFrame, threshold: float) -> None:
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
        f"**🔴 Hồ sơ từ chối (VERY HIGH)**\n\n{tier.get('VERY_HIGH', 0):.1%} ({int(tier.get('VERY_HIGH', 0) * n):,} người)\n\n_Vượt ngưỡng triển khai {threshold:.3f} — không cấp tín dụng._"
    )


def _render_segment_summary_table(df_raw: pd.DataFrame, df_out: pd.DataFrame) -> None:
    """Bảng tóm tắt danh mục theo phân khúc để đọc nhanh."""
    st.markdown("##### Bảng tóm tắt theo phân khúc")
    combined = df_raw[['age', 'MonthlyIncome']].copy()
    combined['probability'] = df_out['probability'].values
    combined['decision'] = df_out['decision'].values

    combined['Nhóm tuổi'] = pd.cut(
        combined['age'],
        bins=[0, 30, 45, 60, 200],
        labels=['<30', '30-45', '45-60', '>60'],
    )

    income_clean = combined['MonthlyIncome'].fillna(combined['MonthlyIncome'].median())
    try:
        combined['Nhóm thu nhập'] = pd.qcut(
            income_clean,
            q=4,
            labels=['Q1 thấp', 'Q2', 'Q3', 'Q4 cao'],
            duplicates='drop',
        )
    except ValueError:
        combined['Nhóm thu nhập'] = 'Không đủ biến thiên'

    age_tbl = (
        combined.groupby('Nhóm tuổi', observed=True)
        .agg(
            so_khach_hang=('probability', 'count'),
            xac_suat_tb=('probability', 'mean'),
            ty_le_tu_choi=('decision', lambda x: (x == 'REJECT').mean()),
        )
        .reset_index()
    )
    age_tbl = age_tbl.rename(columns={
        'so_khach_hang': 'Số khách hàng',
        'xac_suat_tb': 'Xác suất TB',
        'ty_le_tu_choi': 'Tỷ lệ từ chối',
    })
    age_tbl['Xác suất TB'] = age_tbl['Xác suất TB'].map(lambda x: f"{x:.1%}")
    age_tbl['Tỷ lệ từ chối'] = age_tbl['Tỷ lệ từ chối'].map(lambda x: f"{x:.1%}")

    income_tbl = (
        combined.groupby('Nhóm thu nhập', observed=True)
        .agg(
            so_khach_hang=('probability', 'count'),
            xac_suat_tb=('probability', 'mean'),
            ty_le_tu_choi=('decision', lambda x: (x == 'REJECT').mean()),
        )
        .reset_index()
    )
    income_tbl = income_tbl.rename(columns={
        'so_khach_hang': 'Số khách hàng',
        'xac_suat_tb': 'Xác suất TB',
        'ty_le_tu_choi': 'Tỷ lệ từ chối',
    })
    income_tbl['Xác suất TB'] = income_tbl['Xác suất TB'].map(lambda x: f"{x:.1%}")
    income_tbl['Tỷ lệ từ chối'] = income_tbl['Tỷ lệ từ chối'].map(lambda x: f"{x:.1%}")

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Theo nhóm tuổi")
        st.dataframe(age_tbl, use_container_width=True, hide_index=True)
    with c2:
        st.caption("Theo nhóm thu nhập")
        st.dataframe(income_tbl, use_container_width=True, hide_index=True)


def _show_column_guide() -> None:
    """Hiển thị hướng dẫn các cột cần thiết."""
    st.markdown("**10 cột bắt buộc trong file CSV/XLSX:**")
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
    st.caption("Cột tùy chọn: `SeriousDlqin2yrs` (0/1) — nếu có sẽ tính thêm chỉ số hậu kiểm.")
