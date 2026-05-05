"""Plotly + matplotlib visualization helpers cho Streamlit app."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
try:
    from .utils import FEATURES, FEATURE_LABELS, RISK_COLORS, RISK_LABELS_VI, THRESHOLD
except ImportError:
    from utils import FEATURES, FEATURE_LABELS, RISK_COLORS, RISK_LABELS_VI, THRESHOLD


# ─── Single customer: SHAP waterfall ────────────────────────────────────────

def plot_shap_waterfall(X_df: pd.DataFrame, explanation, prob: float) -> plt.Figure:
    """Vẽ biểu đồ SHAP dạng bar (horizontal) — top 10 features."""
    shap_vals = explanation.values[0]
    base_val = explanation.base_values[0]

    order = np.argsort(np.abs(shap_vals))[::-1][:10]
    top_feat = [FEATURES[i] for i in order]
    top_shap = shap_vals[order]
    top_fval = [f"{X_df.iloc[0, i]:.3g}" for i in order]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in top_shap]
    y_pos = np.arange(len(top_feat))[::-1]

    ax.barh(y_pos, top_shap, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax.set_yticks(y_pos)

    labels = [f"{FEATURE_LABELS.get(f, f)[:28]} = {v}" for f, v in zip(top_feat, top_fval)]
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Giá trị SHAP  (đỏ = tăng rủi ro, xanh = giảm rủi ro)', fontsize=10, fontweight='bold')

    base_prob = 1 / (1 + np.exp(-base_val)) if base_val is not None else 0.067
    ax.set_title(f'Giải thích dự báo | Xác suất cơ sở: {base_prob:.1%} → Kết quả: {prob:.1%}',
                 fontsize=11, fontweight='bold', pad=12)

    pos_patch = mpatches.Patch(color='#e74c3c', label='Tăng rủi ro vỡ nợ')
    neg_patch = mpatches.Patch(color='#2ecc71', label='Giảm rủi ro vỡ nợ')
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    return fig


# ─── Batch: overview metrics (no chart — just data) ─────────────────────────

def compute_batch_stats(df_out: pd.DataFrame) -> dict:
    """Tính các chỉ số tổng quan cho batch result."""
    n = len(df_out)
    pct_default = (df_out['probability'] >= THRESHOLD).mean()
    avg_prob = df_out['probability'].mean()
    pct_reject = (df_out['decision'] == 'REJECT').mean()
    return {'n': n, 'pct_default': pct_default, 'avg_prob': avg_prob, 'pct_reject': pct_reject}


# ─── Batch: Risk tier donut ──────────────────────────────────────────────────

def plot_risk_tier_donut(df_out: pd.DataFrame) -> go.Figure:
    """Biểu đồ donut phân bố Risk Tier."""
    tier_counts = df_out['risk_tier'].value_counts().reindex(
        ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH'], fill_value=0
    )
    tier_labels = [f"Thấp ({RISK_LABELS_VI['LOW']})", f"Trung bình ({RISK_LABELS_VI['MEDIUM']})",
                   f"Cao ({RISK_LABELS_VI['HIGH']})", f"Rất cao ({RISK_LABELS_VI['VERY_HIGH']})"]
    colors = [RISK_COLORS['LOW'], RISK_COLORS['MEDIUM'], RISK_COLORS['HIGH'], RISK_COLORS['VERY_HIGH']]

    fig = go.Figure(go.Pie(
        labels=tier_labels,
        values=tier_counts.values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        hovertemplate='%{label}<br>Số lượng: %{value:,}<br>Tỷ lệ: %{percent}<extra></extra>',
    ))
    fig.update_layout(
        title='Phân bố mức độ rủi ro',
        title_font_size=14,
        showlegend=False,
        margin=dict(t=50, b=10, l=10, r=10),
        height=320,
    )
    return fig


# ─── Batch: Probability histogram ───────────────────────────────────────────

def plot_prob_histogram(df_out: pd.DataFrame) -> go.Figure:
    """Histogram xác suất vỡ nợ với đường ngưỡng."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df_out['probability'],
        nbinsx=40,
        name='Xác suất',
        marker_color='#3498db',
        opacity=0.75,
    ))
    for thresh, color, label in [
        (0.10, 'rgba(46,204,113,0.6)', 'Ngưỡng Thấp (10%)'),
        (0.30, 'rgba(243,156,18,0.6)', 'Ngưỡng TB (30%)'),
        (THRESHOLD, 'rgba(231,76,60,0.9)', f'Ngưỡng từ chối ({THRESHOLD})'),
    ]:
        fig.add_vline(x=thresh, line_dash='dash', line_color=color,
                      annotation_text=label, annotation_position='top right',
                      annotation_font_size=10)
    fig.update_layout(
        title='Phân bố xác suất vỡ nợ',
        xaxis_title='Xác suất vỡ nợ',
        yaxis_title='Số lượng khách hàng',
        bargap=0.05,
        height=320,
        margin=dict(t=50, b=40, l=40, r=20),
    )
    return fig


# ─── Batch: Segment analysis ─────────────────────────────────────────────────

def plot_segment_analysis(df_orig: pd.DataFrame, df_out: pd.DataFrame) -> go.Figure:
    """Bar chart tỷ lệ default dự báo theo nhóm tuổi và thu nhập."""
    combined = df_orig[['age', 'MonthlyIncome']].copy()
    combined['probability'] = df_out['probability'].values

    fig = go.Figure()

    # Theo nhóm tuổi
    age_bins = [0, 30, 45, 60, 200]
    age_labels = ['<30', '30-45', '45-60', '>60']
    combined['age_group'] = pd.cut(combined['age'], bins=age_bins, labels=age_labels)
    age_stats = combined.groupby('age_group', observed=True)['probability'].mean()
    fig.add_trace(go.Bar(
        x=age_stats.index.astype(str),
        y=age_stats.values,
        name='Nhóm tuổi',
        marker_color='#3498db',
        text=[f'{v:.1%}' for v in age_stats.values],
        textposition='outside',
    ))

    fig.update_layout(
        title='Tỷ lệ rủi ro trung bình theo nhóm tuổi',
        xaxis_title='Nhóm tuổi',
        yaxis_title='Xác suất vỡ nợ trung bình',
        yaxis_tickformat='.0%',
        height=300,
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


def plot_income_segment(df_orig: pd.DataFrame, df_out: pd.DataFrame) -> go.Figure:
    """Bar chart tỷ lệ default theo phân vị thu nhập."""
    combined = df_orig[['MonthlyIncome']].copy()
    combined['probability'] = df_out['probability'].values

    try:
        combined['income_group'] = pd.qcut(
            combined['MonthlyIncome'].fillna(combined['MonthlyIncome'].median()),
            q=4, labels=['Q1 (thấp)', 'Q2', 'Q3', 'Q4 (cao)']
        )
        income_stats = combined.groupby('income_group', observed=True)['probability'].mean()
        fig = go.Figure(go.Bar(
            x=income_stats.index.astype(str),
            y=income_stats.values,
            marker_color='#9b59b6',
            text=[f'{v:.1%}' for v in income_stats.values],
            textposition='outside',
        ))
        fig.update_layout(
            title='Tỷ lệ rủi ro trung bình theo thu nhập (phân vị)',
            xaxis_title='Nhóm thu nhập',
            yaxis_title='Xác suất vỡ nợ trung bình',
            yaxis_tickformat='.0%',
            height=300,
            margin=dict(t=50, b=40, l=60, r=20),
        )
        return fig
    except Exception:
        return None


# ─── Batch: SHAP global importance ──────────────────────────────────────────

def plot_shap_global(model, X_sample: pd.DataFrame) -> plt.Figure:
    """Bar chart mean |SHAP| trên sample. Dùng matplotlib để reuse với st.pyplot."""
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample).values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feat_importance = pd.Series(mean_abs_shap, index=FEATURES).sort_values(ascending=True)

    labels_vi = [FEATURE_LABELS.get(f, f) for f in feat_importance.index]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#e74c3c' if v > feat_importance.median() else '#3498db' for v in feat_importance.values]
    ax.barh(labels_vi, feat_importance.values, color=colors, alpha=0.8)
    ax.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax.set_title('Tầm quan trọng đặc trưng toàn cục (SHAP)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    return fig


# ─── Benchmark: ROC + PR overlay ────────────────────────────────────────────

def plot_roc_pr(y_true: np.ndarray, probs: np.ndarray) -> go.Figure:
    """Plotly ROC + PR side-by-side."""
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
    from plotly.subplots import make_subplots

    fpr, tpr, _ = roc_curve(y_true, probs)
    auc_roc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f'ROC (AUC = {auc_roc:.4f})',
                                        f'Precision-Recall (AP = {ap:.4f})'))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                             line=dict(color='#e74c3c', width=2.5),
                             name='ROC', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                             line=dict(color='gray', dash='dash', width=1),
                             name='Random', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=rec, y=prec, mode='lines',
                             line=dict(color='#27ae60', width=2.5),
                             name='PR', showlegend=False), row=1, col=2)
    prevalence = y_true.mean()
    fig.add_trace(go.Scatter(x=[0, 1], y=[prevalence, prevalence], mode='lines',
                             line=dict(color='gray', dash='dash', width=1),
                             showlegend=False), row=1, col=2)

    fig.update_xaxes(title_text='Tỷ lệ dương tính giả (FPR)', row=1, col=1, range=[0, 1])
    fig.update_yaxes(title_text='Tỷ lệ dương tính thật (Recall)', row=1, col=1, range=[0, 1])
    fig.update_xaxes(title_text='Recall', row=1, col=2, range=[0, 1])
    fig.update_yaxes(title_text='Precision', row=1, col=2, range=[0, 1])
    fig.update_layout(height=400, margin=dict(t=60, b=50, l=50, r=20))
    return fig


# ─── Benchmark: Lift + Gain chart (decile) ──────────────────────────────────

def plot_lift_gain(y_true: np.ndarray, probs: np.ndarray, n_deciles: int = 10) -> go.Figure:
    """Lift & cumulative gain chart theo decile xác suất."""
    from plotly.subplots import make_subplots

    df = pd.DataFrame({'y': y_true, 'p': probs}).sort_values('p', ascending=False).reset_index(drop=True)
    df['decile'] = pd.qcut(df.index, n_deciles, labels=range(1, n_deciles + 1)).astype(int)

    grp = df.groupby('decile').agg(positives=('y', 'sum'), total=('y', 'count'))
    grp['rate'] = grp['positives'] / grp['total']
    overall_rate = df['y'].mean()
    grp['lift'] = grp['rate'] / overall_rate
    grp['cum_pos'] = grp['positives'].cumsum()
    grp['cum_pos_pct'] = grp['cum_pos'] / df['y'].sum()
    grp['cum_pop_pct'] = grp['total'].cumsum() / len(df)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Biểu đồ Lift theo decile',
                                        'Cumulative Gain (gain ratio)'))
    fig.add_trace(go.Bar(x=grp.index.astype(int), y=grp['lift'],
                         marker_color='#3498db', name='Lift',
                         text=[f'{v:.2f}×' for v in grp['lift']],
                         textposition='outside', showlegend=False), row=1, col=1)
    fig.add_hline(y=1.0, line_dash='dash', line_color='gray',
                  annotation_text='Lift=1 (ngẫu nhiên)', row=1, col=1)

    fig.add_trace(go.Scatter(x=[0] + grp['cum_pop_pct'].tolist(),
                             y=[0] + grp['cum_pos_pct'].tolist(),
                             mode='lines+markers',
                             line=dict(color='#e74c3c', width=2.5),
                             name='Mô hình', showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                             line=dict(color='gray', dash='dash'),
                             name='Ngẫu nhiên', showlegend=False), row=1, col=2)

    fig.update_xaxes(title_text='Decile (1=top rủi ro)', row=1, col=1)
    fig.update_yaxes(title_text='Lift', row=1, col=1)
    fig.update_xaxes(title_text='% dân số thẩm định', tickformat='.0%', row=1, col=2)
    fig.update_yaxes(title_text='% người vỡ nợ bắt được', tickformat='.0%', row=1, col=2)
    fig.update_layout(height=400, margin=dict(t=60, b=50, l=50, r=20))
    return fig


# ─── Benchmark: Calibration ─────────────────────────────────────────────────

def plot_calibration_batch(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> go.Figure:
    """Reliability diagram: predicted prob vs observed default rate."""
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy='quantile')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                             line=dict(color='gray', dash='dash'),
                             name='Hiệu chỉnh hoàn hảo'))
    fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode='lines+markers',
                             line=dict(color='#9b59b6', width=2.5),
                             marker=dict(size=10),
                             name='Mô hình thực tế'))
    fig.update_layout(
        title='Biểu đồ độ tin cậy (Reliability Diagram)',
        xaxis_title='Xác suất dự báo trung bình mỗi nhóm',
        yaxis_title='Tỷ lệ vỡ nợ thực tế',
        xaxis_tickformat='.0%', yaxis_tickformat='.0%',
        height=400, margin=dict(t=60, b=50, l=50, r=20),
    )
    return fig


# ─── Benchmark: KS statistic ────────────────────────────────────────────────

def compute_ks_stat(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Kolmogorov-Smirnov: max distance giữa 2 CDF (positive vs negative class)."""
    pos = np.sort(probs[y_true == 1])
    neg = np.sort(probs[y_true == 0])
    if len(pos) == 0 or len(neg) == 0:
        return {'ks': 0.0, 'threshold_at_ks': 0.5}
    grid = np.unique(np.concatenate([pos, neg, np.linspace(0, 1, 200)]))
    cdf_pos = np.searchsorted(pos, grid, side='right') / len(pos)
    cdf_neg = np.searchsorted(neg, grid, side='right') / len(neg)
    diff = np.abs(cdf_pos - cdf_neg)
    idx_max = int(np.argmax(diff))
    return {'ks': float(diff[idx_max]), 'threshold_at_ks': float(grid[idx_max])}
