"""
evaluation.py — Các chỉ số đánh giá và trực quan hóa cho Loan Default Prediction

Cung cấp hàm đánh giá chuẩn hóa và các hàm vẽ biểu đồ để so sánh 4 mô hình.
Thiết kế để sử dụng được trong cả notebook lẫn Streamlit app.

Lý do phải dùng AUC-ROC thay Accuracy:
    Dataset có class imbalance 6.68% → mô hình "dự báo tất cả là 0" đạt
    Accuracy 93.32% nhưng Recall = 0. AUC-ROC đo khả năng xếp hạng
    (không phụ thuộc ngưỡng), không bị ảnh hưởng trực tiếp bởi mất cân bằng lớp.
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score, fbeta_score,
)


# ─── Tính toán chỉ số đánh giá ───────────────────────────────────────────────

def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
    name: str = 'Model',
) -> Dict:
    """
    Đánh giá mô hình trên 1 tập dữ liệu với ngưỡng phân loại cho trước.

    Các chỉ số trả về:
    - AUC-ROC: Diện tích dưới đường cong ROC (chỉ số chính, không phụ thuộc ngưỡng)
    - AUC-PR: Diện tích dưới đường cong Precision-Recall (tốt hơn với dữ liệu mất cân bằng)
    - F1: trung bình điều hòa của Precision và Recall
    - F2: F-beta với β=2 (ưu tiên Recall hơn — chi phí FN > FP trong tín dụng)
    - Precision, Recall, Specificity
    - Giá trị confusion matrix: TP, FP, TN, FN

    Parameters
    ----------
    model : mô hình đã khớp với phương thức predict_proba
    X : pd.DataFrame
    y : pd.Series
    threshold : float
        Ngưỡng phân loại. Mặc định 0.5, nên tối ưu trên tập val.
    name : str
        Tên mô hình để hiển thị.

    Returns
    -------
    dict
        Tất cả các chỉ số đánh giá.
    """
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        'Model': name,
        'AUC-ROC': roc_auc_score(y, proba),
        'AUC-PR': average_precision_score(y, proba),
        'F1': f1_score(y, pred, zero_division=0),
        'F2': fbeta_score(y, pred, beta=2, zero_division=0),
        'Precision': precision_score(y, pred, zero_division=0),
        'Recall': recall_score(y, pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
        'Threshold': threshold,
    }


def find_optimal_threshold(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str = 'f2',
) -> Tuple[float, float]:
    """
    Tìm ngưỡng phân loại tối ưu trên tập validation.

    Dùng F2-score (β=2) làm mặc định vì trong rủi ro tín dụng:
        Chi phí FN (khoản vay xấu) >> chi phí FP (bỏ lỡ cơ hội)
        → Recall cần được ưu tiên hơn Precision

    Parameters
    ----------
    model : mô hình đã khớp
    X_val, y_val : tập validation
    metric : 'f1' hoặc 'f2'

    Returns
    -------
    best_threshold : float
    best_score : float
    """
    proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.05, 0.95, 0.01)
    scores = []

    for t in thresholds:
        pred = (proba >= t).astype(int)
        if metric == 'f1':
            s = f1_score(y_val, pred, zero_division=0)
        else:  # f2
            s = fbeta_score(y_val, pred, beta=2, zero_division=0)
        scores.append(s)

    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]


def compare_models(
    models: Dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    thresholds: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Tạo bảng so sánh các chỉ số đánh giá cho nhiều mô hình.

    Parameters
    ----------
    models : dict {tên_model: mô_hình_đã_khớp}
    X_val, y_val : tập validation
    thresholds : dict {tên_model: ngưỡng}, optional
        Nếu None, dùng 0.5 cho tất cả mô hình.

    Returns
    -------
    pd.DataFrame
        Bảng so sánh, mỗi mô hình là 1 hàng.
    """
    if thresholds is None:
        thresholds = {name: 0.5 for name in models}

    rows = []
    for name, model in models.items():
        t = thresholds.get(name, 0.5)
        row = evaluate_model(model, X_val, y_val, threshold=t, name=name)
        rows.append(row)

    df = pd.DataFrame(rows).set_index('Model')
    return df.sort_values('AUC-ROC', ascending=False)


# ─── Hàm Vẽ Biểu đồ ─────────────────────────────────────────────────────────

PALETTE = {
    'LR':  '#3498db',
    'DT':  '#e67e22',
    'RF':  '#2ecc71',
    'XGB': '#e74c3c',
}


def plot_confusion_matrix(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
    title: str = 'Confusion Matrix',
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Vẽ confusion matrix dạng heatmap với số lượng tuyệt đối và phần trăm.

    Parameters
    ----------
    model, X, y : đánh giá trên tập dữ liệu này
    threshold : ngưỡng phân loại
    title : tiêu đề biểu đồ
    ax : matplotlib Axes (tạo mới nếu None)

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y, pred)
    n = len(y)

    labels = [
        [f'{cm[0,0]}\n({cm[0,0]/n*100:.1f}%)', f'{cm[0,1]}\n({cm[0,1]/n*100:.1f}%)'],
        [f'{cm[1,0]}\n({cm[1,0]/n*100:.1f}%)', f'{cm[1,1]}\n({cm[1,1]/n*100:.1f}%)'],
    ]

    sns.heatmap(
        cm, annot=labels, fmt='', cmap='Blues',
        xticklabels=['Dự báo: Không vỡ nợ', 'Dự báo: Vỡ nợ'],
        yticklabels=['Thực tế: Không vỡ nợ', 'Thực tế: Vỡ nợ'],
        ax=ax, linewidths=0.5,
    )
    ax.set_title(title, fontweight='bold', pad=10)
    return ax


def plot_roc_pr_curves(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = 'Model',
    color: str = '#3498db',
    ax_roc: Optional[plt.Axes] = None,
    ax_pr: Optional[plt.Axes] = None,
) -> Tuple[plt.Axes, plt.Axes]:
    """
    Vẽ đường cong ROC và Precision-Recall song song.

    Lý do cần cả hai:
    - ROC curve: đo chất lượng xếp hạng tổng thể, ít bị ảnh hưởng bởi mất cân bằng
    - PR curve: phản ánh hiệu suất tốt hơn khi lớp dương rất nhỏ (6.68%)
      Đường cơ sở của PR curve là đường nằm ngang tại y = tỷ lệ hiện diện (6.68%)

    Parameters
    ----------
    model, X, y
    model_name, color : dùng cho chú giải (legend)
    ax_roc, ax_pr : đối tượng Axes (tạo mới nếu None)

    Returns
    -------
    ax_roc, ax_pr
    """
    proba = model.predict_proba(X)[:, 1]
    auc_roc = roc_auc_score(y, proba)
    auc_pr = average_precision_score(y, proba)

    if ax_roc is None or ax_pr is None:
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    fpr, tpr, _ = roc_curve(y, proba)
    ax_roc.plot(fpr, tpr, color=color, lw=2,
                label=f'{model_name} (AUC={auc_roc:.4f})')
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Ngẫu nhiên')
    ax_roc.set_xlabel('Tỷ lệ dương tính giả (1-Specificity)')
    ax_roc.set_ylabel('Tỷ lệ dương tính thật (Recall)')
    ax_roc.set_title('Đường cong ROC', fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=9)

    # PR curve
    precision, recall, _ = precision_recall_curve(y, proba)
    prevalence = y.mean()
    ax_pr.plot(recall, precision, color=color, lw=2,
               label=f'{model_name} (AP={auc_pr:.4f})')
    ax_pr.axhline(prevalence, ls='--', color='gray', alpha=0.5,
                  label=f'Đường cơ sở (tỷ lệ={prevalence:.3f})')
    ax_pr.set_xlabel('Nhạy cảm (Recall)')
    ax_pr.set_ylabel('Độ chính xác (Precision)')
    ax_pr.set_title('Đường cong Precision-Recall', fontweight='bold')
    ax_pr.legend(loc='upper right', fontsize=9)

    return ax_roc, ax_pr


def plot_overlay_roc(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Vẽ các đường cong ROC của nhiều mô hình trên cùng 1 biểu đồ để so sánh.

    Parameters
    ----------
    models : dict {tên: mô_hình_đã_khớp}
    X, y : tập đánh giá
    ax : đối tượng Axes
    save_path : lưu figure nếu không phải None

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c',
              '#9b59b6', '#1abc9c', '#f39c12']

    for (name, model), color in zip(models.items(), colors):
        proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        ax.plot(fpr, tpr, lw=2, color=color, label=f'{name} (AUC={auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Ngẫu nhiên')
    ax.set_xlabel('Tỷ lệ dương tính giả (1-Specificity)')
    ax.set_ylabel('Tỷ lệ dương tính thật (Recall)')
    ax.set_title('So sánh đường cong ROC giữa các mô hình', fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=110)

    return ax


def delong_test(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
) -> dict:
    """
    Kiểm định DeLong (1988) so sánh hai AUC-ROC từ cùng tập test.

    Cơ sở toán học (DeLong et al., Biometrics, 1988):
        AUC ước lượng qua U-statistic:
            V10_i = P(f(x+_i) > f(x-))  ≈  mean(proba_a[pos_i] > proba_a[neg])
        Phương sai và covariance suy ra từ structural components V10, V01.
        Kiểm định: z = (AUC_a - AUC_b) / sqrt(Var(AUC_a - AUC_b))

    Parameters
    ----------
    y_true : array-like, binary (0/1)
    proba_a, proba_b : predicted probabilities từ model A và B (cùng tập test)

    Returns
    -------
    dict với keys: auc_a, auc_b, auc_diff, z_stat, p_value, significant (p<0.05)
    """
    from scipy import stats as scipy_stats

    y_true = np.asarray(y_true)
    proba_a = np.asarray(proba_a)
    proba_b = np.asarray(proba_b)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    def _structural_components(proba):
        pos = proba[pos_idx]
        neg = proba[neg_idx]
        # V10_i: xác suất proba[pos_i] > random neg (tie: 0.5)
        V10 = np.array([
            (np.sum(p > neg) + 0.5 * np.sum(p == neg)) / n_neg
            for p in pos
        ])
        # V01_j: xác suất proba[neg_j] < random pos
        V01 = np.array([
            (np.sum(n < pos) + 0.5 * np.sum(n == pos)) / n_pos
            for n in neg
        ])
        auc = V10.mean()
        var = np.var(V10, ddof=1) / n_pos + np.var(V01, ddof=1) / n_neg
        return auc, var, V10, V01

    auc_a, var_a, V10_a, V01_a = _structural_components(proba_a)
    auc_b, var_b, V10_b, V01_b = _structural_components(proba_b)

    # Covariance giữa hai AUC estimators (cùng tập test)
    cov = (np.cov(V10_a, V10_b, ddof=1)[0, 1] / n_pos +
           np.cov(V01_a, V01_b, ddof=1)[0, 1] / n_neg)

    var_diff = var_a + var_b - 2 * cov
    if var_diff <= 0:
        var_diff = 1e-12  # numerical safety

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p_val = 2 * (1 - scipy_stats.norm.cdf(abs(z)))

    return {
        'auc_a': float(auc_a),
        'auc_b': float(auc_b),
        'auc_diff': float(auc_a - auc_b),
        'z_stat': float(z),
        'p_value': float(p_val),
        'significant': bool(p_val < 0.05),
    }


def compute_calibration_metrics(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_bins: int = 10,
) -> dict:
    """
    Tính Brier Score và Expected Calibration Error (ECE) cho một model.

    Brier Score: BS = (1/N) Σ (p_hat_i - y_i)²
        - Nhỏ hơn = tốt hơn
        - Baseline (predict prevalence): BS_ref = prev × (1-prev) ≈ 0.0623

    ECE: trung bình |fraction_of_positives - mean_predicted| theo bin.

    Parameters
    ----------
    model : fitted model với predict_proba
    X, y : tập đánh giá
    n_bins : số bin cho reliability diagram

    Returns
    -------
    dict với: brier_score, brier_skill_score, ece, fraction_of_positives, mean_predicted_value
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    proba = model.predict_proba(X)[:, 1]
    y_arr = np.asarray(y)

    brier = brier_score_loss(y_arr, proba)
    brier_ref = float(y_arr.mean() * (1 - y_arr.mean()))  # predict prevalence baseline
    bss = 1 - brier / brier_ref  # Brier Skill Score: > 0 = tốt hơn baseline

    frac_pos, mean_pred = calibration_curve(y_arr, proba, n_bins=n_bins)
    ece = float(np.mean(np.abs(frac_pos - mean_pred)))

    return {
        'brier_score': float(brier),
        'brier_ref': brier_ref,
        'brier_skill_score': float(bss),
        'ece': ece,
        'fraction_of_positives': frac_pos,
        'mean_predicted_value': mean_pred,
    }


def plot_calibration_curve(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_bins: int = 10,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Vẽ Reliability Diagram (Calibration Curve) cho nhiều models.

    Đường chéo 45° = perfect calibration.
    Nằm dưới đường chéo = overestimate xác suất (overconfident).
    Nằm trên = underestimate.

    Parameters
    ----------
    models : dict {tên: model đã khớp}
    X, y : tập đánh giá
    n_bins : số bin
    ax, save_path : tùy chọn

    Returns
    -------
    plt.Axes
    """
    from sklearn.calibration import calibration_curve

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c']
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Hiệu chỉnh hoàn hảo')

    for (name, model), color in zip(models.items(), colors):
        proba = model.predict_proba(X)[:, 1]
        frac_pos, mean_pred = calibration_curve(np.asarray(y), proba, n_bins=n_bins)
        ax.plot(mean_pred, frac_pos, marker='o', lw=2, color=color, label=name)

    ax.set_xlabel('Xác suất dự báo trung bình', fontweight='bold')
    ax.set_ylabel('Tỷ lệ vỡ nợ thực tế', fontweight='bold')
    ax.set_title('Biểu đồ độ tin cậy — Chất lượng hiệu chỉnh xác suất', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3, linestyle='--')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=110)

    return ax


def plot_overlay_pr(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Vẽ các đường cong Precision-Recall của nhiều mô hình trên cùng 1 biểu đồ.

    Parameters
    ----------
    models : dict {tên: mô_hình_đã_khớp}
    X, y : tập đánh giá
    ax, save_path : tùy chọn

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c',
              '#9b59b6', '#1abc9c', '#f39c12']
    prevalence = y.mean()

    for (name, model), color in zip(models.items(), colors):
        proba = model.predict_proba(X)[:, 1]
        precision, recall, _ = precision_recall_curve(y, proba)
        ap = average_precision_score(y, proba)
        ax.plot(recall, precision, lw=2, color=color, label=f'{name} (AP={ap:.4f})')

    ax.axhline(prevalence, ls='--', color='gray', alpha=0.5,
               label=f'Đường cơ sở (tỷ lệ={prevalence:.3f})')
    ax.set_xlabel('Nhạy cảm (Recall)')
    ax.set_ylabel('Độ chính xác (Precision)')
    ax.set_title('So sánh đường cong Precision-Recall giữa các mô hình', fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=110)

    return ax
