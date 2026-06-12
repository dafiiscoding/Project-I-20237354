"""Tái sinh các hình có số liệu sai trong reports/ từ model + splits đã lưu.

- fig_25_threshold_optimization.png : quét ngưỡng trên VAL (panel 1-2),
  so sánh 4 ngưỡng trên TEST (panel 3). Bản cũ vẽ chi phí ngưỡng Bayes
  34,5 triệu USD — bất khả thi (tối đa ~10,6 triệu); bản này tính lại.
- fig_31_calibration.png : reliability + Brier của 4 mô hình trên TEST
  (trước hiệu chỉnh), khớp addendum_results.md.
- fig_31b_platt.png (MỚI) : XGBoost trước/sau Platt scaling (khớp trên VAL).
- fig_24_score_distribution.png : phân bố điểm theo nhãn thật; bản cũ vẽ
  vạch ngưỡng 0,77 trong khi ngưỡng triển khai của báo cáo là 0,625.
- model_results.csv : bảng so sánh 4 mô hình. Bản cũ là output của một lần
  chạy notebook 03 trước khi lưu model (LR 0.8432, RF 0.8703 — lệch với
  models/*.pkl); bản này tính lại từ model đã lưu, khớp bảng trong báo cáo.
  Hai cột cv_auc và train_time_s bị bỏ vì không tái lập được nếu không
  chạy lại toàn bộ tìm kiếm siêu tham số.

Chạy: python regen_figs.py
"""

from pathlib import Path
import pickle
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                             average_precision_score)
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.plot_style import setup_vietnamese_style, PALETTE_MODELS  # noqa: E402
from final_report.verify_numbers import (  # noqa: E402
    load_split, metrics_at, ece, C_FN, C_FP,
)

REPORTS = ROOT / 'reports'


def main() -> None:
    setup_vietnamese_style(dpi=150)

    Xv, yv = load_split('val')
    Xt, yt = load_split('test')

    xgb_model = pickle.load(open(ROOT / 'models' / 'best_model.pkl', 'rb'))
    models = {
        'Logistic Regression': joblib.load(ROOT / 'models' / 'model_lr.pkl'),
        'Decision Tree': joblib.load(ROOT / 'models' / 'model_dt.pkl'),
        'Random Forest': joblib.load(ROOT / 'models' / 'model_rf.pkl'),
        'XGBoost': xgb_model,
    }
    probs_t = {n: m.predict_proba(Xt)[:, 1] for n, m in models.items()}
    pv = xgb_model.predict_proba(Xv)[:, 1]
    pt = probs_t['XGBoost']
    for n, p in probs_t.items():
        print(f'AUC test {n}: {roc_auc_score(yt, p):.4f}')

    # ---------- fig_25: tối ưu ngưỡng ----------
    grid = np.arange(0.05, 0.951, 0.005)
    mv = [metrics_at(yv, pv, t) for t in grid]
    f1s = np.array([d['f1'] for d in mv])
    f2s = np.array([d['f2'] for d in mv])
    precs = np.array([d['precision'] for d in mv])
    recs = np.array([d['recall'] for d in mv])
    t_f1 = grid[f1s.argmax()]
    t_f2 = grid[f2s.argmax()]
    print(f'F2-opt (val) = {t_f2:.3f} | F1-opt (val) = {t_f1:.3f}')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle('Phân tích tối ưu ngưỡng vận hành', fontweight='bold')

    ax = axes[0]
    ax.plot(grid, f1s, color='#3498db', lw=2, label='F1')
    ax.plot(grid, f2s, color='#e74c3c', lw=2, label='F2')
    ax.axvline(t_f2, color='black', ls='--', lw=1,
               label=f'F2-opt = {t_f2:.3f}')
    ax.set_xlabel('Ngưỡng'); ax.set_ylabel('Điểm số')
    ax.set_title('F1/F2 theo ngưỡng (tập kiểm định)')
    ax.legend()

    ax = axes[1]
    ax.plot(grid, precs, color='#f39c12', lw=2, label='Precision')
    ax.plot(grid, recs, color='#2ecc71', lw=2, label='Recall')
    ax.axvline(t_f2, color='black', ls='--', lw=1)
    ax.set_xlabel('Ngưỡng'); ax.set_ylabel('Điểm số')
    ax.set_title('Precision và Recall (tập kiểm định)')
    ax.legend()

    ax = axes[2]
    names, ts = ['0,50', '0,625', '0,775', '0,043'], [0.50, 0.625, 0.775, 0.043]
    mt = [metrics_at(yt, pt, t) for t in ts]
    x = np.arange(len(ts)); w = 0.27
    ax.bar(x - w, [d['recall'] * 100 for d in mt], w,
           color='#2ecc71', label='Recall (%)')
    ax.bar(x, [d['precision'] * 100 for d in mt], w,
           color='#3498db', label='Precision (%)')
    ax.bar(x + w, [d['cost'] for d in mt], w,
           color='#e74c3c', label='Chi phí (triệu USD)')
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_xlabel('Ngưỡng')
    ax.set_title('So sánh ngưỡng chính (tập kiểm tra)')
    ax.legend(fontsize=8)
    for d, xi in zip(mt, x):
        print(f"t={d['t']:.3f}: R={d['recall']:.3f} P={d['precision']:.3f} "
              f"F2={d['f2']:.3f} reject={d['reject']:.3f} cost={d['cost']:.2f}")

    fig.tight_layout()
    fig.savefig(REPORTS / 'fig_25_threshold_optimization.png',
                bbox_inches='tight')
    plt.close(fig)

    # ---------- fig_31: reliability + Brier 4 mô hình ----------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Hiệu chỉnh hoàn hảo')
    briers = {}
    for name, p in probs_t.items():
        frac, mean_p = calibration_curve(yt, p, n_bins=10, strategy='uniform')
        ax.plot(mean_p, frac, 'o-', ms=4, lw=1.5,
                color=PALETTE_MODELS[name], label=name)
        briers[name] = brier_score_loss(yt, p)
    ax.set_xlabel('Xác suất dự báo trung bình')
    ax.set_ylabel('Tỷ lệ vỡ nợ thực tế')
    ax.set_title('Biểu đồ độ tin cậy (trước hiệu chỉnh)')
    ax.legend(fontsize=9)

    ax = axes[1]
    base_brier = brier_score_loss(yt, np.full_like(pt, yt.mean()))
    ax.bar(list(briers), list(briers.values()),
           color=[PALETTE_MODELS[n] for n in briers])
    ax.axhline(base_brier, color='gray', ls='--', lw=1.2,
               label=f'Mức nền (dự báo 6,68%): {base_brier:.4f}')
    for i, (n, b) in enumerate(briers.items()):
        ax.text(i, b + 0.002, f'{b:.4f}', ha='center', fontsize=9)
        print(f'Brier {n}: {b:.4f}')
    ax.set_ylabel('Brier Score (thấp hơn là tốt hơn)')
    ax.set_title('Brier Score theo mô hình (tập kiểm tra)')
    ax.tick_params(axis='x', labelsize=8)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORTS / 'fig_31_calibration.png', bbox_inches='tight')
    plt.close(fig)

    # ---------- fig_31b: XGBoost trước/sau Platt ----------
    platt = LogisticRegression(C=1e10, solver='lbfgs')
    platt.fit(pv.reshape(-1, 1), yv)
    pt_cal = platt.predict_proba(pt.reshape(-1, 1))[:, 1]

    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Hiệu chỉnh hoàn hảo')
    for p, lab, c in [
        (pt, f'Trước Platt (Brier {brier_score_loss(yt, pt):.4f}, '
             f'ECE {ece(yt, pt):.1%})', '#e74c3c'),
        (pt_cal, f'Sau Platt (Brier {brier_score_loss(yt, pt_cal):.4f}, '
                 f'ECE {ece(yt, pt_cal):.2%})', '#2ecc71'),
    ]:
        frac, mean_p = calibration_curve(yt, p, n_bins=10, strategy='uniform')
        ax.plot(mean_p, frac, 'o-', ms=4, lw=1.8, color=c, label=lab)
    ax.set_xlabel('Xác suất dự báo trung bình')
    ax.set_ylabel('Tỷ lệ vỡ nợ thực tế')
    ax.set_title('XGBoost: hiệu chỉnh xác suất bằng Platt scaling\n'
                 '(khớp trên tập kiểm định, đánh giá trên tập kiểm tra)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORTS / 'fig_31b_platt.png', bbox_inches='tight')
    plt.close(fig)

    # ---------- fig_24: phân bố điểm theo nhãn, ngưỡng 0,625 ----------
    T = 0.625
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    bins = np.linspace(0, 1, 51)
    ax.hist(pt[yt == 0], bins=bins, color='#6baed6', alpha=0.85,
            label='Không vỡ nợ')
    ax.hist(pt[yt == 1], bins=bins, color='#e74c3c', alpha=0.75,
            label='Vỡ nợ')
    ax.axvline(T, color='black', ls='--', lw=1.5, label=f'Ngưỡng {T:.3f}')
    ax.set_xlabel('Xác suất dự báo'); ax.set_ylabel('Số hồ sơ')
    ax.set_title('Phân phối điểm dự báo theo nhãn thật')
    ax.legend()

    ax = axes[1]
    pos = pt[yt == 1]
    ax.hist(pos[pos < T], bins=bins, color='#e74c3c', alpha=0.85,
            label='FN (bỏ sót)')
    ax.hist(pos[pos >= T], bins=bins, color='#2ecc71', alpha=0.85,
            label='TP (bắt đúng)')
    ax.axvline(T, color='black', ls='--', lw=1.5, label=f'Ngưỡng {T:.3f}')
    ax.set_xlabel('Xác suất dự báo'); ax.set_ylabel('Số hồ sơ')
    ax.set_title('Nhóm vỡ nợ: bắt đúng và bỏ sót')
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / 'fig_24_score_distribution.png',
                bbox_inches='tight')
    plt.close(fig)

    # ---------- model_results.csv: bảng so sánh từ model đã lưu ----------
    rows = []
    for name, m in models.items():
        p_val = m.predict_proba(Xv)[:, 1]
        p_test = probs_t[name]
        t2 = max((metrics_at(yv, p_val, t) for t in grid),
                 key=lambda d: d['f2'])['t']
        d = metrics_at(yt, p_test, t2)
        rows.append({
            'model': name,
            'test_auc': round(roc_auc_score(yt, p_test), 4),
            'avg_precision': round(average_precision_score(yt, p_test), 4),
            'threshold_f2': round(t2, 3),
            'recall': round(d['recall'], 4),
            'precision': round(d['precision'], 4),
            'f1': round(d['f1'], 4),
            'f2': round(d['f2'], 4),
        })
    pd.DataFrame(rows).to_csv(REPORTS / 'model_results.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    print('OK — đã ghi fig_24, fig_25, fig_31, fig_31b, model_results.csv '
          'vào reports/')


if __name__ == '__main__':
    main()
