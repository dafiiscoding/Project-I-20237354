"""
Regen fig_25_threshold_optimization.png voi nhan tieng Viet.
Load: models/best_model.pkl + data/splits/{val,test}.csv
Output: reports/fig_25_threshold_optimization.png

Chay: python notebooks/regen_fig25_threshold.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, fbeta_score, precision_score, recall_score)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from plot_style import setup_vietnamese_style  # noqa: E402

setup_vietnamese_style(dpi=150)

DEPLOY_THRESHOLD = 0.625

model = joblib.load(PROJECT_ROOT / 'models' / 'best_model.pkl')
def _load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(PROJECT_ROOT / 'data' / 'splits' / f'{name}.csv')
    df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')], errors='ignore')
    if 'DebtToIncomeRatio' in df.columns:
        df = df.rename(columns={'DebtToIncomeRatio': 'AbsoluteMonthlyDebt'})
    return df


val = _load_split('val')
test = _load_split('test')

target = 'SeriousDlqin2yrs'
feature_cols = list(model.get_booster().feature_names)
X_val, y_val = val[feature_cols], val[target]
X_test, y_test = test[feature_cols], test[target]

proba_val = model.predict_proba(X_val)[:, 1]
proba_test = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.05, 0.95, 0.005)
f1_scores = [f1_score(y_val, (proba_val >= t).astype(int), zero_division=0) for t in thresholds]
f2_scores = [fbeta_score(y_val, (proba_val >= t).astype(int), beta=2, zero_division=0) for t in thresholds]
prec_scores = [precision_score(y_val, (proba_val >= t).astype(int), zero_division=0) for t in thresholds]
rec_scores = [recall_score(y_val, (proba_val >= t).astype(int), zero_division=0) for t in thresholds]

t_f1 = thresholds[np.argmax(f1_scores)]
t_f2 = thresholds[np.argmax(f2_scores)]

print(f"F1-optimal threshold: {t_f1:.3f} | F1={max(f1_scores):.4f}")
print(f"F2-optimal threshold: {t_f2:.3f} | F2={max(f2_scores):.4f}")
print(f"Deployment threshold: {DEPLOY_THRESHOLD}")

fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))

# 1. F1 vs F2
ax = axes[0]
ax.plot(thresholds, f1_scores, color='#3498db', lw=2.4, label=r'F1 ($\beta=1$)')
ax.plot(thresholds, f2_scores, color='#e74c3c', lw=2.4, label=r'F2 ($\beta=2$, ưu tiên Recall)')
ax.axvline(t_f1, color='#3498db', linestyle='--', alpha=0.75,
           label=f'F1-optimal: t={t_f1:.2f}')
ax.axvline(t_f2, color='#e74c3c', linestyle='--', alpha=0.75,
           label=f'F2-optimal: t={t_f2:.2f}')
ax.axvline(0.5, color='gray', linestyle=':', alpha=0.65, label='t=0,5 (mặc định)')
ax.set_xlabel('Ngưỡng phân loại  t', fontsize=11)
ax.set_ylabel('Điểm số', fontsize=11)
ax.set_title('F1 và F2 theo ngưỡng', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower center')
ax.grid(alpha=0.3, linestyle='--')

# 2. Precision vs Recall tradeoff
ax = axes[1]
ax.plot(thresholds, prec_scores, color='#27ae60', lw=2.4, label='Precision (Độ chính xác)')
ax.plot(thresholds, rec_scores, color='#e74c3c', lw=2.4, label='Recall (Độ phát hiện)')
ax.axvline(t_f2, color='#e74c3c', linestyle='--', alpha=0.75,
           label=f'F2-optimal: t={t_f2:.2f}')
ax.axvline(DEPLOY_THRESHOLD, color='black', linestyle=':', alpha=0.85,
           label=f'Triển khai: t={DEPLOY_THRESHOLD}')
ax.set_xlabel('Ngưỡng phân loại  t', fontsize=11)
ax.set_ylabel('Điểm số', fontsize=11)
ax.set_title('Đánh đổi Precision ↔ Recall', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='center right')
ax.grid(alpha=0.3, linestyle='--')

# 3. Bar chart so sánh metrics
thresh_compare = [
    ('Mặc định\n(t=0,5)', 0.5),
    (f'F1-optimal\n(t={t_f1:.2f})', t_f1),
    (f'F2-optimal\n(t={t_f2:.2f})', t_f2),
    (f'Triển khai\n(t={DEPLOY_THRESHOLD})', DEPLOY_THRESHOLD),
]
labels = [k for k, _ in thresh_compare]
f1_vals, f2_vals, rec_vals = [], [], []
for _, t in thresh_compare:
    pred = (proba_test >= t).astype(int)
    f1_vals.append(f1_score(y_test, pred, zero_division=0))
    f2_vals.append(fbeta_score(y_test, pred, beta=2, zero_division=0))
    rec_vals.append(recall_score(y_test, pred, zero_division=0))

ax = axes[2]
x = np.arange(len(labels))
w = 0.27
b1 = ax.bar(x - w, f1_vals, w, label='F1', color='#3498db', alpha=0.9)
b2 = ax.bar(x, f2_vals, w, label='F2', color='#e74c3c', alpha=0.9)
b3 = ax.bar(x + w, rec_vals, w, label='Recall', color='#27ae60', alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel('Điểm số', fontsize=11)
ax.set_title('So sánh chỉ số trên tập kiểm tra', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8.5)

plt.suptitle('Phân tích tối ưu ngưỡng phân loại (Threshold Optimization)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
out = PROJECT_ROOT / 'reports' / 'fig_25_threshold_optimization.png'
plt.savefig(out, bbox_inches='tight', dpi=150)
plt.close()
print(f"[OK] {out.name}")
