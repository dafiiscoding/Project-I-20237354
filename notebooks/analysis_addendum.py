"""
analysis_addendum.py — DeLong Test + Calibration Analysis
==========================================================
Script bổ sung cho 04_Analysis.ipynb:
  - DeLong test so sánh AUC XGBoost vs Random Forest (cùng test set)
  - Calibration analysis: Brier Score, ECE, Reliability Diagram

Chạy từ project root:
    python notebooks/analysis_addendum.py

Output:
    reports/fig_31_calibration.png
    reports/fig_32_delong_summary.png
    reports/addendum_results.md
"""

import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Đảm bảo src/ có trong path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import (
    delong_test,
    compute_calibration_metrics,
    plot_calibration_curve,
)

REPORTS_DIR = ROOT / "reports"
MODELS_DIR  = ROOT / "models"
SPLITS_DIR  = ROOT / "data" / "splits"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ─── 1. Load dữ liệu và models ───────────────────────────────────────────────

print("Loading test set and models...")

test_df  = pd.read_csv(SPLITS_DIR / "test.csv")
train_df = pd.read_csv(SPLITS_DIR / "train.csv")

FEATURES = [
    'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents', 'TotalDelinquencyScore', 'FinancialStressIndex',
    'AbsoluteMonthlyDebt', 'DelinquencyTrend',
]
TARGET = 'SeriousDlqin2yrs'

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# Load best model (XGBoost)
xgb_model = joblib.load(MODELS_DIR / "best_model.pkl")
print(f"  XGBoost loaded: {xgb_model.__class__.__name__}")

# Load các models khác nếu có (nếu đã lưu riêng)
# Nếu chưa có, sẽ skip DeLong test và chỉ chạy calibration cho XGBoost
models = {'XGBoost': xgb_model}

for name, fname in [('RF', 'model_rf'), ('LR', 'model_lr'), ('DT', 'model_dt')]:
    path = MODELS_DIR / f"{fname}.pkl"
    if path.exists():
        models[name] = joblib.load(path)
        print(f"  {name} loaded from {path}")
    else:
        print(f"  {name} not found at {path} — skipping")


# ─── 2. DeLong Test: XGBoost vs RF ───────────────────────────────────────────

print("\n--- DeLong Test (XGBoost vs RF) ---")

delong_results = {}
if 'RF' in models:
    proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    proba_rf  = models['RF'].predict_proba(X_test)[:, 1]

    result = delong_test(y_test.values, proba_xgb, proba_rf)
    delong_results['XGBoost_vs_RF'] = result

    print(f"  AUC XGBoost: {result['auc_a']:.4f}")
    print(f"  AUC RF:      {result['auc_b']:.4f}")
    print(f"  Δ AUC:       {result['auc_diff']:+.4f}")
    print(f"  z-stat:      {result['z_stat']:.4f}")
    print(f"  p-value:     {result['p_value']:.4f}")
    print(f"  Significant (p<0.05): {result['significant']}")
else:
    print("  RF model not available — run notebook 03 and save with src.models.save_model('rf', 'model_rf')")
    print("  Providing theoretical estimate: with ΔAUC=0.0011 on n=22,500, expected p >> 0.05 (not significant)")
    delong_results['XGBoost_vs_RF'] = {
        'auc_a': 0.8714, 'auc_b': 0.8703,
        'auc_diff': 0.0011,
        'z_stat': float('nan'), 'p_value': float('nan'),
        'significant': False,
        'note': 'RF model not saved — run notebook 03 to compute exact values'
    }


# ─── 3. Calibration Analysis ─────────────────────────────────────────────────

print("\n--- Calibration Analysis ---")

calibration_results = {}
for name, model in models.items():
    cal = compute_calibration_metrics(model, X_test, y_test)
    calibration_results[name] = cal
    print(f"  {name}: Brier={cal['brier_score']:.4f} | BSS={cal['brier_skill_score']:.4f} | ECE={cal['ece']:.4f}")


# ─── 4. Figures ──────────────────────────────────────────────────────────────

# Figure 31: Reliability Diagram
print("\nGenerating figures...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Reliability diagram tất cả models
plot_calibration_curve(models, X_test, y_test, n_bins=10, ax=axes[0])
axes[0].set_title('Hình 4.9: Reliability Diagram\n(đường chéo = calibration hoàn hảo)', fontweight='bold')

# Right: Brier Score bar chart
names = list(calibration_results.keys())
brier_scores = [calibration_results[n]['brier_score'] for n in names]
bss_scores   = [calibration_results[n]['brier_skill_score'] for n in names]
brier_ref    = calibration_results[names[0]]['brier_ref']

x = np.arange(len(names))
bars = axes[1].bar(x, brier_scores, color=['#3498db', '#e67e22', '#2ecc71', '#e74c3c'][:len(names)],
                   alpha=0.8, edgecolor='white', linewidth=1.5)
axes[1].axhline(brier_ref, color='red', linestyle='--', alpha=0.6,
                label=f'Baseline (predict prevalence) = {brier_ref:.4f}')
axes[1].set_xticks(x)
axes[1].set_xticklabels(names, fontsize=11)
axes[1].set_ylabel('Brier Score (thấp hơn = tốt hơn)', fontweight='bold')
axes[1].set_title('Brier Score so với Baseline', fontweight='bold')
axes[1].legend(fontsize=9)
for bar, bss in zip(bars, bss_scores):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                 f'BSS={bss:.3f}', ha='center', va='bottom', fontsize=9)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
fig.savefig(REPORTS_DIR / "fig_31_calibration.png", bbox_inches='tight', dpi=110)
plt.close(fig)
print(f"  Saved: reports/fig_31_calibration.png")


# Figure 32: DeLong summary bảng (nếu có RF)
if 'XGBoost_vs_RF' in delong_results and not np.isnan(delong_results['XGBoost_vs_RF']['z_stat']):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    r = delong_results['XGBoost_vs_RF']
    table_data = [
        ['Model A', 'Model B', 'AUC A', 'AUC B', 'Δ AUC', 'z-stat', 'p-value', 'Kết luận'],
        ['XGBoost', 'RF',
         f"{r['auc_a']:.4f}", f"{r['auc_b']:.4f}",
         f"{r['auc_diff']:+.4f}",
         f"{r['z_stat']:.4f}", f"{r['p_value']:.4f}",
         'Không sig. (p>0.05)' if not r['significant'] else 'Sig. (p<0.05)']
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    ax.set_title('DeLong Test — XGBoost vs Random Forest (same test set)', fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "fig_32_delong_summary.png", bbox_inches='tight', dpi=110)
    plt.close(fig)
    print(f"  Saved: reports/fig_32_delong_summary.png")


# ─── 5. Ghi kết quả ra addendum_results.md ───────────────────────────────────

lines = [
    "# Addendum Results — DeLong Test & Calibration\n",
    f"**Test set:** {len(X_test):,} samples ({y_test.mean()*100:.2f}% positive)\n\n",
    "## DeLong Test: XGBoost vs RF\n\n",
]

r = delong_results['XGBoost_vs_RF']
if np.isnan(r.get('z_stat', float('nan'))):
    lines.append(f"⚠️ RF model chưa được lưu. {r.get('note', '')}\n\n")
else:
    lines += [
        f"| Metric | Giá trị |\n|--------|--------|\n",
        f"| AUC XGBoost | {r['auc_a']:.4f} |\n",
        f"| AUC RF | {r['auc_b']:.4f} |\n",
        f"| Δ AUC | {r['auc_diff']:+.4f} |\n",
        f"| z-statistic | {r['z_stat']:.4f} |\n",
        f"| p-value (two-sided) | {r['p_value']:.4f} |\n",
        f"| Kết luận | {'Có ý nghĩa thống kê (p<0.05)' if r['significant'] else 'Không có ý nghĩa thống kê (p>0.05)'} |\n\n",
    ]

lines.append("## Calibration Analysis\n\n")
lines.append("| Model | Brier Score | Brier Skill Score | ECE |\n|-------|-------------|------------------|-----|\n")
for name, cal in calibration_results.items():
    lines.append(f"| {name} | {cal['brier_score']:.4f} | {cal['brier_skill_score']:.4f} | {cal['ece']:.4f} |\n")

lines += [
    f"\n**Baseline Brier Score** (predict prevalence {y_test.mean()*100:.2f}%): {calibration_results[names[0]]['brier_ref']:.4f}\n",
    "\n**Brier Skill Score > 0** = model tốt hơn baseline; **ECE nhỏ hơn** = calibration tốt hơn.\n",
]

(REPORTS_DIR / "addendum_results.md").write_text("".join(lines), encoding='utf-8')
print(f"  Saved: reports/addendum_results.md")

print("\nDone! Paste numbers from reports/addendum_results.md vào bao_cao_chinh.md §4.6.1 và §4.9.")
