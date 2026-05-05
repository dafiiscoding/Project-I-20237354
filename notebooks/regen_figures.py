"""
regen_figures.py — Re-generate tat ca figures voi plot_style.py (Vietnamese labels + consistent DPI).

Chay script nay de:
1. Load data tu data/splits/
2. Load models tu models/*.pkl
3. Gen lai tat ca figures voi labels tieng Viet
4. Luu PNG vao reports/ (overwrite) + reports/visual_summary/ (anh moi)

Khong can re-train model — chi ve lai.
"""

import os
import sys
import warnings
import io
import codecs
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Force UTF-8 encoding for stdout
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from plot_style import setup_vietnamese_style, LABELS_VI, PALETTE_MODELS, RISK_COLORS, translate_label
from evaluation import (
    plot_confusion_matrix,
    plot_roc_pr_curves,
    plot_overlay_roc,
    plot_overlay_pr,
    plot_calibration_curve,
    compare_models,
)

# Setup directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'splits'
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'
VISUAL_SUMMARY_DIR = REPORTS_DIR / 'visual_summary'

# Create visual_summary directory if not exists
VISUAL_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("REGENERATING FIGURES WITH VIETNAMESE LABELS")
print("=" * 80)

# Setup style
setup_vietnamese_style(dpi=150)
print("[OK] Vietnamese style setup done")

# Load data
print("\nLoading data...")
test_data = pd.read_csv(DATA_DIR / 'test.csv')
y_col = 'SeriousDlqin2yrs'
drop_cols = [y_col, 'Unnamed: 0']
if y_col in test_data.columns:
    y_test = test_data[y_col]
    X_test = test_data.drop(columns=[c for c in drop_cols if c in test_data.columns])
else:
    print(f"ERROR: {y_col} column not found in test.csv")
    sys.exit(1)
print(f"  Test set: {len(X_test)} samples, {len(X_test.columns)} features")
# Fix column name mismatch (data splits use DebtToIncomeRatio, model expects AbsoluteMonthlyDebt)
if 'DebtToIncomeRatio' in X_test.columns and 'AbsoluteMonthlyDebt' not in X_test.columns:
    X_test = X_test.rename(columns={'DebtToIncomeRatio': 'AbsoluteMonthlyDebt'})
    print("  [INFO] Renamed DebtToIncomeRatio -> AbsoluteMonthlyDebt to match model")

# Load models
print("\nLoading models...")
models = {}
for model_file in ['model_lr.pkl', 'model_dt.pkl', 'model_rf.pkl', 'best_model.pkl']:
    path = MODELS_DIR / model_file
    if path.exists():
        models[model_file.replace('model_', '').replace('.pkl', '')] = joblib.load(path)
        print(f"  [OK] {model_file}")
    else:
        print(f"  [SKIP] {model_file} NOT FOUND")

if not models:
    print("ERROR: No models loaded! Check models/ directory.")
    sys.exit(1)

# Alias best_model as xgb for consistency
if 'best_model' in models:
    models['xgb'] = models.pop('best_model')

print(f"\nLoaded models: {', '.join(models.keys())}")

# ───────────────────────────────────────────────────────────────────────────
# Generate figures
# ───────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("GENERATING FIGURES")
print("=" * 80)

# 1. Confusion matrices
print("\n1. Confusion matrices (each model)...")
threshold = 0.625
for name, model in models.items():
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_confusion_matrix(
        model, X_test, y_test,
        threshold=threshold,
        title=f'Ma trận nhầm lẫn — {name.upper()} (Threshold={threshold})',
        ax=ax
    )
    plt.tight_layout()
    save_path = REPORTS_DIR / f'fig_confusion_{name}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  [OK] {save_path.name}")

# 2. ROC curves (overlay)
print("\n2. ROC curves (overlay all models)...")
fig, ax = plt.subplots(figsize=(7, 6))
plot_overlay_roc(models, X_test, y_test, ax=ax)
plt.tight_layout()
save_path = REPORTS_DIR / 'fig_roc_comparison.png'
plt.savefig(save_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"  ✓ {save_path.name}")

# 3. Precision-Recall curves (overlay)
print("\n3. Precision-Recall curves (overlay all models)...")
fig, ax = plt.subplots(figsize=(7, 6))
plot_overlay_pr(models, X_test, y_test, ax=ax)
plt.tight_layout()
save_path = REPORTS_DIR / 'fig_pr_comparison.png'
plt.savefig(save_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"  ✓ {save_path.name}")

# 4. Calibration curves (overlay)
print("\n4. Calibration curves (overlay all models)...")
fig, ax = plt.subplots(figsize=(7, 6))
plot_calibration_curve(models, X_test, y_test, n_bins=10, ax=ax)
plt.tight_layout()
save_path = REPORTS_DIR / 'fig_calibration_comparison.png'
plt.savefig(save_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"  ✓ {save_path.name}")

# 5. Visual summary figures (new)
print("\n5. Visual summary figures (new)...")

# 5a. Pipeline overview (simple schematic)
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

steps = ['Dữ liệu', 'EDA', 'Tiền xử lý', 'Mô hình', 'SHAP', 'Ứng dụng']
x_pos = np.linspace(0.1, 0.9, len(steps))
y_pos = 0.5

for i, (x, step) in enumerate(zip(x_pos, steps)):
    # Box
    rect = plt.Rectangle((x - 0.05, y_pos - 0.1), 0.1, 0.2, fill=True, color='#3498db', alpha=0.7)
    ax.add_patch(rect)
    # Text
    ax.text(x, y_pos, step, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    # Arrow
    if i < len(steps) - 1:
        ax.arrow(x + 0.06, y_pos, 0.08, 0, head_width=0.05, head_length=0.02, fc='gray', ec='gray')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Pipeline Loan Default Prediction', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
save_path = VISUAL_SUMMARY_DIR / 'fig_vs_01_pipeline_overview.png'
plt.savefig(save_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"  ✓ {save_path.name}")

# 5b. KPI callout banner (simple text figure)
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

kpis = [
    ('AUC', '0.8714'),
    ('Recall', '66.9%'),
    ('Threshold', '0.625'),
]

y_start = 0.7
for i, (key, val) in enumerate(kpis):
    y = y_start - i * 0.25
    ax.text(0.15, y, key, fontsize=11, color='gray')
    ax.text(0.4, y, val, fontsize=16, fontweight='bold', color='#e74c3c')

ax.text(0.5, 0.8, 'Chỉ số chính', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
save_path = VISUAL_SUMMARY_DIR / 'fig_vs_02_kpi_callout.png'
plt.savefig(save_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"  ✓ {save_path.name}")

# 5c. What-if scenarios (3 customer cards)
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

scenarios = [
    ('An toàn', 'Prob: 5%', '#2ecc71'),
    ('Biên', 'Prob: 45%', '#f39c12'),
    ('Rủi ro cao', 'Prob: 85%', '#e74c3c'),
]

x_positions = [0.2, 0.5, 0.8]
for x, (label, prob, color) in zip(x_positions, scenarios):
    # Card background
    rect = plt.Rectangle((x - 0.15, 0.2), 0.3, 0.6, fill=True, color=color, alpha=0.2, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    # Text
    ax.text(x, 0.65, label, ha='center', fontsize=12, fontweight='bold')
    ax.text(x, 0.35, prob, ha='center', fontsize=14, fontweight='bold', color=color)

ax.text(0.5, 0.95, 'Các tình huống khách hàng', fontsize=14, fontweight='bold', ha='center')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
save_path = VISUAL_SUMMARY_DIR / 'fig_vs_03_what_if_scenarios.png'
plt.savefig(save_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"  ✓ {save_path.name}")

# 6. fig_04 split: bivariate boxplots (was 5x2 -> split to 3 figures)
print("\n6. fig_04 split - bivariate boxplots by target class...")
try:
    raw_data = pd.read_csv(PROJECT_ROOT / 'data' / 'raw' / 'cs-training.csv')
    # Cap outliers at 1%–99% for visualization
    def cap_series(s, low=0.01, high=0.99):
        return s.clip(s.quantile(low), s.quantile(high))

    target_col = 'SeriousDlqin2yrs'
    if target_col not in raw_data.columns:
        raise ValueError(f"{target_col} not in raw data")

    tick_labels = ['Không vỡ nợ', 'Vỡ nợ']

    # fig_04a: delinquency features (1x3)
    delinq_features = [
        ('NumberOfTime30-59DaysPastDueNotWorse', 'Trễ 30-59 ngày'),
        ('NumberOfTime60-89DaysPastDueNotWorse', 'Trễ 60-89 ngày'),
        ('NumberOfTimes90DaysLate', 'Trễ 90+ ngày'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Phân bố đặc trưng trễ hạn theo nhóm mục tiêu\n(Giới hạn 1%-99%)', fontweight='bold', fontsize=13)
    for ax, (col, label) in zip(axes, delinq_features):
        if col in raw_data.columns:
            data_plot = [
                cap_series(raw_data[raw_data[target_col] == 0][col].dropna()),
                cap_series(raw_data[raw_data[target_col] == 1][col].dropna()),
            ]
            ax.boxplot(data_plot, labels=tick_labels, patch_artist=True,
                       boxprops=dict(facecolor='#a8d8ea', alpha=0.7),
                       medianprops=dict(color='#e74c3c', linewidth=2))
            ax.set_title(label, fontsize=11)
            ax.set_ylabel('Giá trị (giới hạn 1%-99%)')
    plt.tight_layout()
    save_path = REPORTS_DIR / 'fig_04a_delinquency_boxplots.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  [OK] {save_path.name}")

    # fig_04b: financial features (1x3)
    fin_features = [
        ('RevolvingUtilizationOfUnsecuredLines', 'Tỷ lệ sử dụng hạn mức'),
        ('DebtRatio', 'Tỷ lệ nợ'),
        ('MonthlyIncome', 'Thu nhập hàng tháng'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Phân bố đặc trưng tài chính theo nhóm mục tiêu\n(Giới hạn 1%-99%)', fontweight='bold', fontsize=13)
    for ax, (col, label) in zip(axes, fin_features):
        if col in raw_data.columns:
            data_plot = [
                cap_series(raw_data[raw_data[target_col] == 0][col].dropna()),
                cap_series(raw_data[raw_data[target_col] == 1][col].dropna()),
            ]
            ax.boxplot(data_plot, labels=tick_labels, patch_artist=True,
                       boxprops=dict(facecolor='#f9d976', alpha=0.7),
                       medianprops=dict(color='#e74c3c', linewidth=2))
            ax.set_title(label, fontsize=11)
            ax.set_ylabel('Giá trị (giới hạn 1%-99%)')
    plt.tight_layout()
    save_path = REPORTS_DIR / 'fig_04b_financial_boxplots.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  [OK] {save_path.name}")

    # fig_04c: demographic/other features (2x2)
    other_features = [
        ('age', 'Tuổi'),
        ('NumberOfOpenCreditLinesAndLoans', 'Số dòng tín dụng mở'),
        ('NumberRealEstateLoansOrLines', 'Số khoản vay BĐS'),
        ('NumberOfDependents', 'Số người phụ thuộc'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Phân bố đặc trưng nhân khẩu học theo nhóm mục tiêu\n(Giới hạn 1%-99%)', fontweight='bold', fontsize=13)
    for ax, (col, label) in zip(axes.flat, other_features):
        if col in raw_data.columns:
            data_plot = [
                cap_series(raw_data[raw_data[target_col] == 0][col].dropna()),
                cap_series(raw_data[raw_data[target_col] == 1][col].dropna()),
            ]
            ax.boxplot(data_plot, labels=tick_labels, patch_artist=True,
                       boxprops=dict(facecolor='#b8f0b8', alpha=0.7),
                       medianprops=dict(color='#e74c3c', linewidth=2))
            ax.set_title(label, fontsize=11)
            ax.set_ylabel('Giá trị (giới hạn 1%-99%)')
    plt.tight_layout()
    save_path = REPORTS_DIR / 'fig_04c_other_boxplots.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  [OK] {save_path.name}")

except Exception as e:
    print(f"  [SKIP] fig_04 split failed: {e}")

print("\n" + "=" * 80)
print("[SUCCESS] ALL FIGURES REGENERATED")
print("=" * 80)
print(f"\nFigures saved to:")
print(f"  - {REPORTS_DIR}")
print(f"  - {VISUAL_SUMMARY_DIR}")
