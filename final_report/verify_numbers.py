"""Tái tính các con số dùng trong bao_cao_chinh.tex từ model + splits đã lưu.

Chạy:  python verify_numbers.py
In ra: AUC test, bảng ngưỡng (0.50 / F2-opt / F1-opt / Bayes 0.043) với
Recall/Precision/F1/F2, TP/FP/FN/TN, tỷ lệ từ chối, chi phí (c_FN=11.250,
c_FP=500 USD), Brier/ECE trước và sau Platt scaling (khớp trên tập val).
Mọi số trong chương Thực nghiệm của báo cáo phải khớp output này.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
C_FN, C_FP = 11_250, 500

FEATURES = [
    'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents', 'TotalDelinquencyScore', 'FinancialStressIndex',
    'AbsoluteMonthlyDebt', 'DelinquencyTrend',
]


def load_split(name: str):
    df = pd.read_csv(ROOT / 'data' / 'splits' / f'{name}.csv')
    # splits cũ lưu DebtToIncomeRatio, model cần AbsoluteMonthlyDebt
    if 'AbsoluteMonthlyDebt' not in df.columns:
        df['AbsoluteMonthlyDebt'] = df['DebtRatio'] * df['MonthlyIncome']
    return df[FEATURES], df['SeriousDlqin2yrs'].to_numpy()


def metrics_at(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if prec + rec else 0.0
    return dict(t=t, recall=rec, precision=prec, f1=f1, f2=f2,
                tp=tp, fp=fp, fn=fn, tn=tn,
                reject=(tp + fp) / len(y),
                cost=(fn * C_FN + fp * C_FP) / 1e6)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return total


def main() -> None:
    model = pickle.load(open(ROOT / 'models' / 'best_model.pkl', 'rb'))
    Xv, yv = load_split('val')
    Xt, yt = load_split('test')
    pv = model.predict_proba(Xv)[:, 1]
    pt = model.predict_proba(Xt)[:, 1]

    print(f'Test: n={len(yt)}, duong={yt.sum()} ({yt.mean():.4%})')
    print(f'AUC test = {roc_auc_score(yt, pt):.4f}')
    print(f'AUC val  = {roc_auc_score(yv, pv):.4f}')

    # quét ngưỡng trên VAL để xác nhận F2-opt / F1-opt
    grid = np.arange(0.05, 0.951, 0.005)
    mv = [metrics_at(yv, pv, t) for t in grid]
    t_f2 = max(mv, key=lambda d: d['f2'])['t']
    t_f1 = max(mv, key=lambda d: d['f1'])['t']
    print(f'F2-opt (val) = {t_f2:.3f} | F1-opt (val) = {t_f1:.3f}')

    print('\n--- Bang nguong (TEST) ---')
    hdr = ('t', 'recall', 'precision', 'f1', 'f2', 'tp', 'fp', 'fn', 'tn',
           'reject', 'cost(trUSD)')
    print(('{:>8}' * len(hdr)).format(*hdr))
    for t in (0.50, round(t_f2, 3), round(t_f1, 3), 0.043):
        d = metrics_at(yt, pt, t)
        print(f"{d['t']:8.3f}{d['recall']:8.3f}{d['precision']:8.3f}"
              f"{d['f1']:8.3f}{d['f2']:8.3f}{d['tp']:8d}{d['fp']:8d}"
              f"{d['fn']:8d}{d['tn']:8d}{d['reject']:8.3f}{d['cost']:8.2f}")

    # hieu chinh Platt: fit logistic 1 bien tren VAL, ap len TEST
    platt = LogisticRegression(C=1e10, solver='lbfgs')
    platt.fit(pv.reshape(-1, 1), yv)
    pt_cal = platt.predict_proba(pt.reshape(-1, 1))[:, 1]
    base = np.full_like(pt, yt.mean(), dtype=float)
    print('\n--- Hieu chinh xac suat (TEST) ---')
    print(f'Brier truoc Platt = {brier_score_loss(yt, pt):.4f}')
    print(f'Brier sau  Platt = {brier_score_loss(yt, pt_cal):.4f}')
    print(f'Brier nen (du bao 6.68%) = {brier_score_loss(yt, base):.4f}')
    print(f'ECE truoc = {ece(yt, pt):.4f} | ECE sau = {ece(yt, pt_cal):.4f}')
    print(f'AUC sau Platt (bat bien don dieu) = {roc_auc_score(yt, pt_cal):.4f}')


if __name__ == '__main__':
    main()
