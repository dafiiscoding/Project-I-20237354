# Addendum Results — DeLong Test & Calibration
**Test set:** 22,500 samples (6.68% positive)

## DeLong Test: XGBoost vs RF

**Lưu ý tái lập:** Các con số dưới đây tính từ đúng các model đã lưu trong `models/` (RF test AUC = 0.8671), nhất quán với `reports/model_results.csv` (đã được tái sinh từ cùng bộ model bằng `final_report/regen_figs.py`) và với bảng so sánh trong báo cáo chính. Bản `model_results.csv` cũ (RF = 0.8703, LR = 0.8432) là output của một lần chạy notebook trước khi lưu model và đã bị thay thế.

| Metric | Giá trị |
|--------|--------|
| AUC XGBoost | 0.8714 |
| AUC RF | 0.8671 |
| Δ AUC | +0.0043 |
| z-statistic | 4.1833 |
| p-value (two-sided) | 0.0000 |
| Kết luận | Có ý nghĩa thống kê cho cặp XGBoost và RF tái huấn luyện trong addendum (p<0.05) |

## Calibration Analysis

| Model | Brier Score | Brier Skill Score | ECE |
|-------|-------------|------------------|-----|
| XGBoost | 0.1388 | -1.2253 | 0.3590 |
| RF | 0.1189 | -0.9058 | 0.3286 |
| LR | 0.1573 | -1.5214 | 0.3764 |
| DT | 0.1458 | -1.3380 | 0.3695 |

**Baseline Brier Score** (predict prevalence 6.68%): 0.0624

**Brier Skill Score > 0** = model tốt hơn baseline; **ECE nhỏ hơn** = calibration tốt hơn.
