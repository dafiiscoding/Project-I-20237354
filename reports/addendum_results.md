# Addendum Results — DeLong Test & Calibration
**Test set:** 22,500 samples (6.68% positive)

## DeLong Test: XGBoost vs RF

**Lưu ý tái lập:** Bảng so sánh mô hình chính trong `reports/model_results.csv` ghi RF test AUC = 0.8703. Riêng kiểm định DeLong dưới đây dùng RF được huấn luyện lại như một model phụ nội bộ để tạo xác suất so sánh, nên RF AUC = 0.8671. Vì vậy ΔAUC = +0.0043 chỉ áp dụng cho cặp model trong addendum, không phải chênh lệch trực tiếp với dòng RF trong bảng chính.

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
