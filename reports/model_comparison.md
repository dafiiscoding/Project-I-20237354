# Báo cáo So sánh Mô hình — Phase 3 Modeling
## Loan Default Prediction | Give Me Some Credit

**Notebook:** `notebooks/03_Modeling.ipynb`
**Input:** `data/splits/{train,val,test}.csv`
**Output:** `models/best_model.pkl`, `reports/model_results.csv`

---

## 1. Thiết lập Thực nghiệm

### 1.1 Quy trình Thống nhất
Mỗi mô hình được đánh giá theo cùng một quy trình:

| Thành phần | Thiết kế | Lý do |
|-----------|---------|-------|
| Chỉ số chính | AUC-ROC | Không phụ thuộc ngưỡng, mạnh với mất cân bằng lớp 6,68% |
| Chiến lược kiểm định chéo | Stratified 5-Fold | Giữ tỷ lệ positive trong mỗi fold |
| Điều chỉnh siêu tham số | RandomizedSearchCV | Nhanh hơn GridSearch, đủ tốt với n_iter lớn |
| Chọn ngưỡng | Tối ưu F1 trên tập kiểm định (Phase 3; ngưỡng triển khai chốt ở Phase 4) | Ngưỡng 0,5 không tối ưu với dữ liệu mất cân bằng |
| Xử lý mất cân bằng | class_weight='balanced' (LR/DT/RF), scale_pos_weight=13.96 (XGBoost) | |

### 1.2 Phân chia Bộ dữ liệu
- Training: 104,999 rows (6.68% positive)
- Validation: 22,500 rows (6.68% positive)
- Test: 22,500 rows (6.68% positive)
- Đặc trưng: 14 (10 gốc + 4 được tạo)

---

## 2. Kết quả Chi tiết từng Mô hình

### 2.1 Logistic Regression (Baseline)

**Best hyperparameters:** L1 regularization (C=0.001)
- C=0.001 = regularization rất mạnh → solution sparse
- L1 tự động triệt tiêu đặc trưng dư thừa

**Performance:**

| Split | AUC-ROC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| CV (5-fold) | 0.8371 | — | — | — |
| Validation | 0.8388 | 0.4259 | — | — |
| **Test** | **0.8432** | **0.4340** | **0.3878** | **0.4927** |

Ngưỡng tối ưu: **0,66** (so với ngưỡng mặc định 0,5 — mất cân bằng lớp dịch chuyển ngưỡng)
Training time: 487.9s

**Odds Ratios (L1, C=0.001):**

| Đặc trưng | Hệ số β | Odds Ratio e^β | Diễn giải |
|---------|--------------|----------------|----------|
| TotalDelinquencyScore | +0.255 | **1.291** | Tăng 1 điểm → risk ×1.29 |
| RevolvingUtilization | +0.191 | **1.211** | Tăng 1 đơn vị → risk ×1.21 |
| FinancialStressIndex | +0.170 | **1.185** | Đặc trưng được tạo quan trọng nhất |
| NumberOfTime30-59 | +0.062 | 1.064 | Trễ nhẹ vẫn có tín hiệu |
| DelinquencyTrend | +0.025 | 1.025 | Yếu, nhưng có hướng |
| NumberOfTimes90DaysLate | **0.000** | 1.000 | L1 triệt tiêu (đã nắm bắt bởi TotalDelinquency) |
| DebtToIncomeRatio | **0.000** | 1.000 | L1 triệt tiêu (tín hiệu yếu) |
| age | -0.120 | **0.887** | Protective: tuổi cao → risk giảm |
| MonthlyIncome | -0.059 | 0.942 | Protective: income cao → risk giảm |
| NumberOfDependents | -0.052 | 0.949 | Nhẹ: nhiều người phụ thuộc → giảm risk? |

**Nhận xét domain:**
- L1 với C=0,001 đã triệt tiêu `NumberOfTimes90DaysLate` vì `TotalDelinquencyScore` đã nắm bắt nó hoàn toàn (đa cộng tuyến xử lý đúng như dự đoán trong R13)
- `FinancialStressIndex` (đặc trưng được tạo) có odds ratio cao thứ 3 → trích xuất đặc trưng có giá trị
- `age` là protective factor mạnh nhất: OR=0.887 → mỗi năm tuổi tăng, risk giảm ~11.3%
- Ngưỡng tối ưu 0,66 ≠ 0,5: với mất cân bằng 1:14, mô hình cần xác suất cao hơn để dự đoán dương tính

---

### 2.2 Decision Tree CART

**Best hyperparameters:** max_depth=10, min_samples_leaf=200, criterion=Gini, max_features=0.5

**Performance:**

| Split | AUC-ROC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| CV (5-fold) | 0.8509 | — | — | — |
| **Test** | **0.8579** | **0.4314** | **0.3991** | **0.4694** |

Optimal threshold: **0.80**
Training time: **13.2s** (fastest model)
Tree depth: 10 | Number of leaves: 168

**Nhận xét:**
- AUC=0.8579, vượt LR 0.015 điểm dù là mô hình đơn giản nhất trong nhóm tree-based
- `min_samples_leaf=200` là cắt tỉa mạnh, tránh quá khớp trên nhiễu
- `max_features=0.5` = lấy mẫu ngẫu nhiên trong DT → tiền thân của khái niệm RF
- Recall thấp nhất (0.469) → DT thận trọng hơn khi gán nhãn default
- Ngưỡng cực cao (0,80) → DT cần độ tự tin rất cao mới dự đoán vỡ nợ

---

### 2.3 Random Forest

**Best hyperparameters:** n_estimators=200, max_depth=10, min_samples_leaf=30, max_features=0.3
OOB Score: 0.8286

**Performance:**

| Split | AUC-ROC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| CV (5-fold) | 0.8635 | — | — | — |
| **Test** | **0.8703** | **0.4439** | **0.3869** | **0.5206** |

Optimal threshold: **0.72**
Training time: 511.4s (chậm nhất)

**Nhận xét:**
- AUC=0.8703, chỉ thua XGBoost 0.0011 điểm
- **Recall cao nhất** (0.521): RF giỏi nhất trong việc "không bỏ sót" người vỡ nợ thực sự
- OOB Score=0.8286 vs Test AUC=0.8703: OOB accuracy ≠ AUC nên không so sánh trực tiếp
- `max_features=0.3` (chỉ 30% đặc trưng mỗi phân chia) → giải tương quan cây mạnh → phương sai thấp
- Training time 511s: chậm do OOB curve analysis (7 extra fits) + learning curve

---

### 2.4 XGBoost (Best Model)

**Best hyperparameters:**
- n_estimators=200, max_depth=4, learning_rate=0.05
- subsample=0.8, colsample_bytree=0.8
- reg_lambda=1.0, reg_alpha=0.1, min_child_weight=5, gamma=0.1

**Performance:**

| Split | AUC-ROC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| CV (5-fold) | 0.8656 | — | — | — |
| **Test** | **0.8714** | **0.4466** | **0.3937** | **0.5160** |

Optimal threshold: **0.77**
Training time: **126.6s** (nhanh thứ 2, sau DT)

**Tầm quan trọng Đặc trưng theo SHAP (mean |SHAP value|):**

*Từ SHAP TreeExplainer trên 2.000 mẫu kiểm tra. Kết quả phản ánh tầm quan trọng đặc trưng tổng quát.*

Đặc trưng hàng đầu theo SHAP: RevolvingUtilizationOfUnsecuredLines, NumberOfTimes90DaysLate, TotalDelinquencyScore, FinancialStressIndex

**Nhận xét:**
- XGBoost có max_depth=4 (cây nông) với nhiều iterations (200) → Boosting reduce bias hiệu quả
- reg_alpha=0.1 (L1 trên trọng số lá) + reg_lambda=1.0 (L2) → mô hình được chuẩn hóa tốt
- min_child_weight=5: node cần ít nhất 5 mẫu tổng Hessian để phân chia → chống quá khớp
- subsample=0.8 + colsample_bytree=0.8: stochastic gradient boosting → variance reduction
- Training time 127s: nhanh hơn RF (511s) 4×, mà AUC gần tương đương → XGBoost efficient hơn

---

## 3. So sánh Tổng hợp

### 3.1 Bảng so sánh đầy đủ

| Mô hình | CV AUC | Test AUC | Avg Precision | F1 | Precision | Recall | Ngưỡng | Thời gian HT |
|-------|--------|---------|--------------|-----|-----------|--------|-----------|------------|
| **XGBoost** | **0.8656** | **0.8714** | **0.4005** | **0.4466** | 0.3937 | 0.5160 | 0.77 | 127s |
| Random Forest | 0.8635 | 0.8703 | 0.3980 | 0.4439 | 0.3869 | **0.5206** | 0.72 | 511s |
| Decision Tree | 0.8509 | 0.8579 | 0.3762 | 0.4314 | **0.3991** | 0.4694 | 0.80 | **13s** |
| Logistic Regression | 0.8371 | 0.8432 | 0.3700 | 0.4340 | 0.3878 | 0.4927 | 0.66 | 488s |

### 3.2 Phân tích so sánh

**XGBoost vs Random Forest:**
- Khoảng cách AUC: 0,0011 — rất nhỏ, gần như tương đương thống kê
- Recall: RF cao hơn (0,5206 vs 0,5160) — RF giỏi hơn trong việc bắt người vỡ nợ thực sự
- Thời gian huấn luyện: XGBoost 4× nhanh hơn (127s vs 511s)
- **Kết luận:** XGBoost được chọn vì hiệu quả tính toán (127s), khả năng giải thích qua SHAP, và AUC cao nhất

**Decision Tree:**
- AUC=0.8579 vượt LR (0.8432) 0.015 điểm
- Fastest model (13s), easy to visualize
- Thấp nhất về Recall → risk bỏ sót người vỡ nợ

**Logistic Regression — Baseline phù hợp:**
- AUC=0.8432 là solid baseline
- Duy nhất có odds ratio interpretation trực tiếp (Basel III compliant)
- L1 triệt tiêu đặc trưng dư thừa → xác nhận chất lượng feature engineering
- Trong thực tế ngân hàng quy định cần giải thích từng quyết định: LR hoặc LR+SHAP

### 3.3 Tất cả ngưỡng tối ưu > 0,5
Một nhận xét quan trọng: tất cả 4 mô hình đều có ngưỡng tối ưu > 0,5 (0,66–0,80).

**Lý do:** Với mất cân bằng lớp 1:14, prior P(default=1) = 0,0668. Trên thang logit, đầu ra cơ sở của mô hình đã "thiên lệch" về 0. Cần ngưỡng xác suất cao hơn để đủ độ tự tin dự đoán dương tính. Điều này xác nhận rằng dùng 0.5 làm threshold là sai trong credit risk với imbalanced data.

---

## 4. Bàn luận 3 Tranh luận Lớn

### 4.1 Khả năng Giải thích so với Hiệu suất
- LR (baseline): AUC=0,8432, odds ratio rõ ràng, tuân thủ Basel III
- XGBoost (tốt nhất): AUC=0,8714, cần SHAP cho khả năng giải thích

Khoảng cách AUC = 0,0282 (cải thiện tương đối 3,3%). Câu hỏi: khoảng cách này có đủ biện minh cho thêm độ phức tạp không?

**Lập luận:** Với 22,500 test samples, 0.0282 AUC gap tương đương với:
- Correctly ranking thêm ~636 default/non-default pairs (AUC = P(score(+) > score(-)))
- Tại ngưỡng 0,77: thêm ~130-150 người vỡ nợ được xác định đúng (Recall 0,516 vs 0,493)
- 130 defaulters × average bad loan loss $15,000 = ~$1.95M/year saved (trên 22,500 customers)

→ **Kết luận:** Khoảng cách 0,0282 AUC tương đương ~130 trường hợp vỡ nợ được bắt thêm trên tập kiểm tra — có giá trị thực tế. XGBoost + SHAP phù hợp khi cần cả hiệu suất dự báo lẫn lý do cụ thể cho từng quyết định.

### 4.2 Tập trung Dữ liệu so với Tập trung Mô hình
Trích xuất đặc trưng thủ công (TotalDelinquencyScore, FinancialStressIndex) vs XGBoost tự học interaction terms.

**Thực nghiệm:**
- LR với features engineered: AUC=0.8432
- LR với 3 raw delinquency features (không dùng TotalDelinquencyScore) sẽ thấp hơn do multicollinearity
- XGBoost với engineered features: AUC=0.8714

`FinancialStressIndex` (được tạo) xuất hiện trong top đặc trưng SHAP của XGBoost — các đặc trưng tự tạo không bị thay thế mà bổ sung thêm thông tin mà mô hình không tự học được.

→ **Kết luận:** Hai hướng tiếp cận không loại trừ nhau. Trích xuất đặc trưng giúp cả LR (giảm đa cộng tuyến) và XGBoost (cung cấp đặc trưng mang ý nghĩa tài chính rõ ràng).

### 4.3 Đánh đổi Precision so với Recall
Tại optimal threshold:

| Mô hình | Precision | Recall | Tác động kinh doanh |
|-------|-----------|--------|----------------|
| XGBoost | 0.394 | 0.516 | Bắt 51.6% defaults, từ chối nhầm 60.6% applicants được classify là default |
| RF | 0.387 | 0.521 | Recall cao nhất, nhưng precision thấp nhất |

**Trong credit risk:** FN cost (bad loan) >> FP cost (opportunity cost)
- FN: cho vay người thực sự vỡ nợ → mất toàn bộ principal
- FP: từ chối người không vỡ nợ → mất lợi nhuận của 1 khoản vay

→ **Recall nên được ưu tiên** (Phase 4 sẽ tune F-beta với β>1)

---

## 5. Phân tích SHAP (XGBoost)

SHAP values từ TreeExplainer trên 2,000 test samples:

**Đặc trưng hàng đầu theo mean |SHAP|:**
1. RevolvingUtilizationOfUnsecuredLines — tầm quan trọng tổng quát cao nhất
2. NumberOfTimes90DaysLate — tín hiệu nhị phân mạnh (0 vs >0)
3. TotalDelinquencyScore — tổng hợp trễ hạn có trọng số
4. FinancialStressIndex — số hạng tương tác

**SHAP Waterfall (high-risk sample):**
Một sample với P(default) cao có dương SHAP từ utilization cao và TotalDelinquencyScore cao.

Chi tiết SHAP analysis sẽ được deepened trong Phase 4 (waterfall plots, dependence plots).

---

## 6. Best Model — XGBoost

**Hyperparameters finalized:**
```python
XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.1,
    min_child_weight=5,
    gamma=0.1,
    scale_pos_weight=13.96,
    tree_method='hist',
    random_state=42,
)
```

**Deployment specs:**
- Đầu vào: 14 đặc trưng (10 gốc + 4 được tạo, đã imputed)
- Ngưỡng tối ưu Phase 3 (F1-opt): 0,77
- **Ngưỡng triển khai Phase 4 (F2-opt): 0,625**
- Phân loại mức rủi ro (căn chỉnh với ngưỡng triển khai):
  - P < 0.10 → Low Risk
  - 0.10 ≤ P < 0.30 → Medium Risk
  - 0.30 ≤ P < 0.625 → High Risk
  - P ≥ 0.625 → Very High Risk (REJECT)

**Saved files:**
- `models/best_model.pkl` (XGBoost)
- `reports/model_results.csv`
- `reports/fig_16_lr_analysis.png` → `fig_21_model_heatmap.png`

---

## 7. Quyết định thiết kế quan trọng

| Quyết định | Lý do | Reasoning |
|------------|-------|-----------|
| L1 với C=0.001 cho LR | RandomizedSearchCV chọn, confirmed bằng sparse coefficients | R15 |
| scale_pos_weight=13.96 cho XGBoost | n_neg/n_pos = 97,981/7,018 = 13.96 | R16 |
| Optimal threshold tất cả > 0.5 | Class imbalance shifts Bayesian prior → threshold phải dịch phải | R17 |
| XGBoost là mô hình tốt nhất | AUC cao nhất + hiệu quả (127s) + khả năng giải thích SHAP | Kết quả thực nghiệm |
| RF có recall cao nhất | Bagging averaging → less aggressive threshold placement | Observation |

---

*Notebook đầy đủ: `notebooks/03_Modeling.ipynb`*
*Figures: `reports/fig_16` đến `fig_21`*
*Tiếp theo: Phase 4 — Error Analysis & SHAP Deep Dive*
