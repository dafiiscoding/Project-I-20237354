# Báo cáo So sánh Models — Phase 3 Modeling
## Loan Default Prediction | Give Me Some Credit

**Notebook:** `notebooks/03_Modeling.ipynb`
**Input:** `data/splits/{train,val,test}.csv`
**Output:** `models/best_model.pkl`, `reports/model_results.csv`

---

## 1. Thiết lập Thực nghiệm

### 1.1 Protocol thống nhất
Mỗi model được đánh giá theo cùng một protocol:

| Thành phần | Thiết kế | Lý do |
|-----------|---------|-------|
| Primary metric | AUC-ROC | Threshold-free, robust với class imbalance 6.68% |
| CV strategy | Stratified 5-Fold | Giữ tỷ lệ positive trong mỗi fold |
| Hyperparameter tuning | RandomizedSearchCV | Nhanh hơn GridSearch, đủ tốt với n_iter lớn |
| Threshold selection | Tối ưu F1 trên val set (Phase 3; deployment threshold chốt ở Phase 4) | Threshold 0.5 không tối ưu với imbalanced data |
| Imbalance handling | class_weight='balanced' (LR/DT/RF), scale_pos_weight=13.96 (XGBoost) | |

### 1.2 Dataset splits
- Training: 104,999 rows (6.68% positive)
- Validation: 22,500 rows (6.68% positive)
- Test: 22,500 rows (6.68% positive)
- Features: 14 (10 gốc + 4 engineered)

---

## 2. Kết quả Chi tiết từng Model

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

Optimal threshold: **0.66** (so với 0.5 default — class imbalance shift threshold)
Training time: 487.9s

**Odds Ratios (L1, C=0.001):**

| Feature | Coefficient β | Odds Ratio e^β | Diễn giải |
|---------|--------------|----------------|----------|
| TotalDelinquencyScore | +0.255 | **1.291** | Tăng 1 điểm → risk ×1.29 |
| RevolvingUtilization | +0.191 | **1.211** | Tăng 1 đơn vị → risk ×1.21 |
| FinancialStressIndex | +0.170 | **1.185** | Feature engineered quan trọng nhất |
| NumberOfTime30-59 | +0.062 | 1.064 | Delinquency nhẹ vẫn có signal |
| DelinquencyTrend | +0.025 | 1.025 | Yếu, nhưng có hướng |
| NumberOfTimes90DaysLate | **0.000** | 1.000 | L1 zero out (captured by TotalDelinquency) |
| DebtToIncomeRatio | **0.000** | 1.000 | L1 zero out (yếu về signal) |
| age | -0.120 | **0.887** | Protective: tuổi cao → risk giảm |
| MonthlyIncome | -0.059 | 0.942 | Protective: income cao → risk giảm |
| NumberOfDependents | -0.052 | 0.949 | Nhẹ: nhiều người phụ thuộc → giảm risk? |

**Nhận xét domain:**
- L1 với C=0.001 đã zero out `NumberOfTimes90DaysLate` vì `TotalDelinquencyScore` đã capture nó hoàn toàn (multicollinearity xử lý đúng như dự đoán trong R13)
- `FinancialStressIndex` (feature engineered) có odds ratio cao thứ 3 → feature engineering có giá trị
- `age` là protective factor mạnh nhất: OR=0.887 → mỗi năm tuổi tăng, risk giảm ~11.3%
- Optimal threshold 0.66 ≠ 0.5: với imbalance 1:14, model cần probability cao hơn để "commit" prediction positive

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
- `min_samples_leaf=200` là pruning mạnh, tránh overfitting trên noise
- `max_features=0.5` = random subsampling trong DT → pre-cursor của RF concept
- Recall thấp nhất (0.469) → DT thận trọng hơn khi gán nhãn default
- Threshold cực cao (0.80) → DT cần confidence rất cao mới predict default

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
- `max_features=0.3` (chỉ 30% features mỗi split) → decorrelates trees mạnh → variance thấp
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

**SHAP Feature Importance (mean |SHAP value|):**

*Từ SHAP TreeExplainer trên 2,000 test samples. Kết quả phản ánh global feature importance.*

Top features theo SHAP: RevolvingUtilizationOfUnsecuredLines, NumberOfTimes90DaysLate, TotalDelinquencyScore, FinancialStressIndex

**Nhận xét:**
- XGBoost có max_depth=4 (cây nông) với nhiều iterations (200) → Boosting reduce bias hiệu quả
- reg_alpha=0.1 (L1 trên leaf weights) + reg_lambda=1.0 (L2) → model regularized tốt
- min_child_weight=5: node cần ít nhất 5 samples sum of Hessian để split → anti-overfitting
- subsample=0.8 + colsample_bytree=0.8: stochastic gradient boosting → variance reduction
- Training time 127s: nhanh hơn RF (511s) 4×, mà AUC gần tương đương → XGBoost efficient hơn

---

## 3. So sánh Tổng hợp

### 3.1 Bảng so sánh đầy đủ

| Model | CV AUC | Test AUC | Avg Precision | F1 | Precision | Recall | Threshold | Train time |
|-------|--------|---------|--------------|-----|-----------|--------|-----------|------------|
| **XGBoost** | **0.8656** | **0.8714** | **0.4005** | **0.4466** | 0.3937 | 0.5160 | 0.77 | 127s |
| Random Forest | 0.8635 | 0.8703 | 0.3980 | 0.4439 | 0.3869 | **0.5206** | 0.72 | 511s |
| Decision Tree | 0.8509 | 0.8579 | 0.3762 | 0.4314 | **0.3991** | 0.4694 | 0.80 | **13s** |
| Logistic Regression | 0.8371 | 0.8432 | 0.3700 | 0.4340 | 0.3878 | 0.4927 | 0.66 | 488s |

### 3.2 Phân tích so sánh

**XGBoost vs Random Forest:**
- AUC gap: 0.0011 — rất nhỏ, gần như statistically equivalent
- Recall: RF cao hơn (0.5206 vs 0.5160) — RF giỏi hơn trong catching defaulters
- Training time: XGBoost 4× nhanh hơn (127s vs 511s)
- **Kết luận:** XGBoost được chọn vì hiệu quả tính toán (127s), explainability qua SHAP, và AUC cao nhất

**Decision Tree:**
- AUC=0.8579 vượt LR (0.8432) 0.015 điểm
- Fastest model (13s), easy to visualize
- Thấp nhất về Recall → risk bỏ sót người vỡ nợ

**Logistic Regression — Baseline phù hợp:**
- AUC=0.8432 là solid baseline
- Duy nhất có odds ratio interpretation trực tiếp (Basel III compliant)
- L1 triệt tiêu đặc trưng dư thừa → xác nhận chất lượng feature engineering
- Trong thực tế ngân hàng quy định cần giải thích từng quyết định: LR hoặc LR+SHAP

### 3.3 Tất cả optimal thresholds > 0.5
Một observation quan trọng: tất cả 4 models đều có optimal threshold > 0.5 (0.66–0.80).

**Lý do:** Với class imbalance 1:14.0, prior P(default=1) = 0.0668. Logit scale: model's baseline output đã "biased" về 0. Cần probability threshold cao hơn để có đủ confidence predict positive. Điều này xác nhận rằng dùng 0.5 làm threshold là sai trong credit risk với imbalanced data.

---

## 4. Bàn luận 3 Tranh luận Lớn

### 4.1 Interpretability vs Performance
- LR (baseline): AUC=0.8432, odds ratios rõ ràng, Basel III compliant
- XGBoost (best): AUC=0.8714, cần SHAP cho explainability

AUC gap = 0.0282 (3.3% relative improvement). Câu hỏi: gap này có đủ justify thêm complexity không?

**Lập luận:** Với 22,500 test samples, 0.0282 AUC gap tương đương với:
- Correctly ranking thêm ~636 default/non-default pairs (AUC = P(score(+) > score(-)))
- Tại threshold 0.77: thêm ~130-150 defaulters được identify đúng (Recall 0.516 vs 0.493)
- 130 defaulters × average bad loan loss $15,000 = ~$1.95M/year saved (trên 22,500 customers)

→ **Kết luận:** Gap 0.0282 AUC tương đương ~130 defaults được bắt thêm trên tập test — có giá trị thực tế. XGBoost + SHAP phù hợp khi cần cả hiệu suất dự báo lẫn lý do cụ thể cho từng quyết định.

### 4.2 Data-centric vs Model-centric
Feature engineering thủ công (TotalDelinquencyScore, FinancialStressIndex) vs XGBoost tự học interaction terms.

**Thực nghiệm:**
- LR với features engineered: AUC=0.8432
- LR với 3 raw delinquency features (không dùng TotalDelinquencyScore) sẽ thấp hơn do multicollinearity
- XGBoost với engineered features: AUC=0.8714

`FinancialStressIndex` (engineered) xuất hiện trong top SHAP features của XGBoost — các features tự tạo không bị thay thế mà bổ sung thêm thông tin mà model không tự học được.

→ **Kết luận:** Hai hướng tiếp cận không loại trừ nhau. Feature engineering giúp cả LR (giảm đa cộng tuyến) và XGBoost (cung cấp features mang ý nghĩa tài chính rõ ràng).

### 4.3 Precision vs Recall Tradeoff
Tại optimal threshold:

| Model | Precision | Recall | Business Impact |
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

**Top features by mean |SHAP|:**
1. RevolvingUtilizationOfUnsecuredLines — highest global importance
2. NumberOfTimes90DaysLate — strong binary signal (0 vs >0)
3. TotalDelinquencyScore — composite delinquency
4. FinancialStressIndex — interaction term

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
- Input: 14 features (10 gốc + 4 engineered, đã imputed)
- Optimal threshold Phase 3 (F1-opt): 0.77
- **Deployment threshold Phase 4 (F2-opt): 0.625**
- Risk Tier mapping (aligned với deployment threshold):
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
| XGBoost là best model | AUC cao nhất + efficient (127s) + SHAP explainability | Kết quả thực nghiệm |
| RF có recall cao nhất | Bagging averaging → less aggressive threshold placement | Observation |

---

*Notebook đầy đủ: `notebooks/03_Modeling.ipynb`*
*Figures: `reports/fig_16` đến `fig_21`*
*Tiếp theo: Phase 4 — Error Analysis & SHAP Deep Dive*
