# Error Analysis & Diagnostics Report
## Loan Default Prediction | Give Me Some Credit

**Model:** XGBoost (AUC=0.8714, optimal threshold=0.77)  
**Notebook:** `04_Analysis.ipynb`  
**Figures:** fig_22 – fig_30

---

## 1. Error Profiling — FN và FP là ai?

### 1.1 Định nghĩa 4 Error Categories

| Category | Ý nghĩa | Hậu quả |
|----------|---------|---------|
| **TP** (True Positive) | Vỡ nợ thực sự, model cảnh báo đúng | Ngân hàng từ chối → tránh được bad loan |
| **FN** (False Negative) | Vỡ nợ thực sự, model **KHÔNG** cảnh báo | Ngân hàng cho vay → mất principal + interest |
| **FP** (False Positive) | Không vỡ nợ, model cảnh báo nhầm | Ngân hàng từ chối → mất opportunity cost |
| **TN** (True Negative) | Không vỡ nợ, model đúng | Ngân hàng cho vay → normal profit |

FN là error type **nghiêm trọng nhất** trong bối cảnh tín dụng — đây là những khoản vay xấu mà ngân hàng không biết.

### 1.2 Distribution Analysis (fig_22, fig_23)

**FN vs TP comparison (cả hai đều là actual defaults, y=1):**

| Feature | FN median | TP median | Ratio FN/TP | Interpretation |
|---------|-----------|-----------|-------------|----------------|
| RevolvingUtilization | 0.518 | 1.000 | 0.52 | FN ít stressed về credit hơn |
| TotalDelinquencyScore | 0.000 | 5.000 | 0.00 | **FN không có delinquency history!** |
| FinancialStressIndex | 0.000 | 4.000 | 0.00 | FN có index = 0 (không stress) |
| NumberOfTimes90DaysLate | 0.000 | 1.000 | 0.00 | FN chưa từng trễ >90 ngày |
| MonthlyIncome | $4,200 | $3,400 | 1.24 | FN có thu nhập cao hơn TP! |

False Negatives là những người vỡ nợ trông lành mạnh — delinquency history nhẹ hơn và credit utilization thấp hơn TP. Model không cảnh báo được vì features của họ không đủ để vượt threshold 0.77.

**FP vs TN comparison (cả hai đều là actual non-defaults, y=0):**

| Feature | FP median | TN median | Ratio FP/TN | Interpretation |
|---------|-----------|-----------|-------------|----------------|
| RevolvingUtilization | 0.958 | 0.118 | 8.10 | **FP có utilization gần limit** |
| TotalDelinquencyScore | 4.000 | 0.000 | >>1 | FP có delinquency history |
| FinancialStressIndex | 2.856 | 0.000 | >>1 | FP trông rất giống defaulter |
| MonthlyIncome | $3,610 | $4,505 | 0.80 | FP có thu nhập thấp hơn TN |

False Positives là những người có utilization cao và vài lần trễ nhỏ — features trùng với profile người vỡ nợ — nhưng thực tế vẫn trả được nợ. Model nhầm vì không phân biệt được nhóm này.

### 1.3 Probability Score Analysis (fig_24)

Phân tích score distribution cho thấy:

- **FN scores:** median = **0.523** (threshold=0.77)
  → Cách xa threshold ~0.25 điểm — không phải cận ngưỡng, model đánh giá thấp họ
- **TP scores:** median = **0.912** → model tự tin ở defaults điển hình
- **FP scores:** median = **0.866** → model tự tin quá mức khi từ chối nhóm này

FN và FP có cấu trúc lỗi khác nhau. FN (median score=0.523) nằm **xa** threshold 0.77 — không phải cận ngưỡng mà là người vỡ nợ trông lành mạnh: model tự tin phân loại họ là an toàn vì features không có dấu hiệu cảnh báo rõ ràng. Ngược lại, FP (median score=0.866) nằm **trên** threshold — model tự tin quá mức khi từ chối, dù họ không thực sự vỡ nợ. Nhóm thực sự cận ngưỡng là tập nhỏ nằm trong khoảng score 0.7–0.8.

---

## 2. Business Cost Analysis

### 2.1 Cost Framework

Dựa trên model chi phí credit risk thực tế:

```
FN cost per case = Average_loan × (1 - Recovery_rate)
                 = $15,000 × (1 - 0.25) = $11,250 per missed default

FP cost per case = Opportunity_cost_per_rejected_good_customer = $500
```

**Tỉ lệ FN:FP = 22.5:1** — FN đắt gấp 22.5 lần FP. Điều này biện minh cho việc dùng threshold thấp hơn để tăng Recall.

### 2.2 So sánh Cost theo Threshold

| Threshold | FN count | FP count | Total Cost | So với t=0.77 |
|-----------|----------|----------|------------|---------------|
| 0.50 (naive) | 338 | 4,079 | **$5,842,000** | -33.5% (cheaper!) |
| 0.625 (F2 opt) | ~498 | ~2,354 | **~$6,779,500** | -22.9% |
| 0.77 (F1 opt) | 728 | 1,195 | **$8,787,500** | baseline |

t=0.5 có tổng chi phí thấp nhất vì FN đắt gấp 22 lần FP: giảm 390 FN tiết kiệm $4.4M, dù FP tăng thêm 2.884 chỉ tốn thêm $1.4M. Tối ưu ngưỡng phải dựa trên chi phí thực tế, không phải F1.

→ Ngưỡng F2 (0.625) không phải ngưỡng tối ưu chi phí — t=0.5 rẻ hơn. Đây là ngưỡng ưu tiên Recall trong khi giữ FP ở mức kiểm soát được so với t=0.5.

---

## 3. Threshold Optimization — F-beta Score (β=2)

### 3.1 Lý thuyết

$$F_\beta = (1+\beta^2) \cdot \frac{\text{Precision} \times \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

Với β=2: Recall được trọng số **4×** so với Precision:
$$F_2 = 5 \cdot \frac{P \times R}{4P + R}$$

Trong credit risk, 1 FN (bỏ sót vỡ nợ) tốn ~$11.250, còn 1 FP (từ chối nhầm) tốn ~$500. FN:FP ≈ 22.5:1, nên β=2 là lựa chọn thận trọng hợp lý: ưu tiên Recall hơn Precision, còn phần chốt chi phí thực tế được xử lý riêng ở bảng cost analysis.

### 3.2 Optimal Thresholds (validation set)

| Threshold | F1 | F2 | Precision | Recall |
|-----------|----|----|-----------|--------|
| 0.50 (naive) | 0.3455 | 0.5177 | 0.2223 | **0.7753** |
| 0.625 (F2 opt) | 0.4137 | **0.5365** | 0.2994 | 0.6689 |
| 0.775 (F1 opt) | **0.4443** | 0.4801 | 0.3951 | 0.5073 |
| 0.77 (F1 opt) | 0.4466 | 0.4858 | 0.3937 | 0.5160 |

F1-optimized (0.775) và ngưỡng Phase 3 (0.77) gần như giống nhau — nhất quán. F2-optimized (0.625) tăng Recall từ 51.6% → 66.9% (+15.3 pp), đánh đổi bằng Precision giảm từ 39.4% → 29.9%.

**Lựa chọn ngưỡng:** Sử dụng **t=0.625** (F2 opt) cho ngân hàng ưu tiên bắt nhiều defaults hơn. t=0.775 phù hợp hơn khi muốn FP thấp — ít từ chối nhầm hơn nhưng bỏ sót nhiều defaults hơn.

---

## 4. SHAP Deep Dive — Explainability

### 4.1 Global Feature Importance (fig_26, fig_26a)

Dựa trên SHAP TreeExplainer (3,000 test samples, exact Shapley values):

**Top features theo Mean |SHAP value|:**

| Rank | Feature | Type | Mean\|SHAP\| | Insight |
|------|---------|------|-------------|---------|
| 1 | **FinancialStressIndex** | Engineered | **0.577** | Interaction term captures synergy tốt nhất |
| 2 | RevolvingUtilizationOfUnsecuredLines | Original | 0.535 | Credit stress trực tiếp |
| 3 | **TotalDelinquencyScore** | Engineered | 0.410 | Weighted delinquency summary |
| 4 | age | Original | 0.244 | Proxy experience tài chính |
| 5 | NumberOfOpenCreditLinesAndLoans | Original | 0.168 | Diversification/overextension signal |
| 9 | **DebtToIncomeRatio** | Engineered | 0.065 | R14: KEEP (>5% của #2 feature) |
| 14 | **DelinquencyTrend** | Engineered | 0.002 | Thấp nhất — có thể drop sau |

**Observations quan trọng:**
- **2 trong top 3** features là engineered → Phase 2 feature engineering được validate mạnh
- FinancialStressIndex (#1, 0.577) > RevolvingUtilization (#2, 0.535) > TotalDelinquencyScore (#3, 0.410)
  → Interaction term (utilization × delinquency) quan trọng hơn từng thành phần riêng lẻ
- NumberOfTimes90DaysLate chỉ ở rank 12 (SHAP=0.013) mặc dù đây là signal mạnh nhất theo domain
  → XGBoost học được nó qua FinancialStressIndex và TotalDelinquencyScore thay thế
- DelinquencyTrend (SHAP=0.002) rất thấp → có thể drop trong Phase 5
- R14 confirmed: DTI SHAP=0.065 > threshold 0.027 → **KEEP DebtToIncomeRatio**

### 4.2 SHAP Dependence Analysis (fig_27)

**RevolvingUtilization:**
- SHAP gần 0 khi utilization < 0.3 → low risk zone
- SHAP tăng mạnh khi utilization > 0.7 → exponential risk zone
- Nonlinear relationship — cần tree-based model để capture

**TotalDelinquencyScore:**
- SHAP = 0 khi score = 0 (không có lần trễ nào)
- SHAP tăng sharply ngay khi score > 0
- Cực kỳ predictive: bất kỳ delinquency history nào đều tăng risk đáng kể

### 4.3 Individual SHAP Waterfall (fig_28)

**True Positive (TP):** Người vỡ nợ model đúng
- High RevolvingUtil + High TotalDelinquencyScore push prediction về phía default
- SHAP waterfall cho thấy rõ ràng các features nào drive prediction

**False Positive (FP):** Người không vỡ nợ bị nhầm
- Có vài lần delinquency và utilization cao → model overreacts
- Nhưng features khác (income, age) counterbalance không đủ

**False Negative (FN):** Người vỡ nợ model bỏ sót
- RevolvingUtil và TotalDelinquencyScore moderate, không đủ để vượt threshold
- Base value thấp → cần nhiều "positive evidence" hơn để đẩy prediction đủ cao

### 4.4 R14 Verification: DebtToIncomeRatio

Từ Phase 2 (R14): cần verify nếu DTI SHAP < 5% của top-2 feature thì đề xuất remove.

Kết quả từ SHAP analysis:
- Nếu DTI mean|SHAP| < threshold_r14 → **Remove DTI** trong Phase 5 (Streamlit form)
- Nếu DTI mean|SHAP| ≥ threshold_r14 → **Keep DTI** trong feature set

**Kết quả (R14 verification):**
- DTI mean|SHAP| = 0.065, threshold (5% of #2) = 0.027
- **VERDICT: KEEP DebtToIncomeRatio** → DTI SHAP vượt threshold gần 2.5× → không thể drop
- R14 hypothesis bị bác bỏ: DTI có contribution đáng kể mặc dù L1 zero-out ở Logistic Regression

---

## 5. Learning Curve Diagnostics (fig_29)

### 5.1 Bias-Variance Framework

$$\text{Expected Error} = \underbrace{\text{Noise}}_{\text{irreducible}} + \underbrace{\text{Bias}^2}_{\text{underfitting}} + \underbrace{\text{Variance}}_{\text{overfitting}}$$

**Đọc learning curve (XGBoost):**
- **Train AUC:** nên cao và stable
- **Val AUC:** nên tiếp cận Train AUC khi N tăng
- **Gap = Train - Val:** indicator của variance (overfitting)

### 5.2 Kết quả và Chẩn đoán

Từ `fig_29_learning_curve.png`:

| N (training size) | Train AUC | Val AUC | Gap |
|------------------|-----------|---------|-----|
| 6,999 (10%) | 0.9596 | 0.8427 | 0.1169 |
| 13,999 (20%) | 0.9261 | 0.8535 | 0.0726 |
| 27,999 (40%) | 0.9059 | 0.8602 | 0.0457 |
| 45,499 (65%) | 0.8926 | 0.8624 | 0.0302 |
| 69,999 (100%) | 0.8847 | 0.8642 | **0.0205** |

**Pattern diagnosis:**
- Gap giảm đơn điệu từ 0.117 → 0.021 khi N tăng → variance được kiểm soát tốt
- Val AUC không còn tăng đáng kể ở N>45.000 (0.8624 → 0.8642, chỉ +0.0018) → thêm dữ liệu ít cải thiện
- Final gap = 0.021 < 0.03 → **"Good fit — model generalizes well"**
- Train AUC giảm từ 0.96 → 0.88 khi N tăng: training set đa dạng hơn → harder to overfit

### 5.3 Recommendations

| Diagnosis | Action |
|-----------|--------|
| High variance (gap > 0.06) | Increase reg_lambda, decrease max_depth, increase subsample |
| High bias (both AUC < 0.82) | More features, try higher max_depth, reduce regularization |
| Good fit (gap < 0.03) | OK — consider feature engineering for next increment |
| Plateau (val AUC flat) | More data won't help — focus on feature engineering |

**Kết luận cho dataset này:** XGBoost ở sweet spot — regularization effective, không over/underfit. Feature engineering (TotalDelinquencyScore, FinancialStressIndex) đã maximize information extraction từ available features.

---

## 6. Summary & Recommendations

### 6.1 Error Analysis Conclusions

1. **FN — người vỡ nợ bị bỏ sót:** Score median=0.523, cách xa threshold 0.247. Đây không phải cận ngưỡng mà là những người không có delinquency history (TotalDelinquencyScore=0, FinancialStressIndex=0) — model không có bằng chứng để cảnh báo. Thu nhập ($4.200) cao hơn TP ($3.400), trông như hồ sơ tốt nhưng vẫn vỡ nợ, có thể do sự kiện bất ngờ không để lại dấu vết trong credit history.

2. **FP — khách hàng tốt bị từ chối nhầm:** Có một vài lần trễ hạn và utilization cao, trông giống profile người vỡ nợ — nhưng thực tế vẫn trả được. Model chưa nắm được xu hướng cải thiện của nhóm này.

3. **Threshold là lever chính:** Từ threshold=0.77 (F1 opt) sang F2-optimized threshold → Recall tăng đáng kể, business cost giảm, chấp nhận thêm FP.

### 6.2 Feature Engineering Validated

SHAP analysis xác nhận:
- **TotalDelinquencyScore** (engineered) > raw delinquency features → weighting mechanism hoạt động
- **FinancialStressIndex** (engineered) capture nonlinear interaction → không có trong raw features
- Feature engineering cung cấp ~2 trong top-5 SHAP features → **justified**

### 6.3 Model Deployment Recommendations

| Deployment context | Recommended threshold | Rationale |
|-------------------|----------------------|-----------|
| Conservative bank | F2-optimized (~0.30–0.45) | Maximize recall, FN cost >> FP cost |
| FinTech lender | F1-optimized (~0.45–0.55) | Balance growth and risk |
| High-volume screening | Tiered: Low/Medium/High/Very High risk | Multi-threshold categorization |

### 6.4 Technical Debt & Phase 5 Actions

- **DebtToIncomeRatio:** Verify R14 — nếu SHAP thấp, remove khỏi Streamlit form
- **SHAP integration:** Dùng `shap.TreeExplainer` trong Streamlit cho real-time waterfall
- **Threshold parameter:** Expose threshold slider trong dashboard cho user experimentation
- **Model persistence:** `best_model.pkl` (XGBoost) đã sẵn sàng cho Streamlit app

---

*Phase 4 hoàn thành. Figures: fig_22–fig_30. Next: Phase 5 — Streamlit Dashboard.*
