# Báo cáo Preprocessing & Feature Engineering
## Loan Default Prediction | Give Me Some Credit

**Notebook:** `notebooks/02_Preprocessing.ipynb`
**Input:** `data/raw/cs-training.csv` — 150,000 rows × 11 columns
**Output:** `data/splits/{train,val,test}.csv`

---

## 1. Tổng quan Pipeline

```
Raw data (150,000 rows)
    ↓ [Cleaning] Xóa age==0, capping p99
    ↓ [Imputation] KNN(k=5) cho MonthlyIncome, Median cho Dependents
    ↓ [Feature Engineering] +4 features tài chính
    ↓ [Split] Stratified 70/15/15
    ↓ [Model training] final app uses best_model.pkl
    ↓ SMOTE (nếu thử nghiệm so sánh trên training only)
```

**Kết quả cuối:**
- Training set: 104,999 rows × 14 features (6.68% positive)
- Val set:       22,500 rows × 14 features (6.68% positive)
- Test set:      22,500 rows × 14 features (6.68% positive)

---

## 2. Data Cleaning

### 2.1 Xóa age == 0

1 row có `age = 0` bị xóa hoàn toàn. Giá trị này **không thể impute** vì không có age hợp lý nào có thể thay thế — cần phải biết tuổi thực để đánh giá credit risk. Đây là data entry error rõ ràng.

### 2.2 Capping tại 99th Percentile

| Feature | p99 (cap value) | Rows bị cap | Trước max | Sau max |
|---------|----------------|-------------|----------|---------|
| `RevolvingUtilizationOfUnsecuredLines` | ~1.0 | 1,500 | 50,708 | ~1.0 |
| `MonthlyIncome` | 25,000 | 1,168 | 3,008,750 | 25,000 |
| `DebtRatio` | ~3,330 | 1,168 | 329,664 | ~3,330 |
| `NumberOfTimes90DaysLate` | 3 | 873 | 98 | 3 |
| `NumberOfTime30-59DaysPastDueNotWorse` | 4 | 850 | 98 | 4 |
| `NumberOfTime60-89DaysPastDueNotWorse` | 2 | 755 | 98 | 2 |
| `NumberRealEstateLoansOrLines` | 4 | ~600 | 54 | 4 |
| `NumberOfOpenCreditLinesAndLoans` | 27 | ~1,200 | 58 | 27 |

**Lý do capping (không xóa):** Giữ rank information — người có income $3M vẫn là "rất cao thu nhập", chỉ không phải $3M cụ thể. Xóa rows gây sample bias (người giàu ít default hơn → xóa họ làm training set biased).

**Delinquency max = 98:** Đây là sentinel value (giá trị đặc biệt đánh dấu unknown/error), không phải số lần trễ thực tế. Một người không thể trễ hạn 98 lần trong vài năm với tần suất thanh toán hàng tháng.

---

## 3. Missing Value Imputation

### 3.1 MonthlyIncome (19.82% missing)

**Xác định cơ chế missing:** Chi-squared test giữa missing indicator và target:
- χ² = 67.89, p-value ≈ 0 (<<0.001)
- **Kết luận: MAR/MNAR** — missing KHÔNG independent với target
- Domain interpretation: người không khai báo income thường là freelancer/thất nghiệp → higher risk

**So sánh hai phương pháp:**

| Phương pháp | Tất cả imputed = | Std imputed values | Spearman ρ với target |
|-------------|-----------------|-------------------|----------------------|
| Median | 5,400 (identical) | 0 | -0.0617 |
| KNN (k=5) | trung bình ~336–5,400 (diverse) | ~1,157 | -0.0308 |

**Giải thích KNN cho mean thấp hơn:** KNN tìm neighbors dựa trên partial Euclidean distance trong feature space. Người missing income có features tương tự người thu nhập thấp (age trẻ, nhiều delinquency, ít real estate loans) → KNN assign income thấp. Điều này **có thể đúng về domain**: người thiếu income ổn định thực sự có thu nhập thấp hơn median.

**Lý do chọn KNN:**
1. Tạo **diverse imputed values** (không spike tại một điểm)
2. Khai thác correlation với features khác (age, DebtRatio)
3. Consistent với MAR evidence từ chi-squared test

### 3.2 NumberOfDependents (2.62% missing)

Median = 0 (phân phối right-skewed, đa số = 0 hoặc 1). Với chỉ 2.62% missing và pattern gần MCAR, Median Imputation đủ tốt và ít overhead hơn KNN.

**Verify:** 0 missing values sau imputation ✓

---

## 4. Feature Engineering

### 4.1 TotalDelinquencyScore

$$\text{TotalDelinquencyScore} = 3 \times \text{N}_{90+} + 2 \times \text{N}_{60-89} + 1 \times \text{N}_{30-59}$$

**Spearman ρ với target: 0.345** (so với max của 3 features riêng lẻ: 0.342)
- Composite score không tăng correlation mạnh vì 3 features gốc đã rất correlated với nhau (ρ ≈ 0.45–0.49)
- Tuy nhiên, nó **giải quyết multicollinearity** cho Logistic Regression
- Thể hiện domain expertise: weighted by severity, không just count

### 4.2 FinancialStressIndex

$$\text{FSI} = \text{RevolvingUtilization} \times \text{TotalDelinquencyScore}$$

**Spearman ρ với target: 0.346** — cao nhất trong tất cả features!

Interaction term `Util × Delinq` capture profile "double stressed": maxed out credit AND có delinquency history. Đây là highest-risk pattern trong credit scoring.

### 4.3 DebtToIncomeRatio

$$\text{DTI\_abs} = \text{DebtRatio} \times \text{MonthlyIncome}$$

**Spearman ρ với target: 0.016** — yếu, nhưng giữ vì:
- Meaningful về mặt domain (số USD nợ thực tế)
- DebtRatio raw có vấn đề khi income ≈ 0 (ratio explodes)
- Có thể có non-linear effects mà Spearman không capture

### 4.4 DelinquencyTrend

$$\text{DelinquencyTrend} = \text{N}_{30-59} - \text{N}_{90+}$$

**Spearman ρ với target: 0.070** — yếu nhưng capture trajectory information.
- Âm (N90+ > N30-59): pattern xấu dần
- Dương: pattern cải thiện hoặc chỉ miss nhẹ

---

## 5. Multicollinearity — VIF Analysis

| Feature | VIF | Diễn giải |
|---------|-----|----------|
| `NumberOfTimes90DaysLate` | ∞ | Là component của TotalDelinquencyScore → perfect linear dependency |
| `NumberOfTime30-59DaysPastDueNotWorse` | ∞ | Idem |
| `NumberOfTime60-89DaysPastDueNotWorse` | ∞ | Idem |
| `TotalDelinquencyScore` | ∞ | Linear combination của 3 features trên |
| `DelinquencyTrend` | ∞ | Linear function của N30-59 và N90+ |
| `FinancialStressIndex` | 7.43 | Moderate — product của Utilization × TotalDelinquency |
| Các features khác | 1.1–1.6 | OK — không có multicollinearity |

**Xử lý VIF∞ cho Logistic Regression:**
- 3 delinquency features gốc + TotalDelinquencyScore + DelinquencyTrend = 5 features với VIF∞
- Trong LR, sẽ dùng **L2 regularization** để ổn định coefficients
- Hoặc chỉ giữ TotalDelinquencyScore (drop 3 originals) trong LR-only pipeline
- Tree-based models: VIF không ảnh hưởng, giữ tất cả

---

## 6. Stratified Train/Val/Test Split

```python
train_test_split(stratify=y, test_size=0.15, random_state=42)  # test split
train_test_split(stratify=y_trainval, test_size=0.1765, random_state=42)  # val split
```

| Split | Rows | Positive | Positive Rate |
|-------|------|----------|--------------|
| Train | 104,999 | 7,018 | **6.68%** |
| Val | 22,500 | 1,504 | **6.68%** |
| Test | 22,500 | 1,504 | **6.68%** |

Stratification đảm bảo ~6.68% positive trong mọi split ✓ — consistent với tỷ lệ thực tế.

---

## 7. SMOTE — Chỉ là thí nghiệm trên training set

```
Before SMOTE: 7,018 positive (6.68%), 97,981 negative (93.32%)
After  SMOTE: 97,981 positive (50%), 97,981 negative (50%)
Added:        90,963 synthetic minority samples
```

**Quan trọng:** SMOTE chỉ apply trên training set khi so sánh thí nghiệm. Val/test giữ nguyên phân phối thực (6.68%) để metrics phản ánh production performance.

Trong bản final, các model chính dùng `class_weight='balanced'` hoặc `scale_pos_weight` thay vì SMOTE để tránh synthetic noise và giữ pipeline đơn giản.

---

## 8. Artifact cuối

Final deployment chỉ giữ **`models/best_model.pkl`** cho Streamlit app. Các pipeline/model artifacts khác là file lịch sử trong notebook thí nghiệm và không còn là một phần của bundle nộp cuối.

---

## 9. Quyết định thiết kế quan trọng

| Quyết định | Lý do | REASONING_LOG |
|------------|-------|--------------|
| KNN cho MonthlyIncome | MAR evidence: χ²=67.89, p≈0 | R03 |
| Median cho NumberOfDependents | 2.62%, gần MCAR, overhead thấp | R04 |
| Capping không xóa | Giữ rank info, tránh sample bias | R05 |
| Trọng số 3:2:1 | FICO severity tiers, EDA confirms | R06 |
| Phép nhân trong FSI | Interaction term, joint stress | R11 |
| RobustScaler cho LR | Data còn right-skewed sau capping | R10 |
| SMOTE sau split | Tránh data leakage | R09 |
| Stratified split | Imbalanced data 6.68% | R08 |

---

*Notebook đầy đủ: `notebooks/02_Preprocessing.ipynb`*
*Figures: `reports/fig_11` đến `fig_15`*
