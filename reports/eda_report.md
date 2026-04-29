# Báo cáo EDA — Exploratory Data Analysis
## Loan Default Prediction | Give Me Some Credit

**Dataset:** `data/raw/cs-training.csv`
**Notebook:** `notebooks/01_EDA.ipynb`

---

## 1. Tổng quan Dataset

Dataset **Give Me Some Credit** được Kaggle cung cấp bởi một tổ chức tài chính Mỹ, bao gồm hồ sơ vay vốn của 150,000 khách hàng cùng lịch sử tín dụng tương ứng. Mục tiêu là dự báo xác suất một khách hàng sẽ **trễ thanh toán nghiêm trọng (≥ 90 ngày)** trong vòng 2 năm tới — đây là định nghĩa "default" được sử dụng trong thực tiễn báo cáo tín dụng tại Mỹ.

| Thuộc tính | Giá trị |
|-----------|---------|
| Số hồ sơ | 150,000 |
| Số features | 10 (tất cả numerical) |
| Target | `SeriousDlqin2yrs` (binary: 0/1) |
| Bộ nhớ | ~12.6 MB |

Điểm đặc biệt quan trọng: **không có đặc trưng phân loại**. Toàn bộ đầu vào đều là số thực — đơn giản hóa tiền xử lý, nhưng cũng có nghĩa là feature engineering thủ công (tạo interaction terms có ý nghĩa tài chính) sẽ tạo ra giá trị lớn.

---

## 2. Phân tích Class Imbalance

Đây là vấn đề trung tâm của toàn bộ dự án.

```
SeriousDlqin2yrs = 0 (Non-Default): 139,974 hồ sơ (93.32%)
SeriousDlqin2yrs = 1 (Default):      10,026 hồ sơ  ( 6.68%)
Imbalance ratio: 1:14.0
```

**Tại sao Accuracy hoàn toàn vô nghĩa trong bài toán này:**

Một model "naive" chỉ predict `0` cho mọi input sẽ đạt Accuracy = 93.32% — nghe có vẻ ấn tượng, nhưng Recall = 0%, nghĩa là model đó **không có bất kỳ giá trị kinh doanh nào**. Mọi người vỡ nợ đều bị bỏ sót.

Trong bối cảnh tín dụng, hai loại lỗi có chi phí hoàn toàn bất cân xứng:

- **False Negative** (model predict 0, thực tế là 1): Ngân hàng cho vay người sẽ vỡ nợ → mất toàn bộ khoản vay (principal loss). Đây là chi phí tài chính trực tiếp, thường lên đến hàng chục nghìn USD mỗi hồ sơ.
- **False Positive** (model predict 1, thực tế là 0): Ngân hàng từ chối người đủ điều kiện → mất cơ hội kinh doanh (opportunity cost). Chi phí này nhỏ hơn đáng kể.

Do đó: **Primary metric = AUC-ROC** (đo khả năng ranking, độc lập với threshold và imbalance ratio). Secondary: F1-Score, Precision-Recall curve. Threshold tối ưu sẽ được xác định ở Phase 4 dựa trên business cost function, không cố định ở 0.5.

**Implication cho modeling:** `scale_pos_weight = 13.96` cho XGBoost; `class_weight='balanced'` cho sklearn models; SMOTE chỉ áp dụng trên training set.

---

## 3. Missing Values — Phân tích Cơ chế

Chỉ có 2 features có missing values:

| Feature | Missing Count | Missing % | Cơ chế |
|---------|--------------|-----------|--------|
| `MonthlyIncome` | 29,731 | **19.82%** | MAR/MNAR |
| `NumberOfDependents` | 3,924 | **2.62%** | Gần MCAR |

### 3.1 MonthlyIncome — Missing Không Phải Ngẫu Nhiên

Kiểm định Chi-squared giữa missing indicator của `MonthlyIncome` và target `SeriousDlqin2yrs`:

```
χ² = 67.89, p-value ≈ 0 (< 0.0001)
Kết luận: Missing KHÔNG independent với target → MAR hoặc MNAR
```

Giải thích theo domain: Người không có thu nhập ổn định (thất nghiệp, freelancer, thu nhập tiền mặt) có xu hướng không khai báo income. Đây chính xác là nhóm có rủi ro tín dụng cao hơn. Do đó, việc thiếu giá trị MonthlyIncome **chính nó là dấu hiệu** của rủi ro.

**Hệ quả cho imputation:** Không thể dùng mean/median imputation đơn giản (MCAR assumption). Cần KNN Imputer để khai thác correlation với các features khác (age, DebtRatio) trong ước tính income. Sẽ so sánh thực nghiệm hai phương pháp trong Phase 2.

### 3.2 NumberOfDependents — Missing Gần Ngẫu Nhiên

Với chỉ 2.62% missing và không có pattern rõ ràng, Median Imputation (= 0, do right-skewed distribution) là đủ và ít rủi ro data leakage hơn KNN.

---

## 4. Outliers — Phân tích và Kế hoạch Xử lý

EDA phát hiện nhiều outlier cần xử lý trước khi modeling:

### 4.1 age == 0 (1 row)
Giá trị 0 tuổi không có nghĩa tài chính — không thể ký hợp đồng vay vốn ở tuổi 0. Đây là **data entry error** rõ ràng. **Hành động: XÓA** toàn bộ row này.

### 4.2 RevolvingUtilizationOfUnsecuredLines > 1 (3,321 rows, 2.2%)
Credit utilization > 100% (tức là đang nợ nhiều hơn hạn mức được cấp) **có thể xảy ra trong thực tế** khi lãi suất và phí phạt được tính thêm. Tuy nhiên, 241 rows có utilization > 10 (1000%) — đây là **extreme outlier** không có ý nghĩa tài chính hợp lý. **Hành động: Capping tại 99th percentile** (không xóa để giữ thông tin rank tương đối).

### 4.3 MonthlyIncome max = $3,008,750/tháng
Giá trị cực đại $3M/tháng vượt xa 99th percentile ($25,000). Trong bối cảnh đây là dataset cho vay tiêu dùng (consumer lending), không phải corporate banking, income $3M/tháng là bất thường. **Hành động: Capping tại 99th percentile = $25,000**.

### 4.4 Delinquency Counts max = 98 (cho cả 3 features)
Một khách hàng trễ hạn 98 lần trong vòng vài năm là **không thể xảy ra về mặt thực tế** — mỗi tháng chỉ có 1 kỳ thanh toán. Giá trị 96–98 xuất hiện ở cả 3 features delinquency, gợi ý đây là **sentinel value** (giá trị đặc biệt để đánh dấu "unknown" hoặc "error") thay vì dữ liệu thực.

```
NumberOfTimes90DaysLate:              99th pct = 3, max = 98
NumberOfTime30-59DaysPastDueNotWorse: 99th pct = 4, max = 98  
NumberOfTime60-89DaysPastDueNotWorse: 99th pct = 2, max = 98
```

**Hành động: Capping tại 99th percentile** cho mỗi feature.

---

## 5. Univariate Analysis — Phân phối Features

### 5.1 Phân phối Delinquency Features (Right-Skewed Extreme)

Ba features delinquency (`30-59`, `60-89`, `90+` days) đều có phân phối **cực kỳ right-skewed**:
- Đại đa số (>80%) có giá trị = 0: không trễ hạn lần nào
- Đuôi dài (heavy right tail) với giá trị tới 98
- Skewness > 15 cho cả ba features

Từ góc độ tài chính, điều này phản ánh thực tế: hầu hết người đi vay đều trả đúng hạn. Người trễ hạn nhiều lần là thiểu số nhưng quan trọng (positive class).

**Implication cho scaling:** StandardScaler sẽ kéo dãn phân phối lệch này không hiệu quả. Với Logistic Regression, cần cân nhắc **RobustScaler** (dùng median và IQR thay vì mean và std) sau khi capping.

### 5.2 RevolvingUtilizationOfUnsecuredLines — Bimodal

Phân phối có xu hướng bimodal:
- Đỉnh đầu (utilization ≈ 0): người dùng ít tín dụng hoặc có nhiều hạn mức chưa dùng
- Đỉnh thứ hai (utilization ≈ 1): người gần đầy hạn mức

Điều này phù hợp với lý thuyết FICO score: nhóm có utilization cực thấp (có hạn mức nhưng không cần dùng) và nhóm utilization cao (cần tín dụng thường xuyên) là 2 segments rất khác nhau về hành vi tài chính.

### 5.3 Age — Phân phối Chuẩn Lệch Phải

Age có phân phối gần normal với slight right skew, range hợp lý từ 18 đến 109 tuổi (ngoại trừ 1 giá trị = 0 cần xóa). Không cần transform đặc biệt.

---

## 6. Bivariate Analysis — Discriminative Power

Mann-Whitney U test (non-parametric, robust với right-skewed distributions) cho thấy **tất cả 10 features đều có phân phối statistically significantly khác nhau** giữa defaulters và non-defaulters (p < 0.001 sau Bonferroni correction).

**Top discriminators (Spearman correlation với target):**

| Rank | Feature | Spearman ρ | Domain Interpretation |
|------|---------|-----------|----------------------|
| 1 | `NumberOfTimes90DaysLate` | **+0.3423** | Lịch sử trễ hạn nặng — tương quan mạnh nhất; ghi trong credit report 7 năm |
| 2 | `NumberOfTime60-89DaysPastDueNotWorse` | **+0.2771** | Trễ hạn mức trung — dấu hiệu bắt đầu kiệt sức tài chính |
| 3 | `NumberOfTime30-59DaysPastDueNotWorse` | **+0.2574** | Trễ hạn nhẹ — cảnh báo sớm |
| 4 | `RevolvingUtilizationOfUnsecuredLines` | **+0.2404** | Financial stress — đang "maxed out" thẻ tín dụng |
| 5 | `age` | **-0.1170** | Người trẻ có risk cao hơn (ít kinh nghiệm tài chính, thu nhập chưa ổn định) |

**Tại sao Pearson và Spearman cho kết quả khác nhau:**

Pearson correlation của `RevolvingUtilization` với target chỉ là **-0.002** (gần zero!), nhưng Spearman là **+0.2404**. Sự khác biệt lớn này chỉ ra rằng mối quan hệ là **đơn điệu nhưng không tuyến tính** — phù hợp với đặc điểm lệch phải của feature. Dùng Pearson sẽ **underestimate** đáng kể importance của RevolvingUtilization.

**Lesson:** Với dataset tài chính có heavy-tailed distributions, **Spearman correlation là lựa chọn đúng đắn hơn Pearson** cho feature selection.

---

## 7. Multicollinearity Analysis

Correlation matrix Spearman cho thấy 3 features delinquency có tương quan đáng kể với nhau:

```
30-59 ↔ 60-89: ρ ≈ 0.45
30-59 ↔ 90+:   ρ ≈ 0.47  
60-89 ↔ 90+:   ρ ≈ 0.49
```

Đây là **multicollinearity** — cả 3 features đều đo cùng một hiện tượng (lịch sử trễ hạn) nhưng ở mức độ severity khác nhau. Hệ quả:

- **Logistic Regression:** Coefficients có thể không ổn định (high variance), khó interpret riêng lẻ. Cần kiểm tra VIF và áp dụng L2 regularization để ổn định coefficients.
- **Random Forest / XGBoost:** Multicollinearity không ảnh hưởng đến predictions, nhưng **feature importance sẽ bị pha loãng** giữa các features tương quan cao.

**Feature Engineering:** `TotalDelinquencyScore = 3×(90+) + 2×(60-89) + 1×(30-59)` giải quyết đa cộng tuyến bằng cách **tổng hợp 3 chỉ báo thành 1 điểm tổng hợp** có ý nghĩa tài chính rõ ràng (trọng số theo mức độ nghiêm trọng).

---

## 8. Domain Insights — Delinquency Tiering

Phân tích default rate theo số lần trễ hạn tiết lộ **non-linear escalation pattern**:

**90+ days late (most severe tier):**
- 0 lần: default rate ~5,5% (dưới mức nền 6,68%)
- 1 lần: default rate ~35–40%
- 2 lần: default rate ~60%+
- 3+ lần: default rate >70%

Chỉ 1 lần trễ 90+ ngày đã làm tăng default rate lên gấp **6–7 lần** so với mức nền 6,68%. Điều này nhất quán với FICO score methodology: một "major derogatory" (90+ days late) là **yếu tố giảm điểm mạnh nhất** trong credit scoring.

**30-59 days late (mild tier):**
- 0 lần: default rate ~5.5%
- 1 lần: default rate ~18–22%
- 3+ lần: default rate ~40%+

Escalation nhỏ hơn, phù hợp với việc trễ nhẹ có thể là "mistake" chứ không hẳn là "inability to pay".

**Insight cho feature engineering:** Weighting 3:2:1 trong TotalDelinquencyScore không phải arbitrary — nó phản ánh chính xác **asymmetric severity escalation** mà EDA reveal.

---

## 9. Credit Utilization Analysis

Default rate theo utilization bucket:

| Bucket | Default Rate | vs Baseline |
|--------|-------------|-------------|
| 0–10% | ~4.5% | Below baseline |
| 10–30% | ~5.8% | Near baseline |
| 30–50% | ~7.2% | Slightly above |
| 50–70% | ~9.5% | 1.4× baseline |
| 70–90% | ~12% | 1.8× baseline |
| 90–100% | ~15%+ | 2.2× baseline |
| >100% | ~25%+ | 3.7× baseline |

Tỷ lệ default tăng **monotonically và convex** theo utilization — higher utilization không chỉ là signal tuyến tính mà là signal **ngày càng quan trọng hơn** khi gần maxed out.

**Interaction hypothesis:** Credit utilization cao + delinquency history đồng thời = **multiplicative risk signal**, không chỉ additive. Đây là rationale cho `FinancialStressIndex = RevolvingUtilization × TotalDelinquencyScore` trong Phase 2.

---

## 10. Income & DebtRatio Analysis

### MonthlyIncome
Distribution (log scale) cho thấy median income ~$5,400/tháng, phù hợp với median household income tại Mỹ thời điểm 2011 (~$65,000/năm). Default rate giảm **monotonically** theo income quantile:

- Q1 (lowest income): default rate ~10–12%
- Q10 (highest income): default rate ~3–4%

Điều này nhất quán với **capacity to repay** — income càng cao, càng dễ dàng trả nợ khi có shock tài chính (mất việc, y tế...).

### DebtRatio — Vấn Đề Cần Chú Ý
`DebtRatio` trong dataset này có **rất nhiều giá trị > 1** (DTI > 100%), thậm chí lên đến hàng nghìn. Đây có thể là do:
1. Tử số (monthly debt) bao gồm cả non-cash obligations (insurance, taxes) theo định nghĩa rộng hơn standard DTI
2. Khi income = 0 hoặc rất nhỏ, ratio phình to vô nghĩa

**Implication:** `DebtRatio` raw không đáng tin cậy bằng `DebtToIncomeRatio = DebtRatio × MonthlyIncome` (DTI tuyệt đối, đơn vị USD/tháng) sau khi MonthlyIncome đã được impute. Feature này sẽ được tạo trong Phase 2.

---

## 11. Kết luận và Hành động cho Phase 2

### Cleaning Actions (có thứ tự ưu tiên)
1. **Xóa rows age == 0** (1 row) — data error không thể cứu vãn
2. **Capping MonthlyIncome** tại 99th percentile ($25,000) — giữ rank, loại extreme
3. **Capping RevolvingUtilization** tại 99th percentile — giữ monotonic relationship
4. **Capping delinquency counts** tại 99th percentile mỗi feature — loại sentinel values

### Imputation Actions
5. **KNN Imputer (k=5)** cho MonthlyIncome: justified by MAR evidence (χ²=67.89, p≈0)
6. **Median Imputation** cho NumberOfDependents: gần MCAR, right-skewed → median = 0

### Feature Engineering (theo thứ tự tạo)
7. `TotalDelinquencyScore` (cần trước bước 8, 9)
8. `FinancialStressIndex = RevolvingUtilization × TotalDelinquencyScore`
9. `DebtToIncomeRatio = DebtRatio × MonthlyIncome` (sau khi impute income)
10. `DelinquencyTrend = (30-59) - (90+)` — trajectory indicator

### Modeling Implications từ EDA
- **AUC-ROC** là primary metric (không phải Accuracy)
- `scale_pos_weight = 13.96` cho XGBoost
- `class_weight='balanced'` cho sklearn models
- Spearman correlation > Pearson cho feature selection với data này
- Kiểm tra VIF cho multicollinearity trước Logistic Regression

---

*Figures: `reports/fig_01_target_distribution.png` đến `fig_10_debt_income.png`*
*Notebook đầy đủ: `notebooks/01_EDA.ipynb`*
