# Báo cáo EDA — Phân tích Dữ liệu Khám phá
## Loan Default Prediction | Give Me Some Credit

**Bộ dữ liệu:** `data/raw/cs-training.csv`
**Notebook:** `notebooks/01_EDA.ipynb`

---

## 1. Tổng quan Bộ dữ liệu

Bộ dữ liệu **Give Me Some Credit** được Kaggle cung cấp bởi một tổ chức tài chính Mỹ, bao gồm hồ sơ vay vốn của 150,000 khách hàng cùng lịch sử tín dụng tương ứng. Mục tiêu là dự báo xác suất một khách hàng sẽ **trễ thanh toán nghiêm trọng (≥ 90 ngày)** trong vòng 2 năm tới — đây là định nghĩa "vỡ nợ" được sử dụng trong thực tiễn báo cáo tín dụng tại Mỹ.

| Thuộc tính | Giá trị |
|-----------|---------|
| Số hồ sơ | 150,000 |
| Số đặc trưng | 10 (tất cả kiểu số) |
| Target | `SeriousDlqin2yrs` (binary: 0/1) |
| Bộ nhớ | ~12.6 MB |

Điểm đặc biệt quan trọng: **không có đặc trưng phân loại**. Toàn bộ đầu vào đều là số thực — đơn giản hóa tiền xử lý, nhưng cũng có nghĩa là trích xuất đặc trưng thủ công (tạo các số hạng tương tác có ý nghĩa tài chính) sẽ tạo ra giá trị lớn.

---

## 2. Phân tích Mất cân bằng Lớp

Đây là vấn đề trung tâm của toàn bộ dự án.

```
SeriousDlqin2yrs = 0 (Không vỡ nợ): 139,974 hồ sơ (93.32%)
SeriousDlqin2yrs = 1 (Vỡ nợ):       10,026 hồ sơ  ( 6.68%)
Tỷ lệ mất cân bằng: 1:14.0
```

**Tại sao Accuracy hoàn toàn vô nghĩa trong bài toán này:**

Một mô hình "ngây thơ" chỉ dự đoán `0` cho mọi đầu vào sẽ đạt Accuracy = 93.32% — nghe có vẻ ấn tượng, nhưng Recall = 0%, nghĩa là mô hình đó **không có bất kỳ giá trị kinh doanh nào**. Mọi người vỡ nợ đều bị bỏ sót.

Trong bối cảnh tín dụng, hai loại lỗi có chi phí hoàn toàn bất cân xứng:

- **False Negative** (mô hình dự đoán 0, thực tế là 1): Ngân hàng cho vay người sẽ vỡ nợ → mất toàn bộ khoản vay (tổn thất gốc). Đây là chi phí tài chính trực tiếp, thường lên đến hàng chục nghìn USD mỗi hồ sơ.
- **False Positive** (mô hình dự đoán 1, thực tế là 0): Ngân hàng từ chối người đủ điều kiện → mất cơ hội kinh doanh (chi phí cơ hội). Chi phí này nhỏ hơn đáng kể.

Do đó: **Chỉ số chính = AUC-ROC** (đo khả năng xếp hạng, độc lập với ngưỡng và tỷ lệ mất cân bằng). Phụ: F1-Score, đường cong Precision-Recall. Ngưỡng tối ưu sẽ được xác định ở Phase 4 dựa trên hàm chi phí kinh doanh, không cố định ở 0,5.

**Hệ quả cho lập mô hình:** `scale_pos_weight = 13.96` cho XGBoost; `class_weight='balanced'` cho các mô hình sklearn; SMOTE chỉ áp dụng trên tập huấn luyện.

---

## 3. Giá trị Thiếu — Phân tích Cơ chế

Chỉ có 2 đặc trưng có giá trị thiếu:

| Đặc trưng | Số thiếu | % Thiếu | Cơ chế |
|-----------|----------|---------|--------|
| `MonthlyIncome` | 29,731 | **19.82%** | MAR/MNAR |
| `NumberOfDependents` | 3,924 | **2.62%** | Gần MCAR |

### 3.1 MonthlyIncome — Thiếu Không Phải Ngẫu Nhiên

Kiểm định Chi-squared giữa missing indicator của `MonthlyIncome` và target `SeriousDlqin2yrs`:

```
χ² = 67.89, p-value ≈ 0 (< 0.0001)
Kết luận: Thiếu KHÔNG độc lập với target → MAR hoặc MNAR
```

Giải thích theo domain: Người không có thu nhập ổn định (thất nghiệp, freelancer, thu nhập tiền mặt) có xu hướng không khai báo income. Đây chính xác là nhóm có rủi ro tín dụng cao hơn. Do đó, việc thiếu giá trị MonthlyIncome **chính nó là dấu hiệu** của rủi ro.

**Hệ quả cho ước tính thiếu:** Không thể dùng mean/median đơn giản (giả định MCAR). Cần KNN Imputer để khai thác tương quan với các đặc trưng khác (age, DebtRatio) trong ước tính income. Sẽ so sánh thực nghiệm hai phương pháp trong Phase 2.

### 3.2 NumberOfDependents — Thiếu Gần Ngẫu Nhiên

Với chỉ 2.62% thiếu và không có pattern rõ ràng, ước tính trung vị (= 0, do phân phối lệch phải) là đủ và ít rủi ro rò rỉ dữ liệu hơn KNN.

---

## 4. Ngoại lệ — Phân tích và Kế hoạch Xử lý

EDA phát hiện nhiều ngoại lệ cần xử lý trước khi lập mô hình:

### 4.1 age == 0 (1 row)
Giá trị 0 tuổi không có nghĩa tài chính — không thể ký hợp đồng vay vốn ở tuổi 0. Đây là **lỗi nhập liệu** rõ ràng. **Hành động: XÓA** toàn bộ hàng này.

### 4.2 RevolvingUtilizationOfUnsecuredLines > 1 (3,321 rows, 2.2%)
Credit utilization > 100% (tức là đang nợ nhiều hơn hạn mức được cấp) **có thể xảy ra trong thực tế** khi lãi suất và phí phạt được tính thêm. Tuy nhiên, 241 hàng có utilization > 10 (1000%) — đây là **ngoại lệ cực đoan** không có ý nghĩa tài chính hợp lý. **Hành động: Capping tại 99th percentile** (không xóa để giữ thông tin xếp hạng tương đối).

### 4.3 MonthlyIncome max = $3,008,750/tháng
Giá trị cực đại $3M/tháng vượt xa 99th percentile ($25,000). Trong bối cảnh đây là bộ dữ liệu cho vay tiêu dùng, không phải ngân hàng doanh nghiệp, income $3M/tháng là bất thường. **Hành động: Capping tại 99th percentile = $25,000**.

### 4.4 Delinquency Counts max = 98 (cho cả 3 đặc trưng)
Một khách hàng trễ hạn 98 lần trong vòng vài năm là **không thể xảy ra về mặt thực tế** — mỗi tháng chỉ có 1 kỳ thanh toán. Giá trị 96–98 xuất hiện ở cả 3 đặc trưng trễ hạn, gợi ý đây là **giá trị sentinel** (giá trị đặc biệt để đánh dấu "không xác định" hoặc "lỗi") thay vì dữ liệu thực.

```
NumberOfTimes90DaysLate:              99th pct = 3, max = 98
NumberOfTime30-59DaysPastDueNotWorse: 99th pct = 4, max = 98  
NumberOfTime60-89DaysPastDueNotWorse: 99th pct = 2, max = 98
```

**Hành động: Capping tại 99th percentile** cho mỗi đặc trưng.

---

## 5. Phân tích Đơn biến — Phân phối Đặc trưng

### 5.1 Phân phối Đặc trưng Trễ hạn (Lệch Phải Cực độ)

Ba đặc trưng trễ hạn (`30-59`, `60-89`, `90+` days) đều có phân phối **cực kỳ lệch phải**:
- Đại đa số (>80%) có giá trị = 0: không trễ hạn lần nào
- Đuôi dài (lệch phải nặng) với giá trị tới 98
- Skewness > 15 cho cả ba đặc trưng

Từ góc độ tài chính, điều này phản ánh thực tế: hầu hết người đi vay đều trả đúng hạn. Người trễ hạn nhiều lần là thiểu số nhưng quan trọng (lớp dương tính).

**Hệ quả cho chuẩn hóa:** StandardScaler sẽ kéo dãn phân phối lệch này không hiệu quả. Với Logistic Regression, cần cân nhắc **RobustScaler** (dùng median và IQR thay vì mean và std) sau khi capping.

### 5.2 RevolvingUtilizationOfUnsecuredLines — Hai đỉnh

Phân phối có xu hướng hai đỉnh:
- Đỉnh đầu (utilization ≈ 0): người dùng ít tín dụng hoặc có nhiều hạn mức chưa dùng
- Đỉnh thứ hai (utilization ≈ 1): người gần đầy hạn mức

Điều này phù hợp với lý thuyết FICO score: nhóm có utilization cực thấp (có hạn mức nhưng không cần dùng) và nhóm utilization cao (cần tín dụng thường xuyên) là 2 phân khúc rất khác nhau về hành vi tài chính.

### 5.3 Age — Phân phối Chuẩn Lệch Phải

Age có phân phối gần chuẩn với lệch phải nhẹ, khoảng hợp lý từ 18 đến 109 tuổi (ngoại trừ 1 giá trị = 0 cần xóa). Không cần biến đổi đặc biệt.

---

## 6. Phân tích Song biến — Năng lực Phân biệt

Mann-Whitney U test (phi tham số, bền vững với phân phối lệch phải) cho thấy **tất cả 10 đặc trưng đều có phân phối khác biệt có ý nghĩa thống kê** giữa người vỡ nợ và không vỡ nợ (p < 0,001 sau hiệu chỉnh Bonferroni).

**Chỉ số phân biệt hàng đầu (Spearman correlation với target):**

| Hạng | Đặc trưng | Spearman ρ | Diễn giải Lĩnh vực |
|------|-----------|-----------|-------------------|
| 1 | `NumberOfTimes90DaysLate` | **+0,3423** | Lịch sử trễ hạn nặng — tương quan mạnh nhất; ghi trong báo cáo tín dụng 7 năm |
| 2 | `NumberOfTime60-89DaysPastDueNotWorse` | **+0,2771** | Trễ hạn mức trung — dấu hiệu bắt đầu kiệt sức tài chính |
| 3 | `NumberOfTime30-59DaysPastDueNotWorse` | **+0,2574** | Trễ hạn nhẹ — cảnh báo sớm |
| 4 | `RevolvingUtilizationOfUnsecuredLines` | **+0,2404** | Căng thẳng tài chính — đang gần đầy hạn mức thẻ tín dụng |
| 5 | `age` | **-0,1170** | Người trẻ có rủi ro cao hơn (ít kinh nghiệm tài chính, thu nhập chưa ổn định) |

**Tại sao Pearson và Spearman cho kết quả khác nhau:**

Pearson correlation của `RevolvingUtilization` với target chỉ là **-0,002** (gần zero!), nhưng Spearman là **+0,2404**. Sự khác biệt lớn này chỉ ra rằng mối quan hệ là **đơn điệu nhưng không tuyến tính** — phù hợp với đặc điểm lệch phải của đặc trưng này. Dùng Pearson sẽ **đánh giá thấp** đáng kể tầm quan trọng của RevolvingUtilization.

**Bài học:** Với bộ dữ liệu tài chính có phân phối đuôi nặng, **Spearman correlation là lựa chọn đúng đắn hơn Pearson** cho chọn đặc trưng.

---

## 7. Phân tích Đa cộng tuyến

Ma trận tương quan Spearman cho thấy 3 đặc trưng trễ hạn có tương quan đáng kể với nhau:

```
30-59 ↔ 60-89: ρ ≈ 0.45
30-59 ↔ 90+:   ρ ≈ 0.47  
60-89 ↔ 90+:   ρ ≈ 0.49
```

Đây là **đa cộng tuyến** — cả 3 đặc trưng đều đo cùng một hiện tượng (lịch sử trễ hạn) nhưng ở mức độ nghiêm trọng khác nhau. Hệ quả:

- **Logistic Regression:** Hệ số có thể không ổn định (phương sai cao), khó diễn giải riêng lẻ. Cần kiểm tra VIF và áp dụng L2 regularization để ổn định hệ số.
- **Random Forest / XGBoost:** Đa cộng tuyến không ảnh hưởng đến dự đoán, nhưng **tầm quan trọng đặc trưng sẽ bị pha loãng** giữa các đặc trưng tương quan cao.

**Trích xuất đặc trưng:** `TotalDelinquencyScore = 3×(90+) + 2×(60-89) + 1×(30-59)` giải quyết đa cộng tuyến bằng cách **tổng hợp 3 chỉ báo thành 1 điểm tổng hợp** có ý nghĩa tài chính rõ ràng (trọng số theo mức độ nghiêm trọng).

---

## 8. Nhận xét Lĩnh vực — Phân tầng Trễ hạn

Phân tích tỷ lệ vỡ nợ theo số lần trễ hạn tiết lộ **mẫu leo thang phi tuyến**:

**90+ ngày trễ (tầng nghiêm trọng nhất):**
- 0 lần: tỷ lệ vỡ nợ ~5,5% (dưới mức nền 6,68%)
- 1 lần: tỷ lệ vỡ nợ ~35–40%
- 2 lần: tỷ lệ vỡ nợ ~60%+
- 3+ lần: tỷ lệ vỡ nợ >70%

Chỉ 1 lần trễ 90+ ngày đã làm tăng tỷ lệ vỡ nợ lên gấp **6–7 lần** so với mức nền 6,68%. Điều này nhất quán với phương pháp FICO score: một "vi phạm nghiêm trọng" (90+ ngày trễ) là **yếu tố giảm điểm mạnh nhất** trong chấm điểm tín dụng.

**30-59 ngày trễ (tầng nhẹ):**
- 0 lần: tỷ lệ vỡ nợ ~5,5%
- 1 lần: tỷ lệ vỡ nợ ~18–22%
- 3+ lần: tỷ lệ vỡ nợ ~40%+

Leo thang nhỏ hơn, phù hợp với việc trễ nhẹ có thể là "sai lầm" chứ không hẳn là "không có khả năng trả nợ".

**Nhận xét cho trích xuất đặc trưng:** Trọng số 3:2:1 trong TotalDelinquencyScore không phải tùy tiện — nó phản ánh chính xác **leo thang mức độ nghiêm trọng bất đối xứng** mà EDA tiết lộ.

---

## 9. Phân tích Mức sử dụng Tín dụng

Tỷ lệ vỡ nợ theo phân khúc utilization:

| Phân khúc | Tỷ lệ vỡ nợ | so với Mức nền |
|-----------|------------|--------------|
| 0–10% | ~4,5% | Dưới mức nền |
| 10–30% | ~5,8% | Gần mức nền |
| 30–50% | ~7,2% | Hơi trên mức nền |
| 50–70% | ~9,5% | 1,4× mức nền |
| 70–90% | ~12% | 1,8× mức nền |
| 90–100% | ~15%+ | 2,2× mức nền |
| >100% | ~25%+ | 3,7× mức nền |

Tỷ lệ vỡ nợ tăng **đơn điệu và lồi** theo utilization — mức sử dụng cao không chỉ là tín hiệu tuyến tính mà là tín hiệu **ngày càng quan trọng hơn** khi gần đầy hạn mức.

**Giả thuyết tương tác:** Credit utilization cao + lịch sử trễ hạn đồng thời = **tín hiệu rủi ro nhân**, không chỉ cộng. Đây là lý do cho `FinancialStressIndex = RevolvingUtilization × TotalDelinquencyScore` trong Phase 2.

---

## 10. Phân tích Thu nhập & DebtRatio

### MonthlyIncome
Phân phối (thang log) cho thấy thu nhập trung vị ~$5.400/tháng, phù hợp với thu nhập hộ gia đình trung vị tại Mỹ thời điểm 2011 (~$65.000/năm). Tỷ lệ vỡ nợ giảm **đơn điệu** theo phân vị thu nhập:

- Q1 (thu nhập thấp nhất): tỷ lệ vỡ nợ ~10–12%
- Q10 (thu nhập cao nhất): tỷ lệ vỡ nợ ~3–4%

Điều này nhất quán với **khả năng trả nợ** — thu nhập càng cao, càng dễ dàng trả nợ khi có cú sốc tài chính (mất việc, y tế...).

### DebtRatio — Vấn Đề Cần Chú Ý
`DebtRatio` trong bộ dữ liệu này có **rất nhiều giá trị > 1** (DTI > 100%), thậm chí lên đến hàng nghìn. Đây có thể là do:
1. Tử số (monthly debt) bao gồm cả non-cash obligations (insurance, taxes) theo định nghĩa rộng hơn standard DTI
2. Khi income = 0 hoặc rất nhỏ, ratio phình to vô nghĩa

**Hệ quả:** `DebtRatio` thô không đáng tin cậy bằng `DebtToIncomeRatio = DebtRatio × MonthlyIncome` (DTI tuyệt đối, đơn vị USD/tháng) sau khi MonthlyIncome đã được ước tính. Đặc trưng này sẽ được tạo trong Phase 2.

---

## 11. Kết luận và Hành động cho Phase 2

### Hành động Làm sạch (có thứ tự ưu tiên)
1. **Xóa rows age == 0** (1 row) — data error không thể cứu vãn
2. **Capping MonthlyIncome** tại 99th percentile ($25.000) — giữ thứ hạng tương đối, loại ngoại lệ cực đoan
3. **Capping RevolvingUtilization** tại 99th percentile — giữ quan hệ đơn điệu
4. **Capping delinquency counts** tại 99th percentile mỗi đặc trưng — loại giá trị sentinel

### Hành động Ước tính Thiếu
5. **KNN Imputer (k=5)** cho MonthlyIncome: đã xác nhận bằng bằng chứng MAR (χ²=67,89, p≈0)
6. **Ước tính trung vị** cho NumberOfDependents: gần MCAR, lệch phải → median = 0

### Trích xuất Đặc trưng (theo thứ tự tạo)
7. `TotalDelinquencyScore` (cần trước bước 8, 9)
8. `FinancialStressIndex = RevolvingUtilization × TotalDelinquencyScore`
9. `DebtToIncomeRatio = DebtRatio × MonthlyIncome` (sau khi impute income)
10. `DelinquencyTrend = (30-59) - (90+)` — trajectory indicator

### Hệ quả Lập mô hình từ EDA
- **AUC-ROC** là chỉ số chính (không phải Accuracy)
- `scale_pos_weight = 13.96` cho XGBoost
- `class_weight='balanced'` cho các mô hình sklearn
- Spearman correlation > Pearson cho chọn đặc trưng với bộ dữ liệu này
- Kiểm tra VIF cho đa cộng tuyến trước Logistic Regression

---

*Figures: `reports/fig_01_target_distribution.png` đến `fig_10_debt_income.png`*
*Notebook đầy đủ: `notebooks/01_EDA.ipynb`*
