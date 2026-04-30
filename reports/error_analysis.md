# Báo cáo Phân tích Sai số & Chẩn đoán
## Loan Default Prediction | Give Me Some Credit

**Mô hình:** XGBoost (AUC=0,8714, ngưỡng tối ưu=0,77)  
**Notebook:** `04_Analysis.ipynb`  
**Figures:** fig_22 – fig_30

---

## 1. Hồ sơ Lỗi — FN và FP là ai?

### 1.1 Định nghĩa 4 Loại Lỗi

| Category | Ý nghĩa | Hậu quả |
|----------|---------|---------|
| **TP** (True Positive) | Vỡ nợ thực sự, mô hình cảnh báo đúng | Ngân hàng từ chối → tránh được khoản vay xấu |
| **FN** (False Negative) | Vỡ nợ thực sự, mô hình **KHÔNG** cảnh báo | Ngân hàng cho vay → mất gốc + lãi |
| **FP** (False Positive) | Không vỡ nợ, mô hình cảnh báo nhầm | Ngân hàng từ chối → mất chi phí cơ hội |
| **TN** (True Negative) | Không vỡ nợ, mô hình đúng | Ngân hàng cho vay → lợi nhuận bình thường |

FN là error type **nghiêm trọng nhất** trong bối cảnh tín dụng — đây là những khoản vay xấu mà ngân hàng không biết.

### 1.2 Distribution Analysis (fig_22, fig_23)

**FN vs TP comparison (cả hai đều là actual defaults, y=1):**

| Đặc trưng | FN trung vị | TP trung vị | Tỷ lệ FN/TP | Diễn giải |
|---------|-----------|-----------|-------------|----------------|
| RevolvingUtilization | 0,518 | 1,000 | 0,52 | FN ít áp lực tín dụng hơn |
| TotalDelinquencyScore | 0,000 | 5,000 | 0,00 | **FN không có lịch sử trễ hạn!** |
| FinancialStressIndex | 0,000 | 4,000 | 0,00 | FN có chỉ số = 0 (không căng thẳng) |
| NumberOfTimes90DaysLate | 0,000 | 1,000 | 0,00 | FN chưa từng trễ >90 ngày |
| MonthlyIncome | $4.200 | $3.400 | 1,24 | FN có thu nhập cao hơn TP! |

False Negatives là những người vỡ nợ trông lành mạnh — lịch sử trễ hạn nhẹ hơn và tỷ lệ sử dụng tín dụng thấp hơn TP. Mô hình không cảnh báo được vì đặc trưng của họ không đủ để vượt ngưỡng 0,77.

**FP vs TN comparison (cả hai đều là actual non-defaults, y=0):**

| Đặc trưng | FP trung vị | TN trung vị | Tỷ lệ FP/TN | Diễn giải |
|---------|-----------|-----------|-------------|----------------|
| RevolvingUtilization | 0,958 | 0,118 | 8,10 | **FP có tỷ lệ sử dụng gần giới hạn** |
| TotalDelinquencyScore | 4,000 | 0,000 | >>1 | FP có lịch sử trễ hạn |
| FinancialStressIndex | 2,856 | 0,000 | >>1 | FP trông rất giống người vỡ nợ |
| MonthlyIncome | $3.610 | $4.505 | 0,80 | FP có thu nhập thấp hơn TN |

False Positives là những người có tỷ lệ sử dụng tín dụng cao và vài lần trễ nhỏ — đặc trưng trùng với hồ sơ người vỡ nợ — nhưng thực tế vẫn trả được nợ. Mô hình nhầm vì không phân biệt được nhóm này.

### 1.3 Probability Score Analysis (fig_24)

Phân tích score distribution cho thấy:

- **Điểm FN:** trung vị = **0,523** (ngưỡng=0,77)
  → Cách xa ngưỡng ~0,25 điểm — không phải cận ngưỡng, mô hình đánh giá thấp họ
- **Điểm TP:** trung vị = **0,912** → mô hình tự tin ở trường hợp vỡ nợ điển hình
- **Điểm FP:** trung vị = **0,866** → mô hình tự tin quá mức khi từ chối nhóm này

FN và FP có cấu trúc lỗi khác nhau. FN (điểm trung vị=0,523) nằm **xa** ngưỡng 0,77 — không phải cận ngưỡng mà là người vỡ nợ trông lành mạnh: mô hình tự tin phân loại họ là an toàn vì đặc trưng không có dấu hiệu cảnh báo rõ ràng. Ngược lại, FP (điểm trung vị=0,866) nằm **trên** ngưỡng — mô hình tự tin quá mức khi từ chối, dù họ không thực sự vỡ nợ. Nhóm thực sự cận ngưỡng là tập nhỏ nằm trong khoảng score 0.7–0.8.

---

## 2. Phân tích Chi phí Kinh doanh

### 2.1 Khung Chi phí

Dựa trên mô hình chi phí rủi ro tín dụng thực tế:

```
FN cost per case = Average_loan × (1 - Recovery_rate)
                 = $15,000 × (1 - 0.25) = $11,250 per missed default

FP cost per case = Opportunity_cost_per_rejected_good_customer = $500
```

**Tỷ lệ FN:FP = 22,5:1** — FN đắt gấp 22,5 lần FP. Điều này biện minh cho việc dùng ngưỡng thấp hơn để tăng Recall.

### 2.2 So sánh Chi phí theo Ngưỡng

| Ngưỡng | FN count | FP count | Tổng chi phí | So với t=0,77 |
|-----------|----------|----------|------------|---------------|
| 0.50 (naive) | 338 | 4,079 | **$5,842,000** | -33.5% (cheaper!) |
| 0.625 (F2 opt) | ~498 | ~2,354 | **~$6,779,500** | -22.9% |
| 0.77 (F1 opt) | 728 | 1,195 | **$8,787,500** | baseline |

t=0.5 có tổng chi phí thấp nhất vì FN đắt gấp 22 lần FP: giảm 390 FN tiết kiệm $4.4M, dù FP tăng thêm 2.884 chỉ tốn thêm $1.4M. Tối ưu ngưỡng phải dựa trên chi phí thực tế, không phải F1.

→ Ngưỡng F2 (0.625) không phải ngưỡng tối ưu chi phí — t=0.5 rẻ hơn. Đây là ngưỡng ưu tiên Recall trong khi giữ FP ở mức kiểm soát được so với t=0.5.

---

## 3. Tối ưu Ngưỡng — F-beta Score (β=2)

### 3.1 Lý thuyết

$$F_\beta = (1+\beta^2) \cdot \frac{\text{Precision} \times \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

Với β=2: Recall được trọng số **4×** so với Precision:
$$F_2 = 5 \cdot \frac{P \times R}{4P + R}$$

Trong credit risk, 1 FN (bỏ sót vỡ nợ) tốn ~$11.250, còn 1 FP (từ chối nhầm) tốn ~$500. FN:FP ≈ 22.5:1, nên β=2 là lựa chọn thận trọng hợp lý: ưu tiên Recall hơn Precision, còn phần chốt chi phí thực tế được xử lý riêng ở bảng cost analysis.

### 3.2 Ngưỡng Tối ưu (tập kiểm định)

| Ngưỡng | F1 | F2 | Precision | Recall |
|-----------|----|----|-----------|--------|
| 0.50 (naive) | 0.3455 | 0.5177 | 0.2223 | **0.7753** |
| 0.625 (F2 opt) | 0.4137 | **0.5365** | 0.2994 | 0.6689 |
| 0.775 (F1 opt) | **0.4443** | 0.4801 | 0.3951 | 0.5073 |
| 0.77 (F1 opt) | 0.4466 | 0.4858 | 0.3937 | 0.5160 |

F1-optimized (0.775) và ngưỡng Phase 3 (0.77) gần như giống nhau — nhất quán. F2-optimized (0.625) tăng Recall từ 51.6% → 66.9% (+15.3 pp), đánh đổi bằng Precision giảm từ 39.4% → 29.9%.

**Lựa chọn ngưỡng:** Sử dụng **t=0.625** (F2 opt) cho ngân hàng ưu tiên bắt nhiều defaults hơn. t=0.775 phù hợp hơn khi muốn FP thấp — ít từ chối nhầm hơn nhưng bỏ sót nhiều defaults hơn.

---

## 4. SHAP — Phân tích Sâu về Khả năng Giải thích

### 4.1 Tầm quan trọng Đặc trưng Tổng quát (fig_26, fig_26a)

Dựa trên SHAP TreeExplainer (3,000 test samples, exact Shapley values):

**Đặc trưng hàng đầu theo Mean |SHAP value|:**

| Hạng | Đặc trưng | Loại | Mean\|SHAP\| | Nhận xét |
|------|---------|------|-------------|---------|
| 1 | **FinancialStressIndex** | Được tạo | **0,577** | Số hạng tương tác nắm bắt sức mạnh tổng hợp tốt nhất |
| 2 | RevolvingUtilizationOfUnsecuredLines | Gốc | 0,535 | Áp lực tín dụng trực tiếp |
| 3 | **TotalDelinquencyScore** | Được tạo | 0,410 | Tổng hợp trễ hạn có trọng số |
| 4 | age | Gốc | 0,244 | Đại diện kinh nghiệm tài chính |
| 5 | NumberOfOpenCreditLinesAndLoans | Gốc | 0,168 | Tín hiệu đa dạng hóa/quá mức |
| 9 | **DebtToIncomeRatio** | Được tạo | 0,065 | R14: GIỮ (>5% của đặc trưng #2) |
| 14 | **DelinquencyTrend** | Được tạo | 0,002 | Thấp nhất — có thể bỏ sau |

**Nhận xét quan trọng:**
- **2 trong top 3** đặc trưng là được tạo → trích xuất đặc trưng Phase 2 được xác nhận mạnh
- FinancialStressIndex (#1, 0,577) > RevolvingUtilization (#2, 0,535) > TotalDelinquencyScore (#3, 0,410)
  → Số hạng tương tác (utilization × delinquency) quan trọng hơn từng thành phần riêng lẻ
- NumberOfTimes90DaysLate chỉ ở rank 12 (SHAP=0,013) mặc dù đây là tín hiệu mạnh nhất theo lĩnh vực
  → XGBoost học được nó qua FinancialStressIndex và TotalDelinquencyScore thay thế
- DelinquencyTrend (SHAP=0.002) rất thấp → có thể drop trong Phase 5
- R14 xác nhận: DTI SHAP=0,065 > ngưỡng 0,027 → **GIỮ DebtToIncomeRatio**

### 4.2 SHAP Dependence Analysis (fig_27)

**RevolvingUtilization:**
- SHAP gần 0 khi utilization < 0.3 → low risk zone
- SHAP tăng mạnh khi utilization > 0.7 → exponential risk zone
- Quan hệ phi tuyến — cần mô hình dựa trên cây để nắm bắt

**TotalDelinquencyScore:**
- SHAP = 0 khi score = 0 (không có lần trễ nào)
- SHAP tăng mạnh ngay khi score > 0
- Cực kỳ có tính dự đoán: bất kỳ lịch sử trễ hạn nào đều tăng rủi ro đáng kể

### 4.3 Individual SHAP Waterfall (fig_28)

**True Positive (TP):** Người vỡ nợ mô hình đúng
- RevolvingUtil cao + TotalDelinquencyScore cao đẩy dự đoán về phía vỡ nợ
- SHAP waterfall cho thấy rõ ràng các đặc trưng nào quyết định dự đoán

**False Positive (FP):** Người không vỡ nợ bị nhầm
- Có vài lần trễ hạn và tỷ lệ sử dụng tín dụng cao → mô hình phản ứng quá mức
- Nhưng các đặc trưng khác (income, age) cân bằng không đủ

**False Negative (FN):** Người vỡ nợ mô hình bỏ sót
- RevolvingUtil và TotalDelinquencyScore ở mức trung bình, không đủ để vượt ngưỡng
- Giá trị nền thấp → cần nhiều "bằng chứng dương" hơn để đẩy dự đoán đủ cao

### 4.4 R14 Verification: DebtToIncomeRatio

Từ Phase 2 (R14): cần kiểm tra nếu DTI SHAP < 5% của đặc trưng #2 thì đề xuất loại bỏ.

Kết quả từ SHAP analysis:
- Nếu DTI mean|SHAP| < ngưỡng_r14 → **Loại bỏ DTI** trong Phase 5 (form Streamlit)
- Nếu DTI mean|SHAP| ≥ ngưỡng_r14 → **Giữ DTI** trong tập đặc trưng

**Kết quả (R14 verification):**
- DTI mean|SHAP| = 0,065, ngưỡng (5% của #2) = 0,027
- **KẾT LUẬN: GIỮ DebtToIncomeRatio** → DTI SHAP vượt ngưỡng gần 2,5× → không thể loại bỏ
- R14 hypothesis bị bác bỏ: DTI có contribution đáng kể mặc dù L1 zero-out ở Logistic Regression

---

## 5. Learning Curve Diagnostics (fig_29)

### 5.1 Khung Thiên lệch-Phương sai

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
- Gap cuối = 0,021 < 0,03 → **"Khớp tốt — mô hình tổng quát hóa tốt"**
- Train AUC giảm từ 0,96 → 0,88 khi N tăng: tập huấn luyện đa dạng hơn → khó quá khớp hơn

### 5.3 Recommendations

| Diagnosis | Action |
|-----------|--------|
| High variance (gap > 0.06) | Increase reg_lambda, decrease max_depth, increase subsample |
| High bias (both AUC < 0.82) | Thêm đặc trưng, thử max_depth cao hơn, giảm regularization |
| Good fit (gap < 0.03) | Ổn — cân nhắc trích xuất đặc trưng cho vòng lặp tiếp theo |
| Plateau (val AUC flat) | Thêm dữ liệu không giúp — tập trung trích xuất đặc trưng |

**Kết luận cho bộ dữ liệu này:** XGBoost ở điểm cân bằng — chuẩn hóa hiệu quả, không quá khớp/dưới khớp. Trích xuất đặc trưng (TotalDelinquencyScore, FinancialStressIndex) đã tối đa hóa khai thác thông tin từ các đặc trưng hiện có.

---

## 6. Summary & Recommendations

### 6.1 Kết luận Phân tích Sai số

1. **FN — người vỡ nợ bị bỏ sót:** Điểm trung vị=0,523, cách xa ngưỡng 0,247. Đây không phải cận ngưỡng mà là những người không có lịch sử trễ hạn (TotalDelinquencyScore=0, FinancialStressIndex=0) — mô hình không có bằng chứng để cảnh báo. Thu nhập ($4.200) cao hơn TP ($3.400), trông như hồ sơ tốt nhưng vẫn vỡ nợ, có thể do sự kiện bất ngờ không để lại dấu vết trong credit history.

2. **FP — khách hàng tốt bị từ chối nhầm:** Có một vài lần trễ hạn và tỷ lệ sử dụng tín dụng cao, trông giống hồ sơ người vỡ nợ — nhưng thực tế vẫn trả được. Mô hình chưa nắm được xu hướng cải thiện của nhóm này.

3. **Ngưỡng là đòn bẩy chính:** Từ ngưỡng=0,77 (F1 opt) sang ngưỡng F2-tối ưu → Recall tăng đáng kể, chi phí kinh doanh giảm, chấp nhận thêm FP.

### 6.2 Trích xuất Đặc trưng Đã xác nhận

SHAP analysis xác nhận:
- **TotalDelinquencyScore** (được tạo) > đặc trưng trễ hạn gốc → cơ chế tính trọng số hoạt động
- **FinancialStressIndex** (được tạo) nắm bắt tương tác phi tuyến → không có trong đặc trưng gốc
- Trích xuất đặc trưng đóng góp ~2 trong top-5 SHAP → **được chứng minh**

### 6.3 Khuyến nghị Triển khai Mô hình

| Bối cảnh triển khai | Ngưỡng khuyến nghị | Lý do |
|---------------------|-------------------|-------|
| Ngân hàng thận trọng | F2-tối ưu (~0,30–0,45) | Tối đa Recall, chi phí FN >> FP |
| Tổ chức cho vay FinTech | F1-tối ưu (~0,45–0,55) | Cân bằng tăng trưởng và rủi ro |
| Sàng lọc khối lượng lớn | Phân tầng: Thấp/Trung bình/Cao/Rất cao | Phân loại đa ngưỡng |

### 6.4 Nợ Kỹ thuật & Hành động Phase 5

- **DebtToIncomeRatio:** Kiểm tra R14 — nếu SHAP thấp, loại bỏ khỏi form Streamlit
- **Tích hợp SHAP:** Dùng `shap.TreeExplainer` trong Streamlit cho biểu đồ waterfall thời gian thực
- **Tham số ngưỡng:** Đưa thanh trượt ngưỡng vào dashboard để người dùng thử nghiệm
- **Lưu mô hình:** `best_model.pkl` (XGBoost) đã sẵn sàng cho ứng dụng Streamlit

---

*Phase 4 hoàn thành. Figures: fig_22–fig_30. Tiếp theo: Phase 5 — Bảng điều khiển Streamlit.*
