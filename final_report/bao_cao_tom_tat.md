# DỰ BÁO RỦI RO VỠ NỢ TÍN DỤNG BẰNG HỌC MÁY
## Bản Tóm Tắt — Loan Default Prediction Using Machine Learning


> *Bản tóm tắt trình bày các kết quả cốt lõi. Cơ sở lý thuyết đầy đủ, biện giải toán học và phụ lục kỹ thuật nằm trong báo cáo chính (`final_report/bao_cao_chinh.md`).*

---

## TÓM TẮT

Nghiên cứu ứng dụng học máy vào dự báo rủi ro vỡ nợ tín dụng, sử dụng bộ dữ liệu *Give Me Some Credit* (Kaggle, 150.000 hồ sơ vay). Bốn mô hình được xây dựng theo thứ tự tăng dần độ phức tạp: Logistic Regression (AUC=0,8432), Decision Tree (AUC=0,8579), Random Forest (AUC=0,8703), XGBoost (AUC=0,8714). Trích xuất đặc trưng có chủ đích từ lĩnh vực tài chính tạo ra hai đặc trưng xếp top-3 SHAP. Ngưỡng F2-tối ưu t=0,625 tăng Recall từ 51,6% lên 66,9%, giảm chi phí ước tính $2M so với ngưỡng F1-tối ưu (t=0,775). Sản phẩm là ứng dụng Streamlit giải thích từng quyết định tín dụng bằng SHAP waterfall.

---

## MỤC LỤC

1. [Giới thiệu](#chương-1-giới-thiệu)
2. [Cơ sở lý thuyết](#chương-2-cơ-sở-lý-thuyết)
3. [Dữ liệu và Tiền xử lý](#chương-3-dữ-liệu-và-tiền-xử-lý)
4. [Thực nghiệm và Đánh giá](#chương-4-thực-nghiệm-và-đánh-giá)
5. [Sản phẩm — Streamlit Dashboard](#chương-5-sản-phẩm--streamlit-dashboard)
6. [Kết luận và Hướng phát triển](#chương-6-kết-luận-và-hướng-phát-triển)
7. [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

# CHƯƠNG 1: GIỚI THIỆU

## 1.1 Tính cấp thiết

Rủi ro tín dụng là nguy cơ lớn nhất đối với sự ổn định tài chính. Tỷ lệ nợ xấu (NPL — Non-Performing Loan) Việt Nam cuối 2023 là 4,55% [1]. Khủng hoảng tài chính 2008 có nguồn gốc từ định giá sai rủi ro tín dụng trên thị trường thế chấp Mỹ.

Mô hình tính điểm tín dụng (credit scoring) truyền thống (FICO Score, 1956) giải quyết vấn đề quy mô nhưng bỏ qua tương tác phi tuyến. Học máy + kỹ thuật SHAP (SHapley Additive exPlanations) giải quyết đồng thời hai yêu cầu: hiệu suất dự báo cao và giải thích được từng quyết định — điều kiện cần trong quản trị rủi ro tín dụng theo Basel III Pillar 3.

## 1.2 Mục tiêu và kết quả

| Mục tiêu                         | Chỉ tiêu                        | Kết quả                 |
| -------------------------------- | ------------------------------- | ----------------------- |
| Xây dựng mô hình phân loại       | AUC-ROC ≥ 0,87                  | **0,8714** ✓            |
| So sánh 4 thuật toán             | Bảng đầy đủ                     | ✓ (Bảng 4.1)            |
| Trích xuất đặc trưng có cơ sở     | ≥ 2 đặc trưng mới vào top-5 SHAP | **2/3** ✓               |
| Tối ưu ngưỡng theo F2-score      | F2 tối ưu, cost estimate        | t=0,625, Recall=66,9% ✓ |
| Sản phẩm demo                    | App Streamlit + SHAP            | ✓                       |

---

# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

## 2.1 Bài toán và Metrics

Bài toán phân loại nhị phân: học $f: \mathbb{R}^p \to [0,1]$ ước lượng $P(y=1|\mathbf{x})$. Quyết định tại ngưỡng $t$: $\hat{y} = \mathbb{1}[f(\mathbf{x}) \geq t]$.

**Mất cân bằng lớp 1:14** → Accuracy vô nghĩa (mô hình "đoán tất cả 0" đạt 93,3% nhưng Recall=0). Metrics phù hợp: AUC-ROC, F-beta, AUC-PR.

$$F_\beta = (1+\beta^2) \cdot \frac{P \cdot R}{\beta^2 P + R}$$

Với $\beta=2$: Recall ưu tiên gấp 4 lần Precision — phù hợp tín dụng vì FN (bỏ sót vỡ nợ) tốn kém hơn FP (từ chối nhầm) ~22 lần.

## 2.2 Bốn Mô hình

**Logistic Regression** — tối thiểu hóa negative log-likelihood (binary cross-entropy):
$$\mathcal{L}(\boldsymbol{\beta}) = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \ln \sigma(\mathbf{x}_i^\top \boldsymbol{\beta}) + (1-y_i) \ln(1-\sigma(\mathbf{x}_i^\top \boldsymbol{\beta})) \right]$$

Hệ số $e^{\beta_j}$ là odds ratio — tăng $x_j$ lên 1 đơn vị nhân odds vỡ nợ lên $e^{\beta_j}$ lần.

**Decision Tree CART** — phân chia đệ quy bằng Gini impurity: $G(t) = 2p_t(1-p_t)$. Kiểm soát quá khớp bằng `max_depth` và `min_samples_leaf`.

**Random Forest** — bagging $B$ cây với lấy mẫu đặc trưng ngẫu nhiên ($\sqrt{p}$ đặc trưng/phân chia). Phương sai giảm theo: $\text{Var} \to \rho\sigma^2$ khi $B\to\infty$, trong đó $\rho$ là tương quan cặp cây.

**XGBoost** — mô hình additive $F_T(\mathbf{x}) = \sum_{t=1}^T f_t(\mathbf{x})$, tối ưu hàm mục tiêu xấp xỉ Taylor bậc hai:
$$\mathcal{O}^{(t)} \approx \sum_{i=1}^{N} \left[g_i f_t(\mathbf{x}_i) + \frac{1}{2}h_i f_t(\mathbf{x}_i)^2\right] + \gamma T + \frac{\lambda}{2}\sum_j w_j^2$$

Trọng số lá tối ưu dạng closed-form: $w_j^* = -G_j/(H_j+\lambda)$.

**SHAP** — Shapley value đo đóng góp cận biên của đặc trưng $j$:
$$\phi_j(\mathbf{x}) = \sum_{S \subseteq F\setminus\{j\}} \frac{|S|!(|F|-|S|-1)!}{|F|!}\left[v(S\cup\{j\}) - v(S)\right]$$

Tính chất cộng: $f(\mathbf{x}) = \phi_0 + \sum_j \phi_j(\mathbf{x})$ — mỗi dự đoán được giải thích đầy đủ.

---

# CHƯƠNG 3: DỮ LIỆU VÀ TIỀN XỬ LÝ

## 3.1 Bộ dữ liệu

**Give Me Some Credit** (Kaggle, 2011) — 149.999 hồ sơ vay tiêu dùng Mỹ, 10 đặc trưng tài chính, biến mục tiêu `SeriousDlqin2yrs` (trễ hạn ≥ 90 ngày trong 2 năm tiếp theo).

**Tỷ lệ dương: 6,68%** — mất cân bằng 1:14.

![Phân phối biến mục tiêu — 6,68% vỡ nợ so với 93,32% không vỡ nợ](../reports/fig_01_target_distribution.png)
*Hình 3.1: Mất cân bằng lớp 1:14 buộc phải dùng AUC-ROC và F-beta thay vì Accuracy — mô hình "đoán tất cả 0" đạt Accuracy=93,3% nhưng hoàn toàn vô dụng (Recall=0).*

**Vấn đề chất lượng dữ liệu:** `age=0` (xóa), `MonthlyIncome` max=$3M (cap tại P99=$25.000), delinquency counts max=98 (cap tại ngưỡng thực tế). `MonthlyIncome` có **19,8% missing**, `NumberOfDependents` có 2,6% missing.

## 3.2 Xử lý Missing Values

Chi-squared test bác bỏ giả thuyết `MonthlyIncome` missing độc lập với target ($\chi^2=67,89, p\approx0$) → dữ liệu **MAR/MNAR**. Median imputation sẽ ước tính quá cao thu nhập của nhóm missing.

**Giải pháp: KNN Imputer** (k=5, khoảng cách Euclidean):
$$\hat{x}_{i,j} = \frac{\sum_{k \in \mathcal{N}(i)} w_k x_{k,j}}{\sum_{k \in \mathcal{N}(i)} w_k}, \quad w_k = 1/d(\mathbf{x}_i^{-j}, \mathbf{x}_k^{-j})$$

![So sánh phân phối MonthlyIncome sau KNN Imputer và Median Imputation](../reports/fig_11_imputation_comparison.png)
*Hình 3.2: KNN Imputer tạo phân phối đa dạng (std=$1.157) phản ánh thực tế — Median đặt tất cả giá trị thiếu tại $5.400, xóa sạch tín hiệu tương quan giữa thu nhập và các đặc trưng khác.*

## 3.3 Trích xuất Đặc trưng

Bốn đặc trưng mới được tạo dựa trên lĩnh vực tài chính:

| Đặc trưng mới | Công thức | Cơ sở |
|-------------|-----------|-------|
| `TotalDelinquencyScore` | $3 \cdot N_{90+} + 2 \cdot N_{60-89} + 1 \cdot N_{30-59}$ | FICO weighted delinquency tiers |
| `FinancialStressIndex` | RevUtil × TotalDelinquencyScore | Synergy stress ngắn hạn × lịch sử |
| `DebtToIncomeRatio` | DebtRatio × MonthlyIncome | DTI tuyệt đối (không phải tỷ lệ) |
| `DelinquencyTrend` | $N_{30-59} - N_{90+}$ | Âm = đang xấu dần |

**Kết quả:** FinancialStressIndex (Spearman ρ=0,346) và TotalDelinquencyScore (ρ=0,345) dẫn đầu tương quan với biến mục tiêu — vượt tất cả đặc trưng thô gốc.

## 3.4 Xử lý Mất Cân Bằng

SMOTE (Synthetic Minority Oversampling) nội suy giữa các điểm lớp thiểu số:
$$\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_{nn} - \mathbf{x}_i), \quad \lambda \sim \mathcal{U}[0,1]$$

**Lưu ý:** Chỉ áp dụng trên tập huấn luyện — áp dụng trước phân chia là rò rỉ dữ liệu.

**Quyết định cuối:** `class_weight='balanced'` (LR, DT, RF) và `scale_pos_weight=13,96` (XGBoost) — đơn giản hơn SMOTE, kết quả tương đương.

## 3.5 Phân chia Dữ liệu

**Phân chia phân tầng 70/15/15:** Huấn luyện 104.999, Kiểm định 22.500, Kiểm tra 22.500 — mỗi tập giữ tỷ lệ 6,68%.

---

# CHƯƠNG 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ

## 4.1 Thiết lập

- **Kiểm định chéo:** Phân tầng 5 lớp
- **Điều chỉnh siêu tham số:** RandomizedSearchCV tối ưu AUC-ROC
- **Pipeline:** sklearn.Pipeline bọc tiền xử lý để tránh rò rỉ dữ liệu
- **Môi trường:** Python 3.14.2, scikit-learn 1.8.0, XGBoost 3.2.0, SHAP 0.51.0

## 4.2 Kết quả từng mô hình

**Logistic Regression** (penalty=L1, C=0,001): AUC=0,8432, F1=0,4340, Recall=0,4927, train 488s.
- L1 triệt tiêu `NumberOfTimes90DaysLate` → xác nhận `TotalDelinquencyScore` đã tổng hợp thông tin của nó.
- Odds ratio đáng chú ý: TotalDelinquencyScore ($e^\beta=1,291$), age ($e^\beta=0,811$).

**Decision Tree** (max_depth=10, min_samples_leaf=200): AUC=0,8579, train 13s.
- Node gốc phân chia tại TotalDelinquencyScore ≤ 2,5 — nhất quán với SHAP.

**Random Forest** (n_estimators=200, max_features=0,3): AUC=0,8703, OOB=0,8286, train 511s.

**XGBoost** (n_estimators=200, max_depth=4, lr=0,05, scale_pos_weight=13,96): AUC=**0,8714**, F1=0,4466, train 127s.

![Confusion matrix, ROC curve, Precision-Recall curve và learning curve của XGBoost](../reports/fig_19_xgb_analysis.png)
*Hình 4.1: XGBoost đạt AUC=0,8714 — ROC nằm xa đường ngẫu nhiên, PR curve thể hiện hiệu quả ở vùng precision cao, learning curve hội tụ với gap train-val=0,021.*

## 4.3 So sánh Tổng hợp

| Mô hình | AUC-ROC | F1 | Recall | Thời gian huấn luyện | Kích thước mô hình |
|---------|---------|-----|--------|------------|------------|
| Logistic Regression | 0,8432 | 0,4340 | 0,4927 | 488s | ~50 KB |
| Decision Tree | 0,8579 | 0,4314 | 0,4694 | **13s** | ~200 KB |
| Random Forest | 0,8703 | 0,4439 | 0,5206 | 511s | ~12 MB |
| **XGBoost** | **0,8714** | **0,4466** | 0,5160 | 127s | **340 KB** |

![Overlay ROC curves của 4 mô hình trên tập kiểm tra](../reports/fig_20_model_comparison.png)
*Hình 4.2: XGBoost và Random Forest gần như trùng nhau về AUC (gap=0,0011) — nhưng XGBoost có lợi thế rõ ràng về tốc độ (127s vs 511s) và kích thước (340KB vs 12MB).*

**XGBoost đạt AUC=0,8714**, vượt mục tiêu 0,87. So với Random Forest (AUC=0,8703), XGBoost nhanh hơn 4× (127s vs 511s) và nhỏ hơn 35× (340KB vs 12MB).

## 4.4 SHAP Analysis

**Global importance (mean |SHAP|):**

| Hạng | Đặc trưng | Loại | Mean\|SHAP\| |
|------|---------|------|-------------|
| 1 | FinancialStressIndex | **Được tạo** | **0,577** |
| 2 | RevolvingUtilizationOfUnsecuredLines | Gốc | 0,535 |
| 3 | TotalDelinquencyScore | **Được tạo** | 0,410 |
| 4 | age | Gốc | 0,244 |
| 5 | NumberOfOpenCreditLinesAndLoans | Gốc | 0,168 |
| 12 | NumberOfTimes90DaysLate | Gốc | 0,013 |

![Biểu đồ thanh SHAP tổng quát — mean |SHAP value| của 14 đặc trưng](../reports/fig_26a_shap_bar.png)
*Hình 4.3: 2/3 đặc trưng hàng đầu là do trích xuất đặc trưng tạo ra — FinancialStressIndex vượt cả thành phần gốc, xác nhận thiên kiến quy nạp từ lĩnh vực tài chính có giá trị với mô hình phi tuyến mạnh.*

**SHAP base value:** 0,0148 (log-odds), tương ứng xác suất cơ sở 50,37% — hệ quả của `scale_pos_weight` điều chỉnh Bayesian prior.

## 4.5 Phân tích Sai số

**Cấu trúc lỗi tại ngưỡng=0,77:**

| Category | Số lượng | % |
|----------|---------|---|
| **TP** (dự báo đúng vỡ nợ) | 776 | 51,6% tổng số vỡ nợ thực |
| **FN** (bỏ sót vỡ nợ) | 728 | **48,4% tổng số vỡ nợ thực** |
| **FP** (từ chối nhầm) | 1.195 | 5,7% tổng số không vỡ nợ |
| **TN** (duyệt đúng) | 19.801 | 94,3% tổng số không vỡ nợ |

**Profile FN (người vỡ nợ bị bỏ sót):** TotalDelinquencyScore median=0, MonthlyIncome=$4.200 (cao hơn TP). Đây là người vỡ nợ không có dấu hiệu cảnh báo trong lịch sử — giới hạn cấu trúc của các đặc trưng nhìn lại quá khứ.

**Profile FP (khách hàng tốt bị từ chối nhầm):** 1.195 trường hợp với RevolvingUtilization median=0,958 (gấp 8,1× nhóm TN), TotalDelinquencyScore=4,0, FinancialStressIndex=2,856 — trông giống defaulter nhưng thực tế vẫn trả được nợ. Đây là nhóm **"thin-file customers"**: khách hàng dùng thẻ tín dụng tích cực, thu nhập thấp hơn TN (median $3.610 vs $4.505) nhưng có kỷ luật tài chính không thể hiện trong credit history. Chi phí cơ hội: 1.195 × $500 = **$597.500** doanh thu bị bỏ lỡ → giải pháp: **alternative data** (mục 6.3 báo cáo chính).

![Histogram predicted score phân theo nhóm lỗi](../reports/fig_24_score_distribution.png)
*Hình 4.4: FN tập trung ở vùng score thấp (0,2–0,5) — người vỡ nợ "trông lành mạnh", phản ánh giới hạn căn bản của các đặc trưng lịch sử, không phải lỗi mô hình.*

## 4.6 Tối ưu Ngưỡng

Tìm ngưỡng tối ưu F2 trên tập kiểm định:
$$t^* = \arg\max_{t} F_2(t) = 0,625$$

**Kết quả trên tập kiểm tra:**

| Ngưỡng | F2 | Recall | Chi phí ước tính |
|-----------|----|----|-----------|
| 0,50 | 0,518 | 0,775 | $5,84M |
| **0,625** (F2 opt) | **0,537** | **0,669** | **$6,78M** |
| 0,775 (F1 opt) | 0,480 | 0,507 | $8,79M |

**Business cost:** FN=$11.250/case (loan $15.000 × LGD 75%), FP=$500/case. Ngưỡng t=0,5 có tổng chi phí thấp nhất ($5,84M) nhưng tạo ~4.000 FP — khó vận hành. Ngưỡng F2 (0,625) không phải ngưỡng tối ưu chi phí mà là điểm thỏa hiệp ưu tiên Recall: giảm $2M so với F1-optimal (0,775) với FP ở mức kiểm soát được (~2.350).

![F1, F2, Precision và Recall theo ngưỡng — điểm tối ưu F2 tại t=0,625](../reports/fig_25_threshold_optimization.png)
*Hình 4.5: t=0,625 là điểm thỏa hiệp giữa t=0,5 (ít FN nhất nhưng ~4.000 FP) và t=0,775 (ít FP nhưng bỏ sót nhiều defaults) — F2 được tối đa hóa phản ánh chi phí FN:FP=22:1.*

## 4.7 Learning Curve

| Training size | Train AUC | Val AUC | Gap |
|--------------|-----------|---------|-----|
| 10% (7K) | 0,9596 | 0,8427 | 0,117 |
| 65% (45K) | 0,8926 | 0,8624 | 0,030 |
| 100% (70K) | 0,8847 | 0,8642 | **0,021** |

Gap cuối 0,021 < 0,03 → **"Khớp tốt"**. Val AUC không còn tăng đáng kể sau N≈45.000 — giới hạn là chất lượng đặc trưng, không phải số lượng dữ liệu.

## 4.8 Ba Tranh luận Lớn

**1. Khả năng giải thích so với Hiệu suất:** Logistic Regression dễ giải thích nhất nhưng AUC thấp hơn 2,8 điểm. Lựa chọn XGBoost bù đắp phần khả năng giải thích bằng SHAP — thay vì nói "mô hình từ chối", nhân viên tín dụng có thể trình bày: "TotalDelinquencyScore=12 đóng góp +0,45 vào xác suất vỡ nợ." Mức này đủ để giải thích quyết định cho khách hàng và phù hợp với yêu cầu trong quản trị rủi ro tín dụng.

**2. Tập trung Dữ liệu so với Tập trung Mô hình:** XGBoost với 10 đặc trưng gốc → AUC≈0,855; thêm 4 đặc trưng được tạo → AUC=0,8714. Phần cải thiện không lớn về AUC, nhưng 2 trong 3 top-SHAP là đặc trưng tự tạo — thiên kiến quy nạp từ lĩnh vực tài chính vẫn có tác dụng ngay với mô hình phi tuyến mạnh.

**3. Accuracy vs F1/AUC:** "Đoán tất cả 0" đạt Accuracy 93,3% nhưng Recall=0. AUC-ROC và F-beta là metrics phù hợp hơn với dữ liệu mất cân bằng. Ngưỡng mặc định 0,5 không tối ưu: cả 4 mô hình đều cần ngưỡng > 0,5 (từ 0,62 đến 0,80) do class_weight dịch chuyển prior của mô hình.

---

# CHƯƠNG 5: SẢN PHẨM — STREAMLIT DASHBOARD

## 5.1 Kiến trúc và Luồng xử lý

**Đầu vào:** 10 đặc trưng thô → Tự tính 4 đặc trưng được tạo → DataFrame 14 đặc trưng → XGBoost `predict_proba` → SHAP TreeExplainer → Đầu ra.

**Caching:** `@st.cache_resource` cho mô hình và trình giải thích (load một lần). SHAP waterfall tạo mới mỗi dự đoán.

## 5.2 Risk Tier và Quyết định

| P(default) | Risk Tier | Quyết định |
|-----------|-----------|------------|
| < 10% | LOW RISK | APPROVE |
| 10–30% | MEDIUM RISK | APPROVE |
| 30–62,5% | HIGH RISK | APPROVE |
| ≥ 62,5% | VERY HIGH RISK | REJECT |

Ngưỡng 62,5% căn chỉnh với t=0,625 từ tối ưu F2.

![SHAP waterfall cho 3 dự đoán cụ thể — khách hàng thấp/cao/biên](../reports/fig_28_shap_waterfall.png)
*Hình 5.1: Dashboard giải thích từng quyết định: (trái) P=6,7% — an toàn, age và income kéo xuống; (giữa) P=97,8% — rủi ro cao, delinquency đẩy mạnh; (phải) P=41,5% — biên, cần xem xét thêm. Mỗi chart là cơ sở có thể trình bày khi từ chối hồ sơ vay.*

## 5.3 Chạy ứng dụng

```bash
streamlit run app/app.py
```

---

# CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 6.1 Kết quả đạt được

- **XGBoost AUC=0,8714** — vượt mục tiêu 0,87, best trong 4 mô hình
- **Trích xuất đặc trưng đã xác nhận:** FinancialStressIndex (#1 SHAP=0,577), TotalDelinquencyScore (#3 SHAP=0,410) — 2/3 đặc trưng hàng đầu
- **Ngưỡng F2-tối ưu t=0,625:** Recall=66,9% (+15,3pp so với F1-opt), giảm chi phí $2M so với ngưỡng F1 (t=0,775)
- **Sản phẩm:** Streamlit dashboard SHAP-explainable, chạy local và deploy được

## 6.2 Hạn chế chính

1. **Đặc trưng nhìn lại quá khứ:** 48,4% FN là người vỡ nợ không có dấu hiệu cảnh báo trong lịch sử — hạn chế cấu trúc của dữ liệu tín dụng lịch sử
2. **Dữ liệu tĩnh:** Không có chuỗi thời gian → chỉ ảnh chụp tại một thời điểm, không nắm xu hướng
3. **Giới hạn địa lý:** Thị trường Mỹ — cần huấn luyện lại và điều chỉnh ngưỡng cho Việt Nam
4. **Calibration:** AUC cao nhưng probability output chưa được calibrate (Platt scaling có thể cải thiện)

## 6.3 Hướng phát triển

**Gần hạn:** Dữ liệu time series → LSTM/GRU; ensemble stacking (+0,003–0,005 AUC ước tính).

**Trung hạn:** Alternative data (mobile payment, utility bills) cho thin-file customers; fairness audit theo Equal Credit Opportunity Act.

**Dài hạn:** Graph neural networks (mạng lưới bảo lãnh); causal inference; federated learning đa ngân hàng.

---

# TÀI LIỆU THAM KHẢO

[1] Ngân hàng Nhà nước Việt Nam, "Báo cáo thường niên 2023," Hà Nội, 2024.

[2] E. I. Altman, "Financial ratios, discriminant analysis and the prediction of corporate bankruptcy," *Journal of Finance*, vol. 23, no. 4, pp. 589–609, 1968.

[3] D. J. Hand and W. E. Henley, "Statistical classification methods in consumer credit scoring," *J. Royal Statistical Society: Series A*, vol. 160, no. 3, pp. 523–541, 1997.

[4] S. Lessmann et al., "Benchmarking state-of-the-art classification algorithms for credit scoring," *European Journal of Operational Research*, vol. 247, no. 1, pp. 124–136, 2015.

[5] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," *NIPS*, vol. 30, 2017.

[6] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *ACM SIGKDD*, 2016.

[7] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[8] N. V. Chawla et al., "SMOTE: Synthetic minority over-sampling technique," *JAIR*, vol. 16, pp. 321–357, 2002.

[9] M. Bucker et al., "Transparency, auditability, and explainability of ML models in credit decisions," *J. Operational Research Society*, vol. 73, no. 1, pp. 70–90, 2022.

[10] Basel Committee on Banking Supervision, "Revisions to the Standardised Approach for credit risk," BIS, 2017.

---

> **Xem thêm:** Đầy đủ cơ sở lý thuyết (derivations, VIF analysis, hyperparameter search spaces, 30 biểu đồ EDA), code notebooks và phụ lục kỹ thuật tại `final_report/bao_cao_chinh.md`.
