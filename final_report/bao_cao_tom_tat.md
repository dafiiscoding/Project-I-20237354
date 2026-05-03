# DỰ BÁO RỦI RO VỠ NỢ TÍN DỤNG BẰNG MACHINE LEARNING
## Bản Tóm Tắt — Loan Default Prediction Using Machine Learning


> *Bản tóm tắt trình bày các kết quả cốt lõi. Cơ sở lý thuyết đầy đủ, biện giải toán học và phụ lục kỹ thuật nằm trong báo cáo chính (`final_report/bao_cao_chinh.md`).*

---

## TÓM TẮT

Nghiên cứu ứng dụng Machine Learning vào dự báo rủi ro vỡ nợ tín dụng, sử dụng bộ dữ liệu *Give Me Some Credit* (Kaggle, 150.000 hồ sơ vay). Bốn mô hình được xây dựng theo thứ tự tăng dần độ phức tạp: Logistic Regression (AUC=0,8432), Decision Tree (AUC=0,8579), Random Forest (AUC=0,8703), XGBoost (AUC=0,8714). Feature Engineering có chủ đích từ lĩnh vực tài chính tạo ra hai đặc trưng xếp trong nhóm 3 đặc trưng quan trọng nhất theo SHAP. Ngưỡng tối ưu theo hệ số $F_2$ ($t=0,625$) tăng Recall từ 51,6% lên 66,9%, giảm chi phí ước tính 2 triệu USD so với ngưỡng tối ưu theo hệ số $F_1$ ($t=0,775$). Sản phẩm là ứng dụng Streamlit giải thích từng quyết định tín dụng bằng biểu đồ thác nước SHAP.

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

Mô hình tính điểm tín dụng (credit scoring) truyền thống (FICO Score, 1956) giải quyết vấn đề quy mô nhưng bỏ qua tương tác phi tuyến. Machine Learning + kỹ thuật SHAP (SHapley Additive exPlanations) giải quyết đồng thời hai yêu cầu: hiệu suất dự báo cao và giải thích được từng quyết định — điều kiện cần trong quản trị rủi ro tín dụng theo Basel III Pillar 3.

## 1.2 Mục tiêu và kết quả

| Mục tiêu                         | Chỉ tiêu                        | Kết quả                 |
| -------------------------------- | ------------------------------- | ----------------------- |
| Xây dựng mô hình Classification  | AUC-ROC ≥ 0,87                  | **0,8714** ✓            |
| So sánh 4 thuật toán             | Bảng chỉ số đầy đủ              | ✓ (Bảng 4.1)            |
| Feature Engineering có cơ sở     | ≥ 2 đặc trưng mới vào top-5 SHAP | **2/3** ✓               |
| Tối ưu ngưỡng theo hệ số $F_2$   | $F_2$ tối ưu, ước tính chi phí  | $t=0,625$, Recall=66,9% ✓ |
| Sản phẩm thực nghiệm             | Ứng dụng Streamlit + SHAP       | ✓                       |

---

# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

## 2.1 Bài toán và Các chỉ số đánh giá

Bài toán Binary Classification: học $f: \mathbb{R}^p \to [0,1]$ ước lượng $P(y=1|\mathbf{x})$. Quyết định tại ngưỡng $t$: $\hat{y} = \mathbb{1}[f(\mathbf{x}) \geq t]$.

**Class Imbalance 1:14** → Accuracy không phản ánh đúng bản chất (mô hình "đoán tất cả 0" đạt 93,3% nhưng Recall=0). Các chỉ số phù hợp: AUC-ROC, $F_\beta$-score, AUC-PR.

$$F_\beta = (1+\beta^2) \cdot \frac{P \cdot R}{\beta^2 P + R}$$

Với $\beta=2$: Recall được ưu tiên gấp 4 lần so với Precision — phù hợp với lĩnh vực tín dụng vì sai lầm loại II (bỏ sót vỡ nợ) tốn kém hơn sai lầm loại I (từ chối nhầm) khoảng 22 lần.

## 2.2 Bốn Mô hình

**Logistic Regression** — tối thiểu hóa binary cross-entropy loss:
$$\mathcal{L}(\boldsymbol{\beta}) = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \ln \sigma(\mathbf{x}_i^\top \boldsymbol{\beta}) + (1-y_i) \ln(1-\sigma(\mathbf{x}_i^\top \boldsymbol{\beta})) \right]$$

Hệ số $e^{\beta_j}$ là odds ratio — khi đặc trưng $x_j$ tăng thêm 1 đơn vị, odds vỡ nợ thay đổi $e^{\beta_j}$ lần.

**Decision Tree CART** — phân chia đệ quy dựa trên Gini impurity: $G(t) = 2p_t(1-p_t)$. Kiểm soát overfitting qua `max_depth` và `min_samples_leaf`.

**Random Forest** — sử dụng bagging với $B$ cây và lấy mẫu đặc trưng ngẫu nhiên ($\sqrt{p}$ đặc trưng mỗi lần phân chia). Phương sai giảm: $\text{Var} \to \rho\sigma^2$ khi $B\to\infty$, trong đó $\rho$ là tương quan giữa các cặp cây.

**XGBoost** — additive model $F_T(\mathbf{x}) = \sum_{t=1}^T f_t(\mathbf{x})$, tối ưu hàm mục tiêu thông qua xấp xỉ Taylor bậc hai:
$$\mathcal{O}^{(t)} \approx \sum_{i=1}^{N} \left[g_i f_t(\mathbf{x}_i) + \frac{1}{2}h_i f_t(\mathbf{x}_i)^2\right] + \gamma T + \frac{\lambda}{2}\sum_j w_j^2$$

Trọng số lá tối ưu có nghiệm dạng đóng (closed-form): $w_j^* = -G_j/(H_j+\lambda)$.

**SHAP** — Shapley value đo đóng góp cận biên của đặc trưng $j$:
$$\phi_j(\mathbf{x}) = \sum_{S \subseteq F\setminus\{j\}} \frac{|S|!(|F|-|S|-1)!}{|F|!}\left[v(S\cup\{j\}) - v(S)\right]$$

Tính chất cộng: $f(\mathbf{x}) = \phi_0 + \sum_j \phi_j(\mathbf{x})$ — mỗi dự đoán được giải thích đầy đủ.

---

# CHƯƠNG 3: DỮ LIỆU VÀ TIỀN XỬ LÝ

## 3.1 Bộ dữ liệu

**Give Me Some Credit** (Kaggle, 2011) — 149.999 hồ sơ vay tiêu dùng Mỹ, 10 đặc trưng tài chính, biến mục tiêu `SeriousDlqin2yrs` (trễ hạn ≥ 90 ngày trong 2 năm tiếp theo).

**Tỷ lệ dương: 6,68%** — mất cân bằng 1:14.

![Phân phối biến mục tiêu — 6,68% vỡ nợ so với 93,32% không vỡ nợ](../reports/fig_01_target_distribution.png)
*Hình 3.1: Mất cân bằng lớp 1:14 buộc phải sử dụng AUC-ROC và hệ số $F_\beta$ thay vì Accuracy — mô hình "đoán tất cả 0" đạt Accuracy=93,3% nhưng hoàn toàn vô dụng (Recall=0).*

**Vấn đề chất lượng dữ liệu:** `age=0` (xóa), `MonthlyIncome` cao nhất 3 triệu USD (giới hạn tại bách phân vị P99 ở mức 25.000 USD), các chỉ số trễ hạn cao nhất 98 (giới hạn tại ngưỡng thực tế). Biến `MonthlyIncome` có **19,8% giá trị khuyết thiếu**, `NumberOfDependents` có 2,6% giá trị khuyết thiếu.

## 3.2 Xử lý Missing Values

Kiểm định Chi bình phương (Chi-squared test) bác bỏ giả thuyết dữ liệu `MonthlyIncome` khuyết thiếu độc lập với biến mục tiêu ($\chi^2=67,89, p\approx0$) → dữ liệu thuộc dạng **MAR/MNAR**. Phương pháp thay thế bằng trung vị (Median imputation) sẽ ước tính quá cao thu nhập của nhóm khuyết thiếu.

**Giải pháp: Thuật toán KNN Imputer** (k=5, khoảng cách Euclidean):
$$\hat{x}_{i,j} = \frac{\sum_{k \in \mathcal{N}(i)} w_k x_{k,j}}{\sum_{k \in \mathcal{N}(i)} w_k}, \quad w_k = 1/d(\mathbf{x}_i^{-j}, \mathbf{x}_k^{-j})$$

![So sánh phân phối MonthlyIncome sau khi sử dụng KNN Imputer và Median Imputation](../reports/fig_11_imputation_comparison.png)
*Hình 3.2: KNN Imputer tạo ra phân phối đa dạng (độ lệch chuẩn 1.157 USD) phản ánh đúng thực tế — Trong khi thay thế bằng trung vị (Median) đặt tất cả giá trị thiếu tại mức 5.400 USD, làm mất đi tín hiệu tương quan giữa thu nhập và các đặc trưng khác.*

## 3.3 Feature Engineering

Bốn đặc trưng mới được tạo dựa trên lĩnh vực tài chính:

| Đặc trưng mới | Công thức | Cơ sở |
|-------------|-----------|-------|
| `TotalDelinquencyScore` | $3 \cdot N_{90+} + 2 \cdot N_{60-89} + 1 \cdot N_{30-59}$ | Phân lớp trễ hạn có trọng số theo FICO |
| `FinancialStressIndex` | RevUtil × TotalDelinquencyScore | Sự cộng hưởng giữa áp lực ngắn hạn và lịch sử trả nợ |
| `DebtToIncomeRatio` | DebtRatio × MonthlyIncome | Tổng nợ tuyệt đối (không phải tỷ số) |
| `DelinquencyTrend` | $N_{30-59} - N_{90+}$ | Giá trị âm thể hiện xu hướng đang xấu dần |

**Kết quả:** FinancialStressIndex (Spearman ρ=0,346) và TotalDelinquencyScore (ρ=0,345) dẫn đầu tương quan với biến mục tiêu — vượt tất cả đặc trưng thô gốc.

## 3.4 Xử lý Class Imbalance

SMOTE (Synthetic Minority Oversampling Technique) thực hiện nội suy giữa các điểm thuộc lớp thiểu số:
$$\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_{nn} - \mathbf{x}_i), \quad \lambda \sim \mathcal{U}[0,1]$$

**Lưu ý:** Chỉ áp dụng trên tập huấn luyện — áp dụng trước phân chia là rò rỉ dữ liệu.

**Quyết định cuối:** `class_weight='balanced'` (LR, DT, RF) và `scale_pos_weight=13,96` (XGBoost) — đơn giản hơn SMOTE, kết quả tương đương.

## 3.5 Phân chia Dữ liệu

**Phân chia phân tầng 70/15/15:** Huấn luyện 104.999, Kiểm định 22.500, Kiểm tra 22.500 — mỗi tập giữ tỷ lệ 6,68%.

---

# CHƯƠNG 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ

## 4.1 Thiết lập

- **Cross-validation:** Stratified 5-fold
- **Điều chỉnh siêu tham số:** RandomizedSearchCV tối ưu AUC-ROC
- **Pipeline:** sklearn.Pipeline bọc tiền xử lý để tránh rò rỉ dữ liệu
- **Môi trường:** Python 3.14.2, scikit-learn 1.8.0, XGBoost 3.2.0, SHAP 0.51.0

## 4.2 Kết quả từng mô hình

**Logistic Regression** (penalty=L1, C=0,001): AUC=0,8432, F1=0,4340, Recall=0,4927, thời gian huấn luyện 488 giây.
- Hình phạt L1 triệt tiêu biến `NumberOfTimes90DaysLate` → xác nhận `TotalDelinquencyScore` đã mã hóa đủ thông tin của biến này.
- Tỷ số chênh (Odds ratio) đáng chú ý: TotalDelinquencyScore ($e^\beta=1,291$), age ($e^\beta=0,811$).

**Decision Tree** (max_depth=10, min_samples_leaf=200): AUC=0,8579, huấn luyện 13 giây.
- Nút gốc phân chia tại TotalDelinquencyScore ≤ 2,5 — nhất quán với kết quả từ SHAP.

**Random Forest** (n_estimators=200, max_features=0,3): AUC=0,8703, OOB=0,8286, train 511s.

**XGBoost** (n_estimators=200, max_depth=4, lr=0,05, scale_pos_weight=13,96): AUC=**0,8714**, F1=0,4466, train 127s.

![Confusion matrix, ROC curve, Precision-Recall curve và learning curve của XGBoost](../reports/fig_19_xgb_analysis.png)
*Hình 4.1: XGBoost đạt AUC=0,8714 — ROC curve nằm xa đường ngẫu nhiên, PR curve thể hiện hiệu quả ở vùng precision cao, learning curve hội tụ với train/val gap bằng 0,021.*

## 4.3 So sánh Tổng hợp

| Mô hình | AUC-ROC | F1 | Recall | Thời gian huấn luyện | Kích thước mô hình |
|---------|---------|-----|--------|------------|------------|
| Logistic Regression | 0,8432 | 0,4340 | 0,4927 | 488s | ~50 KB |
| Decision Tree | 0,8579 | 0,4314 | 0,4694 | **13s** | ~200 KB |
| Random Forest | 0,8703 | 0,4439 | 0,5206 | 511s | ~12 MB |
| **XGBoost** | **0,8714** | **0,4466** | 0,5160 | 127s | **340 KB** |

![So sánh các đường cong ROC của 4 mô hình trên tập kiểm tra](../reports/fig_20_model_comparison.png)
*Hình 4.2: XGBoost và Random Forest gần như tương đương về AUC (chênh lệch 0,0011) — tuy nhiên XGBoost chiếm ưu thế rõ rệt về tốc độ (127 giây so với 511 giây) và dung lượng mô hình (340KB so với 12MB).*

**XGBoost đạt AUC=0,8714**, vượt mục tiêu 0,87. So với Random Forest (AUC=0,8703), XGBoost nhanh hơn 4× (127s vs 511s) và nhỏ hơn 35× (340KB vs 12MB).

## 4.4 SHAP Analysis

**Độ quan trọng toàn cục (trung bình trị tuyệt đối giá trị SHAP):**

| Hạng | Đặc trưng | Loại | Mean\|SHAP\| |
|------|---------|------|-------------|
| 1 | FinancialStressIndex | **Được tạo** | **0,577** |
| 2 | RevolvingUtilizationOfUnsecuredLines | Gốc | 0,535 |
| 3 | TotalDelinquencyScore | **Được tạo** | 0,410 |
| 4 | age | Gốc | 0,244 |
| 5 | NumberOfOpenCreditLinesAndLoans | Gốc | 0,168 |
| 12 | NumberOfTimes90DaysLate | Gốc | 0,013 |

![Biểu đồ thanh SHAP tổng quát — mean |SHAP value| của 14 đặc trưng](../reports/fig_26a_shap_bar.png)
*Hình 4.3: 2 trong top-3 đặc trưng hàng đầu là do Feature Engineering tạo ra — FinancialStressIndex vượt cả thành phần gốc, xác nhận inductive bias từ lĩnh vực tài chính có giá trị với mô hình phi tuyến mạnh.*

**Giá trị cơ sở SHAP (base value):** 0,0148 (logarit tỷ số chênh), tương ứng xác suất cơ sở 50,37% — hệ quả của tham số `scale_pos_weight` dùng để điều chỉnh Bayesian prior.

## 4.5 Phân tích Sai số

**Cấu trúc sai số tại ngưỡng t=0,77:**

| Phân loại | Số lượng | Tỷ lệ |
|----------|---------|---|
| **TP** (Dự báo đúng vỡ nợ) | 776 | 51,6% tổng số vỡ nợ thực |
| **FN** (Bỏ sót vỡ nợ - Sai lầm loại II) | 728 | **48,4% tổng số vỡ nợ thực** |
| **FP** (Từ chối nhầm - Sai lầm loại I) | 1.195 | 5,7% tổng số không vỡ nợ |
| **TN** (Duyệt đúng) | 19.801 | 94,3% tổng số không vỡ nợ |

**Đặc điểm nhóm FN (người vỡ nợ bị bỏ sót):** Trung vị TDS bằng 0, thu nhập hàng tháng 4.200 USD (cao hơn nhóm TP). Đây là những trường hợp vỡ nợ không có dấu hiệu cảnh báo trong lịch sử — giới hạn cố hữu của các đặc trưng dựa trên dữ liệu quá khứ.

**Đặc điểm nhóm FP (khách hàng tốt bị từ chối nhầm):** 1.195 trường hợp với trung vị RevolvingUtilization đạt 0,958 (gấp 8,1 lần nhóm TN), TDS=4,0, FinancialStressIndex=2,856 — có đặc điểm tương đồng với người vỡ nợ nhưng thực tế vẫn hoàn trả đầy đủ. Đây là nhóm **"khách hàng hồ sơ mỏng" (thin-file customers)**: thường là khách hàng trẻ dùng thẻ tín dụng tích cực, thu nhập thấp hơn nhóm TN (trung vị 3.610 USD so với 4.505 USD) nhưng có kỷ luật tài chính tốt. Chi phí cơ hội: 1.195 × 500 USD = **597.500 USD** doanh thu bị bỏ lỡ → giải pháp: sử dụng **alternative data**.

![Histogram predicted score phân theo nhóm lỗi](../reports/fig_24_score_distribution.png)
*Hình 4.4: FN tập trung ở vùng score thấp (0,2–0,5) — người vỡ nợ "trông lành mạnh", phản ánh giới hạn căn bản của các đặc trưng lịch sử, không phải lỗi mô hình.*

## 4.6 Tối ưu Ngưỡng phân loại

Tìm ngưỡng tối ưu theo hệ số $F_2$ trên tập kiểm định:
$$t^* = \arg\max_{t} F_2(t) = 0,625$$

**Kết quả trên tập kiểm tra:**

| Ngưỡng | $F_2$ | Recall | Chi phí ước tính |
|-----------|----|----|-----------|
| 0,50 | 0,518 | 0,775 | 5,84 triệu USD |
| **0,625** ($F_2$ tối ưu) | **0,537** | **0,669** | **6,78 triệu USD** |
| 0,775 ($F_1$ tối ưu) | 0,480 | 0,507 | 8,79 triệu USD |

**Chi phí kinh doanh:** FN=11.250 USD/trường hợp, FP=500 USD/trường hợp. Ngưỡng t=0,5 có tổng chi phí thấp nhất (5,84 triệu USD) nhưng tạo ra khoảng 4.000 trường hợp FP — gây khó khăn cho vận hành. Ngưỡng tối ưu theo hệ số $F_2$ (0,625) là điểm thỏa hiệp ưu tiên Recall: giảm 2 triệu USD chi phí rủi ro so với ngưỡng tối ưu theo $F_1$ (0,775) trong khi vẫn giữ số lượng FP ở mức kiểm soát được (khoảng 2.350).

![F1, F2, Precision và Recall theo ngưỡng — điểm tối ưu F2 tại t=0,625](../reports/fig_25_threshold_optimization.png)
*Hình 4.5: t=0,625 là điểm thỏa hiệp giữa t=0,5 (ít FN nhất nhưng quá nhiều FP) và t=0,775 (ít FP nhưng bỏ sót nhiều nợ xấu) — Hệ số $F_2$ được tối đa hóa phản ánh tỷ lệ chi phí rủi ro giữa FN và FP là 22:1.*

## 4.7 Learning Curve

| Training size | Train AUC | Val AUC | Gap |
|--------------|-----------|---------|-----|
| 10% (7K) | 0,9596 | 0,8427 | 0,117 |
| 65% (45K) | 0,8926 | 0,8624 | 0,030 |
| 100% (70K) | 0,8847 | 0,8642 | **0,021** |

Train-val gap cuối = 0,021 < 0,03 → **mô hình không overfitting**. AUC trên tập kiểm định không tăng thêm đáng kể sau 45.000 hồ sơ — cho thấy nút thắt cổ chai nằm ở chất lượng đặc trưng, không phải số lượng dữ liệu.

## 4.8 Ba Tranh luận Lớn

**1. Khả năng giải thích so với Hiệu suất:** Logistic Regression dễ giải thích nhất nhưng AUC thấp hơn 2,8 điểm. Lựa chọn XGBoost bù đắp phần khả năng giải thích bằng SHAP — thay vì nói "mô hình từ chối", nhân viên tín dụng có thể trình bày: "TotalDelinquencyScore=12 đóng góp +0,45 vào xác suất vỡ nợ." Mức này đủ để giải thích quyết định cho khách hàng và phù hợp với yêu cầu trong quản trị rủi ro tín dụng.

**2. Tập trung Dữ liệu so với Tập trung Mô hình:** XGBoost với 10 đặc trưng gốc → AUC≈0,855; thêm 4 đặc trưng được tạo → AUC=0,8714. Phần cải thiện không lớn về AUC, nhưng 2 trong 3 top-SHAP là đặc trưng tự tạo — inductive bias từ lĩnh vực tài chính vẫn có tác dụng ngay với mô hình phi tuyến mạnh.

**3. Accuracy vs F1/AUC:** "Đoán tất cả 0" đạt Accuracy 93,3% nhưng Recall=0. AUC-ROC và F-beta là metrics phù hợp hơn với dữ liệu mất cân bằng. Ngưỡng mặc định 0,5 không tối ưu: cả 4 mô hình đều cần ngưỡng > 0,5 (từ 0,62 đến 0,80) do class_weight dịch chuyển prior của mô hình.

---

# CHƯƠNG 5: SẢN PHẨM — STREAMLIT DASHBOARD

## 5.1 Kiến trúc và Luồng xử lý

**Đầu vào:** 10 đặc trưng thô → Tự động tính 4 đặc trưng mới → Cấu trúc DataFrame 14 đặc trưng → Dự báo xác suất bằng XGBoost → Giải thích bằng SHAP TreeExplainer → Kết quả đầu ra.

**Caching:** Sử dụng `@st.cache_resource` để tải mô hình và explainer một lần duy nhất. SHAP waterfall chart được khởi tạo mới cho mỗi dự báo.

## 5.2 Risk Tier và Quyết định

| Xác suất vỡ nợ | Phân lớp rủi ro | Quyết định |
|-----------|-----------|------------|
| < 10% | RỦI RO THẤP | CHẤP THUẬN |
| 10–30% | RỦI RO TRUNG BÌNH | CHẤP THUẬN |
| 30–62,5% | RỦI RO CAO | CHẤP THUẬN |
| ≥ 62,5% | RỦI RO RẤT CAO | TỪ CHỐI |

Ngưỡng 62,5% được đồng bộ với ngưỡng tối ưu $t=0,625$ từ hệ số $F_2$.

![SHAP waterfall chart cho 3 trường hợp dự báo cụ thể](../reports/fig_28_shap_waterfall.png)
*Hình 5.1: Dashboard giải thích từng quyết định: (trái) P=6,7% — an toàn, tuổi và thu nhập giúp giảm rủi ro; (giữa) P=97,8% — rủi ro rất cao, các chỉ số trễ hạn đẩy mạnh xác suất; (phải) P=41,5% — trường hợp biên, cần xem xét thêm. Mỗi biểu đồ là cơ sở minh bạch khi từ chối hồ sơ vay.*

## 5.3 Chạy ứng dụng

```bash
streamlit run app/app.py
```

---

# CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 6.1 Kết quả đạt được

- **XGBoost AUC=0,8714** — vượt mục tiêu 0,87, là mô hình tốt nhất trong 4 thuật toán thử nghiệm.
- **Giá trị Feature Engineering:** FinancialStressIndex (Hạng 1 SHAP = 0,577), TotalDelinquencyScore (Hạng 3 SHAP = 0,410) chiếm vị trí then chốt.
- **Ngưỡng tối ưu theo $F_2$ ($t=0,625$):** Đạt Recall 66,9% (tăng 15,3 điểm phần trăm so với $F_1$), giảm chi phí rủi ro 2 triệu USD so với ngưỡng $F_1$ (t=0,775).
- **Sản phẩm:** Streamlit dashboard tích hợp giải thích SHAP, sẵn sàng chạy nội bộ hoặc triển khai.

## 6.2 Hạn chế chính

1. **Đặc trưng dựa trên lịch sử:** 48,4% FN là các trường hợp vỡ nợ không có dấu hiệu cảnh báo trước — một hạn chế mang tính cấu trúc của dữ liệu tín dụng quá khứ.
2. **Dữ liệu tĩnh:** Thiếu yếu tố chuỗi thời gian nên chỉ phản ánh trạng thái tại một thời điểm, chưa nắm bắt được xu hướng hành vi.
3. **Giới hạn địa lý:** Dữ liệu từ thị trường Mỹ — cần được huấn luyện lại và căn chỉnh ngưỡng phù hợp với thị trường Việt Nam.
4. **Calibration:** Chỉ số AUC cao nhưng xác suất đầu ra chưa được calibrate hoàn toàn (có thể cải thiện bằng Platt Scaling).

## 6.3 Hướng phát triển

**Ngắn hạn:** Sử dụng time series kết hợp với LSTM/GRU; thử nghiệm ensemble stacking để cải thiện thêm AUC.

**Trung hạn:** Tích hợp alternative data như hóa đơn tiện ích, thanh toán di động cho nhóm khách hàng ít lịch sử tín dụng; thực hiện fairness audit theo các tiêu chuẩn quốc tế.

**Dài hạn:** Ứng dụng Graph Neural Networks để mô hình hóa mạng lưới bảo lãnh; nghiên cứu causal inference và Federated learning đa ngân hàng.

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
