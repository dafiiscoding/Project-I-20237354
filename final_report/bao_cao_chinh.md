# DỰ BÁO RỦI RO VỠ NỢ TÍN DỤNG BẰNG MACHINE LEARNING
## Loan Default Prediction Using Machine Learning

**Sinh viên thực hiện:** Đoàn Danh Long  
**Mã số sinh viên:** 20237354  
**Giảng viên hướng dẫn:** Nguyễn Cảnh Nam  
**Học kỳ:** 2025.2 — Năm học 2025–2026  

---

## LỜI CẢM ƠN

Em xin gửi lời cảm ơn chân thành đến Giảng viên hướng dẫn — Thầy Nguyễn Cảnh Nam, Khoa Toán–Tin, Trường Đại học Bách Khoa Hà Nội — đã tận tình hướng dẫn, góp ý và định hướng trong suốt quá trình thực hiện đồ án.

Em cũng xin cảm ơn gia đình và bạn bè đã luôn động viên, tạo điều kiện thuận lợi để em có thể hoàn thành đồ án đúng tiến độ.

Dù đã cố gắng hết sức, báo cáo chắc chắn còn nhiều thiếu sót. Em rất mong nhận được nhận xét và góp ý từ Thầy để tiếp tục cải thiện trong những nghiên cứu tiếp theo.

*Hà Nội, tháng 4 năm 2026 — Đoàn Danh Long*

---

## TÓM TẮT

Báo cáo trình bày nghiên cứu ứng dụng Machine Learning vào bài toán dự báo rủi ro vỡ nợ tín dụng, sử dụng bộ dữ liệu *Give Me Some Credit* (Kaggle, 150.000 hồ sơ vay). Bốn mô hình Supervised Learning được xây dựng và đánh giá theo thứ tự tăng dần độ phức tạp: Logistic Regression (AUC=0,8432), Decision Tree CART (AUC=0,8579), Random Forest (AUC=0,8703), và XGBoost (AUC=0,8714). Mô hình tốt nhất — XGBoost với cơ chế boosting bậc hai — vượt mục tiêu đặt ra (AUC > 0,87) và được phân tích sâu qua SHAP TreeExplainer. Feature Engineering có chủ đích (TotalDelinquencyScore, FinancialStressIndex) được SHAP xác nhận là 2 trong top-3 đặc trưng quan trọng nhất. Ngưỡng F2-optimal t=0,625 tăng Recall từ 51,6% lên 66,9% và giảm ước tính chi phí $2 triệu so với ngưỡng F1-optimal (t=0,77), là điểm thỏa hiệp giữa tối thiểu hóa FN và kiểm soát FP. Sản phẩm cuối là ứng dụng Streamlit tương tác, giải thích từng quyết định tín dụng bằng SHAP waterfall chart.

**Từ khóa:** dự báo vỡ nợ, chấm điểm tín dụng, XGBoost, SHAP, tối ưu F-beta, data imbalance.

---

## MỤC LỤC

1. [Giới thiệu](#chương-1-giới-thiệu)
2. [Cơ sở lý thuyết](#chương-2-cơ-sở-lý-thuyết)
3. [Dữ liệu và Tiền xử lý](#chương-3-dữ-liệu-và-tiền-xử-lý)
4. [Thực nghiệm và Đánh giá](#chương-4-thực-nghiệm-và-đánh-giá)
5. [Sản phẩm — Streamlit Dashboard](#chương-5-sản-phẩm--streamlit-dashboard)
6. [Kết luận và Hướng phát triển](#chương-6-kết-luận-và-hướng-phát-triển)
7. [Tài liệu tham khảo](#tài-liệu-tham-khảo)
8. [Phụ lục](#phụ-lục)

---

# CHƯƠNG 1: GIỚI THIỆU

## 1.1 Tính cấp thiết của vấn đề

Rủi ro tín dụng là một trong những nguy cơ lớn nhất đối với sự ổn định của hệ thống tài chính. Theo Ngân hàng Nhà nước Việt Nam, tỷ lệ nợ xấu (NPL — Non-Performing Loan) toàn hệ thống cuối năm 2023 ở mức 4,55%, tương đương hàng trăm nghìn tỷ đồng tài sản có nguy cơ mất vốn [1]. Trên phạm vi quốc tế, cuộc khủng hoảng tài chính toàn cầu 2008 có nguồn gốc trực tiếp từ việc định giá sai rủi ro tín dụng trong thị trường thế chấp bất động sản Mỹ.

Quy trình thẩm định tín dụng truyền thống dựa vào phán đoán của chuyên viên tín dụng, sàng lọc thủ công một số chỉ số tài chính. Phương pháp này bộc lộ hai yếu điểm căn bản: **(1)** không mở rộng được quy mô khi khối lượng đơn vay tăng đột biến (đặc biệt trong bối cảnh ngân hàng số), và **(2)** thiếu tính nhất quán — cùng một hồ sơ có thể được đánh giá khác nhau bởi hai chuyên viên. Các mô hình tính điểm tín dụng (credit scoring) thống kê ra đời từ những năm 1950 (FICO Score, 1956) giải quyết phần nào vấn đề quy mô, nhưng tính tuyến tính của chúng bỏ qua nhiều tương tác phi tuyến giữa các biến rủi ro.

Sự phát triển của Machine Learning mở ra khả năng xây dựng mô hình chấm điểm tín dụng tự động, học trực tiếp từ lịch sử hàng triệu hồ sơ, nắm bắt được cả những tương tác phi tuyến phức tạp. Đồng thời, kỹ thuật SHAP (SHapley Additive exPlanations) giải quyết vấn đề "black-box" — yêu cầu pháp lý bắt buộc ngân hàng phải giải thích lý do từ chối cho khách hàng (Basel III, Điều 431).

## 1.2 Mục tiêu nghiên cứu

Nghiên cứu này đặt ra các mục tiêu SMART sau:

*Bảng 1 — Mục tiêu SMART và kết quả đạt được*

| Mục tiêu | Chỉ tiêu cụ thể | Kết quả đạt được |
|----------|----------------|-----------------|
| Xây dựng mô hình phân loại | AUC-ROC ≥ 0,87 trên tập kiểm tra | **0,8714** |
| So sánh 4 thuật toán | Bảng đầy đủ: AUC, F1, Precision, Recall, thời gian huấn luyện | Bảng 4.1 |
| Feature Engineering có cơ sở | ≥ 2 đặc trưng mới vào top-5 SHAP | **2 trong top-3** SHAP |
| Tối ưu ngưỡng theo $F_2$-score | $F_2$-score tối ưu, ước tính chi phí | $t=0,625$, Recall=66,9% |
| Sản phẩm demo | Ứng dụng chạy được, giải thích bằng SHAP | Streamlit |

> **Chú thích:** SMART = **S**pecific (Cụ thể) — **M**easurable (Đo lường được) — **A**chievable (Khả thi) — **R**elevant (Thực tế) — **T**ime-bound (Có thời hạn).

## 1.3 Phạm vi nghiên cứu

- **Dữ liệu:** Bộ dữ liệu công khai *Give Me Some Credit* (Kaggle, 2011), 149.999 hồ sơ vay tiêu dùng Mỹ.
- **Mô hình:** 4 mô hình Binary Classification có giám sát: Logistic Regression, Decision Tree CART, Random Forest, XGBoost.
- **Phạm vi không bao gồm:** Neural network sâu, dữ liệu thời gian thực, tích hợp hệ thống ngân hàng, các thị trường ngoài phạm vi bộ dữ liệu.
- **Ngôn ngữ lập trình:** Python 3.14, thư viện scikit-learn 1.8.0, XGBoost 3.2.0, SHAP 0.51.0.

---

# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

## 2.1 Bài toán Binary Classification

### 2.1.1 Định nghĩa hình thức

Cho tập dữ liệu huấn luyện $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ với $\mathbf{x}_i \in \mathbb{R}^p$ là vector đặc trưng $p$ chiều và $y_i \in \{0, 1\}$ là nhãn lớp. Mục tiêu là học hàm $f: \mathbb{R}^p \to [0, 1]$ sao cho $f(\mathbf{x}) = P(y=1|\mathbf{x})$ ước lượng chính xác xác suất hậu nghiệm (posterior probability) của sự kiện vỡ nợ.

Quyết định phân loại tại ngưỡng $t$:

$$\hat{y} = \mathbb{1}[f(\mathbf{x}) \geq t]$$

### 2.1.2 Class Imbalance và hệ quả

Trong dữ liệu này, tỷ lệ vỡ nợ là 6,68%, tạo ra tình trạng class imbalance tỷ lệ 1:14. Điều này dẫn đến hai hệ quả quan trọng:

**Hệ quả 1 — Accuracy không phù hợp và không đáp ứng được yêu cầu của bài toán thực tế:** Mô hình baseline $f(\mathbf{x}) \equiv 0$ (đoán tất cả không vỡ nợ) đạt Accuracy = 93,32% nhưng Recall = 0 — không có giá trị ứng dụng trong phát hiện rủi ro tín dụng.

**Hệ quả 2 — Ngưỡng tối ưu ≠ 0,5:** Với $P(y=1) = 0,0668$, quyết định Bayes tối ưu tối thiểu hóa kỳ vọng chi phí phân loại. Phân loại là 1 (từ chối vay) khi chi phí kỳ vọng của việc dự báo nhầm thành 0 (bỏ sót người vỡ nợ) vượt chi phí kỳ vọng của việc dự báo nhầm thành 1 (từ chối oan):

$$c_{FN} \cdot P(y=1|\mathbf{x}) > c_{FP} \cdot P(y=0|\mathbf{x})$$

Rút gọn (vì $P(y=0|\mathbf{x}) = 1 - P(y=1|\mathbf{x})$): phân loại là 1 khi $P(y=1|\mathbf{x}) > t^*$ với **ngưỡng Bayes tối ưu**:

$$t^* = \frac{c_{FP}}{c_{FP} + c_{FN}}$$

Với $c_{FN} = \$11.250$ và $c_{FP} = \$500$: $t^* = 500/(500+11.250) \approx 0{,}043$. Lưu ý quan trọng: công thức này hoạt động trực tiếp trên *xác suất hậu nghiệm* $P(y=1|\mathbf{x})$, không phụ thuộc vào tỷ lệ prior $P(y=0)/P(y=1)$ vì prior đã được hấp thụ vào $P(y=1|\mathbf{x})$ thông qua định lý Bayes — điều kiện: $P(y=1|\mathbf{x})$ phải là xác suất hậu nghiệm calibrated từ phân phối gốc, không phải đầu ra thô của mô hình huấn luyện trên dữ liệu resampled. Tuy nhiên ngưỡng cực thấp này ($t^*=0{,}043$) khiến ~74% hồ sơ bị gắn nhãn TỪ CHỐI — quá tải hệ thống thẩm định thủ công, không khả thi trong vận hành. Điều này dẫn đến cách tiếp cận thực tế hơn: dùng F2-optimal threshold để tìm ngưỡng cân bằng giữa Recall và tính khả thi trong vận hành, được trình bày ở §4.7.

### 2.1.3 Evaluation Metrics

**Confusion Matrix**:

|  | Pred = 0 | Pred = 1 |
|--|----------|----------|
| **Actual = 0** | TN | FP |
| **Actual = 1** | FN | TP |

Từ đó:
- $\text{Precision} = \frac{TP}{TP + FP}$ — Trong số những người bị từ chối, bao nhiêu % đúng là vỡ nợ?
- $\text{Recall} = \frac{TP}{TP + FN}$ — Trong số người thực sự vỡ nợ, bắt được bao nhiêu %?
- $F_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 P + R}$ — Trung bình điều hòa có trọng số

**AUC-ROC** đo khả năng phân biệt của mô hình, không phụ thuộc ngưỡng:
$$\text{AUC} = P(f(\mathbf{x}^+) > f(\mathbf{x}^-))$$
với $\mathbf{x}^+, \mathbf{x}^-$ lần lượt là một mẫu dương và âm ngẫu nhiên.

**AUC-PR** (Precision-Recall curve area) nhạy hơn với lớp thiểu số, phù hợp đánh giá trong trường hợp class imbalance nghiêm trọng. Nghiên cứu này chọn tối ưu hóa AUC-ROC trong cross-validation để đảm bảo khả năng xếp hạng tổng thể (ranking ability), đồng thời dùng $F_2$-score ở khâu chọn ngưỡng để ưu tiên xử lý lớp thiểu số — qua đó bù đắp hạn chế của AUC-ROC trên dữ liệu mất cân bằng mà không cần tính AUC-PR riêng.

## 2.2 Logistic Regression

### 2.2.1 Hypothesis Function và log-odds

Logistic Regression mô hình hóa xác suất hậu nghiệm qua hàm sigmoid:

$$P(y=1|\mathbf{x}; \boldsymbol{\beta}) = \sigma(\mathbf{x}^\top \boldsymbol{\beta}) = \frac{1}{1 + e^{-\mathbf{x}^\top \boldsymbol{\beta}}}$$

Log-odds (logit) là hàm tuyến tính của các đặc trưng:

$$\ln \frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})} = \mathbf{x}^\top \boldsymbol{\beta} = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p$$

**Ý nghĩa tài chính:** $e^{\beta_j}$ là *tỷ suất chênh* (odds ratio) của đặc trưng $j$ — tăng $x_j$ lên 1 đơn vị nhân tỷ suất chênh vỡ nợ lên $e^{\beta_j}$ lần.

### 2.2.2 Ước lượng tham số — Maximum Likelihood Estimation

Hàm log-likelihood âm (cross-entropy loss):

$$\mathcal{L}(\boldsymbol{\beta}) = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \ln \sigma(\mathbf{x}_i^\top \boldsymbol{\beta}) + (1-y_i) \ln(1 - \sigma(\mathbf{x}_i^\top \boldsymbol{\beta})) \right]$$

Gradient (convex function, có nghiệm toàn cục):

$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\beta}} = \frac{1}{N} \mathbf{X}^\top (\hat{\mathbf{y}} - \mathbf{y}), \quad \hat{y}_i = \sigma(\mathbf{x}_i^\top \boldsymbol{\beta})$$

### 2.2.3 Regularization

**L2 (Ridge):** $\mathcal{L}_{L2} = \mathcal{L}(\boldsymbol{\beta}) + \frac{1}{2C}\|\boldsymbol{\beta}\|_2^2$

Nghiệm bị co về 0 nhưng không bằng 0. Ổn định về số với các đặc trưng có tương quan cao.

**L1 (Lasso):** $\mathcal{L}_{L1} = \mathcal{L}(\boldsymbol{\beta}) + \frac{1}{C}\|\boldsymbol{\beta}\|_1$

Cho nghiệm thưa (sparse) — tự động loại các đặc trưng không quan trọng. $C$ là nghịch đảo cường độ regularization (nhỏ = regularization mạnh).

Trong thực nghiệm (solver SAGA), L1 với $C=0,001$ đưa hệ số `NumberOfTimes90DaysLate` về 0 — cho thấy `TotalDelinquencyScore` đã mã hóa đủ thông tin từ đặc trưng đó.

## 2.3 Decision Tree CART

### 2.3.1 Tiêu chí phân chia — Gini Impurity

CART (Classification and Regression Trees) phân chia đệ quy không gian đặc trưng. Với node $t$ chứa $N_t$ mẫu, Gini impurity:

$$G(t) = 1 - \sum_{c \in \{0,1\}} p(c|t)^2 = 2 p_t (1 - p_t)$$

với $p_t = P(y=1|t)$ là tỷ lệ dương trong node. $G(t) = 0$ khi node thuần khiết ($p_t \in \{0,1\}$), $G(t) = 0,5$ khi node hỗn tạp tối đa ($p_t = 0,5$).

**Giải thích trực quan:** Gini impurity đo mức độ hỗn tạp của một node — xác suất một mẫu ngẫu nhiên bị phân loại sai nếu gán nhãn theo phân phối trong node đó. Node thuần khiết: $G = 0$; node hỗn tạp tối đa ($p_t = 0,5$): $G = 0,5$.

**Tiêu chí phân chia** tại node $t$, chọn đặc trưng $j$ và ngưỡng $\theta$ để maximize:

$$\Delta G(t, j, \theta) = G(t) - \frac{N_L}{N_t} G(t_L) - \frac{N_R}{N_t} G(t_R)$$

### 2.3.2 Kiểm soát Overfitting — Pruning

Cây đầy đủ chiều sâu gây overfitting với dữ liệu huấn luyện. Kiểm soát qua:
- `max_depth`: giới hạn độ sâu tối đa (tìm thấy max_depth=10 là tốt nhất)
- `min_samples_leaf`: số mẫu tối thiểu tại nút lá (200 — tránh nút lá quá nhỏ không đáng tin cậy)

### 2.3.3 class_weight='balanced' (Bù đắp Class Imbalance)

Với mất cân bằng 1:14, scikit-learn gán $w_+ = N/(2 \cdot N_+) = 149.999/(2 \times 10.026) \approx 7{,}48$ cho lớp thiểu số và $w_- \approx 0{,}54$ cho lớp đa số. Tỷ lệ $w_+/w_- \approx 14$ bù đắp chính xác mức mất cân bằng. Điều chỉnh tiêu chí phân chia thành weighted Gini:

$$G_w(t) = 1 - \sum_c \left(\frac{\sum_{i:y_i=c} w_i}{\sum_i w_i}\right)^2$$

## 2.4 Random Forest

### 2.4.1 Bootstrap Aggregating (Bagging)

Random Forest huấn luyện $B$ cây quyết định độc lập, mỗi cây trên một bootstrap sample:

$$\hat{f}_{RF}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B} f_b(\mathbf{x})$$

Mỗi bootstrap sample lấy $N$ mẫu có hoàn trả từ $N$ mẫu gốc. Xác suất để một mẫu *không* được chọn là $(1 - 1/N)^N \to e^{-1} \approx 36,8\%$ — đây chính là **Out-of-Bag (OOB) samples** dùng để ước lượng sai số ngoài mẫu mà không cần tập kiểm định.

### 2.4.2 Giảm Variance — Phân tích lý thuyết

Cho $B$ cây với phương sai $\sigma^2$ và hệ số tương quan $\rho$ giữa mọi cặp cây:

$$\text{Var}\left(\frac{1}{B}\sum_b f_b\right) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$

Khi $B \to \infty$, variance $\to \rho\sigma^2$. Lấy mẫu ngẫu nhiên theo đặc trưng ($\sqrt{p}$ đặc trưng mỗi lần phân chia) giảm $\rho$ → giảm variance. Đây là lý do Random Forest vượt Decision Tree đơn về tổng quát hóa.

**OOB score trong thực nghiệm:** 0,8286 — ước lượng không thiên lệch của AUC trên tập kiểm tra (thực tế AUC tập kiểm tra=0,8703, chênh lệch do OOB estimate hơi thận trọng).

### 2.4.3 Feature Importance

Mean Decrease in Impurity (MDI):

$$\text{FI}(j) = \frac{1}{B}\sum_{b=1}^{B} \sum_{t \in f_b: \text{split on } j} \frac{N_t}{N} \cdot \Delta G(t, j)$$

## 2.5 XGBoost — Gradient Boosting Bậc Hai

### 2.5.1 Mô hình Additive

XGBoost xây dựng additive ensemble: $F_T(\mathbf{x}) = \sum_{t=1}^{T} f_t(\mathbf{x})$, với $f_t$ là cây quyết định (weak learner) thứ $t$. Tại bước $t$:

$$F_t(\mathbf{x}) = F_{t-1}(\mathbf{x}) + \eta f_t(\mathbf{x})$$

$\eta \in (0,1)$ là learning rate (= 0,05), kiểm soát shrinkage.

### 2.5.2 Objective Function và xấp xỉ Taylor bậc hai

Hàm mục tiêu tại bước $t$:

$$\mathcal{O}^{(t)} = \sum_{i=1}^{N} l(y_i, F_{t-1}(\mathbf{x}_i) + f_t(\mathbf{x}_i)) + \Omega(f_t)$$

Trong đó regularization term:

$$\Omega(f_t) = \gamma T + \frac{\lambda}{2}\sum_{j=1}^{T} w_j^2$$

$T$ là số nút lá, $w_j$ là trọng số lá, $\gamma$ là penalty tối thiểu gain để tạo phân chia, $\lambda$ là L2 trên trọng số lá.

Xấp xỉ Taylor bậc hai (bỏ hạng bậc nhất vì đã biết $F_{t-1}$):

$$\mathcal{O}^{(t)} \approx \sum_{i=1}^{N} \left[g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t(\mathbf{x}_i)^2\right] + \Omega(f_t)$$

với gradient $g_i = \partial_{F_{t-1}} l(y_i, F_{t-1})$ và Hessian $h_i = \partial^2_{F_{t-1}} l(y_i, F_{t-1})$.

Với binary cross-entropy: $g_i = \hat{p}_i - y_i$, $h_i = \hat{p}_i(1 - \hat{p}_i)$.

### 2.5.3 Trọng số Lá Tối ưu — Nghiệm Closed-Form

Nhóm các mẫu vào leaf $j$: $I_j = \{i : \mathbf{x}_i \in \text{leaf}_j\}$

$$w_j^* = -\frac{G_j}{H_j + \lambda}, \quad G_j = \sum_{i \in I_j} g_i, \quad H_j = \sum_{i \in I_j} h_i$$

**Gain của một phép phân chia** (phân chia node $I$ thành $I_L, I_R$):

$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda}\right] - \gamma$$

Chỉ tạo phân chia khi Gain > 0, tức $\gamma$ kiểm soát cắt tỉa.

### 2.5.4 Xử lý scale_pos_weight

XGBoost điều chỉnh gradient của mẫu dương:

$$g_i^+ = \text{spw} \cdot g_i, \quad h_i^+ = \text{spw} \cdot h_i$$

với spw = 13,96 (tỷ lệ âm/dương). Tương đương với oversampling lớp thiểu số 13,96 lần trong không gian gradient — hiệu quả hơn SMOTE vì không tạo dữ liệu tổng hợp.

### 2.5.5 SHAP — Shapley Additive exPlanations

SHAP dựa trên lý thuyết Shapley trong lý thuyết trò chơi hợp tác (Cooperative Game Theory). Shapley value của đặc trưng $j$ cho mẫu $\mathbf{x}$ là:

$$\phi_j(\mathbf{x}) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} \left[v(S \cup \{j\}) - v(S)\right]$$

trong đó $F$ là tập tất cả đặc trưng, $v(S)$ là giá trị dự báo khi chỉ có đặc trưng trong $S$.

**TreeExplainer** của Lundberg et al. [5] tính exact Shapley values cho tree-based models trong $O(TLD^2)$ thay vì exponential, với $T$ = số cây, $L$ = số leaf, $D$ = max depth.

Tính chất cộng (additivity):
$$f(\mathbf{x}) = \phi_0 + \sum_{j=1}^{p} \phi_j(\mathbf{x})$$
với $\phi_0 = E[f(\mathbf{x})]$ là giá trị cơ sở (base value, giá trị kỳ vọng trên tập dữ liệu).

## 2.6 Nghiên cứu Liên quan

Chấm điểm tín dụng là một trong những lĩnh vực ứng dụng Machine Learning sớm nhất. Một số công trình nổi bật:

**[Altman, 1968]** — Z-Score model: hàm tuyến tính 5 tỷ số tài chính phân biệt doanh nghiệp phá sản. Đây là nền tảng cho mọi mô hình credit scoring về sau.

**[Hand & Henley, 1997]** — Tổng quan các phương pháp thống kê trong credit scoring. Trong phần lớn tình huống thực tế, Logistic Regression vẫn là mô hình baseline mạnh nhất xét theo cả hiệu suất lẫn khả năng triển khai.

**[Lessmann et al., 2015]** — Benchmark 41 mô hình trên 8 bộ dữ liệu tín dụng. Gradient Boosting và Random Forest nhất quán vượt Logistic Regression và Decision Tree đơn về AUC.

**[Lundberg & Lee, 2017]** — Giới thiệu SHAP framework. Chứng minh TreeExplainer cho exact Shapley values hiệu quả cho tree-based models, giải quyết vấn đề khả năng giải thích trong tín dụng.

**[Bucker et al., 2022]** — Phân tích regulatory requirements cho AI trong chấm điểm tín dụng theo EU AI Act. Các tác giả chỉ ra rằng mô hình gradient boosting kết hợp SHAP phù hợp với yêu cầu về khả năng giải thích trong quản trị rủi ro tín dụng theo Basel III Pillar 3.

---

# CHƯƠNG 3: DỮ LIỆU VÀ TIỀN XỬ LÝ

## 3.1 Mô tả Bộ dữ liệu

### 3.1.1 Tổng quan

**Nguồn:** Give Me Some Credit — Kaggle Competition (2011)  
**Tổng số quan sát:** 149.999 hồ sơ vay tiêu dùng tại Mỹ  
**Biến mục tiêu:** `SeriousDlqin2yrs` = 1 nếu người vay trễ hạn trả nợ ≥ 90 ngày trong 2 năm tiếp theo  
**Tỷ lệ dương:** 10.026/149.999 = **6,68%** (mất cân bằng 1:14)

![Phân phối biến mục tiêu — 6,68% vỡ nợ so với 93,32% không vỡ nợ](../reports/fig_01_target_distribution.png)
*Hình 3.1: Data imbalance 1:14 — mô hình "predict all 0" đạt Accuracy = 93,3% nhưng Recall = 0, cho thấy thước đo Accuracy không phản ánh đúng mục tiêu bài toán, đòi hỏi sử dụng AUC-ROC và $F_\beta$.*

### 3.1.2 Ý nghĩa Tài chính từng Đặc trưng

| Đặc trưng | Ý nghĩa | Chú ý lĩnh vực |
|---------|---------|--------------|
| `RevolvingUtilizationOfUnsecuredLines` | Tỷ lệ sử dụng hạn mức tín dụng xoay vòng (thẻ tín dụng) | FICO chiếm 30% điểm; >70% là ngưỡng cảnh báo |
| `age` | Tuổi người vay | Đại diện kinh nghiệm tài chính; thanh niên và người cao tuổi rủi ro cao hơn |
| `NumberOfTime30-59DaysPastDueNotWorse` | Số lần trả nợ trễ 30–59 ngày | Trễ hạn nhẹ, yếu tố cấu thành Lịch sử thanh toán (chiếm 35% điểm FICO) |
| `DebtRatio` | Tổng nợ / Thu nhập hàng tháng | Tương đương DTI ratio trong underwriting; chuẩn US: DTI < 43% |
| `MonthlyIncome` | Thu nhập hàng tháng (USD) | 19,8% missing — MAR/MNAR |
| `NumberOfOpenCreditLinesAndLoans` | Số tài khoản tín dụng đang mở | Nhiều tài khoản: overextension hoặc diversification |
| `NumberOfTimes90DaysLate` | Số lần trể hạn > 90 ngày | Tín hiệu mạnh nhất — trễ hạn nghiêm trọng |
| `NumberRealEstateLoansOrLines` | Số khoản vay bất động sản | Tài sản thế chấp, giảm rủi ro |
| `NumberOfTime60-89DaysPastDueNotWorse` | Số lần trễ 60–89 ngày | Trễ hạn trung bình |
| `NumberOfDependents` | Số người phụ thuộc | Tăng gánh nặng tài chính thực tế; 2,6% missing |

### 3.1.3 Phân phối và outliers

Phân tích EDA phát hiện các vấn đề chất lượng dữ liệu quan trọng:

- **age = 0:** 1 mẫu — vô nghĩa về domain → xóa
- **MonthlyIncome** max = $3.008.750 — bất thường (1000× median) → giới hạn tại percentile 99 của tập huấn luyện ($25.000)
- **Delinquency counts** max = 98 — sentinel value (thực tế không ai trễ 98 lần) → giới hạn tại ngưỡng thực tế
- **RevolvingUtilization** > 1 (tức > 100% hạn mức) — một số khách hàng vượt hạn mức → giữ nguyên (thông tin thực)

**Phân phối bất đối xứng mạnh (nghiêng phải):** MonthlyIncome (skewness=115), DebtRatio (skewness=2080). Lý do chọn RobustScaler thay vì StandardScaler.

## 3.2 Phân tích Khám phá Dữ liệu — Các Phát hiện Chính

### 3.2.1 Mối quan hệ phi tuyến giữa RevolvingUtilization và tỷ lệ vỡ nợ

Pearson correlation RevolvingUtil — target = −0,002 (gần zero), trong khi Spearman ρ = +0,24 (tương quan đáng kể). Điều này chứng minh tồn tại quan hệ **phi tuyến đơn điệu**: tỷ lệ vỡ nợ tăng theo hàm bậc thang, không theo đường thẳng. Các mô hình tree-based nắm bắt được điều này; Logistic Regression không.

### 3.2.2 Spearman Correlation giữa các biến delinquency

Spearman ρ(90+days, 60-89days) = 0,49, ρ(90+days, 30-59days) = 0,45. Multicollinearity cao → VIF = ∞ cho các cặp này. Giải pháp: tổng hợp vào `TotalDelinquencyScore` thay vì dùng riêng lẻ.

![Ma trận tương quan Spearman giữa các đặc trưng](../reports/fig_05_correlation_heatmap.png)
*Hình 3.2b: Spearman Correlation Matrix — cho thấy multicollinearity cao giữa ba biến delinquency (hệ số 0,45–0,49), là cơ sở kỹ thuật để nén chúng thành `TotalDelinquencyScore`. `DebtRatio` và `MonthlyIncome` gần như không tương quan (ρ ≈ 0), cho thấy thông tin bổ sung từ cả hai biến này.*

### 3.2.3 Tỷ lệ Vỡ nợ theo Số lần Trễ hạn

Phân tích biểu đồ tỷ lệ vỡ nợ cho thấy:
- 0 lần trễ > 90 ngày: tỷ lệ vỡ nợ = 4,1%
- 1 lần: 48,7%
- 2 lần: 70,3%
- ≥ 3 lần: >80%

→ Phi tuyến mạnh, ngưỡng rõ ràng tại count=1. Cơ sở cho TotalDelinquencyScore theo trọng số.

## 3.3 Xử lý Missing Values

### 3.3.1 Kiểm định cơ chế missing

Để lựa chọn phương pháp imputation, kiểm tra cơ chế missing của `MonthlyIncome` (19,8% null):

**Chi-squared test:** $H_0$: MonthlyIncome missing độc lập với target.

$$\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}} = 67,89, \quad p \approx 0$$

Bác bỏ $H_0$ → dữ liệu là **MAR/MNAR** (Missing At Random / Not At Random): người có thu nhập thấp thường không khai báo → Median imputation sẽ ước tính quá cao thu nhập thực của nhóm này → thiên lệch.

### 3.3.2 Lý do chọn KNN Imputer

**KNN Imputation** (k=5, nan-Euclidean distance): ước lượng $x_{i,\text{income}}$ bằng trung bình có trọng số của 5 láng giềng gần nhất trong không gian đặc trưng còn lại:

$$\hat{x}_{i,j} = \frac{\sum_{k \in \mathcal{N}(i)} w_k x_{k,j}}{\sum_{k \in \mathcal{N}(i)} w_k}, \quad w_k = \frac{1}{d(\mathbf{x}_i^{-j}, \mathbf{x}_k^{-j})}$$

Kết quả so sánh thực nghiệm:
- Median imputation: tất cả missing = $5.400 (không có phương sai)
- KNN imputation: trung bình của nhóm missing sau impute = $336 (không phải trung bình toàn bộ dữ liệu; thấp hơn nhiều so với ~$5.400 vì nhóm thiếu dữ liệu thiên về thu nhập thấp), std = $1.157

KNN nắm bắt được thực tế: người 25 tuổi chưa đi làm (age, 0 credit lines) nhận imputed income thấp; người 45 tuổi, nhiều tài khoản nhận imputed income cao hơn.

![So sánh phân phối MonthlyIncome sau KNN Imputer và Median Imputation](../reports/fig_11_imputation_comparison.png)
*Hình 3.2: KNN Imputer tạo ra phân phối đa dạng (std=$1.157) phản ánh thực tế kinh tế — Median đặt tất cả giá trị thiếu tại một điểm duy nhất ($5.400), xóa đi mọi tín hiệu tương quan giữa thu nhập và các đặc trưng khác.*

`NumberOfDependents` (2,6% null, MCAR): Median = 0 đủ.

## 3.4 Feature Engineering

Bốn đặc trưng mới được tạo dựa trên kiến thức lĩnh vực tín dụng:

### 3.4.1 TotalDelinquencyScore

$$\text{TDS} = 3 \cdot N_{90+} + 2 \cdot N_{60-89} + 1 \cdot N_{30-59}$$

Theo FICO score methodology, delinquency được phân loại theo mức nghiêm trọng: trễ > 90 ngày ảnh hưởng gấp 3 lần trễ 30–59 ngày. Trọng số 3-2-1 phản ánh tỷ lệ vỡ nợ quan sát được: 70% với nhóm trễ ≥ 90 ngày từ 2 lần trở lên, so với 8% với nhóm chỉ trễ 30-59 ngày một lần.

**Spearman ρ với biến mục tiêu:** 0,345 — cao hơn bất kỳ đặc trưng trễ hạn thô nào riêng lẻ (max 0,342 với `NumberOfTimes90DaysLate`). Nén thành một điểm số giảm multicollinearity đồng thời tăng sức mạnh dự báo.

### 3.4.2 FinancialStressIndex

$$\text{FSI} = \text{RevolvingUtilization} \times \text{TotalDelinquencyScore}$$

Đặc trưng này nắm bắt tương tác phi tuyến giữa hai chiều rủi ro: người có utilization cao (áp lực tài chính ngắn hạn) VÀ lịch sử trễ hạn (áp lực dài hạn) rủi ro cao hơn nhiều so với từng tín hiệu riêng lẻ. Ví dụ: RevUtil=0,9, TDS=5 → FSI=4,5; RevUtil=0,9, TDS=0 → FSI=0 — cùng mức utilization nhưng kết luận hoàn toàn khác.

**Spearman ρ với biến mục tiêu:** 0,346 — cao nhất trong tất cả đặc trưng, kể cả đặc trưng gốc. Kết quả SHAP ở §4.5 xác nhận: mean|SHAP|=0,577, dẫn đầu toàn bộ tập đặc trưng.

![Tương quan Spearman của 14 đặc trưng với biến mục tiêu — đặc trưng được tạo dẫn đầu](../reports/fig_12_feature_importance.png)
*Hình 3.3: FinancialStressIndex (ρ=0,346) và TotalDelinquencyScore (ρ=0,345) dẫn đầu tương quan với target, vượt các đặc trưng gốc thô — xác nhận Feature Engineering có chủ đích theo lĩnh vực tài chính mang lại giá trị dự báo cao hơn.*

### 3.4.3 AbsoluteDebt — Dư nợ tuyệt đối (tên code: `DebtToIncomeRatio`)

$$\text{AbsoluteDebt} = \text{DebtRatio} \times \text{MonthlyIncome}$$

`DebtRatio` là tỷ lệ nợ/income (không thứ nguyên), không phản ánh quy mô tuyệt đối. DebtRatio=2 với income=$2.000 (nợ $4.000/tháng) khác hoàn toàn DebtRatio=2 với income=$10.000 (nợ $20.000/tháng). Phép nhân triệt tiêu income → kết quả là **số tiền nợ tuyệt đối** (USD/tháng), không phải ratio. SHAP cho mean|SHAP|=0,065 — nhỏ nhưng đủ để giữ lại. *(Lưu ý: tên cột trong code là `DebtToIncomeRatio` — legacy misnomer, giữ nguyên để tương thích với model đã huấn luyện.)*

### 3.4.4 DelinquencyTrend (Cán cân mức độ trễ hạn)

$$\text{DelinquencyTrend} = N_{30-59} - N_{90+}$$

Dương = chủ yếu mắc trễ hạn nhẹ (30–59 ngày), ít trễ hạn nghiêm trọng; Âm = chủ yếu mắc trễ hạn nghiêm trọng (90+ ngày). *Lưu ý: bộ dữ liệu là cross-sectional snapshot, không có timestamp — biến này đo cán cân mức độ trễ hạn, không phản ánh xu hướng thời gian.* SHAP cho mean|SHAP|=0,002 — thấp nhất, có thể loại trong các nghiên cứu tiếp theo nếu cần giảm chiều.

## 3.5 Xử lý Class Imbalance

### 3.5.1 Chiến lược đã thử

**class_weight='balanced':** Gán trọng số $w_+ = N/(2N_+) \approx 7{,}48$ cho lớp thiểu số và $w_- \approx 0{,}54$ cho lớp đa số trong hàm loss (tỷ lệ $w_+/w_- \approx 14$). Không tạo dữ liệu tổng hợp, không ảnh hưởng phân phối đặc trưng.

**SMOTE (Synthetic Minority Oversampling Technique):** Dùng như một phương án **thử nghiệm** trên tập huấn luyện để so sánh với `class_weight`; không phải lựa chọn cuối cùng của quy trình triển khai. Cơ chế là tạo mẫu tổng hợp lớp thiểu số bằng nội suy tuyến tính giữa các điểm gần nhau trong không gian đặc trưng:

$$\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_{nn} - \mathbf{x}_i), \quad \lambda \sim \mathcal{U}[0,1]$$

**Lưu ý quan trọng:** SMOTE chỉ áp dụng trên tập huấn luyện (sau khi phân chia), KHÔNG áp dụng trên tập kiểm định/kiểm tra (val/test). Áp dụng trước khi phân chia là rò rỉ dữ liệu.

![Phân phối lớp trước và sau SMOTE — minority class từ 6,68% lên 50%](../reports/fig_15_smote.png)
*Hình 3.4: SMOTE cân bằng hoàn toàn tập huấn luyện (50/50) bằng cách tạo mẫu tổng hợp nội suy giữa các điểm lớp thiểu số — nhưng không được áp dụng trên tập kiểm định/kiểm tra để tránh rò rỉ dữ liệu.*

### 3.5.2 Quyết định cuối cùng

Dùng `class_weight='balanced'` cho LR, DT, RF. Dùng `scale_pos_weight=13,96` cho XGBoost (tương đương về mặt toán học nhưng được tích hợp sâu vào XGBoost's gradient computation). Không dùng SMOTE trong quy trình cuối cùng vì:
1. `class_weight` đơn giản hơn, không tạo nhiễu từ dữ liệu tổng hợp
2. Kết quả tương đương trong thực nghiệm (đã kiểm tra)
3. Dễ bảo trì và giải thích hơn

## 3.6 Phân chia Dữ liệu

**Phân chia phân tầng 70/15/15:**

| Tập | Kích thước | Positives | Tỷ lệ |
|-----|-----------|-----------|-------|
| Huấn luyện | 104.999 | 7.018 | 6,68% |
| Kiểm định | 22.500 | 1.504 | 6,68% |
| Kiểm tra | 22.500 | 1.504 | 6,68% |

Phân chia phân tầng đảm bảo tỷ lệ lớp thiểu số nhất quán. Tập kiểm định dùng để tối ưu ngưỡng; tập kiểm tra chỉ đụng vào một lần duy nhất để báo cáo kết quả cuối.

---

# CHƯƠNG 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ

## 4.1 Thiết lập Thực nghiệm

**Cross-validation:** Stratified 5-Fold Cross-validation (giữ tỷ lệ 6,68% trong mỗi fold).

**Hyperparameter tuning:** `RandomizedSearchCV` — ưu tiên hơn `GridSearchCV` khi không gian tìm kiếm lớn (thời gian tuyến tính theo số iterations thay vì exponential theo số tham số). Số iterations: LR=12, DT=20, RF=15, XGB=15.

**Metric tối ưu trong CV:** AUC-ROC (không phụ thuộc ngưỡng, phù hợp so sánh mô hình).

**Quy trình xử lý:** Tất cả preprocessing (imputation, scaling) được đóng gói trong `sklearn.Pipeline` để tránh rò rỉ dữ liệu giữa train và val folds. Ngưỡng capping outlier được tính trên tập huấn luyện và áp dụng cố định cho val/test.

**Môi trường:** Python 3.14.2, scikit-learn 1.8.0, XGBoost 3.2.0, SHAP 0.51.0, Windows 11, CPU Intel.

## 4.2 Logistic Regression

### 4.2.1 Cấu hình và kết quả

**Quy trình:** `RobustScaler` → `LogisticRegression(solver='saga', class_weight='balanced')`

`RobustScaler` dùng median và IQR thay vì mean và std — robust với outliers:
$$x_{\text{scaled}} = \frac{x - \text{median}(\mathbf{x})}{\text{IQR}(\mathbf{x})}$$

**Best hyperparameters:** penalty=L1, C=0,001 (regularization mạnh)

**Kết quả trên tập kiểm tra:**

| Metric | Giá trị |
|--------|---------|
| AUC-ROC | **0,8432** |
| F1-score | 0,4340 |
| Precision | 0,3878 |
| Recall | 0,4927 |
| Ngưỡng tối ưu | 0,66 |
| Thời gian huấn luyện | 488s |

### 4.2.2 Phân tích Odds Ratios

| Đặc trưng | Hệ số β | Odds Ratio $e^\beta$ | Diễn giải |
|---------|--------------|---------------------|----------------|
| TotalDelinquencyScore | 0,255 | **1,291** | +1 điểm delinquency → odds vỡ nợ ×1,29 |
| RevolvingUtilization | 0,204 | **1,226** | +10% utilization → odds ×1,12 |
| FinancialStressIndex | 0,186 | **1,205** | Interaction effect |
| age | −0,210 | **0,811** | +10 tuổi → odds ×0,811 (người lớn tuổi ít vỡ nợ) |
| NumberOfTimes90DaysLate | **0** | 1,000 | **L1 triệt tiêu** — dư thừa so với TotalDelinquencyScore |
| DebtToIncomeRatio | **0** | 1,000 | **L1 triệt tiêu** — mô hình tuyến tính không thấy đóng góp |

Đáng chú ý là L1 (C=0,001) đưa hệ số `NumberOfTimes90DaysLate` về 0, trong khi `TotalDelinquencyScore` — vốn đã mã hóa thông tin của đặc trưng đó — vẫn được giữ lại với hệ số dương. Đây là bằng chứng gián tiếp cho thấy Feature Engineering đi đúng hướng.

## 4.3 Decision Tree CART

### 4.3.1 Cấu hình và kết quả

**Best hyperparameters:** max_depth=10, min_samples_leaf=200, criterion=Gini

| Metric | Giá trị |
|--------|---------|
| AUC-ROC | **0,8579** |
| F1-score | 0,4314 |
| Ngưỡng tối ưu | 0,80 |
| Thời gian huấn luyện | 13s |

### 4.3.2 Tree Visualization (top 3 levels)

Node gốc phân chia tại `TotalDelinquencyScore ≤ 2,5`:
- Nhánh trái (TDS ≤ 2,5): 128.000 mẫu, 4,1% vỡ nợ → phân chia tiếp theo `RevolvingUtilization`
- Nhánh phải (TDS > 2,5): 21.000 mẫu, 38,7% vỡ nợ → phân chia tiếp theo `FinancialStressIndex`

Cây học được rằng `TotalDelinquencyScore` là đặc trưng phân biệt bậc nhất — nhất quán với SHAP và kiến thức lĩnh vực.

**Ngưỡng tối ưu cao (0,80):** Decision Tree có xu hướng output xác suất ở đầu dải (gần 0 hoặc gần 1), cần ngưỡng cao để đạt F1 tốt nhất. AUC thấp hơn RF/XGB do variance cao (một cây duy nhất).

## 4.4 Random Forest

### 4.4.1 Cấu hình và kết quả

**Best hyperparameters:** n_estimators=200, max_depth=10, max_features=0,3 (30% đặc trưng ngẫu nhiên mỗi phân chia), min_samples_leaf=1

| Metric | Giá trị |
|--------|---------|
| AUC-ROC | **0,8703** |
| F1-score | 0,4439 |
| Recall | 0,5206 |
| OOB score | 0,8286 |
| Ngưỡng tối ưu | 0,72 |
| Thời gian huấn luyện | 511s |

**OOB estimate:** 0,8286 — nhất quán với AUC tập kiểm tra=0,8703 (ước lượng thận trọng, phù hợp kỳ vọng lý thuyết).

### 4.4.2 Tầm quan trọng Đặc trưng (MDI)

Top 5: TotalDelinquencyScore, FinancialStressIndex, RevolvingUtilization, age, MonthlyIncome — nhất quán với Feature Importance toàn cục qua SHAP của XGBoost. Các đặc trưng được tạo lại xếp đầu.

## 4.5 XGBoost

### 4.5.1 Cấu hình và kết quả

**Best hyperparameters:**

| Hyperparameters | Giá trị | Ý nghĩa |
|---------|---------|---------|
| n_estimators | 200 | Số cây |
| max_depth | 4 | Cây nông — giảm phương sai |
| learning_rate | 0,05 | Shrinkage — học chậm, tổng quát hóa tốt hơn |
| subsample | 0,8 | 80% mẫu mỗi cây — tăng tính ngẫu nhiên |
| colsample_bytree | 0,8 | 80% đặc trưng mỗi cây |
| reg_lambda | 1,0 | L2 trên trọng số lá |
| reg_alpha | 0,1 | L1 trên trọng số lá |
| scale_pos_weight | 13,96 | Bù mất cân bằng dữ liệu |

| Metric | Giá trị |
|--------|---------|
| AUC-ROC | **0,8714** |
| F1-score | 0,4466 |
| Precision | 0,3937 |
| Recall | 0,5160 |
| Ngưỡng tối ưu | 0,77 |
| Thời gian huấn luyện | 127s |

![Confusion matrix, ROC curve, Precision-Recall curve và learning curve của XGBoost](../reports/fig_19_xgb_analysis.png)
*Hình 4.1: XGBoost đạt AUC=0,8714 — ROC curve nằm xa đường ngẫu nhiên, PR curve thể hiện hiệu quả ở ngưỡng precision cao, learning curve hội tụ sau ~45.000 mẫu với khoảng cách huấn luyện-kiểm định (train/val gap) = 0,021.*

### 4.5.2 SHAP Analysis

**Global Feature Importance (mean |SHAP value|):**

| Hạng | Đặc trưng | Loại | Mean\|SHAP\| |
|------|---------|------|-------------|
| 1 | FinancialStressIndex | **Được tạo** | **0,577** |
| 2 | RevolvingUtilizationOfUnsecuredLines | Gốc | 0,535 |
| 3 | TotalDelinquencyScore | **Được tạo** | 0,410 |
| 4 | age | Gốc | 0,244 |
| 5 | NumberOfOpenCreditLinesAndLoans | Gốc | 0,168 |
| 9 | DebtToIncomeRatio | Được tạo | 0,065 |
| 12 | NumberOfTimes90DaysLate | Gốc | 0,013 |

![Biểu đồ thanh SHAP tổng quát — mean |SHAP value| của 14 đặc trưng](../reports/fig_26a_shap_bar.png)
*Hình 4.2: FinancialStressIndex (mean|SHAP|=0,577) và TotalDelinquencyScore (0,410) — 2 đặc trưng Feature Engineering thủ công — chiếm vị trí #1 và #3, xác nhận inductive bias từ domain tài chính có giá trị ngay cả với mô hình phi tuyến mạnh.*

**Phát hiện đáng chú ý:**
- `FinancialStressIndex` (#1, 0,577) > `RevolvingUtilization` (#2, 0,535): tương tác phi tuyến vượt thành phần gốc — nắm bắt sự cộng hưởng giữa tỷ lệ sử dụng tín dụng cao và lịch sử trễ hạn
- `NumberOfTimes90DaysLate` chỉ rank 12 (SHAP=0,013): XGBoost học tín hiệu này gián tiếp qua TotalDelinquencyScore và FinancialStressIndex, không cần đặc trưng thô trực tiếp

**SHAP base value:** 0,0148 (log-odds), tương ứng xác suất cơ sở 50,37% — hệ quả của `scale_pos_weight` điều chỉnh prior probability.

## 4.6 Bảng So sánh Tổng hợp

**Bảng 4.1:** So sánh tổng hợp 4 mô hình trên tập kiểm tra (22.500 hồ sơ)

| Mô hình | AUC-ROC | F1 | Precision | Recall | Ngưỡng | Thời gian huấn luyện | Inf./record | Kích thước mô hình |
|---------|---------|-----|-----------|--------|-----------|------------|-------------|------------|
| Logistic Regression | 0,8432 | 0,4340 | 0,3878 | 0,4927 | 0,66 | 488s | **0,25 µs** | ~50 KB |
| Decision Tree | 0,8579 | 0,4314 | 0,3991 | 0,4694 | 0,80 | **13s** | **0,20 µs** | ~200 KB |
| Random Forest | 0,8703 | 0,4439 | 0,3869 | 0,5206 | 0,72 | 511s | 4,82 µs | ~12 MB |
| **XGBoost** | **0,8714** | **0,4466** | **0,3937** | 0,5160 | 0,77 | 127s | **0,79 µs** | **340 KB** |

*Ghi chú: Inference time đo trên lô 22.500 mẫu (Python 3.14, CPU Intel), lấy minimum của 3 lần chạy để loại bỏ nhiễu đo lường.*

![Overlay ROC curves của 4 mô hình trên tập kiểm tra](../reports/fig_20_model_comparison.png)
*Hình 4.3: XGBoost (AUC=0,8714) và Random Forest (0,8703) gần như trùng nhau trên ROC curve — nhưng XGBoost chiếm ưu thế về tốc độ huấn luyện (127s vs 511s) và kích thước mô hình (340KB vs 12MB), tạo lợi thế triển khai rõ ràng.*

**Nhận xét:**
1. AUC tăng đơn điệu theo độ phức tạp mô hình: LR < DT < RF ≲ XGB
2. XGBoost và RF có AUC point estimate rất gần nhau (gap ≈ 0,001–0,004 tùy lần huấn luyện) — cần kiểm định DeLong (§4.6.1) để khẳng định khác biệt có ý nghĩa thống kê hay không, vì so sánh chỉ qua point estimate là không đủ
3. XGBoost vượt RF về: thời gian huấn luyện (127s vs 511s), kích thước mô hình (340KB vs 12MB), inference time (0,79 µs vs 4,82 µs/record — 6× nhanh hơn), khả năng giải thích (SHAP chính xác)
4. **Mô hình tốt nhất: XGBoost** → lưu `models/best_model.pkl`

### 4.6.1 Kiểm định Thống kê Sự khác biệt AUC — DeLong Test

AUC point estimate giữa các mô hình có khoảng cách nhỏ (RF ≈ XGB) nhưng để khẳng định "XGBoost vượt trội" cần kiểm định thống kê có nguyên tắc. Kiểm định DeLong [13] so sánh hai AUC point estimates từ cùng tập kiểm tra, có tính đến tương quan giữa hai bộ dự báo (cùng dữ liệu nên hai estimator không độc lập — bỏ qua covariance sẽ khiến variance bị ước tính quá cao và mất năng lực kiểm định).

**Phương pháp — U-statistic của AUC:**

Ký hiệu $n_+ = 1.504$ (số positives trong tập kiểm tra), $n_- = 20.996$ (số negatives). AUC được ước lượng qua các thành phần cấu trúc (structural components):

$$V_{10,i} = \frac{1}{n_-}\sum_{j=1}^{n_-} \mathbf{1}[f(x^+_i) > f(x^-_j)] + \frac{1}{2}\mathbf{1}[f(x^+_i) = f(x^-_j)]$$

Phương sai của AUC: $\hat{\text{Var}}(\widehat{\text{AUC}}) = \hat{\sigma}^2_{10}/n_+ + \hat{\sigma}^2_{01}/n_-$ với $\hat{\sigma}^2_{10} = \text{Var}(V_{10})$.

Kiểm định hai chiều: $z = (\widehat{\text{AUC}}_A - \widehat{\text{AUC}}_B) / \sqrt{\hat{\text{Var}}(\widehat{\text{AUC}}_A - \widehat{\text{AUC}}_B)}$

trong đó mẫu số có tính đến covariance giữa hai AUC estimators (cùng tập test).

**Kết quả thực nghiệm** (chạy `python notebooks/analysis_addendum.py`, output đầy đủ ở `reports/addendum_results.md`):

| So sánh | AUC A | AUC B | Δ AUC | z-stat | p-value (2-tailed) | Kết luận |
|---------|-------|-------|-------|--------|--------------------|----------|
| XGBoost vs RF | 0,8714 | 0,8671 | +0,0043 | 4,1833 | < 0,0001 | **Có ý nghĩa thống kê** (p<0,05) |

> *Ghi chú tái lập:* RF được huấn luyện lại với cùng hyperparameters (`n_estimators=200, max_depth=10, max_features=0.3, class_weight='balanced'`) bằng `notebooks/train_supplementary_models.py` để có file `models/model_rf.pkl` cho DeLong test. AUC=0,8671 chênh nhẹ so với 0,8703 ghi nhận trong notebook 03 (sklearn 1.8 vs phiên bản gốc, không ảnh hưởng kết luận định tính).

Với $\Delta$AUC = 0,0043 và $n_{\text{test}} = 22.500$, DeLong cho $z = 4{,}18$, $p < 0{,}0001$ — XGBoost vượt RF về AUC một cách có ý nghĩa thống kê. Tuy nhiên độ chênh tuyệt đối rất nhỏ (≈0,4 điểm AUC), không đủ lớn để là tiêu chí lựa chọn duy nhất. **Quyết định chọn XGBoost** kết hợp ba lý do bổ sung: (i) AUC cao hơn có ý nghĩa thống kê (DeLong p<0,0001), (ii) lợi thế vận hành (tốc độ huấn luyện 4×, kích thước mô hình 35×, inference 6× nhanh hơn), (iii) khả năng giải thích qua SHAP TreeExplainer chính xác thay vì xấp xỉ.

## 4.7 Phân tích Sai số

### 4.7.1 Cấu trúc Lỗi tại ngưỡng=0,77

| Category | Số lượng | % |
|----------|---------|---|
| **TP** (dự báo đúng vỡ nợ) | 776 | 51,6% tổng số vỡ nợ thực |
| **FN** (bỏ sót vỡ nợ) | 728 | **48,4% tổng số vỡ nợ thực** |
| **FP** (từ chối nhầm) | 1.195 | 5,7% tổng số không vỡ nợ |
| **TN** (duyệt đúng) | 19.801 | 94,3% tổng số không vỡ nợ |

### 4.7.2 Profile của False Negatives

FN là những người vỡ nợ mà mô hình không cảnh báo được:

| Đặc trưng | FN trung vị | TP trung vị | Tỷ lệ FN/TP |
|---------|-----------|-----------|------------|
| TotalDelinquencyScore | **0** | 5,0 | 0,00 |
| FinancialStressIndex | **0** | 4,0 | 0,00 |
| RevolvingUtilization | 0,518 | 1,000 | 0,52 |
| MonthlyIncome | $4.200 | $3.400 | **1,24** |
| FN probability score | **0,523** | — | — |

![Histogram predicted score phân theo nhóm lỗi — FN, TP, FP, TN](../reports/fig_24_score_distribution.png)
*Hình 4.4: FN (vỡ nợ bị bỏ sót) tập trung ở vùng score thấp (0,2–0,5) — đây là những người vỡ nợ không có dấu hiệu cảnh báo trước, phản ánh giới hạn căn bản của các đặc trưng nhìn lại quá khứ, không phải lỗi của mô hình.*

Nhóm FN là những người vỡ nợ "trông lành mạnh" — không có lịch sử trễ hạn, thu nhập tương đối ổn định. Mô hình không cảnh báo được là đúng xét trên bộ đặc trưng hiện có; nhiều khả năng những người này vỡ nợ do sự kiện bất ngờ (mất việc, bệnh tật) không để lại dấu vết trong lịch sử tín dụng. Đây là giới hạn vốn có của bộ đặc trưng nhìn lại quá khứ, không phải lỗi của mô hình.

### 4.7.3 Profile của False Positives

FP là 1.195 khách hàng tốt bị mô hình từ chối nhầm (5,7% tổng không vỡ nợ):

**Bảng 4.2:** So sánh profile trung vị giữa nhóm FP và TN

| Đặc trưng | FP trung vị | TN trung vị | Tỷ lệ FP/TN |
|---------|-----------|-----------|------------|
| RevolvingUtilization | **0,958** | 0,118 | 8,1× |
| TotalDelinquencyScore | **4,0** | 0,0 | >>1 |
| FinancialStressIndex | **2,856** | 0,0 | >>1 |
| MonthlyIncome | $3.610 | $4.505 | 0,80 |

Nhóm FP là những người có hành vi tài chính *trông giống* người sẽ vỡ nợ — utilization cao (gần đầy hạn mức), có vài lần trễ hạn nhỏ — nhưng thực tế vẫn trả được nợ. Nhóm này thường là khách hàng trẻ hoặc người dùng thẻ tín dụng tích cực (utilization median 0,958), thu nhập thấp hơn TN nhưng đủ khả năng trả nợ nhờ kỷ luật tài chính mà credit history không phản ánh được.

Hệ quả: 1.195 từ chối nhầm × $500 opportunity cost = **$597.500** doanh thu bị bỏ lỡ. Đây chính là nhóm được hưởng lợi nhất từ **alternative data** — lịch sử thanh toán tiện ích, hành vi mobile payment — như đề xuất trong mục 6.3.

### 4.7.4 Tối ưu Ngưỡng — F-beta Score

**Lý thuyết:** Với $\beta = 2$ (Recall ưu tiên gấp 4 lần Precision):

$$F_2 = 5 \cdot \frac{P \cdot R}{4P + R}$$

Tìm ngưỡng tối ưu trên tập kiểm định:

$$t^* = \arg\max_{t} F_2(t) = 0,625$$

**Kết quả so sánh trên tập kiểm tra:**

| Ngưỡng | F1 | F2 | Precision | Recall | Chi phí ước tính (triệu USD) |
|-----------|----|----|-----------|--------|-----------|
| 0,50 | 0,346 | 0,518 | 0,222 | 0,775 | 5,84 |
| **0,625** (F2 opt) | 0,414 | **0,537** | 0,299 | **0,669** | 6,78 |
| 0,77 (F1 opt) | **0,447** | 0,486 | **0,394** | 0,516 | 8,79 |

**Phân tích chi phí kinh doanh** (giả định: FN=$11.250/case, FP=$500/case, từ loan $15.000 × LGD 75%):

Ngưỡng cơ sở t = 0,5 có tổng chi phí thấp nhất (5,84 triệu USD) vì FN đắt gấp 22 lần FP; nhưng tạo ~4.000 FP — khó vận hành. F2-optimal threshold (0,625) không phải ngưỡng tối ưu chi phí mà là điểm thỏa hiệp ưu tiên Recall: giảm 2 triệu USD so với ngưỡng F1-optimal (0,77) và tăng Recall 15,3 điểm phần trăm, với FP ở mức kiểm soát được (~2.350).

![F1, F2, Precision và Recall theo ngưỡng — điểm tối ưu F2 tại t=0,625](../reports/fig_25_threshold_optimization.png)
*Hình 4.5: Ngưỡng F2-tối ưu t = 0,625 (đường đứt) là điểm thỏa hiệp giữa t = 0,5 (FN thấp nhất, ~4.000 FP) và t = 0,77 (FP thấp, bỏ sót nhiều defaults) — tăng Recall từ 51,6% lên 66,9% và giảm chi phí 2 triệu USD so với ngưỡng F1 t = 0,77.*

**Khuyến nghị deployment:** Sử dụng $t=0,625$ cho ngân hàng thương mại bảo thủ.

### 4.7.5 Learning Curve — Chẩn đoán Bias-Variance

| N (training size) | Train AUC | Val AUC | Gap |
|------------------|-----------|---------|-----|
| 7.000 (10%) | 0,9596 | 0,8427 | 0,1169 |
| 21.000 (30%) | 0,9129 | 0,8582 | 0,0547 |
| 45.500 (65%) | 0,8926 | 0,8624 | 0,0302 |
| 70.000 (100%) | 0,8847 | 0,8642 | **0,0205** |

> *Chú thích: Learning curve chạy trên subsample 70.000 mẫu (67% tập huấn luyện 104.999) để giảm thời gian tính toán — tỷ lệ % là tương đối so với 70.000 này. Kết luận về hội tụ (~45.000 mẫu) vẫn hợp lệ vì đây là phần lớn dữ liệu huấn luyện.*

Khoảng cách cuối (Final gap) = 0,021 < 0,03 → **"Khớp tốt"**: không overfitting nghiêm trọng. AUC tập kiểm định không còn tăng đáng kể sau N≈45.000 → thêm dữ liệu ít giá trị. Giới hạn nằm ở chất lượng đặc trưng, không phải số lượng dữ liệu.

![Learning curve XGBoost — train AUC và val AUC theo kích thước tập huấn luyện](../reports/fig_29_learning_curve.png)
*Hình 4.6: Khoảng cách huấn luyện-kiểm định (train-val gap) thu hẹp từ 0,117 (10% dữ liệu) xuống 0,021 (100% dữ liệu) — mô hình không overfitting và đã hội tụ; AUC kiểm định không còn tăng đáng kể sau N≈45.000, giới hạn là chất lượng đặc trưng, không phải thiếu dữ liệu.*

## 4.8 Bàn luận 3 Tranh luận Lớn

### 4.8.1 Khả năng Giải thích so với Hiệu suất

Logistic Regression (AUC=0,8432) cho hệ số tuyến tính $\beta$ dễ giải thích, phù hợp hoàn toàn với yêu cầu Basel III Pillar 3. Trong bối cảnh này, XGBoost kết hợp với SHAP là lựa chọn hợp lý: hiệu suất cao hơn Logistic Regression, trong khi TreeExplainer vẫn cho phép giải thích từng dự đoán cụ thể. Biểu đồ SHAP waterfall (Hình 4.8) có thể trình bày trực tiếp cho khách hàng bị từ chối: "Xác suất vỡ nợ của anh là 78% vì: điểm lịch sử trả nợ = 12 (+0,45), tỷ lệ sử dụng tín dụng = 95% (+0,38)...". Mức độ giải thích này phù hợp với yêu cầu về khả năng giải thích trong quản trị rủi ro tín dụng.

### 4.8.2 Hướng Tiếp cận Dữ liệu so với Mô hình

Kết quả: Hướng tiếp cận Data-centric (Feature Engineering có chủ đích) kết hợp hướng tiếp cận Model-centric (XGBoost) cho kết quả tốt nhất. Feature Engineering cung cấp inductive bias phù hợp lĩnh vực tài chính giúp mô hình hội tụ nhanh hơn và tốt hơn với cùng lượng dữ liệu.

**Bằng chứng từ SHAP:** 2 trong top-3 đặc trưng hàng đầu là đặc trưng được tạo — chứng minh Feature Engineering không thừa, ngay cả với mô hình phi tuyến mạnh như XGBoost.

### 4.8.3 Accuracy so với Recall — Chiến lược Ngưỡng

Accuracy 93,3% của mô hình "đoán tất cả 0" cho thấy Accuracy không phù hợp để đánh giá với dữ liệu mất cân bằng. Thay vào đó:

- **AUC-ROC = 0,8714:** 87% khả năng xếp hạng đúng (người vỡ nợ có score cao hơn người không vỡ nợ)
- **F1 vs F2:** F1 bình đẳng Precision và Recall không phù hợp tín dụng. F2 (β=2) phản ánh thực tế FN:FP cost = 22:1
- **Ngưỡng không cố định ở 0,5:** Tất cả 4 mô hình đều có ngưỡng tối ưu > 0,5 (từ 0,62 đến 0,80), hệ quả tự nhiên của `class_weight='balanced'` dịch chuyển Bayesian prior.

---

## 4.9 Probability Calibration

### 4.9.1 Động cơ

AUC-ROC đo **khả năng xếp hạng** (chất lượng ranking), không đảm bảo đầu ra xác suất của mô hình là **được hiệu chỉnh tốt**. Một mô hình được hiệu chỉnh tốt cần thỏa mãn: trong số các hồ sơ được dự báo $\hat{p} \approx 0{,}70$, khoảng 70% thực sự vỡ nợ. Điều này quan trọng vì:
1. Dashboard hiển thị "P(default) = 70%" cho nhân viên tín dụng — họ cần hiểu đúng ý nghĩa xác suất tuyệt đối này.
2. Ngưỡng triển khai $t = 0{,}625$ giả định $\hat{p}$ là ước lượng xác suất có ý nghĩa thực sự.

### 4.9.2 Brier Score và Brier Skill Score

Brier Score là mean squared error trên xác suất dự báo:

$$\text{BS} = \frac{1}{N}\sum_{i=1}^{N}(\hat{p}_i - y_i)^2$$

Nhỏ hơn = tốt hơn. Baseline (mô hình "dự báo tỷ lệ phổ biến" $\hat{p} \equiv 0{,}0668$ cho mọi hồ sơ): $\text{BS}_{\text{ref}} = 0{,}0668 \times (1-0{,}0668) \approx 0{,}0623$.

Brier Skill Score (BSS) chuẩn hóa so với baseline:

$$\text{BSS} = 1 - \frac{\text{BS}}{\text{BS}_{\text{ref}}} \in (-\infty, 1]$$

BSS > 0: mô hình tốt hơn baseline; BSS = 1: hiệu chỉnh hoàn hảo.

**Kết quả thực nghiệm** trên tập kiểm tra 22.500 hồ sơ (chi tiết: `reports/addendum_results.md`):

| Mô hình | Brier Score (↓) | Brier Skill Score (↑) | ECE (↓) |
|-------|----------------:|----------------------:|--------:|
| Logistic Regression | 0,1573 | −1,5214 | 0,3764 |
| Decision Tree       | 0,1458 | −1,3380 | 0,3695 |
| Random Forest       | 0,1189 | −0,9058 | 0,3286 |
| XGBoost             | 0,1388 | −1,2253 | 0,3590 |

*Baseline Brier Score (dự báo tỷ lệ phổ biến 6,68%): 0,0624.*

**Phát hiện:** Cả 4 mô hình đều có **BSS < 0** — tức **xấu hơn baseline "dự báo tỷ lệ phổ biến"** về Brier Score. Đây không phải nghịch lý mà là hệ quả trực tiếp của điều chỉnh trọng số (`class_weight='balanced'` cho LR/DT/RF, `scale_pos_weight=13.96` cho XGBoost): các mô hình được tối ưu cho khả năng phân biệt (AUC), được "phép" ước tính quá cao xác suất vỡ nợ lên ~14× để bù mất cân bằng, nên xác suất đầu ra không còn được hiệu chỉnh tốt. ECE ≈ 0,33–0,38 cho thấy độ chênh trung bình giữa $\hat{p}$ và tỷ lệ thực tế trong từng bin lên tới ~33–38 điểm phần trăm — rất lớn.

Trong nhóm này, **Random Forest có Brier Score và ECE tốt nhất** (0,1189 và 0,3286) nhờ averaging nhiều cây làm regularize xác suất, trong khi LR và DT dễ output xác suất cực đoan (gần 0 hoặc gần 1). XGBoost xếp giữa: tốt hơn LR/DT nhưng kém RF do `scale_pos_weight` đẩy mạnh xác suất dương.

### 4.9.3 Reliability Diagram

Reliability diagram (Hình 4.9) chia các dự đoán (predictions) thành 10 khoảng (bin) theo $\hat{p}$, vẽ giá trị trung bình $\hat{p}$ ở trục hoành và tỷ lệ vỡ nợ thực tế (fraction of positives) trong bin ở trục tung. Đường chéo 45° là calibration hoàn hảo; nằm **dưới** đường chéo nghĩa là mô hình **ước tính quá cao** xác suất vỡ nợ — đánh giá cao quá mức nguy cơ thực tế.

**Quan sát từ thực nghiệm:** Cả 4 đường cong đều nằm ở **bên dưới đường chéo 45°** — tức tất cả mô hình đều có xu hướng ước tính quá cao xác suất vỡ nợ. Đây là kết quả trực tiếp của xử lý mất cân bằng (imbalance): với tỷ lệ phổ biến (prevalence) thực tế chỉ 6,68%, các kỹ thuật `class_weight='balanced'` (LR/DT/RF) và `scale_pos_weight=13,96` (XGB) đều ngầm giả định prior 50/50, đẩy $\hat{p}$ lên cao hơn nhiều so với tỷ lệ vỡ nợ thực. Logistic Regression với regularization rất nặng (C=0,001) bị "co" hệ số gần 0, output nằm gần ngưỡng quyết định khoảng 0,5 cho phần lớn hồ sơ — đó là lý do LR có ECE cao nhất (0,3764) bất chấp nền tảng MLE probabilistic. Random Forest có calibration tốt nhất (ECE=0,3286) nhờ averaging nhiều cây làm trung hòa các xác suất cực đoan.

![Reliability diagram và Brier Score của 4 mô hình trên tập kiểm tra](../reports/fig_31_calibration.png)
*Hình 4.9: Reliability diagram (trái) và Brier Score (phải). Tất cả 4 đường nằm dưới đường chéo — ước tính quá cao xác suất, hệ quả của điều chỉnh trọng số để xử lý mất cân bằng. RF gần đường chéo nhất (ECE=0,3286), LR xa nhất (ECE=0,3764). Đường đứt đỏ ở biểu đồ phải là baseline Brier=0,0624 (dự báo tỷ lệ phổ biến) — không mô hình nào đạt mức này, nghĩa là khả năng phân biệt đánh đổi hiệu chỉnh xác suất.*

### 4.9.4 Post-hoc Calibration và Hạn chế

**Hiệu chỉnh Platt (Platt Scaling)** khớp một Logistic Regression trên điểm số đầu ra của mô hình trên tập kiểm định:

$$\hat{p}_{\text{calibrated}} = \sigma(a \cdot \hat{p}_{\text{raw}} + b), \quad a, b \text{ khớp trên tập kiểm định}$$

Phép này không thay đổi ranking (AUC không đổi) nhưng cải thiện Brier Score và ECE.

Nhìn chung, mô hình chưa được hiệu chỉnh xác suất. Với mục tiêu chính là phân loại (chất lượng xếp hạng) và ngưỡng t=0,625 được chọn theo F2-score chứ không phải theo xác suất tuyệt đối, đây là hạn chế có thể chấp nhận trong phạm vi nghiên cứu này. Nếu triển khai thực tế — khi nhân viên tín dụng cần giải thích cho khách hàng rằng "xác suất vỡ nợ của anh là 70%" với ý nghĩa thực sự là 70% — thì cần thực hiện hiệu chỉnh Platt trên tập kiểm định trước.

---

# CHƯƠNG 5: SẢN PHẨM — STREAMLIT DASHBOARD

## 5.1 Kiến trúc Hệ thống

```
┌──────────────────────────────────────────────────────────────┐
│                    app/app.py (Streamlit)                     │
│                                                               │
│  ┌─────────────────────┐    ┌──────────────────────────────┐ │
│  │   INPUT PANEL       │    │     OUTPUT PANEL             │ │
│  │                     │    │                              │ │
│  │  10 đặc trưng thô   │    │  Risk Tier (color)           │ │
│  │  (form widgets)     │───▶│  P(default) = XX%            │ │
│  │                     │    │  Decision: APPROVE/REJECT    │ │
│  └─────────────────────┘    │  Top 3 risk factors          │ │
│           │                 │  SHAP waterfall chart        │ │
│           ▼                 └──────────────────────────────┘ │
│  ┌─────────────────────┐                                     │
│  │  Feature Engineering  │                                     │
│  │  (4 tự tính)        │                                     │
│  └─────────────────────┘                                     │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────────┐    ┌──────────────────────────────┐ │
│  │  XGBoost            │    │  SHAP TreeExplainer          │ │
│  │  best_model.pkl     │    │  (cached @st.cache_resource) │ │
│  │  (@st.cache_resource│    │                              │ │
│  └─────────────────────┘    └──────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

**Luồng xử lý:** Người dùng nhập 10 đặc trưng gốc → Ứng dụng tự tính 4 đặc trưng mới → Xây dựng DataFrame 14 đặc trưng theo đúng thứ tự → `model.predict_proba(X_df)[0,1]` → `shap.TreeExplainer(model)(X_df)` → Hiển thị mức rủi ro + biểu đồ SHAP waterfall.

**Caching strategy:**
- `@st.cache_resource` cho `load_model()` và `load_explainer()` — load một lần, dùng nhiều lần
- SHAP waterfall figure tạo mới mỗi dự đoán (không cache — biểu đồ phụ thuộc đầu vào)
- `plt.close(fig)` sau `st.pyplot(fig)` — tránh rò rỉ bộ nhớ

## 5.2 Đặc trưng Đầu vào

**10 đặc trưng gốc** người dùng nhập thủ công trong biểu mẫu (theo đúng thứ tự trong `app.py`):

| # | Tên đặc trưng | Ý nghĩa tiếng Việt | Phạm vi | Kiểu |
|---|--------------|-------------------|---------|------|
| 1 | `RevolvingUtilizationOfUnsecuredLines` | Tỷ lệ sử dụng hạn mức tín dụng | 0,0 – 10,0 (>1 = vượt hạn mức) | Thực |
| 2 | `age` | Tuổi người vay | 18 – 100 | Nguyên |
| 3 | `NumberOfTime30-59DaysPastDueNotWorse` | Số lần trễ hạn 30–59 ngày | 0 – 20 | Nguyên |
| 4 | `DebtRatio` | Tỷ lệ nợ / thu nhập hàng tháng | 0,0 – 5,0 | Thực |
| 5 | `MonthlyIncome` | Thu nhập hàng tháng (USD) | 0 – 50.000 | Nguyên |
| 6 | `NumberOfOpenCreditLinesAndLoans` | Số tài khoản tín dụng đang mở | 0 – 50 | Nguyên |
| 7 | `NumberOfTimes90DaysLate` | Số lần trễ hạn > 90 ngày | 0 – 20 | Nguyên |
| 8 | `NumberRealEstateLoansOrLines` | Số khoản vay bất động sản | 0 – 20 | Nguyên |
| 9 | `NumberOfTime60-89DaysPastDueNotWorse` | Số lần trễ hạn 60–89 ngày | 0 – 20 | Nguyên |
| 10 | `NumberOfDependents` | Số người phụ thuộc | 0 – 20 | Nguyên |

**4 đặc trưng tự động tính** từ đầu vào trên (người dùng không cần nhập):

| Đặc trưng | Công thức | Ý nghĩa |
|-----------|---------|---------|
| `TotalDelinquencyScore` | $3\times(90+) + 2\times(60\text{–}89) + (30\text{–}59)$ | Điểm tổng hợp trễ hạn có trọng số |
| `FinancialStressIndex` | $\text{RevUtil} \times \text{TotalDelinquencyScore}$ | Tương tác hạn mức × trễ hạn |
| `DebtToIncomeRatio` (AbsoluteDebt) | $\text{DebtRatio} \times \text{MonthlyIncome}$ | Dư nợ tuyệt đối (USD/tháng) |
| `DelinquencyTrend` | $(30\text{–}59) - (90+)$ | Xu hướng cải thiện trễ hạn |

## 5.3 Tính năng chính

**Phân loại mức rủi ro** (căn chỉnh với ngưỡng triển khai):

| P(vỡ nợ) | Mức rủi ro | Màu | Quyết định |
|-----------|-----------|-----|------------|
| < 10% | 🟢 RỦI RO THẤP | Xanh lá | ✅ CHẤP THUẬN |
| 10–30% | 🟡 RỦI RO TRUNG BÌNH | Vàng | ✅ CHẤP THUẬN |
| 30–62,5% | 🟠 RỦI RO CAO | Cam | ✅ CHẤP THUẬN |
| ≥ 62,5% | 🔴 RỦI RO RẤT CAO | Đỏ | ❌ TỪ CHỐI |

**Biểu đồ SHAP Waterfall:** Hiển thị top 10 đặc trưng đóng góp vào quyết định, màu đỏ = tăng rủi ro, màu xanh = giảm rủi ro. Xác suất cơ sở và xác suất cuối hiển thị rõ ràng.

![SHAP waterfall cho 3 dự đoán cụ thể — khách hàng thấp/cao/biên](../reports/fig_28_shap_waterfall.png)
*Hình 5.1: Ba hồ sơ khách hàng: (trái) an toàn — age và MonthlyIncome kéo xuống; (giữa) rủi ro cao — TotalDelinquencyScore và FinancialStressIndex đẩy mạnh lên; (phải) biên — tín hiệu hỗn hợp, cần xem xét thêm. Mỗi biểu đồ là lời giải thích có thể trình bày trực tiếp khi từ chối hồ sơ vay.*

## 5.4 Các Tình huống Minh họa

### Hồ sơ 1 — Khách hàng an toàn
**Input:** Tuổi=55, RevUtil=10%, thu nhập=$8.000/tháng, 0 lần trễ hạn, DebtRatio=0,2  
**Output:** P(vỡ nợ)=6,7% → 🟢 **RỦI RO THẤP** | ✅ CHẤP THUẬN  
**SHAP:** age (−0,18), MonthlyIncome (−0,15) giảm rủi ro; RevUtil gần 0 không đóng góp đáng kể.

### Hồ sơ 2 — Khách hàng rủi ro cao
**Input:** Tuổi=28, RevUtil=95%, thu nhập=$2.500/tháng, 3 lần trễ > 90 ngày, DebtRatio=1,5  
**Output:** P(vỡ nợ)=97,8% → 🔴 **RỦI RO RẤT CAO** | ❌ TỪ CHỐI  
**SHAP:** TotalDelinquencyScore (+0,52), FinancialStressIndex (+0,48), RevUtil (+0,35) đẩy dự đoán về phía vỡ nợ.

### Hồ sơ 3 — Khách hàng cận ngưỡng
**Input:** Tuổi=40, RevUtil=60%, thu nhập=$4.000/tháng, 1 lần trễ 30–59 ngày, DebtRatio=0,5  
**Output:** P(vỡ nợ)=41,5% → 🟠 **RỦI RO CAO** | ✅ CHẤP THUẬN (< 0,625)  
Hồ sơ nằm trong vùng xám — chuyên viên tín dụng nên yêu cầu tài liệu bổ sung trước khi quyết định.

---

# CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 6.1 Tóm tắt Kết quả

Bốn mô hình được xây dựng và đánh giá theo thứ tự tăng dần độ phức tạp. Logistic Regression (AUC=0,8432) làm baseline, có khả năng giải thích tốt nhất qua hệ số hồi quy và odds ratio. Decision Tree (AUC=0,8579) đạt hiệu suất cao hơn nhưng variance lớn hơn. Random Forest (AUC=0,8703) ổn định hơn nhờ ensemble. XGBoost (AUC=0,8714) cho kết quả tốt nhất, vượt mục tiêu đặt ra.

Feature Engineering theo lĩnh vực tài chính cho thấy hiệu quả rõ ràng: `FinancialStressIndex` và `TotalDelinquencyScore` — cả hai đều được tạo thủ công — chiếm vị trí #1 và #3 trong SHAP ranking. Điều đáng chú ý là mô hình L1 (Logistic Regression với C=0,001) tự động triệt tiêu hệ số của `NumberOfTimes90DaysLate`, cho thấy `TotalDelinquencyScore` đã mã hóa đủ thông tin từ đặc trưng đó.

Phân tích sai số cho thấy 48,4% người vỡ nợ thực sự không bị phát hiện tại ngưỡng mặc định — đây là giới hạn vốn có của bộ đặc trưng nhìn lại quá khứ, không phải do mô hình yếu. Tối ưu ngưỡng theo F2-score (t=0,625) tăng Recall thêm 15,3 điểm phần trăm, giảm chi phí ước tính khoảng $2 triệu trên tập kiểm tra 22.500 hồ sơ.

Sản phẩm cuối là ứng dụng Streamlit tương tác, cho phép nhập hồ sơ và xem giải thích SHAP cho từng dự đoán. Chạy bằng `streamlit run app/app.py`.

**Những thách thức kỹ thuật đã giải quyết:**

- **Mất cân bằng dữ liệu 1:14:** Áp dụng `class_weight='balanced'`/`scale_pos_weight` kết hợp tối ưu ngưỡng theo $F_2$-score thay vì Accuracy, tránh được hiện tượng mô hình thiên lệch về lớp đa số.
- **Đa cộng tuyến:** Ba biến delinquency có VIF = ∞ được nén thành `TotalDelinquencyScore` duy nhất, vừa giảm đa cộng tuyến vừa tăng sức mạnh dự báo (Spearman ρ từ 0,342 lên 0,345).
- **Minh bạch quyết định bằng SHAP:** SHAP TreeExplainer cung cấp giải thích tại mức từng hồ sơ (local explanation) và toàn bộ bộ dữ liệu (global explanation), đáp ứng yêu cầu explainability theo Basel III Pillar 3 và GDPR Article 22.

## 6.2 Hạn chế

**1. Dữ liệu nhìn lại quá khứ:** Tất cả 10 đặc trưng gốc phản ánh lịch sử tín dụng, không phản ánh được các sự kiện đột ngột (mất việc, bệnh tật, ly hôn). Đây là lý do 48,4% trường hợp vỡ nợ thực sự bị bỏ sót — không phải mô hình sai mà đặc trưng không đủ.

**2. Dữ liệu tĩnh (ảnh chụp tại một thời điểm):** Bộ dữ liệu chỉ có một thời điểm, không có dữ liệu chuỗi thời gian. Xu hướng thay đổi hành vi (đang xấu dần hay cải thiện) chỉ được đại diện gián tiếp qua `DelinquencyTrend`.

**3. Bộ dữ liệu giới hạn địa lý:** Dữ liệu từ thị trường tiêu dùng Mỹ. Áp dụng trực tiếp cho thị trường Việt Nam cần huấn luyện lại với dữ liệu địa phương và điều chỉnh ngưỡng chi phí.

**4. Hiệu chỉnh xác suất:** Mô hình được tối ưu cho khả năng phân biệt (AUC) nhưng chưa hiệu chỉnh để đầu ra xác suất có ý nghĩa tuyệt đối (tức P(vỡ nợ)=0,7 thực sự nghĩa là 70%). Hiệu chỉnh Platt hoặc hồi quy isotonic có thể cải thiện.

## 6.3 Hướng Phát triển

- **Dữ liệu thời gian:** Thu thập lịch sử thanh toán hàng tháng → LSTM/GRU để nắm bắt xu hướng thay đổi hành vi tín dụng.
- **Ensemble stacking:** Kết hợp dự đoán LR + RF + XGB, học mô hình meta → ước tính AUC cải thiện 0,003–0,005.
- **Giám sát ngưỡng phân loại:** Phát hiện trôi dạt mô hình — ngưỡng tối ưu có thể thay đổi theo biến động kinh tế vĩ mô.
- **Alternative data:** Dữ liệu hành vi (thanh toán di động, hóa đơn tiện ích) → tăng coverage cho nhóm khách hàng ít lịch sử tín dụng.
- **Kiểm toán công bằng:** Kiểm tra mô hình có phân biệt đối xử theo độ tuổi, giới tính không (theo Equal Credit Opportunity Act).
- **Hạ tầng triển khai:** Phục vụ mô hình qua REST API, giám sát hiệu suất với Evidently AI.
- **Graph Neural Networks:** Mô hình hóa mối quan hệ giữa người vay (mạng lưới bảo lãnh) → rủi ro xã hội.
- **Causal inference:** Phân biệt tương quan và quan hệ nhân quả trong các yếu tố tín dụng — quan trọng trong xây dựng chính sách.
- **Federated learning:** Huấn luyện mô hình trên dữ liệu phân tán giữa nhiều tổ chức tín dụng mà không chia sẻ dữ liệu thô (tuân thủ GDPR).

---

# TÀI LIỆU THAM KHẢO

[1] Ngân hàng Nhà nước Việt Nam, "Báo cáo thường niên 2023," Hà Nội, 2024.

[2] E. I. Altman, "Financial ratios, discriminant analysis and the prediction of corporate bankruptcy," *Journal of Finance*, vol. 23, no. 4, pp. 589–609, Sep. 1968.

[3] D. J. Hand and W. E. Henley, "Statistical classification methods in consumer credit scoring: A review," *Journal of the Royal Statistical Society: Series A*, vol. 160, no. 3, pp. 523–541, 1997.

[4] S. Lessmann, B. Baesens, H.-V. Seow, and L. C. Thomas, "Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research," *European Journal of Operational Research*, vol. 247, no. 1, pp. 124–136, 2015.

[5] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NIPS)*, vol. 30, 2017.

[6] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, San Francisco, CA, 2016, pp. 785–794.

[7] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, Oct. 2001.

[8] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002.

[9] M. Bucker, G. van den Heuvel, W. Gebert, and J. Lorenz, "Transparency, auditability, and explainability of machine learning models in credit decisions," *Journal of the Operational Research Society*, vol. 73, no. 1, pp. 70–90, 2022.

[10] A. Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 3rd ed. Sebastopol, CA: O'Reilly Media, 2022.

[11] Basel Committee on Banking Supervision, "Revisions to the Standardised Approach for credit risk," Bank for International Settlements, Basel, Switzerland, 2017.

[12] S. M. Lundberg et al., "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, no. 1, pp. 56–67, Jan. 2020.

[13] E. R. DeLong, D. M. DeLong, and D. L. Clarke-Pearson, "Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach," *Biometrics*, vol. 44, no. 3, pp. 837–845, Sep. 1988.

---

# PHỤ LỤC

## Phụ lục A — Cấu trúc Thư mục

```
loan-default-prediction/
├── app/app.py                   ← Streamlit dashboard
├── data/
│   ├── raw/cs-training.csv     ← Bộ dữ liệu gốc (149.999 rows)
│   ├── processed/              ← Data sau preprocessing
│   └── splits/                 ← train/val/test.csv
├── models/
│   ├── best_model.pkl           ← XGBClassifier (340 KB)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb        ← 4 models, CV, comparison
│   └── 04_Analysis.ipynb        ← Error analysis, SHAP, learning curve
├── reports/                     ← 30 figures + 4 markdown reports
├── src/
│   ├── models.py               ← Training wrappers
│   ├── evaluation.py           ← Metrics, plots
│   ├── preprocessing.py        ← Cleaning, imputation, pipeline
│   └── features.py             ← Hàm Feature Engineering
└── final_report/
    └── bao_cao_chinh.md        ← File này
```

## Phụ lục B — Công thức Tổng hợp

| Metric | Công thức | Ý nghĩa trong tín dụng |
|--------|-----------|----------------------|
| AUC-ROC | $P(f(x^+) > f(x^-))$ | Khả năng phân biệt vỡ nợ/không |
| F1 | $2PR/(P+R)$ | Cân bằng Precision-Recall |
| F2 | $5PR/(4P+R)$ | Recall ưu tiên (FN cost >> FP cost) |
| Gini (Gini coefficient) | $2 \cdot \text{AUC} - 1$ | Phổ biến trong báo cáo rủi ro ngân hàng |
| KS statistic | $\max_t |TPR(t) - FPR(t)|$ | Điểm phân tách tốt nhất |

**Gini coefficient** của mô hình tốt nhất: $2 \times 0,8714 - 1 = 0,7428$ (thường xếp loại "good" nếu > 0,6 theo tiêu chuẩn ngành).

## Phụ lục C — Cài đặt Môi trường

```bash
# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy notebook (từ project root)
jupyter notebook

# Chạy Streamlit app
streamlit run app/app.py
```

**Phiên bản chính:** Python 3.14.2, pandas 3.0.2, scikit-learn 1.8.0, xgboost 3.2.0, shap 0.51.0, streamlit 1.x

## Phụ lục D — Hướng dẫn Reproduce Kết quả

Tất cả thực nghiệm sử dụng `random_state=42` tại mọi điểm có randomness:
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- `RandomizedSearchCV(random_state=42)`
- `XGBClassifier(random_state=42)`
- `np.random.seed(42)` trong mỗi notebook

Với cùng `random_state`, kết quả có thể reproduce chính xác trên cùng phiên bản thư viện.
