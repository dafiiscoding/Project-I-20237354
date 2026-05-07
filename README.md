# Loan Default Prediction

Dự báo xác suất vỡ nợ tín dụng bằng Machine Learning trên dataset [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle, 149.999 hồ sơ vay dùng cho phân tích — tập gốc 150.000 trừ 1 dòng tuổi = 0). Bao gồm toàn bộ pipeline từ EDA, tiền xử lý, huấn luyện 4 mô hình, phân tích sai số, đến Streamlit dashboard cho phép dự báo từng hồ sơ và đánh giá theo lô CSV/XLSX với giải thích SHAP.

## Kết quả

| Mô hình | AUC-ROC | F1 | Precision | Recall | t\* (F1-opt) |
|---------|--------:|----:|----------:|-------:|-------------:|
| Logistic Regression | 0,8432 | 0,434 | 0,388 | 0,493 | 0,66 |
| Decision Tree | 0,8579 | 0,431 | 0,399 | 0,469 | 0,80 |
| Random Forest | 0,8703 | 0,444 | 0,387 | 0,521 | 0,72 |
| **XGBoost** | **0,8714** | **0,447** | **0,394** | **0,516** | **0,77** |

*Bảng trên đo ở ngưỡng tối ưu F1 của từng mô hình.* Ngưỡng triển khai thực tế **t = 0,625** được chọn bằng cách tối ưu F2-score (ưu tiên Recall vì bỏ sót người vỡ nợ tốn kém hơn từ chối nhầm). Tại ngưỡng triển khai, XGBoost đạt **F2 = 0,537, Recall = 0,669, Precision = 0,299** — Recall tăng 15,3 điểm so với ngưỡng F1-opt, ước tính tiết kiệm ~$2M/năm trên 22.500 hồ sơ.

Theo SHAP, hai trong ba features quan trọng nhất là features tự tạo (`FinancialStressIndex`, `TotalDelinquencyScore`), không phải features gốc từ dataset.

## Dataset

Dataset không được đính kèm do điều khoản Kaggle. Tải về tại:

```bash
kaggle competitions download -c GiveMeSomeCredit
# hoặc tải thủ công từ https://www.kaggle.com/c/GiveMeSomeCredit/data
```

Đặt file `cs-training.csv` vào `data/raw/`.

> Lưu ý tái lập: `data/` và các model phụ (`model_lr/dt/rf.pkl`) không track trong git do dung lượng/giấy phép dữ liệu. Chạy notebook/script theo thứ tự bên dưới để tái tạo. Model chính `models/best_model.pkl` được giữ trong repo để chạy dashboard nhanh.

**Quy ước số liệu trong báo cáo:**
- **150.000** — số dòng raw từ Kaggle (`cs-training.csv`).
- **149.999** — số hồ sơ sau khi loại 1 dòng `age = 0` (vô nghĩa về domain), dùng cho toàn bộ phân tích.
- Phân chia: train 104.999 / validation 22.500 / test 22.500. Mô hình **học** trên 104.999 hồ sơ; AUC = 0,8714 báo cáo trên 22.500 hồ sơ test.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy

Chạy các notebook theo thứ tự trong `notebooks/`:

```
01_EDA.ipynb            — phân tích phân phối, outlier, correlation
02_Preprocessing.ipynb  — cleaning, KNN imputation, feature engineering → sinh data/splits/
03_Modeling.ipynb       — train 4 models, CV, SHAP → sinh models/best_model.pkl
04_Analysis.ipynb       — error analysis, SHAP, threshold optimization
```

Sau đó sinh kết quả DeLong test và calibration analysis:

Kết quả kiểm định bổ sung đã được lưu sẵn trong `reports/addendum_results.md` và các hình `reports/fig_*.png`.
Các script sinh lại hình/model phụ là công cụ nội bộ, không đưa lên GitHub trong bản nộp gọn.

Chạy dashboard:

```bash
streamlit run app/app.py
```

Tab đánh giá theo lô hỗ trợ CSV/XLSX, cho phép chọn ngưỡng quyết định, đánh giá danh mục khách hàng và tải file kết quả có `probability`, `risk_tier`, `decision`, `decision_threshold`.

## Cấu trúc

```
├── data/               — raw (cs-training.csv), processed, splits (train/val/test)
├── notebooks/          — 4 notebooks chính + 2 script bổ sung
├── src/                — data_loader, preprocessing, features, models, evaluation, plot_style
├── reports/            — 34+ figures PNG + addendum_results.md
│   └── visual_summary/ — ảnh infographic (pipeline, KPI, what-if scenarios)
├── final_report/       — báo cáo chính (md, tex, pdf) + tóm tắt trực quan
├── models/             — best_model.pkl (XGBoost, ~340KB); model_lr/dt/rf.pkl sinh lại bằng script
├── app/                — Streamlit dashboard (multi-tab: đơn lẻ + theo lô)
└── requirements.txt
```

---

## Hướng dẫn đọc kho mã

### Theo persona (đọc file nào trước?)

**Persona 1 — Giảng viên / người chấm (15 phút)**
1. `README.md` — bạn đang đọc (kết quả + cài đặt)
2. `final_report/tom_tat_truc_quan.md` — tóm tắt 1 trang với ảnh trực quan
3. `final_report/bao_cao_chinh.md` §Tóm tắt, §1, §4.6, §6 — đọc thẳng vào kết quả

**Persona 2 — Người phản biện kỹ thuật (60 phút)**
1. `final_report/bao_cao_chinh.md` — báo cáo chính, tập trung vào kết quả và quyết định kỹ thuật
2. `notebooks/04_Analysis.ipynb` — error analysis tương tác
3. `reports/addendum_results.md` — DeLong test + calibration metrics
4. `src/` — kiểm tra trực tiếp implementation

**Persona 3 — Dev muốn tái hiện (2 giờ)**
1. `README.md` — cài đặt + thứ tự chạy notebooks
2. `notebooks/01_EDA.ipynb` → `02_Preprocessing.ipynb` → `03_Modeling.ipynb` → `04_Analysis.ipynb`
3. `src/` đọc theo thứ tự: `data_loader` → `preprocessing` → `features` → `models` → `evaluation`
4. `app/app.py` — entry point Streamlit dashboard

**Persona 4 — Người đọc nhanh / non-tech (5 phút)**
1. `final_report/tom_tat_truc_quan.md` — **DUY NHẤT cần đọc**

### Bản đồ file quan trọng

| File / Thư mục | Mục đích | Persona ưu tiên |
|---|---|---|
| `data/raw/cs-training.csv` | Dataset gốc Kaggle (phải tải riêng) | Dev |
| `notebooks/01–04_*.ipynb` | Pipeline phân tích + huấn luyện tương tác | Dev |
| `reports/fig_*.png` | Hình đã sinh sẵn để đọc báo cáo và bảo vệ | Báo cáo |
| `src/evaluation.py` | Hàm đánh giá + plot (ROC, PR, calibration) | Dev |
| `src/plot_style.py` | Cấu hình style matplotlib cho tiếng Việt | Dev |
| `models/best_model.pkl` | Mô hình XGBoost đã huấn luyện (AUC=0.8714) | Dev / App |
| `app/app.py` | Streamlit entry point (3 tabs) | Demo |
| `final_report/bao_cao_chinh.md` | Báo cáo chính | Phản biện |
| `final_report/tom_tat_truc_quan.md` | Tóm tắt trực quan 1 trang | Tất cả |
| `reports/visual_summary/` | Infographic pipeline, KPI, what-if | Tất cả |

---

## Báo cáo

Báo cáo chính ở `final_report/bao_cao_chinh.md`, tập trung vào mục tiêu, pipeline, kết quả, dashboard và hạn chế. Các slide/ghi chú defense là tài liệu cá nhân để trình bày, không đưa lên GitHub.

Tóm tắt trực quan dành cho người đọc nhanh: `final_report/tom_tat_truc_quan.md`.

`reports/addendum_results.md` chứa kết quả kiểm định DeLong và calibration metrics (Brier Score, ECE). Lưu ý: bảng mô hình chính dùng RF từ notebook 03 với AUC=0,8703; addendum dùng RF huấn luyện lại để tái lập kiểm định DeLong (AUC=0,8671), nên ΔAUC=+0,0043 chỉ áp dụng cho cặp model trong addendum.
