# Loan Default Prediction

Dự báo xác suất vỡ nợ tín dụng bằng Machine Learning trên dataset [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle, 149.999 hồ sơ vay dùng cho phân tích — tập gốc 150.000 trừ 1 dòng tuổi = 0). Bao gồm toàn bộ pipeline từ EDA, tiền xử lý, huấn luyện 4 mô hình, phân tích sai số, đến Streamlit dashboard cho phép dự báo từng hồ sơ với giải thích SHAP.

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

```bash
python notebooks/train_supplementary_models.py   # lưu model_lr/dt/rf.pkl
python notebooks/analysis_addendum.py            # sinh fig_31, fig_32, addendum_results.md
```

Chạy dashboard:

```bash
streamlit run app/app.py
```

## Cấu trúc

```
├── data/               — raw (cs-training.csv), processed, splits (train/val/test)
├── notebooks/          — 4 notebooks chính + 2 script bổ sung
├── src/                — data_loader, preprocessing, features, models, evaluation, plot_style
├── reports/            — 34+ figures PNG + addendum_results.md
│   └── visual_summary/ — ảnh infographic (pipeline, KPI, what-if scenarios)
├── final_report/       — báo cáo chính 7 chương (md, tex, pdf) + tóm tắt trực quan
├── presentation/       — slide kỹ thuật (29 slides Marp) + slide non-tech (7 slides) + defense guide
├── models/             — best_model.pkl (XGBoost, ~340KB) + model_lr/dt/rf.pkl
├── app/                — Streamlit dashboard (multi-tab: đơn lẻ + theo lô)
└── requirements.txt
```

---

## Hướng dẫn đọc kho mã

### Theo persona (đọc file nào trước?)

**Persona 1 — Giảng viên / người chấm (15 phút)**
1. `README.md` — bạn đang đọc (kết quả + cài đặt)
2. `final_report/tom_tat_truc_quan.md` — tóm tắt 1 trang với ảnh trực quan
3. `presentation/slide_executive_summary.md` — 7 slides ngôn ngữ business
4. `final_report/bao_cao_chinh.md` §1, §4.6, §6 — đọc thẳng vào kết quả

**Persona 2 — Người phản biện kỹ thuật (60 phút)**
1. `final_report/bao_cao_chinh.md` — đầy đủ 7 chương (~800 dòng)
2. `presentation/defense_guide.md` Phần B–C — toán học chi tiết (derivation, proof)
3. `notebooks/04_Analysis.ipynb` — error analysis tương tác
4. `reports/addendum_results.md` — DeLong test + calibration metrics

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
| `notebooks/regen_figures.py` | Tái tạo toàn bộ figures với style tiếng Việt | Dev |
| `src/evaluation.py` | Hàm đánh giá + plot (ROC, PR, calibration) | Dev |
| `src/plot_style.py` | Cấu hình style matplotlib cho tiếng Việt | Dev |
| `models/best_model.pkl` | Mô hình XGBoost đã huấn luyện (AUC=0.8714) | Dev / App |
| `app/app.py` | Streamlit entry point (3 tabs) | Demo |
| `final_report/bao_cao_chinh.md` | Báo cáo chính 7 chương | Phản biện |
| `final_report/tom_tat_truc_quan.md` | Tóm tắt trực quan 1 trang | Tất cả |
| `presentation/slide.md` | 29 slides Marp — defense kỹ thuật | Phản biện |
| `presentation/slide_executive_summary.md` | 7 slides — dành cho non-tech | Giảng viên |
| `presentation/defense_guide.md` | Q&A + toán học chi tiết để chuẩn bị bảo vệ | Phản biện |
| `reports/visual_summary/` | Infographic pipeline, KPI, what-if | Tất cả |

---

## Báo cáo

Báo cáo chính ở `final_report/bao_cao_chinh.md` (~800 dòng), gồm 7 chương với derivation toán học cho từng thuật toán, phân tích thực nghiệm, thảo luận về trade-off và chiến lược threshold. Derivation chi tiết đã được chuyển sang `presentation/defense_guide.md` (Phần B–C).

Tóm tắt trực quan dành cho người đọc nhanh: `final_report/tom_tat_truc_quan.md`.

`reports/addendum_results.md` chứa kết quả kiểm định DeLong (XGB vs RF: ΔAUC=+0.0043, p<0.0001) và calibration metrics (Brier Score, ECE) cho cả 4 mô hình.
