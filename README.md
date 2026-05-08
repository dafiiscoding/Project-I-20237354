# Loan Default Prediction

Dự báo xác suất vỡ nợ tín dụng bằng Machine Learning trên dataset [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle, 149.999 hồ sơ vay dùng cho phân tích — tập gốc 150.000 trừ 1 dòng tuổi = 0). Bao gồm toàn bộ pipeline từ EDA, tiền xử lý, huấn luyện 4 mô hình, phân tích sai số, đến Streamlit dashboard cho phép dự báo từng hồ sơ với giải thích SHAP.

## Đọc nhanh trong 3 phút

1. `final_report/tom_tat_truc_quan.md` — 1 trang, đọc kết quả ngay.
2. `presentation/slide_executive_summary.md` — bản slide cho người không chuyên.
3. `app/app.py` (`streamlit run app/app.py`) — demo dự báo đơn lẻ + batch CSV/XLSX.

## Kết quả

| Mô hình | AUC-ROC | F1 | Precision | Recall | t\* (F1-opt) |
|---------|--------:|----:|----------:|-------:|-------------:|
| Logistic Regression | 0,8432 | 0,434 | 0,388 | 0,493 | 0,66 |
| Decision Tree | 0,8579 | 0,431 | 0,399 | 0,469 | 0,80 |
| Random Forest | 0,8703 | 0,444 | 0,387 | 0,521 | 0,72 |
| **XGBoost** | **0,8714** | **0,447** | **0,394** | **0,516** | **0,77** |

*Bảng trên đo ở ngưỡng tối ưu F1 của từng mô hình.* Ngưỡng triển khai thực tế **t = 0,625** được chọn bằng cách tối ưu F2-score (ưu tiên Recall vì bỏ sót người vỡ nợ tốn kém hơn từ chối nhầm). Tại ngưỡng triển khai, XGBoost đạt **F2 = 0,537, Recall = 0,669, Precision = 0,299** — Recall tăng 15,3 điểm so với ngưỡng F1-opt, ước tính giảm khoảng **2,01 triệu USD** so với ngưỡng F1-optimal trên 22.500 hồ sơ.

Theo SHAP, hai trong ba features quan trọng nhất là features tự tạo (`FinancialStressIndex`, `TotalDelinquencyScore`), không phải features gốc từ dataset.

## Dataset

Dataset không được đính kèm do điều khoản Kaggle. Tải về tại:

```bash
kaggle competitions download -c GiveMeSomeCredit
# hoặc tải thủ công từ https://www.kaggle.com/c/GiveMeSomeCredit/data
```

Đặt file `cs-training.csv`, `cs-test.csv` và `sampleEntry.csv` vào `data/raw/` nếu muốn chạy cả demo batch và thử submission Kaggle.

**Quy ước số liệu trong báo cáo:**
- **150.000** — số dòng raw từ Kaggle (`cs-training.csv`).
- **149.999** — số hồ sơ sau khi loại 1 dòng `age = 0` (vô nghĩa về domain), dùng cho toàn bộ phân tích.
- Phân chia: train 104.999 / validation 22.500 / test 22.500. Mô hình **học** trên 104.999 hồ sơ; AUC = 0,8714 báo cáo trên 22.500 hồ sơ test.
- `cs-test.csv` của Kaggle không có target thật; `sampleEntry.csv` chỉ là mẫu nộp xác suất dự đoán, không phải nhãn 0/1.

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

Batch tab trong Streamlit hỗ trợ đọc file **CSV/XLSX** và xuất file kết quả để tải về. Nếu file upload có cột target `SeriousDlqin2yrs` dạng 0/1, app sẽ tính thêm benchmark AUC/Recall/Precision/F2, ROC/PR, calibration, ma trận nhầm lẫn và tác động chi phí.

Ghi chú demo: Đây là mô phỏng hậu kiểm theo lô bằng dữ liệu có nhãn từ tập training gốc. Với dữ liệu vận hành thật, target sẽ được bổ sung sau kỳ quan sát để đo lại AUC/Recall/Precision/F2.

Kết quả chạy thật trên `data/raw/batch_demo_from_training_5000.csv`:

| Chỉ số | Giá trị |
|---|---:|
| Số hồ sơ / số vỡ nợ | 5.000 / 334 |
| AUC / Recall / Precision / F2 | 0,8780 / 68,26% / 29,53% / 0,5408 |
| TP / FP / FN / TN | 228 / 544 / 106 / 4122 |
| Tỷ lệ từ chối / xác suất TB | 15,44% / 33,44% |
| Chi phí dùng mô hình / duyệt tất cả | 1.464.500 USD / 3.757.500 USD |
| Tiết kiệm mô phỏng | 2.293.000 USD |

## Thử nộp Kaggle

Để thử kết quả trên leaderboard *Give Me Some Credit*, tạo submission từ `cs-test.csv`:

1. Dự đoán xác suất vỡ nợ cho từng dòng trong `data/raw/cs-test.csv` bằng `models/best_model.pkl`.
2. Ghép xác suất với cột `Id` trong `data/raw/sampleEntry.csv`.
3. Lưu file `data/raw/submission_xgb_project_i.csv` gồm đúng hai cột: `Id`, `Probability`.
4. Nộp lên Kaggle:

```powershell
kaggle competitions submit -c GiveMeSomeCredit -f data/raw/submission_xgb_project_i.csv -m "XGBoost Project I submission"
```

File submission đã kiểm tra có 101.503 dòng; xác suất min/mean/max là 0,016970 / 0,330504 / 0,986541. Kết quả nộp thử Kaggle: **Public Score = 0,85785**, **Private Score = 0,86482**. Không dùng `sampleEntry.csv` làm target để benchmark, vì cột `Probability` trong đó là mẫu xác suất submission.

![Kết quả Kaggle](reports/fig_36_kaggle_submission_result.png)

## Cấu trúc

```
├── data/               — raw (cs-training.csv), processed, splits (train/val/test)
├── notebooks/          — 4 notebooks chính + 2 script bổ sung
├── src/                — data_loader, preprocessing, features, models, evaluation, plot_style
├── reports/            — 34+ figures PNG + addendum_results.md
│   └── visual_summary/ — ảnh infographic (pipeline, KPI, what-if scenarios)
├── final_report/       — báo cáo chính bản rút gọn (md, tex, pdf) + tóm tắt trực quan 1 trang
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
1. `final_report/bao_cao_chinh.md` — bản rút gọn, tập trung kết quả và kết luận
2. `presentation/defense_guide.md` Phần B–C — toàn bộ toán học/derivation để hỏi sâu
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
| `final_report/bao_cao_chinh.md` | Báo cáo chính bản rút gọn | Phản biện |
| `final_report/tom_tat_truc_quan.md` | Tóm tắt trực quan 1 trang | Tất cả |
| `presentation/slide.md` | 29 slides Marp — defense kỹ thuật | Phản biện |
| `presentation/slide_executive_summary.md` | 7 slides — dành cho non-tech | Giảng viên |
| `presentation/defense_guide.md` | Q&A + toán học chi tiết để chuẩn bị bảo vệ | Phản biện |
| `reports/visual_summary/` | Infographic pipeline, KPI, what-if | Tất cả |

---

## Báo cáo

Báo cáo chính ở `final_report/bao_cao_chinh.md` là bản rút gọn, tập trung vào mục tiêu, pipeline, kết quả, giá trị ứng dụng và giới hạn. Toàn bộ phần toán học chi tiết đã chuyển sang `presentation/defense_guide.md` để chuẩn bị defense.

Tóm tắt trực quan dành cho người đọc nhanh: `final_report/tom_tat_truc_quan.md`.

`reports/addendum_results.md` chứa kết quả kiểm định DeLong (XGB vs RF: ΔAUC=+0.0043, p<0.0001) và calibration metrics (Brier Score, ECE) cho cả 4 mô hình.
