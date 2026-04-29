# Loan Default Prediction

Dự báo xác suất vỡ nợ tín dụng bằng Machine Learning trên dataset [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle, 150.000 hồ sơ vay). Bao gồm toàn bộ pipeline từ EDA, tiền xử lý, huấn luyện 4 mô hình, phân tích sai số, đến Streamlit dashboard cho phép dự báo từng hồ sơ với giải thích SHAP.

## Kết quả

| Mô hình | AUC-ROC | F1 (t=0,625) | Precision | Recall |
|---------|--------:|-------------:|----------:|-------:|
| Logistic Regression | 0,8432 | 0,434 | 0,388 | 0,493 |
| Decision Tree | 0,8579 | 0,431 | 0,399 | 0,469 |
| Random Forest | 0,8703 | 0,444 | 0,387 | 0,521 |
| **XGBoost** | **0,8714** | **0,447** | **0,394** | **0,516** |

XGBoost cho AUC cao nhất. Ngưỡng phân loại được đặt ở 0,625 thay vì 0,5 vì trong tín dụng bỏ sót người vỡ nợ tốn kém hơn nhiều so với từ chối nhầm — ngưỡng này được chọn bằng cách tối ưu F2-score trên tập validation.

Theo SHAP, hai trong ba features quan trọng nhất là features tự tạo (`FinancialStressIndex`, `TotalDelinquencyScore`), không phải features gốc từ dataset.

## Dataset

Dataset không được đính kèm do điều khoản Kaggle. Tải về tại:

```bash
kaggle competitions download -c GiveMeSomeCredit
# hoặc tải thủ công từ https://www.kaggle.com/c/GiveMeSomeCredit/data
```

Đặt file `cs-training.csv` vào `data/raw/`.

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
├── notebooks/          — 4 notebooks chính + 2 script bổ sung
├── src/                — data_loader, preprocessing, features, models, evaluation
├── reports/            — báo cáo markdown + 32 figures
├── final_report/       — báo cáo chính 7 chương (md, tex, pdf)
├── models/             — best_model.pkl (XGBoost, ~340KB)
├── app/                — Streamlit dashboard
└── requirements.txt
```

## Báo cáo

Báo cáo chính ở `final_report/bao_cao_chinh.md` (và `.pdf`), gồm 7 chương với derivation toán học đầy đủ cho từng thuật toán, phân tích thực nghiệm, và thảo luận về trade-off interpretability vs performance, data-centric vs model-centric, và chiến lược threshold.

`reports/addendum_results.md` chứa kết quả kiểm định DeLong so sánh AUC của XGBoost và Random Forest, và bảng calibration metrics (Brier Score, ECE) cho cả 4 mô hình.
