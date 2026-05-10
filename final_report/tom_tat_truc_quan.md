# Tóm tắt trực quan: Dự báo Rủi ro Vỡ nợ Tín dụng

**Đoàn Danh Long — MSSV 20237354 | GVHD: Nguyễn Cảnh Nam | Đồ án I, 2025–2026**

---

## Kết quả nổi bật

![Chỉ số KPI chính](../reports/visual_summary/fig_vs_02_kpi_callout.png)

| Chỉ số | Giá trị |
|---|---|
| **AUC-ROC tốt nhất** | **0,8714** (XGBoost) |
| **Recall ở ngưỡng triển khai 0,625** | **66,9%** — tăng từ 51,6% ở ngưỡng tối ưu theo F1 |
| **Ngưỡng từ chối** | **0,625** — tối ưu theo F2 |
| **Tiết kiệm ước tính** | **~2,01 triệu USD** so với ngưỡng tối ưu theo F1 trên tập kiểm tra |
| **Đặc trưng quan trọng nhất** | FinancialStressIndex, TotalDelinquencyScore — **2/3 là đặc trưng tự thiết kế** |

---

## Bài toán là gì?

Dự báo khách hàng có vỡ nợ trong **2 năm tới** hay không, dựa trên thông tin tài chính lịch sử.

- **Dữ liệu:** 149.999 hồ sơ vay (Kaggle — Give Me Some Credit)
- **Mất cân bằng:** Chỉ **6,68%** vỡ nợ → không thể chỉ dùng độ chính xác tổng thể (Accuracy)

![Phân bố lớp mục tiêu](../reports/fig_01_target_distribution.png)

---

## Quy trình hoạt động ra sao?

![Sơ đồ quy trình](../reports/visual_summary/fig_vs_01_pipeline_overview.png)

1. **Dữ liệu thô** — 10 đặc trưng tài chính
2. **Khám phá dữ liệu (EDA)** — phát hiện ngoại lệ và tương quan
3. **Tiền xử lý** — điền giá trị thiếu bằng KNN và thiết kế đặc trưng
4. **Mô hình** — so sánh 4 thuật toán: LR, DT, RF, **XGBoost**
5. **Giải thích** — SHAP TreeExplainer
6. **Ứng dụng** — bảng điều khiển Streamlit

---

## Điều gì quan trọng nhất khi mô hình quyết định?

![Tầm quan trọng (SHAP)](../reports/fig_26a_shap_bar.png)

- **FinancialStressIndex** = Tỷ lệ sử dụng × Điểm trễ hạn → **tự tạo**
- **TotalDelinquencyScore** = 3×(trễ 90+) + 2×(trễ 60-89) + 1×(trễ 30-59) → **tự tạo**
- **RevolvingUtilizationOfUnsecuredLines** — tỷ lệ sử dụng hạn mức → gốc

→ **2 trong 3 yếu tố quan trọng nhất là đặc trưng tự thiết kế**

---

## Mô hình "nhìn" khách hàng như thế nào?

![Tình huống điển hình](../reports/visual_summary/fig_vs_03_what_if_scenarios.png)

| Hồ sơ | Đặc điểm | Xác suất vỡ | Quyết định |
|---|---|---|---|
| **An toàn** | Tuổi 45, thu nhập $5K, không trễ, nợ 30% | ~5% | ✅ DUYỆT |
| **Cảnh báo** | Tuổi 35, thu nhập $3K, trễ 1 lần, nợ 55% | ~40% | ⚠️ CẢNH BÁO |
| **Rủi ro cao** | Tuổi 28, thu nhập $2K, trễ 2+ lần, nợ 85% | ~80% | ❌ TỪ CHỐI |

---

## Tại sao chọn ngưỡng 0,625 thay vì 0,5?

![Tối ưu hóa ngưỡng](../reports/fig_25_threshold_optimization.png)

Trong tín dụng: **bỏ sót 1 người sẽ vỡ nợ (FN)** tốn kém hơn nhiều so với **từ chối nhầm người tốt (FP)**.

- Ngưỡng tối ưu theo F1 (0,77): Recall = 51,6% → bỏ sót 48,4%
- Ngưỡng tối ưu theo F2 **(0,625): Recall = 66,9% → giảm khoảng 2,01 triệu USD so với ngưỡng tối ưu theo F1**

F2-score ưu tiên Recall gấp đôi Precision — phù hợp logic kinh doanh.

---

## Demo theo lô và kiểm tra Kaggle

Ứng dụng Streamlit: <https://doandanhlong-loan-prediction.streamlit.app>

Streamlit có thể tải CSV/XLSX để dự báo theo lô. Nếu file có thêm cột nhãn thật `SeriousDlqin2yrs` dạng 0/1, ứng dụng sẽ mô phỏng hậu kiểm và tính AUC/Recall/Precision/F2 cùng tác động chi phí.

**Lưu ý khi demo:** Đây là mô phỏng hậu kiểm theo lô bằng dữ liệu có nhãn từ tập huấn luyện gốc. Với dữ liệu vận hành thật, nhãn thật sẽ được bổ sung sau kỳ quan sát để đo lại AUC/Recall/Precision/F2.

Kết quả chạy thật trên file demo 5.000 hồ sơ:

| Chỉ số | Giá trị |
|---|---:|
| AUC / Recall / Precision / F2 | 0,8780 / 68,26% / 29,53% / 0,5408 |
| TP / FP / FN / TN | 228 / 544 / 106 / 4122 |
| Tỷ lệ từ chối | 15,44% |
| Chi phí dùng mô hình | 1.464.500 USD |
| Tiết kiệm so với duyệt tất cả | 2.293.000 USD |

![Phân bố demo theo lô](../reports/fig_33_batch_demo_distribution.png)

![Chi phí batch demo](../reports/fig_35_batch_demo_cost.png)

Với Kaggle, tập kiểm tra không có nhãn thật nên không dùng để hậu kiểm. Kết quả nộp thử đạt **Public Score = 0,85785**, **Private Score = 0,86482**.

![Kết quả Kaggle](../reports/fig_36_kaggle_submission_result.png)

---

## Các tài liệu liên quan

| Tài liệu | Mô tả |
|---|---|
| `final_report/bao_cao_chinh.pdf` | Báo cáo chính để nộp/chấm |
| `final_report/bao_cao_chinh.md` | Nguồn Markdown của báo cáo |
| `app/app.py` — `streamlit run app/app.py` | Ứng dụng tương tác |
| `README.md` | Hướng dẫn đọc repo và chạy nhanh |
- Demo app: <https://doandanhlong-loan-prediction.streamlit.app>
