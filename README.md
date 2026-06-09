# Dự báo rủi ro vỡ nợ tín dụng

Ứng dụng học máy dự báo xác suất vỡ nợ tín dụng trên bộ dữ liệu **Give Me Some Credit** của Kaggle. Dự án gồm quy trình phân tích dữ liệu, huấn luyện mô hình, báo cáo kết quả và ứng dụng Streamlit để dự báo từng khách hàng hoặc đánh giá theo lô.

Ứng dụng demo: <https://doandanhlong-loan-prediction.streamlit.app>

![So sánh bốn mô hình](reports/fig_20_model_comparison.png)

## Kết quả chính

| Nội dung | Kết quả |
|---|---:|
| Mô hình chính | XGBoost |
| AUC-ROC trên tập kiểm tra | 0,8714 |
| Ngưỡng quyết định triển khai | 0,625 |
| Recall tại ngưỡng 0,625 | 66,9% |
| Precision tại ngưỡng 0,625 | 29,9% |
| F2-score tại ngưỡng 0,625 | 0,537 |
| Kết quả Kaggle | Public 0,85785 / Private 0,86482 |

Ghi chú thuật ngữ:
- **AUC-ROC**: đo khả năng xếp hạng khách hàng rủi ro cao hơn khách hàng an toàn.
- **Recall**: tỷ lệ phát hiện đúng người thật sự vỡ nợ.
- **Precision**: trong nhóm bị từ chối, tỷ lệ thật sự vỡ nợ.
- **F2-score**: chỉ số ưu tiên Recall hơn Precision, phù hợp bài toán tín dụng vì bỏ sót người vỡ nợ thường tốn kém hơn từ chối nhầm.
- **SHAP**: phương pháp giải thích đặc trưng nào làm tăng hoặc giảm rủi ro trong dự báo.

Tầm quan trọng đặc trưng theo SHAP (2 đặc trưng tự thiết kế vào top-3):

![Tầm quan trọng đặc trưng theo SHAP](reports/fig_26a_shap_bar.png)

Tối ưu ngưỡng quyết định theo F2:

![Tối ưu ngưỡng theo F2](reports/fig_25_threshold_optimization.png)

## Chạy nhanh ứng dụng

Cài thư viện:

```powershell
pip install -r requirements.txt
```

Chạy Streamlit:

```powershell
streamlit run app/app.py
```

Ứng dụng có ba phần:
- **Khách hàng đơn lẻ**: nhập 10 thông tin tài chính và xem xác suất vỡ nợ.
- **Đánh giá theo lô**: chọn dữ liệu mẫu hoặc tải CSV/XLSX để dự báo hàng loạt.
- **Hướng dẫn**: giải thích cách đọc xác suất, nhóm rủi ro và kết quả SHAP.

## Dữ liệu mẫu có sẵn

Repo có sẵn dữ liệu nhỏ phục vụ demo Streamlit:

| File | Mục đích |
|---|---|
| `app/sample_data/batch_demo_from_training_5000.csv` | 5.000 hồ sơ có nhãn thật, dùng để mô phỏng hậu kiểm theo lô |
| `app/sample_data/kaggle_test_sample.csv` | 200 hồ sơ không có nhãn thật, dùng để thử luồng dự báo |

Kết quả hậu kiểm trên file 5.000 hồ sơ:

| Chỉ số | Giá trị |
|---|---:|
| Số hồ sơ / số vỡ nợ | 5.000 / 334 |
| AUC-ROC | 0,8780 |
| Recall / Precision / F2 | 68,26% / 29,53% / 0,5408 |
| TP / FP / FN / TN | 228 / 544 / 106 / 4122 |
| Tỷ lệ từ chối | 15,44% |
| Tiết kiệm mô phỏng so với duyệt tất cả | 2.293.000 USD |

Đây là mô phỏng hậu kiểm bằng dữ liệu có nhãn từ tập huấn luyện gốc. Trong vận hành thật, cột nhãn thật chỉ có sau kỳ quan sát.

## Dữ liệu Kaggle đầy đủ

Do điều khoản Kaggle, repo không đính kèm toàn bộ `data/raw/`. Nếu muốn tái lập đầy đủ, tải dữ liệu tại:

<https://www.kaggle.com/c/GiveMeSomeCredit/data>

Đặt các file vào `data/raw/`:
- `cs-training.csv`
- `cs-test.csv`
- `sampleEntry.csv`

Quy ước số liệu:
- Kaggle cung cấp 150.000 dòng huấn luyện ban đầu.
- Dự án loại 1 dòng có `age = 0`, còn 149.999 hồ sơ để phân tích.
- Chia dữ liệu: 104.999 huấn luyện, 22.500 kiểm định, 22.500 kiểm tra.
- `cs-test.csv` không có nhãn thật nên chỉ dùng để tạo file nộp Kaggle, không dùng để hậu kiểm.

## Cấu trúc repo

```text
app/                 Ứng dụng Streamlit
app/sample_data/     Dữ liệu mẫu nhỏ để demo trực tiếp
final_report/        Báo cáo chính, file TeX/PDF và tóm tắt trực quan
models/              Mô hình chính best_model.pkl
notebooks/           4 notebook chính: EDA, tiền xử lý, mô hình, phân tích
reports/             Hình ảnh và bảng kết quả đã dùng trong báo cáo
src/                 Mã nguồn xử lý dữ liệu, đặc trưng, mô hình, đánh giá
requirements.txt     Danh sách thư viện Python
```

Các thư mục/file không đưa lên GitHub gồm `data/`, cache, log LaTeX và các mô hình phụ lớn.

## File nên đọc trước

| File | Khi nào nên đọc |
|---|---|
| `final_report/bao_cao_chinh.pdf` | Bản báo cáo chính để nộp/chấm |
| `final_report/tom_tat_truc_quan.md` | Tóm tắt nhanh cho người không chuyên |
| `app/app.py` | Điểm vào ứng dụng Streamlit |
| `models/best_model.pkl` | Mô hình XGBoost chính đã huấn luyện |
| `reports/model_results.csv` | Bảng kết quả mô hình |

## Ghi chú triển khai

- Mô hình chính là XGBoost, được lưu tại `models/best_model.pkl`.
- Ứng dụng Streamlit có sẵn lựa chọn demo 5.000 dòng có nhãn để kiểm tra hậu kiểm.
- Các thuật ngữ tiếng Anh cần giữ như XGBoost, SHAP, AUC-ROC được giải thích trong báo cáo và tóm tắt.

## Biên dịch báo cáo

Báo cáo được soạn bằng LaTeX (`final_report/bao_cao_chinh.tex`). Cần MiKTeX/TeX Live có `xelatex`. Thả 2 logo `logo_bachkhoa.png` và `logo_toantin.png` vào `final_report/assets/` (thiếu thì bìa hiện ô giữ chỗ), rồi:

```powershell
cd final_report
python build_pdf.py
```

## Giấy phép

Mã nguồn và tài liệu trong kho phát hành theo giấy phép [MIT](LICENSE). Dữ liệu gốc thuộc Kaggle, không nằm trong phạm vi giấy phép này.
