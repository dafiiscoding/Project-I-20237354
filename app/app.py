"""
Bảng điều khiển Streamlit: Dự báo Rủi ro Vỡ nợ Tín dụng
=========================================================
Mô hình: XGBoost | AUC=0.8714 | Ngưỡng triển khai=0.625

Cấu trúc:
  Tab 1 — Khách hàng đơn lẻ : nhập thủ công + SHAP
  Tab 2 — Đánh giá theo lô  : upload CSV/XLSX + batch predict + visualization
  Tab 3 — Hướng dẫn         : giải thích ứng dụng + cách đọc kết quả

Chạy: streamlit run app/app.py (từ thư mục gốc project)
"""

import sys
import warnings
from pathlib import Path

import streamlit as st

warnings.filterwarnings('ignore')

# Đảm bảo import src/ được tìm thấy khi chạy từ thư mục gốc
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'app'))

# Import sau khi đã set sys.path
from utils import load_model        # noqa: E402
import single_customer as tab_single  # noqa: E402
import batch_evaluation as tab_batch  # noqa: E402


# ─── Cấu hình trang ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Dự báo Rủi ro Vỡ nợ",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏦 Dự báo Rủi ro Vỡ nợ")
    st.caption("Đồ án I — Toán Tin, Đại học Bách Khoa")
    st.caption("Bản app: sample-data-selector")
    st.markdown("---")
    st.markdown("**Thông tin mô hình**")
    st.markdown("""
    - **Mô hình**: XGBoost
    - **AUC-ROC**: 0.8714
    - **Ngưỡng từ chối**: 0.625
    - **Tối ưu hóa**: F2-score (ưu tiên Recall)
    """)
    st.markdown("---")
    st.markdown("**Phân loại rủi ro**")
    st.markdown("""
    🟢 **Thấp** — P < 10%
    🟡 **Trung bình** — 10% ≤ P < 30%
    🟠 **Cao** — 30% ≤ P < 62.5%
    🔴 **Rất cao** — P ≥ 62.5% → Từ chối
    """)
    st.markdown("---")
    st.caption("Dữ liệu: Give Me Some Credit (Kaggle)")


# ─── Tải model ──────────────────────────────────────────────────────────────

model = load_model()


# ─── Tab 3: Hướng dẫn (định nghĩa trước khi dùng) ──────────────────────────

def _render_help() -> None:
    st.subheader("ℹ️ Hướng dẫn sử dụng ứng dụng")

    st.markdown("### Tab 1 — Khách hàng đơn lẻ")
    st.markdown("""
    1. Nhập 10 thông tin tài chính của khách hàng vào form bên trái.
    2. Nhấn **Dự báo rủi ro**.
    3. Xem kết quả: xác suất vỡ nợ, mức độ rủi ro, quyết định (Duyệt/Từ chối).
    4. Phân tích SHAP waterfall cho biết đặc trưng nào ảnh hưởng nhiều nhất.
    """)

    st.markdown("### Tab 2 — Đánh giá theo lô")
    st.markdown("""
    1. Chuẩn bị file CSV/XLSX với 10 cột (nhấn **Tải CSV mẫu** để xem đúng định dạng).
    2. Tải file lên, xem trước dữ liệu.
    3. Chọn ngưỡng quyết định phù hợp cho tập hiện tại.
    4. Nhấn **Đánh giá toàn bộ** — ứng dụng tự động tính thêm 4 đặc trưng kỹ thuật.
    5. Xem các biểu đồ: phân bố rủi ro, top khách hàng rủi ro cao, benchmark và trade-off theo ngưỡng.
    6. Tải CSV kết quả với 4 cột bổ sung: `probability`, `risk_tier`, `decision`, `decision_threshold`.
    """)
    st.info(
        "Nếu file có cột `SeriousDlqin2yrs` dạng 0/1, app sẽ mô phỏng hậu kiểm theo lô và tính "
        "AUC/Recall/Precision/F2. Với dữ liệu vận hành thật, target chỉ có sau kỳ quan sát; "
        "`cs-test.csv` của Kaggle không có target thật nên chỉ dùng để tạo submission."
    )

    st.markdown("### Giải thích các chỉ số")
    import pandas as pd
    st.dataframe(pd.DataFrame([
        {"Chỉ số": "probability", "Ý nghĩa": "Xác suất vỡ nợ trong 2 năm tới (0 → 1)"},
        {"Chỉ số": "risk_tier", "Ý nghĩa": "Mức độ rủi ro: LOW / MEDIUM / HIGH / VERY_HIGH"},
        {"Chỉ số": "decision", "Ý nghĩa": "APPROVE nếu P < threshold, REJECT nếu P ≥ threshold"},
        {"Chỉ số": "decision_threshold", "Ý nghĩa": "Ngưỡng quyết định đã dùng cho lần đánh giá batch hiện tại"},
        {"Chỉ số": "SHAP value", "Ý nghĩa": "Đóng góp của đặc trưng vào dự báo; đỏ = tăng rủi ro"},
    ]), hide_index=True, use_container_width=True)

    st.markdown("### 4 đặc trưng kỹ thuật (tự động tính từ raw features)")
    st.dataframe(pd.DataFrame([
        {"Đặc trưng": "TotalDelinquencyScore", "Công thức": "3×(Trễ 90+) + 2×(Trễ 60-89) + 1×(Trễ 30-59)"},
        {"Đặc trưng": "FinancialStressIndex", "Công thức": "RevolvingUtil × TotalDelinquencyScore"},
        {"Đặc trưng": "AbsoluteMonthlyDebt", "Công thức": "DebtRatio × MonthlyIncome"},
        {"Đặc trưng": "DelinquencySeverityBalance", "Công thức": "Trễ 30-59 − Trễ 90+ (cán cân mức độ trễ hạn)"},
    ]), hide_index=True, use_container_width=True)


# ─── Tabs chính ─────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "👤 Khách hàng đơn lẻ",
    "📊 Đánh giá theo lô",
    "ℹ️ Hướng dẫn",
])

with tab1:
    tab_single.render(model)

with tab2:
    tab_batch.render(model)

with tab3:
    _render_help()
