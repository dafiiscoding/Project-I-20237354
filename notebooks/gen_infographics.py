"""
Sinh 3 infographic chat luong cao cho nguoi doc non-tech.
Output: reports/visual_summary/fig_vs_01, fig_vs_02, fig_vs_03 (PNG, dpi=200).

Khong dung emoji (tranh font glyph thieu); dung circle + text monogram.

Chay: python notebooks/gen_infographics.py (tu thu muc goc project)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, Wedge, Circle, FancyArrowPatch,
                                Rectangle, Polygon, Ellipse)
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from plot_style import setup_vietnamese_style  # noqa: E402

setup_vietnamese_style(dpi=200)

OUT_DIR = PROJECT_ROOT / 'reports' / 'visual_summary'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _draw_db_icon(ax, cx, cy, r, color):
    """Hinh tru CSDL."""
    h = r * 1.4
    ax.add_patch(Ellipse((cx, cy + h / 2), r * 1.6, r * 0.5,
                         facecolor=color, edgecolor='white', linewidth=2, zorder=4))
    ax.add_patch(Rectangle((cx - r * 0.8, cy - h / 2), r * 1.6, h,
                           facecolor=color, edgecolor='none', zorder=3))
    ax.add_patch(Ellipse((cx, cy - h / 2), r * 1.6, r * 0.5,
                         facecolor=color, edgecolor='white', linewidth=2, zorder=4))
    for dy in (-0.05, 0.15):
        ax.add_patch(Ellipse((cx, cy + dy), r * 1.6, r * 0.5,
                             facecolor='none', edgecolor='white', linewidth=1.5, zorder=5))


def _draw_eye_icon(ax, cx, cy, r, color):
    """Bieu tuong EDA: hinh con mat."""
    ax.add_patch(Ellipse((cx, cy), r * 1.6, r * 0.85,
                         facecolor='white', edgecolor=color, linewidth=2.5, zorder=4))
    ax.add_patch(Circle((cx, cy), r * 0.42, facecolor=color, zorder=5))
    ax.add_patch(Circle((cx + r * 0.13, cy + r * 0.12), r * 0.12,
                        facecolor='white', zorder=6))


def _draw_gear_icon(ax, cx, cy, r, color):
    """Banh rang."""
    n_teeth = 8
    outer_r = r * 0.95
    tooth_r = r * 1.15
    angles = np.linspace(0, 2 * np.pi, n_teeth, endpoint=False)
    pts = []
    for a in angles:
        a2 = a + np.pi / n_teeth
        pts.append((cx + outer_r * np.cos(a - np.pi / (n_teeth * 2.5)),
                    cy + outer_r * np.sin(a - np.pi / (n_teeth * 2.5))))
        pts.append((cx + tooth_r * np.cos(a),
                    cy + tooth_r * np.sin(a)))
        pts.append((cx + outer_r * np.cos(a + np.pi / (n_teeth * 2.5)),
                    cy + outer_r * np.sin(a + np.pi / (n_teeth * 2.5))))
        pts.append((cx + outer_r * np.cos(a2),
                    cy + outer_r * np.sin(a2)))
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor='white',
                         linewidth=2, zorder=4))
    ax.add_patch(Circle((cx, cy), r * 0.4, facecolor='white', zorder=5))


def _draw_brain_icon(ax, cx, cy, r, color):
    """Mo hinh: 2 hinh tron lien thong."""
    ax.add_patch(Circle((cx - r * 0.4, cy), r * 0.7,
                        facecolor=color, edgecolor='white', linewidth=2, zorder=4))
    ax.add_patch(Circle((cx + r * 0.4, cy), r * 0.7,
                        facecolor=color, edgecolor='white', linewidth=2, zorder=4))
    for ang in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        ax.plot([cx + r * 0.7 * np.cos(ang)], [cy + r * 0.7 * np.sin(ang)],
                'o', color='white', markersize=3.5, zorder=6)


def _draw_bulb_icon(ax, cx, cy, r, color):
    """Bong den: SHAP."""
    ax.add_patch(Circle((cx, cy + r * 0.15), r * 0.85,
                        facecolor=color, edgecolor='white', linewidth=2, zorder=4))
    ax.add_patch(Rectangle((cx - r * 0.32, cy - r * 0.85), r * 0.64, r * 0.3,
                           facecolor=color, edgecolor='white', linewidth=2, zorder=4))
    ax.add_patch(Rectangle((cx - r * 0.25, cy - r * 1.1), r * 0.5, r * 0.18,
                           facecolor=color, edgecolor='white', linewidth=2, zorder=4))
    # Tia sang
    for ang in (np.pi / 4, 3 * np.pi / 4):
        ax.plot([cx + r * 1.1 * np.cos(ang), cx + r * 1.45 * np.cos(ang)],
                [cy + r * 1.1 * np.sin(ang), cy + r * 1.45 * np.sin(ang)],
                color=color, linewidth=2, zorder=5)


def _draw_chart_icon(ax, cx, cy, r, color):
    """Bieu do cot: ung dung dashboard."""
    bar_w = r * 0.32
    bars_x = [cx - r * 0.7, cx - r * 0.18, cx + r * 0.34]
    bars_h = [r * 0.7, r * 1.1, r * 1.5]
    for x, h in zip(bars_x, bars_h):
        ax.add_patch(Rectangle((x, cy - r * 0.85), bar_w, h,
                               facecolor=color, edgecolor='white', linewidth=2, zorder=4))


# ===========================================================================
# INFOGRAPHIC 1 - Pipeline 6 stages
# ===========================================================================

def make_pipeline_overview() -> Path:
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis('off')
    fig.patch.set_facecolor('#f8f9fb')

    ax.text(8, 6.4, 'Quy trinh tu du lieu tho den quyet dinh cho vay',
            ha='center', va='center', fontsize=20, fontweight='bold', color='#1a2740')
    ax.text(8, 5.85,
            '149.999 ho so phan tich  -->  Mo hinh XGBoost  -->  Streamlit dashboard',
            ha='center', va='center', fontsize=12, color='#566173', style='italic')

    stages = [
        ('1', 'Du lieu',     '149.999 ho so\n10 dac trung goc',                 '#2980b9', _draw_db_icon),
        ('2', 'EDA',         'Phan phoi, outlier,\ntuong quan',                  '#3498db', _draw_eye_icon),
        ('3', 'Tien xu ly',  'KNN Imputer +\n4 dac trung tu tao',                '#1abc9c', _draw_gear_icon),
        ('4', 'Mo hinh',     'So sanh 4 thuat toan\nXGBoost = tot nhat',         '#f39c12', _draw_brain_icon),
        ('5', 'SHAP',        'Giai thich tung\nquyet dinh (Basel III)',          '#e67e22', _draw_bulb_icon),
        ('6', 'Ung dung',    'Streamlit dashboard\nDon le + theo lo',            '#c0392b', _draw_chart_icon),
    ]

    n = len(stages)
    margin = 0.6
    card_w = (16 - 2 * margin - (n - 1) * 0.3) / n
    card_h = 3.0
    y_bot = 1.6

    for i, (num, name, desc, color, icon_fn) in enumerate(stages):
        x_left = margin + i * (card_w + 0.3)
        x_center = x_left + card_w / 2

        card = FancyBboxPatch((x_left, y_bot), card_w, card_h,
                              boxstyle="round,pad=0.05,rounding_size=0.18",
                              linewidth=0, facecolor=color, alpha=0.95, zorder=2)
        ax.add_patch(card)

        stripe = FancyBboxPatch((x_left, y_bot + card_h - 0.65), card_w, 0.65,
                                boxstyle="round,pad=0.05,rounding_size=0.18",
                                linewidth=0, facecolor='black', alpha=0.18, zorder=3)
        ax.add_patch(stripe)

        ax.text(x_left + 0.25, y_bot + card_h - 0.32, num,
                ha='left', va='center', fontsize=15, fontweight='bold', color='white', zorder=4)
        ax.text(x_center + 0.18, y_bot + card_h - 0.32, name,
                ha='center', va='center', fontsize=14, fontweight='bold', color='white', zorder=4)

        # Icon
        icon_fn(ax, x_center, y_bot + card_h / 2 + 0.1, 0.42, 'white')

        ax.text(x_center, y_bot + 0.45, desc,
                ha='center', va='center', fontsize=10.5, color='white', zorder=4,
                linespacing=1.3)

        if i < n - 1:
            arrow = FancyArrowPatch((x_left + card_w + 0.02, y_bot + card_h / 2),
                                    (x_left + card_w + 0.28, y_bot + card_h / 2),
                                    arrowstyle='->,head_width=0.25,head_length=0.35',
                                    color='#566173', linewidth=2.5, zorder=1)
            ax.add_patch(arrow)

    band = FancyBboxPatch((0.6, 0.25), 14.8, 1.0,
                          boxstyle="round,pad=0.05,rounding_size=0.15",
                          linewidth=0, facecolor='#1a2740', zorder=1)
    ax.add_patch(band)

    highlights = [
        ('AUC-ROC',     '0,8714'),
        ('Recall',      '66,9%'),
        ('Nguong',      '0,625'),
        ('Tiet kiem',   '~$2M/nam'),
    ]
    # Vietnamese accents (re-render with proper text via ax.text below)
    highlights_vi = [
        ('AUC-ROC',     '0,8714'),
        ('Recall',      '66,9%'),
        ('Ngưỡng',      '0,625'),
        ('Tiết kiệm',   '~$2M/năm'),
    ]
    for i, (k, v) in enumerate(highlights_vi):
        cx = 1.6 + i * 3.7
        ax.text(cx, 0.95, k, ha='left', va='center', fontsize=11, color='#a8b3c7')
        ax.text(cx, 0.55, v, ha='left', va='center', fontsize=18, fontweight='bold', color='white')

    # Re-render Vietnamese accents on titles
    ax.texts[0].set_text('Quy trình từ dữ liệu thô đến quyết định cho vay')
    ax.texts[1].set_text('149.999 hồ sơ phân tích  →  Mô hình XGBoost  →  Streamlit dashboard')

    # Re-render stage titles with accents
    stages_vi = [
        ('1', 'Dữ liệu',     '149.999 hồ sơ\n10 đặc trưng gốc'),
        ('2', 'EDA',         'Phân phối, outlier,\ntương quan'),
        ('3', 'Tiền xử lý',  'KNN Imputer +\n4 đặc trưng tự tạo'),
        ('4', 'Mô hình',     'So sánh 4 thuật toán\nXGBoost = tốt nhất'),
        ('5', 'SHAP',        'Giải thích từng\nquyết định (Basel III)'),
        ('6', 'Ứng dụng',    'Streamlit dashboard\nĐơn lẻ + theo lô'),
    ]
    # ax.texts order: [title, subtitle, then per stage: num, name, desc, then highlights k,v×4]
    text_idx = 2
    for num, name, desc in stages_vi:
        ax.texts[text_idx].set_text(num)
        ax.texts[text_idx + 1].set_text(name)
        ax.texts[text_idx + 2].set_text(desc)
        text_idx += 3

    out = OUT_DIR / 'fig_vs_01_pipeline_overview.png'
    plt.savefig(out, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ===========================================================================
# INFOGRAPHIC 2 - KPI callout (4 cards 2x2)
# ===========================================================================

def make_kpi_callout() -> Path:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor('#f8f9fb')

    ax.text(7, 7.4, 'Bốn chỉ số quan trọng nhất của mô hình',
            ha='center', va='center', fontsize=22, fontweight='bold', color='#1a2740')
    ax.text(7, 6.85,
            'Đo trên tập kiểm tra 22.500 hồ sơ độc lập (chưa thấy lúc huấn luyện)',
            ha='center', va='center', fontsize=12, color='#566173', style='italic')

    kpis = [
        ('AUC-ROC',         '0,8714',   '#2980b9', 'AUC',
         'Khả năng phân biệt người vỡ nợ\nvới người không vỡ nợ',
         'Càng gần 1 càng tốt — mô hình xếp đúng\n2 hồ sơ ngẫu nhiên 87,1% lần thử'),

        ('Recall',          '66,9%',    '#27ae60', 'R',
         'Tỷ lệ phát hiện đúng người\nthực sự sẽ vỡ nợ',
         'Trong 100 người vỡ nợ thực, mô hình bắt\nđược ~67 người trước khi cấp tín dụng'),

        ('Ngưỡng từ chối',  '0,625',    '#e67e22', 'T',
         'Tối ưu F2-score — ưu tiên Recall\ngấp đôi Precision',
         'Hồ sơ có xác suất ≥ 62,5%\n→ TỪ CHỐI cấp tín dụng'),

        ('Tiết kiệm',       '~$2M',     '#c0392b', '$',
         'Ước tính trên 22.500 hồ sơ test/năm\nso với ngưỡng F1-optimal (0,77)',
         'Bắt thêm 15,3 điểm Recall ×\nthiệt hại trung bình $11.250/khoản nợ xấu'),
    ]

    card_w = 6.0
    card_h = 2.7
    gap_x = 0.4
    gap_y = 0.3
    x0 = (14 - 2 * card_w - gap_x) / 2
    y0_top = 3.55
    y0_bot = y0_top - card_h - gap_y
    positions = [(x0, y0_top), (x0 + card_w + gap_x, y0_top),
                 (x0, y0_bot), (x0 + card_w + gap_x, y0_bot)]

    for (px, py), (label, value, color, icon, desc, expl) in zip(positions, kpis):
        shadow = FancyBboxPatch((px + 0.08, py - 0.08), card_w, card_h,
                                boxstyle="round,pad=0.05,rounding_size=0.15",
                                linewidth=0, facecolor='black', alpha=0.08, zorder=1)
        ax.add_patch(shadow)
        card = FancyBboxPatch((px, py), card_w, card_h,
                              boxstyle="round,pad=0.05,rounding_size=0.15",
                              linewidth=0, facecolor='white', zorder=2)
        ax.add_patch(card)
        # Color stripe trai
        stripe = Rectangle((px, py), 0.18, card_h, facecolor=color, zorder=3)
        ax.add_patch(stripe)

        # Icon circle: latin monogram
        icon_cx = px + 0.85
        icon_cy = py + card_h - 0.65
        ax.add_patch(Circle((icon_cx, icon_cy), 0.42, facecolor=color, alpha=0.18, zorder=3))
        ax.text(icon_cx, icon_cy, icon, ha='center', va='center',
                fontsize=18, fontweight='bold', color=color, zorder=4)

        ax.text(px + 1.5, py + card_h - 0.55, label,
                ha='left', va='center', fontsize=12.5, color='#566173',
                fontweight='bold', zorder=4)
        ax.text(px + 1.5, py + card_h - 1.25, value,
                ha='left', va='center', fontsize=34, fontweight='bold', color=color, zorder=4)
        ax.text(px + 0.45, py + 0.95, desc,
                ha='left', va='center', fontsize=10.5, color='#1a2740',
                zorder=4, linespacing=1.4)
        ax.plot([px + 0.45, px + card_w - 0.4], [py + 0.65, py + 0.65],
                color='#dfe4ea', linewidth=0.8, zorder=4)
        ax.text(px + 0.45, py + 0.35, expl,
                ha='left', va='center', fontsize=9.3, color='#7f8c8d', style='italic',
                zorder=4, linespacing=1.4)

    ax.text(7, 0.25,
            'Tham chiếu: bảng 4.1 báo cáo chính | reports/model_results.csv',
            ha='center', va='center', fontsize=9.5, color='#95a5a6', style='italic')

    out = OUT_DIR / 'fig_vs_02_kpi_callout.png'
    plt.savefig(out, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ===========================================================================
# INFOGRAPHIC 3 - What-if scenarios (3 customer profiles)
# ===========================================================================

def _draw_gauge(ax, cx, cy, prob, color, radius=0.7):
    """Gauge ban nguyet xac suat vo no."""
    bg = Wedge((cx, cy), radius, 180, 0, width=0.18,
               facecolor='#ecf0f1', edgecolor='none', zorder=2)
    ax.add_patch(bg)
    angle_end = 180 - prob * 180
    fg = Wedge((cx, cy), radius, angle_end, 180, width=0.18,
               facecolor=color, edgecolor='none', zorder=3)
    ax.add_patch(fg)
    ax.text(cx, cy + 0.05, f'{prob*100:.0f}%', ha='center', va='center',
            fontsize=22, fontweight='bold', color=color, zorder=4)
    ax.text(cx, cy - 0.22, 'xác suất vỡ nợ', ha='center', va='center',
            fontsize=8.5, color='#7f8c8d', zorder=4)
    ax.text(cx - radius - 0.05, cy - 0.05, '0%', ha='right', va='center',
            fontsize=8, color='#95a5a6', zorder=4)
    ax.text(cx + radius + 0.05, cy - 0.05, '100%', ha='left', va='center',
            fontsize=8, color='#95a5a6', zorder=4)


def make_what_if_scenarios() -> Path:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor('#f8f9fb')

    ax.text(8, 8.45, 'Mô hình "nhìn" khách hàng như thế nào?',
            ha='center', va='center', fontsize=22, fontweight='bold', color='#1a2740')
    ax.text(8, 7.95,
            'Ba hồ sơ minh họa — cùng 10 đặc trưng đầu vào, mô hình ra quyết định khác nhau',
            ha='center', va='center', fontsize=12, color='#566173', style='italic')

    customers = [
        {
            'name': 'Khách hàng A',
            'tag':  'AN TOÀN',
            'avatar': 'A',
            'tier_color': '#27ae60',
            'card_bg': '#eafaf1',
            'attrs': [
                ('Tuổi',                '45'),
                ('Thu nhập/tháng',      '$5.000'),
                ('Tỷ lệ nợ',            '0,30 (thấp)'),
                ('Số lần trễ hạn',      '0'),
                ('Vay BĐS',             '1 khoản'),
            ],
            'prob': 0.05,
            'decision': 'DUYỆT',
            'rationale': 'Thu nhập ổn định, không\ntrễ hạn, tỷ lệ nợ thấp',
        },
        {
            'name': 'Khách hàng B',
            'tag':  'CẢNH BÁO',
            'avatar': 'B',
            'tier_color': '#f39c12',
            'card_bg': '#fef5e7',
            'attrs': [
                ('Tuổi',                '35'),
                ('Thu nhập/tháng',      '$3.000'),
                ('Tỷ lệ nợ',            '0,55 (TB)'),
                ('Số lần trễ hạn',      '1× trễ 30-59 ngày'),
                ('Vay BĐS',             '0 khoản'),
            ],
            'prob': 0.40,
            'decision': 'DUYỆT (cảnh báo)',
            'rationale': 'Có dấu hiệu căng thẳng nhưng\nchưa vượt ngưỡng từ chối 62,5%',
        },
        {
            'name': 'Khách hàng C',
            'tag':  'RỦI RO CAO',
            'avatar': 'C',
            'tier_color': '#c0392b',
            'card_bg': '#fdedec',
            'attrs': [
                ('Tuổi',                '28'),
                ('Thu nhập/tháng',      '$2.000'),
                ('Tỷ lệ nợ',            '0,85 (cao)'),
                ('Số lần trễ hạn',      '2× trễ 90+ ngày'),
                ('Vay BĐS',             '0 khoản'),
            ],
            'prob': 0.80,
            'decision': 'TỪ CHỐI',
            'rationale': 'Vượt ngưỡng 62,5% — lịch sử\ntrễ hạn 90+ là tín hiệu mạnh',
        },
    ]

    n = len(customers)
    margin = 0.5
    card_w = (16 - 2 * margin - (n - 1) * 0.4) / n
    card_h = 6.6
    y_bot = 0.7

    for i, c in enumerate(customers):
        x_left = margin + i * (card_w + 0.4)
        x_center = x_left + card_w / 2

        shadow = FancyBboxPatch((x_left + 0.08, y_bot - 0.08), card_w, card_h,
                                boxstyle="round,pad=0.05,rounding_size=0.2",
                                linewidth=0, facecolor='black', alpha=0.07, zorder=1)
        ax.add_patch(shadow)
        card = FancyBboxPatch((x_left, y_bot), card_w, card_h,
                              boxstyle="round,pad=0.05,rounding_size=0.2",
                              linewidth=0, facecolor=c['card_bg'], zorder=2)
        ax.add_patch(card)

        # Header strip
        header = FancyBboxPatch((x_left, y_bot + card_h - 1.0), card_w, 1.0,
                                boxstyle="round,pad=0.05,rounding_size=0.2",
                                linewidth=0, facecolor=c['tier_color'], zorder=3)
        ax.add_patch(header)
        cover = Rectangle((x_left, y_bot + card_h - 1.0), card_w, 0.18,
                          facecolor=c['tier_color'], zorder=3)
        ax.add_patch(cover)

        ax.text(x_center, y_bot + card_h - 0.35, c['name'],
                ha='center', va='center', fontsize=14, fontweight='bold', color='white', zorder=4)
        ax.text(x_center, y_bot + card_h - 0.75, c['tag'],
                ha='center', va='center', fontsize=10.5, color='white',
                fontweight='bold', zorder=4)

        # Avatar circle (with letter)
        avatar_cy = y_bot + card_h - 1.85
        ax.add_patch(Circle((x_center, avatar_cy), 0.6,
                            facecolor='white', edgecolor=c['tier_color'],
                            linewidth=3, zorder=3))
        ax.text(x_center, avatar_cy, c['avatar'],
                ha='center', va='center', fontsize=34, fontweight='bold',
                color=c['tier_color'], zorder=4)

        # Attribute table
        attr_y_start = y_bot + card_h - 2.95
        for j, (k, v) in enumerate(c['attrs']):
            ay = attr_y_start - j * 0.42
            ax.text(x_left + 0.3, ay, k, ha='left', va='center',
                    fontsize=10.5, color='#1a2740', zorder=4)
            ax.text(x_left + card_w - 0.3, ay, v, ha='right', va='center',
                    fontsize=10.5, fontweight='bold', color='#1a2740', zorder=4)
            if j < len(c['attrs']) - 1:
                ax.plot([x_left + 0.3, x_left + card_w - 0.3], [ay - 0.21, ay - 0.21],
                        color='white', linewidth=1.0, alpha=0.7, zorder=3)

        # Gauge
        gauge_y = y_bot + 1.65
        _draw_gauge(ax, x_center, gauge_y, c['prob'], c['tier_color'], radius=0.65)

        # Decision badge
        badge_y = y_bot + 0.52
        badge = FancyBboxPatch((x_left + 0.3, badge_y - 0.27), card_w - 0.6, 0.55,
                               boxstyle="round,pad=0.04,rounding_size=0.15",
                               linewidth=0, facecolor=c['tier_color'], zorder=4)
        ax.add_patch(badge)
        ax.text(x_center, badge_y, c['decision'],
                ha='center', va='center', fontsize=12, fontweight='bold',
                color='white', zorder=5)

        # Rationale
        ax.text(x_center, y_bot + 0.05, c['rationale'],
                ha='center', va='center', fontsize=8.8, color='#566173',
                style='italic', zorder=4, linespacing=1.4)

    out = OUT_DIR / 'fig_vs_03_what_if_scenarios.png'
    plt.savefig(out, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


if __name__ == '__main__':
    print(f"Output dir: {OUT_DIR}")
    for fn in (make_pipeline_overview, make_kpi_callout, make_what_if_scenarios):
        path = fn()
        print(f"  [OK] {path.name}")
    print("Done.")
