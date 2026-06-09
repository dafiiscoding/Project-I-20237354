"""Biên dịch bao_cao_chinh.pdf từ file LaTeX viết tay (bao_cao_chinh.tex).

Báo cáo nay được soạn trực tiếp bằng LaTeX (lớp `report`), KHÔNG còn qua
pandoc. Chỉ cần xelatex 2 lần (lần 1 dựng mục lục/tham chiếu, lần 2 chốt
số trang và cleveref/\\ref). Logo bìa: thả 2 file PNG vào final_report/assets/
- assets/logo_bachkhoa.png
- assets/logo_toantin.png
Nếu thiếu, bìa tự hiển thị ô giữ chỗ (không lỗi build).
"""

from pathlib import Path
import subprocess

REPORT_DIR = Path(__file__).resolve().parent
TEX = REPORT_DIR / "bao_cao_chinh.tex"


def main() -> None:
    for _ in range(3):
        subprocess.run(
            [
                "xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                TEX.name,
            ],
            cwd=REPORT_DIR,
            check=True,
        )
    print(f"OK -> {REPORT_DIR / 'bao_cao_chinh.pdf'}")


if __name__ == "__main__":
    main()
