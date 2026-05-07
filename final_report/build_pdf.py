"""Build bao_cao_chinh.pdf from Markdown with symbol-safe fonts.

This keeps mathematical symbols and emoji glyphs in the PDF by using:
- Times New Roman for body text
- Cascadia Mono for code spans
- Segoe UI Emoji for status icons
"""

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "final_report"
MD = REPORT_DIR / "bao_cao_chinh.md"
TEX = REPORT_DIR / "bao_cao_chinh.tex"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def patch_tex() -> None:
    text = TEX.read_text(encoding="utf-8")
    layout_preamble = r"""
\usepackage{float}
\usepackage{placeins}
\usepackage{needspace}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,skip=6pt}
\setkeys{Gin}{width=\linewidth,height=0.72\textheight,keepaspectratio}
\setlength{\LTpre}{6pt}
\setlength{\LTpost}{6pt}
\renewcommand{\arraystretch}{1.08}
\sloppy
"""
    text = text.replace(r"\begin{document}", layout_preamble + "\n" + r"\begin{document}", 1)
    text = text.replace(
        r"\setmonofont[]{Cascadia Mono}",
        r"\setmonofont[]{Cascadia Mono}" + "\n" + r"\newfontfamily\EmojiFont{Segoe UI Emoji}",
    )
    for char in ["\U0001F7E2", "\U0001F7E1", "\U0001F7E0", "\U0001F534", "\u2705", "\u274C"]:
        text = text.replace(char, r"{\EmojiFont " + char + "}")
    text = text.replace("\u2272", r"$\lesssim$")

    text = re.sub(
        r"(\\section\{CHƯƠNG\s+[1-6]:)",
        r"\\clearpage\n\\FloatBarrier\n\1",
        text,
    )

    page_break_sections = ["TÀI LIỆU THAM KHẢO", "PHỤ LỤC"]
    for section in page_break_sections:
        pattern = re.compile(r"(\\section\{" + re.escape(section) + r"\})")
        text = pattern.sub(r"\\clearpage\n\\FloatBarrier\n\1", text, count=1)

    text = re.sub(r"(\\subsection\{)", r"\\FloatBarrier\n\\Needspace{8\\baselineskip}\n\1", text)
    text = re.sub(r"(\\subsubsection\{)", r"\\Needspace{6\\baselineskip}\n\1", text)
    TEX.write_text(text, encoding="utf-8")


def main() -> None:
    run([
        "pandoc",
        "-s",
        str(MD),
        "-o",
        str(TEX),
        "--pdf-engine=xelatex",
        "-V",
        "mainfont=Times New Roman",
        "-V",
        "monofont=Cascadia Mono",
        "-V",
        "geometry:margin=1in",
    ])
    patch_tex()
    for _ in range(2):
        run([
            "xelatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory=final_report",
            str(TEX),
        ])


if __name__ == "__main__":
    main()
