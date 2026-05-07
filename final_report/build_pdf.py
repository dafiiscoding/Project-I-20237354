"""Build bao_cao_chinh.pdf from Markdown with symbol-safe fonts.

This keeps mathematical symbols and emoji glyphs in the PDF by using:
- Times New Roman for body text
- Cascadia Mono for code spans
- Segoe UI Emoji for status icons
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "final_report"
MD = REPORT_DIR / "bao_cao_chinh.md"
TEX = REPORT_DIR / "bao_cao_chinh.tex"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def patch_tex() -> None:
    text = TEX.read_text(encoding="utf-8")
    text = text.replace(
        r"\setmonofont[]{Cascadia Mono}",
        r"\setmonofont[]{Cascadia Mono}" + "\n" + r"\newfontfamily\EmojiFont{Segoe UI Emoji}",
    )
    for char in ["\U0001F7E2", "\U0001F7E1", "\U0001F7E0", "\U0001F534", "\u2705", "\u274C"]:
        text = text.replace(char, r"{\EmojiFont " + char + "}")
    text = text.replace("\u2272", r"$\lesssim$")
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
