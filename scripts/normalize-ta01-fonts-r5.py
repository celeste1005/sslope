#!/usr/bin/env python3
"""Apply Times New Roman consistently to the final presentation copy."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "Articulo_G25_revision_integrada_DeepSeek_Claude_R4.docx"
OUTPUT = ROOT / "Articulo_G25_revision_integrada_DeepSeek_Claude_R5_TNR.docx"
FIGURES = ROOT / "artifacts/manuscript-review-2026-10-29"


def set_fonts(container) -> None:
    for paragraph in container.paragraphs:
        for run in paragraph.runs:
            run.font.name = "Times New Roman"
            rpr = run._element.get_or_add_rPr()
            rfonts = rpr.rFonts
            if rfonts is None:
                rfonts = rpr.get_or_add_rFonts()
            for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
                rfonts.set(qn(f"w:{attr}"), "Times New Roman")
    for table in getattr(container, "tables", []):
        for row in table.rows:
            for cell in row.cells:
                set_fonts(cell)


def main() -> None:
    doc = Document(SOURCE)

    # Preserve style sizes, weight, spacing, and colors; change only the typeface.
    for style in doc.styles:
        if getattr(style, "font", None) is not None:
            style.font.name = "Times New Roman"
            if style._element.rPr is not None:
                rfonts = style._element.rPr.rFonts
                if rfonts is None:
                    rfonts = style._element.rPr.get_or_add_rFonts()
                for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
                    rfonts.set(qn(f"w:{attr}"), "Times New Roman")

    set_fonts(doc)
    for section in doc.sections:
        set_fonts(section.header)
        set_fonts(section.footer)
        set_fonts(section.first_page_header)
        set_fonts(section.first_page_footer)
        set_fonts(section.even_page_header)
        set_fonts(section.even_page_footer)

    doc.save(OUTPUT)

    replacements = {
        f"word/media/image{i}.jpg": FIGURES / f"figura{i}.jpg"
        for i in range(1, 7)
    }
    temp = OUTPUT.with_suffix(".tmp.docx")
    with ZipFile(OUTPUT, "r") as src, ZipFile(temp, "w", ZIP_DEFLATED) as dst:
        for item in src.infolist():
            content = replacements[item.filename].read_bytes() if item.filename in replacements else src.read(item.filename)
            dst.writestr(item, content)
    temp.replace(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
