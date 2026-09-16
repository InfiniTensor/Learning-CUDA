#!/usr/bin/env python3
"""Render docs/FINAL_REPORT.md as a submission-ready A4 technical report."""

from __future__ import annotations

import argparse
import os
import re
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path

from reportlab.graphics.shapes import (
    Circle,
    Drawing,
    Line,
    Path as GraphicsPath,
    Rect,
    String,
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    PageTemplate,
    Paragraph,
    PageBreak,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "FINAL_REPORT.md"
OUTPUT = ROOT / "output" / "pdf" / "WarpFusion_Technical_Report.pdf"

NAVY = colors.HexColor("#142B4A")
BLUE = colors.HexColor("#275D8C")
LIGHT_BLUE = colors.HexColor("#EAF1F7")
LIGHT_GRAY = colors.HexColor("#F4F5F7")
MID_GRAY = colors.HexColor("#B8C0C8")
TEXT = colors.HexColor("#18212B")


def choose_font(explicit: str | None, environment_name: str,
                candidates: list[str], label: str,
                fallback: str | None = None) -> str:
    requested = explicit or os.environ.get(environment_name)
    if requested:
        path = Path(requested).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{label} font does not exist: {path}")
        return str(path)
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    if fallback is not None:
        return fallback
    raise FileNotFoundError(
        f"cannot find a {label} font; pass the corresponding --*-font option"
    )


def register_fonts(body_font: str | None = None,
                   heading_font: str | None = None,
                   light_font: str | None = None,
                   mono_font: str | None = None) -> dict[str, str]:
    body = choose_font(
        body_font,
        "PURPLEPULSE_REPORT_BODY_FONT",
        [
            "/System/Library/Fonts/Supplemental/Songti.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ],
        "CJK body",
    )
    heading = choose_font(
        heading_font,
        "PURPLEPULSE_REPORT_HEADING_FONT",
        [
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ],
        "CJK heading",
        fallback=body,
    )
    light = choose_font(
        light_font,
        "PURPLEPULSE_REPORT_LIGHT_FONT",
        [
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ],
        "CJK light",
        fallback=body,
    )
    mono = choose_font(
        mono_font,
        "PURPLEPULSE_REPORT_MONO_FONT",
        [
            "/System/Library/Fonts/SFNSMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        ],
        "monospace",
        fallback=light,
    )
    fonts = {"BodyCN": body, "HeadCN": heading, "LightCN": light, "Mono": mono}
    # macOS CJK fonts are TrueType collections.  Their first faces are not the
    # simplified-Chinese regular faces: Songti index 0 is Black and the Heiti
    # collections start with Traditional Chinese.  Select the intended faces
    # explicitly so simplified glyphs and body weight remain consistent.
    collection_faces = {
        ("BodyCN", "/System/Library/Fonts/Supplemental/Songti.ttc"): 6,
        ("HeadCN", "/System/Library/Fonts/STHeiti Medium.ttc"): 1,
        ("LightCN", "/System/Library/Fonts/STHeiti Light.ttc"): 1,
    }
    for name, path in fonts.items():
        subfont_index = collection_faces.get((name, str(Path(path))), 0)
        pdfmetrics.registerFont(TTFont(name, path, subfontIndex=subfont_index))
    return fonts


def inline_markup(text: str) -> str:
    placeholders: list[str] = []

    # Keep punctuation portable across PDF renderers and submission systems.
    text = text.replace("–", "-").replace("—", "-").replace("‑", "-")

    def save_code(match: re.Match[str]) -> str:
        placeholders.append(
            f'<font name="Mono" color="#173A5E">{escape(match.group(1))}</font>'
        )
        return f"@@CODE{len(placeholders) - 1}@@"

    text = re.sub(r"`([^`]+)`", save_code, text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    for index, value in enumerate(placeholders):
        text = text.replace(f"@@CODE{index}@@", value)
    return text


def equation_text(text: str) -> str:
    replacements = {
        r"\sum_j": "Σ_j",
        r"\cdot": "·",
        r"\qquad": "    ",
        r"\lVert": "||",
        r"\rVert": "||",
        r"\frac{q\cdot x}{\lVert q\rVert_2\lVert x\rVert_2}":
            "(q · x) / (||q||₂ ||x||₂)",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("_{L2}", "_L2").replace("_{IP}", "_IP")
    text = text.replace("^2", "²")
    text = text.replace("\\", "")
    return text.strip()


class Rule(Flowable):
    def __init__(self, width: float, color: colors.Color = BLUE) -> None:
        super().__init__()
        self.width = width
        self.height = 1.2
        self.color = color

    def draw(self) -> None:
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(1.2)
        self.canv.line(0, 0, self.width, 0)


class TechnicalReport(BaseDocTemplate):
    def __init__(self, filename: str) -> None:
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=19 * mm,
            rightMargin=19 * mm,
            topMargin=19 * mm,
            bottomMargin=18 * mm,
            title="WarpFusion：面向大规模向量检索的 CUDA 并行设计与性能优化",
            author="花生 (PurplePulse)",
            subject="CUDA GPU 向量检索技术报告",
        )
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="body",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=self.header_footer))

    def header_footer(self, canvas, doc) -> None:  # noqa: ANN001
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(MID_GRAY)
            canvas.setLineWidth(0.35)
            canvas.line(19 * mm, A4[1] - 13 * mm, A4[0] - 19 * mm, A4[1] - 13 * mm)
            canvas.setFont("LightCN", 7.4)
            canvas.setFillColor(colors.HexColor("#59636E"))
            canvas.drawString(19 * mm, A4[1] - 10.5 * mm, "WarpFusion CUDA 向量检索技术报告")
            canvas.drawRightString(A4[0] - 19 * mm, A4[1] - 10.5 * mm, "花生 · ID PurplePulse")
        canvas.setFont("LightCN", 7.4)
        canvas.setFillColor(colors.HexColor("#68727C"))
        canvas.drawCentredString(A4[0] / 2, 9 * mm, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable: Flowable) -> None:
        if isinstance(flowable, Paragraph):
            level = getattr(flowable, "heading_level", None)
            if level in (1, 2):
                key = f"section-{self.seq.nextf('section')}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(flowable.getPlainText(), key, level - 1, False)


def make_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=sample["Title"],
            fontName="HeadCN",
            fontSize=22,
            leading=30,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=8 * mm,
        ),
        "author": ParagraphStyle(
            "Author",
            fontName="LightCN",
            fontSize=10.5,
            leading=16,
            textColor=colors.HexColor("#3D4955"),
            spaceAfter=7 * mm,
        ),
        "h1": ParagraphStyle(
            "Heading1CN",
            fontName="HeadCN",
            fontSize=14.2,
            leading=20,
            textColor=NAVY,
            spaceBefore=7 * mm,
            spaceAfter=3.2 * mm,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Heading2CN",
            fontName="HeadCN",
            fontSize=11.2,
            leading=16,
            textColor=BLUE,
            spaceBefore=5 * mm,
            spaceAfter=2.2 * mm,
            keepWithNext=True,
        ),
        "references_heading": ParagraphStyle(
            "ReferencesHeading",
            fontName="HeadCN",
            fontSize=12.2,
            leading=17,
            textColor=NAVY,
            spaceBefore=3 * mm,
            spaceAfter=1.8 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            fontName="BodyCN",
            fontSize=9.1,
            leading=14.2,
            textColor=TEXT,
            alignment=TA_JUSTIFY,
            firstLineIndent=2 * 9.1,
            spaceAfter=2.1 * mm,
            wordWrap="CJK",
        ),
        "abstract": ParagraphStyle(
            "AbstractCN",
            fontName="BodyCN",
            fontSize=8.9,
            leading=13.8,
            textColor=TEXT,
            alignment=TA_JUSTIFY,
            firstLineIndent=2 * 8.9,
            spaceAfter=2 * mm,
            leftIndent=5 * mm,
            rightIndent=5 * mm,
            wordWrap="CJK",
        ),
        "list": ParagraphStyle(
            "ListCN",
            fontName="BodyCN",
            fontSize=9.1,
            leading=14.2,
            textColor=TEXT,
            alignment=TA_JUSTIFY,
            leftIndent=7 * mm,
            firstLineIndent=-4.5 * mm,
            spaceAfter=1.5 * mm,
            wordWrap="CJK",
        ),
        "references": ParagraphStyle(
            "References",
            fontName="BodyCN",
            fontSize=8.0,
            leading=10.8,
            textColor=TEXT,
            alignment=TA_LEFT,
            leftIndent=5 * mm,
            firstLineIndent=-3.5 * mm,
            spaceAfter=0.8 * mm,
        ),
        "caption": ParagraphStyle(
            "CaptionCN",
            fontName="LightCN",
            fontSize=7.8,
            leading=11,
            textColor=colors.HexColor("#4E5965"),
            alignment=TA_CENTER,
            spaceBefore=1.5 * mm,
            spaceAfter=3 * mm,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "CodeCN",
            fontName="LightCN",
            fontSize=7.7,
            leading=11.2,
            leftIndent=4 * mm,
            rightIndent=4 * mm,
            borderColor=MID_GRAY,
            borderWidth=0.45,
            borderPadding=4,
            backColor=LIGHT_GRAY,
            spaceBefore=1.5 * mm,
            spaceAfter=3 * mm,
            wordWrap="CJK",
        ),
        "equation": ParagraphStyle(
            "Equation",
            fontName="BodyCN",
            fontSize=9.4,
            leading=14,
            textColor=TEXT,
            alignment=TA_CENTER,
            spaceBefore=1.5 * mm,
            spaceAfter=2.5 * mm,
        ),
        "table": ParagraphStyle(
            "TableCell",
            fontName="BodyCN",
            fontSize=7.1,
            leading=9.7,
            textColor=TEXT,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            fontName="HeadCN",
            fontSize=7.1,
            leading=9.7,
            textColor=colors.white,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
    }


def build_table(rows: list[list[str]], width: float, styles: dict[str, ParagraphStyle]) -> Table:
    column_count = max(len(row) for row in rows)
    normalized = [row + [""] * (column_count - len(row)) for row in rows]
    data = []
    for row_index, row in enumerate(normalized):
        style = styles["table_head"] if row_index == 0 else styles["table"]
        data.append([Paragraph(inline_markup(cell.strip()), style) for cell in row])
    weights = [1.0] * column_count
    if column_count >= 3:
        weights[-1] = 1.35
    total = sum(weights)
    column_widths = [width * value / total for value in weights]
    table = Table(data, colWidths=column_widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, MID_GRAY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
                ("TOPPADDING", (0, 0), (-1, -1), 3.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
            ]
            + [
                ("BACKGROUND", (0, row), (-1, row), LIGHT_BLUE)
                for row in range(2, len(data), 2)
            ]
        )
    )
    return table


def simple_svg_drawing(path: Path) -> Drawing:
    """Convert the chart SVG subset used by this report into ReportLab vectors."""
    root = ET.parse(path).getroot()
    width = float(root.attrib["width"].removesuffix("px"))
    height = float(root.attrib["height"].removesuffix("px"))
    drawing = Drawing(width, height)
    text_styles = {
        "tick": (16, "LightCN", TEXT),
        "label": (18, "HeadCN", TEXT),
        "panel-title": (21, "HeadCN", TEXT),
        "figure-title": (26, "HeadCN", TEXT),
        "note": (15, "LightCN", colors.HexColor("#5F6368")),
    }
    line_styles = {
        "grid": (colors.HexColor("#D9DDE3"), 1.0),
        "axis": (colors.HexColor("#51565C"), 1.2),
    }

    def number(value: str | None, default: float = 0.0) -> float:
        return float(value) if value is not None else default

    for element in root:
        tag = element.tag.rsplit("}", 1)[-1]
        attrs = element.attrib
        css_class = attrs.get("class", "")
        if tag in {"title", "desc", "style"}:
            continue
        if tag == "rect":
            rect_width = width if attrs.get("width") == "100%" else number(attrs.get("width"))
            rect_height = height if attrs.get("height") == "100%" else number(attrs.get("height"))
            x = number(attrs.get("x"))
            y = height - number(attrs.get("y")) - rect_height
            drawing.add(
                Rect(
                    x,
                    y,
                    rect_width,
                    rect_height,
                    fillColor=colors.HexColor(attrs.get("fill", "#FFFFFF")),
                    strokeColor=None,
                )
            )
            continue
        if tag == "line":
            stroke, stroke_width = line_styles.get(
                css_class,
                (
                    colors.HexColor(attrs.get("stroke", "#000000")),
                    number(attrs.get("stroke-width"), 1.0),
                ),
            )
            drawing.add(
                Line(
                    number(attrs.get("x1")),
                    height - number(attrs.get("y1")),
                    number(attrs.get("x2")),
                    height - number(attrs.get("y2")),
                    strokeColor=stroke,
                    strokeWidth=stroke_width,
                )
            )
            continue
        if tag == "circle":
            drawing.add(
                Circle(
                    number(attrs.get("cx")),
                    height - number(attrs.get("cy")),
                    number(attrs.get("r")),
                    fillColor=colors.HexColor(attrs.get("fill", "#000000")),
                    strokeColor=colors.HexColor(attrs.get("stroke", "#000000")),
                    strokeWidth=number(attrs.get("stroke-width"), 1.0),
                )
            )
            continue
        if tag == "path":
            tokens = re.findall(r"[ML]|-?\d+(?:\.\d+)?", attrs.get("d", ""))
            graphic_path = GraphicsPath()
            cursor = 0
            command = "M"
            while cursor < len(tokens):
                if tokens[cursor] in {"M", "L"}:
                    command = tokens[cursor]
                    cursor += 1
                x = float(tokens[cursor])
                y = height - float(tokens[cursor + 1])
                cursor += 2
                if command == "M":
                    graphic_path.moveTo(x, y)
                    command = "L"
                else:
                    graphic_path.lineTo(x, y)
            graphic_path.fillColor = None
            graphic_path.strokeColor = colors.HexColor(attrs.get("stroke", "#000000"))
            graphic_path.strokeWidth = number(attrs.get("stroke-width"), 1.0)
            drawing.add(graphic_path)
            continue
        if tag == "text":
            font_size, font_name, fill = text_styles.get(css_class, (11, "LightCN", TEXT))
            angle = 90 if "rotate(-90" in attrs.get("transform", "") else 0
            value = "".join(element.itertext()).replace("–", "-").replace("—", "-")
            drawing.add(
                String(
                    number(attrs.get("x")),
                    height - number(attrs.get("y")),
                    value,
                    fontName=font_name,
                    fontSize=font_size,
                    fillColor=fill,
                    textAnchor=attrs.get("text-anchor", "start"),
                    angle=angle,
                )
            )
    return drawing


def add_svg(story: list[Flowable], path: Path, max_width: float, max_height: float) -> bool:
    if path.name.startswith("report-"):
        drawing = simple_svg_drawing(path)
    else:
        try:
            from svglib.svglib import svg2rlg
        except ImportError:
            drawing = simple_svg_drawing(path)
        else:
            drawing = svg2rlg(str(path))
    if drawing is None:
        return False
    scale = min(max_width / drawing.width, max_height / drawing.height, 1.0)
    drawing.width *= scale
    drawing.height *= scale
    drawing.scale(scale, scale)
    drawing.hAlign = "CENTER"
    story.append(Spacer(1, 2 * mm))
    story.append(drawing)
    return True


def parse_markdown(text: str, doc: TechnicalReport, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    lines = text.splitlines()
    story: list[Flowable] = []
    story.append(Spacer(1, 11 * mm))
    story.append(Paragraph(inline_markup(lines[0].removeprefix("# ")), styles["title"]))
    story.append(Rule(doc.width))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("作者：花生　　学员 ID：PurplePulse", styles["author"]))

    index = 2
    paragraph_lines: list[str] = []
    in_code = False
    code_lines: list[str] = []
    in_equation = False
    equation_lines: list[str] = []
    abstract_mode = False
    references_mode = False

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        value = " ".join(part.strip() for part in paragraph_lines)
        style = styles["abstract"] if abstract_mode else styles["body"]
        story.append(Paragraph(inline_markup(value), style))
        paragraph_lines.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            if in_code:
                story.append(
                    Paragraph("<br/>".join(escape(item) for item in code_lines), styles["code"])
                )
                code_lines.clear()
                in_code = False
            else:
                in_code = True
            index += 1
            continue
        if in_code:
            code_lines.append(line)
            index += 1
            continue

        if stripped == "$$":
            flush_paragraph()
            if in_equation:
                rendered = "<br/>".join(
                    escape(equation_text(item)) for item in equation_lines if item.strip()
                )
                story.append(Paragraph(rendered, styles["equation"]))
                equation_lines.clear()
                in_equation = False
            else:
                in_equation = True
            index += 1
            continue
        if in_equation:
            equation_lines.append(line)
            index += 1
            continue

        if stripped.startswith("!["):
            flush_paragraph()
            match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
            if match:
                image_path = (SOURCE.parent / match.group(2)).resolve()
                add_svg(story, image_path, doc.width * 0.88, 82 * mm)
            index += 1
            continue

        if stripped.startswith("**图"):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped), styles["caption"]))
            index += 1
            continue

        if stripped.startswith("## "):
            flush_paragraph()
            title = stripped[3:].strip()
            abstract_mode = title == "摘要"
            references_mode = title == "参考文献"
            heading_style = styles["references_heading"] if references_mode else styles["h1"]
            paragraph = Paragraph(inline_markup(title), heading_style)
            paragraph.heading_level = 1
            story.append(paragraph)
            index += 1
            continue

        if stripped.startswith("### "):
            flush_paragraph()
            abstract_mode = False
            paragraph = Paragraph(inline_markup(stripped[4:].strip()), styles["h2"])
            paragraph.heading_level = 2
            story.append(paragraph)
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and re.match(
            r"^\s*\|?\s*:?-+", lines[index + 1]
        ):
            flush_paragraph()
            rows: list[list[str]] = []
            rows.append([cell for cell in stripped.strip("|").split("|")])
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append([cell for cell in lines[index].strip().strip("|").split("|")])
                index += 1
            story.append(Spacer(1, 1.5 * mm))
            story.append(build_table(rows, doc.width, styles))
            story.append(Spacer(1, 2.8 * mm))
            continue

        list_match = re.match(r"^(\d+)\.\s+(.*)", stripped)
        if list_match:
            flush_paragraph()
            item_parts = [list_match.group(2)]
            item_number = list_match.group(1)
            index += 1
            while index < len(lines) and lines[index].startswith("   "):
                continuation = lines[index].strip()
                if continuation:
                    item_parts.append(continuation)
                index += 1
            story.append(
                Paragraph(
                    f"{item_number}.　{inline_markup(' '.join(item_parts))}",
                    styles["references"] if references_mode else styles["list"],
                )
            )
            continue

        if not stripped:
            flush_paragraph()
        elif index == 2 and stripped.startswith("**作者"):
            pass
        else:
            paragraph_lines.append(stripped)
        index += 1

    flush_paragraph()
    return story


def main() -> None:
    global SOURCE, OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--body-font")
    parser.add_argument("--heading-font")
    parser.add_argument("--light-font")
    parser.add_argument("--mono-font")
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    OUTPUT = args.output.resolve()
    from build_report_figures import generate_report_figures

    generate_report_figures()
    register_fonts(args.body_font, args.heading_font, args.light_font,
                   args.mono_font)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = TechnicalReport(str(OUTPUT))
    styles = make_styles()
    story = parse_markdown(SOURCE.read_text(encoding="utf-8"), doc, styles)
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
