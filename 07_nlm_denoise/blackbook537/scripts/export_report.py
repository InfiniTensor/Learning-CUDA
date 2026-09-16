#!/usr/bin/env python3
"""Export the checked-in Markdown report; no performance values are generated."""
from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
WIDTH = A4[0] - 80


def register_font() -> str:
    candidates = [os.environ.get('PDF_CJK_FONT', ''), 'C:/Windows/Fonts/simsun.ttc',
                  '/usr/share/fonts/truetype/arphic/uming.ttc']
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            pdfmetrics.registerFont(TTFont('ReportCJK', candidate, subfontIndex=0))
            return 'ReportCJK'
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    return 'STSong-Light'


def inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r'`([^`]+)`', r'<font color="#1d4ed8">\1</font>', text)
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'lab_report.md')
    parser.add_argument('--output', type=Path, default=ROOT / 'lab_report.pdf')
    args = parser.parse_args()
    font = register_font()
    base = dict(fontName=font, textColor=colors.HexColor('#253047'), wordWrap='CJK')
    body = ParagraphStyle('Body', fontSize=10, leading=16, spaceAfter=9, **base)
    title = ParagraphStyle('Title', fontSize=23, leading=32, spaceAfter=14, **base)
    heading = ParagraphStyle('Heading', fontSize=15, leading=23, spaceBefore=9,
                             spaceAfter=10, keepWithNext=True, **base)
    cell = ParagraphStyle('Cell', fontSize=9, leading=13, **base)
    code = ParagraphStyle('Code', fontSize=8.3, leading=12.5, spaceAfter=8,
                          leftIndent=8, rightIndent=8, **base)
    caption = ParagraphStyle('Caption', fontSize=8.5, leading=12, spaceAfter=9, **base)
    story = []
    lines = args.input.read_text(encoding='utf-8').splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == '<!-- pagebreak -->':
            story.append(PageBreak())
        elif line.startswith('```'):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                block.append(html.escape(lines[i]).replace(' ', '&#160;'))
                i += 1
            story.append(Paragraph('<br/>'.join(block), code))
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [v.strip() for v in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', v) for v in cells):
                    rows.append(cells)
                i += 1
            ncols = len(rows[0])
            weights = [min(30, max(10, max(len(row[j]) for row in rows))) for j in range(ncols)]
            widths = [WIDTH * w / sum(weights) for w in weights]
            table = Table([[Paragraph(inline(v), cell) for v in row] for row in rows],
                          colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef9')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7f9fc')]),
                ('LINEBELOW', (0, 0), (-1, 0), 0.7, colors.HexColor('#b9c7df')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 7),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
                ('LEFTPADDING', (0, 0), (-1, -1), 7),
                ('RIGHTPADDING', (0, 0), (-1, -1), 7),
            ]))
            story.extend([table, Spacer(1, 10)])
            continue
        elif line.startswith('!['):
            match = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
            if not match:
                raise ValueError(f'Invalid image: {line}')
            asset = args.input.parent / match[2]
            image = Image(str(asset))
            scale = min(WIDTH / image.imageWidth, 265 / image.imageHeight)
            image.drawWidth = image.imageWidth * scale
            image.drawHeight = image.imageHeight * scale
            story.append(KeepTogether([image, Spacer(1, 5), Paragraph(inline(match[1]), caption)]))
        elif line.startswith('# '):
            story.append(Paragraph(inline(line[2:]), title))
        elif line.startswith('## '):
            story.append(Paragraph(inline(line[3:]), heading))
        else:
            story.append(Paragraph(inline(line), body))
        i += 1

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#cbd5e1'))
        canvas.line(40, 34, A4[0]-40, 34)
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(40, 21, 'CUDA NLM / blackbook537 / 2026')
        canvas.drawRightString(A4[0]-40, 21, str(doc.page))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(args.output), pagesize=A4, leftMargin=40, rightMargin=40,
                            topMargin=36, bottomMargin=47, title='CUDA NLM 实验报告', author='blackbook537')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(args.output)


if __name__ == '__main__':
    main()
