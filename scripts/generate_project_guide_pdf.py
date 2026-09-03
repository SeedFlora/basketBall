#!/usr/bin/env python
"""Render the Indonesian beginner guide as a polished, evidence-backed PDF."""

from __future__ import annotations

import argparse
import html
import re
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    HRFlowable,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "docs" / "PANDUAN_PROYEK.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "pdf" / "panduan_proyek_basketball_pose.pdf"

NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#2563EB")
SKY = colors.HexColor("#E8F1FF")
CYAN = colors.HexColor("#18A4A6")
INK = colors.HexColor("#243B53")
MUTED = colors.HexColor("#627D98")
LINE = colors.HexColor("#D9E2EC")
PAPER = colors.HexColor("#F7F9FC")
ORANGE = colors.HexColor("#D97706")


def _register_fonts() -> tuple[str, str, str, str]:
    candidates = [
        (
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/ariali.ttf"),
            Path("C:/Windows/Fonts/arialbi.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf"),
        ),
    ]
    for regular, bold, italic, bold_italic in candidates:
        if all(path.is_file() for path in (regular, bold, italic, bold_italic)):
            pdfmetrics.registerFont(TTFont("GuideSans", str(regular)))
            pdfmetrics.registerFont(TTFont("GuideSans-Bold", str(bold)))
            pdfmetrics.registerFont(TTFont("GuideSans-Italic", str(italic)))
            pdfmetrics.registerFont(TTFont("GuideSans-BoldItalic", str(bold_italic)))
            pdfmetrics.registerFontFamily(
                "GuideSans",
                normal="GuideSans",
                bold="GuideSans-Bold",
                italic="GuideSans-Italic",
                boldItalic="GuideSans-BoldItalic",
            )
            return (
                "GuideSans",
                "GuideSans-Bold",
                "GuideSans-Italic",
                "GuideSans-BoldItalic",
            )
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique"


FONT, FONT_BOLD, FONT_ITALIC, _ = _register_fonts()


class GuideDocTemplate(BaseDocTemplate):
    """Document template with outline entries and a live table of contents."""

    def afterFlowable(self, flowable: Flowable) -> None:
        if not isinstance(flowable, Paragraph):
            return
        levels = {"GuideH1": 0, "GuideH2": 1, "GuideH3": 2}
        level = levels.get(flowable.style.name)
        if level is None:
            return
        text = flowable.getPlainText()
        key = f"section-{self.seq.nextf('section')}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
        self.notify("TOCEntry", (level, text, self.page, key))


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_label": ParagraphStyle(
            "CoverLabel",
            parent=base["Normal"],
            fontName=FONT_BOLD,
            fontSize=10,
            leading=13,
            textColor=CYAN,
            alignment=TA_CENTER,
            spaceAfter=7,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName=FONT_BOLD,
            fontSize=27,
            leading=33,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=15,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=12,
            leading=18,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=22,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=9.5,
            leading=15,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "GuideH1",
            parent=base["Heading1"],
            fontName=FONT_BOLD,
            fontSize=18,
            leading=23,
            textColor=NAVY,
            spaceBefore=12,
            spaceAfter=9,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "GuideH2",
            parent=base["Heading2"],
            fontName=FONT_BOLD,
            fontSize=13.5,
            leading=18,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "GuideH3",
            parent=base["Heading3"],
            fontName=FONT_BOLD,
            fontSize=11.2,
            leading=15,
            textColor=CYAN,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "GuideBody",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.2,
            leading=13.5,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=5,
            allowWidows=0,
            allowOrphans=0,
        ),
        "bullet": ParagraphStyle(
            "GuideBullet",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9,
            leading=13,
            textColor=INK,
            leftIndent=14,
            firstLineIndent=-8,
            spaceAfter=3,
        ),
        "quote": ParagraphStyle(
            "GuideQuote",
            parent=base["BodyText"],
            fontName=FONT_ITALIC,
            fontSize=9.3,
            leading=14,
            textColor=NAVY,
            leftIndent=13,
            rightIndent=8,
            borderColor=CYAN,
            borderWidth=0,
            borderPadding=(7, 8, 7, 10),
            backColor=SKY,
            spaceBefore=4,
            spaceAfter=7,
        ),
        "code": ParagraphStyle(
            "GuideCode",
            parent=base["Code"],
            fontName="Courier",
            fontSize=6.9,
            leading=9.4,
            textColor=colors.HexColor("#102A43"),
            backColor=colors.HexColor("#EEF2F6"),
            borderColor=LINE,
            borderWidth=0.4,
            borderPadding=7,
            leftIndent=4,
            rightIndent=4,
            spaceBefore=3,
            spaceAfter=7,
            splitLongWords=True,
        ),
        "caption": ParagraphStyle(
            "GuideCaption",
            parent=base["Normal"],
            fontName=FONT_ITALIC,
            fontSize=7.8,
            leading=10.5,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=10,
        ),
        "small": ParagraphStyle(
            "GuideSmall",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
        ),
    }


def _clean_text(value: str) -> str:
    return (
        value.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\u2192", "->")
    )


def _inline(value: str) -> str:
    escaped = html.escape(_clean_text(value), quote=True)
    escaped = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r"<link href='\2' color='#2563EB'>\1</link>",
        escaped,
    )
    return escaped


def _is_table_separator(line: str) -> bool:
    cells = [part.strip() for part in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _parse_table(lines: list[str], styles: dict[str, ParagraphStyle], width: float) -> Table:
    rows: list[list[str]] = []
    for line in lines:
        if _is_table_separator(line):
            continue
        rows.append([part.strip() for part in line.strip().strip("|").split("|")])
    column_count = max(len(row) for row in rows)
    for row in rows:
        row.extend([""] * (column_count - len(row)))
    font_size = 7.4 if column_count <= 4 else 6.4
    leading = font_size + 2.2
    cell = ParagraphStyle(
        "TableCell",
        parent=styles["body"],
        fontName=FONT,
        fontSize=font_size,
        leading=leading,
        spaceAfter=0,
    )
    header = ParagraphStyle(
        "TableHeader",
        parent=cell,
        fontName=FONT_BOLD,
        textColor=colors.white,
    )
    data = [
        [Paragraph(_inline(value), header if index == 0 else cell) for value in row]
        for index, row in enumerate(rows)
    ]
    column_widths = [width / column_count] * column_count
    table = Table(data, colWidths=column_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PAPER]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _markdown_flowables(
    source: Path, styles: dict[str, ParagraphStyle], width: float
) -> list[Flowable]:
    lines = source.read_text(encoding="utf-8").splitlines()
    flowables: list[Flowable] = []
    index = 0
    skipped_title = False
    while index < len(lines):
        raw = lines[index].rstrip()
        stripped = raw.strip()
        if not stripped:
            index += 1
            continue
        if stripped.casefold() == "## daftar isi":
            index += 1
            while index < len(lines) and not re.match(r"^##\s+1\.", lines[index].strip()):
                index += 1
            continue
        if stripped in {"---", "***"}:
            flowables.append(
                HRFlowable(width="100%", thickness=0.45, color=LINE, spaceBefore=5, spaceAfter=7)
            )
            index += 1
            continue
        image_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            caption, target = image_match.groups()
            image_path = (source.parent / target).resolve()
            if image_path.is_file():
                flowables.extend(
                    [
                        CondPageBreak(115 * mm),
                        _scaled_image(image_path, width, 102 * mm),
                        Paragraph(_inline(caption), styles["caption"]),
                    ]
                )
            else:
                flowables.append(
                    Paragraph(
                        f"Visual tidak tersedia: <font face='Courier'>{html.escape(target)}</font>",
                        styles["quote"],
                    )
                )
            index += 1
            continue
        if stripped.startswith("```"):
            block: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index].rstrip())
                index += 1
            index += 1
            code_html = "<br/>".join(html.escape(_clean_text(line)) or "&nbsp;" for line in block)
            flowables.append(Paragraph(code_html, styles["code"]))
            continue
        if stripped.startswith("|") and "|" in stripped[1:]:
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            flowables.extend([_parse_table(table_lines, styles, width), Spacer(1, 7)])
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            index += 1
            if level == 1 and not skipped_title:
                skipped_title = True
                continue
            if title.casefold().startswith("dari konsep paling dasar"):
                continue
            if level == 1:
                flowables.extend([PageBreak(), Paragraph(_inline(title), styles["h1"])])
            elif level == 2:
                flowables.extend([CondPageBreak(32 * mm), Paragraph(_inline(title), styles["h1"])])
            elif level == 3:
                flowables.append(Paragraph(_inline(title), styles["h2"]))
            else:
                flowables.append(Paragraph(_inline(title), styles["h3"]))
            continue
        if stripped.startswith(">"):
            quote_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(lines[index].strip().lstrip("> "))
                index += 1
            flowables.append(Paragraph(_inline(" ".join(quote_lines)), styles["quote"]))
            continue
        bullet = re.match(r"^[-*]\s+(.+)$", stripped)
        numbered = re.match(r"^(\d+)[.)]\s+(.+)$", stripped)
        if bullet or numbered:
            marker = "-" if bullet else f"{numbered.group(1)}."
            text = bullet.group(1) if bullet else numbered.group(2)
            flowables.append(Paragraph(f"{marker}&nbsp;&nbsp;{_inline(text)}", styles["bullet"]))
            index += 1
            continue
        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if (
                not candidate
                or candidate.startswith(("#", "```", ">", "|"))
                or re.match(r"^[-*]\s+", candidate)
                or re.match(r"^\d+[.)]\s+", candidate)
            ):
                break
            paragraph_lines.append(candidate)
            index += 1
        flowables.append(Paragraph(_inline(" ".join(paragraph_lines)), styles["body"]))
    return flowables


def _scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    image = Image(str(path))
    ratio = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * ratio
    image.drawHeight = image.imageHeight * ratio
    image.hAlign = "CENTER"
    return image


def _visual_appendix(styles: dict[str, ParagraphStyle], width: float) -> list[Flowable]:
    figure_root = PROJECT_ROOT / "outputs" / "figures" / "reconstruction"
    figures = [
        (
            "skeleton_examples_by_domain.png",
            "Contoh skeleton dari Indoor, Outdoor, dan Drone. Koordinat adalah anotasi 2D, bukan estimasi 3D.",
        ),
        (
            "example_10_joint_skeleton.png",
            "Satu contoh topology 10 joint TrackID3x3 yang dipakai oleh pipeline.",
        ),
        (
            "normalized_skeleton.png",
            "Skeleton setelah transformasi normalisasi untuk mengurangi perbedaan posisi dan skala kamera.",
        ),
        (
            "normalization_ablation.png",
            "Ablation representasi dan uji shift Indoor-ke-Outdoor. Skala log membantu menampilkan gap raw yang besar.",
        ),
        (
            "error_vs_missing_percentage.png",
            "NMJE dan PCK@0.10 ketika proporsi joint valid yang disembunyikan dinaikkan.",
        ),
        (
            "nmje_by_model.png",
            "Perbandingan model terkontrol pada window T=16 Indoor/Outdoor.",
        ),
        (
            "cross_domain_matrix.png",
            "Matriks generalisasi lintas domain. Sel yang menuju Drone harus dibaca bersama caveat bbox Drone.",
        ),
        (
            "data_efficiency_curve.png",
            "Hubungan jumlah data latih berbasis whole-group dengan error rekonstruksi.",
        ),
    ]
    story: list[Flowable] = [
        PageBreak(),
        Paragraph("Lampiran visual", styles["h1"]),
        Paragraph(
            "Seluruh gambar berikut dihasilkan dari anotasi dan hasil eksperimen nyata. "
            "Tidak ada overlay RGB karena frame/video sumber tidak diunduh.",
            styles["body"],
        ),
    ]
    for filename, caption in figures:
        path = figure_root / filename
        if not path.is_file():
            continue
        story.extend(
            [
                CondPageBreak(115 * mm),
                _scaled_image(path, width, 105 * mm),
                Paragraph(_inline(caption), styles["caption"]),
            ]
        )
    story.extend(
        [
            Spacer(1, 6),
            Table(
                [[Paragraph("RGB overlay tidak dibuat", styles["h2"])],
                 [Paragraph(
                     "Akuisisi lokal bersifat annotation-only. Remote media tidak diunduh karena ukuran file "
                     "tidak dapat diverifikasi aman terhadap kebijakan storage. Tidak ada gambar RGB yang "
                     "direkayasa untuk menggantikan data yang tidak tersedia.",
                     styles["body"],
                 )]],
                colWidths=[width],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FFF7E6")),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#FFFCF5")),
                        ("BOX", (0, 0), (-1, -1), 0.7, ORANGE),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                ),
            ),
        ]
    )
    return story


def _cover(styles: dict[str, ParagraphStyle], width: float) -> list[Flowable]:
    badge = Table(
        [[Paragraph("PANDUAN PEMULA + LAPORAN HASIL NYATA", styles["cover_label"])]],
        colWidths=[90 * mm],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SKY),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#B8D3FF")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        ),
        hAlign="CENTER",
    )
    overview = Table(
        [
            [Paragraph("DATA", styles["small"]), Paragraph("METODE", styles["small"]), Paragraph("HASIL", styles["small"])],
            [
                Paragraph("40,134 pose sumber<br/>2,500 pose subset", styles["body"]),
                Paragraph("Masked keypoint<br/>reconstruction", styles["body"]),
                Paragraph("Micro: NMJE 0.0672<br/>PCK@0.10 0.8052", styles["body"]),
            ],
        ],
        colWidths=[width / 3] * 3,
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, 1), PAPER),
                ("GRID", (0, 0), (-1, -1), 0.5, LINE),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        ),
    )
    return [
        Spacer(1, 25 * mm),
        badge,
        Spacer(1, 13 * mm),
        Paragraph(
            "Rekonstruksi Keypoint yang Disembunyikan pada Pose Pemain Basket 3x3",
            styles["cover_title"],
        ),
        Paragraph(
            "Panduan lengkap dari nol untuk memahami data TrackID3x3, normalisasi skeleton, "
            "model ringan, evaluasi lintas domain, hasil eksperimen, dan istilah penting.",
            styles["cover_subtitle"],
        ),
        HRFlowable(width="72%", thickness=2, color=CYAN, spaceBefore=2, spaceAfter=15),
        overview,
        Spacer(1, 17 * mm),
        Paragraph(
            "Status: eksperimen selesai dan quality gate lulus<br/>"
            "Protokol: seed 42, annotation-only, single-seed exploratory<br/>"
            f"Dibuat: {datetime.now().astimezone().date().isoformat()}",
            styles["cover_meta"],
        ),
        Spacer(1, 12 * mm),
        Paragraph(
            "Batas penting: proyek ini bukan action recognition, bukan penilaian teknik basket, "
            "dan bukan analisis biomekanik atau pose 3D.",
            styles["quote"],
        ),
        NextPageTemplate("body"),
        PageBreak(),
    ]


def _toc(styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOC1",
            fontName=FONT_BOLD,
            fontSize=9.5,
            leading=14,
            leftIndent=0,
            firstLineIndent=0,
            textColor=NAVY,
            spaceBefore=3,
        ),
        ParagraphStyle(
            "TOC2",
            fontName=FONT,
            fontSize=8.5,
            leading=12,
            leftIndent=12,
            firstLineIndent=0,
            textColor=INK,
        ),
        ParagraphStyle(
            "TOC3",
            fontName=FONT,
            fontSize=7.8,
            leading=10.5,
            leftIndent=24,
            firstLineIndent=0,
            textColor=MUTED,
        ),
    ]
    return [
        Paragraph("Daftar isi", styles["h1"]),
        Paragraph(
            "Gunakan bagian konsep dan glosarium bila Anda baru mengenal computer vision. "
            "Bagian hasil selalu menyebut jenis agregasi agar angka tidak tertukar.",
            styles["body"],
        ),
        Spacer(1, 5),
        toc,
        PageBreak(),
    ]


def _page_header_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, A4[1] - 24 * mm, A4[0] - doc.rightMargin, A4[1] - 24 * mm)
    canvas.setFont(FONT_BOLD, 7.5)
    canvas.setFillColor(NAVY)
    canvas.drawString(doc.leftMargin, A4[1] - 19 * mm, "basketBall | TrackID3x3 masked-keypoint reconstruction")
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(A4[0] - doc.rightMargin, 14 * mm, f"Halaman {doc.page}")
    canvas.drawString(doc.leftMargin, 14 * mm, "Seed 42 | Hasil eksploratif satu seed | Annotation-only")
    canvas.restoreState()


def _cover_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#F7FAFF"))
    canvas.rect(0, 0, A4[0], A4[1], stroke=0, fill=1)
    canvas.setFillColor(NAVY)
    canvas.rect(0, A4[1] - 12 * mm, A4[0], 12 * mm, stroke=0, fill=1)
    canvas.setFillColor(CYAN)
    canvas.rect(0, 0, A4[0], 7 * mm, stroke=0, fill=1)
    canvas.restoreState()


def build_pdf(source: Path, output: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Guide source not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    page_width, page_height = A4
    left = right = 18 * mm
    top = 30 * mm
    bottom = 22 * mm
    usable_width = page_width - left - right
    frame = Frame(left, bottom, usable_width, page_height - top - bottom, id="body-frame")
    cover_frame = Frame(left, 12 * mm, usable_width, page_height - 24 * mm, id="cover-frame")
    doc = GuideDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=left,
        rightMargin=right,
        topMargin=top,
        bottomMargin=bottom,
        title="Panduan Proyek basketBall - TrackID3x3 Masked-Keypoint Reconstruction",
        author="basketBall project",
        subject="Data, metode, hasil, reproduksi, dan glosarium untuk pemula",
        creator="scripts/generate_project_guide_pdf.py",
    )
    doc.addPageTemplates(
        [
            PageTemplate(id="cover", frames=[cover_frame], onPage=_cover_page),
            PageTemplate(id="body", frames=[frame], onPage=_page_header_footer),
        ]
    )
    story: list[Flowable] = []
    story.extend(_cover(styles, usable_width))
    story.extend(_toc(styles))
    story.extend(_markdown_flowables(source, styles, usable_width))
    if not re.search(r"^!\[[^\]]*\]\([^)]+\)$", source.read_text(encoding="utf-8"), re.MULTILINE):
        story.extend(_visual_appendix(styles, usable_width))
    doc.multiBuild(story)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    build_pdf(source, output)
    print(f"Generated: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
