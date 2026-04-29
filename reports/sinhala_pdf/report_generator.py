"""Sinhala PDF report generator for fabric defect shift summaries using ReportLab."""

from datetime import datetime
from pathlib import Path
from typing import List, Dict

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


SINHALA_LABELS = {
    "oil_stain": "තෙල් 얼룩",
    "dye_stain": "ඩයි얼룩",
    "hole_snag": "සිදුර / ඇදීම",
    "drop_stitch": "ලූප මඟහැරීම",
    "weave_distortion": "වියමන් විකෘතිය",
    "slub_nep": "නූල් ගැටිත්ත",
    "shade_variation": "වර්ණ විචලනය",
    "shrinkage": "හැකිළීම",
}

REPORT_TITLE = "රෙදි දෝෂ වාර්තාව"
SHIFT_LABEL = "නිෂ්පාදන අවස්ථාව"
DATE_LABEL = "දිනය"
TOTAL_DEFECTS_LABEL = "මුළු දෝෂ"
DEFECT_TYPE_LABEL = "දෝෂ වර්ගය"
COUNT_LABEL = "ගණන"
CONFIDENCE_LABEL = "විශ්වාසය"


def generate_report(
    defect_counts: Dict[str, int],
    avg_confidence: Dict[str, float],
    output_path: str,
    shift_id: str = "Shift-1",
) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=2 * cm, leftMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=18, spaceAfter=12)
    normal_style = styles["Normal"]

    elements = [
        Paragraph(REPORT_TITLE, title_style),
        Spacer(1, 0.4 * cm),
        Paragraph(f"{SHIFT_LABEL}: {shift_id}", normal_style),
        Paragraph(f"{DATE_LABEL}: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style),
        Paragraph(f"{TOTAL_DEFECTS_LABEL}: {sum(defect_counts.values())}", normal_style),
        Spacer(1, 0.6 * cm),
    ]

    table_data = [[DEFECT_TYPE_LABEL, COUNT_LABEL, CONFIDENCE_LABEL]]
    for defect_class, count in sorted(defect_counts.items(), key=lambda x: -x[1]):
        sinhala_name = SINHALA_LABELS.get(defect_class, defect_class)
        conf = avg_confidence.get(defect_class, 0.0)
        table_data.append([sinhala_name, str(count), f"{conf:.1%}"])

    table = Table(table_data, colWidths=[8 * cm, 4 * cm, 4 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, 0), 12),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(table)

    doc.build(elements)
    return output_path
