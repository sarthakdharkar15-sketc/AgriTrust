"""PDF Report Generation for AgriTrust Advisory Decisions.
Produces a clean, publication-quality agronomic advisory report using ReportLab.
"""

import io
from datetime import datetime
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether
)


def generate_advisory_pdf(
    query: str,
    pipeline_result: Dict[str, Any],
    user_context: Optional[Dict[str, str]] = None
) -> bytes:
    """Generate a formatted PDF advisory report from AgriTrust results."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    header_style = ParagraphStyle(
        'DocHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#064e3b')
    )
    
    sub_style = ParagraphStyle(
        'DocSubHeader',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4b5563')
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#064e3b'),
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#1f2937')
    )

    claim_ok_style = ParagraphStyle(
        'ClaimOK',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#166534')
    )

    claim_drop_style = ParagraphStyle(
        'ClaimDrop',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#991b1b')
    )

    story = []

    # Title Banner with Logo
    from pathlib import Path
    from reportlab.platypus import Image as RLImage

    logo_path = Path("assets/logo.png")
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if logo_path.exists():
        rl_logo = RLImage(str(logo_path), width=65, height=65)
        title_block = [
            Paragraph("AGRITRUST® ADVISORY REPORT", header_style),
            Paragraph("Evidence- & Confidence-Aware Agricultural Decision Support", sub_style),
            Paragraph(f"Generated on: {gen_time} | Framework: AgriTrust v0.1.0", sub_style)
        ]
        header_table = Table([[rl_logo, title_block]], colWidths=[75, 465])
        header_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(header_table)
    else:
        story.append(Paragraph("🌾 AGRITRUST® ADVISORY REPORT", header_style))
        story.append(Paragraph("Evidence- & Confidence-Aware Agricultural Decision Support", sub_style))
        story.append(Paragraph(f"Generated on: {gen_time} | Framework: AgriTrust v0.1.0", sub_style))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#10b981'), spaceAfter=14))

    # Query & Context Box
    ctx = user_context or {}
    ctx_parts = []
    if ctx.get("crop"):
        ctx_parts.append(f"<b>Crop:</b> {ctx['crop'].title()}")
    if ctx.get("region"):
        ctx_parts.append(f"<b>Region:</b> {ctx['region']}")
    if ctx.get("season"):
        ctx_parts.append(f"<b>Season:</b> {ctx['season'].title()}")
    if ctx.get("growth_stage"):
        ctx_parts.append(f"<b>Growth Stage:</b> {ctx['growth_stage'].title()}")
    
    ctx_str = " | ".join(ctx_parts) if ctx_parts else "General Agronomic Context"

    query_data = [
        [Paragraph("<b>Farmer / User Query:</b>", section_heading)],
        [Paragraph(query, body_style)],
        [Spacer(1, 4)],
        [Paragraph(f"<b>Context Parameters:</b> {ctx_str}", body_style)]
    ]
    t_query = Table(query_data, colWidths=[540])
    t_query.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_query)
    story.append(Spacer(1, 14))

    # Decision & Metrics Table
    decision = pipeline_result.get("decision", "ABSTAIN")
    conf = pipeline_result.get("overall_confidence", 0.0)
    ess = pipeline_result.get("evidence_support_score", 0.0)
    agr = pipeline_result.get("agent_agreement_score", 1.0)
    cov = pipeline_result.get("evidence_coverage", 0.0)

    dec_bg = {
        "ANSWER": colors.HexColor('#ecfdf5'),
        "CLARIFY": colors.HexColor('#fffbeb'),
        "ABSTAIN": colors.HexColor('#fef2f2'),
        "RETRIEVE_MORE": colors.HexColor('#eff6ff')
    }.get(decision, colors.HexColor('#fef2f2'))

    dec_color = {
        "ANSWER": colors.HexColor('#065f46'),
        "CLARIFY": colors.HexColor('#92400e'),
        "ABSTAIN": colors.HexColor('#991b1b'),
        "RETRIEVE_MORE": colors.HexColor('#1e40af')
    }.get(decision, colors.HexColor('#991b1b'))

    metrics_data = [
        [
            Paragraph("<b>Pipeline Decision</b>", section_heading),
            Paragraph("<b>Overall Confidence</b>", section_heading),
            Paragraph("<b>Evidence Support (ESS)</b>", section_heading),
            Paragraph("<b>Agent Agreement (AGR)</b>", section_heading),
            Paragraph("<b>Claim Coverage</b>", section_heading),
        ],
        [
            Paragraph(f"<b><font size=12 color='{dec_color.hexval()}'>{decision}</font></b>", body_style),
            Paragraph(f"<b>{conf*100:.1f}%</b><br/><font size=7 color='#6b7280'>Calibrated ECE</font>", body_style),
            Paragraph(f"<b>{ess*100:.1f}%</b><br/><font size=7 color='#6b7280'>NLI Entailment</font>", body_style),
            Paragraph(f"<b>{agr*100:.1f}%</b><br/><font size=7 color='#6b7280'>Consensus</font>", body_style),
            Paragraph(f"<b>{cov*100:.1f}%</b><br/><font size=7 color='#6b7280'>Verified / Total</font>", body_style),
        ]
    ]
    t_metrics = Table(metrics_data, colWidths=[108, 108, 108, 108, 108])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('BACKGROUND', (0, 1), (0, 1), dec_bg),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 14))

    # Advisory Recommendation
    story.append(Paragraph("<b>Synthesized Agronomic Recommendation:</b>", section_heading))
    raw_answer = pipeline_result.get("final_answer", "")
    # Clean markdown headers for PDF formatting
    clean_lines = []
    for line in raw_answer.split("\n"):
        line = line.replace("### ", "<b>").replace("## ", "<b>").replace("# ", "<b>")
        if "<b>" in line:
            line += "</b>"
        clean_lines.append(line)
    clean_answer = "<br/>".join(clean_lines)

    ans_table = Table([[Paragraph(clean_answer, body_style)]], colWidths=[540])
    ans_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('PADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(ans_table)
    story.append(Spacer(1, 14))

    # Trust Verification Details
    supp = pipeline_result.get("supported_claims", [])
    drop = pipeline_result.get("dropped_claims", [])

    if supp or drop:
        story.append(Paragraph("<b>Claim-Level Trust & Verification Breakdown:</b>", section_heading))
        claim_rows = []
        for s in supp:
            claim_rows.append([Paragraph(f"<b>[SUPPORTED]</b> {s}", claim_ok_style)])
        for d in drop:
            claim_rows.append([Paragraph(f"<b>[PRUNED]</b> {d} <i>(Dropped due to lack of corpus evidence)</i>", claim_drop_style)])

        if claim_rows:
            t_claims = Table(claim_rows, colWidths=[540])
            t_claims.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#f1f5f9')),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(t_claims)
            story.append(Spacer(1, 14))

    # Cited Knowledge Chunks
    chunks = pipeline_result.get("retrieved_chunks", [])
    if chunks:
        story.append(Paragraph("<b>Retrieved Evidence Citations (Hybrid RAG):</b>", section_heading))
        chk_data = [
            [
                Paragraph("<b>Chunk ID</b>", section_heading),
                Paragraph("<b>Document Title / Source</b>", section_heading),
                Paragraph("<b>Topic</b>", section_heading),
                Paragraph("<b>RRF Score</b>", section_heading),
            ]
        ]
        for c in chunks[:5]:
            chk_data.append([
                Paragraph(c.get("chunk_id", ""), body_style),
                Paragraph(f"{c.get('title', '')}<br/><font size=7 color='#6b7280'>{c.get('source_url', '')}</font>", body_style),
                Paragraph(c.get("topic", ""), body_style),
                Paragraph(f"{c.get('score', 0.0):.3f}", body_style),
            ])
        t_chk = Table(chk_data, colWidths=[90, 270, 110, 70])
        t_chk.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(t_chk)

    # Footer note
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#9ca3af'), spaceAfter=8))
    footer_text = "AgriTrust Framework · Grounded Agricultural Advisory Engine · Verified with ICAR & State Agronomy Guidelines"
    story.append(Paragraph(f"<font size=8 color='#6b7280'>{footer_text}</font>", body_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
