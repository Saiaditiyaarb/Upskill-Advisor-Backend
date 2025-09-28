"""
PDF generation service for Upskill Advisor.

Generates a simple, professional PDF plan including:
- Candidate name (if provided in user_context), goal role
- 3-course plan table (course_id, title if available via passed courses, and why)
- Gap map as a horizontal bar chart (counts of sub-skills per target skill)
- Timeline summary (total weeks and phases)

Implementation uses ReportLab (pure Python). If ReportLab isn't installed,
this module degrades gracefully by writing a plain-text .txt file instead,
so the feature is non-blocking in constrained environments.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import datetime

import logging

logger = logging.getLogger("pdf")


def _safe_text(t: Any) -> str:
    try:
        s = str(t) if t is not None else ""
        return s
    except Exception:
        return ""


def _format_timeline(timeline: Dict[str, Any]) -> str:
    if not timeline:
        return ""
    parts: List[str] = []
    total = timeline.get("total_weeks")
    if total:
        parts.append(f"Total: {total} weeks")
    phases = timeline.get("phases") or []
    for ph in phases:
        parts.append(f"- {ph.get('phase', 'Phase')}: weeks {ph.get('weeks', '?')} — {ph.get('focus', '')}")
    return "\n".join(parts)


def generate_plan_pdf(
    candidate_name: Optional[str],
    goal_role: str,
    plan: List[Dict[str, Any]],
    gap_map: Dict[str, List[str]],
    timeline: Optional[Dict[str, Any]],
    courses_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    output_dir: str = "reports"
) -> str:
    """Generate the PDF and return the path. Falls back to .txt if reportlab missing.

    courses_by_id: optional map course_id -> {title, provider, url}
    metrics: optional dict with keys like {"coverage": float 0-1, "diversity": float 0-1}
    """
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception as e:
        logger.warning("ReportLab not available; writing .txt fallback", extra={"error": str(e)})
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plan_{ts}.txt"
        out = Path(output_dir) / filename
        lines: List[str] = []
        lines.append(f"Upskill Plan for {candidate_name or 'Candidate'} → {goal_role}\n")
        if metrics:
            cov = metrics.get('coverage')
            div = metrics.get('diversity')
            lines.append("Metrics:")
            if cov is not None:
                try:
                    lines.append(f" - Skill Coverage: {round(float(cov) * 100, 1)}%")
                except Exception:
                    lines.append(f" - Skill Coverage: {cov}")
            if div is not None:
                try:
                    lines.append(f" - Path Diversity: {round(float(div) * 100, 1)}%")
                except Exception:
                    lines.append(f" - Path Diversity: {div}")
            lines.append("")
        lines.append("Plan:")
        for step in plan:
            cid = step.get("course_id")
            meta = (courses_by_id or {}).get(cid, {}) if courses_by_id else {}
            title = meta.get("title") if meta else None
            provider = meta.get("provider") if meta else None
            url = meta.get("url") if meta else None
            why = step.get("why", "")
            extra = []
            if provider:
                extra.append(f"provider={provider}")
            if url:
                extra.append(f"url={url}")
            suffix = f" [{' | '.join(extra)}]" if extra else ""
            lines.append(f" - {cid or ''} {f'({title})' if title else ''}{suffix}: {why}")
        lines.append("\nGap Map:")
        for skill, subs in gap_map.items():
            lines.append(f" - {skill}: {len(subs)} sub-skills")
        if timeline:
            lines.append("\nTimeline:")
            lines.append(_format_timeline(timeline))
        out.write_text("\n".join(lines), encoding="utf-8")
        return str(out)

    # ReportLab path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plan_{ts}.pdf"
    out = Path(output_dir) / filename

    doc = SimpleDocTemplate(str(out), pagesize=LETTER, title="Upskill Advisor Plan")
    styles = getSampleStyleSheet()
    story: List[Any] = []

    title_text = f"Upskill Plan for { _safe_text(candidate_name) or 'Candidate' } → {_safe_text(goal_role)}"
    story.append(Paragraph(title_text, styles['Title']))
    story.append(Spacer(1, 12))

    # Metrics section (optional)
    if metrics:
        story.append(Paragraph("Plan Metrics", styles['Heading2']))
        m_rows = [["Metric", "Value"]]
        cov = metrics.get('coverage') if isinstance(metrics, dict) else None
        div = metrics.get('diversity') if isinstance(metrics, dict) else None
        if cov is not None:
            try:
                cov_disp = f"{round(float(cov) * 100, 1)}%"
            except Exception:
                cov_disp = str(cov)
            m_rows.append(["Skill Coverage", cov_disp])
        if div is not None:
            try:
                div_disp = f"{round(float(div) * 100, 1)}%"
            except Exception:
                div_disp = str(div)
            m_rows.append(["Path Diversity", div_disp])
        if len(m_rows) > 1:
            from reportlab.platypus import Table, TableStyle
            m_table = Table(m_rows, colWidths=[150, 100])
            m_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f0f0f0')),
                ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(m_table)
            story.append(Spacer(1, 12))

    # Define paragraph styles for table content
    from reportlab.lib.styles import ParagraphStyle
    body_style = styles.get('BodyText')
    small_style = ParagraphStyle('Small', parent=body_style, fontSize=9, leading=11)

    # Plan table
    story.append(Paragraph("Recommended 3-Course Path", styles['Heading2']))
    data = [["Order", "Course", "Why"]]
    for i, step in enumerate(plan, start=1):
        cid = _safe_text(step.get("course_id"))
        meta = (courses_by_id or {}).get(cid, {}) if courses_by_id else {}
        title = _safe_text(meta.get("title") or cid)
        provider = _safe_text(meta.get("provider"))
        url = _safe_text(meta.get("url"))
        why = _safe_text(step.get("why", ""))
        # Build a rich Course cell with provider and clickable link if available
        course_parts: List[str] = [f"<b>{title}</b>"]
        if provider:
            course_parts.append(f"<br/><font size=9 color='#555555'>Provider: {provider}</font>")
        if url:
            course_parts.append(
                f"<br/><font size=9><u><font color='blue'><link href='{url}'>Open course</link></font></u></font>"
            )
        course_cell = Paragraph("".join(course_parts), body_style)
        why_cell = Paragraph(why, body_style)
        data.append([str(i), course_cell, why_cell])
    table = Table(data, colWidths=[40, 200, 300])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f0f0f0')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # Gap map summary
    story.append(Paragraph("Skill Gap Map", styles['Heading2']))
    if gap_map:
        gap_rows = [["Skill", "Sub-skill count"]]
        # Sort by descending count for readability
        for skill, subs in sorted(gap_map.items(), key=lambda kv: len(kv[1]), reverse=True):
            gap_rows.append([_safe_text(skill), str(len(subs))])
        gtable = Table(gap_rows, colWidths=[240, 80])
        gtable.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f0f0f0')),
        ]))
        story.append(gtable)
    else:
        story.append(Paragraph("No explicit gaps identified.", styles['Normal']))
    story.append(Spacer(1, 12))

    # Timeline
    if timeline:
        story.append(Paragraph("Timeline", styles['Heading2']))
        story.append(Paragraph(_format_timeline(timeline).replace('\n', '<br/>'), styles['Normal']))
        story.append(Spacer(1, 12))

    # Footer
    story.append(Spacer(1, 24))
    story.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Italic']))

    try:
        doc.build(story)
        logger.info("pdf_generated", extra={"path": str(out)})
        return str(out)
    except Exception as e:
        logger.error("pdf_generation_failed", extra={"error": str(e)})
        # Fallback to txt
        txt_path = Path(output_dir) / (filename.replace('.pdf', '.txt'))
        txt_path.write_text("PDF generation failed. See logs.", encoding="utf-8")
        return str(txt_path)
