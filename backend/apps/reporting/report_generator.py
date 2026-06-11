"""
Phase 7 — PDF Report Generator (ReportLab)

Produces a multi-section A4 PDF for a completed interview session.

Sections:
    1. Executive Summary       — candidate name, date, duration, dominant emotion, score
    2. Emotion Timeline        — line chart of per-emotion confidence over session time
    3. Text-Based Insights     — uncertainty phrases, sentiment arc, Hebrew transcript
    4. Model Performance       — per-modality score distributions, fusion breakdown
    5. Recommendations         — rule-based tips triggered by emotion thresholds

Interface used by report_tasks.py:
    generator = InterviewReportGenerator()
    pdf_bytes: bytes = generator.generate(session, emotion_frames, transcript_segments)
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, String
from reportlab.graphics import renderPDF

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_BRAND_DARK   = colors.HexColor("#1a1a2e")
_BRAND_ACCENT = colors.HexColor("#e94560")
_BRAND_MID    = colors.HexColor("#16213e")
_BRAND_LIGHT  = colors.HexColor("#0f3460")
_GREY_LIGHT   = colors.HexColor("#f5f5f5")
_GREY_MID     = colors.HexColor("#cccccc")
_TEXT_DARK    = colors.HexColor("#222222")
_WHITE        = colors.white

_EMOTION_COLORS: dict[str, colors.HexColor] = {
    "neutral":   colors.HexColor("#95a5a6"),
    "confident": colors.HexColor("#27ae60"),
    "anxious":   colors.HexColor("#e67e22"),
    "happy":     colors.HexColor("#f1c40f"),
    "sad":       colors.HexColor("#2980b9"),
    "angry":     colors.HexColor("#c0392b"),
    "surprised": colors.HexColor("#8e44ad"),
    "uncertain": colors.HexColor("#7f8c8d"),
}

EMOTION_CLASSES = [
    "neutral", "confident", "anxious", "happy",
    "sad", "angry", "surprised", "uncertain",
]

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class InterviewReportGenerator:
    """Generates a ReportLab PDF for a completed DeepCue interview session."""

    def generate(
        self,
        session: dict[str, Any],
        emotion_frames: list[dict[str, Any]],
        transcript_segments: list[dict[str, Any]],
    ) -> bytes:
        """
        Build and return PDF bytes.

        Parameters
        ----------
        session             : InterviewSession document from MongoDB
        emotion_frames      : list of EmotionFrame documents, sorted by frame_index
        transcript_segments : list of TranscriptSegment documents, sorted by segment_index
        """
        buf = io.BytesIO()
        doc = _build_doc(buf)
        styles = _build_styles()

        story: list[Any] = []
        story += _section_header(styles)
        story += _section1_executive_summary(session, emotion_frames, styles)
        story += [PageBreak()]
        story += _section2_emotion_timeline(emotion_frames, styles)
        story += [PageBreak()]
        story += _section3_text_insights(transcript_segments, emotion_frames, styles)
        story += [PageBreak()]
        story += _section4_model_performance(emotion_frames, styles)
        story += [PageBreak()]
        story += _section5_recommendations(session, emotion_frames, styles)

        doc.build(story, onFirstPage=_draw_page_border, onLaterPages=_draw_page_border)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Document template
# ---------------------------------------------------------------------------

def _build_doc(buf: io.BytesIO) -> BaseDocTemplate:
    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN + 0.5 * cm,
        title="DeepCue Interview Report",
        author="DeepCue System",
    )
    frame = Frame(
        MARGIN, MARGIN + 0.5 * cm,
        PAGE_W - 2 * MARGIN,
        PAGE_H - 2 * MARGIN - 0.5 * cm,
        id="main",
    )
    template = PageTemplate(id="main", frames=[frame])
    doc.addPageTemplates([template])
    return doc


def _draw_page_border(canvas: Any, doc: Any) -> None:
    """Draw header bar and page footer on every page."""
    canvas.saveState()

    # Header bar.
    canvas.setFillColor(_BRAND_DARK)
    canvas.rect(0, PAGE_H - 1.2 * cm, PAGE_W, 1.2 * cm, fill=1, stroke=0)

    canvas.setFillColor(_BRAND_ACCENT)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(MARGIN, PAGE_H - 0.85 * cm, "DeepCue")
    canvas.setFillColor(_WHITE)
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.85 * cm, "Interview Emotion Analysis Report")

    # Footer.
    canvas.setFillColor(_GREY_MID)
    canvas.setFont("Helvetica", 7)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    canvas.drawString(MARGIN, 0.5 * cm, f"Generated: {ts}")
    canvas.drawRightString(PAGE_W - MARGIN, 0.5 * cm, f"Page {doc.page}")

    canvas.restoreState()


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"],
            fontSize=18, textColor=_BRAND_DARK, spaceAfter=6,
            fontName="Helvetica-Bold",
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"],
            fontSize=13, textColor=_BRAND_LIGHT, spaceAfter=4,
            fontName="Helvetica-Bold",
        ),
        "h3": ParagraphStyle(
            "h3", parent=base["Heading3"],
            fontSize=10, textColor=_BRAND_MID, spaceAfter=3,
            fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontSize=9, textColor=_TEXT_DARK, leading=14,
        ),
        "body_rtl": ParagraphStyle(
            "body_rtl", parent=base["Normal"],
            fontSize=9, textColor=_TEXT_DARK, leading=14,
            wordWrap="RTL",
        ),
        "label": ParagraphStyle(
            "label", parent=base["Normal"],
            fontSize=8, textColor=colors.grey,
        ),
        "callout": ParagraphStyle(
            "callout", parent=base["Normal"],
            fontSize=10, textColor=_BRAND_DARK,
            backColor=_GREY_LIGHT, borderPad=6,
            fontName="Helvetica-Bold",
        ),
        "tip": ParagraphStyle(
            "tip", parent=base["Normal"],
            fontSize=9, textColor=_TEXT_DARK, leading=13,
            leftIndent=10, borderPad=4,
        ),
    }


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def _section_header(styles: dict) -> list:
    return [
        Spacer(1, 1.2 * cm),
        Paragraph("Interview Emotion Analysis Report", styles["h1"]),
        HRFlowable(width="100%", thickness=2, color=_BRAND_ACCENT, spaceAfter=4),
    ]


# ---------------------------------------------------------------------------
# Section 1 — Executive Summary (7.2)
# ---------------------------------------------------------------------------

def _section1_executive_summary(
    session: dict[str, Any],
    emotion_frames: list[dict[str, Any]],
    styles: dict,
) -> list:
    flowables: list[Any] = [
        Paragraph("1. Executive Summary", styles["h2"]),
        Spacer(1, 3 * mm),
    ]

    candidate     = session.get("candidate_name", "Unknown")
    created_at    = session.get("created_at", datetime.now(timezone.utc))
    duration_s    = float(session.get("duration_seconds", 0.0))
    dominant      = session.get("dominant_emotion", "neutral")
    frame_count   = int(session.get("frame_count", len(emotion_frames)))

    # Overall confidence score = mean of confident + happy emotion probabilities.
    overall_score = _compute_overall_score(emotion_frames)

    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at)
        except ValueError:
            created_at = datetime.now(timezone.utc)

    date_str = created_at.strftime("%B %d, %Y  %H:%M")
    dur_str  = _format_duration(duration_s)

    summary_data = [
        ["Candidate",       candidate],
        ["Date",            date_str],
        ["Duration",        dur_str],
        ["Frames analysed", str(frame_count)],
        ["Dominant emotion", dominant.capitalize()],
        ["Confidence score", f"{overall_score:.1%}"],
    ]

    tbl = Table(summary_data, colWidths=[5 * cm, 10 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), _GREY_LIGHT),
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",     (0, 0), (0, -1), _BRAND_DARK),
        ("TEXTCOLOR",     (1, 0), (1, -1), _TEXT_DARK),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_WHITE, _GREY_LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.5, _GREY_MID),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    flowables.append(tbl)
    flowables.append(Spacer(1, 4 * mm))

    # Emotion distribution bar chart (static table-based).
    if emotion_frames:
        flowables.append(Paragraph("Emotion Distribution", styles["h3"]))
        flowables.append(_emotion_distribution_table(emotion_frames, styles))

    return flowables


def _emotion_distribution_table(
    emotion_frames: list[dict],
    styles: dict,
) -> Table:
    """Horizontal bar chart rendered as a ReportLab table."""
    avg_scores = _average_fusion_scores(emotion_frames)
    BAR_MAX = 12 * cm

    rows = []
    for emotion in EMOTION_CLASSES:
        score = avg_scores.get(emotion, 0.0)
        bar_w = max(2, BAR_MAX * score)
        bar_drawing = Drawing(BAR_MAX, 12)
        rect = Rect(0, 1, bar_w, 10,
                    fillColor=_EMOTION_COLORS.get(emotion, _GREY_MID),
                    strokeWidth=0)
        bar_drawing.add(rect)
        rows.append([
            Paragraph(emotion.capitalize(), ParagraphStyle(
                "em", fontSize=8, textColor=_TEXT_DARK
            )),
            bar_drawing,
            Paragraph(f"{score:.1%}", ParagraphStyle(
                "pct", fontSize=8, textColor=_BRAND_DARK, fontName="Helvetica-Bold"
            )),
        ])

    tbl = Table(rows, colWidths=[3 * cm, BAR_MAX, 2 * cm])
    tbl.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return tbl


# ---------------------------------------------------------------------------
# Section 2 — Emotion Timeline (7.3)
# ---------------------------------------------------------------------------

def _section2_emotion_timeline(
    emotion_frames: list[dict[str, Any]],
    styles: dict,
) -> list:
    flowables: list[Any] = [
        Paragraph("2. Emotion Timeline", styles["h2"]),
        Spacer(1, 3 * mm),
    ]

    if not emotion_frames:
        flowables.append(Paragraph("No emotion frames recorded.", styles["body"]))
        return flowables

    flowables.append(Paragraph(
        "Emotion confidence over session time (sampled at 1 data point per 5 frames).",
        styles["body"],
    ))
    flowables.append(Spacer(1, 3 * mm))

    # Sub-sample for chart density.
    sampled = emotion_frames[::5] or emotion_frames

    CHART_W = PAGE_W - 2 * MARGIN - 1 * cm
    CHART_H = 7 * cm
    PADDING_L = 0.8 * cm
    PADDING_B = 0.8 * cm
    PLOT_W = CHART_W - PADDING_L - 0.5 * cm
    PLOT_H = CHART_H - PADDING_B - 0.5 * cm

    drawing = Drawing(CHART_W, CHART_H)

    # Background.
    drawing.add(Rect(PADDING_L, PADDING_B, PLOT_W, PLOT_H,
                     fillColor=colors.HexColor("#f9f9f9"), strokeColor=_GREY_MID,
                     strokeWidth=0.5))

    # Horizontal grid lines at 0.25, 0.5, 0.75.
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = PADDING_B + frac * PLOT_H
        drawing.add(Line(PADDING_L, y, PADDING_L + PLOT_W, y,
                         strokeColor=_GREY_MID, strokeWidth=0.3))
        drawing.add(String(PADDING_L - 2, y - 3, f"{frac:.0%}",
                           fontSize=5, fillColor=colors.grey))

    # Y-axis label.
    drawing.add(String(2, PADDING_B + PLOT_H / 2, "Confidence",
                       fontSize=5, fillColor=colors.grey))

    # Timestamps for X axis.
    t0 = float(sampled[0].get("timestamp", 0.0))
    t_max = max(float(f.get("timestamp", 0.0)) - t0 for f in sampled) or 1.0
    n = len(sampled)

    # Plot a line per emotion.
    for emotion in EMOTION_CLASSES:
        pts: list[float] = []
        for i, frame in enumerate(sampled):
            t = float(frame.get("timestamp", 0.0)) - t0
            x = PADDING_L + (t / t_max) * PLOT_W
            score = frame.get("fusion_scores", {}).get(emotion, 0.0)
            y = PADDING_B + score * PLOT_H
            pts.extend([x, y])

        if len(pts) >= 4:
            drawing.add(PolyLine(
                pts,
                strokeColor=_EMOTION_COLORS.get(emotion, _GREY_MID),
                strokeWidth=0.8,
                strokeLineCap=1,
            ))

    # X-axis time labels (first, middle, last).
    for idx in [0, len(sampled) // 2, len(sampled) - 1]:
        t = float(sampled[idx].get("timestamp", 0.0)) - t0
        x = PADDING_L + (t / t_max) * PLOT_W
        drawing.add(Line(x, PADDING_B - 2, x, PADDING_B,
                         strokeColor=_GREY_MID, strokeWidth=0.5))
        drawing.add(String(x - 5, PADDING_B - 9, f"{t:.0f}s",
                           fontSize=5, fillColor=colors.grey))

    flowables.append(drawing)
    flowables.append(Spacer(1, 3 * mm))

    # Legend.
    legend_items = []
    for i, emotion in enumerate(EMOTION_CLASSES):
        color_hex = _EMOTION_COLORS.get(emotion, _GREY_MID).hexval()
        legend_items.append(
            Paragraph(
                f'<font color="{color_hex}">■</font> {emotion.capitalize()}',
                ParagraphStyle("leg", fontSize=7, textColor=_TEXT_DARK),
            )
        )

    # 4-column legend table.
    rows = [legend_items[i:i+4] for i in range(0, len(legend_items), 4)]
    legend_tbl = Table(rows, colWidths=[(CHART_W / 4)] * 4)
    legend_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    flowables.append(legend_tbl)

    return flowables


# ---------------------------------------------------------------------------
# Section 3 — Text-Based Insights (7.4)
# ---------------------------------------------------------------------------

def _section3_text_insights(
    transcript_segments: list[dict[str, Any]],
    emotion_frames: list[dict[str, Any]],
    styles: dict,
) -> list:
    flowables: list[Any] = [
        Paragraph("3. Text-Based Insights", styles["h2"]),
        Spacer(1, 3 * mm),
    ]

    if not transcript_segments:
        flowables.append(Paragraph("No transcript segments recorded.", styles["body"]))
        return flowables

    # Sentiment arc: divide session into 3 thirds, compute mean text score per third.
    text_scores = [f.get("text_score", 0.5) for f in emotion_frames]
    if text_scores:
        third = max(1, len(text_scores) // 3)
        arc = [
            sum(text_scores[:third]) / third,
            sum(text_scores[third:2*third]) / max(1, third),
            sum(text_scores[2*third:]) / max(1, len(text_scores) - 2*third),
        ]
        arc_row = [
            [Paragraph("Beginning", styles["label"]),
             Paragraph("Middle", styles["label"]),
             Paragraph("End", styles["label"])],
            [Paragraph(f"{arc[0]:.1%}", styles["callout"]),
             Paragraph(f"{arc[1]:.1%}", styles["callout"]),
             Paragraph(f"{arc[2]:.1%}", styles["callout"])],
        ]
        flowables.append(Paragraph("Sentiment Arc", styles["h3"]))
        arc_tbl = Table(arc_row, colWidths=[5 * cm] * 3)
        arc_tbl.setStyle(TableStyle([
            ("ALIGN",   (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",  (0, 0), (-1, -1), "MIDDLE"),
            ("GRID",    (0, 0), (-1, -1), 0.5, _GREY_MID),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        flowables.append(arc_tbl)
        flowables.append(Spacer(1, 4 * mm))

    # Uncertainty phrases: transcript segments where text_score is low (< 0.4).
    uncertain_segs = _find_uncertain_segments(transcript_segments, emotion_frames)
    flowables.append(Paragraph("Uncertainty Phrases", styles["h3"]))
    if uncertain_segs:
        flowables.append(Paragraph(
            "The following phrases were detected with low confidence scores:", styles["body"]
        ))
        flowables.append(Spacer(1, 2 * mm))
        for seg in uncertain_segs[:8]:
            text = seg.get("text", "").strip()
            ts   = float(seg.get("timestamp", 0.0))
            flowables.append(Paragraph(
                f'<i>"{text}"</i>  <font color="grey" size="7">({ts:.0f}s)</font>',
                styles["tip"],
            ))
    else:
        flowables.append(Paragraph("No high-uncertainty phrases detected.", styles["body"]))

    flowables.append(Spacer(1, 4 * mm))

    # Hebrew transcript (RTL, last 8 segments).
    flowables.append(Paragraph("Transcript Excerpts (Hebrew)", styles["h3"]))
    flowables.append(Paragraph(
        "Recent transcript segments (right-to-left):", styles["body"]
    ))
    flowables.append(Spacer(1, 2 * mm))

    recent = transcript_segments[-8:]
    for seg in recent:
        text = seg.get("text", "").strip()
        ts   = float(seg.get("timestamp", 0.0))
        if not text:
            continue
        flowables.append(Paragraph(
            f'<para dir="rtl">{text}</para>  <font color="grey" size="7">({ts:.0f}s)</font>',
            styles["body_rtl"],
        ))

    return flowables


# ---------------------------------------------------------------------------
# Section 4 — Model Performance (7.5)
# ---------------------------------------------------------------------------

def _section4_model_performance(
    emotion_frames: list[dict[str, Any]],
    styles: dict,
) -> list:
    flowables: list[Any] = [
        Paragraph("4. Model Performance Metrics", styles["h2"]),
        Spacer(1, 3 * mm),
    ]

    if not emotion_frames:
        flowables.append(Paragraph("No emotion frames recorded.", styles["body"]))
        return flowables

    video_scores = [f.get("video_score", 0.5) for f in emotion_frames]
    audio_scores = [f.get("audio_score", 0.5) for f in emotion_frames]
    text_scores  = [f.get("text_score",  0.5) for f in emotion_frames]

    def _stats(vals: list[float]) -> tuple[float, float, float]:
        arr = sorted(vals)
        n = len(arr)
        return (
            sum(arr) / n,
            arr[n // 2],
            arr[int(n * 0.05)],
        )

    vm, vmed, v5 = _stats(video_scores)
    am, amed, a5 = _stats(audio_scores)
    tm, tmed, t5 = _stats(text_scores)

    header = ["Modality", "Mean", "Median", "5th Percentile"]
    rows   = [
        ["Video (facial)", f"{vm:.3f}", f"{vmed:.3f}", f"{v5:.3f}"],
        ["Audio (speech)", f"{am:.3f}", f"{amed:.3f}", f"{a5:.3f}"],
        ["Text (language)", f"{tm:.3f}", f"{tmed:.3f}", f"{t5:.3f}"],
    ]
    all_rows = [header] + rows

    tbl = Table(all_rows, colWidths=[5 * cm, 3 * cm, 3 * cm, 4 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), _BRAND_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), _WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_WHITE, _GREY_LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.5, _GREY_MID),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    flowables.append(tbl)
    flowables.append(Spacer(1, 4 * mm))

    # Fusion output breakdown.
    flowables.append(Paragraph("Fusion Model Output (Session Average)", styles["h3"]))
    avg_scores = _average_fusion_scores(emotion_frames)
    fusion_rows = [["Emotion", "Average Confidence"]]
    for emotion in EMOTION_CLASSES:
        score = avg_scores.get(emotion, 0.0)
        fusion_rows.append([emotion.capitalize(), f"{score:.2%}"])

    fusion_tbl = Table(fusion_rows, colWidths=[6 * cm, 5 * cm])
    fusion_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), _BRAND_LIGHT),
        ("TEXTCOLOR",     (0, 0), (-1, 0), _WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_WHITE, _GREY_LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.5, _GREY_MID),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    flowables.append(fusion_tbl)

    return flowables


# ---------------------------------------------------------------------------
# Section 5 — Recommendations (7.6)
# ---------------------------------------------------------------------------

_RECOMMENDATION_RULES: list[tuple[str, float, str, str]] = [
    # (emotion, threshold, title, text)
    ("anxious", 0.30,
     "Managing Interview Anxiety",
     "A notable level of anxiety was detected throughout the session. "
     "Practice deep breathing before your next interview: inhale for 4 seconds, "
     "hold 4 seconds, exhale 6 seconds. Prepare structured answers using the STAR method "
     "to reduce cognitive load."),
    ("uncertain", 0.25,
     "Reducing Verbal Uncertainty",
     "Uncertainty markers appeared frequently in your speech. Replace hedging phrases "
     "(\"I think maybe...\", \"I'm not sure but...\") with assertive language. "
     "Pause briefly before answering — a confident pause reads better than a hesitant filler."),
    ("sad", 0.25,
     "Projecting Positive Energy",
     "Subdued emotional expression was detected. Make a conscious effort to vary your "
     "vocal pitch and maintain appropriate eye contact with the camera. "
     "A brief self-affirmation routine before the interview can help shift your baseline mood."),
    ("angry", 0.20,
     "Keeping Composure Under Pressure",
     "Elevated stress indicators were detected in some segments. "
     "If a question catches you off guard, take a short pause rather than reacting immediately. "
     "Focus on the problem-solving aspect of difficult questions rather than the pressure."),
    ("confident", 0.60,
     "Leveraging Your Confidence",
     "Strong confidence signals were detected. Continue with your current approach: "
     "clear delivery, decisive language, and composed presence. "
     "Use this momentum to drive more specific and structured answers."),
    ("happy", 0.55,
     "Sustaining Positive Engagement",
     "High positive affect was detected throughout the interview. "
     "Your enthusiasm comes through well — make sure it's paired with depth of content "
     "so interviewers see both personality and competence."),
]


def _section5_recommendations(
    session: dict[str, Any],
    emotion_frames: list[dict[str, Any]],
    styles: dict,
) -> list:
    flowables: list[Any] = [
        Paragraph("5. Recommendations", styles["h2"]),
        Spacer(1, 3 * mm),
        Paragraph(
            "The following recommendations are based on emotion patterns detected during "
            "your interview session.",
            styles["body"],
        ),
        Spacer(1, 4 * mm),
    ]

    avg = _average_fusion_scores(emotion_frames)
    fired = False

    for emotion, threshold, title, text in _RECOMMENDATION_RULES:
        score = avg.get(emotion, 0.0)
        if score >= threshold:
            fired = True
            # Title badge.
            badge_color = _EMOTION_COLORS.get(emotion, _GREY_MID).hexval()
            flowables.append(Paragraph(
                f'<font color="{badge_color}">●</font>  <b>{title}</b>'
                f'  <font color="grey" size="7">({emotion}: {score:.1%})</font>',
                styles["h3"],
            ))
            flowables.append(Paragraph(text, styles["tip"]))
            flowables.append(Spacer(1, 3 * mm))

    if not fired:
        flowables.append(Paragraph(
            "No specific recommendations triggered. Overall emotional profile was balanced.",
            styles["body"],
        ))

    return flowables


# ---------------------------------------------------------------------------
# Analytics helpers
# ---------------------------------------------------------------------------

def _average_fusion_scores(emotion_frames: list[dict]) -> dict[str, float]:
    if not emotion_frames:
        return {e: 0.0 for e in EMOTION_CLASSES}
    acc: dict[str, float] = {e: 0.0 for e in EMOTION_CLASSES}
    count = 0
    for f in emotion_frames:
        fs = f.get("fusion_scores", {})
        if fs:
            for e in EMOTION_CLASSES:
                acc[e] += float(fs.get(e, 0.0))
            count += 1
    if count == 0:
        return acc
    return {e: v / count for e, v in acc.items()}


def _compute_overall_score(emotion_frames: list[dict]) -> float:
    avg = _average_fusion_scores(emotion_frames)
    return float(avg.get("confident", 0.0) + avg.get("happy", 0.0)) / 2.0


def _find_uncertain_segments(
    transcript_segments: list[dict],
    emotion_frames: list[dict],
) -> list[dict]:
    """Return transcript segments whose nearest emotion frame has text_score < 0.4."""
    if not emotion_frames:
        return []
    frame_times = [(f.get("timestamp", 0.0), f.get("text_score", 0.5)) for f in emotion_frames]
    result = []
    for seg in transcript_segments:
        t = float(seg.get("timestamp", 0.0))
        nearest_score = min(frame_times, key=lambda ft: abs(ft[0] - t))[1]
        if nearest_score < 0.4 and seg.get("text", "").strip():
            result.append(seg)
    return result


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"
