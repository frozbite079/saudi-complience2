"""
SBC Inspection Report Generator
================================
Renders the HTML/CSS template with violation detection results.
Uses Python's built-in string formatting — **zero external dependencies**.
"""

from __future__ import annotations

import base64
import datetime
import html as html_module
import io
import logging
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent
CSS_FILE = TEMPLATE_DIR / "report.css"


def _esc(value: Any) -> str:
    """HTML-escape a value."""
    return html_module.escape(str(value)) if value else ""


def _load_css() -> str:
    """Read the report stylesheet."""
    if CSS_FILE.exists():
        return CSS_FILE.read_text(encoding="utf-8")
    logger.warning("CSS not found at %s", CSS_FILE)
    return ""


def _severity_class(priority: str) -> str:
    """Map priority string to CSS class."""
    p = (priority or "").upper()
    if p in ("CRITICAL",):
        return "critical"
    elif p in ("HIGH",):
        return "high"
    elif p in ("MEDIUM",):
        return "medium"
    elif p in ("LOW",):
        return "low"
    return "medium"


def _crop_image_b64(annotated_b64: str, bbox: list) -> str | None:
    """
    Crop the annotated image to the bbox region and return a base64 data-URL.
    bbox is normalized [x1, y1, x2, y2] in range 0–1000.
    Returns None on any error.
    """
    try:
        from PIL import Image

        # Strip data-URL prefix if present
        if "," in annotated_b64:
            annotated_b64 = annotated_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(annotated_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size

        x1 = int(bbox[0] / 1000 * w)
        y1 = int(bbox[1] / 1000 * h)
        x2 = int(bbox[2] / 1000 * w)
        y2 = int(bbox[3] / 1000 * h)

        # Clamp and ensure valid crop
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None

        cropped = img.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=85)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as exc:
        logger.warning("Failed to crop evidence image: %s", exc)
        return None


def _render_violation_card(v: dict, index: int, annotated_image: str | None = None) -> str:
    """Render a single violation card HTML block."""
    severity = _severity_class(v.get("priority", "MEDIUM"))
    risk_level = _esc(v.get("priority", "MEDIUM"))
    vio_id = f"VIO-{index:03d}"
    sbc_ref = _esc(v.get("sbc_reference", "N/A"))
    title = _esc(v.get("cv_target") or (v.get("source_text", "")[:60]))
    description = _esc(v.get("source_text", ""))
    rule_text = _esc(v.get("rule_text", "See SBCNC documentation"))
    remediation = _esc(v.get("remediation", ""))
    bbox = v.get("bbox")
    confidence_raw = v.get("confidence", None)
    confidence_pct = int(round(float(confidence_raw) * 100)) if confidence_raw is not None else None

    # Evidence image — prioritize pre-annotated video frame, then fallback to cropped image
    evidence_html = '<div class="evidence-placeholder">No annotated image available</div>'
    frame_b64 = v.get("frame_b64")
    if frame_b64:
        evidence_html = f'<img src="{frame_b64}" alt="Violation frame" style="width:100%;border-radius:6px;">'
    elif bbox and annotated_image:
        cropped_src = _crop_image_b64(annotated_image, bbox)
        if cropped_src:
            evidence_html = f'<img src="{cropped_src}" alt="Violation region" style="width:100%;border-radius:6px;">'
        else:
            evidence_html = '<div class="evidence-placeholder">Bounding box detected at region</div>'
    elif bbox:
        evidence_html = '<div class="evidence-placeholder">Bounding box detected at region</div>'

    remediation_row = ""
    if remediation:
        remediation_row = f"""
            <div class="detail-row">
              <span class="detail-label">Remediation</span>
              <span class="detail-value">{remediation}</span>
            </div>"""

    timestamp_row = ""
    timestamp_sec = v.get("timestamp_sec")
    if timestamp_sec is not None:
        mins, secs = divmod(int(timestamp_sec), 60)
        timestamp_row = f"""
            <div class="detail-row">
              <span class="detail-label">Timestamp</span>
              <span class="detail-value" style="font-family: monospace; background: var(--surface-container); padding: 2px 6px; border-radius: 4px; border: 1px solid var(--surface-border); color: var(--primary);">⏱️ {mins:02d}:{secs:02d}</span>
            </div>"""

    confidence_row = ""
    if confidence_pct is not None:
        # Colour the bar: >=80 green, >=60 orange, else red
        bar_color = "#2e7d32" if confidence_pct >= 80 else "#e65100" if confidence_pct >= 60 else "#c62828"
        confidence_row = f"""
            <div class="detail-row">
              <span class="detail-label">AI Confidence</span>
              <span class="detail-value" style="display:flex;align-items:center;gap:8px;">
                <span style="flex:1;background:#e0e0e0;border-radius:4px;height:8px;overflow:hidden;min-width:80px;">
                  <span style="display:block;height:100%;width:{confidence_pct}%;background:{bar_color};border-radius:4px;transition:width 0.3s;"></span>
                </span>
                <strong style="color:{bar_color};font-size:0.85em;white-space:nowrap;">{confidence_pct}%</strong>
              </span>
            </div>"""

    return f"""
    <div class="violation-card severity-{severity}">
      <div class="card-header">
        <span class="badge badge-{severity}">{risk_level}</span>
        <span class="vio-id">{vio_id}</span>
        <span class="sbc-ref">SBC Ref: {sbc_ref}</span>
      </div>
      <div class="card-body">
        <h3>{title}</h3>
        <div class="card-content">
          <div>{evidence_html}</div>
          <div>{timestamp_row}
            <div class="detail-row">
              <span class="detail-label">Description</span>
              <span class="detail-value">{description}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">SBC Code</span>
              <span class="detail-value">{sbc_ref} — {rule_text}</span>
            </div>{remediation_row}
            <div class="detail-row">
              <span class="detail-label">Risk Level</span>
              <span class="detail-value"><span class="badge badge-{severity}">{risk_level}</span></span>
            </div>{confidence_row}
          </div>
        </div>
      </div>
    </div>"""


def _render_ref_table_row(v: dict, index: int) -> str:
    """Render a single row of the SBC reference table."""
    sbc_ref = _esc(v.get("sbc_reference", "N/A"))
    rule_text = _esc(v.get("rule_text") or v.get("source_text", "")[:80])
    return f"""
        <tr>
          <td>{index}</td>
          <td><strong>{sbc_ref}</strong></td>
          <td>{rule_text}</td>
          <td><span class="badge badge-violation">Violation</span></td>
        </tr>"""


def generate_report_html(
    detection_result: dict[str, Any],
    *,
    project_name: str = "",
    contractor_name: str = "",
    location: str = "",
    inspector_name: str = "",
    building_type: str = "",
    permit_no: str = "",
    classification: str | None = None,
) -> str:
    """
    Render a fully self-contained HTML inspection report.

    Parameters
    ----------
    detection_result : dict
        The JSON dict returned by ``violation_detector.detect_violations()``.

    Returns
    -------
    str
        Fully rendered, self-contained HTML string.
    """
    now = datetime.datetime.now()

    # ── Extract data ────────────────────────────────────────────
    summary = detection_result.get("summary", {})
    # Use _report_verdicts when available (includes frame_b64 + confidence for HTML cards)
    verdicts = detection_result.get("_report_verdicts") or detection_result.get("verdicts", [])
    annotated_image = detection_result.get("annotated_image")
    heatmap_image = detection_result.get("heatmap_image")
    ai_recommendations = detection_result.get("ai_recommendations", [])
    token_usage = detection_result.get("token_usage", {})

    if not project_name:
        project_name = detection_result.get("project_name") or "Building Inspection"
    if not contractor_name:
        contractor_name = detection_result.get("contractor_name") or ""

    if not classification:
        classification = detection_result.get("classification", "General")

    total_items = summary.get("total_rules_evaluated", len(verdicts))
    violation_count = summary.get("violations", len(verdicts))
    compliant_count = summary.get("compliant", 0)
    uncertain_count = summary.get("uncertain", 0)
    compliance_rate = summary.get("compliance_rate", 0.0)
    compliance_int = int(compliance_rate)

    report_id = f"RPT-{now.strftime('%Y-%m')}-{uuid.uuid4().hex[:4].upper()}"
    report_date = now.strftime("%B %d, %Y")
    generated_at = now.strftime("%B %d, %Y %H:%M")

    observe_tokens = token_usage.get("observe", {})
    judge_tokens = token_usage.get("judge", {})
    token_observe_str = f"{observe_tokens.get('total_tokens', 'N/A')} tokens"
    token_judge_str = f"{judge_tokens.get('total_tokens', 'N/A')} tokens"

    # ── Compliance status label ─────────────────────────────────
    if compliance_int >= 90:
        status_html = '<div class="status-label compliant">✓ COMPLIANT</div>'
    elif compliance_int >= 60:
        status_html = '<div class="status-label remediation">⚠ REQUIRES REMEDIATION</div>'
    else:
        status_html = '<div class="status-label critical">✕ NON-COMPLIANT</div>'

    # ── Violation cards ─────────────────────────────────────────
    if verdicts:
        violation_cards = "\n".join(
            _render_violation_card(v, i, annotated_image) for i, v in enumerate(verdicts, 1)
        )
    else:
        violation_cards = """
    <div class="ai-badge" style="text-align:center; padding:40px;">
      <strong style="color: var(--compliant); font-size: 16px;">✓ No Violations Detected</strong><br>
      All inspected items meet Saudi Building Code requirements.
    </div>"""

    annotated_media = detection_result.get("annotated_media")
    is_video = annotated_media and annotated_media.get("type") == "video"

    # ── Evidence section ────────────────────────────────────────
    if annotated_image and not is_video:
        evidence_section = f"""
    <div class="evidence-grid">
      <div class="evidence-item" style="grid-column: span 2;">
        <img src="{annotated_image}" alt="Annotated inspection image">
        <div class="evidence-caption">
          <strong>Figure 3.1</strong> — Full annotated inspection image with violation bounding boxes
        </div>
      </div>
    </div>"""
    else:
        # Don't show global image or video if it's a video (we show frames per-violation instead)
        evidence_section = ""

    # ── Reference table rows ────────────────────────────────────
    if verdicts:
        ref_rows = "\n".join(
            _render_ref_table_row(v, i) for i, v in enumerate(verdicts, 1)
        )
    else:
        ref_rows = """
        <tr>
          <td colspan="4" style="text-align:center; padding:20px; color:var(--on-surface-subtle);">
            No code violations to report.
          </td>
        </tr>"""

    # ── Escape metadata ─────────────────────────────────────────
    e_project = _esc(project_name or "Building Inspection")
    e_contractor = _esc(contractor_name or "Not specified")
    e_location = _esc(location or "Not specified")
    e_inspector = _esc(inspector_name or "AI Inspector")
    e_building = _esc(building_type or "Not specified")
    e_permit = _esc(permit_no or "N/A")
    e_class = _esc(classification)

    # ── Heatmap page ─────────────────────────────────────────────
    if heatmap_image:
        heatmap_page = f"""<div class="report-page">
  <div class="report-header">
    <div class="header-logo"><div class="logo-icon">🏛️</div></div>
    <div class="header-org"><div class="org-title">SBC Inspection Report</div></div>
    <div class="header-meta"><div class="report-id">{report_id}</div></div>
  </div>
  <div class="page-content">
    <h2 class="section-title mt-0">4.0 Violation Heatmap</h2>
    <div class="ai-badge" style="margin-bottom:16px;">
      🔥 Spatial concentration of violations — <strong>Red zones</strong> indicate highest risk density
      (weighted by Risk Level × AI Confidence)
    </div>
    <div class="evidence-grid">
      <div class="evidence-item" style="grid-column:span 2;">
        <img src="{heatmap_image}" alt="Violation Heatmap" style="width:100%;border-radius:8px;">
        <div class="evidence-caption">
          <strong>Figure 4.1</strong> — Violation heatmap overlay. Colour scale: 🔵 Blue = low risk → 🟡 Yellow → 🔴 Red = critical hotspot
        </div>
      </div>
    </div>
    <div style="display:flex;gap:16px;margin-top:16px;flex-wrap:wrap;">
      <div style="display:flex;align-items:center;gap:6px;"><span style="width:20px;height:12px;background:#00008B;border-radius:2px;display:inline-block;"></span><small>Minimal risk</small></div>
      <div style="display:flex;align-items:center;gap:6px;"><span style="width:20px;height:12px;background:#00BFFF;border-radius:2px;display:inline-block;"></span><small>Low risk</small></div>
      <div style="display:flex;align-items:center;gap:6px;"><span style="width:20px;height:12px;background:#00FF00;border-radius:2px;display:inline-block;"></span><small>Medium risk</small></div>
      <div style="display:flex;align-items:center;gap:6px;"><span style="width:20px;height:12px;background:#FFD700;border-radius:2px;display:inline-block;"></span><small>High risk</small></div>
      <div style="display:flex;align-items:center;gap:6px;"><span style="width:20px;height:12px;background:#FF0000;border-radius:2px;display:inline-block;"></span><small>Critical hotspot</small></div>
    </div>
  </div>
  <div class="report-footer">
    <span class="confidential">Confidential — For Authorized Personnel Only</span>
    <span class="page-number">Page 4 of 6</span>
  </div>
</div>"""
    else:
        heatmap_page = ""

    # ── AI Recommendations page ───────────────────────────────────
    _urgency_icons = {"IMMEDIATE": "🚨", "SHORT_TERM": "⚠️", "LONG_TERM": "📋"}
    _effort_colors = {"LOW": "#2e7d32", "MEDIUM": "#e65100", "HIGH": "#c62828"}

    if ai_recommendations:
        rec_cards = ""
        for i, rec in enumerate(ai_recommendations, 1):
            sbc = _esc(rec.get("sbc_reference", "N/A"))
            recommendation = _esc(rec.get("recommendation", ""))
            urgency = str(rec.get("urgency", "SHORT_TERM")).upper()
            effort = str(rec.get("estimated_effort", "MEDIUM")).upper()
            party = _esc(rec.get("responsible_party", "Site Engineer"))
            urgency_icon = _urgency_icons.get(urgency, "📋")
            effort_color = _effort_colors.get(effort, "#e65100")

            rec_cards += f"""
    <div style="border:1px solid var(--surface-border);border-radius:8px;padding:16px;margin-bottom:14px;background:var(--surface-container);">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
        <span style="font-size:1.3em;">{urgency_icon}</span>
        <strong style="font-size:0.95em;">REC-{i:02d} &nbsp;|&nbsp; SBC {sbc}</strong>
        <span style="margin-left:auto;font-size:0.75em;padding:2px 8px;border-radius:12px;background:{effort_color};color:#fff;font-weight:600;">{effort} EFFORT</span>
      </div>
      <p style="margin:0 0 10px;font-size:0.9em;line-height:1.5;">{recommendation}</p>
      <div style="display:flex;gap:16px;font-size:0.8em;color:var(--on-surface-subtle);">
        <span>⏱ Urgency: <strong style="color:var(--on-surface);">{urgency.replace('_', ' ')}</strong></span>
        <span>👷 Responsible: <strong style="color:var(--on-surface);">{party}</strong></span>
      </div>
    </div>"""

        recommendations_page = f"""<div class="report-page">
  <div class="report-header">
    <div class="header-logo"><div class="logo-icon">🏛️</div></div>
    <div class="header-org"><div class="org-title">SBC Inspection Report</div></div>
    <div class="header-meta"><div class="report-id">{report_id}</div></div>
  </div>
  <div class="page-content">
    <h2 class="section-title mt-0">5.0 AI Recommendations</h2>
    <div class="ai-badge" style="margin-bottom:16px;">
      🤖 <strong>AI-Generated Remediation Plan</strong> &nbsp;|&nbsp; Based on detected violations and Saudi Building Code requirements
    </div>
    {rec_cards}
  </div>
  <div class="report-footer">
    <span class="confidential">Confidential — For Authorized Personnel Only</span>
    <span class="page-number">Page 5 of 6</span>
  </div>
</div>"""
    else:
        recommendations_page = ""

    # ── Assemble full HTML ──────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SBC Inspection Report — {report_id}</title>
  <style>
{_load_css()}
  </style>
</head>
<body>

<!-- ═══════════════════════════════════════════════════════════
     PAGE 1 — COVER PAGE
     ═══════════════════════════════════════════════════════════ -->
<div class="report-page">
  <div class="report-header">
    <div class="header-logo">
      <div class="logo-icon">🏛️</div>
    </div>
    <div class="header-org">
      <div class="org-title">Saudi Building Code National Committee</div>
      <div class="org-subtitle">Kingdom of Saudi Arabia</div>
    </div>
    <div class="header-meta">
      <div class="report-id">{report_id}</div>
      <div>{report_date}</div>
    </div>
  </div>

  <div class="page-content">
    <div class="cover-title">
      <h1>BUILDING INSPECTION REPORT</h1>
      <div class="subtitle">Official Compliance Record — AI-Assisted Analysis</div>
      <div class="divider"></div>
    </div>

    <div class="metadata-grid">
      <div class="meta-item">
        <span class="meta-label">Report ID</span>
        <span class="meta-value">{report_id}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Project Name</span>
        <span class="meta-value">{e_project}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Contractor Name</span>
        <span class="meta-value">{e_contractor}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Inspection Date</span>
        <span class="meta-value">{report_date}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Classification</span>
        <span class="meta-value">{e_class}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Permit No.</span>
        <span class="meta-value">{e_permit}</span>
      </div>
    </div>

    <h2 class="section-title mt-0">1.0 Compliance Summary</h2>

    <div class="compliance-box">
      <div class="compliance-gauge" style="--score: {compliance_int}">
        <span class="score">{compliance_int}%</span>
      </div>
      <div class="compliance-details">
        {status_html}
        <div class="compliance-stats">
          <div class="stat">
            <div class="stat-value">{total_items}</div>
            <div class="stat-label">Total Items</div>
          </div>
          <div class="stat">
            <div class="stat-value" style="color: var(--critical)">{violation_count}</div>
            <div class="stat-label">Violations</div>
          </div>
          <div class="stat">
            <div class="stat-value" style="color: var(--compliant)">{compliant_count}</div>
            <div class="stat-label">Compliant</div>
          </div>

        </div>
      </div>
    </div>

    <div class="ai-badge">
      <strong>AI-Powered Analysis</strong> &nbsp;|&nbsp;
      Generated: {generated_at}
    </div>
  </div>

  <div class="report-footer">
    <span class="confidential">Confidential — For Authorized Personnel Only</span>
    <span class="page-number">Page 1 of 6</span>
  </div>
</div>


<!-- ═══════════════════════════════════════════════════════════
     PAGE 2 — VIOLATION DETAILS
     ═══════════════════════════════════════════════════════════ -->
<div class="report-page">
  <div class="report-header">
    <div class="header-logo"><div class="logo-icon">🏛️</div></div>
    <div class="header-org"><div class="org-title">SBC Inspection Report</div></div>
    <div class="header-meta"><div class="report-id">{report_id}</div></div>
  </div>

  <div class="page-content">
    <h2 class="section-title mt-0">2.0 Violation Details</h2>
    {violation_cards}
  </div>

  <div class="report-footer">
    <span class="confidential">Confidential — For Authorized Personnel Only</span>
    <span class="page-number">Page 2 of 6</span>
  </div>
</div>


<!-- ═══════════════════════════════════════════════════════════
     PAGE 3 — MEDIA EVIDENCE & SBC REFERENCES
     ═══════════════════════════════════════════════════════════ -->
<div class="report-page">
  <div class="report-header">
    <div class="header-logo"><div class="logo-icon">🏛️</div></div>
    <div class="header-org"><div class="org-title">SBC Inspection Report</div></div>
    <div class="header-meta"><div class="report-id">{report_id}</div></div>
  </div>

  <div class="page-content">
    <h2 class="section-title mt-0">3.0 Media Evidence</h2>
    {evidence_section}

    <h2 class="section-title">4.0 SBCNC Code References</h2>

    <table class="ref-table">
      <thead>
        <tr>
          <th style="width:40px">#</th>
          <th style="width:140px">SBC Code</th>
          <th>Requirement Summary</th>
          <th style="width:100px">Status</th>
        </tr>
      </thead>
      <tbody>
        {ref_rows}
      </tbody>
    </table>
  </div>

  <div class="report-footer">
    <span class="confidential">Confidential — For Authorized Personnel Only</span>
    <span class="page-number">Page 3 of 6</span>
  </div>
</div>


<!-- ═══════════════════════════════════════════════════════════
     PAGE 4 — VIOLATION HEATMAP
     ═══════════════════════════════════════════════════════════ -->
{heatmap_page}


<!-- ═══════════════════════════════════════════════════════════
     PAGE 5 — AI RECOMMENDATIONS
     ═══════════════════════════════════════════════════════════ -->
{recommendations_page}


<!-- ═══════════════════════════════════════════════════════════
     PAGE 4 — METHODOLOGY & SIGN-OFF
     ═══════════════════════════════════════════════════════════ -->
<div class="report-page">
  <div class="report-header">
    <div class="header-logo"><div class="logo-icon">🏛️</div></div>
    <div class="header-org"><div class="org-title">SBC Inspection Report</div></div>
    <div class="header-meta"><div class="report-id">{report_id}</div></div>
  </div>

  <div class="page-content">
    <h2 class="section-title mt-0">5.0 AI Analysis Methodology</h2>

    <div class="methodology-box">
      <div class="method-row">
        <span class="method-label">Engine</span>
        <span class="method-value">Saudi SBC Compliance AI</span>
      </div>
      <div class="method-row">
        <span class="method-label">Analysis Type</span>
        <span class="method-value">Multi-modal Vision + RAG-augmented Code Matching</span>
      </div>
      <div class="method-row">
        <span class="method-label">Knowledge Base</span>
        <span class="method-value">Saudi Building Code National Committee (SBCNC) Database</span>
      </div>
      <div class="method-row">
        <span class="method-label">Rules Evaluated</span>
        <span class="method-value">{total_items} items against 8 SBC code sections per item</span>
      </div>
      <div class="disclaimer">
        ⚠ This AI-generated analysis is advisory only. Final compliance determination
        must be made by a certified SBC inspector in accordance with SBCNC regulations.
      </div>
    </div>
  </div>

  <div class="report-footer">
    <span class="confidential">End of Report — Confidential</span>
    <span class="page-number">Page 6 of 6</span>
  </div>
</div>

</body>
</html>"""

    return html


def save_report(
    detection_result: dict[str, Any],
    output_path: str | Path,
    **metadata,
) -> str:
    """
    Generate and save a self-contained HTML report.

    Returns
    -------
    str
        The absolute path of the saved HTML report file.
    """
    html = generate_report_html(detection_result, **metadata)
    output_path = Path(output_path).with_suffix(".html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report saved to %s", output_path)
    return str(output_path.resolve())
