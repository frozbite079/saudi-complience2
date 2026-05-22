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
    priority = _esc(v.get("priority", "MEDIUM"))
    vio_id = f"VIO-{index:03d}"
    sbc_ref = _esc(v.get("sbc_reference", "N/A"))
    title = _esc(v.get("cv_target") or (v.get("source_text", "")[:60]))
    description = _esc(v.get("source_text", ""))
    rule_text = _esc(v.get("rule_text", "See SBCNC documentation"))
    remediation = _esc(v.get("remediation", ""))
    bbox = v.get("bbox")

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

    return f"""
    <div class="violation-card severity-{severity}">
      <div class="card-header">
        <span class="badge badge-{severity}">{priority}</span>
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
              <span class="detail-label">Priority</span>
              <span class="detail-value"><span class="badge badge-{severity}">{priority}</span></span>
            </div>
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
    project_name: str = "Building Inspection",
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
    verdicts = detection_result.get("verdicts", [])
    annotated_image = detection_result.get("annotated_image")
    token_usage = detection_result.get("token_usage", {})

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
    e_location = _esc(location or "Not specified")
    e_inspector = _esc(inspector_name or "AI Inspector")
    e_building = _esc(building_type or "Not specified")
    e_permit = _esc(permit_no or "N/A")
    e_class = _esc(classification)

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
    <span class="page-number">Page 1 of 4</span>
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
    <span class="page-number">Page 2 of 4</span>
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
    <span class="page-number">Page 3 of 4</span>
  </div>
</div>


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
    <span class="page-number">Page 4 of 4</span>
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
