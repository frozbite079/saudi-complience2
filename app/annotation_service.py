from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np
import requests

# ── Color palette (BGR) — border/badge fill colors by priority ───────────
PRIORITY_COLORS_BGR = {
    "CRITICAL": (0, 0, 220),
    "HIGH":     (0, 140, 255),
    "MEDIUM":   (0, 200, 255),
    "LOW":      (200, 100, 0),
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
BLACK      = (0, 0, 0)
WHITE      = (255, 255, 255)


def _load_image(media_path_or_url: str) -> np.ndarray:
    if media_path_or_url.startswith(("http://", "https://")):
        response = requests.get(media_path_or_url, timeout=30)
        response.raise_for_status()
        data = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(media_path_or_url)

    if image is None:
        raise ValueError("Unable to load image for annotation")
    return image


def _to_pixel_bbox(
    bbox: list[Any], width: int, height: int
) -> tuple[int, int, int, int] | None:
    if len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None

    if max(x1, y1, x2, y2) <= 1.0:
        x1, x2 = x1 * width,  x2 * width
        y1, y2 = y1 * height, y2 * height
    else:
        x1, x2 = x1 / 1000 * width,  x2 / 1000 * width
        y1, y2 = y1 / 1000 * height, y2 / 1000 * height

    left   = max(0, min(width  - 1, int(round(min(x1, x2)))))
    top    = max(0, min(height - 1, int(round(min(y1, y2)))))
    right  = max(0, min(width  - 1, int(round(max(x1, x2)))))
    bottom = max(0, min(height - 1, int(round(max(y1, y2)))))

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _truncate(text: str, max_chars: int = 38) -> str:
    return text if len(text) <= max_chars else text[:max_chars - 1] + "..."


def _get_dynamic_scaling(width: int, height: int) -> tuple[float, float, int]:
    """Calculate proportional font scales and thickness based on image resolution."""
    # Base resolution 1000px wide -> fs_small=0.45, fs_normal=0.55, thick=1
    scale_factor = max(1.0, width / 1000.0)
    fs_small = 0.45 * scale_factor
    fs_normal = 0.55 * scale_factor
    thick = max(1, int(round(1.5 * scale_factor)))
    return fs_small, fs_normal, thick


def _draw_violation_label(
    image: np.ndarray,
    left: int,
    top: int,
    right: int,
    color: tuple,
    priority: str,
    cv_target: str,
    sbc_ref: str,
) -> None:
    """
    Draw a professional 3-row label tag above a bounding box.
    """
    height, width = image.shape[:2]
    fs_small, fs_normal, thick = _get_dynamic_scaling(width, height)
    pad = max(4, int(5 * (width / 1000.0)))

    short_name = _truncate(cv_target or "Violation", 42)
    ref_text   = f"SBC Art.{sbc_ref}" if sbc_ref else "SBC Ref: N/A"
    badge_text = priority or "N/A"

    # Measure badge
    (bw, bh), _ = cv2.getTextSize(badge_text, FONT, fs_small, thick)
    badge_w = bw + pad * 2 + max(2, thick)

    # Measure name row
    (nw, nh), _ = cv2.getTextSize(short_name, FONT, fs_normal, thick)

    # Measure ref row
    (rw, rh), _ = cv2.getTextSize(ref_text, FONT, fs_small, thick)

    row1_h = max(bh, nh) + pad * 2
    row2_h = rh + pad * 2
    total_h = row1_h + row2_h

    box_w = max(badge_w + nw + pad * 3, rw + pad * 2, right - left)
    box_x1 = left
    box_x2 = min(image.shape[1] - 1, left + box_w)
    box_y1 = max(0, top - total_h)
    box_y2 = top

    # White background for the label
    cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), WHITE, -1)
    # Colored left accent bar
    cv2.rectangle(image, (box_x1, box_y1), (box_x1 + max(4, thick*2), box_y2), color, -1)
    # Outer border
    cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), color, thick)

    # — Row 1: priority badge + short name —
    badge_x1 = box_x1 + max(6, thick*3)
    badge_y1 = box_y1 + pad
    badge_x2 = badge_x1 + badge_w
    badge_y2 = badge_y1 + bh + pad
    cv2.rectangle(image, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
    cv2.putText(
        image, badge_text,
        (badge_x1 + pad, badge_y2 - max(1, pad // 2)),
        FONT, fs_small, BLACK, thick, cv2.LINE_AA,
    )
    cv2.putText(
        image, short_name,
        (badge_x2 + pad, badge_y1 + nh + max(1, pad // 2)),
        FONT, fs_normal, BLACK, thick, cv2.LINE_AA,
    )

    # — Row 2: SBC reference —
    ref_y = box_y1 + row1_h + rh + max(1, pad // 2)
    cv2.putText(
        image, ref_text,
        (box_x1 + max(10, pad*2), ref_y),
        FONT, fs_small, BLACK, thick, cv2.LINE_AA,
    )


def _draw_spatial_violation(
    image: np.ndarray,
    item: dict,
    width: int,
    height: int,
) -> None:
    bbox = _to_pixel_bbox(item["bbox"], width, height)
    if not bbox:
        return
    priority = str(item.get("priority", "LOW")).upper()
    color    = PRIORITY_COLORS_BGR.get(priority, PRIORITY_COLORS_BGR["LOW"])
    cv_target = str(item.get("cv_target") or item.get("label") or "Violation")
    sbc_ref   = str(item.get("sbc_reference") or "")
    left, top, right, bottom = bbox

    # Colored bounding box border (4px)
    cv2.rectangle(image, (left, top), (right, bottom), color, 4)

    # Structured label tag above the box
    _draw_violation_label(image, left, top, right, color, priority, cv_target, sbc_ref)


def _draw_non_spatial_violation(
    image: np.ndarray,
    item: dict,
    index: int,
) -> None:
    """
    Draws a structured bounding box in the left margin for violations
    that GLM could not spatially localize in the image.
    """
    height, width = image.shape[:2]
    fs_small, fs_normal, thick = _get_dynamic_scaling(width, height)
    # Non-spatial labels are slightly smaller
    fs_small *= 0.9
    fs_normal *= 0.9

    priority  = str(item.get("priority", "LOW")).upper()
    color     = PRIORITY_COLORS_BGR.get(priority, PRIORITY_COLORS_BGR["LOW"])
    cv_target = str(item.get("cv_target") or item.get("label") or "Violation")
    sbc_ref   = str(item.get("sbc_reference") or "")

    pad       = max(4, int(5 * (width / 1000.0)))
    box_w     = max(280, int(280 * (width / 1000.0)))
    box_h     = max(60, int(60 * (width / 1000.0)))
    
    x1        = max(8, int(8 * (width / 1000.0)))
    y1        = max(8, int(8 * (width / 1000.0))) + index * (box_h + max(6, thick*3))
    x2        = x1 + box_w
    y2        = y1 + box_h

    short_name = _truncate(cv_target, 36)
    ref_text   = f"SBC Art.{sbc_ref}" if sbc_ref else "SBC Ref: N/A"
    badge_text = priority

    # White background + colored border
    cv2.rectangle(image, (x1, y1), (x2, y2), WHITE, -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thick * 2)
    # Colored left bar
    cv2.rectangle(image, (x1, y1), (x1 + max(4, thick*2), y2), color, -1)

    # Priority badge
    (bw, bh), _ = cv2.getTextSize(badge_text, FONT, fs_small, thick)
    bx1, by1 = x1 + max(8, thick*4), y1 + pad
    bx2, by2 = bx1 + bw + max(8, thick*4), by1 + bh + max(4, thick*2)
    cv2.rectangle(image, (bx1, by1), (bx2, by2), color, -1)
    cv2.putText(image, badge_text, (bx1 + max(4, thick*2), by2 - max(1, thick)), FONT, fs_small, BLACK, thick, cv2.LINE_AA)

    # Short violation name
    cv2.putText(image, short_name, (bx2 + max(6, thick*3), by1 + bh + max(2, thick)), FONT, fs_normal, BLACK, thick, cv2.LINE_AA)

    # SBC reference
    cv2.putText(image, ref_text, (x1 + max(10, pad*2), y2 - max(4, thick*2)), FONT, fs_small, BLACK, thick, cv2.LINE_AA)

    # "Not localizable" note
    cv2.putText(image, "(not spatially localizable)", (bx2 + max(6, thick*3), by2 + max(1, thick)),
                FONT, fs_small * 0.75, (120, 120, 120), max(1, thick-1), cv2.LINE_AA)


def annotate_image_base64(
    media_path_or_url: str,
    localized_violations: list[dict[str, Any]],
) -> str | None:
    if not localized_violations:
        return None

    image = _load_image(media_path_or_url)
    height, width = image.shape[:2]

    spatial     = [v for v in localized_violations if v.get("bbox")]
    non_spatial = [v for v in localized_violations if not v.get("bbox")]

    for item in spatial:
        _draw_spatial_violation(image, item, width, height)

    for i, item in enumerate(non_spatial):
        _draw_non_spatial_violation(image, item, i)

    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("Failed to encode annotated image")

    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


# ── Video frame annotation ─────────────────────────────────────────────

def _extract_frame_at_timestamp(video_path: str, timestamp_sec: float) -> np.ndarray | None:
    """Extract a single frame from a video file at the given timestamp (seconds)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_index = int(round(timestamp_sec * fps))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = max(0, min(frame_index, total - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def annotate_video_frame_base64(
    frame_bgr: np.ndarray,
    violations: list[dict[str, Any]],
) -> str | None:
    """Draw violation bounding boxes onto a raw BGR frame, return as base64 JPEG."""
    if not violations or frame_bgr is None:
        return None

    image = frame_bgr.copy()
    height, width = image.shape[:2]

    spatial     = [v for v in violations if v.get("bbox")]
    non_spatial = [v for v in violations if not v.get("bbox")]

    for item in spatial:
        _draw_spatial_violation(image, item, width, height)

    for i, item in enumerate(non_spatial):
        _draw_non_spatial_violation(image, item, i)

    # Draw timestamp label in bottom-left corner
    ts = violations[0].get("timestamp_sec")
    if ts is not None:
        fs_small, _, thick = _get_dynamic_scaling(width, height)
        ts_text = f"t = {float(ts):.1f}s"
        (tw, th), _ = cv2.getTextSize(ts_text, FONT, fs_small * 1.2, thick)
        pad = max(6, int(8 * (width / 1000.0)))
        cv2.rectangle(image, (0, height - th - pad*3), (tw + pad*3, height), (0, 0, 0), -1)
        cv2.putText(image, ts_text, (pad, height - pad),
                    FONT, fs_small * 1.2, (255, 255, 255), thick, cv2.LINE_AA)

    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return None

    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def annotate_video_frames(
    video_path: str,
    violations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    For each violation with a timestamp, extract the frame from the video,
    draw bounding boxes, and return a list of annotated frame dicts:
    [{"timestamp_sec": 12.5, "annotated_frame": "data:image/jpeg;base64,...", "violations": [...]}]
    """
    if not violations:
        return []

    # Group violations by timestamp (round to nearest 0.5s to cluster nearby items)
    from collections import defaultdict
    ts_groups: dict[float, list[dict]] = defaultdict(list)
    for v in violations:
        ts = v.get("timestamp_sec")
        if ts is not None:
            # Round to 0.5s buckets to group nearby timestamps
            bucket = round(float(ts) * 2) / 2.0
            ts_groups[bucket].append(v)
        else:
            # No timestamp — put in a special -1 bucket
            ts_groups[-1.0].append(v)

    results: list[dict[str, Any]] = []
    for ts_bucket, group_violations in sorted(ts_groups.items()):
        if ts_bucket < 0:
            continue  # skip violations without timestamps

        # Use the actual timestamp of the first violation for extraction
        actual_ts = float(group_violations[0].get("timestamp_sec", ts_bucket))
        frame = _extract_frame_at_timestamp(video_path, actual_ts)
        if frame is None:
            continue

        annotated_b64 = annotate_video_frame_base64(frame, group_violations)
        if annotated_b64:
            results.append({
                "timestamp_sec": actual_ts,
                "annotated_frame": annotated_b64,
                "violations": group_violations,
            })

    return results


def annotate_full_video(
    video_path: str,
    violations: list[dict[str, Any]],
    output_path: str,
    overlay_duration: float = 3.0,
) -> str | None:
    """
    Read the entire source video frame-by-frame.  For each violation that has
    a ``timestamp_sec``, draw the bounding-box overlay on every frame within
    ±overlay_duration seconds of that timestamp.  Write the result as an MP4.

    Returns the *output_path* on success, or ``None`` on failure.
    """
    if not violations:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        return None

    # Build a lookup: for each violation, compute the frame range where we
    # should draw its overlay  (start_frame, end_frame, violation_dict)
    overlay_ranges: list[tuple[int, int, dict]] = []
    for v in violations:
        ts = v.get("timestamp_sec")
        if ts is None:
            continue
        ts = float(ts)
        start = max(0, int((ts - overlay_duration) * fps))
        end   = min(total_frames - 1, int((ts + overlay_duration) * fps))
        overlay_ranges.append((start, end, v))

    # Sort by start frame for efficient processing
    overlay_ranges.sort(key=lambda x: x[0])

    # Use mp4v codec — broadly compatible
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_path = output_path.replace(".mp4", "_temp.mp4")
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Collect all violations whose overlay window covers this frame
        active_violations = [
            v for (s, e, v) in overlay_ranges if s <= frame_idx <= e
        ]

        if active_violations:
            # Draw spatial violations
            spatial = [v for v in active_violations if v.get("bbox")]
            for item in spatial:
                _draw_spatial_violation(frame, item, width, height)

            # Draw non-spatial in top-left margin
            non_spatial = [v for v in active_violations if not v.get("bbox")]
            for i, item in enumerate(non_spatial):
                _draw_non_spatial_violation(frame, item, i)

            # Timestamp badge in bottom-left
            current_sec = frame_idx / fps
            fs_small, _, thick = _get_dynamic_scaling(width, height)
            ts_text = f"t = {current_sec:.1f}s"
            (tw, th), _ = cv2.getTextSize(ts_text, FONT, fs_small * 1.2, thick)
            pad = max(6, int(8 * (width / 1000.0)))
            cv2.rectangle(
                frame,
                (0, height - th - pad * 3),
                (tw + pad * 3, height),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame, ts_text, (pad, height - pad),
                FONT, fs_small * 1.2, WHITE, thick, cv2.LINE_AA,
            )

            # "VIOLATION DETECTED" banner at top-center
            banner = "VIOLATION DETECTED"
            (bw, bh), _ = cv2.getTextSize(banner, FONT, fs_small * 1.4, thick + 1)
            bx = (width - bw) // 2
            by = bh + pad * 3
            cv2.rectangle(frame, (bx - pad * 2, pad), (bx + bw + pad * 2, by + pad), (0, 0, 180), -1)
            cv2.putText(frame, banner, (bx, by), FONT, fs_small * 1.4, WHITE, thick + 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    import os
    import subprocess
    
    # Use FFmpeg to re-encode to h264/yuv420p for max compatibility
    try:
        cmd = [
            "ffmpeg", "-y", "-i", temp_path,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        # Fallback if ffmpeg fails
        if os.path.exists(temp_path):
            os.rename(temp_path, output_path)

    # Verify the output file was actually created and has content
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    return None


# ── Violation Heatmap ──────────────────────────────────────────────────────

_RISK_WEIGHT = {"CRITICAL": 3.0, "HIGH": 2.0, "MEDIUM": 1.0, "LOW": 0.5}


def generate_violation_heatmap(
    media_path_or_url: str,
    violations: list[dict[str, Any]],
) -> str | None:
    """
    Generate a heatmap overlay showing spatial concentration of violations.

    Each violation with a bbox contributes a Gaussian blob weighted by
    risk_level_weight × confidence. The heat layer is colour-mapped
    (COLORMAP_JET: blue→green→yellow→red) and blended on top of the original.

    Returns a ``data:image/jpeg;base64,...`` string, or None on failure.
    """
    try:
        image = _load_image(media_path_or_url)
    except Exception:
        return None

    height, width = image.shape[:2]

    heat = np.zeros((height, width), dtype=np.float32)

    spatial = [v for v in violations if v.get("bbox")]
    if not spatial:
        return None

    for v in spatial:
        bbox = _to_pixel_bbox(v["bbox"], width, height)
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        sigma_x = max(10, (x2 - x1) // 2)
        sigma_y = max(10, (y2 - y1) // 2)

        risk = _RISK_WEIGHT.get(str(v.get("priority", "MEDIUM")).upper(), 1.0)
        conf = float(v.get("confidence", 0.8))
        weight = risk * conf

        ys = np.arange(height, dtype=np.float32)
        xs = np.arange(width, dtype=np.float32)
        gx = np.exp(-0.5 * ((xs - cx) / sigma_x) ** 2)
        gy = np.exp(-0.5 * ((ys - cy) / sigma_y) ** 2)
        blob = np.outer(gy, gx) * weight
        heat += blob

    max_val = heat.max()
    if max_val == 0:
        return None

    heat_norm = (heat / max_val * 255).astype(np.uint8)
    colormap = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image, 0.55, colormap, 0.45, 0)

    ok, buf = cv2.imencode(".jpg", blended, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None

    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"
