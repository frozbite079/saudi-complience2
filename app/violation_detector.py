from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
import shutil
import base64
import mimetypes
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from app.annotation_service import annotate_full_video, annotate_image_base64, generate_violation_heatmap
from app.config import RAG_TOP_K, VIDEO_OUTPUT_DIR
from app.embedding_service import embed_text
from app.llm_service import judge, observe_items, localize_violations, generate_ai_recommendations
from app.weaviate_client import ALLOWED_CLASSIFICATIONS, CLASSIFICATION_TO_DB_CATEGORY, search_rules

logger = logging.getLogger(__name__)

def _get_original_media(media_path_or_url: str, is_video: bool) -> tuple[str | None, str | None]:
    """Returns (annotated_image_b64, annotated_video_filename) by copying original media."""
    annotated_image = None
    annotated_video_filename = None
    
    if is_video:
        annotated_video_filename = f"original_{uuid.uuid4().hex[:12]}.mp4"
        output_path = str(VIDEO_OUTPUT_DIR / annotated_video_filename)
        try:
            if media_path_or_url.startswith("http"):
                urllib.request.urlretrieve(media_path_or_url, output_path)
            else:
                shutil.copy2(media_path_or_url, output_path)
        except Exception as e:
            logger.error("Failed to copy original video: %s", e)
            annotated_video_filename = None
    else:
        try:
            if media_path_or_url.startswith("http"):
                resp = requests.get(media_path_or_url)
                resp.raise_for_status()
                b64 = base64.b64encode(resp.content).decode("utf-8")
                mime = mimetypes.guess_type(media_path_or_url)[0] or "image/jpeg"
            else:
                with open(media_path_or_url, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                mime = mimetypes.guess_type(media_path_or_url)[0] or "image/jpeg"
            annotated_image = f"data:{mime};base64,{b64}"
        except Exception as e:
            logger.error("Failed to read original image: %s", e)
            annotated_image = None
            
    return annotated_image, annotated_video_filename

# ── In-memory result cache keyed by (image_hash, custom_prompt) ──────────
# Cleared when the process restarts. Prevents GLM non-determinism on repeat
# submissions of the same image.
_RESULT_CACHE: dict[str, dict[str, Any]] = {}


def _image_cache_key(media_path_or_url: str, custom_prompt: str) -> str:
    """Return a stable cache key for a given image file + prompt combination."""
    hasher = hashlib.md5()
    try:
        if media_path_or_url.startswith(("http://", "https://")):
            # For URLs just hash the URL string itself
            hasher.update(media_path_or_url.encode())
        else:
            with open(media_path_or_url, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
    except OSError:
        hasher.update(media_path_or_url.encode())
    hasher.update(custom_prompt.encode())
    return hasher.hexdigest()


CANONICAL_CLASSIFICATIONS = {
    "structural safety": "Structural Safety",
    "structural": "Structural Safety",
    "electrical": "Electrical",
    "electricity": "Electrical",
    "plumbing": "Plumbing",
    "fire safety": "Fire Safety",
    "fire": "Fire Safety",
}


def _repair_json(text: str) -> str:
    """
    Multi-pass repair for common LLM JSON failures:
    1. Remove markdown fences
    2. Extract the outermost [...] array
    3. Fix trailing commas before ] or }
    4. Close any unclosed string at end of text (truncation)
    5. Close any unclosed array/object brackets
    """
    # Strip markdown fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Extract first [...] block (greedy to get whole array)
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        text = m.group(0)
    elif not text.startswith("["):
        # Wrap bare objects into an array
        text = "[" + text + "]"

    # Fix trailing commas  e.g.  , ]  or  , }
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # If string appears truncated mid-value, close it
    # Count unmatched quotes, add closing if needed
    # Simple heuristic: if text doesn't end with ] add missing closers
    stripped = text.rstrip()
    if not stripped.endswith("]"):
        # Close any open string
        quote_count = stripped.count('"') - stripped.count('\\"')
        if quote_count % 2 != 0:
            stripped += '"'
        # Close open object and array
        open_braces  = stripped.count("{") - stripped.count("}")
        open_brackets = stripped.count("[") - stripped.count("]")
        stripped += "}" * max(0, open_braces)
        stripped += "]" * max(0, open_brackets)
        text = stripped

    return text


def _extract_objects_from_text(text: str) -> list[dict[str, Any]]:
    """
    Last-resort: find every {...} block in the text and try to parse each one.
    Returns a list of successfully parsed objects.
    """
    results = []
    for match in re.finditer(r"\{[^{}]+\}", text, re.DOTALL):
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                results.append(obj)
        except json.JSONDecodeError:
            pass
    return results


def _parse_json_from_llm(text: str) -> list[dict[str, Any]]:
    if not text or not text.strip():
        logger.warning("LLM returned empty response, skipping JSON parse.")
        return []

    # Stage 1 — direct parse (fast path, no modifications)
    stripped = text.strip()
    try:
        result = json.loads(stripped)
        return result if isinstance(result, list) else [result]
    except json.JSONDecodeError:
        pass

    # Stage 2 — repair common structural issues then parse
    try:
        repaired = _repair_json(stripped)
        result = json.loads(repaired)
        return result if isinstance(result, list) else [result]
    except json.JSONDecodeError:
        pass

    # Stage 3 — object-by-object extraction (most tolerant)
    logger.warning("LLM JSON repair failed, falling back to object extraction.")
    objects = _extract_objects_from_text(stripped)
    if objects:
        return objects

    logger.error("Could not extract any JSON from LLM output. Raw snippet: %.200s", stripped)
    return []

def _parse_localization_json(text: str) -> list[dict[str, Any]]:
    parsed = _parse_json_from_llm(text)
    return [item for item in parsed if isinstance(item, dict)]


def _parse_observed_items(text: str, is_video: bool = False) -> list[dict[str, Any]]:
    items = _parse_json_from_llm(text)
    normalized_items: list[dict[str, Any]] = []

    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue

        category = _normalize_classification(str(item.get("category", "")))
        observation_text = str(item.get("text", "")).strip()
        bbox = item.get("bbox")

        if not category or not observation_text:
            continue

        parsed = {
            "id": f"item_{index}",
            "category": category,
            "db_category": CLASSIFICATION_TO_DB_CATEGORY.get(category, category),
            "text": observation_text,
            "bbox": bbox if isinstance(bbox, list) and len(bbox) == 4 else None,
        }

        # Preserve timestamp from GLM video output
        if is_video and item.get("timestamp_sec") is not None:
            try:
                parsed["timestamp_sec"] = float(item["timestamp_sec"])
            except (TypeError, ValueError):
                pass

        normalized_items.append(parsed)

    return normalized_items


def _build_summary(verdicts: list[dict]) -> dict[str, Any]:
    violations = [v for v in verdicts if v.get("verdict") == "VIOLATION"]
    compliant = [v for v in verdicts if v.get("verdict") == "COMPLIANT"]
    uncertain = [v for v in verdicts if v.get("verdict") == "UNCERTAIN"]

    high_violations = [
        v for v in violations if v.get("priority") in {"CRITICAL", "HIGH"}
    ]

    return {
        "total_rules_evaluated": len(verdicts),
        "violations": len(violations),
        "compliant": len(compliant),
        "uncertain": len(uncertain),
        "high_priority_violations": len(high_violations),
        "compliance_rate": (
            round(len(compliant) / len(verdicts) * 100, 1) if verdicts else 0.0
        ),
    }


def _normalize_classification(value: str) -> str:
    cleaned = value.strip().strip('"').strip("'").lower()
    if cleaned in CANONICAL_CLASSIFICATIONS:
        return CANONICAL_CLASSIFICATIONS[cleaned]

    lines = [line.strip(" -:.\t").lower() for line in cleaned.splitlines()]
    for line in lines:
        if line in CANONICAL_CLASSIFICATIONS:
            return CANONICAL_CLASSIFICATIONS[line]

    for key, classification in sorted(
        CANONICAL_CLASSIFICATIONS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if re.search(rf"\b{re.escape(key)}\b", cleaned):
            return classification
    return ""


def _merge_consecutive_violations(v_list: list[dict], max_gap_sec: float = 5.0) -> list[dict]:
    if not v_list:
        return []
        
    def parse_hms(hms_str: str) -> float:
        try:
            parts = hms_str.split(":")
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(hms_str)
        except Exception:
            return 0.0

    def format_ts_hms(seconds: float) -> str:
        h = int(seconds) // 3600
        m = (int(seconds) % 3600) // 60
        s = int(seconds) % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    # Sort by sbc_reference and timestamp_sec
    sorted_v = sorted(
        v_list,
        key=lambda x: (str(x.get("sbc_reference") or ""), float(x.get("timestamp_sec") or 0.0))
    )
    
    merged: list[dict] = []
    for item in sorted_v:
        # Pre-initialize timestamp_formatted and timestamp_range if they are not already set
        t_val = item.get("timestamp_sec")
        if t_val is not None:
            t_sec = float(t_val)
            if not item.get("timestamp_formatted"):
                item["timestamp_formatted"] = format_ts_hms(t_sec)
            if not item.get("timestamp_range"):
                item["timestamp_range"] = item["timestamp_formatted"]

        if not merged:
            merged.append(dict(item))
            continue
            
        prev = merged[-1]
        same_ref = str(item.get("sbc_reference") or "") == str(prev.get("sbc_reference") or "")
        
        t_curr = float(item.get("timestamp_sec") or 0.0)
        t_prev = float(prev.get("timestamp_sec") or 0.0)
        close_in_time = (t_curr - t_prev) <= max_gap_sec
        
        if same_ref and close_in_time:
            # Keep the one with higher confidence
            conf_curr = float(item.get("confidence") or 0.0)
            conf_prev = float(prev.get("confidence") or 0.0)
            
            if "occurrences" not in prev:
                prev["occurrences"] = [t_prev]
            prev["occurrences"].append(t_curr)
            
            # Combine the evidences cleanly
            evidences = []
            if prev.get("evidence"):
                evidences.append(prev["evidence"])
            if item.get("evidence") and item.get("evidence") != prev.get("evidence"):
                evidences.append(item["evidence"])
            combined_evidence = "; ".join(evidences)
            
            if conf_curr > conf_prev:
                # Update with higher confidence values
                for key in ["confidence", "confidence_pct", "bbox", "frame_b64"]:
                    if key in item and item[key] is not None:
                        prev[key] = item[key]
            
            # Set combined evidence
            prev["evidence"] = combined_evidence
            
            # Build timestamp range string
            t_start = min(prev["occurrences"])
            t_end = max(prev["occurrences"])
            prev["timestamp_range"] = f"{format_ts_hms(t_start)} - {format_ts_hms(t_end)}"
            # Update the main timestamp to be the start timestamp
            prev["timestamp_sec"] = t_start
            prev["timestamp_formatted"] = format_ts_hms(t_start)
        else:
            merged.append(dict(item))
            
    # Clean up temporary occurrences key and ensure all have range and formatted keys
    for item in merged:
        item.pop("occurrences", None)
        t_val = item.get("timestamp_sec")
        if t_val is not None:
            t_sec = float(t_val)
            if not item.get("timestamp_formatted"):
                item["timestamp_formatted"] = format_ts_hms(t_sec)
            if not item.get("timestamp_range"):
                item["timestamp_range"] = item["timestamp_formatted"]
        
    return merged


def detect_violations(
    media_path_or_url: str,
    is_video: bool = False,
    custom_prompt: str = "",
    top_k: int | None = None,
    classification: str | None = None,
    project_name: str = "Building Inspection",
    contractor_name: str = "",
) -> dict[str, Any]:
    logger.info(
        "Starting violation detection | media=%s | video=%s | project=%s | contractor=%s",
        media_path_or_url[:80],
        is_video,
        project_name,
        contractor_name,
    )

    # ── Cache check (skip for videos — frames change each run) ───────────
    if not is_video:
        cache_key = _image_cache_key(media_path_or_url, custom_prompt or "")
        if cache_key in _RESULT_CACHE:
            logger.info("Cache HIT for %s — returning cached result.", media_path_or_url[:60])
            return _RESULT_CACHE[cache_key]
        logger.info("Cache MISS — running full analysis.")
    else:
        cache_key = None

    # ── Step 1: Dynamic visual item extraction ──────────────────────────
    logger.info("Step 1/3: Dynamic visual item extraction via GLM-4.6V")
    items_raw, observe_tokens = observe_items(media_path_or_url, is_video, custom_prompt)
    
    print("\n" + "="*50)
    print("DEBUG: GLM 'observe_items' Raw Output")
    print("="*50)
    print(items_raw)
    print("="*50 + "\n")
    
    observed_items = _parse_observed_items(items_raw, is_video=is_video)

    if classification:
        forced_category = _normalize_classification(classification)
        if forced_category not in ALLOWED_CLASSIFICATIONS:
            raise ValueError(
                f"classification must be one of: {', '.join(sorted(ALLOWED_CLASSIFICATIONS))}"
            )
        for item in observed_items:
            item["category"] = forced_category
            item["db_category"] = CLASSIFICATION_TO_DB_CATEGORY.get(
                forced_category,
                forced_category,
            )

    if not observed_items:
        orig_img, orig_vid = _get_original_media(media_path_or_url, is_video)
        return {
            "error": "GLM did not return usable observed items",
            "raw_observed_items": items_raw,
            "observed_items": [],
            "observation": "",
            "rules_retrieved": [],
            "verdicts": [],
            "summary": _build_summary([]),
            "project_name": project_name,
            "contractor_name": contractor_name,
            "annotated_image": orig_img,
            "annotated_media": (
                {"type": "image", "data_url": orig_img}
                if orig_img
                else {
                    "type": "video",
                    "download_url": f"/api/v1/download/{orig_vid}" if orig_vid else None,
                    "filename": orig_vid,
                }
                if is_video
                else None
            ),
            "token_usage": {"observe": observe_tokens, "judge": {}},
        }

    # ── Step 2/3: Per-item category-filtered RAG + judgment (PARALLEL) ──
    all_rules: list[dict[str, Any]] = []
    all_verdicts: list[dict[str, Any]] = []
    judge_tokens_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def _process_item(item: dict[str, Any]) -> tuple[list, list, dict]:
        """Embed → search → judge for a single observed item. Thread-safe."""
        item_rules: list[dict] = []
        item_verdicts: list[dict] = []
        item_tokens: dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        try:
            logger.info(
                "Searching rules for %s | db_category=%s",
                item["id"],
                item["db_category"],
            )
            query_vector = embed_text(item["text"])
            rules = search_rules(
                query_vector,
                top_k=top_k,
                classification=item["category"],
            )

            for rule in rules:
                rule_copy = dict(rule)
                rule_copy["source_item_id"] = item["id"]
                item_rules.append(rule_copy)

            if not rules:
                return item_rules, item_verdicts, item_tokens

            judgment_raw, judge_tokens = judge(item["text"], rules)
            
            print("\n" + "="*50)
            print(f"DEBUG: GLM 'judge' Raw Output for item '{item['id']}'")
            print("="*50)
            print(judgment_raw)
            print("="*50 + "\n")
            
            for key in item_tokens:
                item_tokens[key] += judge_tokens.get(key, 0)

            verdicts = _parse_json_from_llm(judgment_raw)
            for verdict in verdicts:
                if not isinstance(verdict, dict):
                    continue
                verdict["source_item_id"] = item["id"]
                verdict["source_text"]    = item["text"]
                verdict["bbox"]           = item.get("bbox")
                # Carry timestamp from observed item onto the verdict
                if "timestamp_sec" in item:
                    verdict["timestamp_sec"] = item["timestamp_sec"]
                item_verdicts.append(verdict)

        except Exception:
            logger.exception("Error processing item %s", item.get("id"))

        return item_rules, item_verdicts, item_tokens

    # Run all items concurrently — max_workers capped at item count to avoid over-threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    max_workers = min(len(observed_items), 6)
    logger.info("Processing %d items in parallel (workers=%d)", len(observed_items), max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_item, item): item for item in observed_items}
        for future in as_completed(futures):
            i_rules, i_verdicts, i_tokens = future.result()
            all_rules.extend(i_rules)
            all_verdicts.extend(i_verdicts)
            for key in judge_tokens_total:
                judge_tokens_total[key] += i_tokens.get(key, 0)



    if not all_rules:
        logger.warning("No rules retrieved from Weaviate for observed items")

    # ── Step 3/3: Annotate violations ────────────────────────────────────
    localized: list[dict[str, Any]] = []
    annotated_image: str | None = None
    annotated_video_path: str | None = None
    annotated_video_filename: str | None = None
    violations = [v for v in all_verdicts if v.get("verdict") == "VIOLATION"]

    if violations:
        # Call localize_violations to get mandatory bboxes for every violation
        logger.info("Step 3/3: Localizing %d violations via LLM", len(violations))
        loc_raw, _ = localize_violations(media_path_or_url, is_video, violations)
        loc_parsed = _parse_localization_json(loc_raw)

        # Build a lookup: sbc_reference -> bbox from localization step
        loc_bbox_map: dict[str, list] = {}
        for loc in loc_parsed:
            ref = loc.get("sbc_reference", "")
            bbox = loc.get("bbox")
            if ref and isinstance(bbox, list) and len(bbox) == 4:
                loc_bbox_map[ref] = bbox

        logger.info("Localization returned bboxes for %d/%d violations", len(loc_bbox_map), len(violations))

        localized = []
        for violation in violations:
            # Prefer bbox from localization step; fall back to observe-step bbox
            new_bbox = loc_bbox_map.get(violation.get("sbc_reference", "")) or violation.get("bbox")
            
            # Update the original violation dict so the API return gets the right bbox
            violation["bbox"] = new_bbox
            
            localized.append({
                "sbc_reference":  violation.get("sbc_reference", ""),
                "priority":       violation.get("priority", ""),
                "cv_target":      violation.get("cv_target", ""),
                "label":          violation.get("cv_target") or violation.get("sbc_reference") or "Violation",
                "bbox":           new_bbox,
                "source_item_id": violation.get("source_item_id"),
                "timestamp_sec":  violation.get("timestamp_sec"),
            })

        if is_video:
            from app.annotation_service import _extract_frame_at_timestamp, annotate_video_frame_base64
            import os
            try:
                for violation in violations:
                    ts = violation.get("timestamp_sec")
                    if ts is None:
                        continue
                        
                    # Find the corresponding localized dict for this violation
                    loc_item = next((loc for loc in localized if loc["sbc_reference"] == violation.get("sbc_reference", "")), None)
                    if not loc_item:
                        continue
                        
                    # Extract raw frame
                    frame = _extract_frame_at_timestamp(media_path_or_url, float(ts))
                    if frame is None:
                        continue
                        
                    # Annotate ONLY this specific violation on the frame
                    b64_frame = annotate_video_frame_base64(frame, [loc_item])
                    if b64_frame:
                        violation["frame_b64"] = b64_frame
                        
                logger.info("Extracted and annotated individual video frames for violations.")
                
                # Attempt to generate the full annotated video with bounding boxes
                annotated_video_filename = f"annotated_{uuid.uuid4().hex[:12]}.mp4"
                output_path = str(VIDEO_OUTPUT_DIR / annotated_video_filename)
                
                try:
                    # Resolve local media path for processing
                    if media_path_or_url.startswith("http"):
                        local_temp = str(VIDEO_OUTPUT_DIR / f"temp_{uuid.uuid4().hex[:12]}.mp4")
                        urllib.request.urlretrieve(media_path_or_url, local_temp)
                    else:
                        local_temp = media_path_or_url
                    
                    # Generate the annotated video frame-by-frame
                    res_path = annotate_full_video(
                        video_path=local_temp,
                        violations=violations,
                        output_path=output_path,
                        overlay_duration=3.0,
                    )
                    
                    if media_path_or_url.startswith("http") and os.path.exists(local_temp):
                        try:
                            os.remove(local_temp)
                        except Exception:
                            pass
                    
                    if res_path:
                        temp_ann = output_path.replace(".mp4", "_temp.mp4")
                        if os.path.exists(temp_ann):
                            # Try to compile/convert with ffmpeg to standard browser-compatible h264
                            import subprocess
                            try:
                                subprocess.run([
                                    "ffmpeg", "-y", "-i", temp_ann,
                                    "-vcodec", "libx264", "-acodec", "aac",
                                    "-pix_fmt", "yuv420p", output_path
                                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                                try:
                                    os.remove(temp_ann)
                                except Exception:
                                    pass
                                logger.info("Video processed and encoded with h264 successfully.")
                            except Exception:
                                logger.warning("FFmpeg conversion failed, falling back to raw output.")
                                if os.path.exists(output_path):
                                    try:
                                        os.remove(output_path)
                                    except Exception:
                                        pass
                                os.rename(temp_ann, output_path)
                        logger.info("Full annotated video generated successfully.")
                    else:
                        logger.warning("Full video annotation failed, falling back to original video.")
                        _, orig_vid = _get_original_media(media_path_or_url, True)
                        annotated_video_filename = orig_vid
                except Exception as inner_err:
                    logger.exception("Failed to run full video annotation: %s", inner_err)
                    _, orig_vid = _get_original_media(media_path_or_url, True)
                    annotated_video_filename = orig_vid
                    
            except Exception as e:
                logger.exception("Failed to annotate video frames: %s", e)
                annotated_video_filename = None
        else:
            # Image: annotate the single image
            try:
                annotated_image = annotate_image_base64(media_path_or_url, localized)
            except Exception:
                logger.exception("Failed to annotate image")
    else:
        logger.info("No violations detected; returning original media.")
        annotated_image, annotated_video_filename = _get_original_media(media_path_or_url, is_video)


    summary = _build_summary(all_verdicts)
    primary_category = observed_items[0]["category"]
    primary_db_category = observed_items[0]["db_category"]
    categories = sorted({item["category"] for item in observed_items})
    db_categories = sorted({item["db_category"] for item in observed_items})

    # ── Heatmap (image only — video heatmap not supported yet) ────────────
    heatmap_image: str | None = None
    if violations and not is_video:
        try:
            heatmap_image = generate_violation_heatmap(media_path_or_url, localized)
            if heatmap_image:
                logger.info("Violation heatmap generated successfully.")
        except Exception:
            logger.exception("Failed to generate violation heatmap")

    # ── AI Recommendations ────────────────────────────────────────────────
    ai_recommendations: list[dict] = []
    if violations:
        try:
            ai_recommendations = generate_ai_recommendations(violations)
            logger.info("AI recommendations generated: %d items", len(ai_recommendations))
        except Exception:
            logger.exception("Failed to generate AI recommendations")

    # Build verdict list for HTML report — includes frame_b64 so cards can show screenshots
    _verdicts_for_report = [
        {**{
            "sbc_reference":  v.get("sbc_reference"),
            "category":       v.get("category"),
            "rule_text":      v.get("rule_text"),
            "cv_target":      v.get("cv_target"),
            "priority":       v.get("priority"),  # kept for CSS severity class lookup
            "verdict":        v.get("verdict"),
            "evidence":       v.get("evidence"),
            "confidence":     v.get("confidence"),
            "confidence_pct": int(round(float(v["confidence"]) * 100)) if v.get("confidence") is not None else None,
            "source_text":    v.get("source_text"),
            "bbox":           v.get("bbox"),
            "timestamp_sec":  v.get("timestamp_sec"),
            "frame_b64":      v.get("frame_b64"),  # used only for HTML report rendering
            "remediation":    v.get("remediation"),
        }}
        for v in all_verdicts if v.get("verdict") == "VIOLATION"
    ]

    api_verdicts = [
        {
            "sbc_reference":  v.get("sbc_reference"),
            "category":       v.get("category"),
            "sub_category":   v.get("sub_category"),
            "rule_text":      v.get("rule_text"),
            "cv_target":      v.get("cv_target"),
            "detection_type": v.get("detection_type"),
            "risk_level":     v.get("priority"),
            "verdict":        v.get("verdict"),
            "evidence":       v.get("evidence"),
            "confidence":     v.get("confidence"),
            "confidence_pct": int(round(float(v["confidence"]) * 100)) if v.get("confidence") is not None else None,
            "source_item_id": v.get("source_item_id"),
            "source_text":    v.get("source_text"),
            "bbox":           v.get("bbox"),
            "timestamp_sec":  v.get("timestamp_sec"),
        }
        for v in all_verdicts if v.get("verdict") == "VIOLATION"
    ]

    # Add formatted timestamps for videos/images
    def format_ts_hms(seconds: float | None) -> str | None:
        if seconds is None:
            return None
        h = int(seconds) // 3600
        m = (int(seconds) % 3600) // 60
        s = int(seconds) % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    for v in _verdicts_for_report:
        if v.get("timestamp_sec") is not None:
            v["timestamp_formatted"] = format_ts_hms(float(v["timestamp_sec"]))
            v["timestamp_range"] = v["timestamp_formatted"]

    for v in api_verdicts:
        if v.get("timestamp_sec") is not None:
            v["timestamp_formatted"] = format_ts_hms(float(v["timestamp_sec"]))
            v["timestamp_range"] = v["timestamp_formatted"]

    if is_video:
        _verdicts_for_report = _merge_consecutive_violations(_verdicts_for_report)
        api_verdicts = _merge_consecutive_violations(api_verdicts)
        summary["violations"] = len(api_verdicts)
        total_decided = summary["compliant"] + summary["violations"]
        summary["compliance_rate"] = round(summary["compliant"] / total_decided * 100, 1) if total_decided else 100.0

    result = {
        "classification": primary_category,
        "classifications": categories,
        "db_category": primary_db_category,
        "db_categories": db_categories,
        "verdicts": api_verdicts,
        "annotated_image": annotated_image,
        "annotated_media": (
            {"type": "image", "data_url": annotated_image}
            if annotated_image
            else {
                "type": "video",
                "download_url": f"/api/v1/download/{annotated_video_filename}" if annotated_video_filename else None,
                "filename": annotated_video_filename,
            }
            if is_video
            else None
        ),
        "summary": summary,
        "project_name": project_name,
        "contractor_name": contractor_name,
        "heatmap_image": heatmap_image,
        "ai_recommendations": ai_recommendations,
        "token_usage": {
            "observe": observe_tokens,
            "judge": judge_tokens_total,
        },
        "_report_verdicts": _verdicts_for_report,  # internal: used by HTML report generator only
    }

    logger.info(
        "Detection complete | violations=%d | compliant=%d | uncertain=%d",
        summary["violations"],
        summary["compliant"],
        summary["uncertain"],
    )

    # Store in cache so the same image returns consistent results instantly
    if cache_key is not None:
        _RESULT_CACHE[cache_key] = result
        logger.info("Result cached under key %s", cache_key[:12])

    return result


def detect_violations_batch(
    media_paths_or_urls: list[str],
    is_videos: list[bool] | None = None,
    custom_prompt: str = "",
    top_k: int | None = None,
    classification: str | None = None,
    project_name: str = "Building Inspection",
    contractor_name: str = "",
) -> dict[str, Any]:
    """
    Process a list of image/video paths or URLs in parallel, then aggregate and consolidate results.
    """
    logger.info(
        "Starting batch violation detection | count=%d | project=%s | contractor=%s",
        len(media_paths_or_urls),
        project_name,
        contractor_name,
    )

    if not media_paths_or_urls:
        return {
            "classification": "General",
            "classifications": [],
            "db_category": "General",
            "db_categories": [],
            "verdicts": [],
            "summary": {
                "total_rules_evaluated": 0,
                "violations": 0,
                "compliant": 0,
                "uncertain": 0,
                "compliance_rate": 100.0,
            },
            "project_name": project_name,
            "contractor_name": contractor_name,
            "annotated_media_list": [],
            "ai_recommendations": [],
            "token_usage": {
                "observe": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "judge": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            },
        }

    # Prepare flags
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if is_videos is None:
        is_videos = []
        for path in media_paths_or_urls:
            ext = Path(path).suffix.lower() if not path.startswith("http") else ""
            is_videos.append(ext in video_extensions)
    elif len(is_videos) < len(media_paths_or_urls):
        # Pad with False or detect
        for idx in range(len(is_videos), len(media_paths_or_urls)):
            path = media_paths_or_urls[idx]
            ext = Path(path).suffix.lower() if not path.startswith("http") else ""
            is_videos.append(ext in video_extensions)

    # We import this internally to avoid circular dependencies
    from Voilation_Template.report_generator import _crop_image_b64

    # Run in parallel using ThreadPoolExecutor
    results: list[dict[str, Any] | None] = [None] * len(media_paths_or_urls)

    def _run_single(idx: int):
        path = media_paths_or_urls[idx]
        is_vid = is_videos[idx]
        try:
            res = detect_violations(
                media_path_or_url=path,
                is_video=is_vid,
                custom_prompt=custom_prompt,
                top_k=top_k,
                classification=classification,
                project_name=project_name,
                contractor_name=contractor_name,
            )
            results[idx] = res
        except Exception:
            logger.exception("Failed to run single detection in batch for media: %s", path)

    max_workers = min(len(media_paths_or_urls), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks and track them
        futures = [executor.submit(_run_single, i) for i in range(len(media_paths_or_urls))]
        for future in futures:
            future.result()  # blocks until complete, raises thread exception if any

    # Aggregation
    combined_verdicts = []
    combined_report_verdicts = []
    combined_classifications = set()
    combined_db_categories = set()
    
    total_rules_evaluated = 0
    total_violations = 0
    total_compliant = 0
    total_uncertain = 0

    token_usage_observe = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    token_usage_judge = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    annotated_media_list = []
    all_recommendations_raw = []

    for idx, res in enumerate(results):
        if not res:
            continue
        
        path = media_paths_or_urls[idx]
        media_name = Path(path).name if not path.startswith("http") else path
        is_vid = is_videos[idx]
        
        # 1. Categories
        if res.get("classification"):
            combined_classifications.add(res["classification"])
        if res.get("classifications"):
            combined_classifications.update(res["classifications"])
        if res.get("db_category"):
            combined_db_categories.add(res["db_category"])
        if res.get("db_categories"):
            combined_db_categories.update(res["db_categories"])

        # 2. Summary
        sum_data = res.get("summary", {})
        total_rules_evaluated += sum_data.get("total_rules_evaluated", 0)
        total_violations += sum_data.get("violations", 0)
        total_compliant += sum_data.get("compliant", 0)
        total_uncertain += sum_data.get("uncertain", 0)

        # 3. Tokens
        t_usage = res.get("token_usage", {})
        for key in ["input_tokens", "output_tokens", "total_tokens"]:
            token_usage_observe[key] += t_usage.get("observe", {}).get(key, 0)
            token_usage_judge[key] += t_usage.get("judge", {}).get(key, 0)

        # 4. Recommendations
        if res.get("ai_recommendations"):
            all_recommendations_raw.extend(res["ai_recommendations"])

        # 5. Media list item
        annotated_media_list.append({
            "media_index": idx,
            "media_name": media_name,
            "is_video": is_vid,
            "annotated_image": res.get("annotated_image"),
            "annotated_media": res.get("annotated_media"),
            "heatmap_image": res.get("heatmap_image"),
            "summary": sum_data,
        })

        # 6. Verdicts & Report Verdicts
        item_verdicts = res.get("verdicts", [])
        report_verdicts = res.get("_report_verdicts", [])

        # For report verdicts, if they are from static images, we can pre-crop the region using _crop_image_b64
        # and set it to frame_b64, so that each card in the combined HTML report shows its specific cropped evidence!
        annotated_image = res.get("annotated_image")
        if annotated_image and not is_vid:
            for v in report_verdicts:
                if not v.get("frame_b64") and v.get("bbox"):
                    try:
                        cropped = _crop_image_b64(annotated_image, v["bbox"])
                        if cropped:
                            v["frame_b64"] = cropped
                    except Exception:
                        pass

        # Tag with media source context
        for v in item_verdicts:
            v["media_name"] = media_name
            v["media_index"] = idx
            v["media_type"] = "video" if is_vid else "image"
        for v in report_verdicts:
            v["media_name"] = media_name
            v["media_index"] = idx
            v["media_type"] = "video" if is_vid else "image"

        combined_verdicts.extend(item_verdicts)
        combined_report_verdicts.extend(report_verdicts)

    # Unique Recommendations (Deduplicated based on sbc_reference)
    seen_refs = set()
    dedup_recommendations = []
    for rec in all_recommendations_raw:
        ref = rec.get("sbc_reference")
        if ref not in seen_refs:
            seen_refs.add(ref)
            dedup_recommendations.append(rec)

    # Compliance Rate
    total_decided = total_compliant + total_violations
    compliance_rate = (total_compliant / total_decided * 100.0) if total_decided > 0 else 100.0

    combined_summary = {
        "total_rules_evaluated": total_rules_evaluated,
        "violations": total_violations,
        "compliant": total_compliant,
        "uncertain": total_uncertain,
        "compliance_rate": compliance_rate,
    }

    # Set up some standard default values from first result for backward compatibility
    primary_category = results[0].get("classification", "General") if (results and results[0]) else "General"
    primary_db_category = results[0].get("db_category", "General") if (results and results[0]) else "General"
    first_annotated_image = results[0].get("annotated_image") if (results and results[0]) else None
    first_annotated_media = results[0].get("annotated_media") if (results and results[0]) else None
    first_heatmap = results[0].get("heatmap_image") if (results and results[0]) else None

    # Consolidated Output dict
    batch_result = {
        "classification": primary_category,
        "classifications": sorted(list(combined_classifications)),
        "db_category": primary_db_category,
        "db_categories": sorted(list(combined_db_categories)),
        "verdicts": combined_verdicts,
        "annotated_image": first_annotated_image,
        "annotated_media": first_annotated_media,
        "heatmap_image": first_heatmap,
        "annotated_media_list": annotated_media_list,
        "summary": combined_summary,
        "project_name": project_name,
        "contractor_name": contractor_name,
        "ai_recommendations": dedup_recommendations,
        "token_usage": {
            "observe": token_usage_observe,
            "judge": token_usage_judge,
        },
        "_report_verdicts": combined_report_verdicts,
    }

    return batch_result

