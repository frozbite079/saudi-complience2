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

from app.annotation_service import annotate_full_video, annotate_image_base64
from app.config import RAG_TOP_K, VIDEO_OUTPUT_DIR
from app.embedding_service import embed_text
from app.llm_service import judge, observe_items, localize_violations
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


def detect_violations(
    media_path_or_url: str,
    is_video: bool = False,
    custom_prompt: str = "",
    top_k: int | None = None,
    classification: str | None = None,
) -> dict[str, Any]:
    logger.info(
        "Starting violation detection | media=%s | video=%s",
        media_path_or_url[:80],
        is_video,
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
                
                # Still provide a fallback video download url
                _, orig_vid = _get_original_media(media_path_or_url, True)
                annotated_video_filename = orig_vid
            except Exception:
                logger.exception("Failed to annotate video frames")
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

    result = {
        "classification": primary_category,
        "classifications": categories,
        "db_category": primary_db_category,
        "db_categories": db_categories,
        "verdicts": [v for v in all_verdicts if v.get("verdict") == "VIOLATION"],
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
        "token_usage": {
            "observe": observe_tokens,
            "judge": judge_tokens_total,
        },
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
