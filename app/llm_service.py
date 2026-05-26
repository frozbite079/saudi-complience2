from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import requests

from app.config import (
    GLM_API_KEY,
    GLM_BASE_URL,
    GLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS_OBSERVE,
    LLM_MAX_TOKENS_JUDGE,
)

logger = logging.getLogger(__name__)

_CHAT_URL = f"{GLM_BASE_URL}/chat/completions"

_IMAGE_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _encode_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_image_content(file_path: str) -> dict:
    ext = Path(file_path).suffix.lower()
    mime = _IMAGE_MIME.get(ext, "image/jpeg")
    b64 = _encode_file(file_path)
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _build_video_content(file_path: str) -> dict:
    ext = Path(file_path).suffix.lower()
    b64 = _encode_file(file_path)
    if ext in _VIDEO_EXT:
        mime = f"video/{ext.lstrip('.')}"
        return {
            "type": "video_url",
            "video_url": {"url": f"data:{mime};base64,{b64}"},
        }
    return {"type": "video_url", "video_url": {"url": b64}}


def _encode_image_bytes(data: bytes, mime: str = "image/jpeg") -> dict:
    b64 = base64.b64encode(data).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def extract_scene_change_keyframes(
    video_path: str,
    max_frames: int = 6,
) -> list[dict[str, Any]]:
    """
    Intelligent scene-change detection keyframe extractor.
    Computes absolute pixel differences between sequential frames to extract
    significant visual moments / keyframes.
    
    Returns a list of dicts:
    [
      {
        "frame_index": int,
        "timestamp_sec": float,
        "content": dict,  # standard image_url base64 dict for LLM payload
        "frame": np.ndarray  # raw frame image
      }
    ]
    """
    import urllib.request
    import uuid
    import numpy as np
    import os
    from app.config import VIDEO_OUTPUT_DIR

    # Download remote videos first
    local_path = video_path
    is_temp = False
    if video_path.startswith("http"):
        local_path = str(VIDEO_OUTPUT_DIR / f"temp_kf_{uuid.uuid4().hex[:12]}.mp4")
        logger.info("Downloading remote video for keyframe extraction: %s -> %s", video_path, local_path)
        urllib.request.urlretrieve(video_path, local_path)
        is_temp = True

    try:
        cap = cv2.VideoCapture(local_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {local_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video has no readable frames: {local_path}")

        # Determine sampling rate. To be ultra fast, sample at 5 frames per second
        sample_step = max(1, int(round(fps / 5.0)))
        
        sampled_frames = []
        prev_gray = None
        
        # Read frames, resize and convert to grayscale to compute absolute differences rapidly
        for i in range(0, total_frames, sample_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            
            # Grayscale, resize to small resolution to keep calculations microsecond-fast
            small_gray = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
            # Subtle Gaussian blur to eliminate compression noise
            small_gray_blur = cv2.GaussianBlur(small_gray, (5, 5), 0)
            
            diff_score = 0.0
            if prev_gray is not None:
                # Compute absolute difference between sequential sampled frames
                diff = cv2.absdiff(small_gray_blur, prev_gray)
                diff_score = float(np.mean(diff))
                
            prev_gray = small_gray_blur
            sampled_frames.append({
                "index": i,
                "timestamp_sec": i / fps,
                "score": diff_score,
                "frame": frame
            })
            
        cap.release()
        
        if not sampled_frames:
            raise ValueError("No frames could be sampled from the video")
            
        # Detect peaks in diff scores
        peaks = []
        n_sampled = len(sampled_frames)
        for i in range(n_sampled):
            score = sampled_frames[i]["score"]
            left_score = sampled_frames[i-1]["score"] if i > 0 else 0.0
            right_score = sampled_frames[i+1]["score"] if i < n_sampled - 1 else 0.0
            
            # Peak condition: higher than neighbors and above a minimal change threshold
            if score > left_score and score > right_score and score >= 2.0:
                peaks.append(sampled_frames[i])
                
        # Sort peaks by score descending
        peaks.sort(key=lambda x: x["score"], reverse=True)
        
        selected_keyframes = []
        
        # Always prioritize/include the first frame for baseline context
        selected_keyframes.append(sampled_frames[0])
        
        # Filter other peaks ensuring a minimum temporal distance of 1.5s
        min_dist_sec = 1.5
        for peak in peaks:
            if len(selected_keyframes) >= max_frames:
                break
            too_close = any(abs(peak["timestamp_sec"] - k["timestamp_sec"]) < min_dist_sec for k in selected_keyframes)
            if not too_close:
                selected_keyframes.append(peak)
                
        # Space out remaining frames if we have extra slots
        while len(selected_keyframes) < max_frames and len(sampled_frames) > len(selected_keyframes):
            selected_keyframes.sort(key=lambda x: x["timestamp_sec"])
            max_gap = 0.0
            insert_idx = -1
            
            for j in range(len(selected_keyframes) - 1):
                gap = selected_keyframes[j+1]["timestamp_sec"] - selected_keyframes[j]["timestamp_sec"]
                if gap > max_gap:
                    max_gap = gap
                    insert_idx = j
                    
            last_ts = selected_keyframes[-1]["timestamp_sec"]
            video_duration = sampled_frames[-1]["timestamp_sec"]
            if (video_duration - last_ts) > max_gap:
                max_gap = video_duration - last_ts
                insert_idx = len(selected_keyframes) - 1
                
            if max_gap <= min_dist_sec:
                remaining = [f for f in sampled_frames if f["index"] not in {k["index"] for k in selected_keyframes}]
                if not remaining:
                    break
                remaining.sort(key=lambda x: x["score"], reverse=True)
                selected_keyframes.append(remaining[0])
                continue
                
            if insert_idx == len(selected_keyframes) - 1:
                mid_ts = (last_ts + video_duration) / 2.0
            else:
                mid_ts = (selected_keyframes[insert_idx]["timestamp_sec"] + selected_keyframes[insert_idx+1]["timestamp_sec"]) / 2.0
                
            best_frame_to_insert = min(sampled_frames, key=lambda x: abs(x["timestamp_sec"] - mid_ts))
            if best_frame_to_insert["index"] in {k["index"] for k in selected_keyframes}:
                break
            selected_keyframes.append(best_frame_to_insert)
            
        selected_keyframes.sort(key=lambda x: x["timestamp_sec"])
        
        final_keyframes = []
        for k in selected_keyframes:
            ok, encoded = cv2.imencode(".jpg", k["frame"])
            if not ok:
                continue
            final_keyframes.append({
                "frame_index": k["index"],
                "timestamp_sec": k["timestamp_sec"],
                "content": _encode_image_bytes(encoded.tobytes()),
                "frame": k["frame"]
            })
            
        return final_keyframes
        
    finally:
        if is_temp and os.path.exists(local_path):
            try:
                os.remove(local_path)
            except Exception:
                pass


def _extract_video_frame_contents(
    file_path: str,
    num_frames: int = 4,
) -> list[dict[str, Any]]:
    keyframes = extract_scene_change_keyframes(file_path, max_frames=num_frames)
    return [k["content"] for k in keyframes]


def _build_url_content(url: str, is_video: bool) -> dict:
    key = "video_url" if is_video else "image_url"
    return {"type": key, key: {"url": url}}


def _call_llm(
    system_prompt: str,
    user_content: list[dict[str, Any]],
    max_tokens: int,
    temperature: float | None = None,
    _retries: int = 3,
    timeout: int = 120,
) -> tuple[str, dict[str, int]]:
    headers = {
        "Authorization": f"Bearer {GLM_API_KEY}",
        "Content-Type": "application/json",
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    payload = {
        "model": GLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature or LLM_TEMPERATURE,
    }

    last_error: Exception | None = None
    for attempt in range(1, _retries + 1):
        try:
            resp = requests.post(_CHAT_URL, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            text = data["choices"][0]["message"]["content"]

            if not text or not text.strip():
                logger.warning(
                    "GLM returned empty content on attempt %d/%d. "
                    "finish_reason=%s | usage=%s",
                    attempt,
                    _retries,
                    data["choices"][0].get("finish_reason"),
                    data.get("usage"),
                )
                if attempt < _retries:
                    continue  # retry

            usage = data.get("usage", {})
            token_stats = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            return text, token_stats

        except Exception as e:
            logger.warning("LLM call failed on attempt %d/%d: %s", attempt, _retries, e)
            last_error = e

    # All retries exhausted — return empty so callers degrade gracefully
    logger.error("LLM call failed after %d attempts. Last error: %s", _retries, last_error)
    return "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}



OBSERVE_SYSTEM_PROMPT = (
    "You are a specialized visual observer for Saudi Building Code (SBC) compliance inspections. "
    "Your ONLY job in this step is to OBSERVE and DESCRIBE what is visible — do NOT judge compliance.\n\n"
    "You must identify and list:\n"
    "1. EQUIPMENT: Electrical panels, transformers, switchgear, generators, conduit, wiring, junction boxes\n"
    "2. SIGNAGE & LABELS: Warning signs, equipment labels, room identification signs, voltage markings\n"
    "3. SPACING & CLEARANCES: Working space around panels, access paths, clearance from walls\n"
    "4. SAFETY DEVICES: Circuit breakers, disconnects, grounding, SPDs, RCDs, isolation switches\n"
    "5. PPE: Helmets, vests, gloves, safety footwear\n"
    "6. ENVIRONMENT: Room type, floor condition, lighting, ventilation, fire suppression\n"
    "7. WIRING & CONDUIT: Cable routing, conduit condition, cable tray organization\n\n"
    "Output a structured observation report. Be specific about what you SEE and what you CANNOT see."
)

OBSERVE_USER_TEMPLATE = (
    "Inspect this {media_type} carefully. Provide a detailed, structured observation of all "
    "safety-related elements visible. Focus on equipment, signage, spacing, wiring, and safety devices. "
    "If any element is partially visible or unclear, state that explicitly.\n\n"
    "{custom_prompt}"
)

JUDGE_SYSTEM_PROMPT = (
    "You are a Saudi Building Code (SBC) compliance judge. You will receive:\n"
    "1. A structured VISUAL OBSERVATION of a scene\n"
    "2. A list of SBC RULES with their CV TARGETS (visual indicators)\n\n"
    "For EACH rule you must determine:\n"
    "- VIOLATION: The CV Target is clearly missing or incorrect in the observed scene\n"
    "- COMPLIANT: The CV Target is clearly present and correct in the observed scene\n"
    "- UNCERTAIN: Cannot determine from the available visual evidence\n\n"
    "You MUST respond in the following JSON format (no other text):\n"
    "[\n"
    "  {{\n"
    '    "sbc_reference": "the reference code",\n'
    '    "category": "rule category",\n'
    '    "sub_category": "rule sub-category",\n'
    '    "rule_text": "the full rule text",\n'
    '    "cv_target": "what should be visible",\n'
    '    "detection_type": "PRESENCE|LABEL|SPACING|CONDITION",\n'
    '    "priority": "CRITICAL|HIGH|MEDIUM|LOW",\n'
    '    "verdict": "VIOLATION|COMPLIANT|UNCERTAIN",\n'
    '    "evidence": "specific visual evidence supporting this verdict",\n'
    '    "confidence": 0.0-1.0\n'
    "  }}\n"
    "]\n\n"
    "Be strict: only mark COMPLIANT if the CV Target is clearly visible. "
    "Mark VIOLATION only if the scene clearly shows the CV Target is absent or violated. "
    "Mark UNCERTAIN if the view is obstructed, the area is not visible, or evidence is ambiguous."
)

CLASSIFY_SYSTEM_PROMPT = (
    "You classify construction/compliance inspection media into exactly one rule category. "
    "Return only one category name, with no explanation."
)

CLASSIFY_USER_PROMPT = (
    "Look at this {media_type} and choose the single best category for rule retrieval.\n\n"
    "Allowed categories:\n"
    "- Structural Safety: structure, rebar, concrete, slabs, columns, beams, excavation, scaffolding, PPE/site hazards\n"
    "- Electrical: panels, switchgear, wiring, conduit, transformers, generators, electrical rooms, voltage labels\n"
    "- Plumbing: pipes, valves, drains, pumps, water/sanitary fixtures, sewer systems\n"
    "- Fire Safety: sprinklers, alarms, extinguishers, fire doors, exit signs, emergency lighting\n\n"
    "Return exactly one of: Structural Safety, Electrical, Plumbing, Fire Safety."
)

LOCALIZE_SYSTEM_PROMPT = (
    "You localize visible compliance violations in inspection images. "
    "Return only valid JSON, with no markdown and no explanation."
)

OBSERVE_ITEMS_SYSTEM_PROMPT = (
    "You are a visual inspection extractor. Return only valid JSON, with no markdown. "
    "Extract visible inspection items dynamically from the media."
)

OBSERVE_ITEMS_USER_PROMPT = (
    "Inspect this {media_type} and return a JSON array. Each item must contain only:\n"
    "- category: one of Structural Safety, Electrical, Plumbing, Fire Safety\n"
    "- text: concise observation text for that visible item, including visible missing/unsafe indicators if clear\n"
    "- bbox: [x1,y1,x2,y2] normalized from 0 to 1000 around the visible item/evidence, or null if not localizable\n\n"
    "Rules:\n"
    "- Category must be based on what is actually visible.\n"
    "- Use Electrical for panels, switchgear, wiring, conduit, transformers, generators, electrical rooms, voltage labels.\n"
    "- Use Structural Safety for rebar, slabs, columns, beams, excavation, scaffolding, formwork, structural site hazards.\n"
    "- Use Plumbing for pipes, valves, drains, pumps, water/sanitary systems.\n"
    "- Use Fire Safety for sprinklers, alarms, extinguishers, fire doors, exit signs, emergency lighting.\n"
    "- Return 1 to 8 high-signal items only.\n"
    "- Do not include explanations outside JSON.\n\n"
    "Example:\n"
    "[{{\"category\":\"Electrical\",\"text\":\"Open electrical panel with visible wiring and circuit breakers. Warning signage is not visible.\",\"bbox\":[170,120,650,760]}}]\n\n"
    "{custom_prompt}"
)

OBSERVE_VIDEO_ITEMS_USER_PROMPT = (
    "Watch this entire video carefully and return a JSON array of potential safety/compliance items you observe.\n"
    "Each item must contain:\n"
    "- category: one of Structural Safety, Electrical, Plumbing, Fire Safety\n"
    "- text: concise observation text for that visible item\n"
    "- timestamp_sec: the approximate time in seconds when this item is most clearly visible\n"
    "- bbox: [x1,y1,x2,y2] normalized from 0 to 1000 around the item at that timestamp, or null if not localizable\n\n"
    "Rules:\n"
    "- Category must be based on what is actually visible.\n"
    "- Use Electrical for panels, switchgear, wiring, conduit, transformers, generators, electrical rooms, voltage labels.\n"
    "- Use Structural Safety for rebar, slabs, columns, beams, excavation, scaffolding, formwork, structural site hazards.\n"
    "- Use Plumbing for pipes, valves, drains, pumps, water/sanitary systems.\n"
    "- Use Fire Safety for sprinklers, alarms, extinguishers, fire doors, exit signs, emergency lighting.\n"
    "- Return 1 to 12 high-signal items from throughout the video.\n"
    "- Do not include explanations outside JSON.\n\n"
    "Example:\n"
    "[{{\"category\":\"Electrical\",\"text\":\"Open electrical panel with visible wiring. Warning signage absent.\",\"timestamp_sec\":12.5,\"bbox\":[170,120,650,760]}}]\n\n"
    "{custom_prompt}"
)

OBSERVE_VIDEO_KEYFRAMES_USER_PROMPT = (
    "You are inspecting a set of {num_keyframes} keyframes extracted from a video.\n"
    "The frames are ordered chronologically and correspond to the following exact timestamps:\n"
    "{timeline_text}\n\n"
    "Examine these keyframes carefully and return a JSON array of potential safety/compliance items you observe.\n"
    "Each item in the JSON array must contain:\n"
    "- category: one of Structural Safety, Electrical, Plumbing, Fire Safety\n"
    "- text: concise observation text for that visible item, including visible missing/unsafe indicators if clear\n"
    "- timestamp_sec: the EXACT timestamp from the timeline above corresponding to the frame where the item is observed (choose exactly from: {allowed_timestamps_list})\n"
    "- bbox: [x1,y1,x2,y2] normalized from 0 to 1000 around the item on that specific keyframe, or null if not localizable\n\n"
    "Rules:\n"
    "- Category must be based on what is actually visible.\n"
    "- Use Electrical for panels, switchgear, wiring, conduit, transformers, generators, electrical rooms, voltage labels.\n"
    "- Use Structural Safety for rebar, slabs, columns, beams, excavation, scaffolding, formwork, structural site hazards.\n"
    "- Use Plumbing for pipes, valves, drains, pumps, water/sanitary systems.\n"
    "- Use Fire Safety for sprinklers, alarms, extinguishers, fire doors, exit signs, emergency lighting.\n"
    "- Return 1 to 12 high-signal items from throughout the keyframes.\n"
    "- Do not include explanations outside JSON.\n\n"
    "Example:\n"
    "[{{\"category\":\"Electrical\",\"text\":\"Open electrical panel with visible wiring. Warning signage absent.\",\"timestamp_sec\":12.5,\"bbox\":[170,120,650,760]}}]\n\n"
    "{custom_prompt}"
)


def build_user_content(
    media_path_or_url: str,
    is_video: bool,
    custom_prompt: str = "",
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []

    if media_path_or_url.startswith("http"):
        content.append(_build_url_content(media_path_or_url, is_video))
    elif is_video:
        content.extend(_extract_video_frame_contents(media_path_or_url))
    else:
        content.append(_build_image_content(media_path_or_url))

    media_type = "video frames" if is_video else "image"
    text = OBSERVE_USER_TEMPLATE.format(
        media_type=media_type,
        custom_prompt=custom_prompt,
    )
    content.append({"type": "text", "text": text})
    return content


def observe(
    media_path_or_url: str,
    is_video: bool,
    custom_prompt: str = "",
) -> tuple[str, dict[str, int]]:
    content = build_user_content(media_path_or_url, is_video, custom_prompt)
    return _call_llm(OBSERVE_SYSTEM_PROMPT, content, LLM_MAX_TOKENS_OBSERVE)


def observe_items(
    media_path_or_url: str,
    is_video: bool,
    custom_prompt: str = "",
) -> tuple[str, dict[str, int]]:
    content: list[dict[str, Any]] = []

    if is_video:
        from app.config import VIDEO_FRAMES_COUNT
        keyframes = extract_scene_change_keyframes(media_path_or_url, max_frames=VIDEO_FRAMES_COUNT)
        for kf in keyframes:
            content.append(kf["content"])
            
        timeline_items = []
        allowed_ts = []
        for i, kf in enumerate(keyframes):
            t_val = kf["timestamp_sec"]
            timeline_items.append(f"- Frame {i+1}: t = {t_val:.2f}s")
            allowed_ts.append(f"{t_val:.2f}")
            
        timeline_text = "\n".join(timeline_items)
        allowed_timestamps_list = ", ".join(allowed_ts)
        
        prompt_text = OBSERVE_VIDEO_KEYFRAMES_USER_PROMPT.format(
            num_keyframes=len(keyframes),
            timeline_text=timeline_text,
            allowed_timestamps_list=allowed_timestamps_list,
            custom_prompt=custom_prompt
        )
    else:
        if media_path_or_url.startswith("http"):
            content.append(_build_url_content(media_path_or_url, is_video=False))
        else:
            content.append(_build_image_content(media_path_or_url))
            
        prompt_text = OBSERVE_ITEMS_USER_PROMPT.format(
            media_type="image",
            custom_prompt=custom_prompt
        )

    content.append({"type": "text", "text": prompt_text})

    return _call_llm(
        OBSERVE_ITEMS_SYSTEM_PROMPT,
        content,
        LLM_MAX_TOKENS_OBSERVE,
        temperature=0.0,
        timeout=300 if is_video else 120,
    )


def classify_scene(
    media_path_or_url: str,
    is_video: bool,
) -> tuple[str, dict[str, int]]:
    content: list[dict[str, Any]] = []

    if is_video:
        # Extract a few keyframes to make classification fast and accurate
        keyframes = extract_scene_change_keyframes(media_path_or_url, max_frames=3)
        for kf in keyframes:
            content.append(kf["content"])
    else:
        if media_path_or_url.startswith("http"):
            content.append(_build_url_content(media_path_or_url, is_video=False))
        else:
            content.append(_build_image_content(media_path_or_url))

    media_type = "video" if is_video else "image"
    content.append(
        {
            "type": "text",
            "text": CLASSIFY_USER_PROMPT.format(media_type=media_type),
        }
    )

    return _call_llm(CLASSIFY_SYSTEM_PROMPT, content, max_tokens=32, temperature=0.0)


def classify_observation_text(observation: str) -> tuple[str, dict[str, int]]:
    user_content = [
        {
            "type": "text",
            "text": (
                "Classify this visual observation into the single best rule category for vector DB retrieval.\n\n"
                "Allowed categories:\n"
                "- Structural Safety\n"
                "- Electrical\n"
                "- Plumbing\n"
                "- Fire Safety\n\n"
                "Important: classify by what is actually visible/present, not by checklist items marked "
                "'none visible', 'not visible', 'absent', or 'cannot be assessed'.\n\n"
                "Return exactly one category name only.\n\n"
                f"Observation:\n{observation}"
            ),
        }
    ]
    return _call_llm(CLASSIFY_SYSTEM_PROMPT, user_content, max_tokens=32, temperature=0.0)


def localize_violations(
    media_path_or_url: str,
    is_video: bool,
    violations: list[dict],
) -> tuple[str, dict[str, int]]:
    if is_video or not violations:
        return "[]", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    content: list[dict[str, Any]] = []
    if media_path_or_url.startswith("http"):
        content.append(_build_url_content(media_path_or_url, is_video=False))
    else:
        content.append(_build_image_content(media_path_or_url))

    rules_json = json.dumps(violations, ensure_ascii=False)
    content.append(
        {
            "type": "text",
            "text": (
                "You are annotating a compliance inspection image. For EACH violation listed below, "
                "you MUST provide a bounding box [x1, y1, x2, y2] using normalized coordinates 0–1000.\n\n"
                "RULES — READ CAREFULLY:\n"
                "1. bbox is MANDATORY for every single violation. Never use null.\n"
                "2. For violations where something is PRESENT but incorrect (exposed wire, unlabeled panel): "
                "draw the box tightly around that visible element.\n"
                "3. For violations where something is MISSING (absent sign, missing label, no marking): "
                "draw the box around the area where that item SHOULD be present "
                "(e.g. the panel door, the switch, the cable entry point).\n"
                "4. If the exact element is off-screen, draw a box around the most relevant visible "
                "related element in the image.\n"
                "5. Every box must be meaningful — do not use the full image dimensions.\n\n"
                "Return a JSON array only. No markdown, no explanation.\n\n"
                "Required format for each item:\n"
                "{"
                '"sbc_reference":"...", '
                '"priority":"CRITICAL|HIGH|MEDIUM|LOW", '
                '"label":"short visual label (max 6 words)", '
                '"bbox":[x1,y1,x2,y2]'
                "}\n\n"
                f"Violations to localize:\n{rules_json}"
            ),
        }
    )

    return _call_llm(LOCALIZE_SYSTEM_PROMPT, content, max_tokens=1024, temperature=0.0)


def judge(
    observation: str,
    rules: list[dict],
) -> tuple[str, dict[str, int]]:
    rules_text = ""
    for i, rule in enumerate(rules, 1):
        rules_text += (
            f"\n--- Rule #{i} ---\n"
            f"SBC Reference : {rule['sbc_reference']}\n"
            f"Category      : {rule['category']} > {rule['sub_category']}\n"
            f"Priority      : {rule['priority']}\n"
            f"Detection Type: {rule['detection_type']}\n"
            f"Rule Text     : {rule['rule_text']}\n"
            f"CV Target     : {rule['cv_target']}\n"
        )

    user_text = (
        f"## VISUAL OBSERVATION\n{observation}\n\n"
        f"## SBC RULES TO EVALUATE ({len(rules)} rules)\n{rules_text}\n\n"
        "Evaluate each rule against the observation. Respond ONLY with the JSON array."
    )

    user_content = [{"type": "text", "text": user_text}]
    return _call_llm(JUDGE_SYSTEM_PROMPT, user_content, LLM_MAX_TOKENS_JUDGE, temperature=0.1)


_RECOMMENDATIONS_SYSTEM_PROMPT = (
    "You are a Saudi Building Code (SBC) compliance expert. "
    "Given a list of detected violations, generate clear, actionable remediation recommendations. "
    "Return ONLY a valid JSON array with no markdown or explanation."
)

_RECOMMENDATIONS_USER_TEMPLATE = """\
The following violations were detected during a compliance inspection:

{violations_text}

For each violation, provide a specific, practical recommendation. Return a JSON array where each item contains:
{{
  "sbc_reference": "the SBC code reference",
  "recommendation": "specific action to fix this violation (2-3 sentences, practical and direct)",
  "urgency": "IMMEDIATE|SHORT_TERM|LONG_TERM",
  "estimated_effort": "LOW|MEDIUM|HIGH",
  "responsible_party": "e.g. Licensed Electrician, Site Engineer, Contractor"
}}

Prioritise by risk: CRITICAL violations must be listed first. Be concise and specific to Saudi Building Code standards.
"""


def generate_ai_recommendations(
    violations: list[dict],
) -> list[dict]:
    """
    Call the LLM to generate structured remediation recommendations for each violation.
    Returns a list of recommendation dicts, or an empty list on failure.
    """
    if not violations:
        return []

    violations_text = ""
    for i, v in enumerate(violations, 1):
        violations_text += (
            f"\n{i}. SBC {v.get('sbc_reference', 'N/A')} [{v.get('priority', v.get('risk_level', 'MEDIUM'))}] "
            f"— {v.get('cv_target', v.get('source_text', ''))[:120]}\n"
            f"   Evidence: {v.get('evidence', '')[:200]}\n"
        )

    user_text = _RECOMMENDATIONS_USER_TEMPLATE.format(violations_text=violations_text)
    user_content = [{"type": "text", "text": user_text}]

    try:
        raw, _ = _call_llm(
            _RECOMMENDATIONS_SYSTEM_PROMPT,
            user_content,
            max_tokens=2048,
            temperature=0.2,
        )
        # Parse JSON array
        import re
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception as exc:
        logger.warning("AI recommendations generation failed: %s", exc)

    return []
