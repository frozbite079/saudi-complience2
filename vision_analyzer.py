"""
Vision Analyzer using LangChain + GLM-4.6V
==========================================
Supports: Image upload, Video upload (frame extraction), Token tracking
Model   : z-ai/glm-4.6v
Base URL: https://api.aicredits.in/v1
"""

import base64
import os
import sys
import cv2                  # pip install opencv-python
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import weaviate
import requests
import weaviate.classes as wvc

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
API_KEY  = "sk-live-4bc32310d454c2b90715d42da56e455226dd0441fe1139aba91769e2184a380f"
BASE_URL = "https://api.aicredits.in/v1"
MODEL    = "z-ai/glm-4.6v"

# ──────────────────────────────────────────────
# LLM Client
# ──────────────────────────────────────────────
llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.5,
    max_tokens=2048,
)


# ──────────────────────────────────────────────
# Helper: Encode any file to base64
# ──────────────────────────────────────────────
def file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ──────────────────────────────────────────────
# Helper: Extract frames from video
# ──────────────────────────────────────────────
def extract_video_frames(
    video_path: str,
    num_frames: int = 6,
    output_dir: Optional[str] = None,
) -> list[str]:
    """
    Extracts `num_frames` evenly-spaced frames from a video.
    Returns a list of image file paths.
    """
    output_dir = output_dir or os.path.join(
        os.path.dirname(video_path), "_video_frames"
    )
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / fps if fps > 0 else 0
    print(f"  📹 Video info: {total_frames} frames | {fps:.1f} FPS | {duration_sec:.1f}s")

    # Pick evenly-spaced frame indices
    indices = [
        int(i * total_frames / num_frames)
        for i in range(num_frames)
    ]

    saved_paths = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_path = os.path.join(output_dir, f"frame_{idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        saved_paths.append(frame_path)

    cap.release()
    print(f"  📸 Extracted {len(saved_paths)} frames to: {output_dir}")
    return saved_paths


# ──────────────────────────────────────────────
# Core: Build multimodal message content
# ──────────────────────────────────────────────
def build_image_content(image_paths: list[str], prompt: str) -> list[dict]:
    """
    Constructs the content list for a HumanMessage with one or more images.
    """
    content = []

    # Add all images
    for img_path in image_paths:
        ext = Path(img_path).suffix.lower().lstrip(".")
        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png",  "gif": "image/gif",
                    "webp": "image/webp"}
        mime = mime_map.get(ext, "image/jpeg")
        b64  = file_to_base64(img_path)

        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{b64}",
            },
        })

    # Add the text prompt
    content.append({"type": "text", "text": prompt})
    return content


# ──────────────────────────────────────────────
# Core: Call LLM and print token usage
# ──────────────────────────────────────────────
def analyze_with_token_tracking(content: list[dict], system_prompt: str = "") -> str:
    """
    Sends the multimodal content to the LLM and prints token usage.
    Returns the text response.
    """
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=content))

    print("\n⏳ Sending request to GLM-4.6V …")
    response = llm.invoke(messages)

    # ── Token Usage ──────────────────────────────
    usage = getattr(response, "usage_metadata", None) or {}

    # LangChain exposes usage via response_metadata as fallback
    if not usage:
        usage = response.response_metadata.get("token_usage", {})

    input_tokens  = (
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or "N/A"
    )
    output_tokens = (
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or "N/A"
    )
    total_tokens  = (
        usage.get("total_tokens")
        or (
            (input_tokens + output_tokens)
            if isinstance(input_tokens, int) and isinstance(output_tokens, int)
            else "N/A"
        )
    )

    print("\n" + "─" * 50)
    print("📊  TOKEN USAGE")
    print("─" * 50)
    print(f"  🔵 Input  (prompt)      : {input_tokens:>8} tokens")
    print(f"  🟢 Output (completion)  : {output_tokens:>8} tokens")
    print(f"  🟡 Total                : {total_tokens:>8} tokens")
    print("─" * 50)

    return response.content


# ──────────────────────────────────────────────
# Weaviate RAG Integration
# ──────────────────────────────────────────────
def search_sbc_rules(query: str, limit: int = 3) -> list:
    """
    1. Embeds the search query using the local TEI server.
    2. Queries the local Weaviate 'SBC_Rule' collection.
    3. Returns the top matching structural rules.
    """
    tei_url = "http://localhost:8088/embed"
    
    # 1. Generate Embedding via TEI
    try:
        resp = requests.post(tei_url, json={"inputs": [query]})
        resp.raise_for_status()
        query_vector = resp.json()[0]
    except Exception as e:
        print(f"\n❌ Error generating embedding for RAG: {e}")
        return []
        
    # 2. Query Weaviate Database
    try:
        # Using custom ports 8081 / 50052 per docker-compose
        client = weaviate.connect_to_local(port=8081, grpc_port=50052)
        collection = client.collections.get("SBC_Rule")
        
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        
        results = []
        for obj in response.objects:
            results.append({
                "category": obj.properties.get("category"),
                "sub_category": obj.properties.get("sub_category"),
                "rule_text": obj.properties.get("rule_text"),
                "sbc_reference": obj.properties.get("sbc_reference"),
                "cv_target": obj.properties.get("cv_target"),
                "priority": obj.properties.get("priority"),
                "distance": obj.metadata.distance
            })
        client.close()
        return results
        
    except Exception as e:
        print(f"\n❌ Error querying Weaviate: {e}")
        return []


# ──────────────────────────────────────────────
# Public: Analyze Image
# ──────────────────────────────────────────────
def analyze_image(
    image_path: str,
    prompt: str = "Describe in detail everything you see in this image.",
) -> str:
    """
    Analyze a single image and return a text description.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"\n🖼️  Analyzing image: {image_path}")
    content = build_image_content([image_path], prompt)
    result  = analyze_with_token_tracking(content, system_prompt=(
        "You are an expert visual analyst. Provide clear, structured, "
        "and detailed descriptions of images."
    ))
    return result


# ──────────────────────────────────────────────
# Public: Analyze Video
# ──────────────────────────────────────────────
def analyze_video(
    video_path: str,
    prompt: str = "Analyze this video. Describe the scene, objects, actions, and any text visible across these frames.",
    num_frames: int = 6,
) -> str:
    """
    Analyze a video by extracting key frames and sending them to the LLM.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"\n🎬 Analyzing video: {video_path}")
    print(f"   Extracting {num_frames} key frames …")
    frames = extract_video_frames(video_path, num_frames=num_frames)

    if not frames:
        raise RuntimeError("No frames could be extracted from the video.")

    # Build frame-aware prompt
    full_prompt = (
        f"I'm showing you {len(frames)} frames extracted evenly from a video. "
        f"{prompt}"
    )
    content = build_image_content(frames, full_prompt)
    result  = analyze_with_token_tracking(content, system_prompt=(
        "You are an expert video analyst. Given multiple key frames from a video, "
        "describe the video content, actions, events, and any visible text in detail."
    ))
    return result


# ──────────────────────────────────────────────
# Interactive CLI
# ──────────────────────────────────────────────
def interactive_menu():
    print("\n" + "═" * 55)
    print("  🤖  GLM-4.6V Vision Analyzer (LangChain)")
    print("═" * 55)
    print("  Model   :", MODEL)
    print("  API URL :", BASE_URL)
    print("═" * 55)

    while True:
        print("\nWhat would you like to analyze?")
        print("  [1] Image")
        print("  [2] Video")
        print("  [3] Exit")
        choice = input("\nYour choice (1/2/3): ").strip()

        if choice == "1":
            path   = input("Enter image file path: ").strip().strip('"').strip("'")
            prompt = input(
                "Custom prompt (press Enter for default): "
            ).strip() or "Describe in detail everything you see in this image."

            result = analyze_image(path, prompt)
            print("\n" + "═" * 55)
            print("📝  DESCRIPTION")
            print("═" * 55)
            print(result)
            print("═" * 55)
            
            # --- RAG Lookup ---
            print("\n🔍 Querying Weaviate Vector DB for relevant Saudi Building Code Rules...")
            rules = search_sbc_rules(query=result, limit=3)
            if rules:
                for i, r in enumerate(rules, 1):
                    print(f"\n[Match {i}] (Distance: {r['distance']:.3f})")
                    print(f"Category : {r['category']} > {r['sub_category']}")
                    print(f"SBC Ref  : {r['sbc_reference']}")
                    print(f"Priority : {r['priority']}")
                    print(f"CV Target: {r['cv_target']}")
                    print(f"Rule     : {r['rule_text']}")
            else:
                print("No relevant rules found or database not running.")

        elif choice == "2":
            path   = input("Enter video file path: ").strip().strip('"').strip("'")
            frames = input("Number of frames to extract (default 6): ").strip()
            frames = int(frames) if frames.isdigit() else 6
            prompt = input(
                "Custom prompt (press Enter for default): "
            ).strip() or (
                "Analyze this video. Describe the scene, objects, "
                "actions, and any text visible across these frames."
            )

            result = analyze_video(path, prompt, num_frames=frames)
            print("\n" + "═" * 55)
            print("📝  VIDEO DESCRIPTION")
            print("═" * 55)
            print(result)
            print("═" * 55)
            
            # --- RAG Lookup ---
            print("\n🔍 Querying Weaviate Vector DB for relevant Saudi Building Code Rules...")
            rules = search_sbc_rules(query=result, limit=3)
            if rules:
                for i, r in enumerate(rules, 1):
                    print(f"\n[Match {i}] (Distance: {r['distance']:.3f})")
                    print(f"Category : {r['category']} > {r['sub_category']}")
                    print(f"SBC Ref  : {r['sbc_reference']}")
                    print(f"Priority : {r['priority']}")
                    print(f"CV Target: {r['cv_target']}")
                    print(f"Rule     : {r['rule_text']}")
            else:
                print("No relevant rules found or database not running.")

        elif choice == "3":
            print("👋 Goodbye!")
            sys.exit(0)

        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

        again = input("\nAnalyze another file? (y/n): ").strip().lower()
        if again != "y":
            print("👋 Goodbye!")
            break


# ──────────────────────────────────────────────
# Quick programmatic example (demo mode)
# ──────────────────────────────────────────────
def demo_image_analysis():
    """
    Demonstrates image analysis using the existing PNG in the workspace.
    """
    sample_image = "/home/redspark/Pictures/saudi-complience/c0c2edef-e6a4-4345-a847-a223e8e73d8c.png"
    print(f"\n🚀 Demo: analyzing sample image → {sample_image}")
    result = analyze_image(sample_image)
    print("\n" + "═" * 55)
    print("📝  DESCRIPTION")
    print("═" * 55)
    print(result)
    print("═" * 55)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_image_analysis()
    else:
        interactive_menu()
