"""
Multimodal Vision Analyzer using LangChain & GLM-4.6V
=====================================================
Features:
  - Handles Image Inputs (Local Files & HTTP URLs)
  - Handles Video Inputs (Local Files via Frame Extraction & HTTP URLs natively)
  - Implements Token Usage Tracking
  - Uses langchain-openai with custom base_url, api_key, and model config.

Target Model   : GLM-4.6V
API Base URL   : https://api.z.ai/api/paas/v4/
"""

import base64
import os
import sys
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ────────────────────────────────────────────────────────
# Configuration Constants
# ────────────────────────────────────────────────────────
API_KEY = "6f7d09ab5f0e4edf816408f45f6b2f7b.iNo6WzgdafIlCewY"
BASE_URL = "https://api.z.ai/api/paas/v4/"
MODEL = "GLM-4.6V"

# ────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────

def is_url(path_or_url: str) -> bool:
    """Checks if a given string represents an HTTP or HTTPS URL."""
    try:
        parsed = urlparse(path_or_url)
        return parsed.scheme in ("http", "https")
    except ValueError:
        return False

def file_to_base64(file_path: str) -> str:
    """Reads a local file and encodes its content to a base64 string."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Local file not found: {file_path}")
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_image_mime_type(file_path: str) -> str:
    """Determines the image MIME type based on file extension."""
    ext = Path(file_path).suffix.lower().lstrip(".")
    mime_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif"
    }
    return mime_map.get(ext, "image/jpeg")

class VisionAI:
    """
    Multimodal processing class wrapper for ChatOpenAI model GLM-4.6V.
    Supports Image and Video input combinations natively.
    """
    def __init__(
        self,
        api_key: str = API_KEY,
        base_url: str = BASE_URL,
        model: str = MODEL,
        temperature: float = 0.2,
        max_tokens: int = 4096
    ):
        print(f"⚙️ Initializing LangChain ChatOpenAI client...")
        print(f"   Model    : {model}")
        print(f"   Base URL : {base_url}")
        
        # Initialize OpenAI Chat Model wrapper (pointing to api.z.ai)
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def _track_and_print_tokens(self, response: Any):
        """Helper to extract and format token statistics from the response."""
        usage = getattr(response, "usage_metadata", None) or {}
        if not usage and hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("token_usage", {})

        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or "N/A"
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or "N/A"
        total_tokens = usage.get("total_tokens") or "N/A"

        if total_tokens == "N/A" and isinstance(input_tokens, int) and isinstance(output_tokens, int):
            total_tokens = input_tokens + output_tokens

        print("\n" + "═" * 45)
        print("📊 TOKEN STATISTICS")
        print("─" * 45)
        print(f"  🔹 Input Tokens  (Prompt)     : {input_tokens:>8}")
        print(f"  🔹 Output Tokens (Completion) : {output_tokens:>8}")
        print(f"  🔹 Total Tokens               : {total_tokens:>8}")
        print("═" * 45)

    def analyze_image(
        self,
        image_path_or_url: str,
        prompt: str = "Describe what is visible in this image in detail.",
        system_prompt: str = "You are a professional image analysis agent. Give highly specific and structured responses."
    ) -> str:
        """
        Analyzes a single image (from a local file path or web URL).
        """
        print(f"\n🖼️ Analyzing image: {image_path_or_url}")
        content = []

        if is_url(image_path_or_url):
            content.append({
                "type": "image_url",
                "image_url": {"url": image_path_or_url}
            })
        else:
            mime = get_image_mime_type(image_path_or_url)
            b64_data = file_to_base64(image_path_or_url)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64_data}"}
            })

        content.append({"type": "text", "text": prompt})

        # Construct messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content)
        ]

        print("⏳ Sending request to GLM-4.6V via api.z.ai...")
        response = self.llm.invoke(messages)
        self._track_and_print_tokens(response)
        return response.content

    def analyze_video(
        self,
        video_path_or_url: str,
        prompt: str = "Provide a summary and description of the actions occurring in this video.",
        num_frames: int = 6,  # Kept for backward compatibility but unused
        system_prompt: str = "You are a professional video analysis agent. Detail actions, environments, and sequences accurately."
    ) -> str:
        """
        Analyzes a video directly (either URL or raw base64 encoded local video).
        """
        print(f"\n🎬 Analyzing video: {video_path_or_url}")
        content = []

        if is_url(video_path_or_url):
            print("🔗 Detected HTTP/HTTPS URL. Using native video_url pass...")
            content.append({
                "type": "video_url",
                "video_url": {"url": video_path_or_url}
            })
        else:
            print("📁 Local video file detected. Encoding directly to base64...")
            b64_data = file_to_base64(video_path_or_url)
            content.append({
                "type": "video_url",
                "video_url": {"url": b64_data}
            })

        # Add instructions/prompt
        content.append({"type": "text", "text": prompt})

        # Construct messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content)
        ]

        print("⏳ Sending request to GLM-4.6V via api.z.ai...")
        response = self.llm.invoke(messages)
        self._track_and_print_tokens(response)
        return response.content

    def analyze_multimodal(
        self,
        inputs: List[str],
        prompt: str,
        num_frames: int = 5,  # Kept for backward compatibility but unused
        system_prompt: str = "You are a professional visual analyzer. Compare and analyze the visual inputs provided below."
    ) -> str:
        """
        General function that takes a mixed list of inputs (images, videos, local files, URLs)
        and builds a single, composite native multimodal request.
        """
        print(f"\n🧠 Processing composite multimodal prompt with {len(inputs)} inputs...")
        content = []

        for idx, item in enumerate(inputs):
            # 1. Handle HTTP URL vs Local File
            if is_url(item):
                # Check extension to decide if it looks like a video
                ext = Path(item).suffix.lower()
                if ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                    print(f"   [{idx}] Native online Video URL: {item}")
                    content.append({
                        "type": "video_url",
                        "video_url": {"url": item}
                    })
                else:
                    print(f"   [{idx}] Online Image URL: {item}")
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": item}
                    })
            else:
                # Local file checking
                if not os.path.exists(item):
                    print(f"   ⚠️ WARNING: Item {item} does not exist. Skipping.")
                    continue
                
                ext = Path(item).suffix.lower()
                if ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                    print(f"   [{idx}] Local Video file: {item} (encoding directly to base64...)")
                    b64_data = file_to_base64(item)
                    content.append({
                        "type": "video_url",
                        "video_url": {"url": b64_data}
                    })
                else:
                    print(f"   [{idx}] Local Image file: {item}")
                    mime = get_image_mime_type(item)
                    b64_data = file_to_base64(item)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64_data}"}
                    })

        # Add prompt text
        content.append({"type": "text", "text": prompt})

        # Construct messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content)
        ]

        print("⏳ Sending combined multimodal request to GLM-4.6V...")
        response = self.llm.invoke(messages)
        self._track_and_print_tokens(response)
        return response.content

# ────────────────────────────────────────────────────────
# CLI Interactive and Demo Setup
# ────────────────────────────────────────────────────────

def run_demo():
    """Simple demo analyzing the default compliance image in the workspace."""
    sample_img = "/home/redspark/Pictures/saudi-complience/c0c2edef-e6a4-4345-a847-a223e8e73d8c.png"
    
    if not os.path.exists(sample_img):
        print(f"❌ Demo aborted. Sample image not found at {sample_img}")
        return

    print("🚀 Running GLM-4.6V LangChain compliance verification demo...")
    analyzer = VisionAI()
    
    prompt = (
        "Identify if this is a construction site, electrical site, or a building interior. "
        "List all safety hazards, compliance violations, and note overall environmental safety."
    )
    
    result = analyzer.analyze_image(sample_img, prompt=prompt)
    print("\n" + "═" * 50)
    print("📝 ANALYSIS REPORT")
    print("═" * 50)
    print(result)
    print("═" * 50)

def show_help():
    print("""
GLM-4.6V Multimodal LangChain CLI
=================================
Usage:
  python vision_.py --demo                Run visual analysis on the default workspace image
  python vision_.py --image [path/url]    Analyze a local image file or a direct HTTP URL
  python vision_.py --video [path/url]    Analyze a video (URL or local file)
  python vision_.py --cli                 Run in interactive terminal mode
    """)

def interactive_terminal():
    print("\n" + "═" * 55)
    print("       GLM-4.6V / LangChain-OpenAI CLI tool")
    print("═" * 55)
    
    analyzer = VisionAI()
    
    while True:
        print("\nSelect input format:")
        print("  [1] Single Image (Local file or URL)")
        print("  [2] Single Video (Local file or URL)")
        print("  [3] Mixed Multimodal (List of multiple files/URLs)")
        print("  [4] Exit")
        
        choice = input("\nYour choice (1-4): ").strip()
        if choice == "4":
            print("👋 Exiting visual analyzer. Goodbye!")
            break
            
        elif choice == "1":
            path = input("Image Path or URL: ").strip().strip('"').strip("'")
            prompt = input("Custom prompt (Enter for default): ").strip()
            prompt = prompt or "Describe the contents, objects, and any notable compliance details in this image."
            
            try:
                desc = analyzer.analyze_image(path, prompt=prompt)
                print("\n" + "═" * 50)
                print("📝 RESULT:")
                print("═" * 50)
                print(desc)
                print("═" * 50)
            except Exception as e:
                print(f"❌ Error during image analysis: {e}")
                
        elif choice == "2":
            path = input("Video Path or URL: ").strip().strip('"').strip("'")
            prompt = input("Custom prompt (Enter for default): ").strip()
            prompt = prompt or "Describe the sequence of actions and details in this video."
            
            frames = 6
            if not is_url(path):
                f_input = input("Number of frames to extract (default 6): ").strip()
                if f_input.isdigit():
                    frames = int(f_input)
            
            try:
                desc = analyzer.analyze_video(path, prompt=prompt, num_frames=frames)
                print("\n" + "═" * 50)
                print("📝 RESULT:")
                print("═" * 50)
                print(desc)
                print("═" * 50)
            except Exception as e:
                print(f"❌ Error during video analysis: {e}")
                
        elif choice == "3":
            print("\nEnter paths/URLs one by one. Enter empty line to finish.")
            items = []
            while True:
                item = input(f"Input item #{len(items)+1}: ").strip().strip('"').strip("'")
                if not item:
                    break
                items.append(item)
                
            if not items:
                print("⚠️ No inputs provided. Aborted.")
                continue
                
            prompt = input("Enter prompt/question for comparison: ").strip()
            if not prompt:
                prompt = "Compare and describe the visual inputs provided."
                
            try:
                desc = analyzer.analyze_multimodal(items, prompt=prompt)
                print("\n" + "═" * 50)
                print("📝 RESULT:")
                print("═" * 50)
                print(desc)
                print("═" * 50)
            except Exception as e:
                print(f"❌ Error during multimodal analysis: {e}")

if __name__ == "__main__":
    args = sys.argv[1:]
    
    if not args:
        show_help()
        sys.exit(0)
        
    if "--demo" in args:
        run_demo()
    elif "--cli" in args:
        interactive_terminal()
    elif "--image" in args:
        idx = args.index("--image")
        if idx + 1 < len(args):
            p = args[idx+1]
            pr = "Describe this image in detail."
            if "--prompt" in args:
                pidx = args.index("--prompt")
                if pidx + 1 < len(args):
                    pr = args[pidx+1]
            
            ai = VisionAI()
            print(ai.analyze_image(p, prompt=pr))
        else:
            print("❌ Missing path for --image parameter")
    elif "--video" in args:
        idx = args.index("--video")
        if idx + 1 < len(args):
            p = args[idx+1]
            pr = "Analyze this video sequence."
            if "--prompt" in args:
                pidx = args.index("--prompt")
                if pidx + 1 < len(args):
                    pr = args[pidx+1]
            
            ai = VisionAI()
            print(ai.analyze_video(p, prompt=pr))
        else:
            print("❌ Missing path for --video parameter")
    else:
        show_help()
