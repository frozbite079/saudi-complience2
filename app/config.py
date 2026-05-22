import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

GLM_API_KEY = os.getenv("GLM_API_KEY", "")
GLM_BASE_URL = os.getenv("GLM_BASE_URL", "https://api.z.ai/api/paas/v4")
GLM_MODEL = os.getenv("GLM_MODEL", "GLM-4.6V")

TEI_URL = os.getenv("TEI_URL", "http://localhost:8088/embed")

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "9000"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50052"))
WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "SBC_Rule")

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS_OBSERVE = int(os.getenv("LLM_MAX_TOKENS_OBSERVE", "8192"))
LLM_MAX_TOKENS_JUDGE = int(os.getenv("LLM_MAX_TOKENS_JUDGE", "8192"))

VIDEO_FRAMES_COUNT = int(os.getenv("VIDEO_FRAMES_COUNT", "6"))
VIDEO_OUTPUT_DIR = Path(os.getenv("VIDEO_OUTPUT_DIR", str(BASE_DIR / "outputs")))
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
