from __future__ import annotations

import datetime
import logging
import tempfile
import os
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from app.config import (
    GLM_MODEL,
    WEAVIATE_HOST,
    WEAVIATE_PORT,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_COLLECTION,
    TEI_URL,
    RAG_TOP_K,
    VIDEO_OUTPUT_DIR,
)
from app.violation_detector import detect_violations
from Voilation_Template.report_generator import save_report
from app.embedding_service import embed_text
from app.weaviate_client import (
    weaviate_connection,
    get_collection_stats,
    search_rules,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SBC Compliance Vision API",
    description="Saudi Building Code violation detection via two-level LLM + RAG pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


# ── Pydantic Models ─────────────────────────────────────────────────────

class URLOnlyRequest(BaseModel):
    url: str
    is_video: bool = False
    custom_prompt: str = ""
    top_k: int | None = None
    classification: str | None = None
    project_name: str = "Building Inspection"
    contractor_name: str = ""


class TextSearchRequest(BaseModel):
    query: str
    top_k: int | None = None
    classification: str | None = None


class HealthResponse(BaseModel):
    status: str
    services: dict[str, Any]


# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    services: dict[str, Any] = {"llm": "unknown", "tei": "unknown", "weaviate": "unknown"}

    try:
        import requests as req
        resp = req.get(f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}/v1/meta", timeout=5)
        services["weaviate"] = "ok" if resp.status_code == 200 else "error"
    except Exception:
        services["weaviate"] = "unreachable"

    try:
        import requests as req
        resp = req.post(TEI_URL, json={"inputs": ["ping"]}, timeout=5)
        services["tei"] = "ok" if resp.status_code == 200 else "error"
    except Exception:
        services["tei"] = "unreachable"

    services["llm"] = f"configured:{GLM_MODEL}"

    all_ok = services["weaviate"] == "ok" and services["tei"] == "ok"
    return HealthResponse(status="healthy" if all_ok else "degraded", services=services)


@app.post("/api/v1/analyze/url")
def analyze_url(request: URLOnlyRequest):
    if not request.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="url must start with http:// or https://")

    try:
        result = detect_violations(
            media_path_or_url=request.url,
            is_video=request.is_video,
            custom_prompt=request.custom_prompt,
            top_k=request.top_k,
            classification=request.classification,
            project_name=request.project_name,
            contractor_name=request.contractor_name,
        )
        # Auto-generate HTML report for both images and videos
        try:
            report_name = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.html"
            report_path = VIDEO_OUTPUT_DIR / report_name
            saved_path = save_report(
                result,
                report_path,
                project_name=request.project_name,
                contractor_name=request.contractor_name,
            )
            actual_report_name = Path(saved_path).name
            result["report_url"] = f"/api/v1/download/{actual_report_name}"
            logger.info("HTML Report generated: %s", actual_report_name)
        except Exception as report_err:
            logger.warning("Report generation failed: %s", report_err)

        result.pop("_report_verdicts", None)  # internal — strip before API response
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("URL analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze/upload")
async def analyze_upload(
    file: UploadFile = File(...),
    is_video: bool = Form(False),
    custom_prompt: str = Form(""),
    top_k: int | None = Form(None),
    classification: str | None = Form(None),
    project_name: str = Form("Building Inspection"),
    contractor_name: str = Form(""),
):
    suffix = Path(file.filename or "upload").suffix
    # Auto-detect video from file extension
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if suffix.lower() in video_extensions:
        is_video = True
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = detect_violations(
            media_path_or_url=tmp_path,
            is_video=is_video,
            custom_prompt=custom_prompt,
            top_k=top_k,
            classification=classification,
            project_name=project_name,
            contractor_name=contractor_name,
        )
        # Auto-generate HTML report for both images and videos
        try:
            report_name = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.html"
            report_path = VIDEO_OUTPUT_DIR / report_name
            saved_path = save_report(
                result,
                report_path,
                project_name=project_name,
                contractor_name=contractor_name,
            )
            actual_report_name = Path(saved_path).name
            result["report_url"] = f"/api/v1/download/{actual_report_name}"
            logger.info("HTML Report generated: %s", actual_report_name)
        except Exception as report_err:
            logger.warning("Report generation failed: %s", report_err)

        result.pop("_report_verdicts", None)  # internal — strip before API response
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("Upload analysis failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
@app.post("/api/v1/search/rules")
def search_sbc_rules(request: TextSearchRequest):
    try:
        query_vector = embed_text(request.query)
        results = search_rules(
            query_vector,
            top_k=request.top_k,
            classification=request.classification,
        )
        return JSONResponse(
            content={
                "query": request.query,
                "classification": request.classification,
                "results": results,
            }
        )
    except Exception as e:
        logger.exception("Rule search failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
def db_stats():
    try:
        with weaviate_connection() as client:
            stats = get_collection_stats(client)
        return JSONResponse(content=stats)
    except Exception as e:
        logger.exception("Stats retrieval failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/download/{filename}")
def download_file(filename: str):
    """Serve an annotated video or report file from the outputs directory."""
    # Prevent path traversal attacks
    safe_name = Path(filename).name
    file_path = VIDEO_OUTPUT_DIR / safe_name

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Detect media type from extension
    ext = file_path.suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".html": "text/html",
        ".css": "text/css",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".pdf": "application/pdf",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=safe_name,
    )


class ReportRequest(BaseModel):
    """Request body for on-demand report generation."""
    detection_result: dict
    project_name: str = "Building Inspection"
    contractor_name: str = ""
    location: str = ""
    inspector_name: str = ""
    building_type: str = ""
    permit_no: str = ""
    classification: str | None = None


@app.post("/api/v1/report/generate")
def generate_report(request: ReportRequest):
    """Generate an HTML inspection report from a previous detection result."""
    try:
        report_name = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.html"
        report_path = VIDEO_OUTPUT_DIR / report_name

        saved_path = save_report(
            request.detection_result,
            report_path,
            project_name=request.project_name,
            contractor_name=request.contractor_name,
            location=request.location,
            inspector_name=request.inspector_name,
            building_type=request.building_type,
            permit_no=request.permit_no,
            classification=request.classification,
        )

        actual_report_name = Path(saved_path).name

        return {
            "report_url": f"/api/v1/download/{actual_report_name}",
            "report_file": actual_report_name,
        }
    except Exception as e:
        logger.exception("Report generation failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
