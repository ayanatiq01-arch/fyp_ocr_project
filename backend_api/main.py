"""
main.py - FastAPI server for the Bilingual (Urdu & Arabic) OCR Scanner.

Endpoints
---------
GET  /health        -> liveness + which models are loaded
POST /api/v1/ocr    -> multipart/form-data with field "file" (the cropped image)
                       returns text, language and bounding boxes as JSON

Run
---
    venv\\Scripts\\python main.py                       (0.0.0.0:8000)
    venv\\Scripts\\uvicorn main:app --host 0.0.0.0 --port 8000

Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import image_processing as ip
from ocr_pipeline import OcrPipeline, PipelineResult

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BACKEND_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BACKEND_DIR / "temp_uploads"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# Keep uploaded images after processing (useful for collecting test data).
KEEP_UPLOADS = os.getenv("KEEP_UPLOADS", "0") == "1"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("ocr-api")


# --------------------------------------------------------------------------- #
# Response schema (mirrored by frontend_app/lib/api_service.dart)
# --------------------------------------------------------------------------- #
class Candidate(BaseModel):
    text: str
    confidence: float = Field(..., description="0-100")


class Line(BaseModel):
    bbox: List[int] = Field(..., description="[x1, y1, x2, y2] in uploaded-image pixels")
    text: str
    language: str = Field(..., description="urdu | arabic | unknown")
    engine: str = Field(..., description="UTRNet | PaddleOCR")
    confidence: float = Field(..., description="0-100, of the accepted engine")
    is_bullet: bool
    candidates: Dict[str, Candidate] = Field(
        ..., description="Both engines' raw results, keyed by language")


class BlockOut(BaseModel):
    id: int
    type: str = Field(..., description="Text | Title | List")
    bbox: List[int]
    language: str = Field(..., description="urdu | arabic | mixed | unknown")
    text: str
    lines: List[Line]


class ImageInfo(BaseModel):
    width: int
    height: int


class OcrResponse(BaseModel):
    request_id: str
    image: ImageInfo
    skew_angle: float = Field(..., description="Degrees the page was rotated to deskew it")
    layout_engine: str
    blocks: List[BlockOut]
    formatted_text: str = Field(..., description="Text with paragraphs and bullets preserved")
    processing_ms: int


def _pct(x: float) -> float:
    return round(100.0 * x, 2)


def to_response(request_id: str, result: PipelineResult) -> OcrResponse:
    """Convert pipeline output to the API schema, mapping every box back to
    the coordinate system of the image the client uploaded."""
    pre = result.pre
    blocks_out: List[BlockOut] = []
    for idx, block in enumerate(result.blocks):
        lines = [
            Line(
                bbox=list(pre.to_original(l.box)),
                text=l.text,
                language=l.language,
                engine=l.engine,
                confidence=_pct(l.confidence),
                is_bullet=l.is_bullet,
                candidates={lang: Candidate(text=c.text, confidence=_pct(c.confidence))
                            for lang, c in l.candidates.items()},
            )
            for l in block.lines
        ]
        blocks_out.append(BlockOut(
            id=idx, type=block.type, bbox=list(pre.to_original(block.box)),
            language=block.language, text=block.text(), lines=lines))

    width, height = pre.original_size
    return OcrResponse(
        request_id=request_id,
        image=ImageInfo(width=width, height=height),
        skew_angle=pre.skew_angle,
        layout_engine=result.layout_engine,
        blocks=blocks_out,
        formatted_text=result.formatted_text(),
        processing_ms=result.processing_ms,
    )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at start-up (takes ~30 s; first run also
    downloads the PaddleOCR and LayoutParser models)."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading OCR models ...")
    app.state.pipeline = OcrPipeline()  # device from OCR_DEVICE env (default cpu)
    yield
    app.state.pipeline.close()


app = FastAPI(
    title="Bilingual Urdu & Arabic OCR API",
    version="1.0.0",
    description="Digitises photos of historical books with mixed Urdu (Nastaliq) "
                "and Arabic (Naskh) text using confidence-score routing between "
                "UTRNet and PaddleOCR.",
    lifespan=lifespan,
)

# The mobile app is not a browser, but allowing CORS lets Flutter-web and
# Swagger clients on other origins call the API during development.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/health")
def health(request: Request) -> dict:
    pipeline: Optional[OcrPipeline] = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return {"status": "loading"}
    return {
        "status": "ok",
        "models": {
            "urdu": "UTRNet-Large (HRNet-DBiLSTM-CTC)",
            "arabic": pipeline.arabic.model_name,
            "layout": pipeline.layout.engine,
        },
    }


# Plain `def` (not async): OCR is CPU-bound, so FastAPI runs it in its
# thread pool and the event loop stays responsive for other requests.
@app.post("/api/v1/ocr", response_model=OcrResponse)
def ocr(request: Request, file: UploadFile = File(..., description="Cropped page image")
        ) -> OcrResponse:
    request_id = uuid.uuid4().hex
    ext = Path(file.filename or "").suffix.lower() or ".jpg"
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type '{ext}'. "
                                 f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(400, "Empty file")

    # Store the upload in temp_uploads/ while it is being processed.
    temp_path = UPLOAD_DIR / f"{request_id}{ext}"
    temp_path.write_bytes(data)
    try:
        try:
            image = ip.decode_image(data)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        result = request.app.state.pipeline.process(image)
        response = to_response(request_id, result)
        logger.info("request %s: %d blocks, %d ms", request_id, len(response.blocks),
                    response.processing_ms)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("request %s failed", request_id)
        raise HTTPException(500, "OCR processing failed. See server log for details.") from exc
    finally:
        if not KEEP_UPLOADS:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    # 0.0.0.0 so a phone on the same Wi-Fi network can reach the server.
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
