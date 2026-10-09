"""
main.py - FastAPI server for the Bilingual (Urdu & Arabic) OCR Scanner.

Endpoints
---------
GET  /health             -> liveness + which models are loaded
POST /api/v1/ocr         -> multipart/form-data, field "file" (whole page photo).
                            Returns the finished page as one JSON object.
POST /api/v1/ocr/stream  -> same input; streams NDJSON (one JSON object per
                            line) so the app can show text while the rest of
                            the page is still being read:
                              {"event": "layout", ...page structure, no text}
                              {"event": "cell", "block": b, "row": r, "cell": c, ...}
                              {"event": "done", ...final page (= /api/v1/ocr)}
                              {"event": "error", "detail": "..."}

Run
---
    venv\\Scripts\\python main.py                       (0.0.0.0:8000)
    venv\\Scripts\\uvicorn main:app --host 0.0.0.0 --port 8000

Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
from dotenv import load_dotenv

# API keys (GEMINI_API_KEY) live in backend_api/.env, which git ignores.
# Loaded before ocr_pipeline is imported, as it reads its settings on import.
load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import image_processing as ip
from ocr_pipeline import Cell, OcrPipeline, PipelineResult

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BACKEND_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BACKEND_DIR / "temp_uploads"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
# Keep uploaded images after processing (useful for collecting test data).
KEEP_UPLOADS = os.getenv("KEEP_UPLOADS", "0") == "1"

from logging.handlers import RotatingFileHandler

# Log to the console and to backend_api/server.log (git-ignored, 2 x 1 MB),
# so a problem the phone reported can be looked up afterwards.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(Path(__file__).resolve().parent / "server.log",
                            maxBytes=1_000_000, backupCount=1, encoding="utf-8"),
    ],
)
logger = logging.getLogger("ocr-api")


# --------------------------------------------------------------------------- #
# Response schema (mirrored by frontend_app/lib/api_service.dart)
# --------------------------------------------------------------------------- #
class Candidate(BaseModel):
    text: str
    confidence: float = Field(..., description="0-100")


class CellOut(BaseModel):
    bbox: List[int] = Field(..., description="[x1, y1, x2, y2] in uploaded-image pixels")
    column: int = Field(..., description="Table column, 0 = right-most")
    text: str
    language: str = Field(..., description="urdu | arabic | unknown")
    engine: str = Field(..., description="UTRNet | PaddleOCR (empty until read)")
    confidence: float = Field(..., description="0-100, of the accepted engine")
    candidates: Dict[str, Candidate] = Field(
        default_factory=dict, description="Both engines' raw results, keyed by language")


class RowOut(BaseModel):
    bbox: List[int]
    is_bullet: bool
    text: str
    cells: List[CellOut] = Field(..., description="Right-to-left reading order")


class BlockOut(BaseModel):
    id: int
    type: str = Field(..., description="Text | Title | List | Table")
    bbox: List[int]
    columns: int = Field(..., description="Number of table columns (1 for text)")
    language: str = Field(..., description="urdu | arabic | mixed | unknown")
    text: str
    rows: List[RowOut]


class ReadingLine(BaseModel):
    """One printed line, in the reading order of the page."""
    block: int = Field(..., description="Index into blocks")
    type: str = Field(..., description="Type of the block the line belongs to")
    bbox: List[int]
    language: str = Field(..., description="urdu | arabic | mixed | unknown")
    text: str = Field(..., description="Cells right-to-left; TAB between table columns")


class ImageInfo(BaseModel):
    width: int
    height: int


class OcrResponse(BaseModel):
    request_id: str
    image: ImageInfo
    rotation: int = Field(0, description="0/90/180/270: degrees (counter-clockwise) the "
                                          "page was turned to make it upright")
    skew_angle: float = Field(..., description="Degrees the page was rotated to deskew it")
    layout_engine: str
    blocks: List[BlockOut]
    reading_order: List[ReadingLine] = Field(
        default_factory=list,
        description="Every line of the page top-to-bottom (Y), each line's words "
                    "right-to-left (X): the exact reading order of the original layout")
    formatted_text: str = Field(..., description="ROUGH DRAFT of the local OCR (UTRNet / "
                                                 "EasyOCR): blank line between blocks, "
                                                 "TAB between table columns, '• ' bullets")
    markdown: str = Field("", description="FINAL page as Markdown: corrected by the vision "
                                          "model if ai_correction names a model, else the "
                                          "rough draft as Markdown: ## headings, - bullets, one "
                                          "line per printed line (hard breaks), tables")
    processing_ms: int
    ai_correction: str = Field("off", description="Gemini model that corrected the draft, 'off' "
                                                  "(not corrected), or 'failed: <reason>' (rough "
                                                  "draft returned because Gemini was unavailable)")


def _pct(x: float) -> float:
    return round(100.0 * x, 2)


def cell_out(pre: ip.PreprocessResult, cell: Cell) -> CellOut:
    return CellOut(
        bbox=list(pre.to_original(cell.box)),
        column=cell.column,
        text=cell.text,
        language=cell.language,
        engine=cell.engine,
        confidence=_pct(cell.confidence),
        candidates={lang: Candidate(text=c.text, confidence=_pct(c.confidence))
                    for lang, c in cell.candidates.items()},
    )


def to_response(request_id: str, result: PipelineResult) -> OcrResponse:
    """Convert pipeline output to the API schema, mapping every box back to
    the coordinate system of the image the client uploaded."""
    pre = result.pre
    blocks_out, reading = [], []
    # result.blocks is already in reading order (ocr_pipeline.sort_reading_order).
    for idx, block in enumerate(result.blocks):
        rows = []
        for row in block.rows:
            text = row.text(table=block.type == "Table", columns=block.columns)
            bbox = list(pre.to_original(row.box))
            rows.append(RowOut(bbox=bbox, is_bullet=row.is_bullet, text=text,
                               cells=[cell_out(pre, c) for c in row.cells]))
            langs = {c.language for c in row.cells if c.text}
            reading.append(ReadingLine(
                block=idx, type=block.type, bbox=bbox,
                language=langs.pop() if len(langs) == 1 else ("mixed" if langs else "unknown"),
                text=f"• {text}" if row.is_bullet and text else text))
        blocks_out.append(BlockOut(
            id=idx, type=block.type, bbox=list(pre.to_original(block.box)),
            columns=block.columns, language=block.language, text=block.text(), rows=rows))
    width, height = pre.original_size
    return OcrResponse(
        request_id=request_id,
        image=ImageInfo(width=width, height=height),
        rotation=result.rotation,
        skew_angle=pre.skew_angle,
        layout_engine=result.layout_engine,
        blocks=blocks_out,
        reading_order=reading,
        formatted_text=result.formatted_text(),
        markdown=result.markdown(),
        processing_ms=result.processing_ms,
        ai_correction=result.ai_correction,
    )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at start-up (~40 s; the first run also
    downloads the PaddleOCR and LayoutParser models)."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading OCR models ...")
    app.state.pipeline = OcrPipeline()  # device from OCR_DEVICE env (default cpu)
    yield
    app.state.pipeline.close()


app = FastAPI(
    title="Bilingual Urdu & Arabic OCR API",
    version="2.0.0",
    description="Digitises photos of historical books with mixed Urdu (Nastaliq) "
                "and Arabic (Naskh) text: automatic text detection, UTRNet / PaddleOCR "
                "confidence-score routing, and page-layout reconstruction.",
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
            "detection": "PP-OCRv5_mobile_det",
            "layout": pipeline.layout.engine,
        },
    }


def _read_upload(file: UploadFile) -> tuple[str, Path, np.ndarray]:
    """Validate the upload, store it in temp_uploads/ and decode it.

    The file name / extension is NOT trusted (phone apps send camera files
    with various or missing extensions); the content itself is decoded and a
    non-image is rejected with 400.
    """
    request_id = uuid.uuid4().hex
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".img"  # only used for the temp file name
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(400, "Empty file")

    temp_path = UPLOAD_DIR / f"{request_id}{ext}"
    temp_path.write_bytes(data)
    try:
        image = ip.decode_image(data)
    except ValueError as exc:
        # Keep the file for diagnosis (temp_uploads/failed_<id>...).
        temp_path.rename(temp_path.with_name(f"failed_{temp_path.name}"))
        logger.warning("upload %s (%s, %d bytes, starts %s, ends %s) not decodable: %s",
                       request_id, file.filename, len(data), data[:8].hex(), data[-4:].hex(), exc)
        raise HTTPException(400, str(exc)) from exc
    return request_id, temp_path, image


LANGUAGE_HELP = ("mixed = UTRNet reads every box, Arabic boxes are re-read by EasyOCR "
                 "(default); urdu = UTRNet only; arabic = EasyOCR only")
AI_HELP = ("true = the original image + rough draft go to Gemini, which returns the "
           "corrected page as Markdown; without Gemini the rough draft is returned "
           "(needs GEMINI_API_KEY)")


def _check_language(language: str) -> None:
    if language not in OcrPipeline.LANGUAGES:
        raise HTTPException(422, f"language must be one of {', '.join(OcrPipeline.LANGUAGES)}")


def _cleanup(path: Path) -> None:
    if not KEEP_UPLOADS:
        path.unlink(missing_ok=True)


# Plain `def` (not async): OCR is CPU-bound, so FastAPI runs it in its
# thread pool and the event loop stays responsive for other requests.
@app.post("/api/v1/ocr", response_model=OcrResponse)
def ocr(request: Request, file: UploadFile = File(..., description="Whole page photo"),
        language: str = Form("mixed", description=LANGUAGE_HELP),
        ai_correct: bool = Form(True, description=AI_HELP)) -> OcrResponse:
    _check_language(language)
    request_id, temp_path, image = _read_upload(file)
    try:
        result = request.app.state.pipeline.process(image, language, ai_correct)
        response = to_response(request_id, result)
        logger.info("request %s: %d blocks, %d ms", request_id, len(response.blocks),
                    response.processing_ms)
        return response
    except Exception as exc:
        logger.exception("request %s failed", request_id)
        raise HTTPException(500, "OCR processing failed. See server log for details.") from exc
    finally:
        _cleanup(temp_path)


@app.post("/api/v1/ocr/stream")
def ocr_stream(request: Request, file: UploadFile = File(..., description="Whole page photo"),
               language: str = Form("mixed", description=LANGUAGE_HELP),
               ai_correct: bool = Form(True, description=AI_HELP)) -> StreamingResponse:
    _check_language(language)
    request_id, temp_path, image = _read_upload(file)
    pipeline: OcrPipeline = request.app.state.pipeline

    def events() -> Iterator[bytes]:
        """NDJSON: one compact JSON object per line (UTF-8, Urdu kept as-is)."""
        def line(obj: dict) -> bytes:
            return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

        try:
            pre = None
            for event, payload in pipeline.process_iter(image, language, ai_correct):
                if event == "layout":
                    pre = payload.pre
                    body = to_response(request_id, payload).model_dump()
                    body["total_cells"] = sum(len(r.cells) for b in payload.blocks for r in b.rows)
                    yield line({"event": "layout", **body})
                elif event == "cell":
                    bi, ri, ci, cell = payload
                    yield line({"event": "cell", "block": bi, "row": ri, "cell": ci,
                                **cell_out(pre, cell).model_dump()})
                elif event == "status":
                    yield line({"event": "status", "stage": payload})
                elif event == "done":
                    body = to_response(request_id, payload).model_dump()
                    logger.info("stream %s: %d blocks, %d ms", request_id,
                                len(body["blocks"]), body["processing_ms"])
                    yield line({"event": "done", **body})
        except Exception:
            logger.exception("stream %s failed", request_id)
            yield line({"event": "error",
                        "detail": "OCR processing failed. See server log for details."})
        finally:
            _cleanup(temp_path)

    # Starlette iterates a sync generator in its thread pool, so the CPU-bound
    # OCR does not block the event loop.
    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn

    # 0.0.0.0 so a phone on the same Wi-Fi network can reach the server.
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
