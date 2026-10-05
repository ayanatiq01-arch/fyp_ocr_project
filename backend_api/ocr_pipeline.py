"""
ocr_pipeline.py
===============

Bilingual (Urdu Nastaliq + Arabic Naskh) OCR pipeline with a
**Confidence Score Routing Algorithm**.

    uploaded photo (whole page - no manual cropping needed)
        │
        ▼
    orientation + image_processing.preprocess()   0/90/180/270 · deskew · shadows · denoise
        │
        ▼
    TextDetector (PaddleOCR DB)          AUTO-CROP: one box per text line / table cell
        │
        ▼
    build_layout()                       boxes -> rows -> blocks (paragraph / list /
        │                                table / title) + LayoutParser block hints
        ▼
    for every box, in reading order:
        ├──► UTRNetRecognizer        (Urdu)    ─┐  run at the same time
        └──► PaddleArabicRecognizer  (Arabic)  ─┘
                        │
                        ▼
                route_by_confidence()     higher confidence wins, other discarded
                        │
                        ▼
    results are streamed as they are ready; finally the page is re-assembled
    with its paragraphs, bullets and table columns.

Why confidences are comparable
------------------------------
Both recognisers are CTC models. PaddleOCR's ``rec_score`` is the mean of the
max softmax probability over the frames that survive CTC decoding (non-blank,
not a repeat). :class:`UTRNetRecognizer` computes its score **the same way**,
so the two numbers are on the same 0-1 scale and can be compared directly.

Models (all published, downloaded as-is):
* UTRNet-Large  - ``UTRNet-High-Resolution-Urdu-Text-Recognition/saved_models/
  UTRNet-Large/best_norm_ED.pth`` (from the UTRNet README, Google Drive link).
* PaddleOCR     - ``arabic_PP-OCRv5_mobile_rec`` (recognition) and
  ``PP-OCRv5_mobile_det`` (text detection), auto-downloaded to
  ``~/.paddlex/official_models`` on first run.
* LayoutParser  - ``lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import image_processing as ip

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paths / configuration (override with environment variables if needed)
# --------------------------------------------------------------------------- #
BACKEND_DIR = Path(__file__).resolve().parent
UTRNET_DIR = Path(os.getenv(
    "UTRNET_DIR", BACKEND_DIR / "UTRNet-High-Resolution-Urdu-Text-Recognition"))
UTRNET_WEIGHTS = Path(os.getenv(
    "UTRNET_WEIGHTS", UTRNET_DIR / "saved_models" / "UTRNet-Large" / "best_norm_ED.pth"))
UTRNET_GLYPHS = UTRNET_DIR / "UrduGlyphs.txt"

PADDLE_ARABIC_MODEL = os.getenv("PADDLE_ARABIC_MODEL", "arabic_PP-OCRv5_mobile_rec")
PADDLE_DET_MODEL = os.getenv("PADDLE_DET_MODEL", "PP-OCRv5_mobile_det")
LAYOUT_MODEL_CONFIG = os.getenv(
    "LAYOUT_MODEL_CONFIG", "lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config")
LAYOUT_SCORE_THRESHOLD = float(os.getenv("LAYOUT_SCORE_THRESHOLD", "0.5"))
# "cpu" (default) or "gpu" (needs CUDA builds of PyTorch and PaddlePaddle).
OCR_DEVICE = os.getenv("OCR_DEVICE", "cpu").lower()
# UTRNet (HRNet at full resolution) dominates run time, so PyTorch gets all
# cores; the much lighter Paddle models get half.
TORCH_THREADS = int(os.getenv("TORCH_THREADS", str(os.cpu_count() or 4)))
CPU_THREADS = int(os.getenv("OCR_CPU_THREADS", str(max(1, (os.cpu_count() or 4) // 2))))
# Router calibration (see route_by_confidence). 0.0 = pure score comparison.
ROUTER_URDU_MARGIN = float(os.getenv("ROUTER_URDU_MARGIN", "0.0"))
# Results below this confidence (from BOTH engines) are treated as non-text
# (page ornaments, stains, torn edges) and dropped from the final page.
MIN_TEXT_CONFIDENCE = float(os.getenv("MIN_TEXT_CONFIDENCE", "0.5"))
# AI page reading (GeminiPageReader). Needs GEMINI_API_KEY
# (backend_api/.env); GEMINI_CORRECTION=0 switches it off.
GEMINI_CORRECTION = os.getenv("GEMINI_CORRECTION", "1") != "0"
# Flash models tried in order. The free tier allows only ~20 requests per
# day PER MODEL, so when one model's daily quota is used up (or it is
# overloaded) the next one takes over.
GEMINI_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS", "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,"
                     "gemini-3.5-flash,gemini-3-flash-preview").split(",") if m.strip()]
# Long side of the photo sent to Gemini (pixels).
GEMINI_MAX_SIDE = int(os.getenv("GEMINI_MAX_SIDE", "2000"))
# Gemini gives no per-word score; its text is reported with this confidence.
GEMINI_CONFIDENCE = 0.99
GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "60"))  # seconds for the whole page
# A model that has not answered after this many seconds gets the next model
# as a parallel backup (first good answer wins).
GEMINI_HEDGE_AFTER = float(os.getenv("GEMINI_HEDGE_AFTER", "5"))
# Passes over all models when they are only overloaded (503), before the
# page is read by the (much slower) local OCR instead.
GEMINI_ROUNDS = int(os.getenv("GEMINI_ROUNDS", "3"))
# Gemini 3 models think before answering; "low" keeps a page at ~15-60 s.
GEMINI_THINKING = os.getenv("GEMINI_THINKING", "low")

LANG_URDU = "urdu"
LANG_ARABIC = "arabic"
_ARABIC_SCRIPT_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class EngineResult:
    """Output of one recogniser for one box."""
    engine: str          # "UTRNet" | "PaddleOCR"
    language: str        # LANG_URDU | LANG_ARABIC
    text: str
    confidence: float    # 0.0 - 1.0


@dataclass
class Cell:
    """One auto-cropped text box (a line, part of a line, or a table cell)."""
    box: ip.Box                            # on the pre-processed image
    column: int = 0                        # table column (0 = right-most)
    text: str = ""
    language: str = "unknown"
    engine: str = ""
    confidence: float = 0.0                # of the accepted engine, 0-1
    candidates: Dict[str, EngineResult] = field(default_factory=dict)
    done: bool = False


@dataclass
class Row:
    """Cells that sit on the same printed line, ordered right-to-left."""
    box: ip.Box
    cells: List[Cell] = field(default_factory=list)
    is_bullet: bool = False

    def text(self, table: bool = False, columns: int = 0) -> str:
        if table:
            slots = [""] * max(columns, 1)
            for c in self.cells:
                if c.text:
                    slots[c.column] = (slots[c.column] + " " + c.text).strip()
            return "\t".join(slots).rstrip("\t")
        return " ".join(c.text for c in self.cells if c.text)


@dataclass
class Block:
    """A layout region: paragraph ("Text"), "Title", "List" or "Table"."""
    box: ip.Box
    type: str = "Text"
    rows: List[Row] = field(default_factory=list)
    columns: int = 1

    @property
    def language(self) -> str:
        langs = {c.language for r in self.rows for c in r.cells if c.text}
        if not langs:
            return "unknown"
        return langs.pop() if len(langs) == 1 else "mixed"

    def text(self) -> str:
        """One output line per printed line; tables use TAB between columns,
        list items start with '• '."""
        out = []
        for row in self.rows:
            t = row.text(table=self.type == "Table", columns=self.columns)
            if t.strip():
                out.append(f"• {t}" if row.is_bullet else t)
        return "\n".join(out)


# --------------------------------------------------------------------------- #
# Recogniser 1: UTRNet (Urdu, Nastaliq)
# --------------------------------------------------------------------------- #
class UTRNetRecognizer:
    """Inference wrapper around the cloned UTRNet repository.

    Pre-processing follows ``read.py`` (grayscale, horizontal flip, height 32,
    normalise to [-1, 1]) with one speed change: instead of right-padding
    every image to 400 px, each image is padded only to the next multiple of
    16 (HRNet's down-sampling factor). HRNet cost grows with width, so short
    words / table cells are ~4x faster; accuracy on real book cells was the
    same or better in our tests.
    """

    name = "UTRNet"
    language = LANG_URDU
    IMG_H, MAX_W = 32, 400

    def __init__(self, weights: Path = UTRNET_WEIGHTS, device: str = "cpu"):
        if not weights.is_file():
            raise FileNotFoundError(
                f"UTRNet weights not found at {weights}. Download UTRNet-Large "
                "from the link in the UTRNet README into saved_models/UTRNet-Large/.")

        # The repo uses top-level imports (``from model import Model``,
        # ``from modules... import``), so its folder must be on sys.path.
        if str(UTRNET_DIR) not in sys.path:
            sys.path.insert(0, str(UTRNET_DIR))
        import torch
        from model import Model                       # UTRNet/model.py
        from utils import CTCLabelConverter           # UTRNet/utils.py

        self._torch = torch
        torch.set_num_threads(TORCH_THREADS)
        self.device = torch.device(device)

        with open(UTRNET_GLYPHS, encoding="utf-8") as fh:
            glyphs = "".join(line.strip("\n") for line in fh) + " "
        self.converter = CTCLabelConverter(glyphs)

        opt = SimpleNamespace(
            FeatureExtraction="HRNet", SequenceModeling="DBiLSTM", Prediction="CTC",
            imgH=self.IMG_H, imgW=self.MAX_W, batch_max_length=100, num_fiducial=20,
            input_channel=1, output_channel=32,  # read.py forces 32 for HRNet
            hidden_size=256, num_class=len(self.converter.character),
            device=self.device, rgb=False,
        )
        self.model = Model(opt).to(self.device)
        state = torch.load(str(weights), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        # UTRNet's test-time "temporal dropout" draws from NumPy's global RNG;
        # we seed it per call (under a lock) so results are reproducible.
        self._lock = threading.Lock()
        logger.info("UTRNet loaded from %s", weights)

    def _to_tensor(self, gray: np.ndarray):
        from PIL import Image
        torch = self._torch
        img = Image.fromarray(gray).convert("L")
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)          # as in read.py
        w, h = img.size
        new_w = min(self.MAX_W, max(16, int(np.ceil(self.IMG_H * w / float(h)))))
        img = img.resize((new_w, self.IMG_H), Image.Resampling.BICUBIC)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        x = x.sub_(0.5).div_(0.5)[None, None]                          # [1,1,H,W]
        padded_w = -(-new_w // 16) * 16
        if padded_w != new_w:  # replicate last column, like NormalizePAD
            x = torch.cat([x, x[..., -1:].expand(1, 1, self.IMG_H, padded_w - new_w)], dim=3)
        return x.to(self.device)

    def _decode(self, logits) -> Tuple[str, float]:
        probs = self._torch.softmax(logits, dim=2)
        max_probs, indices = probs.max(dim=2)
        chars, char_probs, prev = [], [], 0
        for p, i in zip(max_probs[0].cpu().numpy(), indices[0].cpu().numpy()):
            if i != 0 and i != prev:           # standard CTC greedy decoding
                chars.append(self.converter.character[i])
                char_probs.append(float(p))
            prev = i
        text = "".join(chars).strip()
        return text, (float(np.mean(char_probs)) if char_probs and text else 0.0)

    def recognize_batch(self, images: Sequence[np.ndarray]) -> List[EngineResult]:
        """Recognise grayscale crops (each at its own width)."""
        results = []
        for gray in images:
            x = self._to_tensor(gray)
            with self._lock, self._torch.no_grad():
                rng_state = np.random.get_state()
                np.random.seed(0)
                try:
                    logits = self.model(x)
                finally:
                    np.random.set_state(rng_state)
            text, conf = self._decode(logits)
            results.append(EngineResult(self.name, self.language, text, conf))
        return results


# --------------------------------------------------------------------------- #
# Recogniser 2: PaddleOCR (Arabic, Naskh)
# --------------------------------------------------------------------------- #
class PaddleArabicRecognizer:
    """PaddleOCR 3.x Arabic-script text-recognition model."""

    name = "PaddleOCR"
    language = LANG_ARABIC

    def __init__(self, model_name: str = PADDLE_ARABIC_MODEL, device: str = "cpu"):
        # Skip PaddleX's network probe of model mirrors on every start-up.
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import TextRecognition
        self.model_name = model_name
        self.model = TextRecognition(model_name=model_name, device=device,
                                     cpu_threads=CPU_THREADS)
        self._lock = threading.Lock()
        logger.info("PaddleOCR recogniser loaded: %s", model_name)

    def recognize_batch(self, images: Sequence[np.ndarray]) -> List[EngineResult]:
        if not images:
            return []
        bgr = [cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) for g in images]
        with self._lock:
            outputs = self.model.predict(input=bgr, batch_size=min(8, len(bgr)))
        results = []
        for out in outputs:
            text = str(out["rec_text"]).strip()
            conf = float(out["rec_score"]) if text else 0.0
            results.append(EngineResult(self.name, self.language, text, conf))
        return results


# --------------------------------------------------------------------------- #
# AI page reading: Gemini reads the whole page (text + boxes + layout)
# --------------------------------------------------------------------------- #
_URDU_ONLY_RE = re.compile(r"[ٹڈڑںےۓھگچپژکیہۂۃ]")


def script_language(text: str) -> str:
    """"urdu" if the text uses Urdu-only letters, "arabic" for other
    Arabic-script text, else "unknown" (used to pick the reading font)."""
    if _URDU_ONLY_RE.search(text):
        return LANG_URDU
    return LANG_ARABIC if _ARABIC_SCRIPT_RE.search(text) else "unknown"


def blocks_from_gemini(page: dict, width: int, height: int, language: str = "mixed",
                       engine: str = "Gemini") -> List[Block]:
    """Turns Gemini's JSON page ({"blocks": [{type, rows: [{is_bullet,
    cells: [{column, box_2d, text}]}]}]}) into the pipeline's Block / Row /
    Cell structure. ``box_2d`` is [ymin, xmin, ymax, xmax] on a 0-1000 scale;
    boxes are converted to pixels of the image (``width`` x ``height``)."""
    def to_box(b) -> Optional[ip.Box]:
        try:
            y1, x1, y2, x2 = (min(1000, max(0, int(v))) for v in b[:4])
        except (TypeError, ValueError):
            return None
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1 * width // 1000, y1 * height // 1000,
                -(-x2 * width // 1000), -(-y2 * height // 1000))

    blocks: List[Block] = []
    for b in page.get("blocks", []):
        kind = b.get("type") if b.get("type") in ("Title", "Text", "List", "Table") else "Text"
        rows: List[Row] = []
        for r in b.get("rows", []):
            cells = []
            for c in r.get("cells", []):
                text = " ".join(str(c.get("text", "")).split())
                box = to_box(c.get("box_2d"))
                if not text or box is None:
                    continue
                lang = language if language in (LANG_URDU, LANG_ARABIC) else script_language(text)
                result = EngineResult(engine, lang, text, GEMINI_CONFIDENCE)
                cells.append(Cell(box=box, column=int(c.get("column") or 0), text=text,
                                  language=lang, engine=engine, confidence=GEMINI_CONFIDENCE,
                                  candidates={lang: result}, done=True))
            if not cells:
                continue
            if kind == "Table":
                cells.sort(key=lambda c: c.column)
            rows.append(Row(box=ip.union_box([c.box for c in cells]), cells=cells,
                            is_bullet=bool(r.get("is_bullet"))))
        if not rows:
            continue
        block = Block(box=ip.union_box([r.box for r in rows]), type=kind, rows=rows)
        if kind == "Table":
            used = sorted({c.column for r in rows for c in r.cells})
            remap = {old: new for new, old in enumerate(used)}
            for r in rows:
                for c in r.cells:
                    c.column = remap[c.column]
            block.columns = len(used)
        else:
            for r in rows:                      # one cell per line outside tables
                for c in r.cells:
                    c.column = 0
        blocks.append(block)
    return blocks


class GeminiPageReader:
    """Reads a whole page photo with Google Gemini (Flash, vision).

    One request per page: Gemini returns every printed line (and every table
    cell) with its text exactly as printed and its bounding box, grouped into
    Title / Text / List / Table blocks in reading order - so text, boxes and
    format come from the same model. The free tier allows ~20 requests per
    day per model, so the next model in ``GEMINI_MODELS`` takes over when
    one is exhausted or overloaded. Failures raise; the pipeline then falls
    back to the local OCR engines.
    """

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    PROMPT = (
        "This is a photo of a page from an Urdu and/or Arabic book. Transcribe ALL the text on "
        "the page exactly as printed and reproduce its layout.\n"
        "Return the page as blocks in reading order (top to bottom; when regions sit side by "
        "side, the right-hand one first). Block type is one of: Title (a heading), Text "
        "(paragraph), List (bulleted or numbered items), Table.\n"
        "Each block has rows; a row is ONE printed line. For every row give its cells:\n"
        "- In Text / Title / List blocks each row has exactly one cell covering the whole line.\n"
        "- In Table blocks each row is one table row and has one cell per column; column 0 is "
        "the RIGHT-most column.\n"
        "For every cell give box_2d = [ymin, xmin, ymax, xmax] on a 0-1000 scale of this image, "
        "tight around that cell's text, and the text exactly as printed: keep the original "
        "spelling, harakat only where printed, digits, brackets and punctuation. Do not "
        "translate, correct or add anything. If a line starts with a bullet symbol set "
        "is_bullet true and leave the symbol out of the text. Include headers, page numbers "
        "and every line - skip nothing.\n"
        "Strictly preserve the exact visual formatting of the page: one row per physical "
        "printed line (the line breaks must match the book exactly - never join or split "
        "lines), a prominent heading is its own Title block, bulleted items are List rows "
        "with is_bullet true, and paragraphs that are visually separated are separate "
        "blocks. The server turns this structure into a Markdown document (## headings, "
        "- bullets, line breaks, tables).")
    SCHEMA = {"type": "OBJECT", "properties": {"blocks": {"type": "ARRAY", "items": {
        "type": "OBJECT", "properties": {
            "type": {"type": "STRING", "enum": ["Title", "Text", "List", "Table"]},
            "rows": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                "is_bullet": {"type": "BOOLEAN"},
                "cells": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                    "column": {"type": "INTEGER"},
                    "box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "text": {"type": "STRING"}},
                    "required": ["box_2d", "text"]}}},
                "required": ["cells"]}}},
        "required": ["type", "rows"]}}}, "required": ["blocks"]}

    def __init__(self, api_key: str, models: Sequence[str] = tuple(GEMINI_MODELS)):
        import requests
        self.api_key = api_key
        self.models = list(models)
        self.last_model = ""                       # model that answered last
        self._resting: Dict[str, float] = {}      # model -> time its quota is back
        self._http = requests.Session()
        logger.info("Gemini page reading enabled: %s", ", ".join(self.models))

    @classmethod
    def from_env(cls) -> Optional["GeminiPageReader"]:
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key or not GEMINI_CORRECTION:
            logger.info("Gemini page reading disabled (no GEMINI_API_KEY or GEMINI_CORRECTION=0)")
            return None
        return cls(key)

    def read_page(self, image_bgr: np.ndarray) -> dict:
        """Gemini's JSON page for the photo (boxes on a 0-1000 scale)."""
        import base64
        h, w = image_bgr.shape[:2]
        s = min(1.0, GEMINI_MAX_SIDE / float(max(h, w)))
        send = cv2.resize(image_bgr, (max(1, int(w * s)), max(1, int(h * s))),
                          interpolation=cv2.INTER_AREA) if s < 1.0 else image_bgr
        ok, buf = cv2.imencode(".jpg", send, [cv2.IMWRITE_JPEG_QUALITY, 92])
        body = {
            "contents": [{"role": "user", "parts": [
                {"inline_data": {"mime_type": "image/jpeg",
                                 "data": base64.b64encode(buf.tobytes()).decode("ascii")}},
                {"text": self.PROMPT}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": self.SCHEMA,
                "mediaResolution": "MEDIA_RESOLUTION_HIGH",
                "thinkingConfig": {"thinkingLevel": GEMINI_THINKING},
            },
        }
        return self._post(body, self._parse_page)

    @staticmethod
    def _parse_page(data: dict) -> dict:
        """The page JSON of a reply; raises if the answer is cut off or not
        valid JSON (that model's answer is then not used)."""
        cand = data["candidates"][0]
        reason = cand.get("finishReason", "STOP")
        if reason != "STOP":
            raise RuntimeError(f"answer incomplete ({reason})")
        answer = "".join(p.get("text", "") for p in cand["content"].get("parts", []))
        page = json.loads(answer)
        if not isinstance(page, dict) or not isinstance(page.get("blocks"), list):
            raise RuntimeError("answer has no blocks")
        return page

    def _post(self, body: dict, parse):
        """Sends the request and returns ``parse(reply)`` of the first good answer.

        Speed: Google's Flash models are often overloaded, so a request is not
        retried after a pause. A model that fails (busy, quota, bad answer)
        hands over to the next model at once, and a model that has not
        answered after GEMINI_HEDGE_AFTER seconds gets the next model as a
        parallel backup - whichever answers first is used.
        """
        deadline = time.time() + GEMINI_TIMEOUT
        errors: List[str] = []
        # Overloaded models often answer a few seconds later: go round the
        # models again (until the deadline) before giving up.
        for round_no in range(GEMINI_ROUNDS):
            models = [m for m in self.models if self._resting.get(m, 0) <= time.time()]
            if not models:
                break                                  # daily quota used up everywhere
            if round_no:
                time.sleep(1.5)
            if time.time() + 5 > deadline:
                break
            try:
                return self._post_round(models, body, parse, deadline, errors)
            except RuntimeError:
                continue
        raise RuntimeError("no Gemini model available (" + "; ".join(errors[-5:] or ["daily quota used up"]) + ")")

    def _post_round(self, models: List[str], body: dict, parse, deadline: float,
                    errors: List[str]):
        """One pass over ``models`` (hedged, see :meth:`_post`)."""
        replies: "queue.Queue[Tuple[str, object, Optional[Exception]]]" = queue.Queue()

        def call(model: str) -> None:
            try:
                replies.put((model, parse(self._call(model, body)), None))
            except Exception as exc:  # reported to the waiting loop
                replies.put((model, None, exc))

        started, pending = 0, 0

        def start_next() -> bool:
            nonlocal started, pending
            if started >= len(models):
                return False
            threading.Thread(target=call, args=(models[started],), daemon=True).start()
            started += 1
            pending += 1
            return True

        start_next()
        while pending:
            wait = min(GEMINI_HEDGE_AFTER, deadline - time.time())
            if wait <= 0:
                break
            try:
                model, result, error = replies.get(timeout=wait)
            except queue.Empty:
                start_next()                       # slow answer: start a backup model
                continue
            pending -= 1
            if error is None:
                self.last_model = model
                return result
            errors.append(f"{model}: {str(error)[:80]}")
            if not pending:
                start_next()                       # failed: next model right away
        raise RuntimeError("round failed")

    def _call(self, model: str, body: dict) -> dict:
        """One request to one model; raises on any non-200 reply."""
        resp = self._http.post(self.URL.format(model=model), json=body, timeout=GEMINI_TIMEOUT,
                               headers={"x-goog-api-key": self.api_key})
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            wait = self._retry_delay(resp)
            if "PerDay" in resp.text or wait > 60:
                self._resting[model] = time.time() + max(wait, 600)
                logger.warning("Gemini %s: daily free quota used up", model)
        raise RuntimeError(f"HTTP {resp.status_code}")

    @staticmethod
    def _retry_delay(resp) -> float:
        """Seconds Google asks us to wait (RetryInfo.retryDelay), 0 if unknown."""
        try:
            for d in resp.json()["error"].get("details", []):
                if "retryDelay" in d:
                    return float(str(d["retryDelay"]).rstrip("s"))
        except Exception:
            pass
        return 0.0


# --------------------------------------------------------------------------- #
# Auto-crop: text detection
# --------------------------------------------------------------------------- #
class TextDetector:
    """Finds every text line / table cell on the page (PaddleOCR DB detector).

    This is the automatic crop: the user photographs the whole page and each
    returned box is read separately. Settings were tuned on a real book page:
    ``unclip_ratio=1.0`` keeps boxes tight so neighbouring rows with heavy
    harakat do not merge, ``limit_side_len=1280`` gives enough resolution to
    separate table cells.
    """

    def __init__(self, model_name: str = PADDLE_DET_MODEL, device: str = "cpu"):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import TextDetection
        # enable_mkldnn=False: Paddle 3.3's oneDNN path fails on this model
        # ("ConvertPirAttribute2RuntimeAttribute not support").
        self.model = TextDetection(
            model_name=model_name, device=device, cpu_threads=CPU_THREADS,
            enable_mkldnn=False, limit_side_len=1280, limit_type="max",
            thresh=0.3, box_thresh=0.5, unclip_ratio=1.0)
        self._lock = threading.Lock()
        logger.info("PaddleOCR text detector loaded: %s", model_name)

    def _detect_raw(self, gray: np.ndarray) -> List[ip.Box]:
        """Axis-aligned boxes of the detector's polygons, clipped to the image."""
        with self._lock:
            out = self.model.predict(input=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))[0]
        h, w = gray.shape[:2]
        boxes: List[ip.Box] = []
        for poly in out["dt_polys"]:
            p = np.asarray(poly)
            x1, y1 = np.floor(p.min(axis=0)).astype(int)
            x2, y2 = np.ceil(p.max(axis=0)).astype(int)
            box = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
            if box[2] - box[0] >= 6 and box[3] - box[1] >= 6:
                boxes.append(box)
        return boxes

    def detect(self, gray: np.ndarray, binary: np.ndarray) -> List[ip.Box]:
        boxes = self._detect_raw(gray)
        if not boxes:
            return self._drop_vertical_strips(self._fallback(binary, []))
        boxes = self._drop_vertical_strips(boxes)
        boxes = self._merge_stacked(self._split_tall(gray, binary, boxes))
        boxes = self._resolve_containment(boxes)
        # The OpenCV fallback can pick up page borders too: filter again.
        return self._drop_vertical_strips(self._fallback(binary, boxes))

    def _split_tall(self, gray: np.ndarray, binary: np.ndarray,
                    boxes: List[ip.Box]) -> List[ip.Box]:
        """A box much taller than a typical line holds several merged lines
        (common for a column of short Arabic words with heavy harakat).

        The region is enlarged 2x and detected again: at the larger scale the
        detector separates the lines itself. If that still gives one box,
        the projection-profile line segmenter is used instead.
        """
        median_h = float(np.median([b[3] - b[1] for b in boxes]))
        result = []
        for b in boxes:
            if b[3] - b[1] <= 1.8 * median_h:
                result.append(b)
                continue
            pad = int(0.2 * median_h)
            x0, y0 = max(0, b[0] - pad), max(0, b[1] - pad)
            region = ip.crop(gray, (x0, y0, b[2] + pad, b[3] + pad))
            big = cv2.resize(region, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            sub = [(x0 + s[0] // 2, y0 + s[1] // 2, x0 + -(-s[2] // 2), y0 + -(-s[3] // 2))
                   for s in self._detect_raw(big)]
            sub = [s for s in sub if s[3] - s[1] >= 0.4 * median_h]
            if len(sub) >= 2:
                result.extend(sub)
                continue
            bands = ip.segment_lines(ip.crop(binary, b))
            if len(bands) >= 2:
                result.extend((b[0], b[1] + t, b[2], b[1] + bt) for t, bt in bands)
            else:
                result.append(b)
        return result

    @staticmethod
    def _resolve_containment(boxes: List[ip.Box]) -> List[ip.Box]:
        """Fix boxes that swallow a neighbouring line.

        With dense harakat / waqf marks (e.g. Quranic text) the detector may
        return one box covering two lines AND a separate box for one of
        those lines. The big box is then replaced by the part of it that no
        other box covers (the missing line); leftovers shorter than half a
        line are dropped. This removes duplicate lines and recovers lost ones.
        """
        if len(boxes) < 2:
            return boxes
        median_h = float(np.median([b[3] - b[1] for b in boxes]))
        result = []
        for i, outer in enumerate(boxes):
            oh = outer[3] - outer[1]
            inner = [b for j, b in enumerate(boxes) if j != i
                     and ip.boxes_overlap_ratio(outer, b) > 0.8
                     and (b[3] - b[1]) * (b[2] - b[0]) < (outer[3] - outer[1]) * (outer[2] - outer[0])
                     and oh > 1.3 * (b[3] - b[1])]
            if not inner:
                result.append(outer)
                continue
            # vertical spans of `outer` not covered by any inner box
            covered = np.zeros(oh, dtype=bool)
            for b in inner:
                covered[max(0, b[1] - outer[1]):max(0, b[3] - outer[1])] = True
            y = 0
            while y < oh:
                if covered[y]:
                    y += 1
                    continue
                start = y
                while y < oh and not covered[y]:
                    y += 1
                piece = (outer[0], outer[1] + start, outer[2], outer[1] + y)
                piece_area = float((piece[2] - piece[0]) * (piece[3] - piece[1]))

                def covered_by(b: ip.Box) -> float:
                    """Share of the piece's OWN area that box b covers."""
                    ix = max(0, min(piece[2], b[2]) - max(piece[0], b[0]))
                    iy = max(0, min(piece[3], b[3]) - max(piece[1], b[1]))
                    return ix * iy / piece_area

                duplicate = any(covered_by(b) > 0.5 for j, b in enumerate(boxes) if j != i)
                if y - start >= 0.5 * median_h and not duplicate:
                    result.append(piece)
        if len(result) != len(boxes):
            logger.info("containment: %d boxes -> %d", len(boxes), len(result))
        return result

    @staticmethod
    def _drop_vertical_strips(boxes: List[ip.Box]) -> List[ip.Box]:
        """Remove tall, narrow boxes: decorative page borders and the book's
        fold, which the detector often reports as text. Urdu/Arabic lines run
        horizontally, so a box much taller than a line yet narrower than one
        line height cannot be text."""
        median_h = float(np.median([b[3] - b[1] for b in boxes]))
        kept = [b for b in boxes
                if not (b[3] - b[1] > 2.5 * median_h and b[2] - b[0] < 0.8 * median_h)
                and not (b[3] - b[1] > 4 * median_h and b[2] - b[0] < 0.2 * (b[3] - b[1]))]
        if len(kept) < len(boxes):
            logger.info("dropped %d border/ornament strips", len(boxes) - len(kept))
        return kept or boxes

    @staticmethod
    def _merge_stacked(boxes: List[ip.Box]) -> List[ip.Box]:
        """Re-join a word the detector cut into an upper and a lower part.

        With heavy harakat (fatha, kasra, shadda) on short Arabic words the
        marks above and the letters below can come out as two boxes stacked
        on top of each other. Two boxes are merged when they overlap
        horizontally for most of the narrower one, the vertical gap is small
        and the result is still no taller than a normal line.
        """
        if len(boxes) < 2:
            return boxes
        median_h = float(np.median([b[3] - b[1] for b in boxes]))
        merged = True
        boxes = list(boxes)
        while merged:
            merged = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    a, b = boxes[i], boxes[j]
                    overlap_x = min(a[2], b[2]) - max(a[0], b[0])
                    narrower = min(a[2] - a[0], b[2] - b[0])
                    gap_y = max(a[1], b[1]) - min(a[3], b[3])      # < 0 if overlapping
                    union = ip.union_box([a, b])
                    if overlap_x > 0.6 * narrower and gap_y < 0.3 * median_h and \
                            union[3] - union[1] <= 1.6 * median_h:
                        boxes[i] = union
                        del boxes[j]
                        merged = True
                        break
                if merged:
                    break
        return boxes

    @staticmethod
    def _fallback(binary: np.ndarray, boxes: List[ip.Box]) -> List[ip.Box]:
        """Safety net: a word or line the detector missed is found with
        OpenCV on the ink not covered by any box.

        On real photos the uncovered ink is mostly NOT text (page borders,
        the other page, the table cloth, stains), so a candidate is only
        accepted when it looks like one printed line: 0.5-1.6 line heights
        tall, at most 8 line heights wide, and centred inside the area where
        the detector found text. With no detected boxes at all (e.g. a very
        faint scan) every paragraph line is accepted."""
        uncovered = binary.copy()
        for b in boxes:
            uncovered[b[1]:b[3], b[0]:b[2]] = 255
        median_h = float(np.median([b[3] - b[1] for b in boxes])) if boxes else 0.0
        area = ip.union_box(boxes) if boxes else None
        extra = []
        for para in ip.detect_paragraphs(uncovered):
            para_bin = ip.crop(uncovered, para)
            for top, bottom in ip.segment_lines(para_bin) or [(0, para[3] - para[1])]:
                ink = ip.ink_bounds(para_bin[top:bottom])
                if ink is None:
                    continue
                box = (para[0] + ink[0], para[1] + top + ink[1], para[0] + ink[2], para[1] + top + ink[3])
                if area is None:
                    extra.append(box)
                    continue
                bh, bw = box[3] - box[1], box[2] - box[0]
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                line_like = 0.5 * median_h <= bh <= 1.6 * median_h and 0.5 * median_h <= bw <= 8 * median_h
                inside = area[0] < cx < area[2] and area[1] < cy < area[3]
                if line_like and inside:
                    extra.append(box)
        if extra:
            logger.info("detector fallback added %d boxes", len(extra))
        return boxes + extra


# --------------------------------------------------------------------------- #
# THE ROUTER - Confidence Score Routing Algorithm
# --------------------------------------------------------------------------- #
def route_by_confidence(urdu: EngineResult, arabic: EngineResult,
                        urdu_margin: float = ROUTER_URDU_MARGIN
                        ) -> Tuple[EngineResult, EngineResult]:
    """Pick the language of a box by comparing the two engines' confidences.

    No language classifier is trained. Each engine only knows its own script
    well, so the engine that is *more confident* about the box is taken to
    be reading the correct language. The other result is discarded.

    ``urdu_margin`` (default 0.0 = plain comparison) is an optional
    calibration offset subtracted from UTRNet's score before comparing.
    Tune it on a labelled validation set; do not guess it.

    Ties go to Urdu (UTRNet), the primary language of the target books.

    Returns:
        (accepted, rejected)
    """
    if urdu.confidence - urdu_margin >= arabic.confidence:
        accepted, rejected = urdu, arabic      # Urdu (Nastaliq) wins
    else:
        accepted, rejected = arabic, urdu      # Arabic (Naskh) wins
    logger.debug("route: urdu=%.3f arabic=%.3f margin=%.2f -> %s",
                 urdu.confidence, arabic.confidence, urdu_margin, accepted.language)
    return accepted, rejected


# --------------------------------------------------------------------------- #
# LayoutParser (block-type hints)
# --------------------------------------------------------------------------- #
class LayoutAnalyzer:
    """LayoutParser PaddleDetection PubLayNet model.

    Used for block *types* (Title / List) when it recognises a region. The
    model was trained on English research papers and usually finds nothing on
    Urdu/Arabic book pages, so the page structure itself comes from
    :func:`build_layout`; the pipeline never depends on this model alone.
    """

    LABEL_MAP = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}

    def __init__(self):
        self.model = None
        self.engine = "unavailable"
        try:
            self.model = self._load_layoutparser()
            self.engine = "layoutparser-paddledetection"
            logger.info("LayoutParser model loaded: %s", LAYOUT_MODEL_CONFIG)
        except Exception:  # network error, missing package ...
            logger.exception("LayoutParser unavailable - continuing without block-type hints")
        self._lock = threading.Lock()

    @staticmethod
    def _load_layoutparser():
        import layoutparser as lp
        import paddle.inference as paddle_infer

        class _CpuSafePaddleLayoutModel(lp.PaddleDetectionLayoutModel):
            """LayoutParser 0.3.4 targets Paddle 2.x. On Paddle 3.x the default
            oneDNN fusion passes crash this model ("OneDnnContext does not
            have the input Filter"), so the predictor is rebuilt with oneDNN
            disabled. Everything else mirrors the original implementation."""

            def load_predictor(self, model_dir, device=None, enable_mkldnn=False,
                               thread_num=10):
                cfg = paddle_infer.Config(os.path.join(model_dir, "inference.pdmodel"),
                                          os.path.join(model_dir, "inference.pdiparams"))
                cfg.disable_gpu()
                cfg.set_cpu_math_library_num_threads(thread_num)
                cfg.disable_onednn()
                cfg.disable_glog_info()
                cfg.enable_memory_optim()
                cfg.switch_use_feed_fetch_ops(False)
                return paddle_infer.create_predictor(cfg)

        return _CpuSafePaddleLayoutModel(
            config_path=LAYOUT_MODEL_CONFIG,
            label_map=LayoutAnalyzer.LABEL_MAP,
            device="cpu",
            extra_config={"threshold": LAYOUT_SCORE_THRESHOLD, "thread_num": CPU_THREADS},
        )

    def detect(self, color_bgr: np.ndarray) -> List[Tuple[ip.Box, str]]:
        """Return (box, type) for Title / List / Text / Table regions."""
        if self.model is None:
            return []
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)  # model expects RGB
        with self._lock:
            layout = self.model.detect(rgb)
        regions = []
        for tb in layout:
            if tb.type == "Figure":
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in tb.coordinates)
            regions.append(((x1, y1, x2, y2), tb.type))
        return regions


# --------------------------------------------------------------------------- #
# Page structure: boxes -> rows -> blocks (paragraph / list / table / title)
# --------------------------------------------------------------------------- #
def _group_rows(boxes: List[ip.Box], line_h: float) -> List[Row]:
    """Chain boxes into printed lines, right-to-left.

    Each box is linked to its nearest neighbour on the LEFT whose vertical
    centre is within half a line height. Because the comparison is always
    local (neighbour to neighbour), a line that slowly drifts up or down
    across a curved book page still stays one row, while a global "same y"
    test would split it or merge it with the next row.
    """
    n = len(boxes)
    cy = [(b[1] + b[3]) / 2 for b in boxes]
    max_gap = 8 * line_h                   # wide table gaps are still joined

    # Candidate links right -> left, best (smallest cost) first.
    links = []
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes):
            if i == j:
                continue
            gap = a[0] - b[2]                        # b lies to the left of a
            overlap_x = min(a[2], b[2]) - max(a[0], b[0])
            if -0.3 * line_h <= gap <= max_gap and overlap_x < 0.5 * min(a[2] - a[0], b[2] - b[0]):
                dy = abs(cy[i] - cy[j])
                # 0.62: tolerates page curl, yet stays below half the usual
                # line pitch (~1.3 line heights) so rows are not crossed.
                if dy < 0.62 * line_h:
                    links.append((max(gap, 0) + 4 * dy, i, j))
    links.sort()

    left_of, right_of = [None] * n, [None] * n
    for _, i, j in links:
        if left_of[i] is None and right_of[j] is None:
            left_of[i], right_of[j] = j, i

    rows: List[Row] = []
    for start in range(n):
        if right_of[start] is not None:
            continue                                 # not the right-most box
        cells, k = [], start
        while k is not None:
            cells.append(Cell(box=boxes[k]))
            k = left_of[k]
        rows.append(Row(box=ip.union_box([c.box for c in cells]), cells=cells))
    # Order rows by the vertical centre of their right-most cell (start of line).
    rows.sort(key=lambda r: (r.cells[0].box[1] + r.cells[0].box[3]) / 2)
    return rows


def _has_column_gap(row: Row, line_h: float) -> bool:
    """True if two neighbouring cells are separated by a table-column gap
    (wider than a line height), not just a word space."""
    return any(a.box[0] - b.box[2] > 1.0 * line_h for a, b in zip(row.cells, row.cells[1:]))


def _assign_columns(block: Block, line_h: float) -> None:
    """Cluster cells into table columns by their RIGHT edge (Urdu/Arabic
    text is right-aligned), 0 = right-most column."""
    cells = [c for r in block.rows for c in r.cells]
    edges = sorted((c.box[2] for c in cells), reverse=True)
    columns: List[List[float]] = [[edges[0]]]
    for x in edges[1:]:
        if columns[-1][-1] - x > 1.0 * line_h:
            columns.append([x])
        else:
            columns[-1].append(x)
    anchors = [float(np.median(col)) for col in columns]
    for c in cells:
        c.column = int(np.argmin([abs(c.box[2] - a) for a in anchors]))
    block.columns = len(anchors)


def _align_table_rows(block: Block, line_h: float) -> None:
    """Rebuild a table's rows column by column with sequence alignment.

    Within one column the cells are simply ordered top to bottom. Each next
    column (right to left) is aligned to the rows built so far with dynamic
    programming (like Needleman-Wunsch): matching two cells costs their
    vertical distance, skipping one costs a fixed gap penalty. The reference
    height of a row is its most recently added cell, so a row that drifts
    down a curved page is followed step by step, and order is never swapped.
    Requires cell.column to be set (see _assign_columns).
    """
    def cy(c: Cell) -> float:
        return (c.box[1] + c.box[3]) / 2

    columns = [sorted((c for r in block.rows for c in r.cells if c.column == k), key=cy)
               for k in range(block.columns)]
    rows: List[List[Cell]] = []
    GAP, MAX_DY = 0.6, 0.9 * line_h      # a match (< 0.9) always beats two gaps (1.2)
    for col in columns:
        if not col:
            continue
        if not rows:
            rows = [[c] for c in col]
            continue
        ref = [cy(r[-1]) for r in rows]
        m, n = len(ref), len(col)
        cost = np.full((m + 1, n + 1), np.inf)
        move = np.zeros((m + 1, n + 1), dtype=np.int8)   # 1 match, 2 skip row, 3 new row
        cost[0, :] = np.arange(n + 1) * GAP
        cost[:, 0] = np.arange(m + 1) * GAP
        move[1:, 0], move[0, 1:] = 2, 3
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dy = abs(ref[i - 1] - cy(col[j - 1]))
                options = [(cost[i - 1, j] + GAP, 2), (cost[i, j - 1] + GAP, 3)]
                if dy < MAX_DY:
                    options.append((cost[i - 1, j - 1] + dy / line_h, 1))
                cost[i, j], move[i, j] = min(options)
        merged: List[List[Cell]] = []
        i, j = m, n
        while i > 0 or j > 0:
            step = move[i, j]
            if step == 1:
                merged.append(rows[i - 1] + [col[j - 1]])
                i, j = i - 1, j - 1
            elif step == 2:
                merged.append(rows[i - 1])
                i -= 1
            else:
                merged.append([col[j - 1]])
                j -= 1
        rows = merged[::-1]

    block.rows = [Row(box=ip.union_box([c.box for c in cells]), cells=cells) for cells in rows]
    block.box = ip.union_box([r.box for r in block.rows])


def select_main_page(boxes: List[ip.Box], rules: List[int],
                     image_size: Tuple[int, int]) -> List[ip.Box]:
    """Keep only text that belongs to the photographed page.

    * Boxes touching the image edge are cut off by the camera frame (the
      facing page, the table cloth) and cannot be read completely: dropped.
    * If a long vertical printed line (page frame / the book's fold) separates
      the main text from a small amount of text beyond it, that text is the
      facing page: dropped. A rule is only used this way when the text beyond
      it is less than a quarter of all text, so a two-column page (whose
      columns are also separated by a rule) is never cut in half.
    """
    w, h = image_size
    edge = 3
    kept = [b for b in boxes if b[0] > edge and b[1] > edge and b[2] < w - edge and b[3] < h - edge]
    if not kept:
        return boxes
    area = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in kept], dtype=np.float64)
    cx = np.array([(b[0] + b[2]) / 2 for b in kept])
    order = np.argsort(cx)
    centre = cx[order][np.searchsorted(np.cumsum(area[order]), area.sum() / 2)]  # area-weighted median
    total = area.sum()
    right = [r for r in rules if r > centre]
    left = [r for r in rules if r < centre]
    keep = np.ones(len(kept), dtype=bool)
    # Only boxes lying ENTIRELY beyond a rule are dropped: a side header that
    # straddles the page frame still belongs to the page.
    bx1 = np.array([b[0] for b in kept], dtype=np.float64)
    bx2 = np.array([b[2] for b in kept], dtype=np.float64)
    if right and area[bx1 > min(right)].sum() < 0.25 * total:
        keep &= ~(bx1 > min(right))
    if left and area[bx2 < max(left)].sum() < 0.25 * total:
        keep &= ~(bx2 < max(left))

    # Text margins: the main page's lines end on common right / left margins
    # (Urdu/Arabic is justified). A box that STARTS beyond the right margin
    # or ENDS before the left margin is the facing page - this also works
    # when a curled page hides the frame line.
    def weighted_percentile(values: np.ndarray, q: float) -> float:
        idx = np.argsort(values)
        cum = np.cumsum(area[idx]) / total
        return float(values[idx][min(len(idx) - 1, np.searchsorted(cum, q))])

    median_h = float(np.median([b[3] - b[1] for b in kept]))
    x1 = np.array([b[0] for b in kept], dtype=np.float64)
    x2 = np.array([b[2] for b in kept], dtype=np.float64)
    right_margin = weighted_percentile(x2, 0.9) + 0.3 * median_h
    left_margin = weighted_percentile(x1, 0.1) - 0.3 * median_h
    width = np.maximum(x2 - x1, 1.0)
    # more than half of the box lies outside the margin
    beyond_right = np.clip(x2 - np.maximum(x1, right_margin), 0, None) / width > 0.5
    before_left = np.clip(np.minimum(x2, left_margin) - x1, 0, None) / width > 0.5
    if area[beyond_right].sum() < 0.25 * total:
        keep &= ~beyond_right
    if area[before_left].sum() < 0.25 * total:
        keep &= ~before_left
    dropped = len(boxes) - int(keep.sum())
    if dropped:
        logger.info("main page: dropped %d boxes outside the page / at the photo edge", dropped)
    return [b for b, k in zip(kept, keep) if k]


def sort_reading_order(blocks: List[Block]) -> List[Block]:
    """Put everything in the reading order of the printed page.

    * inside a row: cells right-to-left (descending X), as Urdu/Arabic are read
    * inside a block: rows top-to-bottom by their vertical centre (median of
      the cells' centres, robust to one tall box with harakat)
    * blocks: top-to-bottom by their first row
    """
    def row_y(row: Row) -> float:
        return float(np.median([(c.box[1] + c.box[3]) / 2 for c in row.cells])) if row.cells else row.box[1]

    for block in blocks:
        for row in block.rows:
            row.cells.sort(key=lambda c: c.box[2], reverse=True)
        block.rows.sort(key=row_y)
    blocks.sort(key=lambda b: row_y(b.rows[0]) if b.rows else b.box[1])
    return blocks


def build_layout(boxes: List[ip.Box], separators: Sequence[int] = ()) -> List[Block]:
    """Re-create the page structure from the auto-cropped boxes.

    * rows   : boxes on the same line, ordered right-to-left
    * blocks : consecutive rows closer than ~0.8 line heights (a larger gap
               starts a new paragraph / title / table)
    * types  : "Table" when most rows have column gaps, "Title" for a lone
               centred or large line (LayoutParser hints: apply_layout_hints)
    """
    if not boxes:
        return []
    # Typical line height = median box height (robust to a few merged boxes).
    line_h = float(np.median([b[3] - b[1] for b in boxes]))
    rows = _group_rows(boxes, line_h)

    page = ip.union_box(boxes)
    page_cx, page_w = (page[0] + page[2]) / 2, page[2] - page[0]

    def is_heading(row: Row) -> bool:
        """A single short box centred on the page (e.g. a chapter heading)."""
        b = row.box
        return len(row.cells) == 1 and (b[2] - b[0]) < 0.5 * page_w and \
            abs((b[0] + b[2]) / 2 - page_cx) < 0.1 * page_w

    # Consecutive rows closer than ~0.8 line heights form one block; a
    # centred heading never merges with the block around it.
    blocks: List[Block] = []
    for row in rows:
        if blocks:
            prev = blocks[-1]
            gap = row.box[1] - prev.rows[-1].box[3]
            h_overlap = min(prev.box[2], row.box[2]) - max(prev.box[0], row.box[0])
            separate = is_heading(row) != is_heading(prev.rows[-1])
            # A printed horizontal rule between the two rows ends the block.
            prev_cy = (prev.rows[-1].box[1] + prev.rows[-1].box[3]) / 2
            row_cy = (row.box[1] + row.box[3]) / 2
            ruled = any(prev_cy < y < row_cy for y in separators)
            if gap < 0.8 * line_h and h_overlap > 0 and not separate and not ruled:
                prev.rows.append(row)
                prev.box = ip.union_box([prev.box, row.box])
                continue
        blocks.append(Block(box=row.box, rows=[row]))

    for i, block in enumerate(blocks):
        gapped = sum(_has_column_gap(r, line_h) for r in block.rows)
        if len(block.rows) >= 2 and gapped >= 0.5 * len(block.rows):
            block.type = "Table"
            _assign_columns(block, line_h)
            _align_table_rows(block, line_h)
        elif len(block.rows) == 1 and len(blocks) > 1 and (
                is_heading(block.rows[0]) or
                (i == 0 and block.box[3] - block.box[1] > 1.2 * line_h)):
            block.type = "Title"                   # centred or large lone line
    return sort_reading_order(blocks)


def apply_layout_hints(blocks: List[Block], hints: List[Tuple[ip.Box, str]]) -> None:
    """Use LayoutParser's Title / List labels for plain paragraphs it covers."""
    for block in blocks:
        if block.type == "Text":
            for hbox, htype in hints:
                if htype in ("Title", "List") and ip.boxes_overlap_ratio(block.box, hbox) > 0.6:
                    block.type = htype


# --------------------------------------------------------------------------- #
# Bullet / list-item detection
# --------------------------------------------------------------------------- #
# Numbered markers at the start (logical order) of a line: "1.", "۱۔", "(٣)", "a)".
_NUMBERED_RE = re.compile(r"^\s*[\(\[]?([0-9۰-۹٠-٩]{1,3}|[a-zA-Z])[\)\]\.\-۔:،]\s*")
# Symbols an OCR engine may emit for a printed bullet dot (UTRNet has no "•"
# in its alphabet and typically reads a bullet as a quote mark or a full stop).
_BULLET_CHARS = "•●▪◦○■□*-–—·٭‘’'\"`´“”.۔،,۰٠°"


def detect_bullet_glyph(line_binary: np.ndarray) -> bool:
    """Geometric bullet check on a binarised line (text = 0).

    In RTL text the bullet is the right-most glyph. It is treated as a bullet
    when it is a small, solid, roughly square blob, vertically centred in the
    line and separated from the text by a clear gap. Nuqta dots are excluded
    because they sit close to their letter.
    """
    ink = (line_binary == 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n < 3:  # need a bullet + at least one text component
        return False
    comps = stats[1:]
    line_top = comps[:, cv2.CC_STAT_TOP].min()
    line_bottom = (comps[:, cv2.CC_STAT_TOP] + comps[:, cv2.CC_STAT_HEIGHT]).max()
    line_h = max(1, line_bottom - line_top)

    right = comps[np.argmax(comps[:, cv2.CC_STAT_LEFT] + comps[:, cv2.CC_STAT_WIDTH])]
    bx, by, bw, bh, area = (int(v) for v in right[:5])
    if not (bw <= 0.45 * line_h and bh <= 0.45 * line_h and 0.5 <= bw / max(bh, 1) <= 2.0):
        return False
    if area / float(bw * bh) < 0.6:          # must be solid
        return False
    centre = (by + bh / 2.0 - line_top) / line_h
    # Loose window: Nastaliq descenders stretch the line box downwards.
    if not 0.15 <= centre <= 0.85:           # roughly vertically centred
        return False
    others = [c for c in comps if not np.array_equal(c, right)]
    nearest_right_edge = max(int(c[0] + c[2]) for c in others)
    return bx - nearest_right_edge >= max(bw, 0.25 * line_h)   # clear gap


def clean_bullet_text(text: str, glyph_bullet: bool) -> Tuple[str, bool]:
    """Normalise list markers. Returns (text, is_bullet)."""
    stripped = text.lstrip()
    if glyph_bullet:
        # Drop whatever the OCR engine produced for the bullet dot.
        if stripped and stripped[0] in _BULLET_CHARS:
            stripped = stripped[1:].lstrip()
        return stripped, True
    if stripped and stripped[0] in "•●▪◦":
        return stripped[1:].lstrip(), True
    return text, False


def _is_noise(cell: Cell) -> bool:
    """Page ornaments / stains read as text: both engines unsure, or output
    with no Arabic-script letters at modest confidence."""
    if not cell.text:
        return True
    best = max((c.confidence for c in cell.candidates.values()), default=0.0)
    if best < MIN_TEXT_CONFIDENCE:
        return True
    if len(cell.text.strip()) == 1 and cell.confidence < 0.9:
        return True                     # a lone uncertain glyph: ornament / stain
    return not _ARABIC_SCRIPT_RE.search(cell.text) and cell.confidence < 0.8


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Markdown output
# --------------------------------------------------------------------------- #
_MD_INLINE = re.compile(r"([\\`*_\[\]<>|~])")
_MD_LINE_START = re.compile(r"^(\s*)([#>+\-=]|\d+[.)])")


def md_escape(text: str, table: bool = False) -> str:
    """Escape Markdown syntax characters in recognised text so it is shown
    literally (Urdu / Arabic text rarely contains them, digits and
    brackets in it must not turn into lists or links)."""
    text = _MD_INLINE.sub(r"\\\1", " ".join(text.split()))
    return text if table else _MD_LINE_START.sub(lambda m: m.group(1) + "\\" + m.group(2), text)


def blocks_to_markdown(blocks: Sequence["Block"]) -> str:
    """Markdown that mirrors the printed page:

    * Title          -> ``## heading`` (one per printed line)
    * Text / List    -> one Markdown line per printed line, joined with hard
                        line breaks (two trailing spaces), bullets as ``- ``
    * Table          -> GitHub table; column 0 (right-most in the book) is the
                        first Markdown column, which right-to-left renderers
                        show on the right. The first row is used as the
                        header row (Markdown tables need one).
    * blocks are separated by a blank line.
    """
    out: List[str] = []
    for block in blocks:
        if block.type == "Table":
            cols = max(block.columns, 1)
            lines = []
            for i, row in enumerate(block.rows):
                slots = [""] * cols
                for c in row.cells:
                    if c.text and c.column < cols:
                        slots[c.column] = (slots[c.column] + " " + md_escape(c.text, True)).strip()
                lines.append("| " + " | ".join(s or " " for s in slots) + " |")
                if i == 0:
                    lines.append("|" + "|".join([" --- "] * cols) + "|")
            if lines:
                out.append("\n".join(lines))
            continue
        lines = []
        for row in block.rows:
            text = " ".join(md_escape(c.text) for c in row.cells if c.text)
            if not text:
                continue
            if block.type == "Title":
                lines.append(f"## {text}")
            elif row.is_bullet:
                lines.append(f"- {text}")
            else:
                lines.append(text)
        if not lines:
            continue
        if block.type == "Title":
            out.append("\n\n".join(lines))
        else:
            # Hard line break after every printed line (not after list items,
            # which already start their own line).
            joined = [line + ("  " if i < len(lines) - 1 and not lines[i + 1].startswith("- ")
                              and not line.startswith("- ") else "")
                      for i, line in enumerate(lines)]
            out.append("\n".join(joined))
    return "\n\n".join(out) + ("\n" if out else "")


@dataclass
class PipelineResult:
    pre: ip.PreprocessResult
    blocks: List[Block]
    layout_engine: str
    processing_ms: int
    rotation: int = 0            # degrees CCW applied to make the page upright
    ai_correction: str = "off"   # "gemini-...": page read by Gemini; "off"; "failed: ..."

    def formatted_text(self) -> str:
        """Plain text that preserves the page structure: blank line between
        blocks, one line per printed line, TAB between table columns, bullets
        as '• '."""
        return "\n\n".join(t for t in (b.text() for b in self.blocks) if t)

    def markdown(self) -> str:
        """The page as a Markdown document (see :func:`blocks_to_markdown`)."""
        return blocks_to_markdown(self.blocks)


class OcrPipeline:
    """Loads all models once and processes whole-page photos end-to-end."""

    # Boxes sent to both engines together; also the streaming granularity.
    CHUNK = 4
    # Detected boxes read upright and flipped for the 0/180 decision.
    ORIENTATION_SAMPLE_LINES = 8

    def __init__(self, device: str = OCR_DEVICE):
        """``device``: "cpu" or "gpu" (PyTorch 'cuda' / Paddle 'gpu:0')."""
        t0 = time.perf_counter()
        use_gpu = device == "gpu"
        self.urdu = UTRNetRecognizer(device="cuda" if use_gpu else "cpu")
        self.arabic = PaddleArabicRecognizer(device="gpu:0" if use_gpu else "cpu")
        self.detector = TextDetector(device="gpu:0" if use_gpu else "cpu")
        self.layout = LayoutAnalyzer()
        self.gemini = GeminiPageReader.from_env()
        # Two workers: one per engine, so both read the same boxes concurrently.
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ocr")
        # LayoutParser runs in the background on its own worker.
        self._layout_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="layout")
        self._warm_up()
        logger.info("OCR pipeline ready in %.1f s", time.perf_counter() - t0)

    def _warm_up(self) -> None:
        """Run every model once on a dummy image. Paddle and PyTorch do
        one-off setup work on the first call; doing it at start-up keeps it
        out of the user's first scan."""
        page = np.full((400, 600), 255, np.uint8)
        for y in range(60, 360, 60):
            cv2.rectangle(page, (60, y), (540, y + 22), 0, -1)
        line = page[50:95, 50:550]
        self.detector.detect(page, page)
        self.layout.detect(cv2.cvtColor(page, cv2.COLOR_GRAY2BGR))
        self.urdu.recognize_batch([line])
        self.arabic.recognize_batch([line])

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._layout_executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ API
    # Values accepted for the ``language`` option.
    LANGUAGES = ("mixed", LANG_URDU, LANG_ARABIC)

    def process(self, image_bgr: np.ndarray, language: str = "mixed",
                ai_correct: bool = True) -> PipelineResult:
        """Run everything and return the final result."""
        result = None
        for event, payload in self.process_iter(image_bgr, language, ai_correct):
            if event == "done":
                result = payload
        return result

    def process_iter(self, image_bgr: np.ndarray, language: str = "mixed",
                     ai_correct: bool = True) -> Iterator[Tuple[str, object]]:
        """Generator for streaming. Yields, in order:

        ("layout", PipelineResult)            page structure, no text yet
        ("cell",  (block_i, row_i, cell_i, Cell))   one per box, reading order
        ("done",  PipelineResult)             final page (noise removed,
                                              bullets marked)

        ``language`` (chosen by the user in the app):
          * "mixed"  - both engines read every box, the confidence router
                       decides (default, unchanged behaviour);
          * "urdu"   - only UTRNet reads the boxes (page known to be Urdu);
          * "arabic" - only PaddleOCR reads the boxes; also much faster,
                       because the slow UTRNet is skipped.

        ``ai_correct``: read the page with Gemini (text, boxes and layout in
        one request; needs GEMINI_API_KEY). Yields ("status", "ai_reading")
        first. If Gemini is unavailable (no key, no internet, quota used up)
        the page is read by the local engines as above.
        """
        if language not in self.LANGUAGES:
            raise ValueError(f"language must be one of {self.LANGUAGES}, got {language!r}")
        t0 = time.perf_counter()
        ai_status = "off"
        if ai_correct and self.gemini is not None:
            yield "status", "ai_reading"
            try:
                result = self._read_with_gemini(image_bgr, language)
                result.processing_ms = int((time.perf_counter() - t0) * 1000)
                logger.info("Gemini (%s) read %d lines in %d ms", result.ai_correction,
                            sum(len(b.rows) for b in result.blocks), result.processing_ms)
                yield "layout", result
                yield "done", result
                return
            except Exception as exc:  # network, quota, bad answer: local OCR instead
                logger.warning("Gemini page reading failed, using local OCR: %s", exc)
                ai_status = f"failed: {str(exc)[:160]}"
            yield "status", "local_ocr"
        # Orientation: text axis from profiles, then 0 vs 180 deg checked on
        # the detected text boxes. Only an upside-down page is processed twice.
        turns, flipped_turns = self._orientation_candidates(image_bgr)
        pre = ip.preprocess(image_bgr, max_side=2000, quarter_turns=turns)
        boxes = self._page_boxes(pre)
        if self._is_upside_down(pre, boxes):
            turns = flipped_turns
            pre = ip.preprocess(image_bgr, max_side=2000, quarter_turns=turns)
            boxes = self._page_boxes(pre)

        # LayoutParser only supplies block-type hints, so it runs in the
        # background and is applied at the end - it never delays the text.
        fut_hints = self._layout_executor.submit(self.layout.detect, pre.color)
        blocks = build_layout(boxes, pre.horizontal_rules)
        result = PipelineResult(pre=pre, blocks=blocks, layout_engine=self.layout.engine,
                                processing_ms=0, rotation=turns * 90)
        logger.info("layout: %d boxes, %d blocks in %d ms (rotated %d deg)", len(boxes),
                    len(blocks), (time.perf_counter() - t0) * 1000, turns * 90)
        yield "layout", result

        # Reading order: block by block, row by row, right-to-left in a row.
        order = [(bi, ri, ci, cell) for bi, b in enumerate(blocks)
                 for ri, r in enumerate(b.rows) for ci, cell in enumerate(r.cells)]
        for start in range(0, len(order), self.CHUNK):
            chunk = order[start:start + self.CHUNK]
            images = [self._crop(pre, item[3].box) for item in chunk]
            # Mixed: send the SAME crops to BOTH engines at the same time.
            # A fixed language runs only that language's engine.
            def skipped(engine) -> List[EngineResult]:
                return [EngineResult(engine.name, engine.language, "", 0.0) for _ in images]

            fut_u = self._executor.submit(self.urdu.recognize_batch, images) \
                if language != LANG_ARABIC else None
            fut_a = self._executor.submit(self.arabic.recognize_batch, images) \
                if language != LANG_URDU else None
            urdu_results = fut_u.result() if fut_u else skipped(self.urdu)
            arabic_results = fut_a.result() if fut_a else skipped(self.arabic)
            for (bi, ri, ci, cell), u, a in zip(chunk, urdu_results, arabic_results):
                if language == LANG_URDU:
                    accepted = u
                elif language == LANG_ARABIC:
                    accepted = a
                else:
                    accepted, _ = route_by_confidence(u, a)
                cell.text = accepted.text
                cell.language = accepted.language if accepted.text else "unknown"
                cell.engine = accepted.engine if accepted.text else ""
                cell.confidence = accepted.confidence
                cell.candidates = {LANG_URDU: u, LANG_ARABIC: a}
                cell.done = True
                yield "cell", (bi, ri, ci, cell)

        apply_layout_hints(result.blocks, fut_hints.result())
        self._finalise(result)
        result.ai_correction = ai_status
        result.processing_ms = int((time.perf_counter() - t0) * 1000)
        logger.info("processed %d boxes in %d ms", len(order), result.processing_ms)
        yield "done", result

    # ------------------------------------------------------------ helpers
    def _read_with_gemini(self, image_bgr: np.ndarray, language: str) -> PipelineResult:
        """Whole page read by Gemini. Boxes are in uploaded-image pixels, so
        the 'pre-processing' is the identity (no rotation, no deskew)."""
        h, w = image_bgr.shape[:2]
        page = self.gemini.read_page(image_bgr)
        blocks = blocks_from_gemini(page, w, h, language, engine="Gemini")
        if not blocks:
            raise RuntimeError("Gemini found no text on the page")
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        pre = ip.PreprocessResult(color=image_bgr, gray=gray, binary=gray, skew_angle=0.0,
                                  original_size=(w, h))
        return PipelineResult(pre=pre, blocks=blocks, layout_engine="gemini",
                              processing_ms=0, rotation=0, ai_correction=self.gemini.last_model)

    def _page_boxes(self, pre: ip.PreprocessResult) -> List[ip.Box]:
        """Auto-cropped text boxes of the main page only."""
        boxes = self.detector.detect(pre.gray, pre.binary)
        h, w = pre.gray.shape[:2]
        return select_main_page(boxes, pre.vertical_rules, (w, h))

    def _crop(self, pre: ip.PreprocessResult, box: ip.Box) -> np.ndarray:
        """Grayscale crop with a little vertical margin (harakat that stick
        out of the tight detector box) and a white border."""
        h = box[3] - box[1]
        extra = max(2, int(0.08 * h))
        gray = ip.crop(pre.gray, (box[0], box[1] - extra, box[2], box[3] + extra))
        return cv2.copyMakeBorder(gray, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)

    def _finalise(self, result: PipelineResult) -> None:
        """Drop noise, mark bullets, re-number empty structures."""
        pre = result.pre
        kept_blocks = []
        for block in result.blocks:
            kept_rows = []
            for row in block.rows:
                row.cells = [c for c in row.cells if not _is_noise(c)]
                if not row.cells:
                    continue
                row.box = ip.union_box([c.box for c in row.cells])
                if block.type != "Table":
                    first = row.cells[0]                       # right-most = start of line
                    glyph = detect_bullet_glyph(ip.crop(pre.binary, row.box))
                    first.text, row.is_bullet = clean_bullet_text(first.text, glyph)
                kept_rows.append(row)
            if not kept_rows:
                continue
            block.rows = kept_rows
            block.box = ip.union_box([r.box for r in kept_rows])
            if block.type == "Table":
                # Re-number columns: a column may have held only noise.
                used = sorted({c.column for r in kept_rows for c in r.cells})
                remap = {old: new for new, old in enumerate(used)}
                for r in kept_rows:
                    for c in r.cells:
                        c.column = remap[c.column]
                block.columns = len(used)
            if block.type == "Text":
                texts = [r.cells[0].text for r in kept_rows]
                listy = sum(1 for r, t in zip(kept_rows, texts) if r.is_bullet or _NUMBERED_RE.match(t))
                if listy >= max(1, len(kept_rows) // 2):
                    block.type = "List"
            kept_blocks.append(block)
        result.blocks = sort_reading_order(self._split_on_language(kept_blocks))

    @staticmethod
    def _split_on_language(blocks: List[Block]) -> List[Block]:
        """Start a new block where the running text switches between Arabic
        and Urdu (e.g. an Arabic verse followed by its Urdu translation), so
        the formatted text keeps them as separate paragraphs. Tables are
        left alone (a table row mixes both languages by design)."""
        out: List[Block] = []
        for block in blocks:
            if block.type == "Table" or len(block.rows) < 2:
                out.append(block)
                continue
            current: List[Row] = []
            current_lang = None
            for row in block.rows:
                langs = [c.language for c in row.cells if c.text]
                lang = max(set(langs), key=langs.count) if langs else None
                if current and lang and current_lang and lang != current_lang:
                    out.append(Block(box=ip.union_box([r.box for r in current]),
                                     type=block.type, rows=current))
                    current = []
                current.append(row)
                current_lang = lang or current_lang
            out.append(Block(box=ip.union_box([r.box for r in current]),
                             type=block.type, rows=current, columns=block.columns))
        return out

    @staticmethod
    def _orientation_candidates(image_bgr: np.ndarray) -> Tuple[int, int]:
        """Projection profiles decide whether text lines run horizontally
        (candidates 0 / 180 deg) or vertically (90 / 270 deg). Returned as
        CCW quarter turns: (most likely, same axis turned 180 deg)."""
        _, binary = ip.quick_binary(image_bgr)
        return (1, 3) if ip.text_runs_vertically(binary) else (0, 2)

    def _is_upside_down(self, pre: ip.PreprocessResult, boxes: List[ip.Box]) -> bool:
        """Read a sample of the DETECTED text boxes with the fast PaddleOCR
        recogniser as they are and turned 180 deg. Upside-down Arabic-script
        text gets a clearly lower confidence. Using detector boxes (not raw
        ink) keeps page borders and background patterns out of the test."""
        if not boxes:
            return False
        median_h = float(np.median([b[3] - b[1] for b in boxes]))
        textlike = [b for b in boxes
                    if 0.5 * median_h <= b[3] - b[1] <= 2 * median_h and b[2] - b[0] >= 1.5 * (b[3] - b[1])]
        sample = sorted(textlike or boxes, key=lambda b: b[2] - b[0],
                        reverse=True)[:self.ORIENTATION_SAMPLE_LINES]
        crops = [self._crop(pre, b) for b in sample]
        upright = self.arabic.recognize_batch(crops)
        flipped = self.arabic.recognize_batch([np.ascontiguousarray(np.rot90(c, 2)) for c in crops])
        up = float(np.mean([r.confidence for r in upright]))
        down = float(np.mean([r.confidence for r in flipped]))
        logger.info("orientation check on %d boxes: upright=%.3f flipped=%.3f", len(crops), up, down)
        # Upside-down photos are rare and give a clear gap (about 0.5 vs 0.3);
        # small, blurry photos give low scores both ways, so demand a clear
        # margin before turning the page over.
        return down > up + 0.12
