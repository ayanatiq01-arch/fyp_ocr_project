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

import logging
import os
import re
import sys
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

    def detect(self, gray: np.ndarray, binary: np.ndarray) -> List[ip.Box]:
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
        if not boxes:
            return self._fallback(binary, [])
        return self._fallback(binary, self._split_tall(binary, boxes))

    @staticmethod
    def _split_tall(binary: np.ndarray, boxes: List[ip.Box]) -> List[ip.Box]:
        """A box much taller than a typical line holds several merged lines:
        split it with the projection-profile line segmenter."""
        median_h = float(np.median([b[3] - b[1] for b in boxes]))
        result = []
        for b in boxes:
            if b[3] - b[1] > 1.8 * median_h:
                bands = ip.segment_lines(ip.crop(binary, b))
                if len(bands) >= 2:
                    result.extend((b[0], b[1] + t, b[2], b[1] + bt) for t, bt in bands)
                    continue
            result.append(b)
        return result

    @staticmethod
    def _fallback(binary: np.ndarray, boxes: List[ip.Box]) -> List[ip.Box]:
        """Safety net: text the detector missed is found with OpenCV on the
        ink that is not covered by any box. Small leftovers (harakat, dots
        just outside a tight box) are ignored."""
        median_h = float(np.median([b[3] - b[1] for b in boxes])) if boxes else 0.0
        uncovered = binary.copy()
        for b in boxes:
            uncovered[b[1]:b[3], b[0]:b[2]] = 255
        extra = []
        for para in ip.detect_paragraphs(uncovered):
            para_bin = ip.crop(uncovered, para)
            for top, bottom in ip.segment_lines(para_bin) or [(0, para[3] - para[1])]:
                ink = ip.ink_bounds(para_bin[top:bottom])
                if ink is None:
                    continue
                box = (para[0] + ink[0], para[1] + top + ink[1], para[0] + ink[2], para[1] + top + ink[3])
                bh, bw = box[3] - box[1], box[2] - box[0]
                if median_h == 0 or (bh >= 0.5 * median_h and bw >= 0.5 * median_h):
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


def build_layout(boxes: List[ip.Box]) -> List[Block]:
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
        return len(row.cells) == 1 and (b[2] - b[0]) < 0.8 * page_w and \
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
            if gap < 0.8 * line_h and h_overlap > 0 and not separate:
                prev.rows.append(row)
                prev.box = ip.union_box([prev.box, row.box])
                continue
        blocks.append(Block(box=row.box, rows=[row]))

    for i, block in enumerate(blocks):
        gapped = sum(_has_column_gap(r, line_h) for r in block.rows)
        if len(block.rows) >= 2 and gapped >= 0.5 * len(block.rows):
            block.type = "Table"
            _assign_columns(block, line_h)
        elif len(block.rows) == 1 and len(blocks) > 1 and (
                is_heading(block.rows[0]) or
                (i == 0 and block.box[3] - block.box[1] > 1.2 * line_h)):
            block.type = "Title"                   # centred or large lone line
    return blocks


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
    return not _ARABIC_SCRIPT_RE.search(cell.text) and cell.confidence < 0.8


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@dataclass
class PipelineResult:
    pre: ip.PreprocessResult
    blocks: List[Block]
    layout_engine: str
    processing_ms: int
    rotation: int = 0            # degrees CCW applied to make the page upright

    def formatted_text(self) -> str:
        """Plain text that preserves the page structure: blank line between
        blocks, one line per printed line, TAB between table columns, bullets
        as '• '."""
        return "\n\n".join(t for t in (b.text() for b in self.blocks) if t)


class OcrPipeline:
    """Loads all models once and processes whole-page photos end-to-end."""

    # Boxes sent to both engines together; also the streaming granularity.
    CHUNK = 4
    # Lines sampled per candidate orientation for the 0/180 decision.
    ORIENTATION_SAMPLE_LINES = 5

    def __init__(self, device: str = OCR_DEVICE):
        """``device``: "cpu" or "gpu" (PyTorch 'cuda' / Paddle 'gpu:0')."""
        t0 = time.perf_counter()
        use_gpu = device == "gpu"
        self.urdu = UTRNetRecognizer(device="cuda" if use_gpu else "cpu")
        self.arabic = PaddleArabicRecognizer(device="gpu:0" if use_gpu else "cpu")
        self.detector = TextDetector(device="gpu:0" if use_gpu else "cpu")
        self.layout = LayoutAnalyzer()
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
    def process(self, image_bgr: np.ndarray) -> PipelineResult:
        """Run everything and return the final result."""
        result = None
        for event, payload in self.process_iter(image_bgr):
            if event == "done":
                result = payload
        return result

    def process_iter(self, image_bgr: np.ndarray
                     ) -> Iterator[Tuple[str, object]]:
        """Generator for streaming. Yields, in order:

        ("layout", PipelineResult)            page structure, no text yet
        ("cell",  (block_i, row_i, cell_i, Cell))   one per box, reading order
        ("done",  PipelineResult)             final page (noise removed,
                                              bullets marked)
        """
        t0 = time.perf_counter()
        turns = self._detect_orientation(image_bgr)
        pre = ip.preprocess(image_bgr, max_side=2000, quarter_turns=turns)

        # LayoutParser only supplies block-type hints, so it runs in the
        # background and is applied at the end - it never delays the text.
        fut_hints = self._layout_executor.submit(self.layout.detect, pre.color)
        boxes = self.detector.detect(pre.gray, pre.binary)
        blocks = build_layout(boxes)
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
            # Send the SAME crops to BOTH engines at the same time.
            fut_u = self._executor.submit(self.urdu.recognize_batch, images)
            fut_a = self._executor.submit(self.arabic.recognize_batch, images)
            for (bi, ri, ci, cell), u, a in zip(chunk, fut_u.result(), fut_a.result()):
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
        result.processing_ms = int((time.perf_counter() - t0) * 1000)
        logger.info("processed %d boxes in %d ms", len(order), result.processing_ms)
        yield "done", result

    # ------------------------------------------------------------ helpers
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
            if block.type == "Text":
                texts = [r.cells[0].text for r in kept_rows]
                listy = sum(1 for r, t in zip(kept_rows, texts) if r.is_bullet or _NUMBERED_RE.match(t))
                if listy >= max(1, len(kept_rows) // 2):
                    block.type = "List"
            kept_blocks.append(block)
        result.blocks = kept_blocks

    def _detect_orientation(self, image_bgr: np.ndarray) -> int:
        """Return how many CCW quarter turns make the page upright.

        1. Projection profiles decide whether text lines currently run
           horizontally (candidates 0 / 180 deg) or vertically (90 / 270).
        2. The two remaining candidates differ by 180 deg. First the
           baseline position is checked (Arabic-script ink is densest in the
           lower part of a line). If that is not clear-cut, both candidates
           are read with the fast PaddleOCR recogniser on a few of the widest
           lines; upside-down text gets a much lower confidence.
        """
        gray, binary = ip.quick_binary(image_bgr)
        candidates = (1, 3) if ip.text_runs_vertically(binary) else (0, 2)

        first, _ = ip.rotate90(binary, candidates[0])
        baseline = ip.baseline_position(first)
        if abs(baseline - 0.5) >= 0.05:
            best = candidates[0] if baseline > 0.5 else candidates[1]
            logger.info("orientation: candidates=%s baseline=%.2f -> rotate %d deg",
                        [c * 90 for c in candidates], baseline, best * 90)
            return best

        scores = {}
        for k in candidates:
            g, _ = ip.rotate90(gray, k)
            b, _ = ip.rotate90(binary, k)
            samples = []
            for para in ip.detect_paragraphs(b):
                para_bin = ip.crop(b, para)
                for top, bottom in ip.segment_lines(para_bin):
                    ink = ip.ink_bounds(para_bin[top:bottom])
                    if ink is None:
                        continue
                    box = (para[0] + ink[0], para[1] + top, para[0] + ink[2], para[1] + bottom)
                    if box[3] - box[1] >= 8:
                        samples.append(ip.crop(g, box))
            samples.sort(key=lambda s: s.shape[1], reverse=True)
            samples = samples[:self.ORIENTATION_SAMPLE_LINES]
            results = self.arabic.recognize_batch(samples) if samples else []
            scores[k] = float(np.mean([r.confidence for r in results])) if results else 0.0

        best = max(candidates, key=lambda k: scores[k])
        logger.info("orientation: candidates=%s scores=%s -> rotate %d deg",
                    [c * 90 for c in candidates],
                    {c * 90: round(s, 3) for c, s in scores.items()}, best * 90)
        return best
