"""
ocr_pipeline.py
===============

Bilingual (Urdu Nastaliq + Arabic Naskh) OCR pipeline with a
**Confidence Score Routing Algorithm**.

    uploaded image
        │
        ▼
    image_processing.preprocess()      deskew · shadow removal · denoise · binarise
        │
        ▼
    LayoutAnalyzer (LayoutParser)      paragraph / title / list blocks (x, y)
        │
        ▼
    segment_lines()                    each block -> text lines
        │
        ├──► UTRNetRecognizer   (Urdu)    ─┐  run concurrently on the same lines
        └──► PaddleArabicRecognizer (Ar.) ─┘
                        │
                        ▼
                route_by_confidence()     higher confidence wins, other discarded
                        │
                        ▼
               blocks + lines + formatted text (paragraphs / bullets preserved)

Why confidences are comparable
------------------------------
Both recognisers are CTC models. PaddleOCR's ``rec_score`` is the mean of the
max softmax probability over the frames that survive CTC decoding (non-blank,
not a repeat). :class:`UTRNetRecognizer` computes its score **the same way**,
so the two numbers are on the same 0-1 scale and can be compared directly.

Models (nothing is invented - both are downloaded/used as published):
* UTRNet-Large  - ``UTRNet-High-Resolution-Urdu-Text-Recognition/saved_models/
  UTRNet-Large/best_norm_ED.pth`` (from the UTRNet README, Google Drive link).
* PaddleOCR     - ``arabic_PP-OCRv5_mobile_rec`` (PaddleOCR 3.x official model,
  auto-downloaded to ``~/.paddlex/official_models`` on first run).
* LayoutParser  - ``lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config``
  (PaddleDetection PubLayNet model, auto-downloaded on first run).
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
from typing import Dict, List, Optional, Sequence, Tuple

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

LANG_URDU = "urdu"
LANG_ARABIC = "arabic"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class EngineResult:
    """Output of one recogniser for one line image."""
    engine: str          # "UTRNet" | "PaddleOCR"
    language: str        # LANG_URDU | LANG_ARABIC
    text: str
    confidence: float    # 0.0 - 1.0


@dataclass
class RoutedLine:
    """A text line after confidence routing."""
    box: ip.Box                  # on the pre-processed image
    text: str
    language: str
    engine: str
    confidence: float            # 0.0 - 1.0 (of the accepted engine)
    candidates: Dict[str, EngineResult]
    is_bullet: bool = False


@dataclass
class Block:
    """A layout region (paragraph, title, list ...)."""
    box: ip.Box                  # on the pre-processed image
    type: str                    # "Text" | "Title" | "List"
    score: float = 1.0
    lines: List[RoutedLine] = field(default_factory=list)

    @property
    def language(self) -> str:
        langs = {l.language for l in self.lines if l.text}
        if not langs:
            return "unknown"
        return langs.pop() if len(langs) == 1 else "mixed"

    def text(self) -> str:
        """Block text with one output line per detected line."""
        out = []
        for line in self.lines:
            if not line.text:
                continue
            out.append(f"• {line.text}" if line.is_bullet else line.text)
        return "\n".join(out)


# --------------------------------------------------------------------------- #
# Recogniser 1: UTRNet (Urdu, Nastaliq)
# --------------------------------------------------------------------------- #
class UTRNetRecognizer:
    """Thin inference wrapper around the cloned UTRNet repository.

    Uses the repository's own ``Model``, ``CTCLabelConverter`` and
    ``NormalizePAD`` so pre-processing matches ``read.py`` exactly
    (grayscale, horizontal flip, resize to height 32, right-pad to 400).
    """

    name = "UTRNet"
    language = LANG_URDU

    # Same hyper-parameters as read.py / README for UTRNet-Large.
    IMG_H, IMG_W = 32, 400

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
        from dataset import NormalizePAD              # UTRNet/dataset.py

        self._torch = torch
        torch.set_num_threads(TORCH_THREADS)
        self.device = torch.device(device)

        with open(UTRNET_GLYPHS, encoding="utf-8") as fh:
            glyphs = "".join(line.strip("\n") for line in fh) + " "
        self.converter = CTCLabelConverter(glyphs)

        opt = SimpleNamespace(
            FeatureExtraction="HRNet", SequenceModeling="DBiLSTM", Prediction="CTC",
            imgH=self.IMG_H, imgW=self.IMG_W, batch_max_length=100, num_fiducial=20,
            input_channel=1, output_channel=32,  # read.py forces 32 for HRNet
            hidden_size=256, num_class=len(self.converter.character),
            device=self.device, rgb=False,
        )
        self.model = Model(opt).to(self.device)
        state = torch.load(str(weights), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

        self.transform = NormalizePAD((1, self.IMG_H, self.IMG_W))
        # UTRNet's test-time "temporal dropout" draws from NumPy's global RNG;
        # we seed it per call (under a lock) so results are reproducible.
        self._lock = threading.Lock()
        logger.info("UTRNet loaded from %s", weights)

    def _to_tensor(self, gray: np.ndarray):
        from PIL import Image
        img = Image.fromarray(gray).convert("L")
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)   # as in read.py
        w, h = img.size
        resized_w = min(self.IMG_W, max(1, int(np.ceil(self.IMG_H * w / float(h)))))
        img = img.resize((resized_w, self.IMG_H), Image.Resampling.BICUBIC)
        return self.transform(img)

    def recognize_batch(self, lines: Sequence[np.ndarray]) -> List[EngineResult]:
        """Recognise a batch of grayscale line images."""
        if not lines:
            return []
        torch = self._torch
        batch = torch.stack([self._to_tensor(g) for g in lines]).to(self.device)

        with self._lock, torch.no_grad():
            rng_state = np.random.get_state()
            np.random.seed(0)
            try:
                logits = self.model(batch)                 # [B, T, C]
            finally:
                np.random.set_state(rng_state)

        probs = torch.softmax(logits, dim=2)
        max_probs, indices = probs.max(dim=2)             # [B, T]
        max_probs, indices = max_probs.cpu().numpy(), indices.cpu().numpy()

        results = []
        for p_row, i_row in zip(max_probs, indices):
            # Standard CTC greedy decoding; keep the prob of every emitted char.
            chars, char_probs, prev = [], [], 0
            for p, i in zip(p_row, i_row):
                if i != 0 and i != prev:
                    chars.append(self.converter.character[i])
                    char_probs.append(float(p))
                prev = i
            text = "".join(chars).strip()
            conf = float(np.mean(char_probs)) if char_probs and text else 0.0
            results.append(EngineResult(self.name, self.language, text, conf))
        return results


# --------------------------------------------------------------------------- #
# Recogniser 2: PaddleOCR (Arabic, Naskh)
# --------------------------------------------------------------------------- #
class PaddleArabicRecognizer:
    """PaddleOCR 3.x text-recognition model for Arabic script (lines only;
    detection is done by our own layout + line segmentation)."""

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

    def recognize_batch(self, lines: Sequence[np.ndarray]) -> List[EngineResult]:
        """Recognise a batch of grayscale line images."""
        if not lines:
            return []
        bgr = [cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) for g in lines]
        with self._lock:
            outputs = self.model.predict(input=bgr, batch_size=min(8, len(bgr)))
        results = []
        for out in outputs:
            text = str(out["rec_text"]).strip()
            conf = float(out["rec_score"]) if text else 0.0
            results.append(EngineResult(self.name, self.language, text, conf))
        return results


# --------------------------------------------------------------------------- #
# THE ROUTER - Confidence Score Routing Algorithm
# --------------------------------------------------------------------------- #
def route_by_confidence(urdu: EngineResult, arabic: EngineResult,
                        urdu_margin: float = ROUTER_URDU_MARGIN
                        ) -> Tuple[EngineResult, EngineResult]:
    """Pick the language of a line by comparing the two engines' confidences.

    No language classifier is trained. Each engine only knows its own script
    well, so the engine that is *more confident* about the line is taken to
    be reading the correct language. The other result is discarded.

    ``urdu_margin`` (default 0.0 = plain comparison) is an optional
    calibration offset subtracted from UTRNet's score before comparing. UTRNet
    tends to be very confident even on Arabic Naskh lines, so on a labelled
    validation set a small margin (e.g. 0.05-0.10) may improve the *language*
    decision. Tune it on real book pages; do not guess it.

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
# Layout analysis (LayoutParser + OpenCV safety net)
# --------------------------------------------------------------------------- #
class LayoutAnalyzer:
    """Detects paragraph / title / list regions and their coordinates.

    Primary engine: LayoutParser's PaddleDetection PubLayNet model.
    Any ink the model misses (e.g. unusual historical layouts) is grouped into
    extra blocks with a morphological OpenCV fallback, so no text is lost.
    """

    LABEL_MAP = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}

    def __init__(self):
        self.model = None
        self.engine = "opencv-fallback"
        try:
            self.model = self._load_layoutparser()
            self.engine = "layoutparser-paddledetection"
            logger.info("LayoutParser model loaded: %s", LAYOUT_MODEL_CONFIG)
        except Exception:  # network error, missing package ...
            logger.exception("LayoutParser unavailable - using OpenCV block detection only")
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

    # ------------------------------------------------------------------ API
    def detect(self, color_bgr: np.ndarray, binary: np.ndarray) -> List[Block]:
        """Return text blocks in reading order.

        Note: the PubLayNet model was trained on English research papers. On
        Urdu/Arabic book photos it often returns nothing, or labels the whole
        page "Figure"; those regions are then handled by the OpenCV paragraph
        detector, so the result never depends on the model alone.
        """
        h, w = binary.shape[:2]
        blocks: List[Block] = []

        if self.model is not None:
            rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)  # model expects RGB
            with self._lock:
                layout = self.model.detect(rgb)
            for tb in layout:
                if tb.type == "Figure":
                    continue  # no text to OCR; any real text in it is picked up below
                x1, y1, x2, y2 = (int(round(v)) for v in tb.coordinates)
                box = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                btype = "Text" if tb.type == "Table" else tb.type
                # The model may put several paragraphs in one box: split it
                # at paragraph gaps so the page structure is preserved.
                sub = ip.detect_paragraphs(ip.crop(binary, box))
                for sx1, sy1, sx2, sy2 in sub or [(0, 0, box[2] - box[0], box[3] - box[1])]:
                    blocks.append(Block(box=(box[0] + sx1, box[1] + sy1, box[0] + sx2, box[1] + sy2),
                                        type=btype, score=float(tb.score)))
            blocks = self._suppress_duplicates(blocks)

        # Safety net: text not covered by any model block.
        uncovered = binary.copy()
        for b in blocks:
            uncovered[b.box[1]:b.box[3], b.box[0]:b.box[2]] = 255
        blocks.extend(Block(box=box, type="Text", score=0.0)
                      for box in ip.detect_paragraphs(uncovered))

        # Tighten every box to its ink; drop empty or hair-thin artefacts.
        tightened = []
        for b in blocks:
            ink = ip.ink_bounds(ip.crop(binary, b.box))
            if ink is None:
                continue
            x1, y1 = b.box[0] + ink[0], b.box[1] + ink[1]
            b.box = (x1, y1, x1 + ink[2] - ink[0], y1 + ink[3] - ink[1])
            if min(b.box[2] - b.box[0], b.box[3] - b.box[1]) >= 8:
                tightened.append(b)
        return self._reading_order(tightened)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _suppress_duplicates(blocks: List[Block]) -> List[Block]:
        """Drop blocks that mostly lie inside a higher-scoring block."""
        kept: List[Block] = []
        for b in sorted(blocks, key=lambda b: b.score, reverse=True):
            if all(ip.boxes_overlap_ratio(b.box, k.box) < 0.7 for k in kept):
                kept.append(b)
        return kept

    @staticmethod
    def _reading_order(blocks: List[Block]) -> List[Block]:
        """Top-to-bottom; blocks side by side on the same row are read
        right-to-left (RTL). A block joins a row only if it overlaps *every*
        block already in that row by more than half of the smaller height."""
        def v_overlap(a: Block, b: Block) -> bool:
            ov = min(a.box[3], b.box[3]) - max(a.box[1], b.box[1])
            return ov > 0.5 * min(a.box[3] - a.box[1], b.box[3] - b.box[1])

        blocks = sorted(blocks, key=lambda b: b.box[1])
        rows: List[List[Block]] = []
        for b in blocks:
            if rows and all(v_overlap(b, other) for other in rows[-1]):
                rows[-1].append(b)
            else:
                rows.append([b])
        ordered = []
        for row in rows:
            ordered.extend(sorted(row, key=lambda b: b.box[2], reverse=True))
        return ordered


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
    because they sit above/below the line centre and close to their letter.
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
    if _NUMBERED_RE.match(stripped):
        return stripped, False  # keep numbering as printed; block becomes a list
    return text, False


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
        """Plain text that preserves the page structure:
        blank line between blocks, one line per printed line, bullets as '• '."""
        return "\n\n".join(t for t in (b.text() for b in self.blocks) if t)


class OcrPipeline:
    """Loads all models once and processes images end-to-end."""

    # Line images are padded with white so glyphs do not touch the border.
    LINE_PAD = 4

    def __init__(self, device: str = OCR_DEVICE):
        """``device``: "cpu" or "gpu" (PyTorch 'cuda' / Paddle 'gpu:0')."""
        t0 = time.perf_counter()
        use_gpu = device == "gpu"
        self.urdu = UTRNetRecognizer(device="cuda" if use_gpu else "cpu")
        self.arabic = PaddleArabicRecognizer(device="gpu:0" if use_gpu else "cpu")
        self.layout = LayoutAnalyzer()
        # Two workers: one per engine, so both read the same lines concurrently.
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ocr")
        logger.info("OCR pipeline ready in %.1f s", time.perf_counter() - t0)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ API
    def process(self, image_bgr: np.ndarray) -> PipelineResult:
        t0 = time.perf_counter()
        turns = self._detect_orientation(image_bgr)
        pre = ip.preprocess(image_bgr, quarter_turns=turns)
        blocks = self.layout.detect(pre.color, pre.binary)

        # 1. Segment every block into line images.
        jobs: List[Tuple[Block, ip.Box, np.ndarray, bool]] = []
        for block in blocks:
            for box, gray_line, bullet in self._lines_of(block, pre):
                jobs.append((block, box, gray_line, bullet))

        # 2. Send ALL lines to BOTH engines simultaneously.
        line_images = [j[2] for j in jobs]
        fut_urdu = self._executor.submit(self.urdu.recognize_batch, line_images)
        fut_arabic = self._executor.submit(self.arabic.recognize_batch, line_images)
        urdu_results, arabic_results = fut_urdu.result(), fut_arabic.result()

        # 3. Route each line by confidence.
        for (block, box, _, glyph_bullet), u, a in zip(jobs, urdu_results, arabic_results):
            accepted, _ = route_by_confidence(u, a)
            text, is_bullet = clean_bullet_text(accepted.text, glyph_bullet)
            block.lines.append(RoutedLine(
                box=box, text=text, language=accepted.language if text else "unknown",
                engine=accepted.engine if text else "", confidence=accepted.confidence,
                candidates={LANG_URDU: u, LANG_ARABIC: a}, is_bullet=is_bullet))

        # 4. Block type: a block made of bullet / numbered lines is a list.
        for block in blocks:
            texts = [l for l in block.lines if l.text]
            listy = sum(1 for l in texts if l.is_bullet or _NUMBERED_RE.match(l.text))
            if texts and listy >= max(1, len(texts) // 2) and block.type == "Text":
                block.type = "List"
        blocks = [b for b in blocks if any(l.text for l in b.lines)]

        ms = int((time.perf_counter() - t0) * 1000)
        logger.info("processed %d blocks / %d lines in %d ms (rotated %d deg)",
                    len(blocks), len(jobs), ms, turns * 90)
        return PipelineResult(pre=pre, blocks=blocks, layout_engine=self.layout.engine,
                              processing_ms=ms, rotation=turns * 90)

    # ------------------------------------------------------------ helpers
    # Lines sampled per candidate orientation for the 0/180 decision.
    ORIENTATION_SAMPLE_LINES = 5

    def _detect_orientation(self, image_bgr: np.ndarray) -> int:
        """Return how many CCW quarter turns make the page upright.

        1. Projection profiles decide whether text lines currently run
           horizontally (candidates 0 / 180 deg) or vertically (90 / 270).
        2. The two remaining candidates differ by 180 deg, which profiles
           cannot tell apart. Both are read with the fast PaddleOCR Arabic-
           script recogniser on a few of the widest lines; upside-down
           Arabic-script text gets a much lower confidence.
        """
        gray, binary = ip.quick_binary(image_bgr)
        candidates = (1, 3) if ip.text_runs_vertically(binary) else (0, 2)

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
    def _lines_of(self, block: Block, pre: ip.PreprocessResult):
        """Yield (line_box, padded grayscale line image, has_bullet_glyph)."""
        bx1, by1, _, _ = block.box
        block_bin = ip.crop(pre.binary, block.box)
        for top, bottom in ip.segment_lines(block_bin):
            band = block_bin[top:bottom]
            ink = ip.ink_bounds(band)
            if ink is None:
                continue
            # Line box in page coordinates, trimmed horizontally to its ink.
            box = (bx1 + ink[0], by1 + top, bx1 + ink[2], by1 + bottom)
            if box[3] - box[1] < 8 or box[2] - box[0] < 8:
                continue
            gray = ip.crop(pre.gray, box)
            gray = cv2.copyMakeBorder(gray, self.LINE_PAD, self.LINE_PAD, self.LINE_PAD,
                                      self.LINE_PAD, cv2.BORDER_CONSTANT, value=255)
            yield box, gray, detect_bullet_glyph(ip.crop(pre.binary, box))
