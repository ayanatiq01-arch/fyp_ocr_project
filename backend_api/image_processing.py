"""
image_processing.py
===================

OpenCV pre-processing for photographs of old printed books.

The pipeline turns a raw phone photo into:

* a **clean grayscale** image (shadow-normalised, denoised, deskewed) that is
  fed to the OCR models, and
* a **binary** image (black text on white) that is used for skew estimation,
  layout fallback and line segmentation.

Every geometric change (down-scaling + rotation) is recorded as a single 2x3
affine matrix so that bounding boxes found on the processed image can be mapped
back to the coordinates of the image the client uploaded.

Public API
----------
decode_image(data)              -> BGR ndarray (EXIF orientation applied)
denoise(gray)                   -> denoised grayscale
remove_shadows(gray)            -> illumination-normalised grayscale
adaptive_binarize(gray)         -> binary image (text = 0, paper = 255)
estimate_skew_angle(binary)     -> angle in degrees
deskew(image, angle)            -> rotated image + affine matrix
preprocess(image_bgr)           -> PreprocessResult (all of the above chained)
segment_lines(binary_block)     -> list of (y_top, y_bottom) text-line bands
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Phone cameras (Samsung, iPhone) often save HEIC/HEIF; teach Pillow to read it.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:  # optional dependency
    logger.warning("pillow-heif not installed: HEIC/HEIF photos cannot be decoded")

# (x1, y1, x2, y2) in pixels, x2/y2 exclusive.
Box = Tuple[int, int, int, int]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes into a BGR image.

    Phone cameras often store rotation in EXIF instead of rotating pixels, so
    the image is decoded with Pillow and ``exif_transpose`` is applied first.

    Raises:
        ValueError: if the bytes are not a readable image.
    """
    try:
        pil = Image.open(io.BytesIO(data))
        pil = ImageOps.exif_transpose(pil).convert("RGB")
    except Exception as exc:  # Pillow raises several exception types
        raise ValueError(f"Could not decode image: {exc}") from exc
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _odd(value: int, minimum: int = 3) -> int:
    """Return ``value`` forced to an odd integer >= ``minimum`` (kernel sizes)."""
    value = max(int(value), minimum)
    return value if value % 2 == 1 else value + 1


# --------------------------------------------------------------------------- #
# 1. Denoising
# --------------------------------------------------------------------------- #
def denoise(gray: np.ndarray, strength: float = 7.0) -> np.ndarray:
    """Remove camera sensor noise and paper grain.

    Non-local means keeps stroke edges sharp, which matters for the thin
    connecting strokes and dots (nuqta) of Nastaliq/Naskh script.

    Args:
        gray: single-channel uint8 image.
        strength: filter strength ``h``; higher removes more noise but can
            erase faint diacritics. 5-10 works well for phone photos.
    """
    if strength <= 0:
        return gray
    # searchWindowSize 11 instead of OpenCV's default 21: ~3.5x faster
    # (cost grows with the window area) with no visible loss on text.
    return cv2.fastNlMeansDenoising(gray, None, h=float(strength),
                                    templateWindowSize=7, searchWindowSize=11)


def remove_small_specks(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Delete isolated black blobs smaller than ``min_area`` pixels.

    ``min_area`` must stay small: Urdu/Arabic dots are tiny connected
    components and must survive this step.
    """
    ink = (binary == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    small = np.where(stats[:, cv2.CC_STAT_AREA] < min_area)[0]
    small = small[small != 0]  # label 0 is the background
    if small.size == 0:
        return binary
    cleaned = binary.copy()
    cleaned[np.isin(labels, small)] = 255
    return cleaned


def remove_line_artifacts(binary: np.ndarray, min_ratio: float = 15.0) -> np.ndarray:
    """Delete long, thin, straight components: page edges, book-fold
    shadows and printed ruling lines.

    These are not text, and a long vertical edge would otherwise glue every
    text line it touches into one block. Letters are safe: even a tall alef
    or a stretched kashida stroke is far from a 15:1 thin bar, and kashida is
    connected to its letters so the component is not a bare line.
    """
    h, w = binary.shape[:2]
    ink = (binary == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    max_thickness = max(6, min(h, w) // 100)
    min_length = 0.05 * min(h, w)
    remove = []
    for i in range(1, count):
        cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        thin, length = min(cw, ch), max(cw, ch)
        if length >= min_length and thin <= max_thickness and length / max(thin, 1) >= min_ratio:
            remove.append(i)
    if not remove:
        return binary
    cleaned = binary.copy()
    cleaned[np.isin(labels, remove)] = 255
    return cleaned


# --------------------------------------------------------------------------- #
# 2. Shadow removal + adaptive binarisation
# --------------------------------------------------------------------------- #
def remove_shadows(gray: np.ndarray) -> np.ndarray:
    """Flatten uneven lighting (page curl shadows, yellowed paper).

    The paper background is estimated by dilating (removes dark text) and
    median-blurring the image; dividing it out leaves text on an evenly lit,
    near-white page. Faded ink gets stretched back to full contrast.
    """
    h, w = gray.shape[:2]
    dilate_k = _odd(min(h, w) // 150, minimum=7)
    blur_k = _odd(min(h, w) // 40, minimum=21)

    background = cv2.dilate(gray, np.ones((dilate_k, dilate_k), np.uint8))
    background = cv2.medianBlur(background, blur_k)

    # gray / background, scaled to 0..255 (division avoids the halo that
    # plain subtraction produces around thick strokes).
    normalized = cv2.divide(gray, background, scale=255)
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)


def adaptive_binarize(gray: np.ndarray, block_size: int | None = None,
                      c: int = 12) -> np.ndarray:
    """Local (adaptive) Gaussian thresholding.

    A global threshold fails on old books because brightness varies across
    the page; the adaptive threshold is computed per neighbourhood.

    Args:
        gray: shadow-normalised grayscale image.
        block_size: neighbourhood size (odd). Defaults to ~1/30 of the short
            side, which is a few text-line heights on a typical page photo.
        c: constant subtracted from the local mean; larger = less noise but
            thinner strokes.

    Returns:
        uint8 image with text = 0 (black) and background = 255 (white).
    """
    h, w = gray.shape[:2]
    if block_size is None:
        block_size = _odd(min(h, w) // 30, minimum=15)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, _odd(block_size), c)


# --------------------------------------------------------------------------- #
# 3. Deskewing
# --------------------------------------------------------------------------- #
def _profile_score(ink: np.ndarray, angle: float) -> float:
    """Sharpness of the horizontal projection profile after rotating by ``angle``.

    When text lines are perfectly horizontal, rows alternate between "full of
    ink" and "empty", so the squared differences between neighbouring row sums
    are maximal.
    """
    h, w = ink.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_NEAREST,
                             borderValue=0)
    profile = rotated.sum(axis=1, dtype=np.float64)
    return float(np.sum(np.diff(profile) ** 2))


def estimate_skew_angle(binary: np.ndarray, max_angle: float = 15.0) -> float:
    """Estimate page tilt with a coarse-to-fine projection-profile search.

    Works on the binary image (text = 0). The image is down-scaled to ~1000 px
    wide for speed; angle precision is ~0.1 degrees.

    Returns:
        Angle in degrees to pass to :func:`deskew` (positive = rotate
        counter-clockwise). 0.0 if the image has almost no ink.
    """
    ink = (binary == 0).astype(np.uint8)
    if ink.mean() < 0.002:  # blank crop: nothing to align
        return 0.0

    scale = min(1.0, 1000.0 / max(ink.shape))
    if scale < 1.0:
        ink = cv2.resize(ink, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)

    def search(center: float, span: float, step: float) -> float:
        angles = np.arange(center - span, center + span + 1e-9, step)
        scores = [_profile_score(ink, float(a)) for a in angles]
        return float(angles[int(np.argmax(scores))])

    coarse = search(0.0, max_angle, 1.0)
    fine = search(coarse, 1.0, 0.1)
    return round(fine, 2)


def deskew(image: np.ndarray, angle: float,
           border_value: int | Tuple[int, int, int] = 255
           ) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate ``image`` by ``angle`` degrees, enlarging the canvas so no text
    is clipped at the corners.

    Returns:
        (rotated_image, 2x3 affine matrix mapping input -> output coordinates)
    """
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w = int(round(h * sin + w * cos))
    new_h = int(round(h * cos + w * sin))
    m[0, 2] += new_w / 2 - w / 2
    m[1, 2] += new_h / 2 - h / 2
    rotated = cv2.warpAffine(image, m, (new_w, new_h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=border_value)
    return rotated, m


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessResult:
    """Everything the OCR pipeline needs from pre-processing."""

    color: np.ndarray            # deskewed BGR image (for layout model)
    gray: np.ndarray             # deskewed, cleaned grayscale (for OCR)
    binary: np.ndarray           # deskewed binary, text = 0 (for segmentation)
    skew_angle: float            # degrees applied during deskew
    original_size: Tuple[int, int]  # (width, height) of the uploaded image
    # 2x3 affine: uploaded-image coords -> processed-image coords
    matrix: np.ndarray = field(repr=False, default_factory=lambda: np.eye(2, 3))

    def to_original(self, box: Box) -> Box:
        """Map a box on the processed image back to the uploaded image.

        The four corners are transformed by the inverse affine and the
        axis-aligned bounding rectangle is returned, clipped to the image.
        """
        x1, y1, x2, y2 = box
        inv = cv2.invertAffineTransform(self.matrix)
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                           dtype=np.float64)
        mapped = corners @ inv[:, :2].T + inv[:, 2]
        w, h = self.original_size
        ox1, oy1 = np.floor(mapped.min(axis=0)).astype(int)
        ox2, oy2 = np.ceil(mapped.max(axis=0)).astype(int)
        return (int(np.clip(ox1, 0, w)), int(np.clip(oy1, 0, h)),
                int(np.clip(ox2, 0, w)), int(np.clip(oy2, 0, h)))


# --------------------------------------------------------------------------- #
# Page orientation (0 / 90 / 180 / 270 degrees)
# --------------------------------------------------------------------------- #
def rotate90(image: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate by ``k`` quarter turns counter-clockwise (like ``np.rot90``).

    Returns:
        (rotated_image, 2x3 affine mapping input -> output coordinates),
        using pixel-edge coordinates so exclusive box edges map exactly.
    """
    k %= 4
    h, w = image.shape[:2]
    m = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    for _ in range(k):
        # One CCW quarter turn of a (w x h) image: (x, y) -> (y, w - x).
        step = np.array([[0, 1, 0], [-1, 0, w], [0, 0, 1]], dtype=np.float64)
        m = step @ m
        w, h = h, w
    return np.ascontiguousarray(np.rot90(image, k)), m[:2]


def quick_binary(image_bgr: np.ndarray, max_side: int = 1200
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Fast, low-resolution (gray, binary) pair for orientation checks.

    Skips denoising (the slow step); good enough to find text lines.
    """
    scale = min(1.0, max_side / float(max(image_bgr.shape[:2])))
    small = cv2.resize(image_bgr, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else image_bgr
    gray = remove_shadows(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    binary = remove_line_artifacts(cv2.medianBlur(adaptive_binarize(gray), 3))
    return gray, binary


def _line_band_score(ink: np.ndarray, max_skew: float = 15.0) -> float:
    """How strongly the ink forms horizontal bands (text lines).

    Coefficient of variation (std / mean) of the smoothed row profile:
    horizontal text lines alternate full / empty rows (high value); text
    running vertically spreads ink evenly over the rows (low value). The best
    value over small tilts is used so a skewed photo is not penalised.
    (Squared row-to-row differences do NOT work: the gaps between Nastaliq
    words create many small jumps across columns too.)
    """
    h, w = ink.shape
    best = 0.0
    for angle in np.arange(-max_skew, max_skew + 1e-9, 3.0):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), float(angle), 1.0)
        rotated = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        profile = np.convolve(rotated.sum(axis=1), np.ones(9) / 9, mode="same")
        mean = profile.mean()
        if mean > 0:
            best = max(best, float(profile.std() / mean))
    return best


def text_runs_vertically(binary: np.ndarray) -> bool:
    """True if the text lines run top-to-bottom (page photographed 90° off).

    The ink is resized to a square so both directions are measured on equal
    terms, then the line-band score of the image is compared with that of
    its transpose.
    """
    ink = (binary == 0).astype(np.float32)
    if ink.mean() < 0.002:
        return False
    ink = cv2.resize(ink, (600, 600), interpolation=cv2.INTER_AREA)
    horizontal = _line_band_score(ink)
    vertical = _line_band_score(np.ascontiguousarray(ink.T))
    logger.debug("orientation bands: horizontal=%.3f vertical=%.3f", horizontal, vertical)
    return vertical > horizontal


def preprocess(image_bgr: np.ndarray, *, max_side: int = 2400,
               denoise_strength: float = 7.0,
               max_skew: float = 15.0,
               quarter_turns: int = 0) -> PreprocessResult:
    """Run the full cleaning pipeline on one uploaded image.

    Steps:
        0. Undo a 90/180/270 degree page rotation (``quarter_turns`` CCW,
           decided by the caller - see OcrPipeline._detect_orientation).
        1. Down-scale very large photos (keeps latency predictable).
        2. Grayscale + non-local-means denoising.
        3. Shadow / illumination normalisation.
        4. Adaptive binarisation -> skew estimation.
        5. Rotate colour + grayscale images by the estimated angle.
        6. Re-binarise the straightened image and remove specks.
    """
    orig_h, orig_w = image_bgr.shape[:2]

    # 0. Quarter-turn rotation (recorded in the affine matrix).
    image_bgr, turn_m = rotate90(image_bgr, quarter_turns)

    # 1. Down-scale (recorded in the affine matrix).
    scale = min(1.0, max_side / float(max(orig_h, orig_w)))
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
    scale_m = np.array([[scale, 0, 0], [0, scale, 0]], dtype=np.float64)

    # 2-3. Clean grayscale.
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = denoise(gray, denoise_strength)
    gray = remove_shadows(gray)

    # 4. Estimate skew on a first binarisation.
    angle = estimate_skew_angle(adaptive_binarize(gray), max_skew)

    # 5. Straighten.
    if abs(angle) >= 0.1:
        gray, rot_m = deskew(gray, angle, border_value=255)
        color, _ = deskew(image_bgr, angle, border_value=(255, 255, 255))
    else:
        angle, rot_m, color = 0.0, np.array([[1, 0, 0], [0, 1, 0]], np.float64), image_bgr

    # 6. Final binary image + speck removal (area threshold scales with size
    #    but stays well below the area of a printed dot).
    binary = adaptive_binarize(gray)
    binary = cv2.medianBlur(binary, 3)
    min_area = max(3, int(gray.shape[0] * gray.shape[1] / 1_500_000))
    binary = remove_small_specks(binary, min_area)
    binary = remove_line_artifacts(binary)

    # Compose quarter-turn, then scale, then deskew: p' = R * (S * (T * p))
    full = (np.vstack([rot_m, [0, 0, 1]]) @ np.vstack([scale_m, [0, 0, 1]])
            @ np.vstack([turn_m, [0, 0, 1]]))
    logger.debug("preprocess: scale=%.3f angle=%.2f size=%s", scale, angle,
                 gray.shape[::-1])
    return PreprocessResult(color=color, gray=gray, binary=binary,
                            skew_angle=angle, original_size=(orig_w, orig_h),
                            matrix=full[:2])


# --------------------------------------------------------------------------- #
# Helpers used by the OCR pipeline
# --------------------------------------------------------------------------- #
def crop(image: np.ndarray, box: Box, pad: int = 0) -> np.ndarray:
    """Crop ``box`` (optionally padded) from ``image``, clipped to its bounds."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    return image[max(0, y1 - pad):min(h, y2 + pad),
                 max(0, x1 - pad):min(w, x2 + pad)]


def ink_bounds(binary: np.ndarray) -> Box | None:
    """Tight bounding box around all ink in a binary image (None if blank)."""
    ys, xs = np.where(binary == 0)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def segment_lines(binary: np.ndarray, min_line_height: int = 8
                  ) -> List[Tuple[int, int]]:
    """Split a text block into horizontal line bands.

    Uses a smoothed horizontal projection profile. Nastaliq lines overlap
    vertically (descenders of one line reach into the next), so:

    * the cut threshold is relative to the profile peak, not zero, and
    * thin bands (usually detached dots / diacritics) are merged into the
      nearest real line.

    Args:
        binary: block image, text = 0.
        min_line_height: bands shorter than this are never kept on their own.

    Returns:
        List of (y_top, y_bottom) sorted top-to-bottom. Empty if no ink.
    """
    ink = (binary == 0).astype(np.float32)
    profile = ink.sum(axis=1)
    if profile.max() <= 0:
        return []

    # Smooth with a window of roughly a third of a typical line height.
    win = max(3, binary.shape[0] // 60)
    profile = np.convolve(profile, np.ones(win) / win, mode="same")

    threshold = 0.12 * profile.max()
    is_text = profile > threshold

    bands: List[List[int]] = []
    start = None
    for y, flag in enumerate(is_text):
        if flag and start is None:
            start = y
        elif not flag and start is not None:
            bands.append([start, y])
            start = None
    if start is not None:
        bands.append([start, len(is_text)])
    if not bands:
        return []

    # Merge thin bands (diacritics) into their nearest neighbour.
    heights = [b[1] - b[0] for b in bands]
    median_h = float(np.median(heights))
    min_keep = max(min_line_height, 0.4 * median_h)
    merged: List[List[int]] = []
    for band in bands:
        if band[1] - band[0] >= min_keep or not merged:
            merged.append(band)
        else:
            merged[-1][1] = band[1]  # attach to the line above
    # A thin first band may still be alone: attach it to the next line.
    if len(merged) > 1 and merged[0][1] - merged[0][0] < min_keep:
        merged[1][0] = merged[0][0]
        merged.pop(0)

    # Grow each band halfway into the gap on each side so ascenders,
    # descenders and dots that fell below the threshold are included.
    result: List[Tuple[int, int]] = []
    height = binary.shape[0]
    for i, (top, bottom) in enumerate(merged):
        prev_bottom = merged[i - 1][1] if i > 0 else 0
        next_top = merged[i + 1][0] if i + 1 < len(merged) else height
        top = max(0, top - max(2, (top - prev_bottom) // 2 + 2))
        bottom = min(height, bottom + max(2, (next_top - bottom) // 2 + 2))
        result.append((top, bottom))
    return result


def detect_paragraphs(binary: np.ndarray, gap_ratio: float = 0.8) -> List[Box]:
    """Find paragraph-like regions with OpenCV only (no ML model).

    1. **Text lines**: ink is smeared horizontally so the words of a line join
       up; each resulting component is a line fragment.
    2. **Row merge**: fragments that sit on the same row (a bullet dot and its
       text, words separated by wide justified gaps, detached dots/diacritics
       above or below a line) are merged into one line box.
    3. **Paragraph grouping**: consecutive lines are put in the same paragraph
       when the vertical gap between them is smaller than
       ``gap_ratio`` x the median line height and they overlap horizontally.
       A larger gap starts a new paragraph / list / title block.

    Args:
        binary: page image, text = 0.
        gap_ratio: paragraph break threshold relative to line height.

    Returns:
        Paragraph boxes (x1, y1, x2, y2), unordered.
    """
    h, w = binary.shape[:2]
    ink = (binary == 0).astype(np.uint8) * 255
    if cv2.countNonZero(ink) == 0:
        return []

    # 1. Line fragments.
    kx = max(9, w // 40)
    smeared = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3)))
    contours, _ = cv2.findContours(smeared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frags = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        # Undo the dilation margin; drop hair-thin artefacts (page edges,
        # rotation borders) that are not text.
        x1, x2 = x + kx // 2, x + bw - kx // 2
        if x2 - x1 < 3 or bh < 3 or (bh < 6 and x2 - x1 > 20 * bh):
            continue
        frags.append([x1, y, x2, y + bh])
    if not frags:
        return []

    heights = sorted(f[3] - f[1] for f in frags)
    # Median of the taller half = typical line height (dots are tiny fragments).
    line_h = float(np.median(heights[len(heights) // 2:]))

    # 2. Merge fragments on the same row.
    frags.sort(key=lambda f: (f[1] + f[3]) / 2)
    lines: List[List[int]] = []
    for f in frags:
        merged = False
        for ln in lines:
            v_overlap = min(ln[3], f[3]) - max(ln[1], f[1])
            h_gap = max(ln[0], f[0]) - min(ln[2], f[2])   # < 0 when overlapping
            fh = f[3] - f[1]
            small = fh < 0.5 * line_h                      # dot / diacritic / bullet
            same_row = v_overlap > 0.5 * min(fh, ln[3] - ln[1])
            if (same_row or (small and v_overlap > -0.4 * line_h)) and h_gap < 4 * line_h:
                ln[:] = [min(ln[0], f[0]), min(ln[1], f[1]), max(ln[2], f[2]), max(ln[3], f[3])]
                merged = True
                break
        if not merged:
            lines.append(list(f))
    lines = [ln for ln in lines if ln[3] - ln[1] >= 0.35 * line_h]
    lines.sort(key=lambda ln: ln[1])

    # 3. Group lines into paragraphs by vertical gap.
    paragraphs: List[List[int]] = []
    for ln in lines:
        if paragraphs:
            p = paragraphs[-1]
            gap = ln[1] - p[3]
            h_overlap = min(p[2], ln[2]) - max(p[0], ln[0])
            if gap < gap_ratio * line_h and h_overlap > 0:
                p[:] = [min(p[0], ln[0]), min(p[1], ln[1]), max(p[2], ln[2]), max(p[3], ln[3])]
                continue
        paragraphs.append(list(ln))
    return [tuple(p) for p in paragraphs]


def boxes_overlap_ratio(a: Box, b: Box) -> float:
    """Intersection area divided by the area of the smaller box."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def union_box(boxes: Sequence[Box]) -> Box:
    """Smallest box containing all ``boxes``."""
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))
