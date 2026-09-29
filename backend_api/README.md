# backend_api — FastAPI + OCR pipeline

| File | Purpose |
|---|---|
| `main.py` | FastAPI app: `POST /api/v1/ocr/stream` (live NDJSON), `POST /api/v1/ocr`, `GET /health` |
| `ocr_pipeline.py` | Auto-crop (text detection) → rows/blocks/tables → UTRNet ‖ Kraken → **confidence router** |
| `image_processing.py` | OpenCV: 90°/180°/270° page orientation, deskew, shadow removal, adaptive binarisation, denoise, line segmentation |
| `UTRNet-High-Resolution-Urdu-Text-Recognition/` | Cloned UTRNet repo + `saved_models/UTRNet-Large/best_norm_ED.pth` |
| `temp_uploads/` | Uploads are stored here while they are processed, then deleted (`KEEP_UPLOADS=1` keeps them) |

## How a page is processed

1. **Orientation and cleaning.** The page is turned upright (0/90/180/270°) and straightened by up to ±15°. Shadows are removed, and the page is denoised and binarised.
2. **Auto-crop.** PaddleOCR's text detector (`PP-OCRv5_mobile_det`) finds every text line and table cell, so the user photographs the whole page and never crops by hand. Any ink the detector misses is picked up by an OpenCV fallback.
3. **Layout.**
   - Boxes are chained into rows, each box to its nearest neighbour on the left. This keeps rows intact on curved pages.
   - Rows are grouped into blocks wherever the vertical gap is less than 0.8 line heights.
   - A block becomes a **Table** when most of its rows have column-sized gaps. Columns are found from the right-aligned edges.
   - A lone centred line becomes a **Title**. LayoutParser adds Title and List hints in the background.
4. **Routing.** Every box goes to **UTRNet** (Urdu) and **Kraken** (Arabic) at the same time. The higher confidence wins; the other result is discarded.
5. **Output.** Results are streamed in reading order as they are ready. At the end, page ornaments are removed and bullets are marked. The formatted text keeps the page structure: a blank line between blocks, TAB between table columns, and `• ` for bullets.

## Setup (Windows, Python 3.11)

```bat
setup_backend.bat
```

The first start downloads these models to `%USERPROFILE%\.paddlex\official_models\` and `%USERPROFILE%\.torch\iopath_cache\`:
- `kraken_models/arabic_best.mlmodel`: Kraken "Printed Arabic Base Model Trained on the OpenITI Corpus" (DOI 10.5281/zenodo.7050296, CC0), downloaded by `setup_backend.bat`
- `PP-OCRv5_mobile_det` (PaddleOCR text detection)
- LayoutParser PubLayNet

## Run

```bat
venv\Scripts\python main.py
```

- API docs: http://localhost:8000/docs
- A phone on the same Wi-Fi uses `http://<PC-LAN-IP>:8000`. If the phone can't connect, allow TCP port 8000 for the local network in Windows Firewall.

```bash
curl -F "file=@page.jpg" http://localhost:8000/api/v1/ocr            # final JSON
curl -N -F "file=@page.jpg" http://localhost:8000/api/v1/ocr/stream  # live events
```

## Response (abridged)

```json
{
  "image": {"width": 1788, "height": 2819}, "rotation": 0, "skew_angle": -4.4,
  "blocks": [
    {"id": 0, "type": "Title", "columns": 1, "text": "سبق نمبر۸کے الفاظ کےمعانی", "rows": [...]},
    {"id": 1, "type": "Table", "columns": 4,
     "rows": [{"text": "سَمعَ\tاس نے سنا\tشَگَرَ\tاس نے شکرکیا",
               "cells": [{"bbox": [1449, 391, 1648, 505], "column": 0, "text": "سَمعَ",
                          "language": "urdu", "engine": "UTRNet", "confidence": 99.5,
                          "candidates": {"urdu": {...}, "arabic": {...}}}, ...]}]}
  ],
  "formatted_text": "…",
  "processing_ms": 80269
}
```

The stream sends a `layout` event first (all boxes, no text), then one `cell` event per box in reading order, then `done` with the object above. All boxes are `[x1, y1, x2, y2]` in pixels of the **uploaded** image.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `OCR_DEVICE` | `cpu` | `gpu` requires the CUDA builds of PyTorch and PaddlePaddle |
| `ROUTER_URDU_MARGIN` | `0.0` | Calibration offset subtracted from UTRNet's score in the router. Tune it on labelled pages. |
| `MIN_TEXT_CONFIDENCE` | `0.5` | Boxes that neither engine reads above this are dropped as ornaments or noise |
| `TORCH_THREADS` / `OCR_CPU_THREADS` | all cores / half | CPU threads for UTRNet / Paddle |
| `KRAKEN_ARABIC_MODEL` | `kraken_models/arabic_best.mlmodel` | Kraken Arabic recognition model file |
| `PADDLE_DET_MODEL` | `PP-OCRv5_mobile_det` | PaddleOCR 3.x text-detection model |
| `MAX_UPLOAD_MB` | `20` | Upload size limit |
| `KEEP_UPLOADS` | `0` | `1` keeps files in `temp_uploads/` |

## Known limitations (measured, not guessed)

- **Speed (CPU only).** On a dual-core i5 laptop with 8 GB RAM, a dictionary page with 58 boxes gives:
  - layout after about 17 s
  - first text after about 22 s
  - the whole page after about 80 s

  UTRNet's HRNet backbone takes about 0.5–1 s per box. Each box is run at its own width (padded to a multiple of 16) instead of the fixed 400 px, which is about 4× faster than in `read.py`. Close other heavy programs, since the models need about 2 GB of RAM. A CUDA GPU (`OCR_DEVICE=gpu`) removes the bottleneck.
- **Kraken vs. the earlier PaddleOCR Arabic recogniser (measured).** Letter error rate on four test pages (harakat ignored):
  - Kraken: 6.1 %, 8.2 %, 1.0 % and 2.8 %
  - PaddleOCR: 2.1 %, 3.7 %, 1.0 % and 1.6 %

  Kraken is about 2x faster, but it is also confident on Urdu boxes and then misreads them (for example "اس نے سنا" becomes "اس ن نا"). `ROUTER_URDU_MARGIN=-0.1` makes Kraken beat UTRNet by 0.1 before it wins, which lowered the average error from 4.8 % to 3.9 % on these pages. The default stays at 0, the plain rule.
- **LayoutParser.** The only available Paddle model (PubLayNet) was trained on English research papers and usually finds nothing on Urdu/Arabic pages. So page structure comes from the detected text boxes, and LayoutParser only adds Title/List hints.
- **Multi-column prose.** Two side-by-side columns of running text are treated as a two-column table. The text is correct, but reading order goes row by row across both columns.
- **License.** UTRNet code and models are CC BY-NC-SA 4.0, for non-commercial and academic use only.
