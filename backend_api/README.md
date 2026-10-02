# backend_api — FastAPI + OCR pipeline

| File | Purpose |
|---|---|
| `main.py` | FastAPI app: `POST /api/v1/ocr/stream` (live NDJSON), `POST /api/v1/ocr`, `GET /health` |
| `ocr_pipeline.py` | Auto-crop (text detection) → rows/blocks/tables → UTRNet ‖ PaddleOCR → **confidence router** → **Gemini vision check** |
| `.env` | `GEMINI_API_KEY=...` (git-ignored, never committed) |
| `test_gemini_corrector.py` | Unit tests for the Gemini step, no network needed (`venv\Scripts\python -m unittest test_gemini_corrector -v`) |
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
4. **Routing.** Every box goes to **UTRNet** and **PaddleOCR** at the same time. The higher confidence wins; the other result is discarded.
5. **AI vision inspection (Gemini Flash).** After the page is read, every box is sent to Google Gemini: the **cropped image of the box together with its raw OCR text**.
   - Gemini compares the two and returns the text exactly as printed: spelling, dots, and missing or extra spaces between words are fixed.
   - The prompt forbids translating, modernising or adding text.
   - The answer is JSON (one entry per box number), so the layout stays intact.
   - Boxes go in batches of 20, sent in parallel.
   - A box is rejected, and its OCR text kept, if Gemini's answer is empty, contains English, or shares less than 65 % with the OCR text. The last case means Gemini rewrote the text instead of correcting it.
   - If Gemini fails, the OCR text is kept and the scan still completes. Failures include no key, no internet, quota used up, or a timeout.
   - The OCR reading is kept as `raw_text`, so the app can show what was changed.
6. **Output.** Results are streamed in reading order as they are ready. At the end, page ornaments are removed and bullets are marked. The formatted text keeps the page structure: a blank line between blocks, TAB between table columns, and `• ` for bullets.

## Setup (Windows, Python 3.11)

```bat
setup_backend.bat
```

The first start downloads these models to `%USERPROFILE%\.paddlex\official_models\` and `%USERPROFILE%\.torch\iopath_cache\`:
- `arabic_PP-OCRv5_mobile_rec` (PaddleOCR recognition)
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
curl -F "file=@page.jpg" -F language=urdu -F ai_correct=false http://localhost:8000/api/v1/ocr
```

Form fields of both OCR endpoints:

| Field | Default | Values |
|---|---|---|
| `file` | — | the page photo (JPEG/PNG/HEIC …) |
| `language` | `mixed` | `mixed` = both engines + confidence router; `urdu` = UTRNet only; `arabic` = PaddleOCR only (faster) |
| `ai_correct` | `true` | Gemini vision inspection of every box (needs `GEMINI_API_KEY` in `.env`) |

### Gemini API key

Put the key in `backend_api/.env`, which git ignores:

```
GEMINI_API_KEY=your-key
```

The free tier allows about **20 requests per day for each model**. A page uses one request per 20 boxes. When a model's daily quota is used up, or the model is overloaded, the next model in `GEMINI_MODELS` takes over.

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
                          "candidates": {"urdu": {...}, "arabic": {...}},
                          "raw_text": "سَمعَ"}, ...]}]}
  ],
  "formatted_text": "…",
  "processing_ms": 80269,
  "ai_correction": "gemini-3.6-flash"
}
```

The stream sends a `layout` event first (all boxes, no text), then one `cell` event per box in reading order with the raw OCR text. Next comes `{"event": "status", "stage": "ai_correction"}` while Gemini checks the boxes, then `done` with the object above. `ai_correction` is the Gemini model that answered, `off`, or `failed: <reason>`. All boxes are `[x1, y1, x2, y2]` in pixels of the **uploaded** image.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `OCR_DEVICE` | `cpu` | `gpu` requires the CUDA builds of PyTorch and PaddlePaddle |
| `ROUTER_URDU_MARGIN` | `0.0` | Calibration offset subtracted from UTRNet's score in the router. Tune it on labelled pages. |
| `MIN_TEXT_CONFIDENCE` | `0.5` | Boxes that neither engine reads above this are dropped as ornaments or noise |
| `TORCH_THREADS` / `OCR_CPU_THREADS` | all cores / half | CPU threads for UTRNet / Paddle |
| `PADDLE_ARABIC_MODEL` / `PADDLE_DET_MODEL` | `arabic_PP-OCRv5_mobile_rec` / `PP-OCRv5_mobile_det` | PaddleOCR 3.x model names |
| `GEMINI_API_KEY` | — | Google AI Studio key; without it the Gemini step is skipped |
| `GEMINI_CORRECTION` | `1` | `0` switches the Gemini step off for every request |
| `GEMINI_MODELS` | `gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-3-flash-preview` | Flash models tried in order (Gemini 1.5 Flash is retired) |
| `GEMINI_BATCH` | `20` | Boxes per request (requests of one page run in parallel) |
| `GEMINI_THINKING` | `low` | Gemini 3 thinking level; `high` was more accurate on the table page (CER 0.30 % vs 1.52 %) but about 3× slower |
| `GEMINI_TIMEOUT` | `240` | Seconds per request |
| `GEMINI_MIN_SIMILARITY` | `0.65` | Answers sharing less than this with the OCR text are rejected |
| `MAX_UPLOAD_MB` | `20` | Upload size limit |
| `KEEP_UPLOADS` | `0` | `1` keeps files in `temp_uploads/` |

## Known limitations (measured, not guessed)

- **Speed (CPU only).** On a dual-core i5 laptop with 8 GB RAM, a dictionary page with 58 boxes gives:
  - layout after about 17 s
  - first text after about 22 s
  - the whole page after about 80 s

  UTRNet's HRNet backbone takes about 0.5–1 s per box. Each box is run at its own width (padded to a multiple of 16) instead of the fixed 400 px, which is about 4× faster than in `read.py`. Close other heavy programs, since the models need about 2 GB of RAM. A CUDA GPU (`OCR_DEVICE=gpu`) removes the bottleneck.
- **Router on Arabic words.** On the test book, UTRNet was *more* confident than PaddleOCR even on Arabic words with harakat. For example, خَلَدَ scored 99.5 % vs 79 %. The text is read correctly, but those words are labelled `urdu`. With the default margin of 0 this follows the specified rule exactly.
- **Gemini correction: measured gain, and its costs.** Character error rate (CER) on the 6 test pages, from the same OCR run with and without the Gemini step (`GEMINI_THINKING=low`):

  | Page | OCR | + Gemini |
  |---|---|---|
  | real1 (table) | 2.13 % | 1.52 % |
  | real2 (table, full photo) | 3.66 % | 1.83 % |
  | real5 (grammar page) | 1.58 % | 0.59 % |
  | page_photo | 1.00 % | 1.00 % |
  | page_photo2 | 0.50 % | 0.00 % |
  | page_clean | 0.00 % | 0.00 % |
  | **Average** | **1.48 %** | **0.82 %** |

  No page got worse. The costs:
  - It adds 5–50 s per page.
  - It needs internet on the server PC.
  - It sends the page crops to Google.
  - The free-tier daily quota is small, so with heavy use it falls back to plain OCR.

  Without an image, Gemini rewrote a test sentence into a different one. That is why the image is always sent and the similarity check exists.
- **LayoutParser.** The only available Paddle model (PubLayNet) was trained on English research papers and usually finds nothing on Urdu/Arabic pages. So page structure comes from the detected text boxes, and LayoutParser only adds Title/List hints.
- **Multi-column prose.** Two side-by-side columns of running text are treated as a two-column table. The text is correct, but reading order goes row by row across both columns.
- **License.** UTRNet code and models are CC BY-NC-SA 4.0, for non-commercial and academic use only.
