# backend_api — FastAPI + OCR pipeline

| File | Purpose |
|---|---|
| `main.py` | FastAPI app: `POST /api/v1/ocr/stream` (live NDJSON), `POST /api/v1/ocr`, `GET /health` |
| `ocr_pipeline.py` | **Gemini page reading** (text + boxes + layout); fallback: auto-crop (text detection) → rows/blocks/tables → UTRNet ‖ PaddleOCR → **confidence router** |
| `.env` | `GEMINI_API_KEY=...` (git-ignored, never committed) |
| `test_gemini_reader.py` | Unit tests for Gemini page reading, no network needed (`venv\Scripts\python -m unittest test_gemini_reader -v`) |
| `image_processing.py` | OpenCV: 90°/180°/270° page orientation, deskew, shadow removal, adaptive binarisation, denoise, line segmentation |
| `UTRNet-High-Resolution-Urdu-Text-Recognition/` | Cloned UTRNet repo + `saved_models/UTRNet-Large/best_norm_ED.pth` |
| `temp_uploads/` | Uploads are stored here while they are processed, then deleted (`KEEP_UPLOADS=1` keeps them) |

## How a page is processed

**Main path: Gemini page reading** (when `GEMINI_API_KEY` is set and the request has `ai_correct=true`).
- The whole photo goes to Google Gemini Flash (vision) in **one request**, resized to at most 2000 px with high media resolution.
- Gemini returns JSON: blocks in reading order (Title / Text / List / Table), each printed line as a row, and table rows split into cells (column 0 = right-most). Every line or cell comes with its **bounding box** and its text exactly as printed, with harakat, digits and punctuation.
- Text, boxes and format therefore come from the same model, so a line is never cut in half. It also works on tilted, sideways or upside-down photos and on low-quality images (see the results below).
- The free tier allows about 20 requests per day per model. When a model's daily quota is used up, or the model is overloaded, the next model in `GEMINI_MODELS` takes over.

**Fallback: local OCR.** Used when Gemini is off or unavailable (no key, no internet, quota used up for every model):

1. **Orientation and cleaning.** The page is turned upright (0/90/180/270°) and straightened by up to ±15°. Shadows are removed, and the page is denoised and binarised.
2. **Auto-crop.** PaddleOCR's text detector (`PP-OCRv5_mobile_det`) finds every text line and table cell. Any ink the detector misses is picked up by an OpenCV fallback.
3. **Layout.**
   - Boxes are chained into rows, each box to its nearest neighbour on the left. This keeps rows intact on curved pages.
   - Rows are grouped into blocks wherever the vertical gap is less than 0.8 line heights.
   - A block becomes a **Table** when most of its rows have column-sized gaps. Columns are found from the right-aligned edges.
   - A lone centred line becomes a **Title**. LayoutParser adds Title and List hints in the background.
4. **Routing.** Every box goes to **UTRNet** and **PaddleOCR** at the same time. The higher confidence wins; the other result is discarded.
5. **Output.** Results are streamed in reading order as they are ready. At the end, page ornaments are removed and bullets are marked.

In both cases the formatted text keeps the page structure: a blank line between blocks, TAB between table columns, and `• ` for bullets.

The response also has **`markdown`**, the page as a Markdown document built from the same blocks (`blocks_to_markdown`):
- A Title becomes a `## heading`.
- Every printed line keeps its own line, joined with a hard line break.
- Bullets become `- ` items.
- Tables become Markdown tables; the first column is the right-most column in the book.
- Blocks are separated by a blank line.
- Markdown characters in the text are escaped.

The app renders this format and uses the same rules for its PDF and Word export.

## Setup (Windows, Python 3.11)

```bat
setup_backend.bat
```

The first start downloads these models to `%USERPROFILE%\.paddlex\official_models\` and `%USERPROFILE%\.torch\iopath_cache\`:
- `arabic_PP-OCRv5_mobile_rec` (PaddleOCR recognition)
- `PP-OCRv5_mobile_det` (PaddleOCR text detection)
- LayoutParser PubLayNet

## Run

Double-click **`start_server.bat`**. It opens a server window, shows the address to use in the app, and starts the server again by itself if it ever stops. Close the window to stop the server.

To start the server automatically every time you log in to Windows, run **`install_autostart.bat`** once. It puts a small launcher in your Startup folder; no admin rights are needed. `uninstall_autostart.bat` undoes it.

- API docs: http://localhost:8000/docs
- A phone on the same Wi-Fi uses `http://<PC-LAN-IP>:8000`. If the phone can't connect, allow TCP port 8000 for the local network in Windows Firewall.
- **The app finds the server by itself.** When the saved address doesn't answer (for example, the router gave the PC a new IP), the app scans the phone's Wi-Fi network on port 8000 and saves the server it finds. Settings also has a **Find server automatically** button.
- The phone still can't connect while the PC is asleep or switched off. Keep the PC awake while scanning.

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
| `ai_correct` | `true` | Read the page with Gemini (needs `GEMINI_API_KEY` in `.env`); `false` = local OCR only |

### Gemini API key

Put the key in `backend_api/.env`, which git ignores:

```
GEMINI_API_KEY=your-key
```

The free tier allows about **20 requests per day for each model**, and each page uses one request. With the five default models that is about 100 pages a day. After that the server reads pages with the local OCR until the quota resets.

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
                          }, ...]}]}
  ],
  "formatted_text": "…",
  "markdown": "## سبق نمبر ۸ کے الفاظ کے معانی

| سَمِعَ | اس نے سنا | … |
| --- | --- | … |
…",
  "processing_ms": 80269,
  "ai_correction": "gemini-3.6-flash"
}
```

With Gemini, the stream sends `{"event": "status", "stage": "ai_reading"}`, then `layout` (the whole page, already read), then `done`. With local OCR, it sends `layout` first (all boxes, no text), then one `cell` event per box in reading order, then `done`. If Gemini fails, a `{"stage": "local_ocr"}` status comes before the local events.

`ai_correction` is the Gemini model that read the page, `off`, or `failed: <reason>` (local OCR was used). Cells read by Gemini have `engine: "Gemini"` and a fixed confidence of 99, because Gemini gives no per-word score. All boxes are `[x1, y1, x2, y2]` in pixels of the **uploaded** image.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `OCR_DEVICE` | `cpu` | `gpu` requires the CUDA builds of PyTorch and PaddlePaddle |
| `ROUTER_URDU_MARGIN` | `0.0` | Calibration offset subtracted from UTRNet's score in the router. Tune it on labelled pages. |
| `MIN_TEXT_CONFIDENCE` | `0.5` | Boxes that neither engine reads above this are dropped as ornaments or noise |
| `TORCH_THREADS` / `OCR_CPU_THREADS` | all cores / half | CPU threads for UTRNet / Paddle |
| `PADDLE_ARABIC_MODEL` / `PADDLE_DET_MODEL` | `arabic_PP-OCRv5_mobile_rec` / `PP-OCRv5_mobile_det` | PaddleOCR 3.x model names |
| `GEMINI_API_KEY` | — | Google AI Studio key; without it pages are read by the local OCR |
| `GEMINI_CORRECTION` | `1` | `0` switches Gemini off for every request |
| `GEMINI_MODELS` | `gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-3-flash-preview` | Flash models tried in order (Gemini 1.5 Flash is retired) |
| `GEMINI_MAX_SIDE` | `2000` | Long side (px) of the photo sent to Gemini |
| `GEMINI_THINKING` | `low` | Gemini 3 thinking level |
| `GEMINI_TIMEOUT` | `60` | Seconds for the whole page before falling back to local OCR |
| `GEMINI_HEDGE_AFTER` | `5` | A model that hasn't answered after this many seconds gets the next model as a parallel backup |
| `GEMINI_ROUNDS` | `3` | Passes over all models when they are only overloaded (HTTP 503) |
| `MAX_UPLOAD_MB` | `20` | Upload size limit |
| `KEEP_UPLOADS` | `0` | `1` keeps files in `temp_uploads/` |

## Known limitations (measured, not guessed)

- **Speed (CPU only).** On a dual-core i5 laptop with 8 GB RAM, a dictionary page with 58 boxes gives:
  - layout after about 17 s
  - first text after about 22 s
  - the whole page after about 80 s

  UTRNet's HRNet backbone takes about 0.5–1 s per box. Each box is run at its own width (padded to a multiple of 16) instead of the fixed 400 px, which is about 4× faster than in `read.py`. Close other heavy programs, since the models need about 2 GB of RAM. A CUDA GPU (`OCR_DEVICE=gpu`) removes the bottleneck.
- **Router on Arabic words.** On the test book, UTRNet was *more* confident than PaddleOCR even on Arabic words with harakat. For example, خَلَدَ scored 99.5 % vs 79 %. The text is read correctly, but those words are labelled `urdu`. With the default margin of 0 this follows the specified rule exactly.
- **Accuracy: Gemini page reading vs local OCR.** Character error rate (CER), letters only, harakat, punctuation and spaces ignored:

  | Page | Local OCR | Gemini page reading |
  |---|---|---|
  | real1 (table) | 2.13 % | 0.00 % |
  | real2 (table, book in hand) | 3.66 % | 0.61 % |
  | real5 (grammar page) | 1.58 % | 0.40 % |
  | page_photo / page_photo2 / page_clean | 1.00 / 0.50 / 0.00 % | 0.00 / 0.00 / 0.00 % |
  | real1, degraded (low resolution, blur, noise) | 6.10 % | 0.00 % |
  | page_photo, degraded | 1.49 % | 0.00 % |
  | real5, degraded (256 px wide, barely readable) | 48.71 % | 8.12 % |
  | real5, turned 90° | — | 3.76 % |
  | page_photo, upside down | — | 0.00 % |

  Line and cell boxes were checked visually on every page, including the sideways one, and fit the printed lines. The costs:
  - 5–80 s per page, depending on how busy Google's servers are.
  - The server PC needs internet.
  - The page photo is sent to Google.
  - The free-tier daily quota.
- **Speed and Gemini 503 "high demand".** When a model answers straight away, a page takes about 6–15 s. Google's Flash models are often overloaded, so:
  - a model that fails (busy, quota used up, or a cut-off answer) hands over to the next model immediately;
  - a model that hasn't answered after 5 s (a "busy" reply alone can take 11 s) gets the next model as a parallel backup, and the first good answer is used;
  - if every model is only busy, they are tried again (up to 3 passes, within 60 s).

  Only after that is the page read by the local OCR, which takes about 80 s on this PC. On 2026-10-05 Google was overloaded for most of the day: answers took 7–36 s, and some pages needed the retry passes.
- **LayoutParser.** The only available Paddle model (PubLayNet) was trained on English research papers and usually finds nothing on Urdu/Arabic pages. So page structure comes from the detected text boxes, and LayoutParser only adds Title/List hints.
- **Multi-column prose.** Two side-by-side columns of running text are treated as a two-column table. The text is correct, but reading order goes row by row across both columns.
- **License.** UTRNet code and models are CC BY-NC-SA 4.0, for non-commercial and academic use only.
