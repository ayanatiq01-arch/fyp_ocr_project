# backend_api — FastAPI + OCR pipeline

| File | Purpose |
|---|---|
| `main.py` | FastAPI app: `POST /api/v1/ocr/stream` (live NDJSON), `POST /api/v1/ocr`, `POST /api/v1/correct` (correct a selected part), `POST /api/v1/log` (app error reports, written to `server.log`), `GET /health` |
| `ocr_pipeline.py` | Text detection → rows/blocks/tables → **UTRNet (Urdu) / EasyOCR (Arabic)** → rough draft → **Gemini vision correction** (original image + draft → corrected lines with their boxes and layout → Markdown) |
| `.env` | `GEMINI_API_KEY=...` (git-ignored, never committed) |
| `test_correction.py` | Unit tests for routing, rough-draft Markdown and the Gemini correction step, no network needed (`venv\Scripts\python -m unittest test_correction -v`) |
| `image_processing.py` | OpenCV: 90°/180°/270° page orientation, deskew, shadow removal, adaptive binarisation, denoise, line segmentation |
| `UTRNet-High-Resolution-Urdu-Text-Recognition/` | Cloned UTRNet repo + `saved_models/UTRNet-Large/best_norm_ED.pth` |
| `temp_uploads/` | Uploads are stored here while they are processed, then deleted (`KEEP_UPLOADS=1` keeps them) |

## How a page is processed

1. **Orientation and cleaning.** The text detector runs on the photo as it is.
   - Upright text lines give almost only wide boxes. If fewer than 80 % are wide, the page is also tried turned 90°, and the direction with wide lines wins.
   - 0° vs 180° is then decided by UTRNet's confidence on four lines, read upright and turned.

   The page is then straightened by up to ±15°. Shadows are removed, and the page is denoised and binarised.
2. **Text detection (boxes).** PaddleOCR's text detector (`PP-OCRv5_mobile_det`, detection only) finds every text line and table cell. Any ink the detector misses is picked up by an OpenCV fallback.
   - Two full-width lines are never merged; only pieces of one word are.
   - Where consecutive lines overlap in height, the boundary is placed in the middle of the overlap, so each crop holds exactly one line.
3. **Layout.**
   - Boxes are chained into rows, each box to its nearest neighbour on the left. This keeps rows intact on curved pages.
   - Rows are grouped into blocks; tables get columns; a lone centred line becomes a Title. LayoutParser adds Title and List hints.
4. **Recognition: each box is routed to one engine by script.** PaddleOCR is no longer used for recognition.
   - `language=urdu`: every box goes to **UTRNet**.
   - `language=arabic`: every box goes to **EasyOCR**'s Arabic model. EasyOCR is local and open source (github.com/JaidedAI/EasyOCR).
   - `language=mixed`: UTRNet reads every box first. Boxes whose reading is Arabic (Arabic-only letters such as ك ي ة أ, or vocalised text with harakat, as in Quran verses) are then read by EasyOCR.

   EasyOCR is not run on every box because it takes about **5 s per line** on this dual-core CPU, against about 1–2 s for UTRNet.
5. **Rough draft.** The boxes, sorted top to bottom (Y) and right to left within a line (X), give the rough draft. It has one line per printed line, a blank line between blocks, TAB between table columns and `• ` for bullets. It is streamed to the app while it is being read, and returned as `formatted_text`.
6. **Vision-LLM correction (the final layer).** The **original full image** and the **rough draft** go to Gemini Flash in a single request.
   - The system prompt is exactly the one specified for the project ("You are an expert document reconstruction AI. … Return ONLY the corrected, perfectly formatted Markdown string.").
   - The user part of the request adds two instructions:
     - keep the rough draft's line breaks;
     - keep every printed mark: āyah numbers (۝۹), rukūʿ ؏, waqf marks, quotation marks, footnotes.
   - The answer is **structured JSON** (`responseSchema`): blocks (Title / Text / List / Table), rows (one per printed line) and cells, each with the corrected text and its `box_2d` on the photo (0–1000 scale).
   - The server turns this answer into the page's blocks, rows and boxes (`blocks_from_gemini`), and builds `markdown` from the same blocks. So the **boxes, layout and text all come from the API**, and the app's boxes, selection, copy, book page, PDF and Word export all show the same corrected lines.
   - If Gemini is unavailable (no key, no internet, quota used up for every model), the rough draft and the local boxes are returned, and `ai_correction` says `failed: …`. Selecting part of such a page sends the part to `POST /api/v1/correct`.

## Setup (Windows, Python 3.11)

```bat
setup_backend.bat
```

The first start downloads these models:
- `PP-OCRv5_mobile_det` (PaddleOCR text detection), to `%USERPROFILE%\.paddlex\official_models\`
- EasyOCR's Arabic recognition model, to `%USERPROFILE%\.EasyOCR\model\`
- LayoutParser PubLayNet, to `%USERPROFILE%\.torch\iopath_cache\`

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
| `language` | `mixed` | `mixed` = UTRNet, Arabic boxes re-read by EasyOCR; `urdu` = UTRNet only; `arabic` = EasyOCR only |
| `ai_correct` | `true` | Correct the rough draft with Gemini (needs `GEMINI_API_KEY` in `.env`); `false` = rough draft only |

### Gemini API key

Put the key in `backend_api/.env`, which git ignores:

```
GEMINI_API_KEY=your-key
```

The free tier allows about **20 requests per day for each model**, and each page uses one request. With the default models that is roughly 100–150 pages a day. After that the server returns the uncorrected rough draft until the quota resets.

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
                          "candidates": {"urdu": {...}},
                          }, ...]}]}
  ],
  "formatted_text": "rough draft: one line per printed line …",
  "markdown": "## آلِ عِمْرٰن ۳\n\nلَا رَیْبَ فِیْہِ ؕ اِنَّ اللّٰہَ لَا یُخْلِفُ الْمِیْعَادَ ۝۹ …",
  "processing_ms": 180000,
  "ai_correction": "gemini-3.6-flash"
}
```

The stream sends:
1. `layout` (all boxes, no text);
2. one `cell` event per box in reading order (in `mixed`, an Arabic box is sent again after EasyOCR re-reads it);
3. `{"event": "status", "stage": "ai_correcting"}` while Gemini corrects the draft;
4. `done` with the object above.

`ai_correction` is the Gemini model that corrected the draft, `off`, or `failed: <reason>` (rough draft returned). All boxes are `[x1, y1, x2, y2]` in pixels of the **uploaded** image.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `OCR_DEVICE` | `cpu` | `gpu` requires the CUDA builds of PyTorch and PaddlePaddle |
| `MIN_TEXT_CONFIDENCE` | `0.5` | UTRNet readings below this are dropped as ornaments or noise |
| `EASYOCR_MIN_CONFIDENCE` | `0.005` | The same for EasyOCR, whose scores are on a much lower scale (correct Quran lines scored 0.01–0.3) |
| `TORCH_THREADS` / `OCR_CPU_THREADS` | all cores / half | CPU threads for UTRNet and EasyOCR / the Paddle detector |
| `PADDLE_DET_MODEL` | `PP-OCRv5_mobile_det` | PaddleOCR 3.x text-detection model |
| `GEMINI_API_KEY` | — | Google AI Studio key; without it the rough draft is returned uncorrected |
| `GEMINI_CORRECTION` | `1` | `0` switches the Gemini correction off for every request |
| `GEMINI_MODELS` | `gemini-3.8-flash,…,gemini-3-flash-preview,gemini-flash-lite-latest,gemini-3.5-flash-lite,gemini-3.1-flash-lite` | Models tried in order (Gemini 1.5 Flash is retired). The lite models come last: own daily quota, but they follow the layout less strictly. |
| `GEMINI_MAX_SIDE` | `2000` | Long side (px) of the photo sent to Gemini |
| `GEMINI_THINKING` | `low` | Gemini 3 thinking level |
| `GEMINI_TIMEOUT` | `60` | Seconds for the correction before the rough draft is returned instead |
| `GEMINI_HEDGE_AFTER` | `5` | A model that hasn't answered after this many seconds gets the next model as a parallel backup |
| `GEMINI_ROUNDS` | `3` | Passes over all models when they are only overloaded (HTTP 503) |
| `MAX_UPLOAD_MB` | `20` | Upload size limit |
| `KEEP_UPLOADS` | `0` | `1` keeps files in `temp_uploads/` |

## Known limitations (measured, not guessed)

- **Speed (CPU only).** On this dual-core i5 laptop with 8 GB RAM, after the speed changes below, a full `mixed` page takes **about 96–106 s** (measured on two of the user's pages). Changes:
  - PaddleOCR sets torch to 1 thread; UTRNet / EasyOCR now reset the thread count before every batch;
  - EasyOCR's second low-contrast pass is off (`contrast_ths=0`, about 30% faster);
  - EasyOCR runs on a thread at the same time as UTRNet;
  - the upside-down check (~26 s) and LayoutParser are skipped when Gemini corrects, because Gemini gives the layout and boxes.

  The target of under 90 s is not reached yet: UTRNet takes about 2–3 s per line on this CPU. `language=urdu` (no EasyOCR) is faster. Before the speed changes, the Quran tafsīr page (21 boxes, 7 of them Arabic āyāt, `mixed`) took about **180–190 s**:

  | Step | Time |
  |---|---|
  | orientation + detection + layout | ~43 s |
  | UTRNet, all boxes | ~80 s |
  | EasyOCR, the 7 Arabic boxes | ~47 s |
  | Gemini correction | ~12 s |

  The local OCR is the slow part. A CUDA GPU (`OCR_DEVICE=gpu`) removes most of it. With `language=urdu`, EasyOCR is skipped. With `language=arabic`, UTRNet is skipped, but EasyOCR then reads every box at about 5 s each.
- **EasyOCR vs PaddleOCR on Arabic.** On the āyāt of that Quran page:
  - PaddleOCR's Arabic model returned almost nothing.
  - EasyOCR read the words, without harakat, with letter errors.
  - EasyOCR scores are very low even when the text is right, so they are not compared with UTRNet's. Boxes are routed by script instead.
- **Quality of the final Markdown** (Quran tafsīr page, checked by eye against the photo; the earlier CER test images are no longer on this PC):
  - **Correct:** every printed line on its own line, all āyah numbers (۝۹ ۝۱۰ ۝۱۱ ۝۱۲), waqf marks (ؕ ۚ), the rukūʿ (؏), the closing quotation mark, full harakat, the Urdu text and the footnote.
  - **Not correct:** Gemini's first answer, before the line-break instruction was added, joined the lines into paragraphs. The page number is sometimes misread on this small photo.
- **Gemini 503 "high demand".** Google's Flash models are often overloaded. Hedging is used:
  - a busy model hands over to the next one at once;
  - a model that hasn't answered after 5 s gets a parallel backup;
  - up to 3 passes are made within 60 s.

  After that the rough draft is returned uncorrected.
- **LayoutParser.** The only available Paddle model (PubLayNet) was trained on English research papers and usually finds nothing on Urdu/Arabic pages. So page structure comes from the detected text boxes, and LayoutParser only adds Title/List hints.
- **Multi-column prose.** Two side-by-side columns of running text are treated as a two-column table. The text is correct, but reading order goes row by row across both columns.
- **License.** UTRNet code and models are CC BY-NC-SA 4.0, for non-commercial and academic use only.
