# backend_api — FastAPI + OCR pipeline

| File | Purpose |
|---|---|
| `main.py` | FastAPI app: `POST /api/v1/ocr`, `GET /health` |
| `ocr_pipeline.py` | Layout → lines → UTRNet ‖ PaddleOCR → **confidence router** |
| `image_processing.py` | OpenCV: deskew, shadow removal, adaptive binarisation, denoise, paragraph/line segmentation |
| `UTRNet-High-Resolution-Urdu-Text-Recognition/` | Cloned UTRNet repo + `saved_models/UTRNet-Large/best_norm_ED.pth` |
| `temp_uploads/` | Uploads are stored here while they are processed, then deleted (`KEEP_UPLOADS=1` keeps them) |

## Setup (Windows, Python 3.11)

```bat
setup_backend.bat
```

The first start downloads two models, about 250 MB in total:
- `arabic_PP-OCRv5_mobile_rec` goes to `%USERPROFILE%\.paddlex\official_models\`.
- LayoutParser PubLayNet goes to `%USERPROFILE%\.torch\iopath_cache\`.

## Run

```bat
venv\Scripts\python main.py
```

- API docs: http://localhost:8000/docs
- A phone on the same Wi-Fi uses `http://<PC-LAN-IP>:8000`. Allow Python through Windows Firewall when Windows asks.

```bash
curl -F "file=@page.jpg" http://localhost:8000/api/v1/ocr
```

## Response (abridged)

```json
{
  "request_id": "5ea6…",
  "image": {"width": 1100, "height": 880},
  "skew_angle": -3.3,
  "layout_engine": "layoutparser-paddledetection",
  "blocks": [{
    "id": 0, "type": "List", "bbox": [753, 393, 1026, 611], "language": "urdu",
    "text": "• میرتقی میرکی شاعری\n• مرزا غالب کے خطوط",
    "lines": [{
      "bbox": [761, 396, 1018, 480], "text": "میرتقی میرکی شاعری",
      "language": "urdu", "engine": "UTRNet", "confidence": 94.84, "is_bullet": true,
      "candidates": {"urdu": {"text": "…", "confidence": 94.84},
                     "arabic": {"text": "…", "confidence": 66.1}}
    }]
  }],
  "formatted_text": "…paragraphs separated by blank lines, bullets as •…",
  "processing_ms": 41347
}
```

All boxes are `[x1, y1, x2, y2]` in pixels of the **uploaded** image. Deskewing is undone before the boxes are returned.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `OCR_DEVICE` | `cpu` | `gpu` requires the CUDA builds of PyTorch and PaddlePaddle |
| `ROUTER_URDU_MARGIN` | `0.0` | Calibration offset subtracted from UTRNet's score in the router. Tune it on labelled pages. |
| `TORCH_THREADS` / `OCR_CPU_THREADS` | all cores / half | CPU threads for UTRNet / Paddle |
| `PADDLE_ARABIC_MODEL` | `arabic_PP-OCRv5_mobile_rec` | Any PaddleOCR 3.x recognition model name |
| `LAYOUT_SCORE_THRESHOLD` | `0.5` | LayoutParser detection threshold |
| `MAX_UPLOAD_MB` | `20` | Upload size limit |
| `KEEP_UPLOADS` | `0` | `1` keeps files in `temp_uploads/` |

## Known limitations (measured, not guessed)

- **Speed:** on a dual-core i5 (CPU only), UTRNet's HRNet backbone takes about 3 s per text line. A page with 8 lines takes about 35–40 s. A CUDA GPU (`OCR_DEVICE=gpu`) removes this bottleneck.
- **LayoutParser model:** the PubLayNet model was trained on English research papers. On Urdu/Arabic pages it usually returns nothing, or a single "Figure". `detect_paragraphs()` in `image_processing.py` (lines first, then paragraph grouping by vertical gap) covers that case, and it also splits any large model block into its paragraphs.
- **Router bias:** UTRNet is often *more* confident than PaddleOCR even on Arabic Naskh lines. On test pages it scored 96–99 % against PaddleOCR's 93–97 %. With the default margin of 0 those lines are labelled `urdu`, although the text itself is read correctly apart from ی/ي. On Urdu lines PaddleOCR trails by 20 points or more. So a margin of about 0.05–0.10 fixed the Arabic labels on the test pages. Validate it on real scans before relying on it.
- **Line width:** UTRNet squeezes each line to 32×400 px, as in its `read.py`. Very long lines lose some resolution.
- **License:** UTRNet code and models are CC BY-NC-SA 4.0, for non-commercial and academic use only.
