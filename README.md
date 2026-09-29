# Bilingual (Urdu & Arabic) OCR Scanner for Historical Books

Final-year project. It digitises photos of old books that mix **Urdu (Nastaliq)** and **Arabic (Naskh)** text:

1. Take a photo in the Flutter app and crop the paragraphs you want.
2. The FastAPI backend cleans the image and detects the layout.
3. Each line is read by **UTRNet** (Urdu) and **Kraken** (Arabic) at the same time.
4. A **confidence-score router** keeps whichever engine is more confident.
5. The app shows the text with its paragraphs and bullet points preserved.

```
fyp_ocr_project/
├── frontend_app/     Flutter app: camera, interactive cropping, results + box overlay
└── backend_api/      FastAPI server, OCR pipeline, OpenCV preprocessing, UTRNet
```

| Part | Details |
|---|---|
| Frontend | [frontend_app/README.md](frontend_app/README.md): Flutter 3.47, `image_picker`, `image_cropper`, `http` |
| Backend | [backend_api/README.md](backend_api/README.md): FastAPI, Kraken 7.1 (Arabic), PaddleOCR 3.7 (text detection), LayoutParser 0.3.4, PyTorch, OpenCV |

## Quick start

```bat
:: Backend (Windows, Python 3.11)
cd backend_api
setup_backend.bat
venv\Scripts\python main.py            :: http://localhost:8000/docs
```

Download the UTRNet-Large weights before the first run. Use the link in
[`backend_api/UTRNet-High-Resolution-Urdu-Text-Recognition/README.md`](backend_api/UTRNet-High-Resolution-Urdu-Text-Recognition/README.md)
and save the file to
`backend_api/UTRNet-High-Resolution-Urdu-Text-Recognition/saved_models/UTRNet-Large/best_norm_ED.pth`.
The file is 190 MB, too large for GitHub, so it is not in this repo.

```bash
# Frontend
cd frontend_app
flutter pub get
flutter build apk --release --dart-define=API_BASE_URL=http://<PC-LAN-IP>:8000
```

A prebuilt Android APK is attached to the repository's **Releases**.

## How the pipeline works

```
photo ─► deskew · shadow removal · adaptive binarisation · denoise   (image_processing.py)
      ─► layout blocks (LayoutParser + OpenCV paragraph detection)
      ─► text lines
      ─► UTRNet (Urdu)  ‖  Kraken arabic_best (Arabic)                 (run concurrently)
      ─► route_by_confidence(): higher confidence wins                  (ocr_pipeline.py)
      ─► JSON: blocks, lines, language, confidence, bounding boxes, formatted text
```

Measured limitations are listed in [backend_api/README.md](backend_api/README.md#known-limitations-measured-not-guessed).

## Credits & licenses

- **UTRNet** by Abdur Rahman, Arjun Ghosh and Chetan Arora ([paper](https://arxiv.org/abs/2306.15782), [repo](https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition)).
  - Included under `backend_api/UTRNet-High-Resolution-Urdu-Text-Recognition/`, which is licensed **CC BY-NC-SA 4.0** (non-commercial use only).
  - One compatibility change for newer PyTorch: `dataset.py` line 21.
- [Kraken](https://github.com/mittagessen/kraken) (Apache-2.0) with the OpenITI printed Arabic model (Benjamin Kiessling, DOI 10.5281/zenodo.7050296, CC0).
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache-2.0, text detection) and [LayoutParser](https://github.com/Layout-Parser/layout-parser) (Apache-2.0).
