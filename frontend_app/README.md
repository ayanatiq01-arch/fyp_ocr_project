# frontend_app — Flutter client

Document scanner (or camera) → upload the whole page to the FastAPI backend → RTL text + layout boxes.

**Scan page (best quality)** opens Google's ML Kit Document Scanner (`google_mlkit_document_scanner`).

- **What it does:** it detects the page edges, flattens the perspective and removes shadows and stains before the page is sent for OCR.
- **Requirements:** Android only. It needs Google Play services and no camera permission.
- **Fallback:** on iOS, or if the scanner is unavailable, the app uses the plain camera.

| File | Purpose |
|---|---|
| `lib/main.dart` | UI, camera/gallery capture, results + bounding-box overlay |
| `lib/image_cropper.dart` | Interactive paragraph selection (`image_cropper` package) |
| `lib/api_service.dart` | Multipart `POST /api/v1/ocr`, typed JSON models |
| `test/api_service_test.dart` | Parses a real backend response (keeps Dart models in sync with the API) |

Built with Flutter 3.47.5 / Dart 3.13.4.

## Platform configuration (already applied)

- **Android** (`android/app/src/main/AndroidManifest.xml`):
  - `INTERNET` permission.
  - `usesCleartextTraffic="true"`, so the app can reach the backend over plain http.
  - The `UCropActivity` entry that `image_cropper` requires.
- **iOS** (`ios/Runner/Info.plist`):
  - Camera and photo-library usage descriptions.
  - `NSAllowsArbitraryLoads`, which allows the http backend during development.

## Build & run

The backend must be running (see `../backend_api/README.md`).

```bash
flutter pub get
flutter test                      # unit tests
flutter analyze                   # lints

# Real phone on the same Wi-Fi as the PC running the backend:
flutter build apk --release --dart-define=API_BASE_URL=http://<PC-LAN-IP>:8000
#   -> build/app/outputs/flutter-apk/app-release.apk

# Android emulator (10.0.2.2 = the host PC):
flutter run
```

To change the server address at runtime, tap ⚙ in the app bar. The dot next
to it turns green when `GET /health` succeeds.

iOS builds need macOS with Xcode.

The release APK is signed with the debug key, as Flutter's template does.
That's fine for testing and demos. Before publishing to the Play Store,
configure a real signing key.
