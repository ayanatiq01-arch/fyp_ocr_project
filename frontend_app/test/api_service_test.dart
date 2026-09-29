// Parses a response captured from the real backend (POST /api/v1/ocr) to
// make sure the Dart models stay in sync with backend_api/main.py.

import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:frontend_app/api_service.dart';

const _sampleResponse = r'''
{
  "request_id": "5ea6799b487347379b1a173cdbefa52a",
  "image": {"width": 1100, "height": 880},
  "skew_angle": -3.3,
  "layout_engine": "layoutparser-paddledetection",
  "blocks": [
    {
      "id": 0, "type": "List", "bbox": [753, 393, 1026, 611], "language": "urdu",
      "text": "• میرتقی میرکی شاعری",
      "lines": [
        {
          "bbox": [761, 396, 1018, 480], "text": "میرتقی میرکی شاعری",
          "language": "urdu", "engine": "UTRNet", "confidence": 94.84, "is_bullet": true,
          "candidates": {
            "urdu": {"text": "‘میرتقی میرکی شاعری", "confidence": 94.84},
            "arabic": {"text": "هيرتى مري شاعرى", "confidence": 66.1}
          }
        }
      ]
    }
  ],
  "formatted_text": "• میرتقی میرکی شاعری",
  "processing_ms": 41347
}
''';

void main() {
  test('OcrResult.fromJson parses a real backend response', () {
    final result = OcrResult.fromJson(jsonDecode(_sampleResponse) as Map<String, dynamic>);

    expect(result.imageWidth, 1100);
    expect(result.imageHeight, 880);
    expect(result.skewAngle, closeTo(-3.3, 1e-9));
    expect(result.processingMs, 41347);
    expect(result.blocks, hasLength(1));

    final block = result.blocks.single;
    expect(block.type, 'List');
    expect(block.bbox.x2, 1026);

    final line = block.lines.single;
    expect(line.text, 'میرتقی میرکی شاعری');
    expect(line.engine, 'UTRNet');
    expect(line.language, 'urdu');
    expect(line.candidates['arabic']!.confidence, closeTo(66.1, 1e-9));
  });

  test('ApiService strips a trailing slash from the base URL', () {
    final api = ApiService(baseUrl: 'http://10.0.2.2:8000/');
    expect(api.baseUrl, 'http://10.0.2.2:8000');
    api.dispose();
  });
}
