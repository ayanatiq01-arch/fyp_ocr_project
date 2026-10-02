// Parses responses shaped exactly like the backend's (POST /api/v1/ocr and
// the NDJSON stream) to keep the Dart models in sync with backend_api/main.py.

import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:frontend_app/api_service.dart';

const _cell1 = '{"bbox": [1300, 300, 1500, 400], "column": 0, "text": "سَمِعَ", '
    '"language": "urdu", "engine": "UTRNet", "confidence": 99.5, '
    '"candidates": {"urdu": {"text": "سَمِعَ", "confidence": 99.5}, '
    '"arabic": {"text": "سَمع", "confidence": 79.0}}}';
const _cell2 = '{"bbox": [900, 300, 1200, 400], "column": 1, "text": "اس نے سنا", '
    '"language": "urdu", "engine": "UTRNet", "confidence": 97.0, '
    '"candidates": {"urdu": {"text": "اس نے سنا", "confidence": 97.0}, '
    '"arabic": {"text": "اس سا", "confidence": 50.0}}}';
const _unread = '{"bbox": [900, 300, 1200, 400], "column": 1, "text": "", '
    '"language": "unknown", "engine": "", "confidence": 0.0, "candidates": {}}';

String _page(String cells) => '''
{
  "request_id": "abc", "image": {"width": 1788, "height": 2819},
  "rotation": 0, "skew_angle": -4.4, "layout_engine": "layoutparser-paddledetection",
  "blocks": [{"id": 0, "type": "Table", "bbox": [900, 300, 1500, 400], "columns": 2,
              "language": "urdu", "text": "",
              "rows": [{"bbox": [900, 300, 1500, 400], "is_bullet": false, "text": "",
                        "cells": [$cells]}]}],
  "formatted_text": "", "processing_ms": 0, "total_cells": 2
}''';

void main() {
  test('Table rows join columns with TAB in right-to-left column order', () {
    final r = OcrResult.fromJson(jsonDecode(_page('$_cell1, $_cell2')) as Map<String, dynamic>);
    final block = r.blocks.single;
    expect(block.isTable, isTrue);
    expect(block.columns, 2);
    expect(r.currentText(), 'سَمِعَ\tاس نے سنا');
    expect(r.readCells, 2);
  });

  test('Streaming: a cell event fills in an unread cell', () {
    final r = OcrResult.fromJson(jsonDecode(_page('$_cell1, $_unread')) as Map<String, dynamic>);
    expect(r.readCells, 1);
    expect(r.totalCells, 2);

    // Same shape as {"event": "cell", "block": 0, "row": 0, "cell": 1, ...cell}
    final event = jsonDecode('{"event": "cell", "block": 0, "row": 0, "cell": 1, '
        '${_cell2.substring(1)}') as Map<String, dynamic>;
    r.blocks[event['block'] as int].rows[event['row'] as int].cells[event['cell'] as int] =
        OcrCell.fromJson(event);

    expect(r.readCells, 2);
    expect(r.blocks.single.rows.single.cells[1].candidates['arabic']!.confidence, 50.0);
  });

  test('ApiService strips a trailing slash from the base URL', () {
    final api = ApiService(baseUrl: 'http://10.0.2.2:8000/');
    expect(api.baseUrl, 'http://10.0.2.2:8000');
    api.dispose();
  });

  test('Gemini: corrected boxes keep the OCR reading in raw_text', () {
    final fixed = OcrCell.fromJson(
        {'bbox': [0, 0, 10, 10], 'text': 'برصغیر کی', 'raw_text': 'برصغیرکی', 'confidence': 62.5});
    expect(fixed.isAiCorrected, isTrue);
    expect(fixed.rawText, 'برصغیرکی');

    final same = OcrCell.fromJson(
        {'bbox': [0, 0, 10, 10], 'text': 'اردو', 'raw_text': 'اردو', 'confidence': 90});
    final oldServer = OcrCell.fromJson({'bbox': [0, 0, 10, 10], 'text': 'اردو', 'confidence': 90});
    expect(same.isAiCorrected, isFalse);
    expect(oldServer.isAiCorrected, isFalse);
  });
}
