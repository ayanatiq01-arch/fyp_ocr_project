// api_service.dart
//
// HTTP client for the FastAPI backend (backend_api/main.py).
//
//   POST {baseUrl}/api/v1/ocr/stream   multipart "file" -> NDJSON event stream
//   POST {baseUrl}/api/v1/ocr          multipart "file" -> final JSON
//   GET  {baseUrl}/health
//
// Also contains the typed models for the JSON the backend returns, so the UI
// never touches raw maps.

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

/// Thrown for any failure talking to the backend; [message] is user-facing.
class ApiException implements Exception {
  ApiException(this.message, {this.statusCode});

  final String message;
  final int? statusCode;

  @override
  String toString() =>
      statusCode == null ? message : 'HTTP $statusCode: $message';
}

// --------------------------------------------------------------------------
// Models (mirror the Pydantic models in backend_api/main.py)
// --------------------------------------------------------------------------

/// Axis-aligned box in pixels of the uploaded image.
class BBox {
  const BBox(this.x1, this.y1, this.x2, this.y2);

  factory BBox.fromJson(List<dynamic> json) => BBox(
        (json[0] as num).toDouble(),
        (json[1] as num).toDouble(),
        (json[2] as num).toDouble(),
        (json[3] as num).toDouble(),
      );

  final double x1, y1, x2, y2;
}

/// One engine's reading of a cell (kept to show how the router decided).
class EngineCandidate {
  const EngineCandidate({required this.text, required this.confidence});

  factory EngineCandidate.fromJson(Map<String, dynamic> json) =>
      EngineCandidate(
        text: json['text'] as String? ?? '',
        confidence: (json['confidence'] as num?)?.toDouble() ?? 0,
      );

  final String text;

  /// 0-100.
  final double confidence;
}

/// One auto-cropped text box: a line, part of a line or a table cell.
class OcrCell {
  const OcrCell({
    required this.bbox,
    required this.column,
    required this.text,
    required this.language,
    required this.engine,
    required this.confidence,
    required this.candidates,
  });

  factory OcrCell.fromJson(Map<String, dynamic> json) => OcrCell(
        bbox: BBox.fromJson(json['bbox'] as List<dynamic>),
        column: json['column'] as int? ?? 0,
        text: json['text'] as String? ?? '',
        language: json['language'] as String? ?? 'unknown',
        engine: json['engine'] as String? ?? '',
        confidence: (json['confidence'] as num?)?.toDouble() ?? 0,
        candidates: (json['candidates'] as Map<String, dynamic>? ?? {}).map(
          (k, v) => MapEntry(k, EngineCandidate.fromJson(v as Map<String, dynamic>)),
        ),
      );

  final BBox bbox;

  /// Table column, 0 = right-most.
  final int column;
  final String text;

  /// "urdu" | "arabic" | "unknown"
  final String language;

  /// "UTRNet" | "PaddleOCR" | "" (empty = not read yet)
  final String engine;

  /// 0-100, confidence of the accepted engine.
  final double confidence;

  /// Keyed by language ("urdu", "arabic").
  final Map<String, EngineCandidate> candidates;

  /// True once the backend has read this cell.
  bool get isRead => candidates.isNotEmpty;
}

/// Cells on one printed line, in right-to-left reading order.
class OcrRow {
  OcrRow({required this.bbox, required this.isBullet, required this.cells});

  factory OcrRow.fromJson(Map<String, dynamic> json) => OcrRow(
        bbox: BBox.fromJson(json['bbox'] as List<dynamic>),
        isBullet: json['is_bullet'] as bool? ?? false,
        cells: (json['cells'] as List<dynamic>? ?? [])
            .map((e) => OcrCell.fromJson(e as Map<String, dynamic>))
            .toList(),
      );

  final BBox bbox;
  final bool isBullet;

  /// Mutable: cells are replaced one by one while the page is streamed.
  final List<OcrCell> cells;
}

/// A layout block: paragraph, title, list or table.
class OcrBlock {
  OcrBlock({
    required this.id,
    required this.type,
    required this.bbox,
    required this.columns,
    required this.language,
    required this.rows,
  });

  factory OcrBlock.fromJson(Map<String, dynamic> json) => OcrBlock(
        id: json['id'] as int,
        type: json['type'] as String? ?? 'Text',
        bbox: BBox.fromJson(json['bbox'] as List<dynamic>),
        columns: json['columns'] as int? ?? 1,
        language: json['language'] as String? ?? 'unknown',
        rows: (json['rows'] as List<dynamic>? ?? [])
            .map((e) => OcrRow.fromJson(e as Map<String, dynamic>))
            .toList(),
      );

  final int id;

  /// "Text" | "Title" | "List" | "Table"
  final String type;
  final BBox bbox;

  /// Number of table columns (1 for text).
  final int columns;

  /// "urdu" | "arabic" | "mixed" | "unknown"
  final String language;
  final List<OcrRow> rows;

  bool get isTable => type == 'Table';

  /// Text of one row: table columns joined by TAB, otherwise by spaces.
  String rowText(OcrRow row) {
    if (!isTable) {
      return row.cells.where((c) => c.text.isNotEmpty).map((c) => c.text).join(' ');
    }
    final slots = List<String>.filled(columns < 1 ? 1 : columns, '');
    for (final c in row.cells) {
      if (c.text.isEmpty || c.column >= slots.length) continue;
      slots[c.column] = slots[c.column].isEmpty ? c.text : '${slots[c.column]} ${c.text}';
    }
    return slots.join('\t').replaceAll(RegExp(r'\t+$'), '');
  }
}

/// A (possibly still streaming) page result.
class OcrResult {
  OcrResult({
    required this.requestId,
    required this.imageWidth,
    required this.imageHeight,
    required this.rotation,
    required this.skewAngle,
    required this.layoutEngine,
    required this.blocks,
    required this.formattedText,
    required this.processingMs,
    required this.totalCells,
  });

  factory OcrResult.fromJson(Map<String, dynamic> json) {
    final image = json['image'] as Map<String, dynamic>;
    final blocks = (json['blocks'] as List<dynamic>? ?? [])
        .map((e) => OcrBlock.fromJson(e as Map<String, dynamic>))
        .toList();
    return OcrResult(
      requestId: json['request_id'] as String? ?? '',
      imageWidth: (image['width'] as num).toDouble(),
      imageHeight: (image['height'] as num).toDouble(),
      rotation: json['rotation'] as int? ?? 0,
      skewAngle: (json['skew_angle'] as num?)?.toDouble() ?? 0,
      layoutEngine: json['layout_engine'] as String? ?? '',
      blocks: blocks,
      formattedText: json['formatted_text'] as String? ?? '',
      processingMs: json['processing_ms'] as int? ?? 0,
      totalCells: json['total_cells'] as int? ??
          blocks.fold<int>(0, (n, b) => n + b.rows.fold<int>(0, (m, r) => m + r.cells.length)),
    );
  }

  final String requestId;
  final double imageWidth;
  final double imageHeight;

  /// Degrees (CCW) the backend turned the photo to make the page upright.
  final int rotation;
  final double skewAngle;
  final String layoutEngine;
  final List<OcrBlock> blocks;

  /// Final page text from the backend (empty while streaming).
  final String formattedText;
  final int processingMs;
  final int totalCells;

  Iterable<OcrCell> get cells => blocks.expand((b) => b.rows).expand((r) => r.cells);

  int get readCells => cells.where((c) => c.isRead).length;

  /// Page text built from what has been read so far (same rules as the
  /// backend: blank line between blocks, TAB between table columns).
  String currentText() {
    if (formattedText.isNotEmpty) return formattedText;
    return blocks
        .map((b) => b.rows
            .where((r) => b.rowText(r).trim().isNotEmpty)
            .map((r) => r.isBullet ? '• ${b.rowText(r)}' : b.rowText(r))
            .join('\n'))
        .where((t) => t.isNotEmpty)
        .join('\n\n');
  }
}

/// Events of POST /api/v1/ocr/stream.
sealed class OcrEvent {
  const OcrEvent();
}

/// Page structure is known (boxes found); no text yet.
class LayoutEvent extends OcrEvent {
  const LayoutEvent(this.result);
  final OcrResult result;
}

/// One cell has been read.
class CellEvent extends OcrEvent {
  const CellEvent(this.block, this.row, this.index, this.cell);
  final int block, row, index;
  final OcrCell cell;
}

/// Whole page finished (noise removed, bullets marked, final text).
class DoneEvent extends OcrEvent {
  const DoneEvent(this.result);
  final OcrResult result;
}

// --------------------------------------------------------------------------
// Client
// --------------------------------------------------------------------------

class ApiService {
  ApiService({
    required String baseUrl,
    http.Client? client,
    this.timeout = const Duration(seconds: 300),
  })  : baseUrl = _normalise(baseUrl),
        _client = client ?? http.Client();

  /// Default backend address. 10.0.2.2 is the host machine as seen from the
  /// Android emulator. On a real phone use the PC's LAN IP, e.g.
  /// `flutter run --dart-define=API_BASE_URL=http://192.168.1.20:8000`.
  static const String defaultBaseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://10.0.2.2:8000',
  );

  final String baseUrl;

  /// Longest silence allowed between two streamed events / for a whole
  /// non-streamed request (OCR runs on the server's CPU).
  final Duration timeout;
  final http.Client _client;

  static String _normalise(String url) {
    final u = url.trim();
    return u.endsWith('/') ? u.substring(0, u.length - 1) : u;
  }

  /// Returns true if the backend answers GET /health with status "ok".
  Future<bool> healthCheck() async {
    try {
      final res = await _client
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 5));
      if (res.statusCode != 200) return false;
      final body = jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
      return body['status'] == 'ok';
    } catch (_) {
      return false;
    }
  }

  Future<http.StreamedResponse> _post(String path, File image, String language) async {
    final request = http.MultipartRequest('POST', Uri.parse('$baseUrl$path'))
      ..fields['language'] = language
      ..files.add(await http.MultipartFile.fromPath('file', image.path));
    try {
      final response = await _client.send(request).timeout(const Duration(seconds: 60));
      if (response.statusCode != 200) {
        final body = await response.stream.bytesToString();
        throw ApiException(_errorDetail(body), statusCode: response.statusCode);
      }
      return response;
    } on TimeoutException {
      throw ApiException('The server did not respond. Is it running?');
    } on SocketException catch (e) {
      throw ApiException('Cannot reach the server at $baseUrl (${e.message}).');
    } on http.ClientException catch (e) {
      throw ApiException('Network error: ${e.message}');
    }
  }

  /// Uploads the whole page photo and yields results as the server reads
  /// them, in reading order.
  ///
  /// [language]: "mixed" (both engines + confidence router), "urdu" (UTRNet
  /// only) or "arabic" (PaddleOCR only).
  Stream<OcrEvent> scanStream(File image, {String language = 'mixed'}) async* {
    final response = await _post('/api/v1/ocr/stream', image, language);
    final lines = response.stream
        .transform(utf8.decoder) // Urdu/Arabic: always decode as UTF-8
        .transform(const LineSplitter())
        .timeout(timeout, onTimeout: (sink) {
      sink.addError(ApiException('The server stopped responding.'));
      sink.close();
    });
    try {
      await for (final line in lines) {
        if (line.trim().isEmpty) continue;
        final json = jsonDecode(line) as Map<String, dynamic>;
        switch (json['event']) {
          case 'layout':
            yield LayoutEvent(OcrResult.fromJson(json));
          case 'cell':
            yield CellEvent(json['block'] as int, json['row'] as int, json['cell'] as int,
                OcrCell.fromJson(json));
          case 'done':
            yield DoneEvent(OcrResult.fromJson(json));
          case 'error':
            throw ApiException(json['detail'] as String? ?? 'OCR failed on the server.');
        }
      }
    } on http.ClientException catch (e) {
      throw ApiException('Connection lost: ${e.message}');
    }
  }

  /// Non-streaming variant: returns the finished page.
  Future<OcrResult> scanImage(File image, {String language = 'mixed'}) async {
    final response = await _post('/api/v1/ocr', image, language);
    final body = await response.stream.bytesToString().timeout(timeout);
    try {
      return OcrResult.fromJson(jsonDecode(body) as Map<String, dynamic>);
    } catch (e) {
      throw ApiException('Unexpected response from server: $e');
    }
  }

  /// Extracts FastAPI's {"detail": ...} message when present.
  static String _errorDetail(String body) {
    try {
      final decoded = jsonDecode(body);
      if (decoded is Map && decoded['detail'] != null) {
        return decoded['detail'].toString();
      }
    } catch (_) {
      // Not JSON - fall through.
    }
    return body.isEmpty ? 'Unknown server error' : body;
  }

  void dispose() => _client.close();
}
