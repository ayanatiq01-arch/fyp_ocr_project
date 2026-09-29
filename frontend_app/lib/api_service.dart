// api_service.dart
//
// HTTP client for the FastAPI backend (backend_api/main.py).
//
//   POST {baseUrl}/api/v1/ocr   multipart/form-data, field name "file"
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
// Response models (mirror the Pydantic models in backend_api/main.py)
// --------------------------------------------------------------------------

/// Axis-aligned box in pixels of the uploaded (cropped) image.
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

/// Result of one OCR engine for one line (kept for transparency / debugging).
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

/// One recognised text line, after confidence routing.
class OcrLine {
  const OcrLine({
    required this.bbox,
    required this.text,
    required this.language,
    required this.engine,
    required this.confidence,
    required this.candidates,
  });

  factory OcrLine.fromJson(Map<String, dynamic> json) => OcrLine(
        bbox: BBox.fromJson(json['bbox'] as List<dynamic>),
        text: json['text'] as String? ?? '',
        language: json['language'] as String? ?? 'unknown',
        engine: json['engine'] as String? ?? '',
        confidence: (json['confidence'] as num?)?.toDouble() ?? 0,
        candidates: (json['candidates'] as Map<String, dynamic>? ?? {}).map(
          (k, v) => MapEntry(k, EngineCandidate.fromJson(v as Map<String, dynamic>)),
        ),
      );

  final BBox bbox;
  final String text;

  /// "urdu" | "arabic" | "unknown"
  final String language;

  /// "UTRNet" | "PaddleOCR" | ""
  final String engine;

  /// 0-100, confidence of the accepted engine.
  final double confidence;

  /// Keyed by language ("urdu", "arabic").
  final Map<String, EngineCandidate> candidates;
}

/// A layout block (paragraph, title, list ...) with its lines.
class OcrBlock {
  const OcrBlock({
    required this.id,
    required this.type,
    required this.bbox,
    required this.text,
    required this.language,
    required this.lines,
  });

  factory OcrBlock.fromJson(Map<String, dynamic> json) => OcrBlock(
        id: json['id'] as int,
        type: json['type'] as String? ?? 'Text',
        bbox: BBox.fromJson(json['bbox'] as List<dynamic>),
        text: json['text'] as String? ?? '',
        language: json['language'] as String? ?? 'unknown',
        lines: (json['lines'] as List<dynamic>? ?? [])
            .map((e) => OcrLine.fromJson(e as Map<String, dynamic>))
            .toList(),
      );

  final int id;

  /// "Text" | "Title" | "List"
  final String type;
  final BBox bbox;
  final String text;

  /// "urdu" | "arabic" | "mixed" | "unknown"
  final String language;
  final List<OcrLine> lines;
}

/// Whole response of POST /api/v1/ocr.
class OcrResult {
  const OcrResult({
    required this.requestId,
    required this.imageWidth,
    required this.imageHeight,
    required this.skewAngle,
    required this.layoutEngine,
    required this.blocks,
    required this.formattedText,
    required this.processingMs,
  });

  factory OcrResult.fromJson(Map<String, dynamic> json) {
    final image = json['image'] as Map<String, dynamic>;
    return OcrResult(
      requestId: json['request_id'] as String? ?? '',
      imageWidth: (image['width'] as num).toDouble(),
      imageHeight: (image['height'] as num).toDouble(),
      skewAngle: (json['skew_angle'] as num?)?.toDouble() ?? 0,
      layoutEngine: json['layout_engine'] as String? ?? '',
      blocks: (json['blocks'] as List<dynamic>? ?? [])
          .map((e) => OcrBlock.fromJson(e as Map<String, dynamic>))
          .toList(),
      formattedText: json['formatted_text'] as String? ?? '',
      processingMs: json['processing_ms'] as int? ?? 0,
    );
  }

  final String requestId;
  final double imageWidth;
  final double imageHeight;
  final double skewAngle;
  final String layoutEngine;
  final List<OcrBlock> blocks;

  /// Plain text with paragraph breaks and bullet markers preserved.
  final String formattedText;
  final int processingMs;
}

// --------------------------------------------------------------------------
// Client
// --------------------------------------------------------------------------

class ApiService {
  ApiService({
    required String baseUrl,
    http.Client? client,
    this.timeout = const Duration(seconds: 180),
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

  /// OCR runs on CPU on the server and can take a while for large crops.
  final Duration timeout;
  final http.Client _client;

  static String _normalise(String url) =>
      url.trim().endsWith('/') ? url.trim().substring(0, url.trim().length - 1) : url.trim();

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

  /// Uploads [image] (the user's cropped region) and returns the OCR result.
  Future<OcrResult> scanImage(File image) async {
    final uri = Uri.parse('$baseUrl/api/v1/ocr');
    final request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath('file', image.path));

    final http.Response response;
    try {
      final streamed = await _client.send(request).timeout(timeout);
      response = await http.Response.fromStream(streamed).timeout(timeout);
    } on TimeoutException {
      throw ApiException('The server took too long to respond. Try a smaller crop.');
    } on SocketException catch (e) {
      throw ApiException('Cannot reach the server at $baseUrl (${e.message}).');
    } on http.ClientException catch (e) {
      throw ApiException('Network error: ${e.message}');
    }

    // Always decode as UTF-8: Urdu/Arabic text is garbled otherwise.
    final bodyText = utf8.decode(response.bodyBytes);
    if (response.statusCode != 200) {
      throw ApiException(_errorDetail(bodyText), statusCode: response.statusCode);
    }
    try {
      return OcrResult.fromJson(jsonDecode(bodyText) as Map<String, dynamic>);
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
