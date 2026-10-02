// app_settings.dart
//
// User settings, remembered between app launches (shared_preferences):
// backend server URL, low-confidence review threshold and highlighting,
// Gemini AI correction.
// Also owns the ApiService for the current server URL.

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'api_service.dart';

class AppSettings extends ChangeNotifier {
  AppSettings._(this._prefs)
      : _serverUrl = _prefs.getString(_kServerUrl) ?? ApiService.defaultBaseUrl,
        _threshold = _prefs.getDouble(_kThreshold) ?? 70,
        _highlight = _prefs.getBool(_kHighlight) ?? true,
        _aiCorrect = _prefs.getBool(_kAiCorrect) ?? true {
    _api = ApiService(baseUrl: _serverUrl);
  }

  static Future<AppSettings> load() async =>
      AppSettings._(await SharedPreferences.getInstance());

  static const _kServerUrl = 'server_url';
  static const _kThreshold = 'low_confidence_threshold';
  static const _kHighlight = 'highlight_low_confidence';
  static const _kAiCorrect = 'gemini_ai_correct';

  final SharedPreferences _prefs;
  late ApiService _api;
  String _serverUrl;
  double _threshold;
  bool _highlight;
  bool _aiCorrect;
  bool? _serverOnline;

  ApiService get api => _api;
  String get serverUrl => _api.baseUrl;

  /// null = checking / unknown.
  bool? get serverOnline => _serverOnline;

  /// Words read with confidence (0-100) below this are highlighted for review.
  double get lowConfidenceThreshold => _threshold;
  bool get highlightLowConfidence => _highlight;

  /// Ask the server to let Gemini visually check and correct the OCR text.
  bool get aiCorrect => _aiCorrect;

  set aiCorrect(bool value) {
    _aiCorrect = value;
    _prefs.setBool(_kAiCorrect, value);
    notifyListeners();
  }

  /// True if [confidence] should be highlighted for review.
  bool isLowConfidence(double confidence) => _highlight && confidence < _threshold;

  Future<bool> checkServer() async {
    _serverOnline = null;
    notifyListeners();
    final api = _api;
    final ok = await api.healthCheck();
    if (identical(api, _api)) {
      _serverOnline = ok;
      notifyListeners();
    }
    return ok;
  }

  Future<void> setServerUrl(String url) async {
    url = url.trim();
    if (url.isEmpty) return;
    if (!url.startsWith('http://') && !url.startsWith('https://')) url = 'http://$url';
    _api.dispose();
    _api = ApiService(baseUrl: url);
    _serverUrl = _api.baseUrl;
    await _prefs.setString(_kServerUrl, _serverUrl);
    await checkServer();
  }

  set lowConfidenceThreshold(double value) {
    _threshold = value;
    _prefs.setDouble(_kThreshold, value);
    notifyListeners();
  }

  set highlightLowConfidence(bool value) {
    _highlight = value;
    _prefs.setBool(_kHighlight, value);
    notifyListeners();
  }

  @override
  void dispose() {
    _api.dispose();
    super.dispose();
  }
}
