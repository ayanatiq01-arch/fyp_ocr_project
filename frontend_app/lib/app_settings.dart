// app_settings.dart
//
// User settings, remembered between app launches (shared_preferences):
// backend server URL, low-confidence review threshold and highlighting,
// Gemini AI page reading.
// Also owns the ApiService for the current server URL.

import 'dart:async';
import 'dart:io';

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
  bool _discovering = false;

  /// True while [discoverServer] scans the Wi-Fi network.
  bool get discovering => _discovering;

  ApiService get api => _api;
  String get serverUrl => _api.baseUrl;

  /// null = checking / unknown.
  bool? get serverOnline => _serverOnline;

  /// Words read with confidence (0-100) below this are highlighted for review.
  double get lowConfidenceThreshold => _threshold;
  bool get highlightLowConfidence => _highlight;

  /// Ask the server to read pages with Gemini (falls back to local OCR).
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

  /// Checks the saved server; if it does not answer (PC switched off, server
  /// restarted, or the PC got a new IP from the router), scans the Wi-Fi
  /// network for it. Returns true if a server is reachable afterwards.
  Future<bool> ensureServer() async {
    if (await checkServer()) return true;
    return discoverServer();
  }

  /// Finds the HarfScan server on the phone's Wi-Fi network: every address
  /// of the phone's /24 subnet is tried on the server's port (fast TCP
  /// connect, then GET /health). The first server found is saved.
  Future<bool> discoverServer() async {
    if (_discovering) return false;
    _discovering = true;
    _serverOnline = null;
    notifyListeners();
    try {
      final saved = Uri.tryParse(serverUrl);
      final port = (saved != null && saved.hasPort) ? saved.port : 8000;
      final hosts = <String>[];
      if (saved != null && saved.host.isNotEmpty) hosts.add(saved.host);
      final interfaces = await NetworkInterface.list(type: InternetAddressType.IPv4);
      for (final iface in interfaces) {
        for (final addr in iface.addresses) {
          if (addr.isLoopback) continue;
          final parts = addr.address.split('.');
          if (parts.length != 4) continue;
          final prefix = parts.take(3).join('.');
          for (var i = 1; i < 255; i++) {
            final host = '$prefix.$i';
            if (host != addr.address && !hosts.contains(host)) hosts.add(host);
          }
        }
      }
      const wave = 64;
      for (var start = 0; start < hosts.length; start += wave) {
        final batch = hosts.sublist(start, start + wave > hosts.length ? hosts.length : start + wave);
        final found = await _firstServer(batch, port);
        if (found != null) {
          await setServerUrl('http://$found:$port');
          return _serverOnline == true;
        }
      }
      _serverOnline = false;
      return false;
    } finally {
      _discovering = false;
      notifyListeners();
    }
  }

  static Future<String?> _firstServer(List<String> hosts, int port) async {
    final results = await Future.wait(hosts.map((host) async {
      try {
        final socket = await Socket.connect(host, port, timeout: const Duration(milliseconds: 700));
        socket.destroy();
      } catch (_) {
        return null;
      }
      final api = ApiService(baseUrl: 'http://$host:$port');
      try {
        return await api.healthCheck() ? host : null;
      } finally {
        api.dispose();
      }
    }));
    return results.firstWhere((h) => h != null, orElse: () => null);
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
