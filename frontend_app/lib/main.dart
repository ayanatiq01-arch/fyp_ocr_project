// main.dart
//
// HarfScan - Bilingual (Urdu & Arabic) OCR Scanner for historical books.
//
// Pages:
//   1. LanguageScreen       - choose the script: Urdu, Arabic or Mixed
//   2. DashboardScreen      - Camera Scan / Gallery Upload, book workspace
//   3. SelectTextScreen     - drag from the first word to the last word
//   4. BookWorkspaceScreen  - multi-page book, review highlights, export
//   +  SettingsScreen       - server address, review options, about
//
// One BookSession (the active book) and one AppSettings are shared by all
// pages.

import 'package:flutter/material.dart';

import 'app_settings.dart';
import 'book_session.dart';
import 'language_screen.dart';
import 'theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final settings = await AppSettings.load();
  runApp(HarfScanApp(settings: settings));
}

class HarfScanApp extends StatefulWidget {
  const HarfScanApp({super.key, required this.settings});

  final AppSettings settings;

  @override
  State<HarfScanApp> createState() => _HarfScanAppState();
}

class _HarfScanAppState extends State<HarfScanApp> {
  final BookSession _session = BookSession();

  @override
  void dispose() {
    _session.dispose();
    widget.settings.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'HarfScan',
      debugShowCheckedModeBanner: false,
      theme: buildHarfTheme(),
      home: LanguageScreen(session: _session, settings: widget.settings),
    );
  }
}
