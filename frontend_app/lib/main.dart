// main.dart
//
// HarfScan - Bilingual (Urdu & Arabic) OCR Scanner for historical books.
//
// Screens:
//   1. DashboardScreen      - language toggle, Camera / Gallery, workspace
//   2. CropScreen           - crop box or whole page -> extract & append
//   3. BookWorkspaceScreen  - multi-page book, review highlights, export
//
// One BookSession (the active book) is shared by all screens.

import 'package:flutter/material.dart';

import 'book_session.dart';
import 'dashboard_screen.dart';
import 'theme.dart';

void main() => runApp(const HarfScanApp());

class HarfScanApp extends StatefulWidget {
  const HarfScanApp({super.key});

  @override
  State<HarfScanApp> createState() => _HarfScanAppState();
}

class _HarfScanAppState extends State<HarfScanApp> {
  final BookSession _session = BookSession();

  @override
  void dispose() {
    _session.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'HarfScan',
      debugShowCheckedModeBanner: false,
      theme: buildHarfTheme(),
      home: DashboardScreen(session: _session),
    );
  }
}
