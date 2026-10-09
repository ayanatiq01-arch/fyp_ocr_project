// pdf_viewer_screen.dart
//
// Opens an exported PDF inside the app right after it is saved: the pages
// are shown as they are in the file (zoomable), with a Share button for
// sending it on (WhatsApp, Drive, email ...).

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:printing/printing.dart';

import 'export_service.dart';
import 'theme.dart';

class PdfViewerScreen extends StatefulWidget {
  const PdfViewerScreen({super.key, required this.file, required this.savedTo});

  /// The PDF in the app's folder (shown and shared).
  final File file;

  /// Where the copy in the phone's Downloads is ("Download/HarfScan/...").
  final String savedTo;

  @override
  State<PdfViewerScreen> createState() => _PdfViewerScreenState();
}

class _PdfViewerScreenState extends State<PdfViewerScreen> {
  final _pages = <Uint8List>[];
  Object? _error;
  bool _done = false;

  String get _name => widget.file.uri.pathSegments.last;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final bytes = await widget.file.readAsBytes();
      await for (final page in Printing.raster(bytes, dpi: 144)) {
        final png = await page.toPng();
        if (!mounted) return;
        setState(() => _pages.add(png));
      }
    } catch (e) {
      _error = e;
    }
    if (mounted) setState(() => _done = true);
  }

  Future<void> _share() => ExportService.share(widget.file, _name);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF3A3F47),
      appBar: AppBar(
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(_name, overflow: TextOverflow.ellipsis, style: const TextStyle(fontSize: 16)),
            Text('Saved to ${widget.savedTo}',
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(fontSize: 11, color: Colors.white70)),
          ],
        ),
        actions: [
          IconButton(tooltip: 'Share', icon: const Icon(Icons.share), onPressed: _share),
        ],
      ),
      body: _error != null && _pages.isEmpty
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text('The PDF was saved but cannot be shown here:\n$_error',
                    textAlign: TextAlign.center, style: const TextStyle(color: Colors.white)),
              ),
            )
          : InteractiveViewer(
              minScale: 1,
              maxScale: 4,
              child: ListView.builder(
                padding: const EdgeInsets.fromLTRB(12, 12, 12, 96),
                itemCount: _pages.length + (_done ? 0 : 1),
                itemBuilder: (context, i) {
                  if (i == _pages.length) {
                    return const Padding(
                      padding: EdgeInsets.all(32),
                      child: Center(child: CircularProgressIndicator(color: HarfColors.gold)),
                    );
                  }
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: DecoratedBox(
                      decoration: const BoxDecoration(
                        color: Colors.white,
                        boxShadow: [BoxShadow(color: Colors.black38, blurRadius: 6)],
                      ),
                      child: Image.memory(_pages[i], fit: BoxFit.fitWidth, gaplessPlayback: true),
                    ),
                  );
                },
              ),
            ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _share,
        backgroundColor: HarfColors.gold,
        foregroundColor: Colors.black,
        icon: const Icon(Icons.share),
        label: const Text('Share'),
      ),
    );
  }
}
