// page_capture.dart
//
// Renders book pages exactly as the app shows them (same fonts - Nastaliq
// for Urdu, Naskh for Arabic - same line breaks, bullets and tables) on
// white paper, and captures them as images for the PDF. Flutter's own text
// engine shapes the Urdu/Arabic letters, so the PDF looks the same as the
// extracted text in the app.

import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:google_fonts/google_fonts.dart';

import 'book_session.dart';
import 'page_markdown.dart';
import 'theme.dart';

/// Content area of an A4 PDF page (height / width), used to cut long pages.
const double kA4ContentRatio = (842.0 - 2 * 40 - 30) / (595.0 - 2 * 40);

/// Captures every page; returns, per book page, one or more PNG slices that
/// each fit on one A4 sheet. Shows a short "Preparing PDF" screen meanwhile.
Future<List<List<Uint8List>>?> capturePages(BuildContext context, List<BookPage> pages) =>
    Navigator.of(context).push<List<List<Uint8List>>>(PageRouteBuilder(
      opaque: true,
      pageBuilder: (_, _, _) => _CaptureScreen(pages: pages),
    ));

/// A page as it is printed: dark text on white paper.
class PrintedPage extends StatelessWidget {
  const PrintedPage({super.key, required this.page, required this.width});

  final BookPage page;
  final double width;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: width,
      color: Colors.white,
      padding: const EdgeInsets.fromLTRB(14, 10, 14, 14),
      child: MarkdownPage(
        markdown: pageMarkdown(page.result),
        language: pageLanguage(page.result),
        paper: true,
      ),
    );
  }
}

class _CaptureScreen extends StatefulWidget {
  const _CaptureScreen({required this.pages});

  final List<BookPage> pages;

  @override
  State<_CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<_CaptureScreen> {
  late final List<GlobalKey> _keys = [for (final _ in widget.pages) GlobalKey()];
  static const double _pixelRatio = 3;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _capture());
  }

  Future<void> _capture() async {
    final navigator = Navigator.of(context);
    try {
      if (harfUseGoogleFonts) await GoogleFonts.pendingFonts(); // Nastaliq / Naskh first
      await WidgetsBinding.instance.endOfFrame;
      await Future<void>.delayed(const Duration(milliseconds: 100));
      await WidgetsBinding.instance.endOfFrame;
      final out = <List<Uint8List>>[];
      for (final key in _keys) {
        final boundary = key.currentContext!.findRenderObject()! as RenderRepaintBoundary;
        final image = await boundary.toImage(pixelRatio: _pixelRatio);
        out.add(await _slices(image));
        image.dispose();
      }
      navigator.pop(out);
    } catch (e) {
      debugPrint('Page capture failed: $e');
      navigator.pop(null);
    }
  }

  /// Cuts a tall page into A4-sized slices, preferring a blank row between
  /// two lines so no line of text is cut in half.
  Future<List<Uint8List>> _slices(ui.Image image) async {
    final w = image.width, h = image.height;
    final maxH = (w * kA4ContentRatio).floor();
    final raw = (await image.toByteData(format: ui.ImageByteFormat.rawRgba))!;
    bool blankRow(int y) {
      for (var x = 0; x < w; x += 2) {
        final i = (y * w + x) * 4;
        if (raw.getUint8(i) < 245 || raw.getUint8(i + 1) < 245 || raw.getUint8(i + 2) < 245) {
          return false;
        }
      }
      return true;
    }

    final cuts = <int>[0];
    while (h - cuts.last > maxH) {
      var cut = cuts.last + maxH;
      for (var y = cut; y > cuts.last + maxH * 3 ~/ 4; y--) {
        if (blankRow(y)) {
          cut = y;
          break;
        }
      }
      cuts.add(cut);
    }
    cuts.add(h);

    final pngs = <Uint8List>[];
    for (var i = 0; i < cuts.length - 1; i++) {
      final top = cuts[i], sliceH = cuts[i + 1] - cuts[i];
      final recorder = ui.PictureRecorder();
      Canvas(recorder).drawImageRect(
        image,
        Rect.fromLTWH(0, top.toDouble(), w.toDouble(), sliceH.toDouble()),
        Rect.fromLTWH(0, 0, w.toDouble(), sliceH.toDouble()),
        Paint(),
      );
      final slice = await recorder.endRecording().toImage(w, sliceH);
      final png = await slice.toByteData(format: ui.ImageByteFormat.png);
      slice.dispose();
      pngs.add(png!.buffer.asUint8List());
    }
    return pngs;
  }

  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.sizeOf(context).width;
    return Scaffold(
      backgroundColor: HarfColors.navy,
      body: Stack(
        children: [
          // The pages are laid out (and painted) behind the progress card.
          SingleChildScrollView(
            child: Column(
              children: [
                for (var i = 0; i < widget.pages.length; i++)
                  RepaintBoundary(
                    key: _keys[i],
                    child: PrintedPage(page: widget.pages[i], width: width),
                  ),
              ],
            ),
          ),
          Positioned.fill(
            child: ColoredBox(
              color: HarfColors.navy,
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const CircularProgressIndicator(color: HarfColors.brightGold),
                    const SizedBox(height: 16),
                    Text('Preparing PDF (${widget.pages.length} pages)...',
                        style: const TextStyle(color: HarfColors.ink)),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
