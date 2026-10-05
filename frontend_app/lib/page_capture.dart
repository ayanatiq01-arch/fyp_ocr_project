// page_capture.dart
//
// Renders book pages exactly as the app shows them (same fonts - Nastaliq
// for Urdu, Naskh for Arabic - same line breaks, bullets and tables) as
// dark text on white paper, and turns them into images for the PDF.
// Flutter's own text engine shapes the Urdu/Arabic letters, so the PDF
// looks the same as the extracted text in the app.
//
// Every page is drawn in its own off-screen render tree (not on the
// visible screen), so nothing on the screen can hide or blank it.

import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:google_fonts/google_fonts.dart';

import 'book_session.dart';
import 'page_markdown.dart';
import 'theme.dart';

/// Content area of an A4 PDF page (height / width), used to cut long pages.
const double kA4ContentRatio = (842.0 - 30 - 40 - 30) / (595.0 - 2 * 40);

/// Logical width a page is laid out at (about the width of a phone screen,
/// so lines wrap like in the app).
const double kPrintWidth = 420;
const double _pixelRatio = 3;

/// Thrown when a page was rendered without any visible text.
class BlankPageException implements Exception {
  const BlankPageException(this.page);
  final int page;

  @override
  String toString() => 'Page $page rendered empty';
}

/// A page as it is printed: dark text on white paper.
class PrintedPage extends StatelessWidget {
  const PrintedPage({super.key, required this.page, this.width = kPrintWidth});

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

/// Renders every page; returns, per book page, one or more PNG slices that
/// each fit on one A4 sheet. Throws [BlankPageException] if a page with
/// text came out empty (the caller then uses the text-based PDF).
Future<List<List<Uint8List>>> capturePages(BuildContext context, List<BookPage> pages) async {
  if (harfUseGoogleFonts) {
    // Ask for both reading fonts, then wait until they are loaded.
    scriptStyle('urdu');
    scriptStyle('arabic');
    await GoogleFonts.pendingFonts();
  }
  final out = <List<Uint8List>>[];
  for (final page in pages) {
    if (!context.mounted) throw StateError('Export screen closed');
    final image = await renderOffscreen(context, PrintedPage(page: page), kPrintWidth);
    try {
      final hasText = page.text.trim().isNotEmpty;
      out.add(await _slices(image, requireInk: hasText, pageNumber: page.number));
    } finally {
      image.dispose();
    }
  }
  return out;
}

/// Lays out and paints [widget] in a separate render tree [width] logical
/// pixels wide (height = whatever the widget needs) and returns the image.
@visibleForTesting
Future<ui.Image> renderOffscreen(BuildContext context, Widget widget, double width) async {
  final boundary = RenderRepaintBoundary();
  final renderView = RenderView(
    view: View.of(context),
    child: RenderPositionedBox(alignment: Alignment.topCenter, heightFactor: 1, child: boundary),
    configuration: ViewConfiguration(
      logicalConstraints: BoxConstraints(maxWidth: width, maxHeight: 100000),
      physicalConstraints:
          BoxConstraints(maxWidth: width * _pixelRatio, maxHeight: 100000 * _pixelRatio),
      devicePixelRatio: _pixelRatio,
    ),
  );
  final pipelineOwner = PipelineOwner()..rootNode = renderView;
  renderView.prepareInitialFrame();
  final buildOwner = BuildOwner(focusManager: FocusManager());
  final root = RenderObjectToWidgetAdapter<RenderBox>(
    container: boundary,
    child: InheritedTheme.captureAll(
      context,
      MediaQuery(
        data: MediaQuery.of(context),
        child: Directionality(
          textDirection: TextDirection.rtl,
          child: Material(color: Colors.white, child: widget),
        ),
      ),
    ),
  ).attachToRenderTree(buildOwner);
  buildOwner.buildScope(root);
  buildOwner.finalizeTree();
  pipelineOwner.flushLayout();
  pipelineOwner.flushCompositingBits();
  pipelineOwner.flushPaint();
  final image = await boundary.toImage(pixelRatio: _pixelRatio);
  // Tear the temporary tree down.
  root.update(RenderObjectToWidgetAdapter<RenderBox>(container: boundary));
  buildOwner.finalizeTree();
  return image;
}

/// Cuts a tall page into A4-sized slices, preferring a blank row between
/// two lines so no line of text is cut in half.
Future<List<Uint8List>> _slices(ui.Image image, {required bool requireInk, required int pageNumber}) async {
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

  if (requireInk) {
    var ink = false;
    for (var y = 0; y < h && !ink; y += 3) {
      ink = !blankRow(y);
    }
    if (!ink) throw BlankPageException(pageNumber);
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
