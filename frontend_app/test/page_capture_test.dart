import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend_app/api_service.dart';
import 'package:frontend_app/book_session.dart';
import 'package:frontend_app/export_service.dart';
import 'package:frontend_app/page_capture.dart';
import 'package:frontend_app/theme.dart';

void main() {
  setUpAll(() => harfUseGoogleFonts = false);
  testWidgets('PDF pages are rendered off-screen and are not blank', (tester) async {
    Map<String, dynamic> cell(String t) => {'bbox': [0, 0, 1, 1], 'column': 0, 'text': t, 'language': 'urdu',
        'engine': 'Gemini', 'confidence': 99, 'candidates': {'urdu': {'text': t, 'confidence': 99}}};
    final result = OcrResult.fromJson({
      'image': {'width': 100, 'height': 100},
      'blocks': [
        {'id': 0, 'type': 'Title', 'bbox': [0, 0, 1, 1], 'columns': 1, 'language': 'urdu',
          'rows': [{'bbox': [0, 0, 1, 1], 'is_bullet': false, 'cells': [cell('Heading line')]}]},
        {'id': 1, 'type': 'Text', 'bbox': [0, 0, 1, 1], 'columns': 1, 'language': 'urdu',
          'rows': [for (var i = 0; i < 60; i++) {'bbox': [0, 0, 1, 1], 'is_bullet': false, 'cells': [cell('Line number $i of the page')]}]},
      ],
    });
    final book = BookSession(title: 'T')..addPage(result, '');
    List<List<Uint8List>>? images;
    Object? error;
    await tester.pumpWidget(MaterialApp(home: Builder(builder: (context) => Scaffold(body: Center(
        child: ElevatedButton(
          onPressed: () async {
            try { images = await capturePages(context, book.pages); } catch (e) { error = e; }
          },
          child: const Text('go')))))));
    await tester.runAsync(() async {
      await tester.tap(find.text('go'));
      for (var i = 0; i < 50 && images == null && error == null; i++) {
        await Future<void>.delayed(const Duration(milliseconds: 100));
      }
    });
    expect(error, isNull); // capturePages throws BlankPageException for an empty page
    expect(images, hasLength(1));
    expect(images!.single.length, greaterThan(1)); // 60 lines: more than one A4 sheet
    await tester.runAsync(() async {
      final pdf = await buildImagePdf(images!);
      expect(String.fromCharCodes(pdf.take(5)), '%PDF-');
    });
  });
}
