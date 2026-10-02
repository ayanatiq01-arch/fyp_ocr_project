import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend_app/export_service.dart';

const _pages = [
  ExportPage('=== Page 1 ===', 'پاکستان ایک خوبصورت ملک ہے\n\n• العلم نور\nنام\tعمر'),
  ExportPage('=== Page 2 ===', 'A & B <test>'),
];

void main() {
  test('DOCX contains every page header and escaped RTL text', () {
    final bytes = buildBookDocx(_pages);
    final archive = ZipDecoder().decodeBytes(bytes);
    expect(archive.findFile('[Content_Types].xml'), isNotNull);
    expect(archive.findFile('_rels/.rels'), isNotNull);
    final xml = utf8.decode(archive.findFile('word/document.xml')!.content as List<int>);
    expect(xml, contains('=== Page 1 ==='));
    expect(xml, contains('=== Page 2 ==='));
    expect(xml, contains('پاکستان ایک خوبصورت ملک ہے'));
    expect(xml, contains('نام   |   عمر')); // TAB -> column separator
    expect(xml, contains('A &amp; B &lt;test&gt;'));
    expect(xml, contains('<w:bidi/>'));
    expect('<w:br w:type="page"/>'.allMatches(xml).length, 1);
  });

  test('PDF is produced with the bundled Arabic-script font', () async {
    final regular = File('assets/fonts/NotoNaskhArabic-Regular.ttf').readAsBytesSync();
    final bold = File('assets/fonts/NotoNaskhArabic-Bold.ttf').readAsBytesSync();
    final pdf = await buildBookPdf(
      _pages,
      regularFont: ByteData.sublistView(Uint8List.fromList(regular)),
      boldFont: ByteData.sublistView(Uint8List.fromList(bold)),
    );
    expect(String.fromCharCodes(pdf.take(5)), '%PDF-');
    expect(pdf.length, greaterThan(1000));
  });
}
