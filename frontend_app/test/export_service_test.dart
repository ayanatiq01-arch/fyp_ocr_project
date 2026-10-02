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

  // Like rootBundle.load() on a phone: the font is a view at a non-zero
  // offset inside a larger buffer (used to throw RangeError on export).
  ByteData fontView(String name) {
    final bytes = File('assets/fonts/$name').readAsBytesSync();
    final padded = Uint8List(bytes.length + 64)..setRange(64, 64 + bytes.length, bytes);
    return ByteData.sublistView(padded, 64);
  }

  test('PDF is produced from real OCR pages with the bundled font', () async {
    final pages = [
      ..._pages,
      ExportPage('=== Page 3 ===', File('test/fixtures/real_page_urdu.txt').readAsStringSync()),
      ExportPage('=== Page 4 ===', File('test/fixtures/real_page_bullets.txt').readAsStringSync()),
    ];
    final pdf = await buildBookPdf(
      pages,
      regularFont: fontView('NotoNaskhArabic-Regular.ttf'),
      boldFont: fontView('NotoNaskhArabic-Bold.ttf'),
    );
    expect(String.fromCharCodes(pdf.take(5)), '%PDF-');
    expect(pdf.length, greaterThan(1000));
  });
}
