import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend_app/api_service.dart';
import 'package:frontend_app/export_service.dart';
import 'package:frontend_app/page_markdown.dart';

Map<String, dynamic> _cell(String text, {int column = 0, double confidence = 99}) => {
      'bbox': [0, 0, 10, 10],
      'column': column,
      'text': text,
      'language': 'urdu',
      'engine': 'Gemini',
      'confidence': confidence,
      'candidates': {
        'urdu': {'text': text, 'confidence': confidence}
      },
    };

Map<String, dynamic> _row(List<Map<String, dynamic>> cells, {bool bullet = false}) =>
    {'bbox': [0, 0, 10, 10], 'is_bullet': bullet, 'cells': cells};

Map<String, dynamic> _block(String type, List<Map<String, dynamic>> rows, {int columns = 1}) =>
    {'id': 0, 'type': type, 'bbox': [0, 0, 10, 10], 'columns': columns, 'rows': rows};

/// Same page as backend_api/test_gemini_reader.py MarkdownTest.
final _page = OcrResult.fromJson({
  'image': {'width': 100, 'height': 100},
  'blocks': [
    _block('Title', [
      _row([_cell('اردو ادب کی')])
    ]),
    _block('Table', [
      _row([_cell('سَمِعَ'), _cell('اس نے سنا', column: 1)])
    ], columns: 2),
    _block('List', [
      _row([_cell('مرزا غالب')], bullet: true),
      _row([_cell('الحمد لله')]),
    ]),
    _block('Text', [
      _row([_cell('پہلی سطر')]),
      _row([_cell('دوسری سطر', confidence: 40)]),
    ]),
  ],
});

void main() {
  test('Page Markdown matches the backend rules', () {
    expect(
        pageMarkdown(_page),
        '## اردو ادب کی\n\n'
        '| سَمِعَ | اس نے سنا |\n| --- | --- |\n\n'
        '- مرزا غالب\n'
        'الحمد لله\n\n'
        'پہلی سطر  \nدوسری سطر\n');
  });

  test('Low-confidence words are marked for highlighting', () {
    final md = pageMarkdown(_page, isLow: (c) => c < 70);
    expect(md, contains('==دوسری سطر=='));
    expect(md, isNot(contains('==پہلی')));
  });

  test('Markdown characters in OCR text are escaped', () {
    expect(mdEscape('# ۱۔ *x* == [a]'), r'\# ۱۔ \*x\* \=\= \[a\]');
    expect(mdEscape('1. y'), r'\1. y');
  });

  test('Word export keeps headings, bullets and real tables', () {
    final page = ExportPage('=== Page 1 ===', _page.currentText(),
        blocks: ExportBlock.ofResult(_page));
    final xml = utf8.decode(ZipDecoder()
        .decodeBytes(buildBookDocx([page]))
        .findFile('word/document.xml')!
        .content as List<int>);
    expect(xml, contains('<w:jc w:val="center"/>')); // centred heading
    expect(xml, contains('<w:tbl>'));
    expect(xml, contains('<w:bidiVisual/>')); // column 0 on the right
    expect(xml, contains('• مرزا غالب'));
  });

  test('PDF export with the structured page', () async {
    ByteData font(String n) => ByteData.sublistView(File('assets/fonts/$n').readAsBytesSync());
    final pdf = await buildBookPdf(
      [ExportPage('=== Page 1 ===', _page.currentText(), blocks: ExportBlock.ofResult(_page))],
      regularFont: font('NotoNaskhArabic-Regular.ttf'),
      boldFont: font('NotoNaskhArabic-Bold.ttf'),
    );
    expect(String.fromCharCodes(pdf.take(5)), '%PDF-');
  });

  test('Corrected Markdown: plain text and Word blocks', () {
    final md = [
      '## آلِ عِمْرٰن',
      '',
      'لَا رَیْبَ فِیْہِ ۝۹  ',
      'دوسری سطر',
      '',
      '- پہلا نکتہ',
      '- دوسرا',
      '',
      '| سَمِعَ | اس نے سنا |',
      '| --- | --- |',
      '| عَلِمَ | اس نے جانا |',
    ].join('\n');
    expect(
        markdownToPlain(md),
        [
          'آلِ عِمْرٰن',
          '',
          'لَا رَیْبَ فِیْہِ ۝۹',
          'دوسری سطر',
          '',
          '• پہلا نکتہ',
          '• دوسرا',
          '',
          'سَمِعَ\tاس نے سنا',
          'عَلِمَ\tاس نے جانا',
        ].join('\n'));
    final blocks = ExportBlock.ofMarkdown(md);
    expect(blocks.map((b) => b.type), ['Title', 'Text', 'List', 'Table']);
    expect(blocks[1].rows.length, 2); // the book's two lines stay two lines
    expect(blocks[2].rows.every((r) => r.bullet), isTrue);
    expect(blocks[3].rows.length, 2); // header rule dropped
    expect(blocks[3].rows.first.cells, ['سَمِعَ', 'اس نے سنا']);
  });

  test('Arabic lines are marked for the Naskh font, Urdu lines and tables are not', () {
    const table = '| **اوقات الصلوۃ** | **۴۵۱** |';
    const urdu = 'ذِی الحجہ کی تیسری اور آٹھویں تاریخ کو کلمات پڑھنے ہیں۔';
    const arabic = 'اَشْهَدُ اَنْ لَّا اِلٰهَ اِلَّا اللّٰهُ وَحْدَهٗ لَا شَرِيْكَ لَهٗ';
    final md = [table, '| :-: | :-: |', urdu, '# $arabic'].join('\n');
    final lines = markArabicLines(md).split('\n');
    expect(lines[0], table); // table row untouched
    expect(lines[2], urdu); // Urdu line untouched
    expect(lines[3], '# ⁅$arabic⁆');
    expect(markdownToPlain(md), isNot(contains(':-:'))); // short table rules dropped
    expect(looksArabic('جن لوگوں نے کفر کا رویہ اختیار کیا ہے'), isFalse);
  });
}
