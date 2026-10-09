// export_service.dart
//
// Exports the compiled book (all pages, in order) as PDF or Word (.docx)
// and hands the file to the system share sheet (Files, Drive, WhatsApp ...).
//
// The builders are pure functions (bytes in, bytes out) so they can be unit
// tested without a device.

import 'dart:io';
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter/foundation.dart' show debugPrint;
import 'package:flutter/widgets.dart' show BuildContext;
import 'package:path_provider/path_provider.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:share_plus/share_plus.dart';

import 'api_service.dart';
import 'book_session.dart';
import 'page_capture.dart';
import 'page_markdown.dart';

/// One printed line: its cells (one for text lines, one per column for
/// table rows, column 0 = right-most) and whether it is a bullet item.
class ExportRow {
  const ExportRow(this.cells, {this.bullet = false});

  final List<String> cells;
  final bool bullet;
}

/// A layout block of the page: "Title", "Text", "List" or "Table".
class ExportBlock {
  const ExportBlock(this.type, this.rows);

  /// The page's blocks with their lines, in reading order.
  static List<ExportBlock> ofResult(OcrResult result) => [
        for (final b in result.blocks)
          ExportBlock(b.type, [
            for (final r in b.rows)
              if (r.cells.any((c) => c.text.isNotEmpty))
                ExportRow(b.isTable ? _tableCells(b, r) : [b.rowText(r)], bullet: r.isBullet),
          ]),
      ].where((b) => b.rows.isNotEmpty).toList();

  /// The blocks of page Markdown: '#' headings -> Title, '- ' items ->
  /// List (bullets), '|' rows -> Table, other lines -> Text; a blank line
  /// starts a new block.
  static List<ExportBlock> ofMarkdown(String md) {
    final blocks = <ExportBlock>[];
    var type = '';
    var rows = <ExportRow>[];
    void flush() {
      if (rows.isNotEmpty) blocks.add(ExportBlock(type, rows));
      rows = <ExportRow>[];
      type = '';
    }

    for (final raw in md.split('\n')) {
      final t = raw.trim();
      if (t.isEmpty || RegExp(r'^(\*{3,}|-{3,}|_{3,})$').hasMatch(t)) {
        flush();
        continue;
      }
      final String kind;
      final ExportRow row;
      if (t.startsWith('#')) {
        kind = 'Title';
        row = ExportRow([markdownToPlain(t)]);
      } else if (t.startsWith('|')) {
        if (RegExp(r'^\|?\s*:?-{3,}').hasMatch(t)) continue; // header rule
        kind = 'Table';
        row = ExportRow(t
            .replaceAll(RegExp(r'^\||\|$'), '')
            .split('|')
            .map((c) => markdownToPlain(c.trim()))
            .toList());
      } else if (t.startsWith('- ') || t.startsWith('* ') || t.startsWith('+ ')) {
        kind = 'List';
        row = ExportRow([markdownToPlain(t.substring(2))], bullet: true);
      } else {
        kind = type == 'List' ? 'List' : 'Text';
        row = ExportRow([markdownToPlain(t)]);
      }
      if (type.isNotEmpty && type != kind) flush();
      type = kind;
      rows.add(row);
    }
    flush();
    return blocks;
  }

  static List<String> _tableCells(OcrBlock block, OcrRow row) {
    final slots = List<String>.filled(block.columns < 1 ? 1 : block.columns, '');
    for (final c in row.cells) {
      if (c.text.isEmpty || c.column >= slots.length) continue;
      slots[c.column] = slots[c.column].isEmpty ? c.text : '${slots[c.column]} ${c.text}';
    }
    return slots;
  }

  final String type;
  final List<ExportRow> rows;

  bool get isTable => type == 'Table';
}

/// One page as it is exported: "=== Page N ===" header + its content.
class ExportPage {
  const ExportPage(this.header, this.text, {this.blocks = const []});

  /// The page header in exports is just its number.
  factory ExportPage.of(BookPage page) =>
      ExportPage('${page.number}', page.text,
          blocks: page.result.isCorrected
              ? ExportBlock.ofMarkdown(page.result.markdown)
              : ExportBlock.ofResult(page.result));

  final String header;

  /// Page text: one printed line per text line, blank line between blocks,
  /// TAB between table columns, "• " bullets. Used when [blocks] is empty.
  final String text;

  /// The page layout (headings, paragraphs, lists, tables) - exported with
  /// the same formatting as the printed page.
  final List<ExportBlock> blocks;
}

/// Table columns are separated by TAB in the OCR text; shown as " | ".
String _displayLine(String line) => line.replaceAll('\t', '   |   ');

// --------------------------------------------------------------------------
// PDF
// --------------------------------------------------------------------------

/// Copies [data] into its own buffer starting at offset 0. On a phone,
/// rootBundle.load() returns a view into a larger buffer, and the pdf
/// package's TTF parser reads `data.buffer` ignoring `offsetInBytes`, which
/// threw a RangeError on export.
ByteData _ownBuffer(ByteData data) => ByteData.sublistView(Uint8List.fromList(
    data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes)));

/// Builds the book PDF. [regularFont] / [boldFont] must cover Arabic-script
/// glyphs (Noto Naskh Arabic is bundled); the pdf package shapes Arabic and
/// Urdu letters and lays the lines out right-to-left.
Future<Uint8List> buildBookPdf(List<ExportPage> pages,
    {required ByteData regularFont, required ByteData boldFont, String title = 'HarfScan Book'}) {
  final regular = pw.Font.ttf(_ownBuffer(regularFont));
  final bold = pw.Font.ttf(_ownBuffer(boldFont));
  final doc = pw.Document(
    title: title,
    creator: 'HarfScan',
    // Naskh has no Latin glyphs: headers, digits and "|" fall back to Helvetica.
    theme: pw.ThemeData.withFont(
        base: regular, bold: bold, fontFallback: [pw.Font.helvetica(), pw.Font.helveticaBold()]),
  );
  doc.addPage(
    pw.MultiPage(
      pageFormat: PdfPageFormat.a4,
      margin: const pw.EdgeInsets.fromLTRB(40, 30, 40, 40),
      textDirection: pw.TextDirection.rtl,
      build: (context) => [
        for (final (i, page) in pages.indexed) ...[
          // Every book page on a new sheet; its header is just the number.
          if (i > 0) pw.NewPage(),
          pw.Center(
            child: pw.Text(page.header,
                textDirection: pw.TextDirection.ltr,
                style: const pw.TextStyle(fontSize: 11, color: PdfColors.grey700)),
          ),
          pw.SizedBox(height: 14),
          if (page.blocks.isEmpty)
            for (final line in page.text.split('\n'))
              line.trim().isEmpty
                  ? pw.SizedBox(height: 8)
                  : _pdfLine(_displayLine(line).replaceAll('•', _pdfBullet))
          else
            for (final block in page.blocks) ...[
              ..._pdfBlock(block),
              pw.SizedBox(height: 10),
            ],
        ],
      ],
    ),
  );
  return doc.save();
}

// "•" is in neither font: the Arabic star "٭" is used as the bullet.
const _pdfBullet = '٭';

/// U+200F (right-to-left mark) in front of every line: the bidi package used
/// by pdf throws a RangeError when a line STARTS with a hamza letter and a
/// diacritic (e.g. "أُرِيدُ", "إِيْمَانٌ"); the invisible mark avoids it.
const _rlm = '\u200F';

/// The pdf package lays right-to-left lines out without mirroring paired
/// brackets, so "(۱۹)" came out as ")۱۹(". Swapping them first displays
/// them the right way round.
const _mirror = {'(': ')', ')': '(', '[': ']', ']': '[', '{': '}', '}': '{', '<': '>', '>': '<'};

/// The pdf package does not shape Urdu yeh (U+06CC) when a diacritic follows
/// it inside a word ("ایْمان" lost its dots). In that position Arabic yeh
/// (U+064A) looks identical, so it is drawn instead.
final _urduYehBeforeMark = RegExp('\u06CC(?=[\u064B-\u065F\u0670]+[\u0621-\u064A\u0671-\u06D3])');

String _pdfText(String text) => text.isEmpty
    ? text
    : '$_rlm${text.replaceAll(_urduYehBeforeMark, '\u064A').split('').map((c) => _mirror[c] ?? c).join()}';

pw.Widget _pdfLine(String text,
        {double size = 13, bool bold = false, pw.TextAlign align = pw.TextAlign.right}) =>
    pw.Text(_pdfText(text),
        textDirection: pw.TextDirection.rtl,
        textAlign: align,
        style: pw.TextStyle(
            fontSize: size, lineSpacing: 4, fontWeight: bold ? pw.FontWeight.bold : null));

/// One block laid out like the printed page.
List<pw.Widget> _pdfBlock(ExportBlock block) {
  switch (block.type) {
    case 'Title':
      return [
        for (final r in block.rows)
          pw.Container(
            width: double.infinity,
            alignment: pw.Alignment.center,
            child: _pdfLine(r.cells.join(' '), size: 17, bold: true, align: pw.TextAlign.center),
          ),
      ];
    case 'Table':
      final cols = block.rows.fold<int>(1, (n, r) => r.cells.length > n ? r.cells.length : n);
      return [
        pw.Table(
          border: pw.TableBorder.all(color: PdfColors.grey600, width: 0.6),
          defaultVerticalAlignment: pw.TableCellVerticalAlignment.middle,
          children: [
            for (final r in block.rows)
              pw.TableRow(children: [
                // Column 0 is the right-most column of the book: PDF tables
                // are laid out left-to-right, so the cells go in reverse.
                for (var c = cols - 1; c >= 0; c--)
                  pw.Padding(
                    padding: const pw.EdgeInsets.symmetric(horizontal: 6, vertical: 3),
                    child: _pdfLine(c < r.cells.length ? r.cells[c] : ''),
                  ),
              ]),
          ],
        ),
      ];
    default: // Text, List: one line per printed line
      return [
        for (final r in block.rows)
          _pdfLine(r.bullet ? '$_pdfBullet ${r.cells.join(' ')}' : r.cells.join(' ')),
      ];
  }
}

/// Builds the book PDF from page images rendered by the app (see
/// page_capture.dart): every book page starts on a new A4 sheet, long pages
/// continue on the next sheet, and the only header is the page number.
Future<Uint8List> buildImagePdf(List<List<Uint8List>> pages, {String title = 'HarfScan Book'}) {
  final doc = pw.Document(title: title, creator: 'HarfScan');
  for (var i = 0; i < pages.length; i++) {
    for (final slice in pages[i]) {
      doc.addPage(pw.Page(
        pageFormat: PdfPageFormat.a4,
        margin: const pw.EdgeInsets.fromLTRB(40, 30, 40, 40),
        build: (context) => pw.Column(
          crossAxisAlignment: pw.CrossAxisAlignment.stretch,
          children: [
            pw.Center(
              child: pw.Text('${i + 1}',
                  style: const pw.TextStyle(fontSize: 11, color: PdfColors.grey700)),
            ),
            pw.SizedBox(height: 14),
            pw.Image(pw.MemoryImage(slice), fit: pw.BoxFit.fitWidth, alignment: pw.Alignment.topCenter),
          ],
        ),
      ));
    }
  }
  return doc.save();
}

// --------------------------------------------------------------------------
// Word (.docx)
// --------------------------------------------------------------------------

String _xmlEscape(String s) => s
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');

const _font = 'Noto Naskh Arabic';

String _rtlParagraph(String text) => '<w:p><w:pPr><w:bidi/><w:jc w:val="right"/>'
    '<w:spacing w:after="80" w:line="360" w:lineRule="auto"/></w:pPr>'
    '<w:r><w:rPr><w:rFonts w:ascii="$_font" w:hAnsi="$_font" w:cs="$_font"/>'
    '<w:rtl/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr>'
    '<w:t xml:space="preserve">${_xmlEscape(text)}</w:t></w:r></w:p>';

/// Page number at the top of each book page (centred, small, grey).
String _headerParagraph(String text) => '<w:p><w:pPr><w:jc w:val="center"/>'
    '<w:spacing w:before="0" w:after="200"/></w:pPr>'
    '<w:r><w:rPr><w:color w:val="666666"/><w:sz w:val="22"/></w:rPr>'
    '<w:t xml:space="preserve">${_xmlEscape(text)}</w:t></w:r></w:p>';

const _pageBreak = '<w:p><w:r><w:br w:type="page"/></w:r></w:p>';

String _run(String text, {int size = 28, bool bold = false}) =>
    '<w:r><w:rPr><w:rFonts w:ascii="$_font" w:hAnsi="$_font" w:cs="$_font"/>'
    '${bold ? '<w:b/><w:bCs/>' : ''}<w:rtl/><w:sz w:val="$size"/><w:szCs w:val="$size"/></w:rPr>'
    '<w:t xml:space="preserve">${_xmlEscape(text)}</w:t></w:r>';

String _titleParagraph(String text) => '<w:p><w:pPr><w:bidi/><w:jc w:val="center"/>'
    '<w:spacing w:before="120" w:after="120"/></w:pPr>${_run(text, size: 34, bold: true)}</w:p>';

String _cellParagraph(String text) => '<w:p><w:pPr><w:bidi/><w:jc w:val="right"/>'
    '<w:spacing w:after="0"/></w:pPr>${_run(text, size: 26)}</w:p>';

/// A bordered table; w:bidiVisual puts the first cell (column 0) on the right.
String _docxTable(ExportBlock block) {
  final cols = block.rows.fold<int>(1, (n, r) => r.cells.length > n ? r.cells.length : n);
  const border = 'w:val="single" w:sz="4" w:space="0" w:color="808080"';
  final out = StringBuffer('<w:tbl><w:tblPr><w:bidiVisual/><w:tblW w:w="5000" w:type="pct"/>'
      '<w:tblBorders><w:top $border/><w:left $border/><w:bottom $border/><w:right $border/>'
      '<w:insideH $border/><w:insideV $border/></w:tblBorders></w:tblPr><w:tblGrid>');
  for (var c = 0; c < cols; c++) {
    out.write('<w:gridCol/>');
  }
  out.write('</w:tblGrid>');
  for (final r in block.rows) {
    out.write('<w:tr>');
    for (var c = 0; c < cols; c++) {
      out.write('<w:tc><w:tcPr><w:tcW w:w="0" w:type="auto"/></w:tcPr>'
          '${_cellParagraph(c < r.cells.length ? r.cells[c] : '')}</w:tc>');
    }
    out.write('</w:tr>');
  }
  out.write('</w:tbl><w:p/>');
  return out.toString();
}

String _docxBlock(ExportBlock block) {
  switch (block.type) {
    case 'Title':
      return block.rows.map((r) => _titleParagraph(r.cells.join(' '))).join();
    case 'Table':
      return _docxTable(block);
    default:
      return '${block.rows.map((r) => _rtlParagraph(r.bullet ? '• ${r.cells.join(' ')}' : r.cells.join(' '))).join()}<w:p/>';
  }
}

/// Builds a minimal, valid WordprocessingML document: each book page starts
/// on a new Word page with its "=== Page N ===" header, followed by the page
/// laid out like the book: centred bold headings, one right-to-left
/// paragraph per printed line, bullets, and real bordered tables.
Uint8List buildBookDocx(List<ExportPage> pages) {
  final body = StringBuffer();
  for (var i = 0; i < pages.length; i++) {
    if (i > 0) body.write(_pageBreak);
    body.write(_headerParagraph(pages[i].header));
    if (pages[i].blocks.isNotEmpty) {
      pages[i].blocks.map(_docxBlock).forEach(body.write);
      continue;
    }
    for (final line in pages[i].text.split('\n')) {
      body.write(line.trim().isEmpty ? '<w:p/>' : _rtlParagraph(_displayLine(line)));
    }
  }
  final document = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
      '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
      '<w:body>$body<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
      '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" '
      'w:header="708" w:footer="708" w:gutter="0"/><w:bidi/></w:sectPr></w:body></w:document>';
  const contentTypes = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
      '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
      '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
      '<Default Extension="xml" ContentType="application/xml"/>'
      '<Override PartName="/word/document.xml" '
      'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
      '</Types>';
  const rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
      '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
      '<Relationship Id="rId1" '
      'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
      'Target="word/document.xml"/></Relationships>';

  final archive = Archive()
    ..addFile(ArchiveFile.string('[Content_Types].xml', contentTypes))
    ..addFile(ArchiveFile.string('_rels/.rels', rels))
    ..addFile(ArchiveFile.string('word/document.xml', document));
  return ZipEncoder().encodeBytes(archive);
}

// --------------------------------------------------------------------------
// Save + share
// --------------------------------------------------------------------------

enum ExportFormat { pdf, docx }

class ExportService {
  const ExportService._();

  /// Builds the file for the whole book, saves it in the app's documents
  /// folder and opens the share sheet. Returns the saved file, or null if
  /// the user left the export screen.
  ///
  /// The PDF is made from the pages as the app draws them (same fonts and
  /// layout as on screen); if that fails, from the text with the bundled
  /// Naskh font.
  static Future<File?> exportBook(BuildContext context, BookSession session, ExportFormat format) async {
    final pages = session.pages.map(ExportPage.of).toList();
    final Uint8List bytes;
    final String extension;
    switch (format) {
      case ExportFormat.pdf:
        List<List<Uint8List>>? images;
        try {
          images = await capturePages(context, session.pages);
        } catch (e) {
          debugPrint('Rendered PDF not possible, using the text PDF: $e');
        }
        bytes = images != null && images.length == pages.length
            ? await buildImagePdf(images, title: session.title)
            : await buildBookPdf(
                pages,
                title: session.title,
                regularFont: await rootBundle.load('assets/fonts/NotoNaskhArabic-Regular.ttf'),
                boldFont: await rootBundle.load('assets/fonts/NotoNaskhArabic-Bold.ttf'),
              );
        extension = 'pdf';
      case ExportFormat.docx:
        bytes = buildBookDocx(pages);
        extension = 'docx';
    }
    final dir = await getApplicationDocumentsDirectory();
    final stamp = DateTime.now().toIso8601String().replaceAll(RegExp(r'[:.]'), '-');
    final name = session.title.replaceAll(RegExp(r'[^\w\u0600-\u06FF -]'), '').trim();
    final file = File('${dir.path}/${name.isEmpty ? 'HarfScan_Book' : name}_$stamp.$extension');
    await file.writeAsBytes(bytes, flush: true);

    await SharePlus.instance.share(ShareParams(
      files: [XFile(file.path)],
      subject: '${session.title} (${session.pageCount} pages)',
    ));
    return file;
  }
}
