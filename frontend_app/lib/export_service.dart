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
import 'package:path_provider/path_provider.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:share_plus/share_plus.dart';

import 'book_session.dart';

/// One page as it is exported: "=== Page N ===" header + text.
class ExportPage {
  const ExportPage(this.header, this.text);

  factory ExportPage.of(BookPage page) => ExportPage(page.header, page.text);

  final String header;

  /// Page text: one printed line per text line, blank line between blocks,
  /// TAB between table columns, "• " bullets.
  final String text;
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
  const gold = PdfColor.fromInt(0xFFD4AF37);
  const navy = PdfColor.fromInt(0xFF0F2027);

  doc.addPage(
    pw.MultiPage(
      pageFormat: PdfPageFormat.a4,
      margin: const pw.EdgeInsets.fromLTRB(40, 40, 40, 48),
      textDirection: pw.TextDirection.rtl,
      header: (context) => pw.Container(
        alignment: pw.Alignment.centerLeft,
        margin: const pw.EdgeInsets.only(bottom: 12),
        child: pw.Text('$title  -  ${context.pageNumber}/${context.pagesCount}',
            textDirection: pw.TextDirection.ltr,
            style: const pw.TextStyle(fontSize: 9, color: PdfColors.grey600)),
      ),
      build: (context) => [
        for (final page in pages) ...[
          pw.Container(
            width: double.infinity,
            margin: const pw.EdgeInsets.only(top: 14, bottom: 8),
            padding: const pw.EdgeInsets.symmetric(vertical: 4, horizontal: 8),
            decoration: const pw.BoxDecoration(
              color: navy,
              borderRadius: pw.BorderRadius.all(pw.Radius.circular(4)),
            ),
            child: pw.Text(page.header,
                textDirection: pw.TextDirection.ltr,
                style: pw.TextStyle(color: gold, fontWeight: pw.FontWeight.bold, fontSize: 12)),
          ),
          for (final line in page.text.split('\n'))
            line.trim().isEmpty
                ? pw.SizedBox(height: 8)
                // "•" is in neither font: use the Arabic star "٭" as the bullet.
                : pw.Text(_displayLine(line).replaceAll('•', '٭'),
                    textDirection: pw.TextDirection.rtl,
                    textAlign: pw.TextAlign.right,
                    style: const pw.TextStyle(fontSize: 13, lineSpacing: 4)),
        ],
      ],
    ),
  );
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

String _headerParagraph(String text) => '<w:p><w:pPr><w:spacing w:before="240" w:after="120"/>'
    '<w:shd w:val="clear" w:color="auto" w:fill="0F2027"/></w:pPr>'
    '<w:r><w:rPr><w:b/><w:color w:val="D4AF37"/><w:sz w:val="24"/></w:rPr>'
    '<w:t xml:space="preserve">${_xmlEscape(text)}</w:t></w:r></w:p>';

const _pageBreak = '<w:p><w:r><w:br w:type="page"/></w:r></w:p>';

/// Builds a minimal, valid WordprocessingML document: each book page starts
/// on a new Word page with its "=== Page N ===" header; every text line is a
/// right-to-left paragraph.
Uint8List buildBookDocx(List<ExportPage> pages) {
  final body = StringBuffer();
  for (var i = 0; i < pages.length; i++) {
    if (i > 0) body.write(_pageBreak);
    body.write(_headerParagraph(pages[i].header));
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
  /// folder and opens the share sheet. Returns the saved file.
  static Future<File> exportBook(BookSession session, ExportFormat format) async {
    final pages = session.pages.map(ExportPage.of).toList();
    final Uint8List bytes;
    final String extension;
    switch (format) {
      case ExportFormat.pdf:
        bytes = await buildBookPdf(
          pages,
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
    final file = File('${dir.path}/HarfScan_Book_$stamp.$extension');
    await file.writeAsBytes(bytes, flush: true);

    await SharePlus.instance.share(ShareParams(
      files: [XFile(file.path)],
      subject: 'HarfScan book (${session.pageCount} pages)',
    ));
    return file;
  }
}
