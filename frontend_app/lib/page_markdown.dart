// page_markdown.dart
//
// A digitised page as Markdown (same rules as the backend's
// ocr_pipeline.blocks_to_markdown) and the widget that renders it:
//
//   Title          -> "## heading"               (centred, bold)
//   Text / List    -> one line per printed line  (hard line breaks), "- " bullets
//   Table          -> Markdown table, column 0 (right-most in the book) first;
//                     rendered right-to-left so it is on the right again
//
// Words read with low confidence are wrapped in ==...== and shown with a
// gold background.

import 'package:flutter/material.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:markdown/markdown.dart' as md;

import 'api_service.dart';
import 'theme.dart';

final _mdInline = RegExp(r'([\\`*_\[\]<>|~=])');
final _mdLineStart = RegExp(r'^(\s*)([#>+\-]|\d+[.)])');

/// Escapes Markdown syntax characters so OCR text is shown literally.
String mdEscape(String text, {bool table = false}) {
  final t = text.split(RegExp(r'\s+')).where((w) => w.isNotEmpty).join(' ')
      .replaceAllMapped(_mdInline, (m) => '\\${m[1]}');
  return table ? t : t.replaceFirstMapped(_mdLineStart, (m) => '${m[1]}\\${m[2]}');
}

/// Markdown for the page. [isLow] marks cells to highlight for review.
String pageMarkdown(OcrResult result, {bool Function(double confidence)? isLow}) {
  String cellText(OcrCell c, {bool table = false}) {
    final t = mdEscape(c.text, table: table);
    return (isLow != null && isLow(c.confidence)) ? '==$t==' : t;
  }

  final out = <String>[];
  for (final block in result.blocks) {
    if (block.isTable) {
      final cols = block.columns < 1 ? 1 : block.columns;
      final lines = <String>[];
      for (var i = 0; i < block.rows.length; i++) {
        final slots = List<String>.filled(cols, '');
        for (final c in block.rows[i].cells) {
          if (c.text.isEmpty || c.column >= cols) continue;
          final t = cellText(c, table: true);
          slots[c.column] = slots[c.column].isEmpty ? t : '${slots[c.column]} $t';
        }
        lines.add('| ${slots.map((s) => s.isEmpty ? ' ' : s).join(' | ')} |');
        if (i == 0) lines.add('|${List.filled(cols, ' --- ').join('|')}|');
      }
      if (lines.isNotEmpty) out.add(lines.join('\n'));
      continue;
    }
    final lines = <String>[];
    for (final row in block.rows) {
      final text = row.cells.where((c) => c.text.isNotEmpty).map(cellText).join(' ');
      if (text.isEmpty) continue;
      if (block.type == 'Title') {
        lines.add('## $text');
      } else if (row.isBullet) {
        lines.add('- $text');
      } else {
        lines.add(text);
      }
    }
    if (lines.isEmpty) continue;
    if (block.type == 'Title') {
      out.add(lines.join('\n\n'));
    } else {
      final joined = <String>[];
      for (var i = 0; i < lines.length; i++) {
        final hardBreak = i < lines.length - 1 &&
            !lines[i].startsWith('- ') &&
            !lines[i + 1].startsWith('- ');
        joined.add(hardBreak ? '${lines[i]}  ' : lines[i]);
      }
      out.add(joined.join('\n'));
    }
  }
  return out.isEmpty ? '' : '${out.join('\n\n')}\n';
}

/// What is shown, printed and exported for a page: the vision model's
/// corrected Markdown, or the rough draft as Markdown ([isLow] marks the
/// uncertain words of a rough draft).
String displayMarkdown(OcrResult result, {bool Function(double confidence)? isLow}) =>
    pageMarkdown(result, isLow: result.isCorrected ? null : isLow);

final _mdTableRule = RegExp(r'^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?$');
final _mdEscape = RegExp(r'\\([\\`*_\[\]<>|~=#+\-.!()])');

/// Plain text of page Markdown (for Copy and the master text): headings
/// without '#', bullets as '• ', table cells separated by TAB, inline
/// markers and escapes removed.
String markdownToPlain(String md) {
  final out = <String>[];
  for (var line in md.split('\n')) {
    line = line.trimRight();
    final t = line.trim();
    if (_mdTableRule.hasMatch(t) || RegExp(r'^(\*{3,}|-{3,}|_{3,})$').hasMatch(t)) continue;
    if (t.startsWith('#')) {
      line = t.replaceFirst(RegExp(r'^#+\s*'), '');
    } else if (t.startsWith('- ') || t.startsWith('* ') || t.startsWith('+ ')) {
      line = '• ${t.substring(2)}';
    } else if (t.startsWith('|')) {
      line = t.replaceAll(RegExp(r'^\||\|$'), '').split('|').map((c) => c.trim()).join('\t');
    }
    line = line
        .replaceAll(RegExp(r'(\*\*|__|==)'), '')
        .replaceAllMapped(_mdEscape, (m) => m[1]!);
    out.add(line);
  }
  return out.join('\n').replaceAll(RegExp(r'\n{3,}'), '\n\n').trim();
}

/// Most common script on the page ("urdu" / "arabic"), for the reading font.
String pageLanguage(OcrResult result) {
  var urdu = 0, arabic = 0;
  for (final c in result.cells) {
    if (c.language == 'arabic') arabic += c.text.length;
    if (c.language == 'urdu') urdu += c.text.length;
  }
  return arabic > urdu ? 'arabic' : 'urdu';
}

/// Renders page Markdown right-to-left in the book's reading font.
class MarkdownPage extends StatelessWidget {
  const MarkdownPage(
      {super.key, required this.markdown, required this.language, this.paper = false});

  final String markdown;
  final String language;

  /// Printed look (dark text on white, for the PDF) instead of the app's
  /// gold-on-navy look.
  final bool paper;

  @override
  Widget build(BuildContext context) {
    final ink = paper ? Colors.black : HarfColors.ink;
    final accent = paper ? HarfColors.navy : HarfColors.brightGold;
    final base = scriptStyle(language, size: paper ? 16 : 18, color: ink);
    TextStyle heading(double scale) => base.copyWith(
        fontSize: (base.fontSize ?? 18) * scale, fontWeight: FontWeight.w700, color: accent);
    final sheet = MarkdownStyleSheet(
      p: base,
      // Every heading level in the reading font (Gemini uses # .. ###),
      // centred like headings in the book.
      h1: heading(1.3),
      h2: heading(1.22),
      h3: heading(1.15),
      h4: heading(1.1),
      h5: heading(1.05),
      h6: heading(1.0),
      h1Align: WrapAlignment.center,
      h2Align: WrapAlignment.center,
      h3Align: WrapAlignment.center,
      h4Align: WrapAlignment.center,
      h5Align: WrapAlignment.center,
      h6Align: WrapAlignment.center,
      strong: const TextStyle(fontWeight: FontWeight.w700),
      listBullet: base.copyWith(color: paper ? Colors.black : HarfColors.gold),
      tableHead: base,
      tableBody: base,
      tableBorder: TableBorder.all(
          color: paper ? Colors.black54 : HarfColors.gold.withValues(alpha: 0.45), width: 0.8),
      tableCellsPadding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      tableColumnWidth: const IntrinsicColumnWidth(),
      blockSpacing: 10,
      textAlign: WrapAlignment.start,
    );
    final body = MarkdownBody(
      data: markArabicLines(markdown),
      styleSheet: sheet,
      extensionSet: md.ExtensionSet(
        md.ExtensionSet.gitHubFlavored.blockSyntaxes,
        [_ArabicSyntax(), _MarkSyntax(), ...md.ExtensionSet.gitHubFlavored.inlineSyntaxes],
      ),
      builders: {'mark': _MarkBuilder(), 'ar': _ArabicBuilder()},
    );
    if (paper) return Directionality(textDirection: TextDirection.rtl, child: body);
    return Directionality(
      textDirection: TextDirection.rtl,
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal, // wide tables scroll instead of overflowing
        reverse: true,
        child: ConstrainedBox(
          constraints: BoxConstraints(minWidth: MediaQuery.sizeOf(context).width - 72),
          child: IntrinsicWidth(child: body),
        ),
      ),
    );
  }
}

final _arabicOnly = RegExp(r'[\u0643\u064A\u0629\u0649\u0623\u0625]'); // ك ي ة ى أ إ
final _harakat = RegExp(r'[\u064B-\u0652\u0670]');
final _letters = RegExp(r'[\u0621-\u064A\u0671-\u06D3]');

/// True for Arabic text (Quran verses, duas): vocalised (one haraka per 4
/// letters or more) or with Arabic-only letters. Same rule as the server's
/// ``looks_arabic``.
bool looksArabic(String text) {
  final letters = _letters.allMatches(text).length;
  if (letters < 3) return false;
  return _arabicOnly.allMatches(text).length / letters >= 0.08 ||
      _harakat.allMatches(text).length / letters >= 0.25;
}

const _arOpen = '\u2045', _arClose = '\u2046'; // ⁅ ⁆ - never in book text

/// Wraps the text of Arabic lines in ⁅...⁆ so they are drawn in Naskh (as
/// Arabic is printed) while Urdu lines stay in Nastaliq. Table rows are
/// left alone (they mix both scripts by cell).
String markArabicLines(String markdown) => markdown.split('\n').map((line) {
      final m = RegExp(r'^(\s*(?:#{1,6}\s+|[-*+]\s+)?)(.*?)(\s*)$').firstMatch(line)!;
      final body = m[2]!;
      if (body.isEmpty || body.startsWith('|') || !looksArabic(body)) return line;
      return '${m[1]}$_arOpen${body.replaceAll(_arOpen, '').replaceAll(_arClose, '')}$_arClose${m[3]}';
    }).join('\n');

/// `⁅text⁆` -> `<ar>text</ar>`
class _ArabicSyntax extends md.InlineSyntax {
  _ArabicSyntax() : super('$_arOpen([^$_arClose]*)$_arClose');

  @override
  bool onMatch(md.InlineParser parser, Match match) {
    parser.addNode(md.Element.text('ar', match[1]!));
    return true;
  }
}

class _ArabicBuilder extends MarkdownElementBuilder {
  @override
  Widget? visitElementAfterWithContext(BuildContext context, md.Element element,
      TextStyle? preferredStyle, TextStyle? parentStyle) {
    final style = parentStyle ?? preferredStyle ?? const TextStyle();
    final naskh = scriptStyle('arabic', size: style.fontSize ?? 18, color: style.color)
        .copyWith(fontWeight: style.fontWeight);
    return Text.rich(TextSpan(text: element.textContent, style: naskh));
  }
}

/// ==text== -> <mark>text</mark>
class _MarkSyntax extends md.InlineSyntax {
  _MarkSyntax() : super(r'==([^=\n]+)==');

  @override
  bool onMatch(md.InlineParser parser, Match match) {
    parser.addNode(md.Element.text('mark', match[1]!));
    return true;
  }
}

class _MarkBuilder extends MarkdownElementBuilder {
  @override
  Widget? visitElementAfterWithContext(BuildContext context, md.Element element,
      TextStyle? preferredStyle, TextStyle? parentStyle) {
    return Text.rich(TextSpan(
      text: element.textContent,
      style: (parentStyle ?? preferredStyle)?.copyWith(backgroundColor: HarfColors.lowConfidence),
    ));
  }
}
