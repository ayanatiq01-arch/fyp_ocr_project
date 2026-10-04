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
  const MarkdownPage({super.key, required this.markdown, required this.language});

  final String markdown;
  final String language;

  @override
  Widget build(BuildContext context) {
    final base = scriptStyle(language, size: 18);
    final sheet = MarkdownStyleSheet(
      p: base,
      h2: base.copyWith(fontSize: (base.fontSize ?? 18) * 1.25, fontWeight: FontWeight.w700,
          color: HarfColors.brightGold),
      h2Align: WrapAlignment.center,
      listBullet: base.copyWith(color: HarfColors.gold),
      tableHead: base,
      tableBody: base,
      tableBorder: TableBorder.all(color: HarfColors.gold.withValues(alpha: 0.45), width: 0.8),
      tableCellsPadding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      tableColumnWidth: const IntrinsicColumnWidth(),
      blockSpacing: 10,
      textAlign: WrapAlignment.start,
    );
    return Directionality(
      textDirection: TextDirection.rtl,
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal, // wide tables scroll instead of overflowing
        reverse: true,
        child: ConstrainedBox(
          constraints: BoxConstraints(minWidth: MediaQuery.sizeOf(context).width - 72),
          child: IntrinsicWidth(
            child: MarkdownBody(
              data: markdown,
              styleSheet: sheet,
              extensionSet: md.ExtensionSet(
                md.ExtensionSet.gitHubFlavored.blockSyntaxes,
                [_MarkSyntax(), ...md.ExtensionSet.gitHubFlavored.inlineSyntaxes],
              ),
              builders: {'mark': _MarkBuilder()},
            ),
          ),
        ),
      ),
    );
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
