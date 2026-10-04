// book_workspace_screen.dart
//
// Screen 3 - Multi-Page Book Workspace & Exporter.
// Every extracted page is appended to one master document:
//
//     === Page 1 ===
//     [text]
//     === Page 2 ===
//     [text] ...
//
// Words read with low confidence (threshold in Settings) are highlighted in
// gold for review.
// "Scan Next Page" adds the next page; the toolbar exports the whole book.

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import 'api_service.dart';
import 'app_settings.dart';
import 'book_session.dart';
import 'select_text_screen.dart';
import 'export_service.dart';
import 'theme.dart';

class BookWorkspaceScreen extends StatefulWidget {
  const BookWorkspaceScreen({super.key, required this.session, required this.settings});

  final BookSession session;
  final AppSettings settings;

  @override
  State<BookWorkspaceScreen> createState() => _BookWorkspaceScreenState();
}

class _BookWorkspaceScreenState extends State<BookWorkspaceScreen> {
  bool _exporting = false;

  BookSession get _session => widget.session;

  Future<void> _scanNextPage() async {
    final source = await showModalBottomSheet<ImageSource>(
      context: context,
      backgroundColor: HarfColors.slate,
      builder: (ctx) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Padding(
              padding: const EdgeInsets.all(16),
              child: Text('Add page ${_session.pageCount + 1}',
                  style: Theme.of(ctx).textTheme.titleMedium?.copyWith(color: HarfColors.gold)),
            ),
            ListTile(
              leading: const Icon(Icons.photo_camera, color: HarfColors.gold),
              title: const Text('Camera Scan'),
              onTap: () => Navigator.pop(ctx, ImageSource.camera),
            ),
            ListTile(
              leading: const Icon(Icons.photo_library, color: HarfColors.gold),
              title: const Text('Gallery Upload'),
              onTap: () => Navigator.pop(ctx, ImageSource.gallery),
            ),
            const SizedBox(height: 8),
          ],
        ),
      ),
    );
    if (source == null || !mounted) return;
    await SelectTextScreen.pickAndSelect(context,
        source: source, session: _session, settings: widget.settings);
  }

  Future<void> _export(ExportFormat format) async {
    if (_session.isEmpty) return;
    setState(() => _exporting = true);
    try {
      final file = await ExportService.exportBook(_session, format);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Saved ${file.uri.pathSegments.last}')));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Export failed: $e')));
    } finally {
      if (mounted) setState(() => _exporting = false);
    }
  }

  Future<void> _confirmRemove(BookPage page) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: HarfColors.slate,
        title: Text('Remove ${page.header}?'),
        content: const Text('The following pages are renumbered.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Remove')),
        ],
      ),
    );
    if (ok == true) _session.removePage(page);
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Listenable.merge([_session, widget.settings]),
      builder: (context, _) => Scaffold(
        extendBodyBehindAppBar: true,
        appBar: AppBar(
          title: Text('Book Workspace (${_session.pageCount})'),
          actions: [
            IconButton(
              tooltip: 'Export as PDF',
              icon: const Icon(Icons.picture_as_pdf),
              onPressed: _session.isEmpty || _exporting ? null : () => _export(ExportFormat.pdf),
            ),
            IconButton(
              tooltip: 'Export as Word (.docx)',
              icon: const Icon(Icons.description),
              onPressed: _session.isEmpty || _exporting ? null : () => _export(ExportFormat.docx),
            ),
            IconButton(
              tooltip: 'Copy whole book',
              icon: const Icon(Icons.copy_all),
              onPressed: _session.isEmpty
                  ? null
                  : () {
                      Clipboard.setData(ClipboardData(text: _session.masterText));
                      ScaffoldMessenger.of(context)
                          .showSnackBar(const SnackBar(content: Text('Book copied to clipboard')));
                    },
            ),
          ],
        ),
        floatingActionButton: FloatingActionButton.extended(
          onPressed: _exporting ? null : _scanNextPage,
          icon: const Icon(Icons.add),
          label: const Text('Scan Next Page'),
        ),
        body: HarfBackground(
          child: SafeArea(
            child: Column(
              children: [
                if (_exporting) const LinearProgressIndicator(color: HarfColors.brightGold),
                Expanded(child: _session.isEmpty ? _buildEmpty() : _buildPages()),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildEmpty() => Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Text('No pages yet.\nTap "Scan Next Page" to add page 1.',
              textAlign: TextAlign.center, style: Theme.of(context).textTheme.titleMedium),
        ),
      );

  Widget _buildPages() {
    final s = widget.settings;
    final low = s.highlightLowConfidence ? _session.lowConfidenceCount(s.lowConfidenceThreshold) : 0;
    return ListView(
      padding: const EdgeInsets.fromLTRB(12, 4, 12, 96),
      children: [
        if (low > 0) _ReviewLegend(count: low, threshold: s.lowConfidenceThreshold),
        for (final page in _session.pages)
          _PageCard(page: page, isLow: s.isLowConfidence, onRemove: () => _confirmRemove(page)),
      ],
    );
  }
}

class _ReviewLegend extends StatelessWidget {
  const _ReviewLegend({required this.count, required this.threshold});

  final int count;
  final double threshold;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 8),
      child: Row(
        children: [
          Container(
            width: 18,
            height: 18,
            decoration: BoxDecoration(
                color: HarfColors.lowConfidence, borderRadius: BorderRadius.circular(4)),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
                '$count word(s) read with < ${threshold.round()}% confidence - please review',
                style: Theme.of(context).textTheme.bodySmall),
          ),
        ],
      ),
    );
  }
}

/// One page of the master document: "=== Page N ===" + its text, laid out
/// like the printed page (title, paragraphs, bullets, table columns).
class _PageCard extends StatelessWidget {
  const _PageCard({required this.page, required this.isLow, required this.onRemove});

  final BookPage page;
  final bool Function(double confidence) isLow;
  final VoidCallback onRemove;

  @override
  Widget build(BuildContext context) {
    final blocks = page.result.blocks;
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 8, 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Text(page.header,
                    style: const TextStyle(
                        color: HarfColors.brightGold, fontWeight: FontWeight.w700, letterSpacing: 1)),
                const Spacer(),
                IconButton(
                  tooltip: 'Copy page',
                  icon: const Icon(Icons.copy, size: 20),
                  onPressed: () {
                    Clipboard.setData(ClipboardData(text: '${page.header}\n${page.text}'));
                    ScaffoldMessenger.of(context)
                        .showSnackBar(SnackBar(content: Text('${page.header} copied')));
                  },
                ),
                IconButton(
                  tooltip: 'Remove page',
                  icon: const Icon(Icons.delete_outline, size: 20),
                  onPressed: onRemove,
                ),
              ],
            ),
            const Divider(color: HarfColors.gold, height: 8),
            if (blocks.isEmpty)
              const Padding(
                padding: EdgeInsets.all(12),
                child: Text('- no text found on this page -'),
              )
            else
              Directionality(
                textDirection: TextDirection.rtl,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    for (final block in blocks) ...[
                      _BlockView(block: block, isLow: isLow),
                      const SizedBox(height: 12),
                    ],
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _BlockView extends StatelessWidget {
  const _BlockView({required this.block, required this.isLow});

  final OcrBlock block;
  final bool Function(double confidence) isLow;

  @override
  Widget build(BuildContext context) {
    final title = block.type == 'Title';
    return Column(
      crossAxisAlignment: title ? CrossAxisAlignment.center : CrossAxisAlignment.start,
      children: [
        for (final row in block.rows)
          Wrap(
            alignment: title ? WrapAlignment.center : WrapAlignment.start,
            spacing: block.isTable ? 4 : 6,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              if (row.isBullet) Text('•', style: scriptStyle('urdu')),
              for (var i = 0; i < row.cells.length; i++) ...[
                if (block.isTable && i > 0)
                  const Text('|', style: TextStyle(color: HarfColors.gold)),
                _Word(cell: row.cells[i], bold: title, isLow: isLow),
              ],
            ],
          ),
      ],
    );
  }
}

/// A recognised word / box. Low-confidence words get a gold background.
class _Word extends StatelessWidget {
  const _Word({required this.cell, required this.bold, required this.isLow});

  final OcrCell cell;
  final bool bold;
  final bool Function(double confidence) isLow;

  @override
  Widget build(BuildContext context) {
    if (cell.text.isEmpty) return const SizedBox.shrink();
    final low = isLow(cell.confidence);
    final style = scriptStyle(cell.language, size: bold ? 22 : 19)
        .copyWith(fontWeight: bold ? FontWeight.w700 : FontWeight.w400);
    return Tooltip(
      message: cell.engine == 'Gemini'
          ? 'Read by Gemini AI'
          : '${cell.engine} - ${cell.confidence.toStringAsFixed(0)}% confidence',
      triggerMode: TooltipTriggerMode.tap,
      child: Container(
        padding: low ? const EdgeInsets.symmetric(horizontal: 4) : EdgeInsets.zero,
        decoration: low
            ? BoxDecoration(color: HarfColors.lowConfidence, borderRadius: BorderRadius.circular(4))
            : null,
        child: Text(cell.text, style: style),
      ),
    );
  }
}
