// select_text_screen.dart
//
// Page 3 - Select text (like Google Lens).
// The whole photo is sent to the backend and read; every recognised word is
// then laid over the photo. The user puts a finger on the FIRST word and
// drags to the LAST word - everything in between (in reading order, across
// lines) is selected. The gold handles can be dragged to adjust the
// selection, a magnifier shows the words under the finger, and
// "Add to Book" appends the selected text as the next book page.

import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import 'api_service.dart';
import 'app_settings.dart';
import 'book_session.dart';
import 'theme.dart';

class SelectTextScreen extends StatefulWidget {
  const SelectTextScreen(
      {super.key, required this.image, required this.session, required this.settings});

  final File image;
  final BookSession session;
  final AppSettings settings;

  /// Picks a photo (camera or gallery), opens the selection screen and
  /// returns true if a page was added to the book.
  static Future<bool> pickAndSelect(BuildContext context,
      {required ImageSource source,
      required BookSession session,
      required AppSettings settings}) async {
    final messenger = ScaffoldMessenger.of(context);
    final navigator = Navigator.of(context);
    try {
      final XFile? photo = await ImagePicker().pickImage(
        source: source,
        preferredCameraDevice: CameraDevice.rear,
        requestFullMetadata: false,
        // At most 2400 px: plenty for small print (the server reads pages at
        // 2000 px) and the upload is several times smaller and faster.
        maxWidth: 2400,
        maxHeight: 2400,
        imageQuality: 92,
      );
      if (photo == null) return false; // user cancelled
      final added = await navigator.push<bool>(MaterialPageRoute(
        builder: (_) =>
            SelectTextScreen(image: File(photo.path), session: session, settings: settings),
      ));
      return added ?? false;
    } on PlatformException catch (e) {
      messenger.showSnackBar(SnackBar(
          content: Text(
              'Could not open ${source == ImageSource.camera ? 'camera' : 'gallery'}: ${e.message}')));
      return false;
    }
  }

  @override
  State<SelectTextScreen> createState() => _SelectTextScreenState();
}

/// One selectable word laid over the photo (image pixel coordinates).
class _Word {
  const _Word(this.block, this.row, this.cell, this.text, this.rect);

  final int block, row, cell;
  final String text;
  final Rect rect;
}

enum _Phase { reading, selecting, failed }

class _SelectTextScreenState extends State<SelectTextScreen> {
  StreamSubscription<OcrEvent>? _scan;
  _Phase _phase = _Phase.reading;
  bool _retried = false; // one automatic retry after finding the server again
  String _stage = ''; // 'ai_correcting' (vision model) or 'searching' (server)
  String _error = '';
  OcrResult? _result;
  List<_Word> _words = const [];
  List<Rect> _boxes = const []; // line / table-cell boxes, shown on the photo

  // Selection = words [min(_anchor,_focus) .. max(_anchor,_focus)].
  int? _anchor;
  int? _focus;
  Offset? _finger; // screen position while dragging (for the magnifier)

  int? get _lo => _anchor == null ? null : math.min(_anchor!, _focus!);
  int? get _hi => _anchor == null ? null : math.max(_anchor!, _focus!);
  bool get _hasSelection => _anchor != null;
  int get _selectedCount => _hasSelection ? _hi! - _lo! + 1 : 0;

  @override
  void initState() {
    super.initState();
    _read();
  }

  @override
  void dispose() {
    _scan?.cancel();
    super.dispose();
  }

  // ------------------------------------------------------------- reading

  void _read() {
    _scan?.cancel();
    setState(() {
      _phase = _Phase.reading;
      _result = null;
      _words = const [];
      _boxes = const [];
      _anchor = _focus = null;
      _stage = '';
    });
    _scan = widget.settings.api
        .scanStream(widget.image,
            language: widget.settings.language.apiValue, aiCorrect: widget.settings.aiCorrect)
        .listen(
      (event) {
        if (!mounted) return;
        setState(() {
          switch (event) {
            case LayoutEvent(:final result):
              _result = result;
            case CellEvent(:final block, :final row, :final index, :final cell):
              final r = _result;
              if (r != null && block < r.blocks.length) {
                final cells = r.blocks[block].rows[row].cells;
                if (index < cells.length) cells[index] = cell;
              }
            case StatusEvent(:final stage):
              _stage = stage;
            case DoneEvent(:final result):
              _result = result;
              _words = _buildWords(result);
              _boxes = [
                for (final c in result.cells)
                  if (c.text.isNotEmpty) Rect.fromLTRB(c.bbox.x1, c.bbox.y1, c.bbox.x2, c.bbox.y2),
              ];
              _phase = _Phase.selecting;
          }
        });
      },
      onError: (Object e) async {
        if (!mounted) return;
        final unreachable = e is ApiException &&
            (e.message.startsWith('Cannot reach') || e.message.startsWith('The server did not respond')) &&
            !_retried;
        if (unreachable) {
          // Server not reachable (restarted / new IP): look for it on the
          // Wi-Fi network and try once more before showing an error.
          _retried = true;
          setState(() => _stage = 'searching');
          if (await widget.settings.discoverServer() && mounted) {
            _read();
            return;
          }
        }
        if (!mounted) return;
        setState(() {
          _phase = _Phase.failed;
          _error = e is ApiException ? e.message : 'Reading failed: $e';
          if (unreachable) {
            _error += '\n\nThe HarfScan server was not found on this Wi-Fi. Make sure the PC '
                'is on, the server window is open, and the phone is on the same Wi-Fi.';
          }
        });
      },
      cancelOnError: true,
    );
  }

  /// Splits every recognised box into words. The backend reads whole boxes
  /// (a line or part of a line); each word gets a share of the box width
  /// proportional to its length, right-to-left as Urdu/Arabic is written.
  static List<_Word> _buildWords(OcrResult r) {
    final words = <_Word>[];
    for (var b = 0; b < r.blocks.length; b++) {
      final rows = r.blocks[b].rows;
      for (var ri = 0; ri < rows.length; ri++) {
        final cells = rows[ri].cells;
        for (var ci = 0; ci < cells.length; ci++) {
          final cell = cells[ci];
          final parts = cell.text.trim().split(RegExp(r'\s+')).where((p) => p.isNotEmpty).toList();
          if (parts.isEmpty) continue;
          final box = Rect.fromLTRB(cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2);
          final rects = _splitBox(box, parts, r.rotation);
          for (var i = 0; i < parts.length; i++) {
            words.add(_Word(b, ri, ci, parts[i], rects[i]));
          }
        }
      }
    }
    return words;
  }

  static List<Rect> _splitBox(Rect box, List<String> parts, int rotation) {
    // A sideways photo (90/270, or a tall line box when the page was read
    // as photographed): text runs vertically - keep the whole box for each
    // word rather than guessing.
    final vertical = rotation % 180 != 0 || (parts.length > 1 && box.height > box.width * 1.5);
    if (parts.length == 1 || vertical) return [for (final _ in parts) box];
    final total = parts.fold<int>(0, (n, p) => n + p.length) + parts.length - 1;
    final rects = <Rect>[];
    var cum = 0;
    for (final p in parts) {
      final a = cum / total, e = (cum + p.length) / total;
      cum += p.length + 1;
      rects.add(rotation == 0
          // upright page: first word on the right
          ? Rect.fromLTRB(box.right - e * box.width, box.top, box.right - a * box.width, box.bottom)
          // upside-down photo: first word on the left
          : Rect.fromLTRB(box.left + a * box.width, box.top, box.left + e * box.width, box.bottom));
    }
    return rects;
  }

  // ----------------------------------------------------------- selection

  /// Word under (or nearest to) [p], in image coordinates.
  int? _wordAt(Offset p) {
    if (_words.isEmpty) return null;
    var best = 0;
    var bestDist = double.infinity;
    for (var i = 0; i < _words.length; i++) {
      final r = _words[i].rect;
      if (r.contains(p)) return i;
      final dx = math.max(0.0, math.max(r.left - p.dx, p.dx - r.right));
      final dy = math.max(0.0, math.max(r.top - p.dy, p.dy - r.bottom));
      final d = dx * dx + dy * dy * 4; // prefer the same line
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    }
    return best;
  }

  /// Start handle sits before the first word, end handle after the last
  /// (reading direction right-to-left on an upright page).
  Offset _startHandle(double scale) {
    final r = _words[_lo!].rect;
    final upright = (_result?.rotation ?? 0) != 180;
    return (upright ? r.topRight : r.topLeft) * scale;
  }

  Offset _endHandle(double scale) {
    final r = _words[_hi!].rect;
    final upright = (_result?.rotation ?? 0) != 180;
    return (upright ? r.bottomLeft : r.bottomRight) * scale;
  }

  void _onPanStart(Offset local, double scale) {
    final idx = _wordAt(local / scale);
    if (idx == null) return;
    setState(() {
      _finger = local;
      if (_hasSelection && (local - _startHandle(scale)).distance < 36) {
        _anchor = _hi; // drag the start handle
        _focus = _lo;
      } else if (_hasSelection && (local - _endHandle(scale)).distance < 36) {
        _anchor = _lo; // drag the end handle
        _focus = _hi;
      } else {
        _anchor = _focus = idx; // new selection from the first word
      }
    });
  }

  void _onPanUpdate(Offset local, double scale) {
    final idx = _wordAt(local / scale);
    if (idx == null) return;
    setState(() {
      _finger = local;
      _focus = idx;
    });
  }

  void _onTap(Offset local, double scale) {
    final idx = _wordAt(local / scale);
    if (idx == null) return;
    setState(() {
      if (_selectedCount == 1 && _lo == idx) {
        _anchor = _focus = null;
      } else {
        _anchor = _focus = idx;
      }
    });
  }

  void _selectAll() => setState(() {
        _anchor = 0;
        _focus = _words.length - 1;
      });

  void _clear() => setState(() => _anchor = _focus = null);

  /// The page restricted to the selected words, keeping lines, blocks,
  /// bullets and table columns, so the book shows it with its structure.
  OcrResult _selectionResult() {
    final r = _result!;
    if (_lo == 0 && _hi == _words.length - 1) return r; // whole page
    final picked = <(int, int, int), List<String>>{};
    for (var i = _lo!; i <= _hi!; i++) {
      final w = _words[i];
      (picked[(w.block, w.row, w.cell)] ??= []).add(w.text);
    }
    final blocks = <OcrBlock>[];
    for (var b = 0; b < r.blocks.length; b++) {
      final block = r.blocks[b];
      final rows = <OcrRow>[];
      for (var ri = 0; ri < block.rows.length; ri++) {
        final row = block.rows[ri];
        final cells = <OcrCell>[];
        for (var ci = 0; ci < row.cells.length; ci++) {
          final words = picked[(b, ri, ci)];
          if (words == null) continue;
          final c = row.cells[ci];
          cells.add(OcrCell(
            bbox: c.bbox,
            column: c.column,
            text: words.join(' '),
            language: c.language,
            engine: c.engine,
            confidence: c.confidence,
            candidates: c.candidates,
          ));
        }
        if (cells.isNotEmpty) {
          rows.add(OcrRow(bbox: row.bbox, isBullet: row.isBullet, cells: cells));
        }
      }
      if (rows.isNotEmpty) {
        blocks.add(OcrBlock(
            id: block.id,
            type: block.type,
            bbox: block.bbox,
            columns: block.columns,
            language: block.language,
            rows: rows));
      }
    }
    return OcrResult(
      requestId: r.requestId,
      imageWidth: r.imageWidth,
      imageHeight: r.imageHeight,
      rotation: r.rotation,
      skewAngle: r.skewAngle,
      layoutEngine: r.layoutEngine,
      blocks: blocks,
      formattedText: '',
      processingMs: r.processingMs,
      totalCells: picked.length,
    );
  }

  String get _selectedText => _hasSelection ? _selectionResult().currentText() : '';

  void _addToBook() {
    final page = widget.session.addPage(_selectionResult(), widget.image.path);
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text('${page.header} added to the book')));
    Navigator.of(context).pop(true);
  }

  // ------------------------------------------------------------------ UI

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        title: Text(_phase == _Phase.selecting ? 'Select text' : 'Finding text'),
        actions: [
          if (_phase == _Phase.selecting)
            TextButton(onPressed: _selectAll, child: const Text('Select all')),
        ],
      ),
      body: HarfBackground(
        child: SafeArea(
          child: Column(
            children: [
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(12, 4, 12, 8),
                  child: Center(child: _buildPhoto()),
                ),
              ),
              _buildPanel(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildPhoto() {
    final r = _result;
    final photo = Image.file(widget.image, fit: BoxFit.fill, cacheWidth: 1600);
    if (r == null || r.imageWidth <= 0 || r.imageHeight <= 0) {
      return Stack(alignment: Alignment.center, children: [
        Image.file(widget.image, fit: BoxFit.contain, cacheWidth: 1600),
        const _ScanSweep(),
      ]);
    }
    return AspectRatio(
      aspectRatio: r.imageWidth / r.imageHeight,
      child: LayoutBuilder(builder: (context, box) {
        final scale = box.maxWidth / r.imageWidth;
        final selecting = _phase == _Phase.selecting;
        return Stack(
          clipBehavior: Clip.none,
          children: [
            Positioned.fill(
              child: DecoratedBox(
                position: DecorationPosition.foreground,
                decoration: BoxDecoration(border: Border.all(color: HarfColors.gold, width: 1.2)),
                child: photo,
              ),
            ),
            Positioned.fill(
              child: CustomPaint(
                painter: selecting
                    ? _SelectionPainter(
                        words: _words,
                        boxes: _boxes,
                        lo: _lo,
                        hi: _hi,
                        scale: scale,
                        start: _hasSelection ? _startHandle(scale) : null,
                        end: _hasSelection ? _endHandle(scale) : null,
                      )
                    : _ReadingPainter(result: r, scale: scale, read: r.readCells),
              ),
            ),
            if (selecting)
              Positioned.fill(
                child: GestureDetector(
                  behavior: HitTestBehavior.opaque,
                  onTapUp: (d) => _onTap(d.localPosition, scale),
                  onPanStart: (d) => _onPanStart(d.localPosition, scale),
                  onPanUpdate: (d) => _onPanUpdate(d.localPosition, scale),
                  onPanEnd: (_) => setState(() => _finger = null),
                  onPanCancel: () => setState(() => _finger = null),
                ),
              ),
            if (_finger != null)
              Positioned(
                left: _finger!.dx - 55,
                top: _finger!.dy - 55 - 90,
                child: const IgnorePointer(
                  child: RawMagnifier(
                    size: Size(110, 110),
                    magnificationScale: 2,
                    focalPointOffset: Offset(0, 90),
                    decoration: MagnifierDecoration(
                      shape: CircleBorder(side: BorderSide(color: HarfColors.brightGold, width: 2)),
                      shadows: [BoxShadow(color: Colors.black54, blurRadius: 10)],
                    ),
                  ),
                ),
              ),
          ],
        );
      }),
    );
  }

  Widget _buildPanel() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
      decoration: BoxDecoration(
        color: HarfColors.navy.withValues(alpha: 0.92),
        borderRadius: const BorderRadius.vertical(top: Radius.circular(22)),
        border: Border(top: BorderSide(color: HarfColors.gold.withValues(alpha: 0.5))),
      ),
      child: switch (_phase) {
        _Phase.reading => _readingPanel(),
        _Phase.failed => _failedPanel(),
        _Phase.selecting => _selectingPanel(),
      },
    );
  }

  Widget _readingPanel() {
    final r = _result;
    final total = r?.totalCells ?? 0;
    final read = r?.readCells ?? 0;
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        LinearProgressIndicator(
          value: total == 0 || _stage == 'ai_correcting' ? null : read / total,
          color: HarfColors.brightGold,
          backgroundColor: HarfColors.slate,
          minHeight: 6,
          borderRadius: BorderRadius.circular(3),
        ),
        const SizedBox(height: 10),
        Text(_stage == 'searching'
            ? 'Connecting to the server...'
            : 'Finding text...'),
        const SizedBox(height: 4),
        TextButton(onPressed: () => Navigator.of(context).pop(false), child: const Text('Cancel')),
      ],
    );
  }

  Widget _failedPanel() => Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.cloud_off, color: Colors.redAccent),
          const SizedBox(height: 6),
          Text(_error, textAlign: TextAlign.center),
          const SizedBox(height: 10),
          FilledButton.icon(onPressed: _read, icon: const Icon(Icons.refresh), label: const Text('Try again')),
        ],
      );

  Widget _selectingPanel() {
    if (_words.isEmpty) {
      return Column(mainAxisSize: MainAxisSize.min, children: [
        const Text('No text was found on this photo.'),
        TextButton(onPressed: () => Navigator.of(context).pop(false), child: const Text('Back')),
      ]);
    }
    final nextPage = widget.session.pageCount + 1;
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        if (!_hasSelection)
          const Row(
            children: [
              Icon(Icons.touch_app, color: HarfColors.gold),
              SizedBox(width: 10),
              Expanded(
                child: Text('Put your finger on the first word and drag to the last word.'),
              ),
            ],
          )
        else ...[
          Row(
            children: [
              Text('$_selectedCount word${_selectedCount == 1 ? '' : 's'} selected',
                  style: const TextStyle(color: HarfColors.gold, fontWeight: FontWeight.w600)),
              const Spacer(),
              TextButton(onPressed: _clear, child: const Text('Clear')),
            ],
          ),
          ConstrainedBox(
            constraints: const BoxConstraints(maxHeight: 96),
            child: SingleChildScrollView(
              child: Text(
                _selectedText.replaceAll('\t', '  |  '),
                textDirection: TextDirection.rtl,
                textAlign: TextAlign.right,
                style: scriptStyle(widget.settings.language == OcrLanguage.arabic ? 'arabic' : 'urdu',
                    size: 17),
              ),
            ),
          ),
        ],
        const SizedBox(height: 12),
        FilledButton.icon(
          onPressed: _hasSelection ? _addToBook : null,
          icon: const Icon(Icons.auto_stories),
          label: Text('Add to Book as Page $nextPage'),
          style: FilledButton.styleFrom(minimumSize: const Size(0, 54)),
        ),
      ],
    );
  }
}

// --------------------------------------------------------------------------
// Painters
// --------------------------------------------------------------------------

/// While reading: boxes found on the page; gold once read.
class _ReadingPainter extends CustomPainter {
  _ReadingPainter({required this.result, required this.scale, required this.read});

  final OcrResult result;
  final double scale;
  final int read; // repaint as more boxes are read

  @override
  void paint(Canvas canvas, Size size) {
    final pending = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1
      ..color = Colors.white54;
    final done = Paint()..color = HarfColors.gold.withValues(alpha: 0.30);
    for (final c in result.cells) {
      final r = Rect.fromLTRB(c.bbox.x1, c.bbox.y1, c.bbox.x2, c.bbox.y2);
      canvas.drawRect(Rect.fromLTRB(r.left * scale, r.top * scale, r.right * scale, r.bottom * scale),
          c.isRead ? done : pending);
    }
  }

  @override
  bool shouldRepaint(covariant _ReadingPainter old) =>
      old.result != result || old.read != read || old.scale != scale;
}

/// The text boxes found on the page (gold outlines), the selectable words
/// and the selection (gold) with its two handles.
class _SelectionPainter extends CustomPainter {
  _SelectionPainter({
    required this.words,
    required this.boxes,
    required this.lo,
    required this.hi,
    required this.scale,
    required this.start,
    required this.end,
  });

  final List<_Word> words;
  final List<Rect> boxes;
  final int? lo, hi;
  final double scale;
  final Offset? start, end;

  @override
  void paint(Canvas canvas, Size size) {
    final outline = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.6
      ..color = HarfColors.brightGold.withValues(alpha: 0.85);
    final shade = Paint()..color = HarfColors.gold.withValues(alpha: 0.10);
    for (final r in boxes) {
      final rect = RRect.fromRectAndRadius(
          Rect.fromLTRB(r.left * scale, r.top * scale, r.right * scale, r.bottom * scale).inflate(1),
          const Radius.circular(3));
      canvas.drawRRect(rect, shade);
      canvas.drawRRect(rect, outline);
    }
    final idle = Paint()..color = Colors.transparent;
    final selected = Paint()..color = HarfColors.brightGold.withValues(alpha: 0.42);
    for (var i = 0; i < words.length; i++) {
      final r = words[i].rect;
      final rect = Rect.fromLTRB(r.left * scale, r.top * scale, r.right * scale, r.bottom * scale);
      final on = lo != null && i >= lo! && i <= hi!;
      canvas.drawRRect(RRect.fromRectAndRadius(rect.deflate(0.5), const Radius.circular(2)),
          on ? selected : idle);
    }
    if (start != null && end != null && lo != null) {
      final h = words[lo!].rect.height * scale;
      _handle(canvas, start!, h, up: true);
      _handle(canvas, end!, words[hi!].rect.height * scale, up: false);
    }
  }

  void _handle(Canvas canvas, Offset at, double lineHeight, {required bool up}) {
    final stem = Paint()
      ..color = HarfColors.brightGold
      ..strokeWidth = 2.5;
    final knob = Paint()..color = HarfColors.brightGold;
    final other = up ? at.translate(0, lineHeight) : at.translate(0, -lineHeight);
    canvas.drawLine(at, other, stem);
    canvas.drawCircle(up ? at.translate(0, -7) : at.translate(0, 7), 8, knob);
  }

  @override
  bool shouldRepaint(covariant _SelectionPainter old) =>
      old.lo != lo || old.hi != hi || old.words != words || old.boxes != boxes || old.scale != scale;
}

/// Gold line sweeping over the photo while the page structure is detected.
class _ScanSweep extends StatefulWidget {
  const _ScanSweep();

  @override
  State<_ScanSweep> createState() => _ScanSweepState();
}

class _ScanSweepState extends State<_ScanSweep> with SingleTickerProviderStateMixin {
  late final AnimationController _c =
      AnimationController(vsync: this, duration: const Duration(milliseconds: 1600))
        ..repeat(reverse: true);

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Positioned.fill(
        child: AnimatedBuilder(
          animation: _c,
          builder: (context, _) => Align(
            alignment: Alignment(0, _c.value * 2 - 1),
            child: Container(
              height: 3,
              decoration: BoxDecoration(
                gradient: HarfColors.goldSheen,
                boxShadow: [BoxShadow(color: HarfColors.brightGold.withValues(alpha: 0.6), blurRadius: 12)],
              ),
            ),
          ),
        ),
      );
}
