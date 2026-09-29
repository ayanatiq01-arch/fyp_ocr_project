// main.dart
//
// Bilingual (Urdu & Arabic) OCR Scanner - Flutter UI.
//
// Flow:  Camera / Gallery  ->  upload the WHOLE page (no manual cropping;
//        the backend finds every text box itself)  ->  text streams in, in
//        reading order  ->  page is shown with its original structure
//        (title, paragraphs, bullets, tables).
//        Optional: "Select area" lets the user crop before scanning.

import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import 'api_service.dart';
import 'image_cropper.dart';

void main() => runApp(const OcrScannerApp());

class OcrScannerApp extends StatelessWidget {
  const OcrScannerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Urdu & Arabic OCR',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF00695C),
        useMaterial3: true,
      ),
      home: const ScannerPage(),
    );
  }
}

class ScannerPage extends StatefulWidget {
  const ScannerPage({super.key});

  @override
  State<ScannerPage> createState() => _ScannerPageState();
}

class _ScannerPageState extends State<ScannerPage> {
  final ImagePicker _picker = ImagePicker();
  late ApiService _api = ApiService(baseUrl: ApiService.defaultBaseUrl);

  File? _image; // photo being scanned
  OcrResult? _result; // grows while the page is streamed
  StreamSubscription<OcrEvent>? _scan;
  bool _busy = false;
  bool _done = false;
  bool? _serverOnline;
  bool _showBoxes = true;
  final Stopwatch _clock = Stopwatch();

  @override
  void initState() {
    super.initState();
    _checkServer();
    _recoverLostPhoto();
  }

  @override
  void dispose() {
    _scan?.cancel();
    _api.dispose();
    super.dispose();
  }

  // ------------------------------------------------------------------ actions

  Future<void> _checkServer() async {
    final ok = await _api.healthCheck();
    if (mounted) setState(() => _serverOnline = ok);
  }

  /// Android may kill the app while the camera is open (low memory). The
  /// photo is then delivered on next launch via retrieveLostData().
  Future<void> _recoverLostPhoto() async {
    if (!Platform.isAndroid) return;
    final LostDataResponse lost = await _picker.retrieveLostData();
    if (lost.isEmpty || lost.file == null || !mounted) return;
    _startScan(File(lost.file!.path));
  }

  Future<void> _pick(ImageSource source) async {
    try {
      // Full resolution: fine print in old books needs every pixel.
      final XFile? photo = await _picker.pickImage(
        source: source,
        preferredCameraDevice: CameraDevice.rear,
        requestFullMetadata: false,
      );
      if (photo == null) return; // user cancelled
      _startScan(File(photo.path));
    } on PlatformException catch (e) {
      _snack('Could not open ${source == ImageSource.camera ? 'camera' : 'gallery'}: ${e.message}');
    }
  }

  /// Optional: let the user limit the scan to part of the page.
  Future<void> _selectArea() async {
    if (_image == null) return;
    final File? cropped = await ParagraphCropper.crop(context, _image!.path);
    if (cropped != null && mounted) _startScan(cropped);
  }

  void _startScan(File image) {
    _scan?.cancel();
    setState(() {
      _image = image;
      _result = null;
      _busy = true;
      _done = false;
    });
    _clock
      ..reset()
      ..start();
    _scan = _api.scanStream(image).listen(
      _onEvent,
      onError: (Object e) {
        _snack(e is ApiException ? e.message : 'Scan failed: $e');
        _finish();
      },
      onDone: _finish,
      cancelOnError: true,
    );
  }

  void _onEvent(OcrEvent event) {
    if (!mounted) return;
    setState(() {
      _serverOnline = true;
      switch (event) {
        case LayoutEvent(:final result):
          _result = result;
        case CellEvent(:final block, :final row, :final index, :final cell):
          final r = _result;
          if (r != null && block < r.blocks.length) {
            final cells = r.blocks[block].rows[row].cells;
            if (index < cells.length) cells[index] = cell;
          }
        case DoneEvent(:final result):
          _result = result;
          _done = true;
      }
    });
  }

  void _finish() {
    _clock.stop();
    if (mounted) setState(() => _busy = false);
  }

  void _cancelScan() {
    _scan?.cancel();
    _finish();
  }

  void _reset() {
    _scan?.cancel();
    setState(() {
      _image = null;
      _result = null;
      _busy = false;
      _done = false;
    });
  }

  Future<void> _editServerUrl() async {
    final controller = TextEditingController(text: _api.baseUrl);
    final String? url = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Backend server URL'),
        content: TextField(
          controller: controller,
          keyboardType: TextInputType.url,
          decoration: const InputDecoration(
            hintText: 'http://192.168.1.20:8000',
            helperText: 'Emulator: http://10.0.2.2:8000',
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, controller.text),
            child: const Text('Save'),
          ),
        ],
      ),
    );
    controller.dispose();
    if (url == null || url.trim().isEmpty) return;
    _api.dispose();
    setState(() {
      _api = ApiService(baseUrl: url);
      _serverOnline = null;
    });
    _checkServer();
  }

  void _snack(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), behavior: SnackBarBehavior.floating),
    );
  }

  void _copyText() {
    final text = _result?.currentText() ?? '';
    if (text.isEmpty) return;
    Clipboard.setData(ClipboardData(text: text));
    _snack('Copied to clipboard');
  }

  // ---------------------------------------------------------------------- UI

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Urdu & Arabic OCR'),
        actions: [
          _ServerStatusDot(online: _serverOnline, onTap: _checkServer),
          IconButton(
            tooltip: 'Server settings',
            icon: const Icon(Icons.settings),
            onPressed: _busy ? null : _editServerUrl,
          ),
        ],
      ),
      body: SafeArea(child: _image == null ? _buildPickView() : _buildScanView()),
    );
  }

  Widget _buildPickView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.menu_book, size: 96, color: Theme.of(context).colorScheme.primary),
            const SizedBox(height: 16),
            Text(
              'Photograph a full book page.\nThe app finds and reads all the text by itself.',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 32),
            FilledButton.icon(
              onPressed: () => _pick(ImageSource.camera),
              icon: const Icon(Icons.photo_camera),
              label: const Text('Take photo'),
              style: FilledButton.styleFrom(minimumSize: const Size(220, 52)),
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              onPressed: () => _pick(ImageSource.gallery),
              icon: const Icon(Icons.photo_library),
              label: const Text('Choose from gallery'),
              style: OutlinedButton.styleFrom(minimumSize: const Size(220, 52)),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildScanView() {
    final result = _result;
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        _ImageWithBoxes(image: _image!, result: _showBoxes ? result : null),
        const SizedBox(height: 12),
        _buildProgress(result),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          alignment: WrapAlignment.center,
          children: [
            if (_busy)
              FilledButton.tonalIcon(
                onPressed: _cancelScan,
                icon: const Icon(Icons.stop),
                label: const Text('Stop'),
              )
            else
              FilledButton.icon(
                onPressed: () => _startScan(_image!),
                icon: const Icon(Icons.refresh),
                label: const Text('Scan again'),
              ),
            OutlinedButton.icon(
              onPressed: _busy ? null : _selectArea,
              icon: const Icon(Icons.crop),
              label: const Text('Select area'),
            ),
            OutlinedButton.icon(
              onPressed: _reset,
              icon: const Icon(Icons.add_a_photo),
              label: const Text('New page'),
            ),
          ],
        ),
        if (result != null) ...[
          const SizedBox(height: 16),
          _PageCard(
            result: result,
            done: _done,
            onCopy: _copyText,
            onCellTap: (cell) => _showCellDetails(cell),
          ),
        ],
      ],
    );
  }

  Widget _buildProgress(OcrResult? r) {
    final theme = Theme.of(context).textTheme.bodySmall;
    final seconds = (_clock.elapsedMilliseconds / 1000).toStringAsFixed(0);
    if (r == null) {
      return Column(children: [
        const LinearProgressIndicator(),
        const SizedBox(height: 6),
        Text(_busy ? 'Finding text on the page…' : 'No result', style: theme),
      ]);
    }
    final total = r.totalCells == 0 ? 1 : r.totalCells;
    final read = _done ? total : r.readCells;
    return Column(children: [
      LinearProgressIndicator(value: read / total),
      const SizedBox(height: 6),
      Row(
        children: [
          Expanded(
            child: Text(
              _done
                  ? 'Done · ${r.blocks.length} blocks · ${(r.processingMs / 1000).toStringAsFixed(0)} s'
                  : 'Reading $read / ${r.totalCells} · $seconds s',
              style: theme,
            ),
          ),
          const Text('Boxes'),
          Switch(value: _showBoxes, onChanged: (v) => setState(() => _showBoxes = v)),
        ],
      ),
    ]);
  }

  /// Bottom sheet showing how the router decided for one box.
  void _showCellDetails(OcrCell cell) {
    showModalBottomSheet<void>(
      context: context,
      builder: (ctx) {
        Widget engine(String label, String key) {
          final c = cell.candidates[key];
          final won = cell.language == key;
          return ListTile(
            leading: Icon(won ? Icons.check_circle : Icons.cancel,
                color: won ? Colors.green : Colors.grey),
            title: Directionality(
              textDirection: TextDirection.rtl,
              child: Text(c?.text.isNotEmpty == true ? c!.text : '—',
                  style: const TextStyle(fontSize: 20)),
            ),
            subtitle: Text('$label · ${c?.confidence.toStringAsFixed(1) ?? '-'}%'),
          );
        }

        return SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Padding(
                padding: EdgeInsets.all(16),
                child: Text('Confidence routing: the higher score wins',
                    style: TextStyle(fontWeight: FontWeight.bold)),
              ),
              engine('UTRNet (Urdu)', 'urdu'),
              engine('Kraken (Arabic)', 'arabic'),
              const SizedBox(height: 8),
            ],
          ),
        );
      },
    );
  }
}

// --------------------------------------------------------------------------
// Page reconstruction
// --------------------------------------------------------------------------

/// Shows the page the way it is printed: centred title, paragraph lines,
/// bullets and tables, all right-to-left. Cells not read yet show "…".
class _PageCard extends StatelessWidget {
  const _PageCard({
    required this.result,
    required this.done,
    required this.onCopy,
    required this.onCellTap,
  });

  final OcrResult result;
  final bool done;
  final VoidCallback onCopy;
  final ValueChanged<OcrCell> onCellTap;

  static const _textStyle = TextStyle(fontSize: 20, height: 1.7);

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Text('Extracted page', style: Theme.of(context).textTheme.titleMedium),
                const Spacer(),
                IconButton(tooltip: 'Copy text', icon: const Icon(Icons.copy), onPressed: onCopy),
              ],
            ),
            const Divider(),
            if (result.blocks.isEmpty)
              const Center(child: Text('— no text found —'))
            else
              Directionality(
                textDirection: TextDirection.rtl,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    for (final block in result.blocks) ...[
                      _buildBlock(context, block),
                      const SizedBox(height: 18),
                    ],
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildBlock(BuildContext context, OcrBlock block) {
    switch (block.type) {
      case 'Table':
        return _buildTable(context, block);
      case 'Title':
        return Center(
          child: Wrap(
            alignment: WrapAlignment.center,
            children: [
              for (final c in block.rows.expand((r) => r.cells))
                _cellText(c, _textStyle.copyWith(fontSize: 24, fontWeight: FontWeight.bold)),
            ],
          ),
        );
      default: // Text / List: one line per printed line
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            for (final row in block.rows)
              Wrap(
                spacing: 6,
                children: [
                  if (row.isBullet) const Text('•', style: _textStyle),
                  for (final c in row.cells) _cellText(c, _textStyle),
                ],
              ),
          ],
        );
    }
  }

  Widget _buildTable(BuildContext context, OcrBlock block) {
    final cols = block.columns < 1 ? 1 : block.columns;
    final border = Theme.of(context).dividerColor;
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      reverse: true, // RTL: start at the right-hand column
      child: Table(
        defaultColumnWidth: const IntrinsicColumnWidth(),
        border: TableBorder.all(color: border, width: 0.5),
        defaultVerticalAlignment: TableCellVerticalAlignment.middle,
        children: [
          for (final row in block.rows)
            TableRow(children: [
              for (var col = 0; col < cols; col++)
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  child: Wrap(
                    spacing: 4,
                    children: [
                      for (final c in row.cells.where((c) => c.column == col))
                        _cellText(c, _textStyle.copyWith(fontSize: 18)),
                    ],
                  ),
                ),
            ]),
        ],
      ),
    );
  }

  /// One cell: tap to see both engines' readings. Unread cells show "…".
  Widget _cellText(OcrCell cell, TextStyle style) {
    if (!cell.isRead) {
      return Text('…', style: style.copyWith(color: Colors.grey));
    }
    if (cell.text.isEmpty) return const SizedBox.shrink();
    final color = cell.language == 'arabic' ? Colors.indigo.shade900 : Colors.black;
    return InkWell(
      onTap: () => onCellTap(cell),
      child: Text(cell.text, style: style.copyWith(color: color)),
    );
  }
}

// --------------------------------------------------------------------------
// Small widgets
// --------------------------------------------------------------------------

class _ServerStatusDot extends StatelessWidget {
  const _ServerStatusDot({required this.online, required this.onTap});

  final bool? online;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final color = online == null
        ? Colors.grey
        : online!
            ? Colors.green
            : Colors.red;
    final label = online == null
        ? 'Checking server…'
        : online!
            ? 'Server online'
            : 'Server offline - tap to retry';
    return IconButton(
      tooltip: label,
      onPressed: onTap,
      icon: Icon(Icons.circle, size: 14, color: color),
    );
  }
}

/// Shows the photo and, once the layout is known, every text box the backend
/// found: grey = waiting, teal = read as Urdu, indigo = read as Arabic.
class _ImageWithBoxes extends StatelessWidget {
  const _ImageWithBoxes({required this.image, required this.result});

  final File image;
  final OcrResult? result;

  @override
  Widget build(BuildContext context) {
    final r = result;
    final picture = ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: Image.file(image, fit: BoxFit.contain),
    );
    if (r == null || r.imageWidth <= 0 || r.imageHeight <= 0) return picture;

    // AspectRatio makes the painted area exactly match the image, so a
    // single scale factor maps server pixels to screen pixels.
    return AspectRatio(
      aspectRatio: r.imageWidth / r.imageHeight,
      child: Stack(
        fit: StackFit.expand,
        children: [picture, CustomPaint(painter: _BoxPainter(r))],
      ),
    );
  }
}

class _BoxPainter extends CustomPainter {
  _BoxPainter(this.result) : _readCells = result.readCells;

  final OcrResult result;
  final int _readCells; // repaint when more cells have been read

  @override
  void paint(Canvas canvas, Size size) {
    final sx = size.width / result.imageWidth;
    final sy = size.height / result.imageHeight;
    Rect toRect(BBox b) => Rect.fromLTRB(b.x1 * sx, b.y1 * sy, b.x2 * sx, b.y2 * sy);

    Paint stroke(Color c, double w) => Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = w
      ..color = c;
    final blockPaint = stroke(Colors.orange, 2);
    final pending = stroke(Colors.grey, 1.2);
    final urdu = stroke(Colors.teal, 1.5);
    final arabic = stroke(Colors.indigo, 1.5);

    for (final block in result.blocks) {
      canvas.drawRect(toRect(block.bbox), blockPaint);
      for (final cell in block.rows.expand((r) => r.cells)) {
        final paint = !cell.isRead ? pending : (cell.language == 'arabic' ? arabic : urdu);
        canvas.drawRect(toRect(cell.bbox), paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _BoxPainter old) =>
      old.result != result || old._readCells != _readCells;
}
