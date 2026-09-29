// main.dart
//
// Bilingual (Urdu & Arabic) OCR Scanner - Flutter UI.
//
// Flow:  Camera / Gallery  ->  interactive crop (image_cropper.dart)
//        ->  upload to FastAPI (api_service.dart)  ->  show text + layout.

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

  String? _originalPath; // full photo, kept so the user can re-crop
  File? _croppedImage; // region sent to the server
  OcrResult? _result;
  bool _busy = false;
  bool? _serverOnline;
  bool _showBoxes = true;

  @override
  void initState() {
    super.initState();
    _checkServer();
    _recoverLostPhoto();
  }

  @override
  void dispose() {
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
    _originalPath = lost.file!.path;
    await _crop();
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
      _originalPath = photo.path;
      await _crop();
    } on PlatformException catch (e) {
      _showError('Could not open ${source == ImageSource.camera ? 'camera' : 'gallery'}: ${e.message}');
    }
  }

  Future<void> _crop() async {
    if (_originalPath == null) return;
    final File? cropped = await ParagraphCropper.crop(context, _originalPath!);
    if (cropped == null || !mounted) return;
    setState(() {
      _croppedImage = cropped;
      _result = null;
    });
  }

  Future<void> _scan() async {
    if (_croppedImage == null) return;
    setState(() => _busy = true);
    try {
      final result = await _api.scanImage(_croppedImage!);
      if (!mounted) return;
      setState(() {
        _result = result;
        _serverOnline = true;
      });
    } on ApiException catch (e) {
      _showError(e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  void _reset() => setState(() {
        _originalPath = null;
        _croppedImage = null;
        _result = null;
      });

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

  void _showError(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), behavior: SnackBarBehavior.floating),
    );
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
      body: SafeArea(
        child: _croppedImage == null ? _buildPickView() : _buildScanView(),
      ),
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
              'Photograph a book page, then select the paragraphs to digitise.',
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
        _ImageWithBoxes(
          image: _croppedImage!,
          result: _showBoxes ? result : null,
        ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          alignment: WrapAlignment.center,
          children: [
            FilledButton.icon(
              onPressed: _busy ? null : _scan,
              icon: _busy
                  ? const SizedBox(
                      width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                  : const Icon(Icons.document_scanner),
              label: Text(_busy ? 'Scanning…' : 'Scan text'),
            ),
            OutlinedButton.icon(
              onPressed: _busy ? null : _crop,
              icon: const Icon(Icons.crop),
              label: const Text('Re-crop'),
            ),
            OutlinedButton.icon(
              onPressed: _busy ? null : _reset,
              icon: const Icon(Icons.add_a_photo),
              label: const Text('New page'),
            ),
          ],
        ),
        if (result != null) ...[
          const SizedBox(height: 16),
          _buildSummary(result),
          const SizedBox(height: 12),
          _buildFormattedText(result),
          const SizedBox(height: 12),
          ...result.blocks.map(_buildBlockCard),
        ],
      ],
    );
  }

  Widget _buildSummary(OcrResult r) {
    final lines = r.blocks.expand((b) => b.lines).toList();
    final avg = lines.isEmpty
        ? 0.0
        : lines.map((l) => l.confidence).reduce((a, b) => a + b) / lines.length;
    return Row(
      children: [
        Expanded(
          child: Text(
            '${r.blocks.length} blocks · ${lines.length} lines · '
            'avg confidence ${avg.toStringAsFixed(1)}% · '
            '${(r.processingMs / 1000).toStringAsFixed(1)} s',
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ),
        const Text('Boxes'),
        Switch(value: _showBoxes, onChanged: (v) => setState(() => _showBoxes = v)),
      ],
    );
  }

  Widget _buildFormattedText(OcrResult r) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Text('Extracted text', style: Theme.of(context).textTheme.titleMedium),
                const Spacer(),
                IconButton(
                  tooltip: 'Copy',
                  icon: const Icon(Icons.copy),
                  onPressed: r.formattedText.isEmpty
                      ? null
                      : () {
                          Clipboard.setData(ClipboardData(text: r.formattedText));
                          _showError('Copied to clipboard');
                        },
                ),
              ],
            ),
            const Divider(),
            // Urdu and Arabic are right-to-left.
            Directionality(
              textDirection: TextDirection.rtl,
              child: SelectableText(
                r.formattedText.isEmpty ? '— no text found —' : r.formattedText,
                style: const TextStyle(fontSize: 20, height: 1.8),
                textAlign: TextAlign.right,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBlockCard(OcrBlock block) {
    return Card(
      child: ExpansionTile(
        title: Text('Block ${block.id + 1} · ${block.type}'),
        subtitle: Text('${block.lines.length} lines · ${block.language}'),
        children: block.lines
            .map(
              (line) => ListTile(
                dense: true,
                title: Directionality(
                  textDirection: TextDirection.rtl,
                  child: Text(line.text, style: const TextStyle(fontSize: 18)),
                ),
                subtitle: Text(
                  '${line.engine.isEmpty ? '—' : line.engine} · '
                  '${line.confidence.toStringAsFixed(1)}%  '
                  '(Urdu ${line.candidates['urdu']?.confidence.toStringAsFixed(1) ?? '-'}% / '
                  'Arabic ${line.candidates['arabic']?.confidence.toStringAsFixed(1) ?? '-'}%)',
                ),
                trailing: _LanguageChip(language: line.language),
              ),
            )
            .toList(),
      ),
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

class _LanguageChip extends StatelessWidget {
  const _LanguageChip({required this.language});

  final String language;

  @override
  Widget build(BuildContext context) {
    final (label, color) = switch (language) {
      'urdu' => ('اردو', Colors.teal),
      'arabic' => ('عربي', Colors.indigo),
      _ => ('?', Colors.grey),
    };
    return Chip(
      label: Text(label, style: const TextStyle(color: Colors.white)),
      backgroundColor: color,
      visualDensity: VisualDensity.compact,
    );
  }
}

/// Shows the cropped image and, when a result is available, draws the
/// block (thick) and line (thin) boxes returned by the backend on top.
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
        children: [
          picture,
          CustomPaint(painter: _BoxPainter(r)),
        ],
      ),
    );
  }
}

class _BoxPainter extends CustomPainter {
  _BoxPainter(this.result);

  final OcrResult result;

  @override
  void paint(Canvas canvas, Size size) {
    final sx = size.width / result.imageWidth;
    final sy = size.height / result.imageHeight;
    Rect toRect(BBox b) => Rect.fromLTRB(b.x1 * sx, b.y1 * sy, b.x2 * sx, b.y2 * sy);

    final blockPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5
      ..color = Colors.orange;
    final urduPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2
      ..color = Colors.teal;
    final arabicPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2
      ..color = Colors.indigo;

    for (final block in result.blocks) {
      canvas.drawRect(toRect(block.bbox), blockPaint);
      for (final line in block.lines) {
        canvas.drawRect(
          toRect(line.bbox),
          line.language == 'arabic' ? arabicPaint : urduPaint,
        );
      }
    }
  }

  @override
  bool shouldRepaint(covariant _BoxPainter oldDelegate) => oldDelegate.result != result;
}
