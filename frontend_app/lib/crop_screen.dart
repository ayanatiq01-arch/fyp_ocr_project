// crop_screen.dart
//
// Screen 2 - Interactive Crop & Selection.
// Shows the picked photo; the user can drag a crop box around the text
// (image_cropper) or keep the whole page, then "Extract Text & Append to
// Book" sends it to the backend and appends the result as the next page.

import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import 'api_service.dart';
import 'book_session.dart';
import 'image_cropper.dart';
import 'theme.dart';

class CropScreen extends StatefulWidget {
  const CropScreen({super.key, required this.image, required this.session, required this.api});

  final File image;
  final BookSession session;
  final ApiService api;

  /// Picks a photo (camera or gallery), opens the crop screen and returns
  /// true if a page was extracted and appended to the book.
  static Future<bool> pickAndExtract(BuildContext context,
      {required ImageSource source, required BookSession session, required ApiService api}) async {
    final messenger = ScaffoldMessenger.of(context);
    final navigator = Navigator.of(context);
    try {
      final XFile? photo = await ImagePicker().pickImage(
        source: source,
        preferredCameraDevice: CameraDevice.rear,
        requestFullMetadata: false, // full resolution: fine print needs every pixel
      );
      if (photo == null) return false; // user cancelled
      final added = await navigator.push<bool>(MaterialPageRoute(
        builder: (_) => CropScreen(image: File(photo.path), session: session, api: api),
      ));
      return added ?? false;
    } on PlatformException catch (e) {
      messenger.showSnackBar(SnackBar(
          content: Text('Could not open ${source == ImageSource.camera ? 'camera' : 'gallery'}: ${e.message}')));
      return false;
    }
  }

  @override
  State<CropScreen> createState() => _CropScreenState();
}

class _CropScreenState extends State<CropScreen> {
  late File _selection = widget.image; // what will be sent: crop or whole page
  bool get _isWholePage => _selection.path == widget.image.path;

  StreamSubscription<OcrEvent>? _scan;
  bool _extracting = false;
  int _read = 0;
  int _total = 0;

  @override
  void dispose() {
    _scan?.cancel();
    super.dispose();
  }

  Future<void> _adjustCrop() async {
    final cropped = await ParagraphCropper.crop(context, widget.image.path);
    if (cropped != null && mounted) setState(() => _selection = cropped);
  }

  void _selectWholePage() => setState(() => _selection = widget.image);

  void _extract() {
    setState(() {
      _extracting = true;
      _read = 0;
      _total = 0;
    });
    _scan = widget.api.scanStream(_selection, language: widget.session.language.apiValue).listen(
      (event) {
        if (!mounted) return;
        switch (event) {
          case LayoutEvent(:final result):
            setState(() => _total = result.totalCells);
          case CellEvent():
            setState(() => _read++);
          case DoneEvent(:final result):
            final page = widget.session.addPage(result, _selection.path);
            ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text('${page.header} added to the book')));
            Navigator.of(context).pop(true);
        }
      },
      onError: (Object e) {
        if (!mounted) return;
        setState(() => _extracting = false);
        ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text(e is ApiException ? e.message : 'Extraction failed: $e')));
      },
      cancelOnError: true,
    );
  }

  void _cancelExtraction() {
    _scan?.cancel();
    setState(() => _extracting = false);
  }

  @override
  Widget build(BuildContext context) {
    final nextPage = widget.session.pageCount + 1;
    return PopScope(
      canPop: !_extracting,
      child: Scaffold(
        extendBodyBehindAppBar: true,
        appBar: AppBar(title: Text('Page $nextPage - Select text')),
        body: HarfBackground(
          child: SafeArea(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Expanded(child: _ImagePreview(file: _selection, wholePage: _isWholePage)),
                  const SizedBox(height: 14),
                  if (_extracting) _buildProgress() else _buildActions(),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildActions() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            Expanded(
              child: OutlinedButton.icon(
                onPressed: _adjustCrop,
                icon: const Icon(Icons.crop),
                label: const Text('Adjust Crop Box'),
                style: OutlinedButton.styleFrom(minimumSize: const Size(0, 48)),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: _isWholePage ? null : _selectWholePage,
                icon: const Icon(Icons.fullscreen),
                label: const Text('Select Whole Page'),
                style: OutlinedButton.styleFrom(minimumSize: const Size(0, 48)),
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        FilledButton.icon(
          onPressed: _extract,
          icon: const Icon(Icons.auto_stories),
          label: const Text('Extract Text & Append to Book'),
          style: FilledButton.styleFrom(minimumSize: const Size(0, 56)),
        ),
        const SizedBox(height: 6),
        Text('Language: ${widget.session.language.label}',
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: HarfColors.gold)),
      ],
    );
  }

  Widget _buildProgress() {
    final value = _total == 0 ? null : _read / _total;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            LinearProgressIndicator(
              value: value,
              color: HarfColors.brightGold,
              backgroundColor: HarfColors.navy,
              minHeight: 6,
            ),
            const SizedBox(height: 10),
            Text(_total == 0 ? 'Finding text on the page...' : 'Reading $_read / $_total'),
            TextButton(onPressed: _cancelExtraction, child: const Text('Cancel')),
          ],
        ),
      ),
    );
  }
}

class _ImagePreview extends StatelessWidget {
  const _ImagePreview({required this.file, required this.wholePage});

  final File file;
  final bool wholePage;

  @override
  Widget build(BuildContext context) {
    return Stack(
      fit: StackFit.expand,
      children: [
        DecoratedBox(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: HarfColors.gold, width: 1.5),
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(15),
            child: Image.file(file, fit: BoxFit.contain, key: ValueKey(file.path)),
          ),
        ),
        Positioned(
          top: 10,
          left: 10,
          child: Chip(
            avatar: Icon(wholePage ? Icons.description : Icons.crop, size: 18, color: HarfColors.navy),
            label: Text(wholePage ? 'Whole page' : 'Cropped region'),
            labelStyle: const TextStyle(color: HarfColors.navy, fontWeight: FontWeight.w600),
            backgroundColor: HarfColors.gold,
            side: BorderSide.none,
          ),
        ),
      ],
    );
  }
}
