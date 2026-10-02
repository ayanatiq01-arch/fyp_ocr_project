// dashboard_screen.dart
//
// Screen 1 - Home Dashboard.
// Gold HarfScan logo, language toggle [Urdu | Arabic | Mixed (Auto)],
// Camera / Gallery action cards and the "Active Book Workspace" indicator.

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';

import 'api_service.dart';
import 'book_session.dart';
import 'book_workspace_screen.dart';
import 'crop_screen.dart';
import 'theme.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key, required this.session});

  final BookSession session;

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  ApiService _api = ApiService(baseUrl: ApiService.defaultBaseUrl);
  bool? _serverOnline;

  BookSession get _session => widget.session;

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

  Future<void> _checkServer() async {
    setState(() => _serverOnline = null);
    final ok = await _api.healthCheck();
    if (mounted) setState(() => _serverOnline = ok);
  }

  /// Android may kill the app while the camera is open (low memory); the
  /// photo is then delivered on the next launch via retrieveLostData().
  Future<void> _recoverLostPhoto() async {
    if (!Platform.isAndroid) return;
    final lost = await ImagePicker().retrieveLostData();
    if (lost.isEmpty || lost.file == null || !mounted) return;
    final added = await Navigator.of(context).push<bool>(MaterialPageRoute(
      builder: (_) => CropScreen(image: File(lost.file!.path), session: _session, api: _api),
    ));
    if (added == true) _openWorkspace();
  }

  Future<void> _scan(ImageSource source) async {
    final added =
        await CropScreen.pickAndExtract(context, source: source, session: _session, api: _api);
    if (added) _openWorkspace();
  }

  void _openWorkspace() {
    if (!mounted) return;
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => BookWorkspaceScreen(session: _session, api: _api),
    ));
  }

  Future<void> _newBook() async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: HarfColors.slate,
        title: const Text('Start a new book?'),
        content: Text('The ${_session.pageCount} page(s) in the current workspace will be cleared. '
            'Export them first if you need them.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('New book')),
        ],
      ),
    );
    if (ok == true) _session.clear();
  }

  Future<void> _editServerUrl() async {
    final controller = TextEditingController(text: _api.baseUrl);
    final url = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: HarfColors.slate,
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
          FilledButton(onPressed: () => Navigator.pop(ctx, controller.text), child: const Text('Save')),
        ],
      ),
    );
    controller.dispose();
    if (url == null || url.trim().isEmpty) return;
    _api.dispose();
    setState(() => _api = ApiService(baseUrl: url));
    _checkServer();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        actions: [
          _ServerStatusDot(online: _serverOnline, onTap: _checkServer),
          IconButton(
            tooltip: 'Server settings',
            icon: const Icon(Icons.settings),
            onPressed: _editServerUrl,
          ),
        ],
      ),
      body: HarfBackground(
        child: SafeArea(
          child: ListenableBuilder(
            listenable: _session,
            builder: (context, _) => ListView(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
              children: [
                const _Header(),
                const SizedBox(height: 28),
                _sectionLabel('Document language'),
                _LanguageToggle(
                  value: _session.language,
                  onChanged: (v) => _session.language = v,
                ),
                const SizedBox(height: 28),
                _sectionLabel('Add a page'),
                Row(
                  children: [
                    Expanded(
                      child: _ActionCard(
                        icon: Icons.photo_camera,
                        title: 'Camera Scan',
                        subtitle: 'Photograph a page',
                        onTap: () => _scan(ImageSource.camera),
                      ),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: _ActionCard(
                        icon: Icons.photo_library,
                        title: 'Gallery Upload',
                        subtitle: 'Pick a saved photo',
                        onTap: () => _scan(ImageSource.gallery),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 28),
                _sectionLabel('Active Book Workspace'),
                _WorkspaceCard(
                  pageCount: _session.pageCount,
                  onOpen: _openWorkspace,
                  onNewBook: _session.isEmpty ? null : _newBook,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _sectionLabel(String text) => Padding(
        padding: const EdgeInsets.only(left: 4, bottom: 10),
        child: Text(text.toUpperCase(),
            style: const TextStyle(
                color: HarfColors.gold, fontSize: 12, letterSpacing: 1.6, fontWeight: FontWeight.w600)),
      );
}

// --------------------------------------------------------------------------
// Widgets
// --------------------------------------------------------------------------

class _Header extends StatelessWidget {
  const _Header();

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          width: 92,
          height: 92,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            gradient: HarfColors.goldSheen,
            boxShadow: [
              BoxShadow(color: HarfColors.gold.withValues(alpha: 0.35), blurRadius: 24),
            ],
          ),
          alignment: Alignment.center,
          // "ح" (Harf) - the first letter of the word, as the logo mark.
          child: Text('ح',
              style: GoogleFonts.notoNaskhArabic(
                  fontSize: 46, fontWeight: FontWeight.w700, color: HarfColors.navy, height: 1.2)),
        ),
        const SizedBox(height: 14),
        ShaderMask(
          shaderCallback: HarfColors.goldSheen.createShader,
          child: Text('HarfScan',
              style: GoogleFonts.cinzel(
                  fontSize: 36, fontWeight: FontWeight.w700, color: Colors.white, letterSpacing: 2)),
        ),
        const SizedBox(height: 4),
        Text('Urdu & Arabic OCR for historical books',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: HarfColors.ink.withValues(alpha: 0.75))),
      ],
    );
  }
}

class _LanguageToggle extends StatelessWidget {
  const _LanguageToggle({required this.value, required this.onChanged});

  final OcrLanguage value;
  final ValueChanged<OcrLanguage> onChanged;

  @override
  Widget build(BuildContext context) {
    return SegmentedButton<OcrLanguage>(
      segments: [
        for (final l in OcrLanguage.values)
          ButtonSegment(value: l, label: Text(l.label, maxLines: 1, overflow: TextOverflow.ellipsis)),
      ],
      selected: {value},
      showSelectedIcon: false,
      onSelectionChanged: (s) => onChanged(s.first),
      style: SegmentedButton.styleFrom(
        backgroundColor: HarfColors.slate,
        foregroundColor: HarfColors.ink,
        selectedBackgroundColor: HarfColors.gold,
        selectedForegroundColor: HarfColors.navy,
        side: const BorderSide(color: HarfColors.gold),
        minimumSize: const Size(0, 48),
      ),
    );
  }
}

class _ActionCard extends StatelessWidget {
  const _ActionCard(
      {required this.icon, required this.title, required this.subtitle, required this.onTap});

  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: EdgeInsets.zero,
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 12),
          child: Column(
            children: [
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: HarfColors.gold.withValues(alpha: 0.15),
                  border: Border.all(color: HarfColors.gold),
                ),
                child: Icon(icon, size: 32, color: HarfColors.brightGold),
              ),
              const SizedBox(height: 12),
              Text(title,
                  textAlign: TextAlign.center,
                  style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
              const SizedBox(height: 2),
              Text(subtitle,
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.bodySmall),
            ],
          ),
        ),
      ),
    );
  }
}

class _WorkspaceCard extends StatelessWidget {
  const _WorkspaceCard({required this.pageCount, required this.onOpen, required this.onNewBook});

  final int pageCount;
  final VoidCallback onOpen;
  final VoidCallback? onNewBook;

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: EdgeInsets.zero,
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onOpen,
        child: Padding(
          padding: const EdgeInsets.all(18),
          child: Row(
            children: [
              const Icon(Icons.menu_book, size: 40, color: HarfColors.gold),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('$pageCount',
                        style: const TextStyle(
                            fontSize: 30, fontWeight: FontWeight.w700, color: HarfColors.brightGold)),
                    Text(pageCount == 1 ? 'page digitised' : 'pages digitised'),
                  ],
                ),
              ),
              if (onNewBook != null)
                IconButton(
                  tooltip: 'Start a new book',
                  icon: const Icon(Icons.restart_alt),
                  onPressed: onNewBook,
                ),
              const Icon(Icons.chevron_right, color: HarfColors.gold),
            ],
          ),
        ),
      ),
    );
  }
}

class _ServerStatusDot extends StatelessWidget {
  const _ServerStatusDot({required this.online, required this.onTap});

  final bool? online;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final (color, label) = switch (online) {
      null => (Colors.grey, 'Checking server...'),
      true => (Colors.greenAccent, 'Server online'),
      false => (Colors.redAccent, 'Server offline - tap to retry'),
    };
    return IconButton(
      tooltip: label,
      onPressed: onTap,
      icon: Icon(Icons.circle, size: 14, color: color),
    );
  }
}
