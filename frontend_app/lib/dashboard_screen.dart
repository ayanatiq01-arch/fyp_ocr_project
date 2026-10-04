// dashboard_screen.dart
//
// Page 2 - Add a page: Camera Scan or Gallery Upload, plus the
// "Active Book Workspace" with the number of digitised pages.
// The chosen script is shown at the top; back goes to the script choice.

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'app_settings.dart';
import 'book_session.dart';
import 'book_workspace_screen.dart';
import 'select_text_screen.dart';
import 'settings_screen.dart';
import 'theme.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key, required this.session, required this.settings});

  final BookSession session;
  final AppSettings settings;

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  BookSession get _session => widget.session;
  AppSettings get _settings => widget.settings;

  @override
  void initState() {
    super.initState();
    _settings.ensureServer(); // finds the server again if the PC's IP changed
    _recoverLostPhoto();
  }

  /// Android may kill the app while the camera is open (low memory); the
  /// photo is then delivered on the next launch via retrieveLostData().
  Future<void> _recoverLostPhoto() async {
    if (!Platform.isAndroid) return;
    final lost = await ImagePicker().retrieveLostData();
    if (lost.isEmpty || lost.file == null || !mounted) return;
    final added = await Navigator.of(context).push<bool>(MaterialPageRoute(
      builder: (_) =>
          SelectTextScreen(image: File(lost.file!.path), session: _session, settings: _settings),
    ));
    if (added == true) _openWorkspace();
  }

  Future<void> _scan(ImageSource source) async {
    final added = await SelectTextScreen.pickAndSelect(context,
        source: source, session: _session, settings: _settings);
    if (added) _openWorkspace();
  }

  void _openWorkspace() {
    if (!mounted) return;
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => BookWorkspaceScreen(session: _session, settings: _settings),
    ));
  }

  void _openSettings() => Navigator.of(context).push(MaterialPageRoute(
        builder: (_) => SettingsScreen(settings: _settings, session: _session),
      ));

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Listenable.merge([_session, _settings]),
      builder: (context, _) => Scaffold(
        extendBodyBehindAppBar: true,
        appBar: AppBar(
          title: const Text('HarfScan'),
          actions: [
            _ServerStatusDot(online: _settings.serverOnline, onTap: _settings.checkServer),
            IconButton(tooltip: 'Settings', icon: const Icon(Icons.settings), onPressed: _openSettings),
          ],
        ),
        body: HarfBackground(
          child: SafeArea(
            child: ListView(
              padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
              children: [
                _LanguageBanner(
                  language: _session.language,
                  onChange: () => Navigator.of(context).pop(),
                ),
                if (_settings.serverOnline == false) ...[
                  const SizedBox(height: 12),
                  _OfflineBanner(url: _settings.serverUrl, onTap: _openSettings),
                ],
                const SizedBox(height: 24),
                const SectionLabel('Add a page'),
                _ActionCard(
                  icon: Icons.photo_camera,
                  title: 'Camera Scan',
                  subtitle: 'Take a photo of the book page',
                  onTap: () => _scan(ImageSource.camera),
                ),
                const SizedBox(height: 14),
                _ActionCard(
                  icon: Icons.photo_library,
                  title: 'Gallery Upload',
                  subtitle: 'Choose a photo you already took',
                  onTap: () => _scan(ImageSource.gallery),
                ),
                const SizedBox(height: 28),
                const SectionLabel('Active Book Workspace'),
                _WorkspaceCard(pageCount: _session.pageCount, onOpen: _openWorkspace),
                const SizedBox(height: 24),
                const _HowItWorks(),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// --------------------------------------------------------------------------
// Widgets
// --------------------------------------------------------------------------

class _LanguageBanner extends StatelessWidget {
  const _LanguageBanner({required this.language, required this.onChange});

  final OcrLanguage language;
  final VoidCallback onChange;

  @override
  Widget build(BuildContext context) {
    final sample = switch (language) {
      OcrLanguage.urdu => 'اردو',
      OcrLanguage.arabic => 'العربية',
      OcrLanguage.mixed => 'اردو + عربی',
    };
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 8, 8, 8),
      decoration: BoxDecoration(
        color: HarfColors.gold.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: HarfColors.gold.withValues(alpha: 0.6)),
      ),
      child: Row(
        children: [
          Text(sample,
              textDirection: TextDirection.rtl,
              style: scriptStyle(language == OcrLanguage.arabic ? 'arabic' : 'urdu', size: 20)),
          const SizedBox(width: 12),
          Expanded(child: Text('Script: ${language.label}')),
          TextButton(onPressed: onChange, child: const Text('Change')),
        ],
      ),
    );
  }
}

class _OfflineBanner extends StatelessWidget {
  const _OfflineBanner({required this.url, required this.onTap});

  final String url;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.redAccent.withValues(alpha: 0.15),
      borderRadius: BorderRadius.circular(14),
      child: ListTile(
        onTap: onTap,
        leading: const Icon(Icons.cloud_off, color: Colors.redAccent),
        title: const Text('OCR server not reachable'),
        subtitle: Text('$url\nTap to open Settings'),
        isThreeLine: true,
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
          padding: const EdgeInsets.all(18),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: HarfColors.gold.withValues(alpha: 0.15),
                  border: Border.all(color: HarfColors.gold),
                ),
                child: Icon(icon, size: 30, color: HarfColors.brightGold),
              ),
              const SizedBox(width: 18),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 17)),
                    const SizedBox(height: 2),
                    Text(subtitle, style: Theme.of(context).textTheme.bodySmall),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right, color: HarfColors.gold),
            ],
          ),
        ),
      ),
    );
  }
}

class _WorkspaceCard extends StatelessWidget {
  const _WorkspaceCard({required this.pageCount, required this.onOpen});

  final int pageCount;
  final VoidCallback onOpen;

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
              const Text('Open', style: TextStyle(color: HarfColors.gold)),
              const Icon(Icons.chevron_right, color: HarfColors.gold),
            ],
          ),
        ),
      ),
    );
  }
}

class _HowItWorks extends StatelessWidget {
  const _HowItWorks();

  @override
  Widget build(BuildContext context) {
    const steps = [
      (Icons.photo_camera, 'Take or choose a photo of the page'),
      (Icons.touch_app, 'Drag from the first word to the last word'),
      (Icons.auto_stories, 'Add it to the book, then scan the next page'),
      (Icons.ios_share, 'Export the whole book as PDF or Word'),
    ];
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SectionLabel('How it works'),
        for (final (icon, text) in steps)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 5, horizontal: 4),
            child: Row(children: [
              Icon(icon, size: 20, color: HarfColors.gold),
              const SizedBox(width: 12),
              Expanded(child: Text(text, style: Theme.of(context).textTheme.bodyMedium)),
            ]),
          ),
      ],
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
