// settings_screen.dart
//
// Settings: OCR server address (with a connection test), review
// highlighting of low-confidence words, the current book, and About.
// Everything is remembered between launches (see app_settings.dart).

import 'package:flutter/material.dart';

import 'app_settings.dart';
import 'book_session.dart';
import 'theme.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key, required this.settings, required this.session});

  final AppSettings settings;
  final BookSession session;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final TextEditingController _url = TextEditingController(text: widget.settings.serverUrl);
  bool _saving = false;

  AppSettings get _settings => widget.settings;

  @override
  void dispose() {
    _url.dispose();
    super.dispose();
  }

  Future<void> _saveAndTest() async {
    FocusScope.of(context).unfocus();
    setState(() => _saving = true);
    await _settings.setServerUrl(_url.text);
    if (!mounted) return;
    _url.text = _settings.serverUrl;
    setState(() => _saving = false);
    final ok = _settings.serverOnline == true;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text(ok ? 'Connected to the OCR server' : 'Server not reachable at ${_settings.serverUrl}')));
  }

  Future<void> _clearBook() async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: HarfColors.slate,
        title: const Text('Start a new book?'),
        content: Text('All ${widget.session.pageCount} page(s) in the workspace will be removed. '
            'Export them first if you need them.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Clear book')),
        ],
      ),
    );
    if (ok == true) widget.session.clear();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(title: const Text('Settings')),
      body: HarfBackground(
        child: SafeArea(
          child: ListenableBuilder(
            listenable: Listenable.merge([_settings, widget.session]),
            builder: (context, _) => ListView(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 32),
              children: [
                const SectionLabel('OCR server'),
                _serverCard(),
                const SizedBox(height: 24),
                const SectionLabel('Review'),
                _reviewCard(),
                const SizedBox(height: 24),
                const SectionLabel('Book'),
                Card(
                  margin: EdgeInsets.zero,
                  child: ListTile(
                    leading: const Icon(Icons.delete_sweep, color: HarfColors.gold),
                    title: const Text('Start a new book'),
                    subtitle: Text('${widget.session.pageCount} page(s) in the current workspace'),
                    enabled: !widget.session.isEmpty,
                    onTap: _clearBook,
                  ),
                ),
                const SizedBox(height: 24),
                const SectionLabel('About'),
                _aboutCard(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _serverCard() {
    final online = _settings.serverOnline;
    final (color, status) = switch (online) {
      null => (Colors.grey, 'Checking...'),
      true => (Colors.greenAccent, 'Connected'),
      false => (Colors.redAccent, 'Not reachable'),
    };
    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(children: [
              Icon(Icons.circle, size: 12, color: color),
              const SizedBox(width: 8),
              Text(status, style: TextStyle(color: color, fontWeight: FontWeight.w600)),
              const Spacer(),
              TextButton.icon(
                onPressed: _settings.checkServer,
                icon: const Icon(Icons.refresh, size: 18),
                label: const Text('Check'),
              ),
            ]),
            const SizedBox(height: 8),
            TextField(
              controller: _url,
              keyboardType: TextInputType.url,
              autocorrect: false,
              decoration: InputDecoration(
                labelText: 'Server address',
                hintText: 'http://192.168.100.14:8000',
                border: const OutlineInputBorder(),
                focusedBorder: const OutlineInputBorder(
                    borderSide: BorderSide(color: HarfColors.gold, width: 1.5)),
                prefixIcon: const Icon(Icons.dns_outlined),
                suffixIcon: _saving
                    ? const Padding(
                        padding: EdgeInsets.all(14),
                        child: SizedBox(
                            width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2)),
                      )
                    : null,
              ),
              onSubmitted: (_) => _saveAndTest(),
            ),
            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: _saving ? null : _saveAndTest,
              icon: const Icon(Icons.save),
              label: const Text('Save & test connection'),
              style: FilledButton.styleFrom(minimumSize: const Size(0, 48)),
            ),
            const SizedBox(height: 10),
            Text(
              'Run the backend on your PC and keep the phone on the same Wi-Fi. '
              'Use the PC\'s Wi-Fi IP address with port 8000.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }

  Widget _reviewCard() {
    final t = _settings.lowConfidenceThreshold;
    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Column(
          children: [
            SwitchListTile(
              value: _settings.highlightLowConfidence,
              onChanged: (v) => _settings.highlightLowConfidence = v,
              activeThumbColor: HarfColors.gold,
              secondary: const Icon(Icons.highlight, color: HarfColors.gold),
              title: const Text('Highlight uncertain words'),
              subtitle: const Text('Gold background on words the OCR is unsure about'),
            ),
            ListTile(
              enabled: _settings.highlightLowConfidence,
              leading: const Icon(Icons.tune, color: HarfColors.gold),
              title: Text('Highlight below ${t.round()}% confidence'),
              subtitle: Slider(
                value: t,
                min: 50,
                max: 95,
                divisions: 9,
                label: '${t.round()}%',
                activeColor: HarfColors.gold,
                onChanged: _settings.highlightLowConfidence
                    ? (v) => _settings.lowConfidenceThreshold = v
                    : null,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _aboutCard() => Card(
        margin: EdgeInsets.zero,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Center(child: HarfLogo(size: 56, showTagline: false)),
              const SizedBox(height: 12),
              const Text('Bilingual (Urdu & Arabic) OCR Scanner for Historical Books',
                  style: TextStyle(fontWeight: FontWeight.w600)),
              const SizedBox(height: 8),
              for (final line in const [
                'Urdu: UTRNet (HRNet + BiLSTM + CTC)',
                'Arabic: PaddleOCR PP-OCRv5 Arabic',
                'Text detection: PaddleOCR PP-OCRv5',
                'Layout: LayoutParser + confidence routing',
              ])
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 2),
                  child: Text('• $line', style: Theme.of(context).textTheme.bodySmall),
                ),
              const SizedBox(height: 8),
              Text('Version 2.1.0', style: Theme.of(context).textTheme.bodySmall),
            ],
          ),
        ),
      );
}
