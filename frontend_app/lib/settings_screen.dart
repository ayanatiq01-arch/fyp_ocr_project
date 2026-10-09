// settings_screen.dart
//
// Settings: how the app works, server connection, review highlighting,
// account and About. Everything is
// remembered between launches (see app_settings.dart).

import 'package:flutter/material.dart';

import 'app_settings.dart';
import 'auth_service.dart';
import 'theme.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key, required this.settings, required this.auth});

  final AppSettings settings;
  final AuthService auth;

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
        content: Text(ok ? 'Connected to the server' : 'Server not reachable at ${_settings.serverUrl}')));
  }

  Future<void> _findServer() async {
    final ok = await _settings.discoverServer();
    if (!mounted) return;
    _url.text = _settings.serverUrl;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text(ok
            ? 'Server found at ${_settings.serverUrl}'
            : 'No HarfScan server found on this Wi-Fi')));
  }

  Future<void> _signOut() async {
    // Back to the home page first; once its transition has finished, the
    // home page is replaced by the login page.
    Navigator.of(context).popUntil((route) => route.isFirst);
    await Future<void>.delayed(const Duration(milliseconds: 450));
    await widget.auth.signOut();
  }

  Future<void> _changePassword() async {
    final current = TextEditingController();
    final next = TextEditingController();
    String? error;
    await showDialog<void>(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setDialog) => AlertDialog(
          backgroundColor: HarfColors.slate,
          title: const Text('Change password'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                  controller: current,
                  obscureText: true,
                  decoration: const InputDecoration(labelText: 'Current password')),
              TextField(
                  controller: next,
                  obscureText: true,
                  decoration: const InputDecoration(
                      labelText: 'New password', helperText: 'At least 6 characters, with a number')),
              if (error != null)
                Padding(
                  padding: const EdgeInsets.only(top: 10),
                  child: Text(error!, style: const TextStyle(color: Colors.redAccent)),
                ),
            ],
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
            FilledButton(
              onPressed: () async {
                final problem = await widget.auth.changePassword(current.text, next.text);
                if (problem != null) {
                  setDialog(() => error = problem);
                  return;
                }
                if (ctx.mounted) Navigator.pop(ctx);
                if (mounted) {
                  ScaffoldMessenger.of(context)
                      .showSnackBar(const SnackBar(content: Text('Password changed')));
                }
              },
              child: const Text('Change'),
            ),
          ],
        ),
      ),
    );
    current.dispose();
    next.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(title: const Text('Settings')),
      body: HarfBackground(
        child: SafeArea(
          child: ListenableBuilder(
            listenable: Listenable.merge([_settings, widget.auth]),
            builder: (context, _) => ListView(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 32),
              children: [
                const SectionLabel('How it works'),
                const _HowItWorks(),
                const SizedBox(height: 24),
                const SectionLabel('Server'),
                _serverCard(),
                const SizedBox(height: 24),
                const SectionLabel('Review'),
                _reviewCard(),
                const SizedBox(height: 24),
                const SectionLabel('Account'),
                _accountCard(),
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
                onPressed: _settings.discovering ? null : _settings.checkServer,
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
            const SizedBox(height: 8),
            OutlinedButton.icon(
              onPressed: _settings.discovering ? null : _findServer,
              icon: _settings.discovering
                  ? const SizedBox(
                      width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                  : const Icon(Icons.wifi_find),
              label: Text(_settings.discovering ? 'Searching the Wi-Fi...' : 'Find server automatically'),
              style: OutlinedButton.styleFrom(minimumSize: const Size(0, 48)),
            ),
            const SizedBox(height: 10),
            Text(
              'The server runs on your PC (start_server.bat). Keep the phone on the same Wi-Fi; '
              'the app finds the server by itself if its address changes.',
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
              subtitle: const Text('Gold background on words the reader is unsure about'),
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

  Widget _accountCard() {
    final user = widget.auth.user;
    return Card(
      margin: EdgeInsets.zero,
      child: Column(
        children: [
          ListTile(
            leading: CircleAvatar(
              backgroundColor: HarfColors.gold,
              foregroundColor: HarfColors.navy,
              child: Text((user?.name.isNotEmpty ?? false) ? user!.name[0].toUpperCase() : '?'),
            ),
            title: Text(user?.name ?? ''),
            subtitle: Text(user?.email ?? ''),
          ),
          const Divider(height: 1),
          ListTile(
            leading: const Icon(Icons.password, color: HarfColors.gold),
            title: const Text('Change password'),
            onTap: _changePassword,
          ),
          ListTile(
            leading: const Icon(Icons.logout, color: HarfColors.gold),
            title: const Text('Sign out'),
            subtitle: const Text('Your books stay saved on this phone'),
            onTap: _signOut,
          ),
        ],
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
                'Arabic: EasyOCR (local, open source)',
                'Text detection, layout and AI correction to Markdown',
                'Book export: PDF and Word',
              ])
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 2),
                  child: Text('• $line', style: Theme.of(context).textTheme.bodySmall),
                ),
              const SizedBox(height: 8),
              Text('Version 4.0.1', style: Theme.of(context).textTheme.bodySmall),
            ],
          ),
        ),
      );
}

class _HowItWorks extends StatelessWidget {
  const _HowItWorks();

  @override
  Widget build(BuildContext context) {
    const steps = [
      (Icons.translate, 'On the home page, choose the script of the page'),
      (Icons.photo_camera, 'Take or choose a photo of the page'),
      (Icons.touch_app, 'Drag from the first word to the last word'),
      (Icons.auto_stories, 'Add it to the book, then scan the next page'),
      (Icons.ios_share, 'Export the whole book as PDF or Word'),
    ];
    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 14),
        child: Column(
          children: [
            for (final (i, (icon, text)) in steps.indexed)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 6),
                child: Row(children: [
                  CircleAvatar(
                    radius: 13,
                    backgroundColor: HarfColors.gold.withValues(alpha: 0.2),
                    child: Text('${i + 1}',
                        style: const TextStyle(
                            fontSize: 12, color: HarfColors.brightGold, fontWeight: FontWeight.w700)),
                  ),
                  const SizedBox(width: 12),
                  Icon(icon, size: 20, color: HarfColors.gold),
                  const SizedBox(width: 10),
                  Expanded(child: Text(text, style: Theme.of(context).textTheme.bodyMedium)),
                ]),
              ),
          ],
        ),
      ),
    );
  }
}
