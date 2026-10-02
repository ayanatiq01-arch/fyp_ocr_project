// language_screen.dart
//
// Page 1 - Choose the script of the book: Urdu, Arabic or Mixed.
// Tapping a script opens the scan page (camera / gallery).

import 'package:flutter/material.dart';

import 'app_settings.dart';
import 'book_session.dart';
import 'dashboard_screen.dart';
import 'settings_screen.dart';
import 'theme.dart';

class LanguageScreen extends StatelessWidget {
  const LanguageScreen({super.key, required this.session, required this.settings});

  final BookSession session;
  final AppSettings settings;

  void _choose(BuildContext context, OcrLanguage language) {
    session.language = language;
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => DashboardScreen(session: session, settings: settings),
    ));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        actions: [
          IconButton(
            tooltip: 'Settings',
            icon: const Icon(Icons.settings),
            onPressed: () => Navigator.of(context).push(MaterialPageRoute(
              builder: (_) => SettingsScreen(settings: settings, session: session),
            )),
          ),
        ],
      ),
      body: HarfBackground(
        child: SafeArea(
          child: ListView(
            padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
            children: [
              const HarfLogo(),
              const SizedBox(height: 32),
              Text('Which script is your book in?',
                  textAlign: TextAlign.center, style: Theme.of(context).textTheme.titleLarge),
              const SizedBox(height: 6),
              Text('Choose the language of the pages you want to digitise.',
                  textAlign: TextAlign.center, style: Theme.of(context).textTheme.bodySmall),
              const SizedBox(height: 24),
              _ScriptCard(
                sample: 'اردو',
                sampleLanguage: 'urdu',
                title: 'Urdu',
                subtitle: 'Nastaliq script - read by UTRNet',
                onTap: () => _choose(context, OcrLanguage.urdu),
              ),
              const SizedBox(height: 14),
              _ScriptCard(
                sample: 'العربية',
                sampleLanguage: 'arabic',
                title: 'Arabic',
                subtitle: 'Naskh script - read by PaddleOCR',
                onTap: () => _choose(context, OcrLanguage.arabic),
              ),
              const SizedBox(height: 14),
              _ScriptCard(
                sample: 'اردو + عربی',
                sampleLanguage: 'urdu',
                title: 'Mixed (Auto)',
                subtitle: 'Both engines read, the more confident one wins',
                onTap: () => _choose(context, OcrLanguage.mixed),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ScriptCard extends StatelessWidget {
  const _ScriptCard({
    required this.sample,
    required this.sampleLanguage,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  final String sample;
  final String sampleLanguage;
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
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
          child: Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title,
                        style: const TextStyle(
                            fontSize: 19, fontWeight: FontWeight.w600, color: HarfColors.brightGold)),
                    const SizedBox(height: 2),
                    Text(subtitle, style: Theme.of(context).textTheme.bodySmall),
                  ],
                ),
              ),
              const SizedBox(width: 12),
              Text(sample,
                  textDirection: TextDirection.rtl,
                  style: scriptStyle(sampleLanguage, size: 26, color: HarfColors.ink)),
              const SizedBox(width: 8),
              const Icon(Icons.chevron_right, color: HarfColors.gold),
            ],
          ),
        ),
      ),
    );
  }
}
