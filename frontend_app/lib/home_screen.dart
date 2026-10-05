// home_screen.dart
//
// Main page after sign-in: animated "Welcome to HarfScan", Camera Scan /
// Gallery Upload into the current book, and the user's library of books
// (create, open, rename, delete). Script choice and "How it works" live in
// Settings.

import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';

import 'app_settings.dart';
import 'auth_service.dart';
import 'book_session.dart';
import 'book_workspace_screen.dart';
import 'select_text_screen.dart';
import 'settings_screen.dart';
import 'theme.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, required this.library, required this.settings, required this.auth});

  final BookLibrary library;
  final AppSettings settings;
  final AuthService auth;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  late final AnimationController _intro =
      AnimationController(vsync: this, duration: const Duration(milliseconds: 1400))..forward();
  late final AnimationController _glow =
      AnimationController(vsync: this, duration: const Duration(seconds: 6))..repeat();

  BookLibrary get _library => widget.library;
  AppSettings get _settings => widget.settings;

  @override
  void initState() {
    super.initState();
    _settings.ensureServer(); // finds the server again if the PC's IP changed
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      await _recoverLostPhoto();
      // Continue where the user left off.
      if (mounted && _settings.lastScreen == 'workspace') _openWorkspace();
    });
  }

  @override
  void dispose() {
    _intro.dispose();
    _glow.dispose();
    super.dispose();
  }

  /// Android may kill the app while the camera is open (low memory); the
  /// photo is then delivered on the next launch via retrieveLostData().
  Future<void> _recoverLostPhoto() async {
    if (!Platform.isAndroid) return;
    final lost = await ImagePicker().retrieveLostData();
    if (lost.isEmpty || lost.file == null || !mounted) return;
    final added = await Navigator.of(context).push<bool>(MaterialPageRoute(
      builder: (_) => SelectTextScreen(
          image: File(lost.file!.path), session: _library.active, settings: _settings),
    ));
    if (added == true) _openWorkspace();
  }

  Future<void> _scan(ImageSource source) async {
    final added = await SelectTextScreen.pickAndSelect(context,
        source: source, session: _library.active, settings: _settings);
    if (added) _openWorkspace();
  }

  Future<void> _openWorkspace([BookSession? book]) async {
    if (book != null) await _library.setActive(book);
    if (!mounted) return;
    await Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => BookWorkspaceScreen(session: _library.active, settings: _settings),
    ));
  }

  void _openSettings() => Navigator.of(context).push(MaterialPageRoute(
        builder: (_) => SettingsScreen(settings: _settings, auth: widget.auth),
      ));

  Future<String?> _askTitle({required String title, String initial = ''}) async {
    final controller = TextEditingController(text: initial);
    final result = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: HarfColors.slate,
        title: Text(title),
        content: TextField(
          controller: controller,
          autofocus: true,
          textCapitalization: TextCapitalization.sentences,
          decoration: const InputDecoration(hintText: 'Book name'),
          onSubmitted: (v) => Navigator.pop(ctx, v),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, controller.text), child: const Text('Save')),
        ],
      ),
    );
    controller.dispose();
    return result == null || result.trim().isEmpty ? null : result.trim();
  }

  Future<void> _newBook() async {
    final title = await _askTitle(title: 'New book');
    if (title == null) return;
    await _library.create(title);
    if (mounted) {
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('"$title" created - new scans go into it')));
    }
  }

  Future<void> _bookMenu(BookSession book, String action) async {
    if (action == 'rename') {
      final title = await _askTitle(title: 'Rename book', initial: book.title);
      if (title != null) book.rename(title);
    } else if (action == 'delete') {
      final ok = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          backgroundColor: HarfColors.slate,
          title: Text('Delete "${book.title}"?'),
          content: Text('Its ${book.pageCount} page(s) will be deleted from this phone. '
              'Export it first if you need it.'),
          actions: [
            TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
            FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Delete')),
          ],
        ),
      );
      if (ok == true) await _library.delete(book);
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Listenable.merge([_library, _settings]),
      builder: (context, _) => Scaffold(
        extendBodyBehindAppBar: true,
        appBar: AppBar(
          actions: [
            _ServerStatusDot(online: _settings.serverOnline, onTap: _settings.ensureServer),
            IconButton(tooltip: 'Settings', icon: const Icon(Icons.settings), onPressed: _openSettings),
          ],
        ),
        body: HarfBackground(
          child: SafeArea(
            child: ListView(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 28),
              children: [
                _Welcome(intro: _intro, glow: _glow, name: widget.auth.user?.name ?? ''),
                const SizedBox(height: 26),
                _staggered(0.35, _currentBookCard()),
                const SizedBox(height: 22),
                _staggered(0.45, const SectionLabel('Add a page')),
                _staggered(
                  0.5,
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
                          subtitle: 'Choose a photo',
                          onTap: () => _scan(ImageSource.gallery),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 26),
                _staggered(
                  0.6,
                  Row(
                    children: [
                      const Expanded(child: SectionLabel('My books')),
                      TextButton.icon(
                        onPressed: _newBook,
                        icon: const Icon(Icons.add, size: 18),
                        label: const Text('New book'),
                      ),
                    ],
                  ),
                ),
                for (final book in _library.books)
                  _staggered(
                    0.65,
                    _BookTile(
                      book: book,
                      active: identical(book, _library.active),
                      onOpen: () => _openWorkspace(book),
                      onMenu: (action) => _bookMenu(book, action),
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _currentBookCard() {
    final book = _library.active;
    return Card(
      margin: EdgeInsets.zero,
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: _openWorkspace,
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
                    const Text('CONTINUE READING',
                        style: TextStyle(
                            color: HarfColors.gold, fontSize: 11, letterSpacing: 1.4, fontWeight: FontWeight.w600)),
                    const SizedBox(height: 2),
                    Text(book.title,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
                    Text('${book.pageCount} ${book.pageCount == 1 ? 'page' : 'pages'} digitised',
                        style: Theme.of(context).textTheme.bodySmall),
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

  /// Fades and slides [child] in, starting at [start] (0-1) of the intro.
  Widget _staggered(double start, Widget child) {
    final anim = CurvedAnimation(
        parent: _intro, curve: Interval(start.clamp(0, 0.9), 1, curve: Curves.easeOutCubic));
    return FadeTransition(
      opacity: anim,
      child: SlideTransition(
        position: Tween(begin: const Offset(0, 0.15), end: Offset.zero).animate(anim),
        child: child,
      ),
    );
  }
}

// --------------------------------------------------------------------------
// Widgets
// --------------------------------------------------------------------------

/// "Welcome to HarfScan" with a slowly turning gold ring around the logo.
class _Welcome extends StatelessWidget {
  const _Welcome({required this.intro, required this.glow, required this.name});

  final Animation<double> intro;
  final Animation<double> glow;
  final String name;

  @override
  Widget build(BuildContext context) {
    final logoIn = CurvedAnimation(parent: intro, curve: const Interval(0, 0.5, curve: Curves.elasticOut));
    final textIn = CurvedAnimation(parent: intro, curve: const Interval(0.15, 0.6, curve: Curves.easeOut));
    return Column(
      children: [
        const SizedBox(height: 4),
        ScaleTransition(
          scale: logoIn,
          child: SizedBox(
            width: 120,
            height: 120,
            child: Stack(
              alignment: Alignment.center,
              children: [
                AnimatedBuilder(
                  animation: glow,
                  builder: (context, _) => Transform.rotate(
                    angle: glow.value * 2 * math.pi,
                    child: Container(
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        gradient: SweepGradient(colors: [
                          HarfColors.gold.withValues(alpha: 0),
                          HarfColors.brightGold,
                          HarfColors.gold.withValues(alpha: 0),
                        ]),
                      ),
                    ),
                  ),
                ),
                Container(
                  width: 108,
                  height: 108,
                  decoration: const BoxDecoration(shape: BoxShape.circle, color: HarfColors.navy),
                ),
                const HarfLogo(size: 92, showTagline: false, showName: false),
              ],
            ),
          ),
        ),
        const SizedBox(height: 14),
        FadeTransition(
          opacity: textIn,
          child: Column(
            children: [
              Text('Welcome to',
                  style: Theme.of(context)
                      .textTheme
                      .titleMedium
                      ?.copyWith(color: HarfColors.ink.withValues(alpha: 0.8))),
              ShaderMask(
                shaderCallback: HarfColors.goldSheen.createShader,
                child: Text('HarfScan',
                    style: harfFont(GoogleFonts.cinzel,
                        fontSize: 40, fontWeight: FontWeight.w700, color: Colors.white, letterSpacing: 2)),
              ),
              if (name.isNotEmpty)
                Text('Hello, ${name.split(' ').first}!',
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: HarfColors.gold)),
            ],
          ),
        ),
      ],
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
          padding: const EdgeInsets.symmetric(vertical: 22, horizontal: 12),
          child: Column(
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
              const SizedBox(height: 12),
              Text(title,
                  textAlign: TextAlign.center,
                  style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
              const SizedBox(height: 2),
              Text(subtitle, textAlign: TextAlign.center, style: Theme.of(context).textTheme.bodySmall),
            ],
          ),
        ),
      ),
    );
  }
}

class _BookTile extends StatelessWidget {
  const _BookTile(
      {required this.book, required this.active, required this.onOpen, required this.onMenu});

  final BookSession book;
  final bool active;
  final VoidCallback onOpen;
  final ValueChanged<String> onMenu;

  @override
  Widget build(BuildContext context) {
    final d = book.updatedAt;
    final date = '${d.day}/${d.month}/${d.year}';
    return Card(
      margin: const EdgeInsets.only(bottom: 10),
      child: ListTile(
        onTap: onOpen,
        leading: Icon(active ? Icons.auto_stories : Icons.book_outlined,
            color: active ? HarfColors.brightGold : HarfColors.gold),
        title: Text(book.title, maxLines: 1, overflow: TextOverflow.ellipsis),
        subtitle: Text('${book.pageCount} ${book.pageCount == 1 ? 'page' : 'pages'}  ·  $date'
            '${active ? '  ·  current' : ''}'),
        trailing: PopupMenuButton<String>(
          onSelected: onMenu,
          color: HarfColors.slate,
          itemBuilder: (_) => const [
            PopupMenuItem(value: 'rename', child: Text('Rename')),
            PopupMenuItem(value: 'delete', child: Text('Delete')),
          ],
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
      null => (Colors.grey, 'Connecting...'),
      true => (Colors.greenAccent, 'Server online'),
      false => (Colors.redAccent, 'Server offline - tap to search again'),
    };
    return IconButton(
      tooltip: label,
      onPressed: onTap,
      icon: Icon(Icons.circle, size: 14, color: color),
    );
  }
}
