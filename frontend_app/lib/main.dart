// main.dart
//
// HarfScan - Bilingual (Urdu & Arabic) OCR Scanner for historical books.
//
// Pages:
//   1. LoginScreen          - sign in / create account (stays signed in)
//   2. HomeScreen           - welcome, Camera Scan / Gallery Upload, my books
//   3. SelectTextScreen     - drag from the first word to the last word
//   4. BookWorkspaceScreen  - the book's pages, review highlights, export
//   +  SettingsScreen       - script, how it works, server, review, account
//
// Books are saved on the phone per user (book_session.dart) and the app
// reopens where the user left off.

import 'package:flutter/material.dart';

import 'app_settings.dart';
import 'auth_service.dart';
import 'book_session.dart';
import 'home_screen.dart';
import 'login_screen.dart';
import 'theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final settings = await AppSettings.load();
  final auth = await AuthService.load();
  runApp(HarfScanApp(settings: settings, auth: auth));
}

class HarfScanApp extends StatelessWidget {
  const HarfScanApp({super.key, required this.settings, required this.auth});

  final AppSettings settings;
  final AuthService auth;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'HarfScan',
      debugShowCheckedModeBanner: false,
      theme: buildHarfTheme(),
      home: _AuthGate(settings: settings, auth: auth),
    );
  }
}

/// Login page when signed out; the user's library and home page when
/// signed in.
class _AuthGate extends StatefulWidget {
  const _AuthGate({required this.settings, required this.auth});

  final AppSettings settings;
  final AuthService auth;

  @override
  State<_AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends State<_AuthGate> {
  BookLibrary? _library;
  Future<BookLibrary>? _opening;
  late final AppLifecycleListener _lifecycle = AppLifecycleListener(
    // Write pending changes before Android may stop the app.
    onPause: () => _library?.flush(),
    onDetach: () => _library?.flush(),
  );

  @override
  void initState() {
    super.initState();
    _lifecycle; // start listening
    widget.auth.addListener(_userChanged);
    _userChanged(rebuild: false);
  }

  @override
  void dispose() {
    widget.auth.removeListener(_userChanged);
    _lifecycle.dispose();
    _library?.dispose();
    super.dispose();
  }

  void _userChanged({bool rebuild = true}) {
    final user = widget.auth.user;
    if (user?.id == _library?.userId && (user == null) == (_library == null)) return;
    _library?.flush();
    _library?.dispose();
    _library = null;
    _opening = user == null ? null : BookLibrary.open(user.id);
    if (user == null) widget.settings.lastScreen = 'home';
    if (rebuild) setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final opening = _opening;
    if (opening == null) return LoginScreen(auth: widget.auth);
    return FutureBuilder<BookLibrary>(
      future: opening,
      builder: (context, snap) {
        if (snap.hasError) {
          return Scaffold(body: Center(child: Text('Could not open your books: ${snap.error}')));
        }
        final library = snap.data;
        if (library == null) {
          return const Scaffold(
            body: HarfBackground(child: Center(child: CircularProgressIndicator(color: HarfColors.gold))),
          );
        }
        _library = library;
        return HomeScreen(library: library, settings: widget.settings, auth: widget.auth);
      },
    );
  }
}
