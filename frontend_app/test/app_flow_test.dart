import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend_app/app_settings.dart';
import 'package:frontend_app/auth_service.dart';
import 'package:frontend_app/main.dart';
import 'package:frontend_app/theme.dart';
import 'package:path_provider_platform_interface/path_provider_platform_interface.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';
import 'package:shared_preferences/shared_preferences.dart';

class _FakePaths extends Fake with MockPlatformInterfaceMixin implements PathProviderPlatform {
  _FakePaths(this.dir);
  final String dir;

  @override
  Future<String?> getApplicationDocumentsPath() async => dir;
}

/// Lets real file I/O run (the library is read and written with dart:io)
/// and pumps frames until [finder] finds something.
Future<void> pumpUntil(WidgetTester tester, Finder finder) async {
  for (var i = 0; i < 40 && finder.evaluate().isEmpty; i++) {
    await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 100)));
    await tester.pump(const Duration(milliseconds: 100));
  }
}

void main() {
  setUpAll(() => harfUseGoogleFonts = false);

  testWidgets('Create account -> home -> new book -> settings -> sign out', (tester) async {
    final docs = Directory.systemTemp.createTempSync('harfscan_test');
    PathProviderPlatform.instance = _FakePaths(docs.path);
    SharedPreferences.setMockInitialValues({'server_url': 'http://127.0.0.1:9'});

    late AppSettings settings;
    late AuthService auth;
    await tester.runAsync(() async {
      settings = await AppSettings.load();
      auth = await AuthService.load();
    });
    await tester.pumpWidget(HarfScanApp(settings: settings, auth: auth));
    await tester.pump(const Duration(seconds: 2));

    // 1. Login page.
    expect(find.text('Sign in to continue'), findsOneWidget);
    await tester.tap(find.text('New to HarfScan? Create an account'));
    await tester.pump(const Duration(milliseconds: 400));
    await tester.enterText(find.widgetWithText(TextFormField, 'Full name'), 'Ayan Atiq');
    await tester.enterText(find.widgetWithText(TextFormField, 'Email'), 'ayan@example.com');
    await tester.enterText(find.widgetWithText(TextFormField, 'Password'), 'secret12');
    await tester.enterText(find.widgetWithText(TextFormField, 'Confirm password'), 'secret12');
    await tester.runAsync(() async {
      await tester.tap(find.text('Create account'));
      await Future<void>.delayed(const Duration(seconds: 2)); // hashing + opening the library
    });
    await tester.pump(const Duration(seconds: 3));

    // 2. Home page with a first book.
    await pumpUntil(tester, find.text('Welcome to'));
    await tester.pump(const Duration(seconds: 2)); // intro animation
    expect(find.text('Welcome to'), findsOneWidget);
    expect(find.text('Hello, Ayan!'), findsOneWidget);
    expect(find.text('My First Book'), findsWidgets);
    expect(find.text('Camera Scan'), findsOneWidget);
    expect(find.textContaining('Urdu'), findsNothing); // scripts live in Settings now

    // New book.
    await tester.scrollUntilVisible(find.text('New book'), 200);
    await tester.tap(find.text('New book'));
    await tester.pump(const Duration(milliseconds: 500));
    await tester.enterText(find.byType(TextField).last, 'Qawaid');
    await tester.runAsync(() async {
      await tester.tap(find.text('Save'));
      await Future<void>.delayed(const Duration(milliseconds: 800));
    });
    await pumpUntil(tester, find.text('"Qawaid" created - new scans go into it'));
    await tester.pump(const Duration(seconds: 1));
    expect(find.text('Qawaid'), findsWidgets);
    expect(find.text('CONTINUE READING'), findsOneWidget);
    expect(Directory('${docs.path}/books/${auth.user!.id}').listSync().length, 2); // saved on "disk"

    // 3. Settings: scripts, how it works, account; no API name anywhere.
    await tester.tap(find.byIcon(Icons.settings));
    await pumpUntil(tester, find.text('Mixed (Auto)'));
    expect(find.text('Urdu'), findsOneWidget);
    expect(find.text('Mixed (Auto)'), findsOneWidget);
    expect(find.textContaining('Gemini'), findsNothing);
    await tester.scrollUntilVisible(find.text('Sign out'), 300, scrollable: find.byType(Scrollable).first);
    await tester.runAsync(() async {
      await tester.tap(find.text('Sign out'));
      await Future<void>.delayed(const Duration(milliseconds: 50));
    });
    for (var i = 0; i < 6; i++) {
      await tester.pump(const Duration(milliseconds: 100)); // settings closes
    }
    await tester.runAsync(() async {
      await Future<void>.delayed(const Duration(milliseconds: 600));
    });
    await pumpUntil(tester, find.text('Sign in to continue'));
    expect(find.text('Sign in to continue'), findsOneWidget);
    await tester.pumpWidget(const SizedBox()); // stop the home page animations
    await tester.pump(const Duration(seconds: 10)); // let snack bar timers finish
  });
}
