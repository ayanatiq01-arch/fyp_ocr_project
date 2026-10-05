import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:frontend_app/api_service.dart';
import 'package:frontend_app/auth_service.dart';
import 'package:frontend_app/book_session.dart';
import 'package:shared_preferences/shared_preferences.dart';

String _hex(List<int> b) => b.map((x) => x.toRadixString(16).padLeft(2, '0')).join();

final _result = OcrResult.fromJson({
  'request_id': 'r1',
  'image': {'width': 100, 'height': 200},
  'rotation': 0,
  'skew_angle': 0,
  'layout_engine': 'gemini',
  'ai_correction': 'gemini-3.8-flash',
  'formatted_text': '',
  'processing_ms': 900,
  'blocks': [
    {
      'id': 0,
      'type': 'Table',
      'bbox': [0, 0, 10, 10],
      'columns': 2,
      'language': 'urdu',
      'rows': [
        {
          'bbox': [0, 0, 10, 10],
          'is_bullet': false,
          'cells': [
            {
              'bbox': [5, 0, 10, 10],
              'column': 0,
              'text': 'سَمِعَ',
              'language': 'arabic',
              'engine': 'Gemini',
              'confidence': 99,
              'candidates': {
                'arabic': {'text': 'سَمِعَ', 'confidence': 99}
              }
            },
            {
              'bbox': [0, 0, 5, 10],
              'column': 1,
              'text': 'اس نے سنا',
              'language': 'urdu',
              'engine': 'Gemini',
              'confidence': 99,
              'candidates': {
                'urdu': {'text': 'اس نے سنا', 'confidence': 99}
              }
            },
          ]
        }
      ]
    }
  ],
});

void main() {
  test('PBKDF2-HMAC-SHA256 matches the RFC 7914 / published test vectors', () {
    expect(_hex(AuthService.pbkdf2('password', utf8.encode('salt'), iterations: 1)),
        '120fb6cffcf8b32c43e7225256c4f837a86548c92ccc35480805987cb70be17b');
    expect(_hex(AuthService.pbkdf2('password', utf8.encode('salt'), iterations: 2)),
        'ae4d0c95af6b46d32d0adff928f06dd02a303f8ef3c251dfd6e2d85a95474c43');
  });

  test('Sign up, sign in, wrong password, sign out', () async {
    SharedPreferences.setMockInitialValues({});
    final auth = await AuthService.load();
    expect(await auth.signUp(name: 'Ayan', email: 'ayan@example.com', password: 'abc'),
        'Password must be at least 6 characters');
    expect(await auth.signUp(name: 'Ayan', email: 'Ayan@Example.com', password: 'secret12'), isNull);
    expect(auth.user?.email, 'ayan@example.com');
    await auth.signOut();
    expect(auth.signedIn, isFalse);
    expect(await auth.signIn(email: 'ayan@example.com', password: 'wrong123'),
        'Email or password is incorrect');
    expect(await auth.signIn(email: 'AYAN@example.com ', password: 'secret12'), isNull);
    expect(await auth.signUp(name: 'X', email: 'ayan@example.com', password: 'secret12'),
        'Please enter your name');

    // "Stay signed in": a new AuthService (next app launch) is still signed in.
    expect((await AuthService.load()).user?.name, 'Ayan');
    // The password itself is never stored.
    final prefs = await SharedPreferences.getInstance();
    expect(prefs.getString('auth_accounts'), isNot(contains('secret12')));
  });

  test('OCR result survives a save / load round trip', () {
    final back = OcrResult.fromJson(jsonDecode(jsonEncode(_result.toJson())) as Map<String, dynamic>);
    expect(back.currentText(), _result.currentText());
    expect(back.blocks.single.isTable, isTrue);
    expect(back.cells.first.candidates['arabic']?.text, 'سَمِعَ');
    expect(back.aiCorrection, 'gemini-3.8-flash');
  });

  test('A book with pages survives a save / load round trip', () {
    final book = BookSession(title: 'Qawaid')
      ..addPage(_result, '/tmp/p1.jpg')
      ..addPage(_result, '/tmp/p2.jpg');
    book.removePage(book.pages.first);
    final back = BookSession.fromJson(jsonDecode(jsonEncode(book.toJson())) as Map<String, dynamic>);
    expect(back.id, book.id);
    expect(back.title, 'Qawaid');
    expect(back.pageCount, 1);
    expect(back.pages.single.number, 1);
    expect(back.pages.single.imagePath, '/tmp/p2.jpg');
    expect(back.masterText, book.masterText);
  });
}
