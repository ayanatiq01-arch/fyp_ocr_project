// auth_service.dart
//
// Sign-in for HarfScan. Accounts live on this phone (no extra server
// needed): the password is never stored - only a random salt and a
// PBKDF2-HMAC-SHA256 hash (20 000 rounds) of it. "Stay signed in" keeps
// the user logged in between app launches; each user has their own books.

import 'dart:convert';
import 'dart:math';

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

class UserAccount {
  const UserAccount({required this.id, required this.name, required this.email});

  /// Stable id, used for the folder that holds the user's books.
  final String id;
  final String name;
  final String email;
}

class AuthService extends ChangeNotifier {
  AuthService._(this._prefs) {
    final session = _prefs.getString(_kSession);
    if (session != null) _user = _account(session);
  }

  static Future<AuthService> load() async => AuthService._(await SharedPreferences.getInstance());

  static const _kAccounts = 'auth_accounts';
  static const _kSession = 'auth_session';
  static const _iterations = 20000;
  static final _emailRe = RegExp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$');

  final SharedPreferences _prefs;
  UserAccount? _user;

  /// The signed-in user, or null.
  UserAccount? get user => _user;
  bool get signedIn => _user != null;

  Map<String, dynamic> get _accounts =>
      jsonDecode(_prefs.getString(_kAccounts) ?? '{}') as Map<String, dynamic>;

  UserAccount? _account(String email) {
    final a = _accounts[email] as Map<String, dynamic>?;
    if (a == null) return null;
    return UserAccount(id: a['id'] as String, name: a['name'] as String, email: email);
  }

  /// Creates an account and signs in. Returns an error message, or null.
  Future<String?> signUp({
    required String name,
    required String email,
    required String password,
    bool remember = true,
  }) async {
    name = name.trim();
    email = email.trim().toLowerCase();
    final problem = validateName(name) ?? validateEmail(email) ?? validatePassword(password);
    if (problem != null) return problem;
    final accounts = _accounts;
    if (accounts.containsKey(email)) return 'An account with this email already exists';
    final salt = _randomBytes(16);
    accounts[email] = {
      'id': base64Url.encode(_randomBytes(9)),
      'name': name,
      'salt': base64.encode(salt),
      'hash': base64.encode(_pbkdf2(password, salt)),
      'created': DateTime.now().toIso8601String(),
    };
    await _prefs.setString(_kAccounts, jsonEncode(accounts));
    return _startSession(email, remember);
  }

  /// Signs in. Returns an error message, or null.
  Future<String?> signIn({required String email, required String password, bool remember = true}) async {
    email = email.trim().toLowerCase();
    final a = _accounts[email] as Map<String, dynamic>?;
    // Same message for "no account" and "wrong password".
    if (a == null) return 'Email or password is incorrect';
    final hash = _pbkdf2(password, base64.decode(a['salt'] as String));
    if (!_sameBytes(hash, base64.decode(a['hash'] as String))) {
      return 'Email or password is incorrect';
    }
    return _startSession(email, remember);
  }

  Future<String?> _startSession(String email, bool remember) async {
    _user = _account(email);
    if (remember) {
      await _prefs.setString(_kSession, email);
    } else {
      await _prefs.remove(_kSession);
    }
    notifyListeners();
    return null;
  }

  Future<void> signOut() async {
    _user = null;
    await _prefs.remove(_kSession);
    notifyListeners();
  }

  /// Changes the signed-in user's password. Returns an error message, or null.
  Future<String?> changePassword(String current, String next) async {
    final user = _user;
    if (user == null) return 'Not signed in';
    final accounts = _accounts;
    final a = accounts[user.email] as Map<String, dynamic>;
    if (!_sameBytes(_pbkdf2(current, base64.decode(a['salt'] as String)),
        base64.decode(a['hash'] as String))) {
      return 'Current password is incorrect';
    }
    final problem = validatePassword(next);
    if (problem != null) return problem;
    final salt = _randomBytes(16);
    a['salt'] = base64.encode(salt);
    a['hash'] = base64.encode(_pbkdf2(next, salt));
    await _prefs.setString(_kAccounts, jsonEncode(accounts));
    return null;
  }

  // ------------------------------------------------------------ validation

  static String? validateName(String name) =>
      name.trim().length < 2 ? 'Please enter your name' : null;

  static String? validateEmail(String email) =>
      _emailRe.hasMatch(email.trim()) ? null : 'Please enter a valid email address';

  static String? validatePassword(String password) {
    if (password.length < 6) return 'Password must be at least 6 characters';
    if (!password.contains(RegExp(r'[0-9]')) || !password.contains(RegExp(r'[A-Za-z]'))) {
      return 'Use letters and at least one number';
    }
    return null;
  }

  // ------------------------------------------------------------ crypto

  static List<int> _randomBytes(int n) {
    final r = Random.secure();
    return List<int>.generate(n, (_) => r.nextInt(256));
  }

  /// PBKDF2-HMAC-SHA256 with a 32-byte output (one block).
  @visibleForTesting
  static List<int> pbkdf2(String password, List<int> salt, {int iterations = _iterations}) {
    final hmac = Hmac(sha256, utf8.encode(password));
    var u = hmac.convert([...salt, 0, 0, 0, 1]).bytes;
    final out = List<int>.from(u);
    for (var i = 1; i < iterations; i++) {
      u = hmac.convert(u).bytes;
      for (var j = 0; j < out.length; j++) {
        out[j] ^= u[j];
      }
    }
    return out;
  }

  static List<int> _pbkdf2(String password, List<int> salt) => pbkdf2(password, salt);

  /// Constant-time comparison.
  static bool _sameBytes(List<int> a, List<int> b) {
    if (a.length != b.length) return false;
    var diff = 0;
    for (var i = 0; i < a.length; i++) {
      diff |= a[i] ^ b[i];
    }
    return diff == 0;
  }
}
