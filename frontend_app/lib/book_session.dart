// book_session.dart
//
// Books and the user's library. A BookSession is one book: its pages, in
// order. BookLibrary holds all books of the signed-in user, remembers which
// one was open, and saves every change to the phone right away, so the app
// opens where the user left off.

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'api_service.dart';

/// Script of the book (sent to the backend as "language").
enum OcrLanguage {
  urdu('urdu', 'Urdu'),
  arabic('arabic', 'Arabic'),
  mixed('mixed', 'Mixed (Auto)');

  const OcrLanguage(this.apiValue, this.label);

  /// Value of the backend's `language` form field.
  final String apiValue;
  final String label;

  static OcrLanguage parse(String? value) =>
      values.firstWhere((l) => l.apiValue == value, orElse: () => mixed);
}

/// One digitised page of a book.
class BookPage {
  BookPage({required this.number, required this.result, required this.imagePath, DateTime? createdAt})
      : createdAt = createdAt ?? DateTime.now();

  factory BookPage.fromJson(Map<String, dynamic> json) => BookPage(
        number: json['number'] as int,
        result: OcrResult.fromJson(json['result'] as Map<String, dynamic>),
        imagePath: json['image_path'] as String? ?? '',
        createdAt: DateTime.tryParse(json['created_at'] as String? ?? ''),
      );

  /// 1-based page number in the book.
  int number;
  final OcrResult result;
  final String imagePath;
  final DateTime createdAt;

  /// Page text with paragraphs, bullets and table columns (TAB) preserved.
  String get text => result.currentText();

  /// Header line used in the master document.
  String get header => '=== Page $number ===';

  Map<String, dynamic> toJson() => {
        'number': number,
        'result': result.toJson(),
        'image_path': imagePath,
        'created_at': createdAt.toIso8601String(),
      };
}

/// One book: its pages in order.
class BookSession extends ChangeNotifier {
  BookSession({String? id, this.title = 'My Book', DateTime? createdAt, DateTime? updatedAt})
      : id = id ?? DateTime.now().microsecondsSinceEpoch.toRadixString(36),
        createdAt = createdAt ?? DateTime.now(),
        updatedAt = updatedAt ?? DateTime.now();

  factory BookSession.fromJson(Map<String, dynamic> json) {
    final book = BookSession(
      id: json['id'] as String,
      title: json['title'] as String? ?? 'My Book',
      createdAt: DateTime.tryParse(json['created_at'] as String? ?? ''),
      updatedAt: DateTime.tryParse(json['updated_at'] as String? ?? ''),
    );
    book._pages.addAll((json['pages'] as List<dynamic>? ?? [])
        .map((p) => BookPage.fromJson(p as Map<String, dynamic>)));
    return book;
  }

  final String id;
  String title;
  final DateTime createdAt;
  DateTime updatedAt;
  final List<BookPage> _pages = [];

  List<BookPage> get pages => List.unmodifiable(_pages);
  int get pageCount => _pages.length;
  bool get isEmpty => _pages.isEmpty;

  void _changed() {
    updatedAt = DateTime.now();
    notifyListeners();
  }

  /// Appends an extracted page as the next page of the book.
  BookPage addPage(OcrResult result, String imagePath) {
    final page = BookPage(number: _pages.length + 1, result: result, imagePath: imagePath);
    _pages.add(page);
    _changed();
    return page;
  }

  /// Removes a page and renumbers the following ones.
  void removePage(BookPage page) {
    _pages.remove(page);
    for (var i = 0; i < _pages.length; i++) {
      _pages[i].number = i + 1;
    }
    _changed();
  }

  /// Removes every page.
  void clear() {
    _pages.clear();
    _changed();
  }

  void rename(String newTitle) {
    title = newTitle.trim().isEmpty ? title : newTitle.trim();
    _changed();
  }

  /// The whole book as one document:
  ///
  ///     === Page 1 ===
  ///     [text of page 1]
  ///
  ///     === Page 2 ===
  ///     [text of page 2]
  String get masterText => _pages.map((p) => '${p.header}\n${p.text}').join('\n\n');

  /// Number of words (cells) read with confidence below [threshold] (0-100).
  int lowConfidenceCount(double threshold) => _pages
      .expand((p) => p.result.cells)
      .where((c) => c.isRead && c.text.isNotEmpty && c.confidence < threshold)
      .length;

  Map<String, dynamic> toJson() => {
        'id': id,
        'title': title,
        'created_at': createdAt.toIso8601String(),
        'updated_at': updatedAt.toIso8601String(),
        'pages': [for (final p in _pages) p.toJson()],
      };
}

/// All books of one user, saved as JSON files in the app's documents folder
/// (`books/<user id>/<book id>.json`).
class BookLibrary extends ChangeNotifier {
  BookLibrary._(this._dir, this._prefs, this.userId);

  /// Loads the user's books; creates a first book if there is none.
  static Future<BookLibrary> open(String userId) async {
    final docs = await getApplicationDocumentsDirectory();
    final dir = Directory('${docs.path}/books/$userId');
    await dir.create(recursive: true);
    final library = BookLibrary._(dir, await SharedPreferences.getInstance(), userId);
    await library._load();
    return library;
  }

  final Directory _dir;
  final SharedPreferences _prefs;
  final String userId;
  final List<BookSession> _books = [];
  BookSession? _active;
  final Map<String, Timer> _pendingSaves = {};

  String get _kActive => 'active_book_$userId';

  /// Most recently changed first.
  List<BookSession> get books =>
      List.unmodifiable(_books..sort((a, b) => b.updatedAt.compareTo(a.updatedAt)));

  /// The book that new pages are added to.
  BookSession get active => _active!;

  Future<void> _load() async {
    await for (final f in _dir.list()) {
      if (f is! File || !f.path.endsWith('.json')) continue;
      try {
        _track(BookSession.fromJson(jsonDecode(await f.readAsString()) as Map<String, dynamic>));
      } catch (e) {
        debugPrint('Skipping unreadable book ${f.path}: $e');
      }
    }
    if (_books.isEmpty) {
      await create('My First Book');
      return;
    }
    final activeId = _prefs.getString(_kActive);
    _active = _books.firstWhere((b) => b.id == activeId, orElse: () => books.first);
  }

  void _track(BookSession book) {
    _books.add(book);
    book.addListener(() {
      _scheduleSave(book);
      notifyListeners();
    });
  }

  File _file(BookSession book) => File('${_dir.path}/${book.id}.json');

  /// Saves shortly after the last change (several quick changes = one write).
  void _scheduleSave(BookSession book) {
    _pendingSaves[book.id]?.cancel();
    _pendingSaves[book.id] = Timer(const Duration(milliseconds: 300), () => _save(book));
  }

  Future<void> _save(BookSession book) async {
    _pendingSaves.remove(book.id)?.cancel();
    if (!_books.contains(book)) return;
    final file = _file(book);
    final tmp = File('${file.path}.tmp');
    await tmp.writeAsString(jsonEncode(book.toJson()), flush: true);
    await tmp.rename(file.path); // atomic: a crash never leaves half a book
  }

  /// Writes every pending change now (e.g. when the app goes to background).
  Future<void> flush() async {
    for (final book in _books.where((b) => _pendingSaves.containsKey(b.id)).toList()) {
      await _save(book);
    }
  }

  Future<BookSession> create(String title) async {
    final book = BookSession(title: title.trim().isEmpty ? 'Untitled Book' : title.trim());
    _track(book);
    await _save(book);
    await setActive(book);
    return book;
  }

  Future<void> setActive(BookSession book) async {
    _active = book;
    await _prefs.setString(_kActive, book.id);
    notifyListeners();
  }

  Future<void> delete(BookSession book) async {
    _pendingSaves.remove(book.id)?.cancel();
    _books.remove(book);
    final file = _file(book);
    if (await file.exists()) await file.delete();
    if (_books.isEmpty) {
      await create('My First Book');
    } else if (identical(_active, book)) {
      await setActive(books.first);
    }
    notifyListeners();
  }

  @override
  void dispose() {
    for (final t in _pendingSaves.values) {
      t.cancel();
    }
    super.dispose();
  }
}
