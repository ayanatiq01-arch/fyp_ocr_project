// book_session.dart
//
// The "Active Book Workspace": every page extracted in this session, kept in
// order, plus the user's language choice. Shared by all three screens.

import 'package:flutter/foundation.dart';

import 'api_service.dart';

/// Language the user tells the backend to expect (sent as "language").
enum OcrLanguage {
  urdu('urdu', 'Urdu'),
  arabic('arabic', 'Arabic'),
  mixed('mixed', 'Mixed (Auto)');

  const OcrLanguage(this.apiValue, this.label);

  /// Value of the backend's `language` form field.
  final String apiValue;
  final String label;
}

/// One digitised page of the book.
class BookPage {
  BookPage({required this.number, required this.result, required this.imagePath})
      : createdAt = DateTime.now();

  /// 1-based page number in the current book.
  int number;
  final OcrResult result;
  final String imagePath;
  final DateTime createdAt;

  /// Page text with paragraphs, bullets and table columns (TAB) preserved.
  String get text => result.currentText();

  /// Header line used in the master document and in exports.
  String get header => '=== Page $number ===';
}

class BookSession extends ChangeNotifier {
  final List<BookPage> _pages = [];
  OcrLanguage _language = OcrLanguage.mixed;

  List<BookPage> get pages => List.unmodifiable(_pages);
  int get pageCount => _pages.length;
  bool get isEmpty => _pages.isEmpty;

  OcrLanguage get language => _language;
  set language(OcrLanguage value) {
    if (value == _language) return;
    _language = value;
    notifyListeners();
  }

  /// Appends an extracted page as the next page of the book.
  BookPage addPage(OcrResult result, String imagePath) {
    final page = BookPage(number: _pages.length + 1, result: result, imagePath: imagePath);
    _pages.add(page);
    notifyListeners();
    return page;
  }

  /// Removes a page and renumbers the following ones.
  void removePage(BookPage page) {
    _pages.remove(page);
    for (var i = 0; i < _pages.length; i++) {
      _pages[i].number = i + 1;
    }
    notifyListeners();
  }

  /// Starts a new, empty book.
  void clear() {
    _pages.clear();
    notifyListeners();
  }

  /// The whole book as one document:
  ///
  ///     === Page 1 ===
  ///     [text of page 1]
  ///
  ///     === Page 2 ===
  ///     [text of page 2]
  String get masterText =>
      _pages.map((p) => '${p.header}\n${p.text}').join('\n\n');

  /// Number of boxes whose OCR text Gemini corrected.
  int get aiCorrectedCount =>
      _pages.expand((p) => p.result.cells).where((c) => c.isAiCorrected).length;

  /// Number of words (cells) read with confidence below [threshold] (0-100).
  int lowConfidenceCount(double threshold) => _pages
      .expand((p) => p.result.cells)
      .where((c) => c.isRead && c.text.isNotEmpty && c.confidence < threshold)
      .length;
}
