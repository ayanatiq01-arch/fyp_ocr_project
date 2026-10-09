import 'package:flutter_test/flutter_test.dart';
import 'package:frontend_app/book_session.dart';
import 'package:frontend_app/export_service.dart';

void main() {
  test('Typed file names are cleaned', () {
    expect(ExportService.safeFileName(' My Book.pdf '), 'My Book');
    expect(ExportService.safeFileName(r'a/b\c:d*e?"f<g>h|i'), 'abcdefghi');
    expect(ExportService.safeFileName('اردو کتاب'), 'اردو کتاب'); // Urdu names are kept
    expect(ExportService.safeFileName('notes.DOCX'), 'notes');
  });

  test('Suggested file name: Latin title and date', () {
    final name = ExportService.defaultFileName(BookSession(title: 'My First Book'));
    expect(name, startsWith('My_First_Book_'));
    expect(ExportService.defaultFileName(BookSession(title: 'کتاب')), startsWith('HarfScan_Book_'));
  });
}
