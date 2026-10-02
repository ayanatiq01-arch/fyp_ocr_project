// theme.dart
//
// HarfScan look: deep royal dark blue with metallic gold accents.

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class HarfColors {
  const HarfColors._();

  static const Color navy = Color(0xFF0F2027); // deep royal dark blue
  static const Color slate = Color(0xFF203A43); // lighter blue (cards, bars)
  static const Color gold = Color(0xFFD4AF37); // metallic gold
  static const Color brightGold = Color(0xFFFFD700); // highlights
  static const Color ink = Color(0xFFF5F1E6); // warm off-white text

  /// Background gradient used behind every screen.
  static const LinearGradient background = LinearGradient(
    begin: Alignment.topCenter,
    end: Alignment.bottomCenter,
    colors: [navy, slate],
  );

  /// Gold sheen for logo, headings and primary buttons.
  static const LinearGradient goldSheen = LinearGradient(
    colors: [gold, brightGold, gold],
  );

  /// Background behind words read with low confidence (< 70 %).
  static const Color lowConfidence = Color(0x66FFD700);
}

ThemeData buildHarfTheme() {
  final base = ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    colorScheme: const ColorScheme.dark(
      primary: HarfColors.gold,
      onPrimary: HarfColors.navy,
      secondary: HarfColors.brightGold,
      onSecondary: HarfColors.navy,
      surface: HarfColors.slate,
      onSurface: HarfColors.ink,
    ),
    scaffoldBackgroundColor: HarfColors.navy,
  );
  final text = GoogleFonts.poppinsTextTheme(base.textTheme)
      .apply(bodyColor: HarfColors.ink, displayColor: HarfColors.ink);
  return base.copyWith(
    textTheme: text,
    appBarTheme: AppBarTheme(
      backgroundColor: Colors.transparent,
      elevation: 0,
      foregroundColor: HarfColors.gold,
      titleTextStyle: GoogleFonts.poppins(
          fontSize: 20, fontWeight: FontWeight.w600, color: HarfColors.gold),
    ),
    cardTheme: CardThemeData(
      color: HarfColors.slate.withValues(alpha: 0.85),
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(18),
        side: BorderSide(color: HarfColors.gold.withValues(alpha: 0.35)),
      ),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        backgroundColor: HarfColors.gold,
        foregroundColor: HarfColors.navy,
        textStyle: GoogleFonts.poppins(fontWeight: FontWeight.w600),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        foregroundColor: HarfColors.gold,
        side: const BorderSide(color: HarfColors.gold),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      ),
    ),
    floatingActionButtonTheme: const FloatingActionButtonThemeData(
      backgroundColor: HarfColors.gold,
      foregroundColor: HarfColors.navy,
    ),
    snackBarTheme: const SnackBarThemeData(
      backgroundColor: HarfColors.slate,
      contentTextStyle: TextStyle(color: HarfColors.ink),
      behavior: SnackBarBehavior.floating,
    ),
  );
}

/// Reading font for recognised text: Nastaliq for Urdu (as printed in the
/// books), Naskh for Arabic.
TextStyle scriptStyle(String language, {double size = 20, Color? color}) {
  final c = color ?? HarfColors.ink;
  return language == 'arabic'
      ? GoogleFonts.notoNaskhArabic(fontSize: size, height: 1.8, color: c)
      : GoogleFonts.notoNastaliqUrdu(fontSize: size * 0.9, height: 2.2, color: c);
}

/// Full-screen gradient background used by every screen.
class HarfBackground extends StatelessWidget {
  const HarfBackground({super.key, required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) => DecoratedBox(
        decoration: const BoxDecoration(gradient: HarfColors.background),
        child: child,
      );
}
