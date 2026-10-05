// theme.dart
//
// HarfScan look: deep royal dark blue with metallic gold accents.

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

/// False in widget tests, where fonts cannot be downloaded: the default
/// font is used instead of Google Fonts.
bool harfUseGoogleFonts = true;

/// [GoogleFonts] style, or a plain [TextStyle] when [harfUseGoogleFonts] is off.
TextStyle harfFont(TextStyle Function({double? fontSize, FontWeight? fontWeight, Color? color,
            double? height, double? letterSpacing})
        font,
    {double? fontSize, FontWeight? fontWeight, Color? color, double? height, double? letterSpacing}) {
  final style = TextStyle(
      fontSize: fontSize, fontWeight: fontWeight, color: color, height: height, letterSpacing: letterSpacing);
  if (!harfUseGoogleFonts) return style;
  return font(
      fontSize: fontSize, fontWeight: fontWeight, color: color, height: height, letterSpacing: letterSpacing);
}

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
  final text = (harfUseGoogleFonts ? GoogleFonts.poppinsTextTheme(base.textTheme) : base.textTheme)
      .apply(bodyColor: HarfColors.ink, displayColor: HarfColors.ink);
  return base.copyWith(
    textTheme: text,
    appBarTheme: AppBarTheme(
      backgroundColor: Colors.transparent,
      elevation: 0,
      foregroundColor: HarfColors.gold,
      titleTextStyle: harfFont(GoogleFonts.poppins,
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
        textStyle: harfFont(GoogleFonts.poppins, fontWeight: FontWeight.w600),
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
      ? harfFont(GoogleFonts.notoNaskhArabic, fontSize: size, height: 1.8, color: c)
      : harfFont(GoogleFonts.notoNastaliqUrdu, fontSize: size * 0.9, height: 2.2, color: c);
}

/// Gold "ح" (Harf) medallion with the HarfScan wordmark.
class HarfLogo extends StatelessWidget {
  const HarfLogo({super.key, this.size = 92, this.showTagline = true, this.showName = true});

  final double size;
  final bool showTagline;
  final bool showName;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: size,
          height: size,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            gradient: HarfColors.goldSheen,
            boxShadow: [BoxShadow(color: HarfColors.gold.withValues(alpha: 0.35), blurRadius: 24)],
          ),
          alignment: Alignment.center,
          child: Text('ح',
              style: harfFont(GoogleFonts.notoNaskhArabic,
                  fontSize: size * 0.5, fontWeight: FontWeight.w700, color: HarfColors.navy, height: 1.2)),
        ),
        if (showName) ...[
        SizedBox(height: size * 0.15),
        ShaderMask(
          shaderCallback: HarfColors.goldSheen.createShader,
          child: Text('HarfScan',
              style: harfFont(GoogleFonts.cinzel,
                  fontSize: size * 0.39, fontWeight: FontWeight.w700, color: Colors.white, letterSpacing: 2)),
        ),
        ],
        if (showTagline) ...[
          const SizedBox(height: 4),
          Text('Urdu & Arabic OCR for historical books',
              style: Theme.of(context)
                  .textTheme
                  .bodyMedium
                  ?.copyWith(color: HarfColors.ink.withValues(alpha: 0.75))),
        ],
      ],
    );
  }
}

/// Small gold section caption ("ADD A PAGE").
class SectionLabel extends StatelessWidget {
  const SectionLabel(this.text, {super.key});

  final String text;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(left: 4, bottom: 10),
        child: Text(text.toUpperCase(),
            style: const TextStyle(
                color: HarfColors.gold, fontSize: 12, letterSpacing: 1.6, fontWeight: FontWeight.w600)),
      );
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
