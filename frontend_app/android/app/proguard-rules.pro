# R8 keep rules for the release build.
#
# Google ML Kit Document Scanner looks up its internal components at run time.
# Without these rules R8 shrinks / renames them and scanDocument() crashes with
# a NullPointerException inside com.google.android.gms.internal.mlkit_* (seen on
# a Samsung A05s with the minified release APK).
-keep class com.google.mlkit.** { *; }
-keep class com.google.android.gms.internal.mlkit_** { *; }
-keep class com.google_mlkit_document_scanner.** { *; }
-dontwarn com.google.mlkit.**
-dontwarn com.google.android.gms.internal.mlkit_**
