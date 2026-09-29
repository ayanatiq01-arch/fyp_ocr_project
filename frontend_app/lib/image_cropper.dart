// image_cropper.dart
//
// Interactive cropping step. After the user takes a photo of a book page they
// drag the crop handles around the paragraph(s) they want digitised; only
// that region is uploaded, which keeps OCR fast and accurate.
//
// Wraps the `image_cropper` package (uCrop on Android, TOCropViewController
// on iOS). Android additionally needs the UCropActivity entry in
// AndroidManifest.xml - see frontend_app/README.md.

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_cropper/image_cropper.dart';

/// Opens the native cropping screen for a picked page photo.
///
/// Named `ParagraphCropper` (not `ImageCropper`) to avoid clashing with the
/// class exported by the `image_cropper` package.
class ParagraphCropper {
  const ParagraphCropper._();

  /// Lets the user select a region of [sourcePath].
  ///
  /// Returns the cropped image file, or `null` if the user cancelled.
  /// Free-form aspect ratio is the default because paragraphs have
  /// arbitrary shapes; presets are offered for convenience.
  static Future<File?> crop(BuildContext context, String sourcePath) async {
    final theme = Theme.of(context).colorScheme;

    final CroppedFile? cropped = await ImageCropper().cropImage(
      sourcePath: sourcePath,
      // PNG is lossless: JPEG artefacts around thin Nastaliq strokes and
      // dots measurably hurt recognition.
      compressFormat: ImageCompressFormat.png,
      compressQuality: 100,
      uiSettings: [
        AndroidUiSettings(
          toolbarTitle: 'Select paragraph',
          toolbarColor: theme.primary,
          toolbarWidgetColor: theme.onPrimary,
          activeControlsWidgetColor: theme.primary,
          initAspectRatio: CropAspectRatioPreset.original,
          lockAspectRatio: false,
          hideBottomControls: false,
          aspectRatioPresets: const [
            CropAspectRatioPreset.original,
            CropAspectRatioPreset.ratio4x3,
            CropAspectRatioPreset.ratio16x9,
          ],
        ),
        IOSUiSettings(
          title: 'Select paragraph',
          aspectRatioLockEnabled: false,
          resetAspectRatioEnabled: true,
          aspectRatioPresets: const [
            CropAspectRatioPreset.original,
            CropAspectRatioPreset.ratio4x3,
            CropAspectRatioPreset.ratio16x9,
          ],
        ),
      ],
    );

    if (cropped == null) return null;
    return File(cropped.path);
  }
}
