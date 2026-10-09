package com.fyp.frontend_app

import android.content.ContentValues
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File

/**
 * Saves exported books (PDF / Word) into the phone's Downloads folder
 * (Download/HarfScan), where Files, WhatsApp, Drive ... can open them.
 * Android 10+: MediaStore (no permission needed); older: the public folder.
 */
class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "harfscan/downloads")
            .setMethodCallHandler { call, result ->
                if (call.method != "save") {
                    result.notImplemented()
                    return@setMethodCallHandler
                }
                try {
                    val path = call.argument<String>("path")!!
                    val name = call.argument<String>("name")!!
                    val mime = call.argument<String>("mime")!!
                    result.success(saveToDownloads(File(path), name, mime))
                } catch (e: Exception) {
                    result.error("SAVE_FAILED", e.toString(), null)
                }
            }
    }

    private fun saveToDownloads(src: File, name: String, mime: String): String {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val resolver = contentResolver
            val values = ContentValues().apply {
                put(MediaStore.Downloads.DISPLAY_NAME, name)
                put(MediaStore.Downloads.MIME_TYPE, mime)
                put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/HarfScan")
                put(MediaStore.Downloads.IS_PENDING, 1)
            }
            val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                ?: throw IllegalStateException("Downloads folder not available")
            resolver.openOutputStream(uri).use { out ->
                if (out == null) throw IllegalStateException("Cannot write to Downloads")
                src.inputStream().use { it.copyTo(out) }
            }
            values.clear()
            values.put(MediaStore.Downloads.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
            return "Download/HarfScan/$name"
        }
        @Suppress("DEPRECATION")
        val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "HarfScan")
        dir.mkdirs()
        val dst = File(dir, name)
        src.copyTo(dst, overwrite = true)
        return dst.absolutePath
    }
}
