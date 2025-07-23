package com.appliedrec.verid3.facedetection.retinaface

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.RectF
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.appliedrec.verid3.common.EulerAngle
import com.appliedrec.verid3.common.Face
import com.appliedrec.verid3.common.Image
import com.appliedrec.verid3.common.serialization.fromBitmap
import com.appliedrec.verid3.common.use
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert
import org.junit.Ignore
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.hypot
import kotlin.system.measureTimeMillis

/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class FaceDetectionRetinaFaceTest {

    @Test
    fun testDetectFace() = runBlocking {
        val bitmap = InstrumentationRegistry.getInstrumentation()
            .context.assets.open("image.jpg").use(BitmapFactory::decodeStream)
        val image = Image.fromBitmap(bitmap)
        val faces = FaceDetectionRetinaFace.create(
            InstrumentationRegistry.getInstrumentation().targetContext
        ).use {
            faceDetection -> faceDetection.detectFacesInImage(image, 1)
        }
        Assert.assertEquals(1, faces.size)
        val expectedFace = loadExpectedFace()
        Assert.assertTrue(compareFaces(faces[0], expectedFace, image.width.toFloat() * 0.1f))
        return@runBlocking
    }

    @Test
    @Ignore
    fun testDetectFaceWithDifferentModelVariants() = runBlocking {
        val bitmap = InstrumentationRegistry.getInstrumentation()
            .context.assets.open("image.jpg").use(BitmapFactory::decodeStream)
        val image = Image.fromBitmap(bitmap)
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val outputImage = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputImage)
        val colours = mapOf(ModelVariant.FP32 to Color.RED, ModelVariant.FP16 to Color.GREEN, ModelVariant.INT8 to Color.BLUE)
        val faces: Map<SessionConfiguration, Face?> = setOf(
            SessionConfiguration.FP32, SessionConfiguration.FP16, SessionConfiguration.INT8
        ).associateWith { config ->
            val outputFile = context.filesDir.resolve(config.modelVariant.modelName)
            context.assets.open(config.modelVariant.modelName).use { inputStream ->
                outputFile.outputStream().use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
            val faces = FaceDetectionRetinaFace(
                context, config
            ).use { faceDetection ->
                faceDetection.detectFacesInImage(image, 1)
            }
            faces.firstOrNull()
        }
        faces.forEach { (_, face) ->
            if (face != null) {
                val paint = Paint().apply {
                    color = Color.WHITE
                    style = Paint.Style.STROKE
                    strokeWidth = 8f
                }
                canvas.drawRect(face.bounds, paint)
                paint.style = Paint.Style.FILL
                face.landmarks.forEach { point ->
                    canvas.drawCircle(point.x, point.y, 8f, paint)
                }
            }
        }
        faces.forEach { (variant, face) ->
            if (face != null) {
                val paint = Paint().apply {
                    color = colours[variant.modelVariant]!!
                    style = Paint.Style.STROKE
                    strokeWidth = 8f
                    isAntiAlias = true
                    xfermode = PorterDuffXfermode(PorterDuff.Mode.MULTIPLY)
                }
                canvas.drawRect(face.bounds, paint)
                paint.style = Paint.Style.FILL
                face.landmarks.forEach { point ->
                    canvas.drawCircle(point.x, point.y, 8f, paint)
                }
            }
        }
        context.openFileOutput("output.jpg", Context.MODE_PRIVATE).use {
            outputImage.compress(Bitmap.CompressFormat.JPEG, 100, it)
        }
        return@runBlocking
    }

    @Test
    fun testFaceDetectionSpeed() = runBlocking {
        val bitmap = InstrumentationRegistry.getInstrumentation()
            .context.assets.open("image.jpg").use(BitmapFactory::decodeStream)
        val image = Image.fromBitmap(bitmap)
        FaceDetectionRetinaFace.create(
            InstrumentationRegistry.getInstrumentation().targetContext
        ).use { faceDetection ->
            val times = mutableListOf<Long>()
            for (i in 1..100) {
                val time = measureTimeMillis {
                    faceDetection.detectFacesInImage(image, 1)
                }
                Log.d("Ver-ID", "Full detection pass: %d ms".format(time))
                times.add(time)
            }
            val average = times.average()
            Log.d("Ver-ID", "Average detection time: %.02f ms".format(average))
            val median = times.median()
            Log.d("Ver-ID", "Median detection time: %d ms".format(median))
        }
        return@runBlocking
    }

    @Test
    fun testFaceDetectionSpeedWithVariousSettings() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val iterationCount = 5
        val bitmap = InstrumentationRegistry.getInstrumentation()
            .context.assets.open("image.jpg").use(BitmapFactory::decodeStream)
        val image = Image.fromBitmap(bitmap)
        for (setting in SessionConfiguration.all) {
            try {
                FaceDetectionRetinaFace(context, setting).use { faceDetection ->
                    val times = mutableListOf<Long>()
                    for (i in 0..<iterationCount) {
                        val time = measureTimeMillis {
                            faceDetection.detectFacesInImage(image, 1)
                        }
                        times.add(time)
                    }
                    val average = times.average()
                    Log.d(
                        "Ver-ID",
                        "Average detection time for %s: %.0f ms".format(setting.toString(), average)
                    )
                    val median = times.median()
                    Log.d(
                        "Ver-ID",
                        "Median detection time for %s: %d ms".format(setting.toString(), median)
                    )
                }
            } catch (e: Exception) {
                Log.e("Ver-ID", "Detection cannot run with configuration: %s".format(setting.toString()))
            }
        }
    }

    private fun compareFaces(face1: Face, face2: Face, maxPointDistance: Float = 10f): Boolean {
        return face1.landmarks.zip(face2.landmarks).map { (p1, p2) ->
            p1.distanceTo(p2)
        }.max() <= maxPointDistance
    }

    private fun loadExpectedFace(): Face {
        return InstrumentationRegistry.getInstrumentation().context.assets.open("face.json").use {
            val json = it.readAllBytes().toString(Charsets.UTF_8)
            val face = JSONObject(json)
            val bounds = face.getJSONObject("bounds")
            val angle = face.getJSONObject("angle")
            val landmarksJsonArray = face.getJSONArray("landmarks")
            val landmarks = Array(landmarksJsonArray.length()) { i ->
                landmarksJsonArray.getDouble(i).toFloat()
            }.toList().chunked(2).map { (x, y) -> PointF(x, y) }.toTypedArray()
            Face(
                bounds = RectF(
                    bounds.getDouble("x").toFloat(),
                    bounds.getDouble("y").toFloat(),
                    bounds.getDouble("x").toFloat() + bounds.getDouble("width").toFloat(),
                    bounds.getDouble("y").toFloat() + bounds.getDouble("height").toFloat()
                ),
                angle = EulerAngle(
                    angle.getDouble("yaw").toFloat(),
                    angle.getDouble("pitch").toFloat(),
                    angle.getDouble("roll").toFloat()
                ),
                quality = face.getDouble("quality").toFloat(),
                landmarks = landmarks,
                leftEye = landmarks[0],
                rightEye = landmarks[1],
                noseTip = landmarks[2],
                mouthLeftCorner = landmarks[3],
                mouthRightCorner = landmarks[4]
            )
        }
    }
}

fun PointF.distanceTo(other: PointF): Float {
    return hypot(other.x - this.x, other.y - this.y)
}

fun MutableList<Long>.median(): Long {
    if (isEmpty()) throw NoSuchElementException("Cannot compute median of empty array")
    val sorted = sorted()
    val mid = size / 2
    return if (size % 2 == 0) {
        ((sorted[mid - 1] + sorted[mid]) / 2.0).toLong()
    } else {
        sorted[mid]
    }
}