package com.appliedrec.verid3.facedetection.retinaface

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.PointF
import android.graphics.RectF
import com.appliedrec.verid3.common.EulerAngle
import com.appliedrec.verid3.common.Face
import com.appliedrec.verid3.common.FaceDetection
import com.appliedrec.verid3.common.IImage
import com.appliedrec.verid3.common.serialization.toBitmap
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.math.max

/**
 * Face detection using RetinaFace model.
 *
 * @constructor
 * Create an instance with specific settings. We recommend using `FaceDetectionRetinaFace.create()`
 * factory method to take advantage of optimal configuration calibration.
 *
 * @param modelPath Absolute path to the model file.
 * @param useNnapi `true` to use NNAPI for inference.
 * @param nnapiFlags Flags for NNAPI.
 */
class FaceDetectionRetinaFace(val modelPath: String, val useNnapi: Boolean, val nnapiFlags: Int) : FaceDetection {

    companion object {
        init {
            System.loadLibrary("FaceDetectionRetinaFace")
        }
        const val MAX_FACES = 100
        const val IMAGE_SIZE = 320

        /**
         * Factory constructor for FaceDetectionRetinaFace
         *
         * The function will run a calibration pass to determine the optimal model configuration.
         * The optimal configuration is stored in the device's shared preferences.
         *
         * @param context Application context.
         * @param forceCalibrate If true, the function will always run a calibration pass.
         * Otherwise it will attempt to read previously stored configuration from the device's
         * shared preferences.
         * @return Instance of FaceDetectionRetinaFace
         */
        suspend fun create(context: Context, forceCalibrate: Boolean=false): FaceDetectionRetinaFace {
            val modelVariants = mutableMapOf<ModelVariant,String>()
            for (variant in ModelVariant.entries) {
                val modelFile: File = context.filesDir.resolve(variant.modelName)
                if (!modelFile.exists()) {
                    context.assets.open(modelFile.name).use { input ->
                        modelFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                }
                modelVariants[variant] = modelFile.absolutePath
            }
            val configurationManager = SessionConfigurationManager(context, modelVariants)
            val configuration = configurationManager.getOptimalSessionConfiguration(forceCalibrate)
            val modelPath = modelVariants[configuration.modelVariant]
            requireNotNull(modelPath)
            return FaceDetectionRetinaFace(modelPath, configuration.useNnapi, configuration.nnapiFlags)
        }
    }

    private var nativeContext: Long
    private val buffer: ByteBuffer = ByteBuffer.allocateDirect(MAX_FACES * 18 * 4).order(ByteOrder.nativeOrder())
    private val paddedBitmap = Bitmap.createBitmap(IMAGE_SIZE, IMAGE_SIZE, Bitmap.Config.ARGB_8888)
    private val paddedCanvas = Canvas(paddedBitmap)
    private val lock = ReentrantLock()
    var confidenceThreshold: Float = 0.6f

    init {
        nativeContext = createNativeContext(modelPath, useNnapi, nnapiFlags)
    }

    /**
     * Detect faces in image
     *
     * @param image Image in which to detect faces
     * @param limit Maximum number of faces to detect. Capped at 100.
     * @return Array of detected faces.
     */
    override suspend fun detectFacesInImage(image: IImage, limit: Int): Array<Face> {
        require(limit in 1..MAX_FACES) { "Limit must be between 1 and $MAX_FACES" }
        return lock.withLock {
            val scale = minOf(1.0f, IMAGE_SIZE.toFloat() / max(image.width, image.height).toFloat());
//            val scale = minOf(IMAGE_SIZE.toFloat() / image.width, IMAGE_SIZE.toFloat() / image.height)
//            val (bitmap, scale) = scaleAndPadBitmap(image.toBitmap())
            val numFaces = detectFacesInBuffer(nativeContext, image.toDirectByteBuffer(), image.width, image.height, image.bytesPerRow, image.format.ordinal, limit, buffer)
//            val numFaces = detectFaces(nativeContext, image.toBitmap(), limit, buffer)
            facesFromBuffer(numFaces, 1f / scale)
        }
    }

    /**
     * Close the instance and release its resources
     *
     * It's recommended that you call this function
     * to free up resources when you no longer need an instance of FaceDetectionRetinaFace.
     */
    fun close() {
        lock.withLock {
            destroyNativeContext(nativeContext)
            paddedBitmap.recycle()
        }
    }

    private fun scaleAndPadBitmap(original: Bitmap): Pair<Bitmap, Float> {
        val srcWidth = original.width
        val srcHeight = original.height
        val scale = minOf(IMAGE_SIZE.toFloat() / srcWidth, IMAGE_SIZE.toFloat() / srcHeight)
        val newWidth = (srcWidth * scale).toInt()
        val newHeight = (srcHeight * scale).toInt()
        val scaledBitmap = Bitmap.createScaledBitmap(original, newWidth, newHeight, true)
        paddedCanvas.drawColor(Color.BLACK)
        paddedCanvas.drawBitmap(scaledBitmap, 0f, 0f, null)
        return paddedBitmap to scale
    }

    private fun facesFromBuffer(count: Int, scale: Float): Array<Face> {
        buffer.rewind()
        val floatBufer = buffer.asFloatBuffer()
        val faces = mutableListOf<Face>()
        for (i in 0..<count) {
            val index = i * 18
            val x = floatBufer[index] * scale
            val y = floatBufer[index+1] * scale
            val width = floatBufer[index+2] * scale
            val height = floatBufer[index+3] * scale
            val yaw = floatBufer[index+4]
            val pitch = floatBufer[index+5]
            val roll = floatBufer[index+6]
            val leftEyeX = floatBufer[index+7] * scale
            val leftEyeY = floatBufer[index+8] * scale
            val rightEyeX = floatBufer[index+9] * scale
            val rightEyeY = floatBufer[index+10] * scale
            val noseX = floatBufer[index+11] * scale
            val noseY = floatBufer[index+12] * scale
            val leftMouthX = floatBufer[index+13] * scale
            val leftMouthY = floatBufer[index+14] * scale
            val rightMouthX = floatBufer[index+15] * scale
            val rightMouthY = floatBufer[index+16] * scale
            val confidence = floatBufer[index+17]
            if (confidence < confidenceThreshold) continue
            faces.add(Face(
                bounds = RectF(x, y, x+width, y+height),
                angle = EulerAngle(yaw, pitch, roll),
                quality = confidence,
                landmarks = arrayOf(
                    PointF(leftEyeX, leftEyeY),
                    PointF(rightEyeX, rightEyeY),
                    PointF(noseX, noseY),
                    PointF(leftMouthX, leftMouthY),
                    PointF(rightMouthX, rightMouthY)
                ),
                leftEye = PointF(leftEyeX, leftEyeY),
                rightEye = PointF(rightEyeX, rightEyeY),
                noseTip = PointF(noseX, noseY),
                mouthLeftCorner = PointF(leftMouthX, leftMouthY),
                mouthRightCorner = PointF(rightMouthX, rightMouthY)
            ))
        }
        return faces.toTypedArray()
    }

    private external fun createNativeContext(modelPath: String, useNnapi: Boolean, nnapiFlags: Int): Long

    private external fun destroyNativeContext(context: Long)

    private external fun detectFaces(context: Long, image: Bitmap, limit: Int, buffer: ByteBuffer): Int

    private external fun detectFacesInBuffer(context: Long, imageBuffer: ByteBuffer, width:Int, height: Int, bytesPerRow:Int, imageFormat:Int, limit: Int, buffer: ByteBuffer): Int
}

private fun IImage.toDirectByteBuffer(): ByteBuffer {
    val buffer = ByteBuffer.allocateDirect(data.size)
        .order(ByteOrder.nativeOrder())
    buffer.put(data)
    buffer.rewind()  // reset position to 0 for JNI reading
    return buffer
}