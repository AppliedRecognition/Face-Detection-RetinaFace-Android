package com.appliedrec.verid3.facedetection.retinaface

import android.content.Context
import android.graphics.PointF
import android.graphics.RectF
import com.appliedrec.verid3.common.EulerAngle
import com.appliedrec.verid3.common.Face
import com.appliedrec.verid3.common.FaceDetection
import com.appliedrec.verid3.common.IImage
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.future.future
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CompletableFuture
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.jvm.Throws
import kotlin.math.max

/**
 * Face detection using RetinaFace model.
 *
 * @constructor
 * Create an instance with specific settings. We recommend using [FaceDetectionRetinaFace.create]
 * (Kotlin) or [FaceDetectionRetinaFace.createAsync] (Java) factory methods to take advantage of
 * optimal configuration calibration.
 *
 * @param context Application context.
 * @param modelVariant Model file variant.
 * @param useNnapi `true` to use NNAPI for inference.
 * @param nnapiFlags Flags for NNAPI.
 */
@Suppress("MemberVisibilityCanBePrivate")
class FaceDetectionRetinaFace
@Throws(Exception::class)
constructor(context: Context, val configuration: SessionConfiguration) : FaceDetection {

    companion object {
        init {
            System.loadLibrary("FaceDetectionRetinaFace")
        }
        const val MAX_FACES = 100
        const val IMAGE_SIZE = 320

        /**
         * Factory constructor for FaceDetectionRetinaFace
         *
         * In Java use [FaceDetectionRetinaFace.createAsync].
         *
         * The function will run a calibration pass to determine the optimal model configuration.
         * The optimal configuration is stored in the device's shared preferences.
         *
         * @param context Application context.
         * @param forceCalibrate If `true`, the function will always run a calibration pass.
         * Otherwise it will attempt to read previously stored configuration from the device's
         * shared preferences.
         * @return Instance of FaceDetectionRetinaFace
         */
        suspend fun create(context: Context, forceCalibrate: Boolean=false): FaceDetectionRetinaFace {
            val modelVariants = mutableMapOf<ModelVariant,String>()
            val appContext = context.applicationContext
            for (variant in ModelVariant.entries) {
                val modelFile: File = appContext.filesDir.resolve(variant.modelName)
                if (!modelFile.exists()) {
                    withContext(Dispatchers.IO) {
                        appContext.assets.open(modelFile.name).use { input ->
                            modelFile.outputStream().use { output ->
                                input.copyTo(output)
                            }
                        }
                    }
                }
                modelVariants[variant] = modelFile.absolutePath
            }
            val configurationManager = SessionConfigurationManager(appContext, modelVariants)
            val configuration = withContext(Dispatchers.Default) {
                configurationManager.getOptimalSessionConfiguration(forceCalibrate)
            }
            return FaceDetectionRetinaFace(context, configuration)
        }

        /**
         * Factory constructor for FaceDetectionRetinaFace
         *
         * For Java only. In Kotlin use [FaceDetectionRetinaFace.create]
         *
         * The function will run a calibration pass to determine the optimal model configuration.
         * The optimal configuration is stored in the device's shared preferences.
         *
         * @param context Application context.
         * @param forceCalibrate If `true`, the function will always run a calibration pass.
         * Otherwise it will attempt to read previously stored configuration from the device's
         * shared preferences.
         * @return Completable future that resolves to an instance of FaceDetectionRetinaFace
         */
        @JvmStatic
        @Deprecated("Java only", level = DeprecationLevel.HIDDEN)
        fun createAsync(context: Context, forceCalibrate: Boolean = false): CompletableFuture<FaceDetectionRetinaFace> {
            return CoroutineScope(Dispatchers.Default).future {
                create(context, forceCalibrate)
            }
        }
    }

    private var nativeContext: Long
    private val buffer: ByteBuffer = ByteBuffer.allocateDirect(MAX_FACES * 18 * 4)
        .order(ByteOrder.nativeOrder())
    private val lock = ReentrantLock()

    /**
     * Minimum confidence threshold for detected faces.
     */
    @Suppress("unused")
    var confidenceThreshold: Float = 0.6f

    init {
        val appContext = context.applicationContext
        val modelFile: File = appContext.filesDir.resolve(configuration.modelVariant.modelName)
        if (!modelFile.exists()) {
            appContext.assets.open(configuration.modelVariant.modelName).use { input ->
                modelFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        val modelPath = modelFile.absolutePath
        nativeContext = createNativeContext(modelPath, configuration.useNnapi, configuration.nnapiOptions.toFlags())
    }

    /**
     * Detect faces in image
     *
     * @param image [Image][IImage] in which to detect faces
     * @param limit Maximum number of faces to detect. Capped at 100.
     * @return Array of detected [faces][Face].
     */
    override suspend fun detectFacesInImage(image: IImage, limit: Int): List<Face> {
        require(limit in 1..MAX_FACES) { "Limit must be between 1 and $MAX_FACES" }
        return lock.withLock {
            val scale = minOf(1.0f, IMAGE_SIZE.toFloat() / max(image.width, image.height).toFloat())
            val numFaces = detectFacesInBuffer(nativeContext, image.toDirectByteBuffer(), image.width, image.height, image.bytesPerRow, image.format.ordinal, limit, buffer)
            facesFromBuffer(numFaces, 1f / scale)
        }
    }

    /**
     * Close the instance and release its resources
     *
     * It's recommended that you call this function
     * to free up resources when you no longer need an instance of FaceDetectionRetinaFace.
     */
    @Suppress("unused")
    override suspend fun close() {
        lock.withLock {
            destroyNativeContext(nativeContext)
        }
    }

    private fun facesFromBuffer(count: Int, scale: Float): List<Face> {
        buffer.rewind()
        val floatBuffer = buffer.asFloatBuffer()
        val faces = mutableListOf<Face>()
        for (i in 0..<count) {
            val index = i * 18
            val x = floatBuffer[index] * scale
            val y = floatBuffer[index+1] * scale
            val width = floatBuffer[index+2] * scale
            val height = floatBuffer[index+3] * scale
            val yaw = floatBuffer[index+4]
            val pitch = floatBuffer[index+5]
            val roll = floatBuffer[index+6]
            val leftEyeX = floatBuffer[index+7] * scale
            val leftEyeY = floatBuffer[index+8] * scale
            val rightEyeX = floatBuffer[index+9] * scale
            val rightEyeY = floatBuffer[index+10] * scale
            val noseX = floatBuffer[index+11] * scale
            val noseY = floatBuffer[index+12] * scale
            val leftMouthX = floatBuffer[index+13] * scale
            val leftMouthY = floatBuffer[index+14] * scale
            val rightMouthX = floatBuffer[index+15] * scale
            val rightMouthY = floatBuffer[index+16] * scale
            val confidence = floatBuffer[index+17]
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
        return faces
    }

    private external fun createNativeContext(modelPath: String, useNnapi: Boolean, nnapiFlags: Int): Long

    private external fun destroyNativeContext(context: Long)

    private external fun detectFacesInBuffer(context: Long, imageBuffer: ByteBuffer, width:Int, height: Int, bytesPerRow:Int, imageFormat:Int, limit: Int, buffer: ByteBuffer): Int
}

private fun IImage.toDirectByteBuffer(): ByteBuffer {
    val buffer = ByteBuffer.allocateDirect(data.size)
        .order(ByteOrder.nativeOrder())
    buffer.put(data)
    buffer.rewind()
    return buffer
}