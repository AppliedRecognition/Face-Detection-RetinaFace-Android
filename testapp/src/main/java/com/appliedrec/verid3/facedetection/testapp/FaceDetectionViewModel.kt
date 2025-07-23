package com.appliedrec.verid3.facedetection.testapp

import android.app.Application
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.collection.mutableLongListOf
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.appliedrec.verid3.common.Face
import com.appliedrec.verid3.common.Image
import com.appliedrec.verid3.common.serialization.fromBitmap
import com.appliedrec.verid3.facedetection.retinaface.FaceDetectionRetinaFace
import com.appliedrec.verid3.facedetection.retinaface.toFlags
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlin.system.measureTimeMillis

class FaceDetectionViewModel(application: Application) : AndroidViewModel(application) {

    private val _annotatedBitmap = MutableStateFlow<Bitmap?>(null)
    private val _isLoaded = MutableStateFlow(false)
    private val _minDetectionSpeedMs = MutableStateFlow<Long?>(null)
    private val _maxDetectionSpeedMs = MutableStateFlow<Long?>(null)
    private val _medianDetectionSpeedMs = MutableStateFlow<Long?>(null)
    private val _detectionModelPath = MutableStateFlow<String?>(null)
    private val _detectionUseNnapi = MutableStateFlow<Boolean?>(null)
    private val _detectionNnapiFlags = MutableStateFlow<Int?>(null)
    val annotatedBitmap: StateFlow<Bitmap?> = _annotatedBitmap.asStateFlow()
    val isLoaded: StateFlow<Boolean> = _isLoaded.asStateFlow()
    val minDetectionSpeedMs: StateFlow<Long?> = _minDetectionSpeedMs.asStateFlow()
    val maxDetectionSpeedMs: StateFlow<Long?> = _maxDetectionSpeedMs.asStateFlow()
    val medianDetectionSpeedMs: StateFlow<Long?> = _medianDetectionSpeedMs.asStateFlow()
    val detectionModelPath: StateFlow<String?> = _detectionModelPath.asStateFlow()
    val detectionUseNnapi: StateFlow<Boolean?> = _detectionUseNnapi.asStateFlow()
    val detectionNnapiFlags: StateFlow<Int?> = _detectionNnapiFlags.asStateFlow()

    private var faceDetector: FaceDetectionRetinaFace? = null

    init {
        viewModelScope.launch {
            faceDetector = FaceDetectionRetinaFace.create(application)
            _detectionModelPath.value = faceDetector?.configuration?.modelVariant?.modelName
            _detectionUseNnapi.value = faceDetector?.configuration?.useNnapi
            _detectionNnapiFlags.value = faceDetector?.configuration?.nnapiOptions?.toFlags()
            _isLoaded.value = true
        }
    }

    fun setBitmap(bitmap: Bitmap) {
        viewModelScope.launch {
            try {
                faceDetector?.let { faceDetection ->
                    _minDetectionSpeedMs.value = null
                    _maxDetectionSpeedMs.value = null
                    _medianDetectionSpeedMs.value = null
                    val safeBitmap = if (bitmap.config == Bitmap.Config.HARDWARE) {
                        bitmap.copy(Bitmap.Config.ARGB_8888, false)
                    } else {
                        bitmap
                    }
                    val image = Image.fromBitmap(safeBitmap)
                    val faces: Array<Face> =
                        faceDetection.detectFacesInImage(image, limit = 1)
                    if (faces.isNotEmpty()) {
                        val annotated = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                        val canvas = Canvas(annotated)
                        val strokeWidth = bitmap.width.toFloat() / 100f
                        val rectPaint = Paint().apply {
                            color = Color.GREEN
                            style = Paint.Style.STROKE
                            this.strokeWidth = strokeWidth
                            isAntiAlias = true
                        }
                        val landmarkPaint = Paint().apply {
                            color = Color.GREEN
                            style = Paint.Style.FILL
                            isAntiAlias = true
                        }
                        faces.forEach { face ->
                            canvas.drawRect(face.bounds, rectPaint)
                            face.landmarks.forEach { landmark ->
                                canvas.drawCircle(landmark.x, landmark.y, strokeWidth, landmarkPaint)
                            }
                        }
                        _annotatedBitmap.value = annotated
                    } else {
                        _annotatedBitmap.value = bitmap
                    }
                    val detectionSpeeds = mutableListOf<Long>()
                    for (i in 0 until 10) {
                        val time = measureTimeMillis {
                            faceDetection.detectFacesInImage(image, limit = 1)
                        }
                        detectionSpeeds.add(time)
                    }
                    _minDetectionSpeedMs.value = detectionSpeeds.min()
                    _maxDetectionSpeedMs.value = detectionSpeeds.max()
                    _medianDetectionSpeedMs.value = detectionSpeeds.median()
                }
            } catch (e: Exception) {
                e.printStackTrace()
                _annotatedBitmap.value = null
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        viewModelScope.launch {
            faceDetector?.close()
        }
    }
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