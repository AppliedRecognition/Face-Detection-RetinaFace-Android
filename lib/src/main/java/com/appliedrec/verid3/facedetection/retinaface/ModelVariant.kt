package com.appliedrec.verid3.facedetection.retinaface

/**
 * Represents a variant of the model used for face detection.
 *
 * @property modelName The name of the model file.
 */
enum class ModelVariant(val modelName: String) {
    /**
     * Non-quantised model using 32-bit floating point precision.
     */
    FP32("RetinaFace320_FP32.onnx"),

    /**
     * Quantised model using 16-bit floating point precision.
     */
    FP16("RetinaFace320_FP16.onnx"),

    /**
     * Quantised model using 8-bit integer precision.
     */
    INT8("RetinaFace320_INT8.onnx");

    companion object {
        fun fromModelName(modelName: String): ModelVariant? =
            entries.find { it.modelName == modelName }
    }
}