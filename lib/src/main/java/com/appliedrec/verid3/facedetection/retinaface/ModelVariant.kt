package com.appliedrec.verid3.facedetection.retinaface

enum class ModelVariant(val modelName: String) {
    FP32("RetinaFace320_FP32.onnx"),
    FP16("RetinaFace320_FP16.onnx"),
    INT8("RetinaFace320_INT8.onnx");

    companion object {
        fun fromModelName(modelName: String): ModelVariant? =
            entries.find { it.modelName == modelName }
    }
}