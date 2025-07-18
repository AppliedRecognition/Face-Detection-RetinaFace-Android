package com.appliedrec.verid3.facedetection.retinaface

data class SessionConfiguration(
    val modelVariant: ModelVariant,
    val useNnapi: Boolean,
    val nnapiFlags: Int
)
