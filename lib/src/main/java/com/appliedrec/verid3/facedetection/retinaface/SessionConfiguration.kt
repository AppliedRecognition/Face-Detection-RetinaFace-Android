package com.appliedrec.verid3.facedetection.retinaface

/**
 * Inference session configuration options
 *
 * @property modelVariant Model variant
 * @property useNnapi `true` if NNAPI should be used
 * @property nnapiOptions Set of NNAPI options
 */
sealed class SessionConfiguration(
    val modelVariant: ModelVariant,
    val useNnapi: Boolean,
    val nnapiOptions: Set<NnapiOptions> = emptySet()
) {
    /**
     * Run inference with nonquantised model without using NNAPI
     */
    data object FP32 : SessionConfiguration(ModelVariant.FP32, false)

    /**
     * Run inference with nonquantised model using NNAPI
     */
    data object FP32_NNAPI : SessionConfiguration(ModelVariant.FP32, true)

    /**
     * Run inference with nonquantised model using NNAPI and disabling CPU inference
     */
    data object FP32_NNAPI_CPU_DISABLED : SessionConfiguration(ModelVariant.FP32, true, setOf(NnapiOptions.CPU_DISABLED))

    /**
     * Run inference with 16-bit floating point quantised model without using NNAPI
     */
    data object FP16 : SessionConfiguration(ModelVariant.FP16, false)

    /**
     * Run inference with 16-bit floating point quantised model using NNAPI
     */
    data object FP16_NNAPI : SessionConfiguration(ModelVariant.FP16, true, setOf(NnapiOptions.USE_FP16))

    /**
     * Run inference with 16-bit floating point quantised model using NNAPI and disabling CPU inference
     */
    data object FP16_NNAPI_CPU_DISABLED :SessionConfiguration(ModelVariant.FP16, true, setOf(NnapiOptions.USE_FP16, NnapiOptions.CPU_DISABLED))

    /**
     * Run inference with 8-bit integer quantised model without using NNAPI
     */
    data object INT8 : SessionConfiguration(ModelVariant.INT8, false)

    /**
     * Run inference with 8-bit integer quantised model using NNAPI
     */
    data object INT8_NNAPI : SessionConfiguration(ModelVariant.INT8, true)

    /**
     * Custom configuration, use with caution
     *
     * @property customModelVariant Model variant
     * @property customUseNnapi Use NNAPI
     * @property customNnapiOptions NNAPI options
     */
    data class Custom(
        val customModelVariant: ModelVariant,
        val customUseNnapi: Boolean,
        val customNnapiOptions: Set<NnapiOptions> = emptySet()
    ) : SessionConfiguration(
        customModelVariant,
        customUseNnapi,
        customNnapiOptions
    )

    override fun toString(): String {
        val nnapiFlagsString = nnapiOptions.map { it.name }.joinToString(", ")
        val useNnapiString = if (useNnapi) "yes" else "no"
        return "Model variant: %s, use Nnapi: %s, Nnapi flags: %s".format(
            modelVariant.modelName,
            useNnapiString,
            nnapiFlagsString
        )
    }

    companion object {
        /**
         * Set of all supported configurations
         */
        @JvmStatic
        val all = setOf(
            FP32, FP32_NNAPI, FP32_NNAPI_CPU_DISABLED,
            FP16, FP16_NNAPI, FP16_NNAPI_CPU_DISABLED,
            INT8, INT8_NNAPI
        )
    }
}