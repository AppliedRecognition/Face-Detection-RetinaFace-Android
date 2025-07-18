package com.appliedrec.verid3.facedetection.retinaface

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class SessionConfigurationManager(context: Context, val modelPaths: Map<ModelVariant,String>) {

    private val prefs = context.getSharedPreferences("SessionConfiguration", Context.MODE_PRIVATE)

    suspend fun getOptimalSessionConfiguration(forceCalibrate: Boolean=false): SessionConfiguration {
        val allPrefs = prefs.all
        if (!forceCalibrate
            && allPrefs.containsKey(PreferenceKeys.MODEL_VARIANT)
            && allPrefs.containsKey(PreferenceKeys.USE_NNAPI)
            && allPrefs.containsKey(PreferenceKeys.NNAPI_FLAGS)
        ) {
            return SessionConfiguration(
                ModelVariant.valueOf(allPrefs[PreferenceKeys.MODEL_VARIANT] as String),
                allPrefs[PreferenceKeys.USE_NNAPI] as Boolean,
                allPrefs[PreferenceKeys.NNAPI_FLAGS] as Int
            )
        }
        val config = withContext(Dispatchers.Default) {
            calculateOptimalSessionConfiguration()
        }
        prefs.edit()
            .putString(PreferenceKeys.MODEL_VARIANT, config.modelVariant.name)
            .putBoolean(PreferenceKeys.USE_NNAPI, config.useNnapi)
            .putInt(PreferenceKeys.NNAPI_FLAGS, config.nnapiFlags)
            .commit()
        return config
    }

    fun reset() {
        prefs.edit()
            .remove(PreferenceKeys.MODEL_VARIANT)
            .remove(PreferenceKeys.USE_NNAPI)
            .remove(PreferenceKeys.NNAPI_FLAGS)
            .commit()
    }

    private fun getModelPath(variant: ModelVariant): String = modelPaths[variant] ?: ""

    private external fun calculateOptimalSessionConfiguration(): SessionConfiguration
}

private object PreferenceKeys {
    const val MODEL_VARIANT: String = "modelVariant"
    const val USE_NNAPI: String = "useNnapi"
    const val NNAPI_FLAGS: String = "nnapiFlags"
}