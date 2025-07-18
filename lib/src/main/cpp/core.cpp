#include <jni.h>
#include <string>
#include <android/bitmap.h>
#include <android/log.h>
#include <cassert>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <chrono>
#include "FaceDetection.h"
#include "Preprocessing.h"
#include <onnxruntime/core/providers/nnapi/nnapi_provider_factory.h>
#include "OptimalSessionSettingsSelector.h"

#define LOG_TAG "Ver-ID"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

template <typename F>
class FinalAction {
public:
    explicit FinalAction(F f) : f_(std::move(f)), active_(true) {}
    ~FinalAction() { if (active_) f_(); }

    void dismiss() { active_ = false; }

private:
    F f_;
    bool active_;
};

template <typename F>
FinalAction<F> finally(F f) {
    return FinalAction<F>(std::move(f));
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_appliedrec_verid3_facedetection_retinaface_FaceDetectionRetinaFace_createNativeContext(
        JNIEnv *env, jobject thiz, jstring model_file, jboolean useNnapi, jint nnapiFlags) {
    const char *modelPathCStr = env->GetStringUTFChars(model_file, nullptr);
    std::string modelPath(modelPathCStr);
    env->ReleaseStringUTFChars(model_file, modelPathCStr);
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    sessionOptions.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
    if (useNnapi) {
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_Nnapi(sessionOptions, nnapiFlags);
        if (status != nullptr) {
            const char* msg = Ort::GetApi().GetErrorMessage(status);
            Ort::GetApi().ReleaseStatus(status);
            throw std::runtime_error(std::string("NNAPI setup error: ") + msg);
        }
    }
    auto *detection = new verid::FaceDetection(modelPath, std::move(sessionOptions));
    return reinterpret_cast<jlong>(detection);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_appliedrec_verid3_facedetection_retinaface_FaceDetectionRetinaFace_destroyNativeContext(
        JNIEnv *env, jobject thiz, jlong context) {
    auto *detection = reinterpret_cast<verid::FaceDetection *>(context);
    delete detection;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_appliedrec_verid3_facedetection_retinaface_FaceDetectionRetinaFace_detectFacesInBuffer(JNIEnv *env,
                                                                                        jobject thiz,
                                                                                        jlong context,
                                                                                        jobject imageBuffer,
                                                                                        jint width,
                                                                                        jint height,
                                                                                        jint bytesPerRow,
                                                                                        jint imageFormat,
                                                                                        jint limit,
                                                                                        jobject buffer) {
    auto *detection = reinterpret_cast<verid::FaceDetection *>(context);
    if (!detection) {
        throw std::runtime_error("Invalid context");
    }
    void *in = env->GetDirectBufferAddress(imageBuffer);
    if (!in) {
        return 0;
    }
    auto* out = static_cast<float*>(env->GetDirectBufferAddress(buffer));
    if (!out) {
        return 0;
    }
    int numFaces = detection->detectFaces(in, width, height, bytesPerRow, imageFormat, limit, out);
    return numFaces;
}
extern "C"
JNIEXPORT jint JNICALL
Java_com_appliedrec_verid3_facedetection_retinaface_FaceDetectionRetinaFace_detectFaces(JNIEnv *env,
                                                                                        jobject thiz,
                                                                                        jlong context,
                                                                                        jobject image,
                                                                                        jint limit,
                                                                                        jobject buffer) {
    auto *detection = reinterpret_cast<verid::FaceDetection *>(context);
    if (!detection) {
        throw std::runtime_error("Invalid context");
    }
//    auto start = std::chrono::high_resolution_clock::now();
//    std::vector<float> input;
//    verid::preprocessBitmapOnGPU(env, image, 640, input);
////    auto input = verid::inputTensorFromAndroidBitmap(env, image, 640);
//    auto end = std::chrono::high_resolution_clock::now();
//    float ms = std::chrono::duration<float, std::milli>(end - start).count();
//    LOGI("Input prep time: %.03f ms", ms);
    auto* out = static_cast<float*>(env->GetDirectBufferAddress(buffer));
    if (!out) return 0;
    int numFaces = detection->detectFaces(env, image, limit, out);
    return numFaces;
//    jfloatArray result = env->NewFloatArray(faces.size());
//    if (result == nullptr) {
//        throw std::runtime_error("Failed to allocated result array");
//    }
//    env->SetFloatArrayRegion(result, 0, faces.size(), faces.data());
//    return result;
}
extern "C"
JNIEXPORT jobject JNICALL
Java_com_appliedrec_verid3_facedetection_retinaface_SessionConfigurationManager_calculateOptimalSessionConfiguration(
        JNIEnv *env, jobject thiz) {
    // Get model paths
    jclass cls = env->GetObjectClass(thiz);
    jmethodID getModelPathMID = env->GetMethodID(cls, "getModelPath", "(Lcom/appliedrec/verid3/facedetection/retinaface/ModelVariant;)Ljava/lang/String;");

    // Get ModelVariant enum class and values
    jclass modelVariantCls = env->FindClass("com/appliedrec/verid3/facedetection/retinaface/ModelVariant");
    jfieldID fp32Field = env->GetStaticFieldID(modelVariantCls, "FP32", "Lcom/appliedrec/verid3/facedetection/retinaface/ModelVariant;");
    jfieldID fp16Field = env->GetStaticFieldID(modelVariantCls, "FP16", "Lcom/appliedrec/verid3/facedetection/retinaface/ModelVariant;");
    jfieldID int8Field = env->GetStaticFieldID(modelVariantCls, "INT8", "Lcom/appliedrec/verid3/facedetection/retinaface/ModelVariant;");

    jobject fp32Enum = env->GetStaticObjectField(modelVariantCls, fp32Field);
    jobject fp16Enum = env->GetStaticObjectField(modelVariantCls, fp16Field);
    jobject int8Enum = env->GetStaticObjectField(modelVariantCls, int8Field);

    auto fp32PathJ = (jstring) env->CallObjectMethod(thiz, getModelPathMID, fp32Enum);
    auto fp16PathJ = (jstring) env->CallObjectMethod(thiz, getModelPathMID, fp16Enum);
    auto int8PathJ = (jstring) env->CallObjectMethod(thiz, getModelPathMID, int8Enum);

    const char *fp32Path = env->GetStringUTFChars(fp32PathJ, nullptr);
    const char *fp16Path = env->GetStringUTFChars(fp16PathJ, nullptr);
    const char *int8Path = env->GetStringUTFChars(int8PathJ, nullptr);

    auto [modelPath, useNnapi, nnapiFlags] = verid::createOptimalSessionOptions(fp32Path, fp16Path, int8Path);

    env->ReleaseStringUTFChars(fp32PathJ, fp32Path);
    env->ReleaseStringUTFChars(fp16PathJ, fp16Path);
    env->ReleaseStringUTFChars(int8PathJ, int8Path);

    // Determine ModelVariant from selected modelPath suffix
    jobject selectedVariant;
    if (modelPath.find("_FP16.onnx") != std::string::npos) {
        selectedVariant = fp16Enum;
    } else if (modelPath.find("_INT8.onnx") != std::string::npos) {
        selectedVariant = int8Enum;
    } else {
        selectedVariant = fp32Enum;
    }

    // Create SessionConfiguration(modelVariant, useNnapi, nnapiFlags)
    jclass sessionConfigCls = env->FindClass("com/appliedrec/verid3/facedetection/retinaface/SessionConfiguration");
    jmethodID ctor = env->GetMethodID(sessionConfigCls, "<init>",
                                      "(Lcom/appliedrec/verid3/facedetection/retinaface/ModelVariant;ZI)V");

    jobject config = env->NewObject(sessionConfigCls, ctor,
                                    selectedVariant,
                                    (jboolean ) useNnapi,
                                    (jint) nnapiFlags
    );

    return config;
}