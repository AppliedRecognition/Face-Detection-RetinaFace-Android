//
// Created by Jakub Dolejs on 14/07/2025.
//

#ifndef FACE_DETECTION_PREPROCESSING_H
#define FACE_DETECTION_PREPROCESSING_H

#include <vector>
#include <jni.h>

namespace verid {

    std::vector<float> inputTensorFromAndroidBitmap(JNIEnv *env, jobject bitmap, int targetSize);

} // verid

#endif //FACE_DETECTION_PREPROCESSING_H
