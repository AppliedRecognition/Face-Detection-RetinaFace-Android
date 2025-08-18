//
// Created by Jakub Dolejs on 11/07/2025.
//

#ifndef FACE_DETECTION_FACEDETECTION_H
#define FACE_DETECTION_FACEDETECTION_H

#include <string>
#include <vector>
#include <jni.h>
#include <android/bitmap.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include "Postprocessing.h"
#include "Preprocessing.h"

namespace verid {

    class FaceDetection {
    public:
        explicit FaceDetection(const std::string &modelPath, Ort::SessionOptions options);
        ~FaceDetection() = default;
        int detectFaces(std::vector<float> &input, int limit, float *buffer);
        int detectFaces(void *input, int width, int height, int bytesPerRow, int format, int limit, float *buffer);
    private:
        Ort::Env env_;
        Ort::Session session_;
        Ort::AllocatorWithDefaultOptions allocator_;

        std::vector<const char*> inputNames_;
        std::vector<const char*> outputNames_;

        Postprocessing postprocessing_;
        Preprocessing preprocessing_;
        std::vector<float> inputBuffer_;
        std::vector<float> boxes_;
        std::vector<float> scores_;
        std::vector<float> landmarks_;

        void loadModelIO();
    };

} // verid

#endif //FACE_DETECTION_FACEDETECTION_H
