//
// Created by Jakub Dolejs on 11/07/2025.
//

#include "FaceDetection.h"
#include "Postprocessing.h"
#include "OptimalSessionSettingsSelector.h"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/nnapi/nnapi_provider_factory.h>
#include <android/log.h>
#include <chrono>
#include <vector>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <sstream>
#include "Logger.h"

constexpr int IMAGE_SIZE = 320;

namespace verid {

    FaceDetection::FaceDetection(const std::string &modelPath, Ort::SessionOptions options)
            : env_(ORT_LOGGING_LEVEL_WARNING, LOG_TAG),
              session_(env_, modelPath.c_str(), options),
              postprocessing_(IMAGE_SIZE, IMAGE_SIZE),
              preprocessing_(IMAGE_SIZE)
    {
        loadModelIO();
    }

    void FaceDetection::loadModelIO() {
        size_t inputCount = session_.GetInputCount();
        size_t outputCount = session_.GetOutputCount();
        inputNames_.clear();
        outputNames_.clear();
        for (size_t i = 0; i < inputCount; ++i) {
            Ort::AllocatedStringPtr name = session_.GetInputNameAllocated(i, allocator_);
            inputNames_.push_back(strdup(name.get()));  // strdup to persist
        }
        for (size_t i = 0; i < outputCount; ++i) {
            Ort::AllocatedStringPtr name = session_.GetOutputNameAllocated(i, allocator_);
            outputNames_.push_back(strdup(name.get()));  // strdup to persist
        }
    }

    static std::vector<float> toFloatVector(const Ort::Value& output) {
        const auto* dataPtr = output.GetTensorData<float>();
        auto typeInfo = output.GetTensorTypeAndShapeInfo();
        auto shape = typeInfo.GetShape();
        size_t totalElements = 1;
        for (auto dim : shape) {
            totalElements *= dim;
        }
        return {dataPtr, dataPtr + totalElements};
    }

    int FaceDetection::detectFaces(void *imageData, int width, int height, int bytesPerRow, int format, int limit, float *buffer) {
        std::vector<float> input;
        preprocessing_.preprocessBitmap(imageData, width, height, bytesPerRow, format, input);
        return detectFaces(input, limit, buffer);
    }

    int FaceDetection::detectFaces(std::vector<float> &input, const int limit, float *buffer) {
        std::vector<int64_t> inputShape = {1, 3, IMAGE_SIZE, IMAGE_SIZE};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo,
                input.data(),
                input.size(),
                inputShape.data(),
                inputShape.size()
        );
        std::vector<Ort::Value> inputTensors;
        inputTensors.emplace_back(std::move(inputTensor));
        std::vector<Ort::Value> outputTensors;
        outputTensors.resize(outputNames_.size());
        // Run inference
        session_.Run(
                Ort::RunOptions{nullptr},
                inputNames_.data(),
                inputTensors.data(),
                inputTensors.size(),
                outputNames_.data(),
                outputTensors.data(),
                outputTensors.size()
        );
        std::unordered_map<std::string, Ort::Value> outputMap;
        for (size_t i = 0; i < outputNames_.size(); ++i) {
            outputMap[outputNames_[i]] = std::move(outputTensors[i]);
        }
        std::vector<float> boxes = toFloatVector(outputMap["boxes"]);
        std::vector<float> scores = toFloatVector(outputMap["scores"]);
        std::vector<float> landmarks = toFloatVector(outputMap["landmarks"]);
        // Decode boxes
        std::vector<DetectionBox> detections = postprocessing_.decode(boxes, scores, landmarks);
        // NMS
        detections = verid::Postprocessing::nonMaxSuppression(detections, 0.4f, limit);
        int numFaces = std::min(static_cast<int>(detections.size()), limit);
        // Fill the face buffer
        for (int i = 0; i < numFaces; ++i) {
            const auto& det = detections[i];
            buffer[0] = det.bounds.x;
            buffer[1] = det.bounds.y;
            buffer[2] = det.bounds.width;
            buffer[3] = det.bounds.height;
            buffer[4] = det.angle.yaw;
            buffer[5] = det.angle.pitch;
            buffer[6] = det.angle.roll;
            buffer[7] = det.landmarks[0].x;
            buffer[8] = det.landmarks[0].y;
            buffer[9] = det.landmarks[1].x;
            buffer[10] = det.landmarks[1].y;
            buffer[11] = det.landmarks[2].x;
            buffer[12] = det.landmarks[2].y;
            buffer[13] = det.landmarks[3].x;
            buffer[14] = det.landmarks[3].y;
            buffer[15] = det.landmarks[4].x;
            buffer[16] = det.landmarks[4].y;
            buffer[17] = det.quality;
            buffer += 18;  // advance pointer by one face block
        }
        return numFaces;
    }
} // verid