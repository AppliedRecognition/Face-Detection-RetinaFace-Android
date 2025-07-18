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

#define LOG_TAG "Ver-ID"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

constexpr int IMAGE_SIZE = 320;

namespace verid {

    FaceDetection::FaceDetection(const std::string &modelPath, Ort::SessionOptions&& options)
            : env_(ORT_LOGGING_LEVEL_WARNING, LOG_TAG),
              session_(env_, modelPath.c_str(), std::move(options)),
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

    static std::vector<uint8_t> generateTestInputBuffer(int width = 600, int height = 800) {
        const int channels = 4;
        std::vector<uint8_t> buffer(width * height * channels, 0);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = (y * width + x) * channels;

                // Top-left 32x32: red
                if (x < 32 && y < 32) {
                    buffer[idx + 0] = 255;  // R
                    buffer[idx + 1] = 0;    // G
                    buffer[idx + 2] = 0;    // B
                    buffer[idx + 3] = 255;  // A
                }
                    // Bottom-right 32x32: blue
                else if (x >= width - 32 && y >= height - 32) {
                    buffer[idx + 0] = 0;    // R
                    buffer[idx + 1] = 0;    // G
                    buffer[idx + 2] = 255;  // B
                    buffer[idx + 3] = 255;  // A
                }
                    // Everywhere else: green
                else {
                    buffer[idx + 0] = 0;    // R
                    buffer[idx + 1] = 255;  // G
                    buffer[idx + 2] = 0;    // B
                    buffer[idx + 3] = 255;  // A
                }
            }
        }

        return buffer;
    }

    static void writeToFile(std::vector<uint8_t> &rgba, int width, int height) {
        std::string fileName = "/data/user/0/com.appliedrec.verid3.facedetection.testapp/files/debug_input.ppm";
        int fd = open(fileName.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (fd >= 0) {
            std::ostringstream oss;
            oss << "P6\n" << width << " " << height << "\n255\n";
            std::string header = oss.str();
            write(fd, header.c_str(), header.size());
            for (int i = 0; i < width * height; ++i) {
                uint8_t r = rgba[i * 4 + 0];
                uint8_t g = rgba[i * 4 + 1];
                uint8_t b = rgba[i * 4 + 2];
                write(fd, &r, 1);
                write(fd, &g, 1);
                write(fd, &b, 1);
            }
            close(fd);
            LOGI("Wrote RGBA to file %s", fileName.c_str());
        } else {
            LOGI("Failed to write RGBA to file %s", fileName.c_str());
        }
    }

    int FaceDetection::detectFaces(void *imageData, int width, int height, int bytesPerRow, int format, int limit, float *buffer) {
        std::vector<float> input;
//        std::vector<uint8_t> testBuffer = generateTestInputBuffer();
//        writeToFile(testBuffer, 600, 800);
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

        auto start = std::chrono::high_resolution_clock::now();
        session_.Run(
                Ort::RunOptions{nullptr},
                inputNames_.data(),
                inputTensors.data(),
                inputTensors.size(),
                outputNames_.data(),
                outputTensors.data(),
                outputTensors.size()
        );
        auto end = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(end - start).count();
        LOGI("Inference time: %.03f ms", ms);

        std::unordered_map<std::string, Ort::Value> outputMap;
        for (size_t i = 0; i < outputNames_.size(); ++i) {
            outputMap[outputNames_[i]] = std::move(outputTensors[i]);
        }
        start = std::chrono::high_resolution_clock::now();
        std::vector<float> boxes = toFloatVector(outputMap["boxes"]);
        std::vector<float> scores = toFloatVector(outputMap["scores"]);
        std::vector<float> landmarks = toFloatVector(outputMap["landmarks"]);
        end = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<float, std::milli>(end - start).count();
        LOGI("Output tensor float conversion time: %.03f ms", ms);

        start = std::chrono::high_resolution_clock::now();
        std::vector<DetectionBox> detections = postprocessing_.decode(boxes, scores, landmarks);
        end = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<float, std::milli>(end - start).count();
        LOGI("Output decoding time: %.03f ms", ms);

        start = std::chrono::high_resolution_clock::now();
        detections = verid::Postprocessing::nonMaxSuppression(detections, 0.4f, limit);
        end = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<float, std::milli>(end - start).count();
        LOGI("NMS time: %.03f ms", ms);
//        std::vector<float> faces;
//        faces.reserve(detections.size() * 18);
//        for (auto& det : detections) {
//            faces.push_back(det.bounds.x);
//            faces.push_back(det.bounds.y);
//            faces.push_back(det.bounds.width);
//            faces.push_back(det.bounds.height);
//            faces.push_back(det.angle.yaw);
//            faces.push_back(det.angle.pitch);
//            faces.push_back(det.angle.roll);
//            faces.push_back(det.landmarks[0].x);
//            faces.push_back(det.landmarks[0].y);
//            faces.push_back(det.landmarks[1].x);
//            faces.push_back(det.landmarks[1].y);
//            faces.push_back(det.landmarks[2].x);
//            faces.push_back(det.landmarks[2].y);
//            faces.push_back(det.landmarks[3].x);
//            faces.push_back(det.landmarks[3].y);
//            faces.push_back(det.landmarks[4].x);
//            faces.push_back(det.landmarks[4].y);
//            faces.push_back(det.quality);
//        }
//        return faces;

        int numFaces = std::min(static_cast<int>(detections.size()), limit);

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

    int FaceDetection::detectFaces(JNIEnv *env, jobject bitmap, const int limit, float *buffer) {
        throw std::runtime_error("Not implemented");
//        std::vector<float> input;
//        preprocessing_.preprocessBitmapOnGPU(env, bitmap, input);
//        return detectFaces(input, limit, buffer);
    }
} // verid