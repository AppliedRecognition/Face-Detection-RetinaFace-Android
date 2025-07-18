//
// Created by Jakub Dolejs on 15/07/2025.
//

#include "OptimalSessionSettingsSelector.h"

#include <vector>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <unistd.h>
#include "Logger.h"

namespace verid {

    struct Options {
        std::string modelPath;
        bool useNnapi;
        uint32_t nnapiFlags;
    };

    struct Result {
        Options options;
        double averageTimeMs;
    };

    double runInference(Ort::Session& session, size_t warmupRuns, size_t testRuns) {
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputNameAllocated = session.GetInputNameAllocated(0, allocator);
        const char* inputName = inputNameAllocated.get();

        auto inputShapeInfo = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        auto inputShape = inputShapeInfo.GetShape();
        size_t inputSize = 1;
        for (auto dim : inputShape) {
            inputSize *= (dim > 0) ? dim : 1;  // handle dynamic shapes as 1
        }

        std::vector<float> inputTensorValues(inputSize, 0.5f);
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        const char* inputNames[] = {inputName};

        size_t outputCount = session.GetOutputCount();
        std::vector<const char*> outputNames;
        for (size_t i = 0; i < outputCount; ++i) {
            auto nameAllocated = session.GetOutputNameAllocated(i, allocator);
            outputNames.push_back(nameAllocated.release());  // we take ownership
        }

        for (size_t i = 0; i < warmupRuns; ++i) {
            auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputSize, inputShape.data(), inputShape.size());
            session.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames.data(), outputCount);
        }

        double totalMs = 0.0;
        for (size_t i = 0; i < testRuns; ++i) {
            auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputSize, inputShape.data(), inputShape.size());
            auto start = std::chrono::high_resolution_clock::now();
            auto output = session.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames.data(), outputCount);
            auto end = std::chrono::high_resolution_clock::now();
            totalMs += std::chrono::duration<double, std::milli>(end - start).count();
        }

        return totalMs / static_cast<double>(testRuns);
    }

    Ort::SessionOptions createSessionOptions(bool useNnapi, uint32_t nnapiFlags) {
        Ort::SessionOptions options;
        int cores = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
        int numThreads = std::min(cores, 4);
        options.SetIntraOpNumThreads(numThreads);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
        if (useNnapi) {
            OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_Nnapi(options, nnapiFlags);
            if (status != nullptr) {
                const char* msg = Ort::GetApi().GetErrorMessage(status);
                Ort::GetApi().ReleaseStatus(status);
                throw std::runtime_error(std::string("NNAPI setup error: ") + msg);
            }
        }
        return options;
    }

    std::tuple<std::string, bool, uint32_t> createOptimalSessionOptions(
            const std::string& fp32modelPath,
            const std::string& fp16modelPath,
            const std::string& int8modelPath)
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, LOG_TAG);

        std::vector<Options> combinations = {
                {fp32modelPath, false, 0},
                {fp32modelPath, true, 0},
                {fp32modelPath, true, NNAPI_FLAG_CPU_DISABLED},

                {fp16modelPath, false, 0},
                {fp16modelPath, true, NNAPI_FLAG_USE_FP16},
                {fp16modelPath, true, NNAPI_FLAG_USE_FP16 | NNAPI_FLAG_CPU_DISABLED},

                {int8modelPath, false, 0},
                {int8modelPath, true, 0}
        };

        std::vector<Result> results;

        for (const auto& opt : combinations) {
            try {
                auto sessionOptions = createSessionOptions(opt.useNnapi, opt.nnapiFlags);
                Ort::Session session(env, opt.modelPath.c_str(), sessionOptions);
                double avgMs = runInference(session, 2, 2);
                results.push_back({opt, avgMs});
            } catch (const std::exception& e) {
                LOGI("Error with %s: %s", opt.modelPath.c_str(), e.what());
            }
        }

        if (results.empty()) {
            throw std::runtime_error("No successful inference runs.");
        }

        auto best = std::min_element(results.begin(), results.end(),
                                     [](const Result& a, const Result& b) {
                                         return a.averageTimeMs < b.averageTimeMs;
                                     });

        LOGI("Best configuration:\nModel: %s\nNNAPI: %s\nFlags: %d\nAverage time: %.03f",
             best->options.modelPath.c_str(),
             (best->options.useNnapi ? "ON" : "OFF"),
             best->options.nnapiFlags,
             best->averageTimeMs);

        return {best->options.modelPath, best->options.useNnapi, best->options.nnapiFlags};
    }

}  // namespace