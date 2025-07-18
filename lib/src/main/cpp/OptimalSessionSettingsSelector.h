//
// Created by Jakub Dolejs on 15/07/2025.
//

#pragma once

#include <string>
#include <tuple>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/nnapi/nnapi_provider_factory.h>

namespace verid {

    std::tuple<std::string, bool, uint32_t> createOptimalSessionOptions(
            const std::string &fp32modelPath,
            const std::string &fp16modelPath,
            const std::string &int8modelPath);

}