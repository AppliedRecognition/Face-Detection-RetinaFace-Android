//
// Created by Jakub Dolejs on 14/07/2025.
//

#include "Preprocessing.h"
#include <android/bitmap.h>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <chrono>
#include <android/log.h>

#define LOG_TAG "Ver-ID"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace verid {

//    std::vector<float> inputTensorFromAndroidBitmap(JNIEnv *env, jobject bitmap, int targetSize) {
//        AndroidBitmapInfo info;
//        void *pixels;
//
//        if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
//            throw std::runtime_error("AndroidBitmap_getInfo failed!");
//        }
//
//        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
//            throw std::runtime_error("Only RGBA_8888 bitmaps supported!");
//        }
//
//        if (info.width != 640 || info.height != 640) {
//            throw std::runtime_error("Bitmap must be 640x640!");
//        }
//
//        if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
//            throw std::runtime_error("AndroidBitmap_lockPixels failed!");
//        }
//
//        int pixelCount = targetSize * targetSize;
//        std::vector<float> input(3 * pixelCount);
//
//        auto *src = static_cast<uint8_t *>(pixels);
//
//        float *R = input.data();
//        float *G = R + pixelCount;
//        float *B = G + pixelCount;
//
//        const float32x4_t biasR = vdupq_n_f32(-104.f);
//        const float32x4_t biasG = vdupq_n_f32(-117.f);
//        const float32x4_t biasB = vdupq_n_f32(-123.f);
//
//        int i = 0;
//        for (; i <= pixelCount - 8; i += 8) {
//            // Load 32 bytes (8 pixels × 4 channels, RGBA)
//            uint8x16x2_t pixels_u8 = vld2q_u8(src + i * 4);  // 2x16 bytes (deinterleaved pairs)
//            uint8x16_t rg = pixels_u8.val[0];                // R0 G0 R1 G1 ...
//            uint8x16_t ba = pixels_u8.val[1];                // B0 A0 B1 A1 ...
//
//            // Extract R and G channels
//            uint8x8_t r_u8 = vuzp_u8(vget_low_u8(rg), vget_high_u8(rg)).val[0];
//            uint8x8_t g_u8 = vuzp_u8(vget_low_u8(rg), vget_high_u8(rg)).val[1];
//
//            // Extract B (skip A)
//            uint8x8_t b_u8 = vuzp_u8(vget_low_u8(ba), vget_high_u8(ba)).val[0];
//
//            // Widen to uint16 → uint32 → float
//            float32x4_t r_f32_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(r_u8))));
//            float32x4_t r_f32_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(r_u8))));
//
//            float32x4_t g_f32_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(g_u8))));
//            float32x4_t g_f32_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(g_u8))));
//
//            float32x4_t b_f32_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(b_u8))));
//            float32x4_t b_f32_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(b_u8))));
//
//            // Apply bias
//            r_f32_0 = vaddq_f32(r_f32_0, biasR);
//            r_f32_1 = vaddq_f32(r_f32_1, biasR);
//            g_f32_0 = vaddq_f32(g_f32_0, biasG);
//            g_f32_1 = vaddq_f32(g_f32_1, biasG);
//            b_f32_0 = vaddq_f32(b_f32_0, biasB);
//            b_f32_1 = vaddq_f32(b_f32_1, biasB);
//
//            // Store results
//            vst1q_f32(R + i, r_f32_0);
//            vst1q_f32(R + i + 4, r_f32_1);
//
//            vst1q_f32(G + i, g_f32_0);
//            vst1q_f32(G + i + 4, g_f32_1);
//
//            vst1q_f32(B + i, b_f32_0);
//            vst1q_f32(B + i + 4, b_f32_1);
//        }
//
//        // Scalar fallback for leftover pixels
//        for (; i < pixelCount; ++i) {
//            uint8_t *p = src + i * 4;
//            R[i] = static_cast<float>(p[0]) - 104.f;
//            G[i] = static_cast<float>(p[1]) - 117.f;
//            B[i] = static_cast<float>(p[2]) - 123.f;
//        }
//
//        AndroidBitmap_unlockPixels(env, bitmap);
//        return input;
//    }

    // No NEON
    std::vector<float> inputTensorFromAndroidBitmap(JNIEnv *env, jobject bitmap, int targetSize) {
        AndroidBitmapInfo info;
        void *pixels;

        if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
            throw std::runtime_error("AndroidBitmap_getInfo failed!");
        }

        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            throw std::runtime_error("Only RGBA_8888 bitmaps supported!");
        }

        if (info.width != targetSize || info.height != targetSize) {
            throw std::runtime_error("Bitmap must be 640x640!");
        }

        if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
            throw std::runtime_error("AndroidBitmap_lockPixels failed!");
        }

        int pixelCount = targetSize * targetSize;

        std::vector<float> input(3 * pixelCount);

        auto *src = static_cast<uint8_t *>(pixels);

        float *R = input.data();
        float *G = R + pixelCount;
        float *B = G + pixelCount;

        for (int i = 0; i <= pixelCount - 4; i += 4) {
            uint8_t *p0 = src + (i + 0) * 4;
            uint8_t *p1 = src + (i + 1) * 4;
            uint8_t *p2 = src + (i + 2) * 4;
            uint8_t *p3 = src + (i + 3) * 4;

            R[i + 0] = p0[0] - 104.f;
            R[i + 1] = p1[0] - 104.f;
            R[i + 2] = p2[0] - 104.f;
            R[i + 3] = p3[0] - 104.f;

            G[i + 0] = p0[1] - 117.f;
            G[i + 1] = p1[1] - 117.f;
            G[i + 2] = p2[1] - 117.f;
            G[i + 3] = p3[1] - 117.f;

            B[i + 0] = p0[2] - 123.f;
            B[i + 1] = p1[2] - 123.f;
            B[i + 2] = p2[2] - 123.f;
            B[i + 3] = p3[2] - 123.f;
        }

        AndroidBitmap_unlockPixels(env, bitmap);

        return input;
    }
} // verid