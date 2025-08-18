//
// Created by Jakub Dolejs on 17/07/2025.
//

#ifndef FACE_DETECTION_PREPROCESSING_H
#define FACE_DETECTION_PREPROCESSING_H

#include <vector>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <chrono>
#include "Logger.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace verid {

    class Preprocessing {
    public:
        explicit Preprocessing(int targetSize)
                : targetSize(targetSize),
                  squareBuffer(targetSize * targetSize * 3, 0) {}

        ~Preprocessing() {
            squareBuffer.clear();
            squareBuffer.shrink_to_fit();
        }

        void preprocessBitmap(void* inputBuffer, int width, int height, int bytesPerRow, int imageFormat, std::vector<float>& outRGB) {
            const int bpp = bytesPerPixel(imageFormat);
            if (bpp < 3) throw std::runtime_error("Unsupported format for RGB extraction");
            if (!inputBuffer)
                throw std::invalid_argument("inputBuffer is null");
            if (width <= 0 || height <= 0)
                throw std::invalid_argument("Invalid image dimensions");
            if (bytesPerRow < width * bpp)
                throw std::invalid_argument("bytesPerRow too small for width");
            const unsigned char* src = static_cast<unsigned char*>(inputBuffer);
            // Compute scale
            float scale = std::min(1.0f, static_cast<float>(targetSize) / static_cast<float>(std::max(width, height)));
            int scaledWidth = static_cast<int>(static_cast<float>(width) * scale);
            int scaledHeight = static_cast<int>(static_cast<float>(height) * scale);
            unsigned char* square = squareBuffer.data();
            // Nearest neighbour resampling
            for (int y = 0; y < scaledHeight; ++y) {
                int nearestY = static_cast<int>(static_cast<float>(y) / scale);
                for (int x = 0; x < scaledWidth; ++x) {
                    int nearestX = static_cast<int>(static_cast<float>(x) / scale);
                    const unsigned char* p = src + nearestY * bytesPerRow + nearestX * bpp;
                    square[(y * targetSize + x) * 3 + 0] = p[channelIndex(imageFormat, 0)];
                    square[(y * targetSize + x) * 3 + 1] = p[channelIndex(imageFormat, 1)];
                    square[(y * targetSize + x) * 3 + 2] = p[channelIndex(imageFormat, 2)];
                }
                std::memset(&square[(y * targetSize + scaledWidth) * 3], 0, (targetSize - scaledWidth) * 3);
            }
            for (int y = scaledHeight; y < targetSize; ++y) {
                std::memset(&square[y * targetSize * 3], 0, targetSize * 3);
            }
            // Split into R, G, B planes
            size_t N = static_cast<size_t>(targetSize) * targetSize;
            outRGB.resize(3 * N);
            float* R = outRGB.data();
            float* G = R + N;
            float* B = G + N;
#ifdef __AVX2__
            LOGI("Using AVX2");
            simdSplitAVX2(square, R, G, B, N);
#elif defined(__ARM_NEON)
            LOGI("Using NEON");
            simdSplitNEON(square, R, G, B, N);
#else
#pragma omp parallel for if (N > 10000)
            for (size_t i = 0; i < N; ++i) {
                R[i] = square[i * 3 + 0] - 104.0f;
                G[i] = square[i * 3 + 1] - 117.0f;
                B[i] = square[i * 3 + 2] - 123.0f;
            }
#endif
        }

    private:
        int targetSize;
        std::vector<unsigned char> squareBuffer;

        static int bytesPerPixel(int format) {
            switch (format) {
                case 0: case 1: return 3;  // RGB, BGR
                case 2: case 3: case 4: case 5: return 4; // ARGB, BGRA, ABGR, RGBA
                case 6: return 1;           // Grayscale
                default: return 0;
            }
        }

        static int channelIndex(int format, int c) {
            // Map to RGB order
            switch (format) {
                case 0: return c;                      // RGB
                case 1: return 2 - c;                 // BGR
                case 2: return c + 1;                 // ARGB, skip alpha
                case 3: return (c == 0) ? 2 : (c == 2) ? 0 : 1;  // BGRA
                case 4: return (c == 0) ? 3 : (c == 1) ? 2 : 1;  // ABGR
                case 5: return c;                     // RGBA
                default: return 0;
            }
        }

        void simdSplitAVX2(const unsigned char* sq, float* R, float* G, float* B, size_t N) {
#ifdef __AVX2__
            size_t i = 0;
        for (; i + 8 <= N; i += 8) {
            __m256i pixels = _mm256_loadu_si256((__m256i*)(sq + i * 3));
            __m256i r = _mm256_and_si256(pixels, _mm256_set1_epi32(0xFF));
            __m256i g = _mm256_and_si256(_mm256_srli_epi32(pixels, 8), _mm256_set1_epi32(0xFF));
            __m256i b = _mm256_and_si256(_mm256_srli_epi32(pixels, 16), _mm256_set1_epi32(0xFF));
            float r_f[8], g_f[8], b_f[8];
            _mm256_storeu_si256((__m256i*)r_f, r);
            _mm256_storeu_si256((__m256i*)g_f, g);
            _mm256_storeu_si256((__m256i*)b_f, b);
            for (int j = 0; j < 8; ++j) {
                R[i + j] = r_f[j] - 114.0f;
                G[i + j] = g_f[j] - 117.0f;
                B[i + j] = b_f[j] - 123.0f;
            }
        }
        for (; i < N; ++i) {
            R[i] = sq[i * 3 + 0] - 114.0f;
            G[i] = sq[i * 3 + 1] - 117.0f;
            B[i] = sq[i * 3 + 2] - 123.0f;
        }
#endif
        }

        static void simdSplitNEON(const unsigned char* sq, float* R, float* G, float* B, size_t N) {
#ifdef __ARM_NEON
            size_t i = 0;

            const float32x4_t meanR = vdupq_n_f32(104.0f);
            const float32x4_t meanG = vdupq_n_f32(117.0f);
            const float32x4_t meanB = vdupq_n_f32(123.0f);

            for (; i + 8 <= N; i += 8) {
                uint8x8x3_t pixels = vld3_u8(sq + i * 3);

                uint16x8_t r16 = vmovl_u8(pixels.val[0]);
                uint16x8_t g16 = vmovl_u8(pixels.val[1]);
                uint16x8_t b16 = vmovl_u8(pixels.val[2]);

                float32x4_t r_f1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(r16)));
                float32x4_t r_f2 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(r16)));
                float32x4_t g_f1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(g16)));
                float32x4_t g_f2 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(g16)));
                float32x4_t b_f1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b16)));
                float32x4_t b_f2 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b16)));

                // Subtract mean
                r_f1 = vsubq_f32(r_f1, meanR);
                r_f2 = vsubq_f32(r_f2, meanR);
                g_f1 = vsubq_f32(g_f1, meanG);
                g_f2 = vsubq_f32(g_f2, meanG);
                b_f1 = vsubq_f32(b_f1, meanB);
                b_f2 = vsubq_f32(b_f2, meanB);

                vst1q_f32(R + i,     r_f1);
                vst1q_f32(R + i + 4, r_f2);
                vst1q_f32(G + i,     g_f1);
                vst1q_f32(G + i + 4, g_f2);
                vst1q_f32(B + i,     b_f1);
                vst1q_f32(B + i + 4, b_f2);
            }

            for (; i < N; ++i) {
                R[i] = static_cast<float>(sq[i * 3 + 0]) - 104.0f;
                G[i] = static_cast<float>(sq[i * 3 + 1]) - 117.0f;
                B[i] = static_cast<float>(sq[i * 3 + 2]) - 123.0f;
            }
#endif
        }
    };

}

#endif //FACE_DETECTION_PREPROCESSING_H
