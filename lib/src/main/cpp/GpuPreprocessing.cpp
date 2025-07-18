#include "GpuPreprocessing.h"
#include <android/bitmap.h>
#include <android/log.h>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>

#define LOG_TAG "Ver-ID"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace verid {

    static void writeToFile(std::vector<uint8_t> &rgba, int targetSize) {
        std::string fileName = "/data/user/0/com.appliedrec.verid3.facedetection.testapp/files/debug_output.ppm";
        int fd = open(fileName.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (fd >= 0) {
            std::string header = "P6\n640 640\n255\n";
            write(fd, header.c_str(), header.size());
            for (int i = 0; i < targetSize * targetSize; ++i) {
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

    static GLuint compileShader(GLenum type, const char* src) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);
        GLint success = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char log[512];
            glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
            throw std::runtime_error(std::string("Shader compile error: ") + log);
        }
        return shader;
    }

    static GLuint createProgram(const char* vsSrc, const char* fsSrc) {
        GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
        GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
        GLuint program = glCreateProgram();
        glAttachShader(program, vs);
        glAttachShader(program, fs);
        glLinkProgram(program);
        GLint success = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            char log[512];
            throw std::runtime_error(std::string("Program link error: ") + log);
        }
        glDeleteShader(vs);
        glDeleteShader(fs);
        return program;
    }

    const char* VERTEX_SHADER = R"(
#version 300 es
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTex;
out vec2 vTexCoord;
void main() {
    vTexCoord = aTex;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

    const char* FRAGMENT_SHADER = R"(
#version 300 es
precision mediump float;
uniform sampler2D uTexture;
in vec2 vTexCoord;
out vec4 fragColor;
void main() {
    fragColor = texture(uTexture, vTexCoord);
}
)";

    GpuPreprocessing::GpuPreprocessing(int targetSize) : targetSize(targetSize) {
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        eglInitialize(display, nullptr, nullptr);

        EGLint attr[] = { EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8,
                          EGL_BLUE_SIZE, 8, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE };
        EGLint num;
        eglChooseConfig(display, attr, &config, 1, &num);

        EGLint ctxAttr[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
        ctx = eglCreateContext(display, config, EGL_NO_CONTEXT, ctxAttr);

        EGLint surfAttr[] = { EGL_WIDTH, targetSize, EGL_HEIGHT, targetSize, EGL_NONE };
        surf = eglCreatePbufferSurface(display, config, surfAttr);

        eglMakeCurrent(display, surf, surf, ctx);

        program = createProgram(VERTEX_SHADER, FRAGMENT_SHADER);
        glUseProgram(program);

        // Setup output framebuffer and texture
        glGenTextures(1, &texOut);
        glBindTexture(GL_TEXTURE_2D, texOut);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, targetSize, targetSize, 0, GL_RGBA, GL_FLOAT, nullptr);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texOut, 0);
    }

    GpuPreprocessing::~GpuPreprocessing() {
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
        glDeleteTextures(1, &texOut);
        glDeleteFramebuffers(1, &fbo);
        glDeleteProgram(program);
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroySurface(display, surf);
        eglDestroyContext(display, ctx);
        eglTerminate(display);
        if (texIn != 0) {
            glDeleteTextures(1, &texIn);
        }
    }

    void GpuPreprocessing::preprocessBitmapOnGPU(
            void* inputBuffer,
            int width,
            int height,
            int bytesPerRow,
            int imageFormat,
            std::vector<float>& outRGB
    ) {
        const int channels = 4;  // Uploading as RGBA

        // Allocate packed buffer if needed
        const uint8_t* src = static_cast<uint8_t*>(inputBuffer);
        std::vector<uint8_t> packedData;
        const void* uploadPtr = inputBuffer;

        if (bytesPerRow != width * channels) {
            packedData.resize(width * height * channels);
            for (int y = 0; y < height; ++y) {
                memcpy(
                        packedData.data() + y * width * channels,
                        src + y * bytesPerRow,
                        width * channels
                );
            }
            uploadPtr = packedData.data();
        }

        auto start = std::chrono::high_resolution_clock::now();
        if (texIn == 0 || texInWidth != width || texInHeight != height) {
            if (texIn != 0) {
                glDeleteTextures(1, &texIn);
            }
            glGenTextures(1, &texIn);
            glBindTexture(GL_TEXTURE_2D, texIn);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            texInWidth = width;
            texInHeight = height;

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        } else {
            glBindTexture(GL_TEXTURE_2D, texIn);
        }

        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, uploadPtr);
        auto end = std::chrono::high_resolution_clock::now();
        LOGI("glTexImage2D: %.03f ms", std::chrono::duration<float, std::milli>(end - start).count());

        float scale = std::min(float(targetSize) / width, float(targetSize) / height);
        float scaledW = (width * scale) / targetSize * 2.0f;
        float scaledH = (height * scale) / targetSize * 2.0f;

        LOGI("Input size: %d x %d, targetSize: %d", width, height, targetSize);
        LOGI("Computed scale: %f", scale);
        LOGI("Scaled NDC width: %f, height: %f", scaledW, scaledH);

        float quadVertices[] = {
                -1.0f,            -1.0f,            0.0f, 1.0f,  // bottom-left
                -1.0f + scaledW,  -1.0f,            1.0f, 1.0f,  // bottom-right
                -1.0f,            -1.0f + scaledH,  0.0f, 0.0f,  // top-left
                -1.0f + scaledW,  -1.0f + scaledH,  1.0f, 0.0f   // top-right
        };

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_DYNAMIC_DRAW);

        glViewport(0, 0, targetSize, targetSize);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texIn);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        std::vector<uint8_t> rgba(targetSize * targetSize * 4);
        start = std::chrono::high_resolution_clock::now();
        glReadPixels(0, 0, targetSize, targetSize, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
        end = std::chrono::high_resolution_clock::now();
        LOGI("glReadPixels: %.03f ms", std::chrono::duration<float, std::milli>(end - start).count());

        // Set channel indices once
        int rIndex = 0, gIndex = 1, bIndex = 2;
        switch (imageFormat) {
            case 0:  // RGB
            case 2:  // ARGB
            case 5:  // RGBA
                rIndex = 0; gIndex = 1; bIndex = 2;
                break;
            case 1:  // BGR
            case 3:  // BGRA
            case 4:  // ABGR
                rIndex = 2; gIndex = 1; bIndex = 0;
                break;
            case 6:  // GRAYSCALE
                rIndex = gIndex = bIndex = 0;
                break;
            default:
                throw std::runtime_error("Unsupported image format");
        }

        outRGB.resize(targetSize * targetSize * 3);
        for (int i = 0; i < targetSize * targetSize; ++i) {
            outRGB[i] = static_cast<float>(rgba[i * 4 + rIndex]) - 104.f;
            outRGB[i + targetSize * targetSize] = static_cast<float>(rgba[i * 4 + gIndex]) - 117.f;
            outRGB[i + 2 * targetSize * targetSize] = static_cast<float>(rgba[i * 4 + bIndex]) - 123.f;
        }
        if (!wroteFile) {
            verid::writeToFile(rgba, targetSize);
            wroteFile = true;
        }
    }

} // namespace verid