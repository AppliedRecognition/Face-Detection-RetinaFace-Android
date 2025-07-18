//
// Created by Jakub Dolejs on 17/07/2025.
//

#include "GpuPreprocessing.h"
#include <android/bitmap.h>
#include <fcntl.h>
#include <unistd.h>
#include <android/log.h>

#define LOG_TAG "Ver-ID"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace verid {

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
            glGetProgramInfoLog(program, sizeof(log), nullptr, log);
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
uniform vec2 uScale;
in vec2 vTexCoord;
out vec4 fragColor;
void main() {
    vec2 srcCoord = vTexCoord / uScale;
    vec4 pixel = texture(uTexture, srcCoord);
    fragColor = vec4(pixel.rgb, 1.0);
}
)";

    static void writeToFile(std::vector<float> &rgba, int targetSize) {
        int fd = open("/data/user/0/com.appliedrec.verid3.facedetection.testapp/files/debug_output.ppm", O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (fd >= 0) {
            std::string header = "P6\n640 640\n255\n";
            write(fd, header.c_str(), header.size());
            for (int i = 0; i < targetSize * targetSize; ++i) {
                uint8_t r = std::clamp(rgba[i * 4 + 0] * 255.f, 0.f, 255.f);
                uint8_t g = std::clamp(rgba[i * 4 + 1] * 255.f, 0.f, 255.f);
                uint8_t b = std::clamp(rgba[i * 4 + 2] * 255.f, 0.f, 255.f);
                write(fd, &r, 1);
                write(fd, &g, 1);
                write(fd, &b, 1);
            }
            close(fd);
            LOGI("Wrote RGBA to file /sdcard/debug_output.ppm");
        } else {
            LOGI("Failed to write RGBA to file /sdcard/debug_output.ppm");
        }
    }

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

        // Compile shader program
        program = createProgram(VERTEX_SHADER, FRAGMENT_SHADER);
        glUseProgram(program);

        // Setup quad
        float quadVertices[] = {
                -1, -1, 0, 0,
                1, -1, 1, 0,
                -1, 1, 0, 1,
                1, 1, 1, 1
        };
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // Setup output framebuffer + texture (fixed size)
        glGenTextures(1, &texOut);
        glBindTexture(GL_TEXTURE_2D, texOut);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, targetSize, targetSize, 0, GL_RGBA, GL_FLOAT, nullptr);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texOut, 0);

        // Setup shader uniforms
        glUniform1i(glGetUniformLocation(program, "uTexture"), 0);
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
    }

    void GpuPreprocessing::preprocessBitmapOnGPU(JNIEnv* env, jobject bitmap, std::vector<float>& outRGB) {
        AndroidBitmapInfo info;
        void* pixels;

        if (AndroidBitmap_getInfo(env, bitmap, &info) < 0 || info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            throw std::runtime_error("Invalid bitmap format");
        }
        if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
            throw std::runtime_error("Failed to lock pixels");
        }

        GLuint texIn;
        glGenTextures(1, &texIn);
        glBindTexture(GL_TEXTURE_2D, texIn);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, info.width, info.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        AndroidBitmap_unlockPixels(env, bitmap);

        float scale = std::min(float(targetSize) / info.width, float(targetSize) / info.height);
        glUniform2f(glGetUniformLocation(program, "uScale"), scale, scale);

        glViewport(0, 0, targetSize, targetSize);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texIn);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        std::vector<float> rgba(targetSize * targetSize * 4);
        glReadPixels(0, 0, targetSize, targetSize, GL_RGBA, GL_FLOAT, rgba.data());

        outRGB.resize(targetSize * targetSize * 3);
        for (int i = 0; i < targetSize * targetSize; ++i) {
            outRGB[i] = rgba[i * 4 + 0] * 255.f - 104.f;  // R
            outRGB[i + targetSize * targetSize] = rgba[i * 4 + 1] * 255.f - 117.f;  // G
            outRGB[i + 2 * targetSize * targetSize] = rgba[i * 4 + 2] * 255.f - 123.f;  // B
        }
        writeToFile(rgba, targetSize);
        glDeleteTextures(1, &texIn);
    }
} // verid