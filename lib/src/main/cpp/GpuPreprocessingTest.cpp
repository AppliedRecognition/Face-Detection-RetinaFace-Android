#include "GpuPreprocessing.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <vector>
#include <stdexcept>
#include <cstring>

namespace verid {

    GLenum mapFormat(int imageFormat) {
        // Map Kotlin ImageFormat enum ordinal to OpenGL format
        switch (imageFormat) {
            case 0:  // RGB
            case 1: return GL_RGB;  // BGR
            case 2:                  // ARGB
            case 3:                  // BGRA
            case 4:                  // ABGR
            case 5: return GL_RGBA; // RGBA or variants (treat as RGBA)
            case 6: return GL_RED;  // Grayscale
            default: throw std::invalid_argument("Unsupported image format");
        }
    }

    int bytesPerPixel(int imageFormat) {
        switch (imageFormat) {
            case 0: case 1: return 3;  // RGB, BGR
            case 2: case 3: case 4: case 5: return 4; // ARGB, BGRA, ABGR, RGBA
            case 6: return 1;          // Grayscale
            default: return 0;
        }
    }

    GLuint compileShader() {
        const char* vertexShaderSrc = R"(
            #version 330 core
            out vec2 TexCoord;
            void main() {
                vec2 pos[4] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(1, 1));
                vec2 uv[4]  = vec2[](vec2(0, 0), vec2(1, 0), vec2(0, 1), vec2(1, 1));
                gl_Position = vec4(pos[gl_VertexID], 0, 1);
                TexCoord = uv[gl_VertexID];
            }
        )";

        const char* fragmentShaderSrc = R"(
            #version 330 core
            in vec2 TexCoord;
            out vec4 FragColor;
            uniform sampler2D inputTex;
            uniform float u_scale;

            void main() {
                vec2 scaledUV = TexCoord;
                if (TexCoord.x > u_scale || TexCoord.y > u_scale) {
                    FragColor = vec4(0, 0, 0, 1);
                } else {
                    FragColor = texture(inputTex, TexCoord / u_scale);
                }
            }
        )";

        GLuint vert = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vert, 1, &vertexShaderSrc, nullptr);
        glCompileShader(vert);

        GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(frag, 1, &fragmentShaderSrc, nullptr);
        glCompileShader(frag);

        GLuint prog = glCreateProgram();
        glAttachShader(prog, vert);
        glAttachShader(prog, frag);
        glLinkProgram(prog);

        glDeleteShader(vert);
        glDeleteShader(frag);
        return prog;
    }

    void renderQuad() {
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }

    GpuPreprocessing::GpuPreprocessing(int targetSize)
            : targetSize(targetSize), fbo(0), texture(0), shaderProgram(0) {

        // Initialize OpenGL context (external, assumed ready)

        // Create texture for rendering
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, targetSize, targetSize, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Create framebuffer object
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("Framebuffer not complete");
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Compile shader (vertex & fragment) for scaling & padding
        shaderProgram = compileShader();
    }

    GpuPreprocessing::~GpuPreprocessing() {
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &texture);
        glDeleteProgram(shaderProgram);
    }

    void GpuPreprocessing::preprocessBitmapOnGPU(void* inputBuffer, int width, int height, int bytesPerRow, int imageFormat, std::vector<float>& outRGB) {
        // Upload inputBuffer as GL texture
        GLuint inputTex;
        glGenTextures(1, &inputTex);
        glBindTexture(GL_TEXTURE_2D, inputTex);
        GLenum format = mapFormat(imageFormat);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, bytesPerRow / (bytesPerPixel(imageFormat)));

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, format, GL_UNSIGNED_BYTE, inputBuffer);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Calculate scale factor and set viewport
        float scale = std::min(1.0f, static_cast<float>(targetSize) / std::max(width, height));
        int scaledWidth = static_cast<int>(width * scale);
        int scaledHeight = static_cast<int>(height * scale);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, targetSize, targetSize);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glBindTexture(GL_TEXTURE_2D, inputTex);

        // Render quad with scale uniform
        glUniform1f(glGetUniformLocation(shaderProgram, "u_scale"), scale);
        renderQuad();

        // Read back result
        std::vector<unsigned char> rgbBuffer(targetSize * targetSize * 3);
        glReadPixels(0, 0, targetSize, targetSize, GL_RGB, GL_UNSIGNED_BYTE, rgbBuffer.data());

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteTextures(1, &inputTex);

        // Split into R, G, B planes
        outRGB.resize(3 * targetSize * targetSize);
        size_t N = targetSize * targetSize;
        for (size_t i = 0; i < N; ++i) {
            outRGB[i] = rgbBuffer[3 * i] / 255.0f;             // R
            outRGB[i + N] = rgbBuffer[3 * i + 1] / 255.0f;     // G
            outRGB[i + 2 * N] = rgbBuffer[3 * i + 2] / 255.0f; // B
        }
    }

}