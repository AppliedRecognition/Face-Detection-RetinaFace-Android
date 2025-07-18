//
// Created by Jakub Dolejs on 17/07/2025.
//

#ifndef FACE_DETECTION_GPUPREPROCESSING_H
#define FACE_DETECTION_GPUPREPROCESSING_H

#include <vector>
#include <jni.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>

namespace verid {

    class GpuPreprocessing {
    public:
        explicit GpuPreprocessing(int targetSize);
        ~GpuPreprocessing();
        void preprocessBitmapOnGPU(void* inputBuffer, int width, int height, int bytesPerRow, int imageFormat, std::vector<float>& outRGB);
    private:
        int targetSize;
        EGLDisplay display;
        EGLConfig config;
        EGLContext ctx;
        EGLSurface surf;
        GLuint program;
        GLuint vao, vbo;
        GLuint fbo, texOut;
        int texInWidth = 0;
        int texInHeight = 0;
        GLuint texIn = 0;
        bool wroteFile = false;
        GLuint texture, shaderProgram;
    };

} // verid

#endif //FACE_DETECTION_GPUPREPROCESSING_H
