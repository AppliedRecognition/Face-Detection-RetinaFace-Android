# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("FaceDetectionRetinaFace")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS_RELEASE "-O3 -ftree-vectorize")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ftree-vectorize")

# Add the native source code
add_library(
        ${CMAKE_PROJECT_NAME}
        SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        core.cpp
        FaceDetection.cpp
        OptimalSessionSettingsSelector.cpp
        Postprocessing.cpp
)

# Path to ONNX Runtime headers
include_directories(
        ${CMAKE_SOURCE_DIR}/onnxruntime/include
        ${CMAKE_SOURCE_DIR}/onnxruntime/include/onnxruntime/core/session
)

# Import ONNX Runtime prebuilt shared library
add_library(
        onnxruntime
        SHARED
        IMPORTED
)

set_target_properties(
        onnxruntime
        PROPERTIES
        IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libonnxruntime.so
)

find_library(log-lib log)
find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)

# Link libraries to your native code
target_link_libraries(
        ${CMAKE_PROJECT_NAME}
        onnxruntime
        ${log-lib}
        ${android-lib}
        ${jnigraphics-lib}
        EGL
        GLESv3
)