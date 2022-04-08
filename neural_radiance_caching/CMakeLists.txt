set(TARGET_NAME "neural_radiance_caching")

file(
    GLOB_RECURSE SOURCES
    *.h *.hpp *.c *.cpp)

set(
    CUDA_KERNELS
    "kernels.cu"
    "copy_buffers.cu"
)

set(
    OPTIX_KERNELS
    "optix_kernels.cu"
)

set(
    CUDA_SOURCES
    "network_interface.cu"
)

file(
    GLOB_RECURSE SHADERS
    "shaders/*")

set(
    GPU_KERNEL_DEPENDENCIES
    "../utils/cuda_util.h"
    "../utils/optix_util.h"
    "../utils/optixu_on_cudau.h"
    "../common/common_shared.h"
    "../common/common_device.cuh"
    "${TARGET_NAME}_shared.h"
)

nvcuda_compile_ptx(
    SOURCES ${CUDA_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/ptxes"
    GENERATED_FILES CUDA_PTXES
    NVCC_OPTIONS
    "$<$<CONFIG:Debug>:-G>"
    "--gpu-architecture=compute_52"
    "-std=c++17"
    "-cudart" "shared"
    "--use_fast_math"
    "--relocatable-device-code=true"
    "-I${OPTIX_INCLUDE_DIR}"
)

nvcuda_compile_ptx(
    SOURCES ${OPTIX_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/ptxes"
    GENERATED_FILES OPTIX_PTXES
    NVCC_OPTIONS
    "$<$<CONFIG:Debug>:-lineinfo>"
    "--gpu-architecture=compute_52"
    "-std=c++17"
    "-cudart" "shared"
    "--use_fast_math"
    "--relocatable-device-code=true"
    "-I${OPTIX_INCLUDE_DIR}"
)

copy_files(
    SOURCES ${SHADERS}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/shaders"
    GENERATED_FILES COPIED_SHADERS
)

set(ASSIMP_DLL "${CMAKE_SOURCE_DIR}/ext/assimp/assimp-vc143-mt.dll")
copy_files(
    SOURCES "${ASSIMP_DLL}"
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}"
    GENERATED_FILES COPIED_FILES
)

# non-essentials
source_group(
    "non-essentials" REGULAR_EXPRESSION
    "${CMAKE_SOURCE_DIR}/common/.*\.(h|hpp|c|cpp)$")
source_group(
    "non-essentials/ext/gl3w" REGULAR_EXPRESSION
    "${CMAKE_SOURCE_DIR}/ext/gl3w/.*$")
source_group(
    "non-essentials/ext/glfw" REGULAR_EXPRESSION
    "${CMAKE_SOURCE_DIR}/ext/glfw/.*\.h$")
source_group(
    "non-essentials/ext/imgui/core" REGULAR_EXPRESSION
    "${CMAKE_SOURCE_DIR}/ext/imgui/[^/]*\.(h|cpp)$")
source_group(
    "non-essentials/ext/imgui" REGULAR_EXPRESSION
    "${CMAKE_SOURCE_DIR}/ext/imgui/backends/.*\.(h|cpp)$")
source_group(
    "non-essentials/ext/miniz" REGULAR_EXPRESSION
    "${CMAKE_SOURCE_DIR}/ext/miniz/[^/]*\.(h|hpp|c|cpp)$")
source_group(
    "non-essentials/ext/assimp" REGULAR_EXPRESSION
    "${CMAKE_SOURCE_DIR}/ext/assimp/[^/]*\.(dll)$")
source_group(
    "non-essentials/ext" REGULAR_EXPRESSION
    "${CMAKE_SOURCE_DIR}/ext/[^/]*\.(h|hpp|c|cc|cpp)$")
source_group(
    "non-essentials/utils" REGULAR_EXPRESSION
    "${CMAKE_SOURCE_DIR}/utils/.*$")
source_group(
    "non-essentials/shaders" REGULAR_EXPRESSION
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/.*$")

# essentials
source_group(
    "essentials" REGULAR_EXPRESSION
    "${CMAKE_CURRENT_SOURCE_DIR}/[^/]*\.(h|hpp|c|cpp)$")
source_group(
    "essentials" FILES
    ${CUDA_SOURCES})
source_group(
    "essentials/GPU kernels" REGULAR_EXPRESSION
    "${CMAKE_CURRENT_SOURCE_DIR}/[^/]*\.(cu)$")

list(GET CMAKE_CUDA_ARCHITECTURES 0 MIN_GPU_ARCH)
list(APPEND DEFINITIONS -DTCNN_MIN_GPU_ARCH=${MIN_GPU_ARCH})

list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_NVCC_FLAGS "$<$<CONFIG:Debug>:-G>")

add_executable(
    "${TARGET_NAME}"
    ${COMMON_SOURCES}
    ${SOURCES}
    ${CUDA_KERNELS}
    ${OPTIX_KERNELS}
    ${CUDA_SOURCES}
    ${SHADERS}
    ${ASSIMP_DLL}
)
target_compile_features("${TARGET_NAME}" PRIVATE cxx_std_20)
set_target_properties("${TARGET_NAME}" PROPERTIES CXX_EXTENSIONS OFF CUDA_EXTENSIOINS OFF)
target_compile_definitions("${TARGET_NAME}" PRIVATE ${DEFINITIONS})
target_compile_options("${TARGET_NAME}" PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>")
target_include_directories(
    "${TARGET_NAME}" PRIVATE
    "${CMAKE_SOURCE_DIR}/ext/miniz"
    "${CMAKE_SOURCE_DIR}/ext/assimp/include"
    "${CMAKE_SOURCE_DIR}/ext/gl3w/include"
    "${CMAKE_SOURCE_DIR}/ext/glfw/include"
    "${CMAKE_SOURCE_DIR}/ext/imgui"
    "${CMAKE_SOURCE_DIR}/ext/tiny-cuda-nn/dependencies"
    "${CMAKE_SOURCE_DIR}/ext/tiny-cuda-nn/include"
    "${OPTIX_INCLUDE_DIR}"
)
target_link_directories(
    "${TARGET_NAME}" PRIVATE
    "${CMAKE_SOURCE_DIR}/ext/glfw/lib"
    "${CMAKE_SOURCE_DIR}/ext/assimp"
)
target_link_libraries(
    "${TARGET_NAME}"
    fakelib
    CUDA::cuda_driver
    opengl32
    glfw3
    assimp-vc143-mt
    tiny-cuda-nn
)
