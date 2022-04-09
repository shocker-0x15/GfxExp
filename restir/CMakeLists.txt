set(TARGET_NAME "restir")

file(
    GLOB_RECURSE SOURCES
    *.h *.hpp *.c *.cpp)

set(
    CUDA_KERNELS
    "copy_buffers.cu"
    "per_pixel_ris.cu"
)

set(
    OPTIX_KERNELS
    "optix_kernels.cu"
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

set(ASSIMP_DLL "../ext/assimp/assimp-vc143-mt.dll")
copy_files(
    SOURCES "${ASSIMP_DLL}"
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}"
    GENERATED_FILES COPIED_FILES
)

# non-essentials
source_group(
    "non-essentials" REGULAR_EXPRESSION
    "../common/.*\.(h|hpp|c|cpp)$")
source_group(
    "non-essentials/ext/gl3w" REGULAR_EXPRESSION
    "../ext/gl3w/.*$")
source_group(
    "non-essentials/ext/glfw" REGULAR_EXPRESSION
    "../ext/glfw/.*\.h$")
source_group(
    "non-essentials/ext/imgui/core" REGULAR_EXPRESSION
    "../ext/imgui/[^/]*\.(h|cpp)$")
source_group(
    "non-essentials/ext/imgui" REGULAR_EXPRESSION
    "../ext/imgui/backends/.*\.(h|cpp)$")
source_group(
    "non-essentials/ext/miniz" REGULAR_EXPRESSION
    "../ext/miniz/[^/]*\.(h|hpp|c|cpp)$")
source_group(
    "non-essentials/ext/assimp" REGULAR_EXPRESSION
    "../ext/assimp/[^/]*\.(dll)$")
source_group(
    "non-essentials/ext" REGULAR_EXPRESSION
    "../ext/[^/]*\.(h|hpp|c|cc|cpp)$")
source_group(
    "non-essentials/utils" REGULAR_EXPRESSION
    "../utils/.*$")
source_group(
    "non-essentials/shaders" REGULAR_EXPRESSION
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/.*$")

# essentials
source_group(
    "essentials" REGULAR_EXPRESSION
    "${CMAKE_CURRENT_SOURCE_DIR}/[^/]*\.(h|hpp|c|cpp)$")
source_group(
    "essentials/GPU kernels" REGULAR_EXPRESSION
    "${CMAKE_CURRENT_SOURCE_DIR}/[^/]*\.(cu)$")

add_executable(
    "${TARGET_NAME}"
    ${COMMON_SOURCES}
    ${SOURCES}
    ${CUDA_KERNELS}
    ${OPTIX_KERNELS}
    ${SHADERS}
    ${ASSIMP_DLL}
)
target_compile_features("${TARGET_NAME}" PRIVATE cxx_std_20)
set_target_properties("${TARGET_NAME}" PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(
    "${TARGET_NAME}" PRIVATE
    "../ext/miniz"
    "../ext/assimp/include"
    "../ext/gl3w/include"
    "../ext/glfw/include"
    "../ext/imgui"
    "${OPTIX_INCLUDE_DIR}"
)
target_link_directories(
    "${TARGET_NAME}" PRIVATE
    "../ext/glfw/lib"
    "../ext/assimp"
)
target_link_libraries(
    "${TARGET_NAME}"
    fakelib
    CUDA::cuda_driver
    opengl32
    glfw3
    assimp-vc143-mt
)
    