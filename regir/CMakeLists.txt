set(TARGET_NAME "regir")

file(
    GLOB_RECURSE SOURCES
    *.h *.hpp *.c *.cpp)

set(
    CUDA_KERNELS
    "copy_buffers.cu"
    "build_cell_reservoirs.cu"
)

set(
    OPTIX_KERNELS
    "optix_kernels.cu"
)

file(
    GLOB_RECURSE COMMON_SHADERS
    "../common/shaders/*")

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
    SOURCES ${COMMON_SHADERS}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/shaders"
    GENERATED_FILES COPIED_COMMON_SHADERS
)

# non-essentials
source_group(
    "non-essentials" REGULAR_EXPRESSION
    "../common/[^/]*\.(h|hpp|c|cpp)$")
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
    "non-essentials/ext/tinyexr" REGULAR_EXPRESSION
    "../ext/tinyexr/[^/]*\.(h|hpp|c|cpp)$")
source_group(
    "non-essentials/ext/tinyexr/miniz" REGULAR_EXPRESSION
    "../ext/tinyexr/deps/miniz/[^/]*\.(h|hpp|c|cpp)$")
source_group(
    "non-essentials/ext" REGULAR_EXPRESSION
    "../ext/[^/]*\.(h|hpp|c|cc|cpp)$")
source_group(
    "non-essentials/utils" REGULAR_EXPRESSION
    "../utils/.*$")
source_group(
    "non-essentials/shaders" REGULAR_EXPRESSION
    "../common/shaders/.*")

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
    ${COMMON_SHADERS}
)
target_compile_features("${TARGET_NAME}" PRIVATE cxx_std_20)
set_target_properties("${TARGET_NAME}" PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(
    "${TARGET_NAME}" PRIVATE
    "../ext/tinyexr"
    "../ext/tinyexr/deps/miniz"
    "../ext/assimp/include"
    "${CMAKE_BINARY_DIR}/ext/assimp/include"
    "../ext/gl3w/include"
    "../ext/glfw/include"
    "../ext/imgui"
    "${OPTIX_INCLUDE_DIR}"
)
target_link_directories(
    "${TARGET_NAME}" PRIVATE
    "../ext/glfw/lib"
)
target_link_libraries(
    "${TARGET_NAME}"
    fakelib
    CUDA::cuda_driver
    opengl32
    glfw3
    assimp::assimp
)
    