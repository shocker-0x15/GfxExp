set(TARGET_NAME "regir")

file(
    GLOB_RECURSE SOURCES
    *.h *.hpp *.c *.cpp)

set(
    CUDA_KERNELS
    "gpu_kernels/copy_buffers.cu"
    "gpu_kernels/build_cell_reservoirs.cu"
)

set(
    OPTIX_KERNELS
    "gpu_kernels/optix_gbuffer_kernels.cu"
    "gpu_kernels/optix_pathtracing_kernels.cu"
)

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

set_common_sources()
set_source_files_properties(
    ${GL3W_SOURCES} PROPERTIES
    GENERATED TRUE
)

# essentials
source_group(
    "essentials" REGULAR_EXPRESSION
    "${CMAKE_CURRENT_SOURCE_DIR}/[^/]*\.(h|hpp|c|cpp)$")
source_group(
    "essentials/GPU kernels" REGULAR_EXPRESSION
    "${CMAKE_CURRENT_SOURCE_DIR}/gpu_kernels/[^/]*\.(cu)$")

add_executable(
    "${TARGET_NAME}"
    ${COMMON_SOURCES}
    ${SOURCES}
    ${GL3W_SOURCES}
    ${CUDA_KERNELS}
    ${OPTIX_KERNELS}
)
target_compile_features("${TARGET_NAME}" PRIVATE cxx_std_20)
set_target_properties("${TARGET_NAME}" PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(
    "${TARGET_NAME}"
    fakelib
)
    