cmake_minimum_required(VERSION 3.26)
project(
    GfxExp_cmake
    VERSION 1.0
    DESCRIPTION "Sandbox for graphics paper implementation"
    LANGUAGES C CXX CUDA)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_LIBRARY_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(
    CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmake"
    ${CMAKE_MODULE_PATH})
find_package(CUDAToolkit 12.2 REQUIRED)
find_package(OptiX77)
include("copy_files")
include("nvcuda_compile_ptx")

if (OptiX77_FOUND)
    set(OPTIX_INCLUDE_DIR "${OPTIX77_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "OptiX SDK 7.7 not found.")
endif()



# ----------------------------------------------------------------
# JP: プロジェクト間で共通のソースコードのリスト作成。

file(
    GLOB_RECURSE COMMON_SOURCES
    "common/*"
    "utils/*"
    "ext/cubd/cubd.h"
    "ext/glfw/include/*"
    "ext/imgui/*"
    "ext/tinyexr/*"
)
list(
    FILTER COMMON_SOURCES INCLUDE REGEX
    ".*\.(h|hpp|cuh|c|cc|cpp|cu|vert|frag)$")
list(
    FILTER COMMON_SOURCES EXCLUDE REGEX
    "ext/prebuilt/.*")
list(
    FILTER COMMON_SOURCES EXCLUDE REGEX
    "common/obj_loader\..*")
list(
    FILTER COMMON_SOURCES EXCLUDE REGEX
    "ext/imgui/imgui_demo\.cpp$")
list(
    FILTER COMMON_SOURCES EXCLUDE REGEX
    "ext/imgui/(backends|misc|examples)/.*")
list(
    FILTER COMMON_SOURCES EXCLUDE REGEX
    "ext/tinyexr/deps/miniz/examples/.*")
list(
    FILTER COMMON_SOURCES EXCLUDE REGEX
    "ext/tinyexr/deps/ZFP/.*")
list(
    FILTER COMMON_SOURCES EXCLUDE REGEX
    "ext/tinyexr/(examples|experimental|test)/.*")
list(
    FILTER COMMON_SOURCES EXCLUDE REGEX
    "ext/tinyexr/test_tinyexr.cc$")
list(
    FILTER COMMON_SOURCES EXCLUDE REGEX
    "ext/cubd/.*(cpp|cu)$")
list(
    FILTER COMMON_SOURCES EXCLUDE REGEX
    "ext/cubd/cuda_util.*$")
file(GLOB STB_SOURCES "ext/stb_*")
file(
    GLOB IMGUI_BACKEND_SOURCES
    "ext/imgui/backends/imgui_impl_glfw*"
    "ext/imgui/backends/imgui_impl_opengl3*"
)
set(
    COMMON_SOURCES
    ${COMMON_SOURCES}
    ${STB_SOURCES}
    ${IMGUI_BACKEND_SOURCES}
)
# foreach(f IN LISTS COMMON_SOURCES)
#     message("${f}")
# endforeach()

# END:JP: プロジェクト間で共通のソースコードのリスト作成。
# ----------------------------------------------------------------



# Called from the sub directories
function(set_common_sources)
    set(
        GL3W_SOURCES
        "${CMAKE_BINARY_DIR}/ext/gl3w/include/GL/gl3w.h"
        "${CMAKE_BINARY_DIR}/ext/gl3w/include/GL/glcorearb.h"
        "${CMAKE_BINARY_DIR}/ext/gl3w/include/KHR/khplatform.h"
        "${CMAKE_BINARY_DIR}/ext/gl3w/src/gl3w.c"
        PARENT_SCOPE
    )

    file(
        GLOB_RECURSE COMMON_CUDA_KERNELS
        "../common/gpu_kernels/*")
    file(
        GLOB_RECURSE COMMON_SHADERS
        "../common/shaders/*")

    nvcuda_compile_ptx(
        SOURCES ${COMMON_CUDA_KERNELS}
        DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
        TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/ptxes"
        GENERATED_FILES CUDA_PTXES
        NVCC_OPTIONS
        "$<$<CONFIG:Debug>:-G>"
        "$<$<CONFIG:Debug>:-D_DEBUG=1>"
        "--gpu-architecture=compute_52"
        "-std=c++20"
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

    copy_files(
        SOURCES "../common/common_shader.h"
        TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}"
        GENERATED_FILES COPIED_COMMON_HEADER_FOR_SHADERS
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
        "non-essentials/ext/cubd" REGULAR_EXPRESSION
        "../ext/cubd/[^/]*\.h$")
    source_group(
        "non-essentials/ext" REGULAR_EXPRESSION
        "../ext/[^/]*\.(h|hpp|c|cc|cpp)$")
    source_group(
        "non-essentials/utils" REGULAR_EXPRESSION
        "../utils/.*$")
    source_group(
        "non-essentials/GPU kernels" REGULAR_EXPRESSION
        "../common/gpu_kernels/.*")
    source_group(
        "non-essentials/shaders" REGULAR_EXPRESSION
        "../common/shaders/.*")
endfunction()



# ----------------------------------------------------------------
# JP: CMakeLists.txtのあるサブディレクトリ内のターゲット全てにフォルダーを設定する。
# https://stackoverflow.com/questions/45092198/cmake-how-do-i-change-properties-on-subdirectory-project-targets

# _dir以下の(CMakeが有効な)サブディレクトリにあるターゲットを再帰的に取得する。
function(get_all_targets _result _dir)
    get_property(_subdirs DIRECTORY "${_dir}" PROPERTY SUBDIRECTORIES)
    foreach(_subdir IN LISTS _subdirs)
        get_all_targets(${_result} "${_subdir}")
    endforeach()
    get_property(_sub_targets DIRECTORY "${_dir}" PROPERTY BUILDSYSTEM_TARGETS)
    set(${_result} ${${_result}} ${_sub_targets} PARENT_SCOPE)
endfunction()

# _dir以下の(CMakeが有効な)サブディレクトリにあるターゲットにフォルダーを設定する。
function(set_project_group _folder_name _dir)
    get_all_targets(_targets "${_dir}")
    foreach(_target IN LISTS _targets)
        # message(${_target})
        set_target_properties(
            ${_target}
            PROPERTIES FOLDER "${_folder_name}"
        )
    endforeach()
endfunction()

# END: JP: CMakeLists.txtのあるサブディレクトリ内のターゲット全てにフォルダーを設定する。
# ----------------------------------------------------------------



# ----------------------------------------------------------------
# JP: インストールされているGPUの中から最小のCUDAアーキテクチャーを特定し、デフォルトのターゲットアーキテクチャーとする。

# adapted from https://stackoverflow.com/a/69353718
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
# JP: 前後のスペース削除。
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
# スペース区切りをセミコロン区切りに。
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
# 小数点の削除。
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

# Remove unsupported architectures
list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES "86+PTX")

list(SORT CMAKE_CUDA_ARCHITECTURES COMPARE NATURAL ORDER ASCENDING)
list(GET CMAKE_CUDA_ARCHITECTURES 0 MIN_GPU_ARCH)

set(TARGET_CUDA_ARCH ${MIN_GPU_ARCH} CACHE STRING "Target CUDA architecture")
set_property(
    CACHE TARGET_CUDA_ARCH PROPERTY STRINGS
    70 75 80 86)

# tiny-cuda-nnがターゲットアーキテクチャーを使うように環境変数を設定する。
set(ENV{TCNN_CUDA_ARCHITECTURES} ${TARGET_CUDA_ARCH})

# END: JP: インストールされているGPUの中から最小のCUDAアーキテクチャーを特定し、デフォルトのターゲットアーキテクチャーとする。
# ----------------------------------------------------------------



# gl3w
add_subdirectory(ext/gl3w)
set_project_group("External" "ext/gl3w")

# GLFW
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
option(GLFW_INSTALL "Generate installation target" OFF)
add_subdirectory(ext/glfw)
set_project_group("External" "ext/GLFW")

# Assimp
option(ASSIMP_BUILD_ASSIMP_TOOLS "If the supplementary tools for Assimp are built in addition to the library." OFF)
option(ASSIMP_BUILD_SAMPLES "If the official samples are built as well (needs Glut)." OFF)
option(ASSIMP_BUILD_TESTS "If the test suite for Assimp is built in addition to the library." OFF)
option(ASSIMP_INSTALL "Disable this if you want to use assimp as a submodule." OFF)
if(MSVC)
    option(ASSIMP_BUILD_ASSIMP_VIEW "If the Assimp view tool is built. (requires DirectX)" OFF)
endif()
add_subdirectory(ext/assimp)
set_project_group("External" "ext/assimp")

# Tiny CUDA Neural Networks
option(TCNN_BUILD_BENCHMARK "Build tiny-cuda-nn example benchmark?" OFF)
option(TCNN_BUILD_EXAMPLES "Build tiny-cuda-nn example applications?" OFF)
add_subdirectory(ext/tiny-cuda-nn)
set_project_group("External" "ext/tiny-cuda-nn")

# CUBd
option(CUBD_BUILD_SAMPLES "Build sample projects for cubd." OFF)
option(CUBD_BUILD_DYNAMIC_LIB "Build dynamic-link cubd library." OFF)
option(CUBD_BUILD_STATIC_LIB "Build static-link cubd library." ON)
add_subdirectory(ext/cubd)
set_project_group("External" "ext/cubd")

add_library(fakelib INTERFACE)
target_compile_definitions(
    fakelib INTERFACE
    "USE_CUBD_LIB"
    "$<$<CONFIG:Debug>:_DEBUG=1>"
)
target_compile_options(
    fakelib INTERFACE
    # if (compilerID == MSVC && compilerLanguage != CUDA) set(/MP);
    "$<$<AND:$<C_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>"
)
target_include_directories(
    fakelib INTERFACE
    "ext/tinyexr"
    "ext/tinyexr/deps/miniz"
    "ext/glfw/include"
    "ext/imgui"
    "ext/assimp/include"
    "${CMAKE_BINARY_DIR}/ext/assimp/include"
    "${CMAKE_BINARY_DIR}/ext/gl3w/include"
    "${OPTIX_INCLUDE_DIR}"
)
target_link_libraries(
    fakelib INTERFACE
    libcubd_static
    CUDA::cuda_driver
    opengl32
    gl3w
    glfw
    assimp::assimp
)

add_subdirectory(path_tracing)
add_subdirectory(restir)
add_subdirectory(regir)
add_subdirectory(neural_radiance_caching)
add_subdirectory(svgf)
