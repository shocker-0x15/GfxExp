cmake_minimum_required(VERSION 3.18)
project(
    GfxExp_cmake
    VERSION 1.0
    DESCRIPTION "Sandbox for graphics paper implementation"
    LANGUAGES C CXX CUDA)

set(
    CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmake"
    ${CMAKE_MODULE_PATH})
find_package(CUDAToolkit 11.6 REQUIRED)
find_package(OptiX74)
include("copy_files")
include("nvcuda_compile_ptx")

if (OptiX74_FOUND)
    set(OPTIX_INCLUDE_DIR "${OPTIX74_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "OptiX SDK 7.4 not found.")
endif()

file(
    GLOB_RECURSE COMMON_SOURCES
    "${CMAKE_SOURCE_DIR}/common/*.h"
    "${CMAKE_SOURCE_DIR}/common/*.hpp"
    "${CMAKE_SOURCE_DIR}/common/*.c"
    "${CMAKE_SOURCE_DIR}/common/*.cpp"
    "${CMAKE_SOURCE_DIR}/common/*.cuh"
    "${CMAKE_SOURCE_DIR}/utils/*.h"
    "${CMAKE_SOURCE_DIR}/utils/*.hpp"
    "${CMAKE_SOURCE_DIR}/utils/*.c"
    "${CMAKE_SOURCE_DIR}/utils/*.cpp"
    "${CMAKE_SOURCE_DIR}/ext/gl3w/*"
    "${CMAKE_SOURCE_DIR}/ext/glfw/*.h"
)
file(
    GLOB IMGUI_SOURCES
    "${CMAKE_SOURCE_DIR}/ext/imgui/*.h"
    "${CMAKE_SOURCE_DIR}/ext/imgui/*.cpp"
    "${CMAKE_SOURCE_DIR}/ext/imgui/backends/imgui_impl_glfw*"
    "${CMAKE_SOURCE_DIR}/ext/imgui/backends/imgui_impl_opengl3*"
)
file(
    GLOB MINIZ_SOURCES
    "${CMAKE_SOURCE_DIR}/ext/miniz/*.h"
    "${CMAKE_SOURCE_DIR}/ext/miniz/*.c"
)
file(
    GLOB STB_SOURCES
    "${CMAKE_SOURCE_DIR}/ext/stb_*"
)
file(
    GLOB TINYEXR_SOURCES
    "${CMAKE_SOURCE_DIR}/ext/tinyexr*"
)
set(
    COMMON_SOURCES
    ${COMMON_SOURCES}
    ${IMGUI_SOURCES}
    ${MINIZ_SOURCES}
    ${STB_SOURCES}
    ${TINYEXR_SOURCES}
)
list(
    FILTER COMMON_SOURCES
    EXCLUDE REGEX "${CMAKE_SOURCE_DIR}/common/obj_loader\..*")
list(
    FILTER COMMON_SOURCES
    EXCLUDE REGEX "${CMAKE_SOURCE_DIR}/ext/imgui/imgui_demo\.cpp")
# foreach(f IN LISTS COMMON_SOURCES)
#     message("${f}")
# endforeach()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_LIBRARY_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

option(TCNN_BUILD_BENCHMARK "Build tiny-cuda-nn example benchmark?" OFF)
option(TCNN_BUILD_EXAMPLES "Build tiny-cuda-nn example applications?" OFF)
add_subdirectory(ext/tiny-cuda-nn)

add_library(fakelib INTERFACE)
target_compile_definitions(
    fakelib INTERFACE
    "$<$<CONFIG:Debug>:_DEBUG=1>"
)
target_compile_options(
    fakelib INTERFACE
    "$<$<AND:$<C_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>"
)

add_subdirectory(path_tracing)
add_subdirectory(restir)
add_subdirectory(regir)
add_subdirectory(neural_radiance_caching)
add_dependencies(neural_radiance_caching tiny-cuda-nn)
