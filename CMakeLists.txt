cmake_minimum_required(VERSION 3.18)
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
find_package(CUDAToolkit 11.6 REQUIRED)
find_package(OptiX74)
include("copy_files")
include("nvcuda_compile_ptx")

if (OptiX74_FOUND)
    set(OPTIX_INCLUDE_DIR "${OPTIX74_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "OptiX SDK 7.4 not found.")
endif()



# ----------------------------------------------------------------
# JP: プロジェクト間で共通のソースコードのリスト作成。

file(
    GLOB_RECURSE COMMON_SOURCES
    "common/*"
    "utils/*"
    "ext/*"
)
list(
    FILTER COMMON_SOURCES INCLUDE REGEX
    ".*\.(h|hpp|cuh|c|cc|cpp)$")
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
    "ext/assimp/.*")
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
    "ext/tiny-cuda-nn/.*")
file(
    GLOB IMGUI_BACKEND_SOURCES
    "ext/imgui/backends/imgui_impl_glfw*"
    "ext/imgui/backends/imgui_impl_opengl3*"
)
set(
    COMMON_SOURCES
    ${COMMON_SOURCES}
    ${IMGUI_BACKEND_SOURCES}
)
# foreach(f IN LISTS COMMON_SOURCES)
#     message("${f}")
# endforeach()

# END:JP: プロジェクト間で共通のソースコードのリスト作成。
# ----------------------------------------------------------------



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

add_library(fakelib INTERFACE)
target_compile_definitions(
    fakelib INTERFACE
    "$<$<CONFIG:Debug>:_DEBUG=1>"
)
target_compile_options(
    fakelib INTERFACE
    # if (compilerID == MSVC && compilerLanguage != CUDA) set(/MP);
    "$<$<AND:$<C_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>"
)

add_subdirectory(path_tracing)
add_subdirectory(restir)
add_subdirectory(regir)
add_subdirectory(neural_radiance_caching)
