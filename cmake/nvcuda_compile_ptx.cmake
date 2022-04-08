# Generate a custom build rule to translate *.cu files to *.ptx files.
# NVCUDA_COMPILE_PTX(
#   SOURCES file1.cu file2.cu
#   DEPENDENCIES header1.h header2.h
#   TARGET_PATH <path where ptxs should be stored>
#   FILENAME_SUFFIX <suffix of the output filename that will be inserted before ".ptx">
#   GENERATED_FILES ptx_sources
#   NVCC_OPTIONS -arch=sm_30
# )

# Generates *.ptx files for the given source files.
# ptx_sources will contain the list of generated files.
# DAR Using this becasue I do not want filenames like "cuda_compile_ptx_generated_raygeneration.cu.ptx" but just "raygeneration.ptx".

function(NVCUDA_COMPILE_PTX)
  if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "ERROR: Only 64-bit programs supported.")
  endif()

  set(options "")
  set(oneValueArgs TARGET_PATH GENERATED_FILES FILENAME_SUFFIX)
  set(multiValueArgs NVCC_OPTIONS SOURCES DEPENDENCIES)

  CMAKE_PARSE_ARGUMENTS(NVCUDA_COMPILE_PTX "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  if(NOT WIN32) # Do not create a folder with the name ${ConfigurationName} under Windows.
    # Under Linux make sure the target directory exists. 
    file(MAKE_DIRECTORY ${NVCUDA_COMPILE_PTX_TARGET_PATH})
  endif()
  
  # Custom build rule to generate ptx files from cuda files
  foreach(input ${NVCUDA_COMPILE_PTX_SOURCES})
    get_filename_component(input_we "${input}" NAME_WE)

    set(output "${NVCUDA_COMPILE_PTX_TARGET_PATH}/${input_we}${NVCUDA_COMPILE_PTX_FILENAME_SUFFIX}.ptx")

    list(APPEND PTX_FILES "${output}")

    add_custom_command(
      OUTPUT  "${output}"
      MAIN_DEPENDENCY "${input}"
      DEPENDS ${NVCUDA_COMPILE_PTX_DEPENDENCIES}
      COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} --machine=64 --ptx ${NVCUDA_COMPILE_PTX_NVCC_OPTIONS} "${input}" -o "${output}" WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endforeach()

  set(${NVCUDA_COMPILE_PTX_GENERATED_FILES} ${PTX_FILES} PARENT_SCOPE)
endfunction()
