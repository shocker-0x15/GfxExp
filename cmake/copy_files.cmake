function(COPY_FILES)
  set(options "")
  set(oneValueArgs TARGET_PATH GENERATED_FILES)
  set(multiValueArgs SOURCES)

  CMAKE_PARSE_ARGUMENTS(COPY_FILES "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  foreach(INPUT ${COPY_FILES_SOURCES})
    get_filename_component(INPUT_NAME "${INPUT}" NAME)

    set(OUTPUT "${COPY_FILES_TARGET_PATH}/${INPUT_NAME}")

    list(APPEND DST_FILES "${OUTPUT}")

    add_custom_command(
      OUTPUT  "${OUTPUT}"
      MAIN_DEPENDENCY "${INPUT}"
      COMMAND ${CMAKE_COMMAND} -E copy "${INPUT}" "${OUTPUT}" WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endforeach()

  set(${COPY_FILES_GENERATED_FILES} ${DST_FILES} PARENT_SCOPE)
endfunction()
