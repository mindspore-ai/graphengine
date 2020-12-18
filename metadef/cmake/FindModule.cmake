#[[
  module - the name of export imported target
  name   - find the library name
  path   - find the library path
#]]
function(find_module module name path)
    if (TARGET ${module})
        return()
    endif()
    find_library(${module}_LIBRARY_DIR NAMES ${name} NAMES_PER_DIR PATHS ${path}
      PATH_SUFFIXES lib
    )

    message(STATUS "find ${name} location ${${module}_LIBRARY_DIR}")
    if ("${${module}_LIBRARY_DIR}" STREQUAL "${module}_LIBRARY_DIR-NOTFOUND")
      message(FATAL_ERROR "${name} not found in ${path}")
    endif()
    add_library(${module} SHARED IMPORTED)
    set_target_properties(${module} PROPERTIES
      IMPORTED_LOCATION ${${module}_LIBRARY_DIR}
    )
endfunction()
