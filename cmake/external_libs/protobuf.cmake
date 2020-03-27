if (NOT TARGET protobuf::libprotobuf)
graphengine_add_pkg(protobuf
        VER 3.8.0
        HEAD_ONLY ./
        URL https://github.com/protocolbuffers/protobuf/archive/v3.8.0.tar.gz
        MD5 3d9e32700639618a4d2d342c99d4507a)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "Disahble protobuf test")
set(protobuf_BUILD_SHARED_LIBS ON CACHE BOOL "Gen shared library")
set(_ms_tmp_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
string(REPLACE " -Wall" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE " -Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(PROTOBUF_CMAKE_FILE "${protobuf_DIRPATH}/cmake/libprotobuf.cmake" )
FILE(READ ${PROTOBUF_CMAKE_FILE} GE_MR_PROTOBUF_CMAKE)
STRING(REPLACE "VERSION \${protobuf_VERSION}" "VERSION 19" GE_MR_PROTOBUF_CMAKE_V19 "${GE_MR_PROTOBUF_CMAKE}" )
FILE(WRITE ${PROTOBUF_CMAKE_FILE} "${GE_MR_PROTOBUF_CMAKE_V19}")

add_subdirectory(${protobuf_DIRPATH}/cmake ${protobuf_DIRPATH}/build)
set(CMAKE_CXX_FLAGS ${_ms_tmp_CMAKE_CXX_FLAGS})
endif()

set(PROTOBUF_LIBRARY protobuf::libprotobuf)
include_directories(${protobuf_DIRPATH}/src)
add_library(ge_protobuf::protobuf ALIAS libprotobuf)

function(ge_protobuf_generate comp c_var h_var)
    if(NOT ARGN)
        message(SEND_ERROR "Error: ge_protobuf_generate() called without any proto files")
        return()
    endif()

    set(${c_var})
    set(${h_var})

    foreach(file ${ARGN})
        get_filename_component(abs_file ${file} ABSOLUTE)
        get_filename_component(file_name ${file} NAME_WE)
        get_filename_component(file_dir ${abs_file} PATH)

        list(APPEND ${c_var} "${CMAKE_BINARY_DIR}/proto/${comp}/proto/${file_name}.pb.cc")
        list(APPEND ${h_var} "${CMAKE_BINARY_DIR}/proto/${comp}/proto/${file_name}.pb.h")

        add_custom_command(
                OUTPUT "${CMAKE_BINARY_DIR}/proto/${comp}/proto/${file_name}.pb.cc"
                "${CMAKE_BINARY_DIR}/proto/${comp}/proto/${file_name}.pb.h"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/proto/${comp}/proto"
                COMMAND protobuf::protoc -I${file_dir} --cpp_out=${CMAKE_BINARY_DIR}/proto/${comp}/proto ${abs_file}
                DEPENDS protobuf::protoc ${abs_file}
                COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM )
    endforeach()

    set_source_files_properties(${${c_var}} ${${h_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)

endfunction()
