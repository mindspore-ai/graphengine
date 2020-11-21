if (NOT TARGET ge_protobuf::ascend_protobuf)
if (AS_MS_COMP)
    set(protobuf_USE_STATIC_LIBS OFF)
    set(protobuf_CMAKE_OPTION -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=ON -DLIB_PREFIX=ascend_
        -DCMAKE_C_FLAGS=\"-Dgoogle=ascend_private\" -DCMAKE_CXX_FLAGS=\"-Dgoogle=ascend_private\")
else ()
    set(protobuf_USE_STATIC_LIBS ON)
    set(protobuf_CMAKE_OPTION -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -DLIB_PREFIX=ascend_)
endif ()
set(ge_protobuf_CXXFLAGS "-Wno-maybe-uninitialized -Wno-unused-parameter -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -O2 -D_GLIBCXX_USE_CXX11_ABI=0")
set(ge_protobuf_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
set(_ge_tmp_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
string(REPLACE " -Wall" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE " -Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

if (ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/protobuf_source/repository/archive/v3.8.0.tar.gz")
    set(MD5 "eba86ae9f07ba5cfbaf8af3bc4e84236")
else()
    set(REQ_URL "https://github.com/protocolbuffers/protobuf/archive/v3.8.0.tar.gz")
    set(MD5 "3d9e32700639618a4d2d342c99d4507a")
endif ()

graphengine_add_pkg(ge_protobuf
        VER 3.8.0
        LIBS ascend_protobuf
        EXE protoc
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_PATH ../cmake/
        CMAKE_OPTION ${protobuf_CMAKE_OPTION})
set(CMAKE_CXX_FLAGS ${_ge_tmp_CMAKE_CXX_FLAGS})
endif()
add_library(graphengine::protobuf ALIAS ge_protobuf::ascend_protobuf)
set(PROTOBUF_LIBRARY ge_protobuf::ascend_protobuf)
include_directories(${ge_protobuf_INC})
include_directories(${ge_protobuf_DIRPATH}/src)

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
                COMMAND ge_protobuf::protoc -I${file_dir} --cpp_out=${CMAKE_BINARY_DIR}/proto/${comp}/proto ${abs_file}
                DEPENDS ge_protobuf::protoc ${abs_file}
                COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM )
    endforeach()

    set_source_files_properties(${${c_var}} ${${h_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)

endfunction()
