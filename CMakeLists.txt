# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

cmake_minimum_required(VERSION 3.14)
project (GraphEngine[CXX])
set(CMAKE_CXX_STANDARD 17)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${PROJECT_BINARY_DIR})
set(GE_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})
set(GE_PROTO_DIR ${GE_SOURCE_DIR}/src)

if (NOT BUILD_PATH)
    set(BUILD_PATH "${CMAKE_SOURCE_DIR}/build")
endif()
# architecture: aarch64 or x86_64
message(STATUS "System architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
# system: euleros or ubuntu
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    execute_process(
            COMMAND bash "-c" "cat /etc/os-release | grep ^ID= | awk -F '=' '{print $2}'"
            OUTPUT_VARIABLE SYSTEM_TYPE
    )
    MESSAGE(STATUS "System type: ${SYSTEM_TYPE}.")
endif()

# download json headers, rather than whole repository
include(${GE_SOURCE_DIR}/cmake/ge_utils.cmake)
include(${GE_SOURCE_DIR}/cmake/external_libs/json.cmake)
include(${GE_SOURCE_DIR}/cmake/external_libs/eigen.cmake)
include(${GE_SOURCE_DIR}/cmake/external_libs/gtest.cmake)
include(${GE_SOURCE_DIR}/cmake/external_libs/protobuf.cmake)
include(${GE_SOURCE_DIR}/cmake/external_libs/onnx.cmake)
include(${GE_SOURCE_DIR}/cmake/external_libs/securec.cmake)
set(CMAKE_SKIP_RPATH TRUE)

# for CPU/GPU mode, find c_sec and slog from local prebuild
if(NOT ENABLE_D AND NOT GE_ONLY)
    set(GE_PREBUILD_PATH ${GE_SOURCE_DIR}/third_party/prebuild/${CMAKE_HOST_SYSTEM_PROCESSOR})
    find_library(slog libslog.so ${GE_PREBUILD_PATH})
# if D_LINK_PATH is set in environment variables, search libraries in given path
elseif(DEFINED ENV{D_LINK_PATH})
    # D_LINK_PATH is set
    set(GE_LIB_PATH $ENV{D_LINK_PATH})
    set(GE_SYS_ARCH "")
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
        # x86 ubuntu
        set(GE_SYS_ARCH "x86_64")
    elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
        # arm euleros
        set(GE_SYS_ARCH "aarch64")
    else()
        message(FATAL_ERROR "Running on a unsupported architecture: ${SYSTEM_TYPE}, build terminated")
    endif()
    set(GE_LIB_PATH ${GE_LIB_PATH}/${GE_SYS_ARCH})
    find_library(slog libslog.so ${GE_LIB_PATH})
    find_library(mmpa libmmpa.so ${GE_LIB_PATH})
    find_library(runtime libruntime.so ${GE_LIB_PATH})
    find_library(msprof libmsprof.so ${GE_LIB_PATH})
    find_library(register libregister.so ${GE_LIB_PATH})
    find_library(hccl libhccl.so ${GE_LIB_PATH})
    find_library(resource libresource.so ${GE_LIB_PATH})
    find_library(error_manager liberror_manager.so ${GE_LIB_PATH})
else()
    # Ascend mode
    if(DEFINED ENV{ASCEND_CUSTOM_PATH})
        set(ASCEND_DIR $ENV{ASCEND_CUSTOM_PATH})
    else()
        set(ASCEND_DIR /usr/local/Ascend)
    endif()
    set(ASCEND_DRIVER_DIR ${ASCEND_DIR}/driver/lib64/common)
    set(ASCEND_RUNTIME_DIR ${ASCEND_DIR}/fwkacllib/lib64)
    find_library(slog libslog.so ${ASCEND_DRIVER_DIR})
    find_library(mmpa libmmpa.so ${ASCEND_DRIVER_DIR})
    find_library(msprof libmsprof.so ${ASCEND_DRIVER_DIR})

    find_library(hccl libhccl.so ${ASCEND_RUNTIME_DIR})
    find_library(runtime libruntime.so ${ASCEND_RUNTIME_DIR})
    find_library(register libregister.so ${ASCEND_RUNTIME_DIR})
    find_library(resource libresource.so ${ASCEND_RUNTIME_DIR})
    find_library(error_manager liberror_manager.so ${ASCEND_RUNTIME_DIR})
endif()

# add compile flags
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message("Build in Debug mode")
    set(CMAKE_C_FLAGS "-O0 -g -Wall -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -pipe -fPIC ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -pipe -fPIC ${CMAKE_CXX_FLAGS}")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -rdynamic")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic")
    endif()
else()
    set(CMAKE_C_FLAGS "-O2 -Wall -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -pipe ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -pipe ${CMAKE_CXX_FLAGS}")
endif ()

# force __FILE__ to show relative path of file, from source directory, as cmake project makes __FILE__ absolute directory
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FILE__='\"$(subst $(realpath ${CMAKE_SOURCE_DIR})/,,$(abspath $<))\"' -Wno-builtin-macro-redefined")

# compile libraries from following directories
# libgraph is compiled in any situation
add_subdirectory(${GE_SOURCE_DIR}/src/common/graph)
if(ENABLE_D)
    # if MindSpore compiles in D mode, compile the following libraries
    add_subdirectory(${GE_SOURCE_DIR}/src/ge/common)
    add_subdirectory(${GE_SOURCE_DIR}/src/ge/ge_runtime)
elseif(GE_ONLY)
    # standalone GraphEngine compiles all following libraries
    add_subdirectory(${GE_SOURCE_DIR}/src/ge/common)
    add_subdirectory(${GE_SOURCE_DIR}/src/ge/ge_runtime)
    add_subdirectory(${GE_SOURCE_DIR}/src/ge/ge_local_engine)
    add_subdirectory(${GE_SOURCE_DIR}/src/ge/graph/build/memory)
    add_subdirectory(${GE_SOURCE_DIR}/src/ge/)
    add_subdirectory(${GE_SOURCE_DIR}/src/ge/executor)
    add_subdirectory(${GE_SOURCE_DIR}/src/ge/client)
    add_subdirectory(${GE_SOURCE_DIR}/src/ge/plugin/engine)
endif()

# if (ENABLE_GE_COV OR ENABLE_GE_UT OR ENABLE_GE_ST)
#     add_subdirectory(tests)
# endif()

