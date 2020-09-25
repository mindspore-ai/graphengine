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

# libge_client.so
# add all proto files, generate corresponding .h and .cc files
set(CMAKE_CXX_FLAGS "-Wno-unused-variable ${CMAKE_CXX_FLAGS}")
file(GLOB PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "../proto/ge_api.proto"
        )

file(GLOB PROTO_HEADER_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "../proto/ge_ir.proto"
        "../proto/task.proto"
        "../proto/om.proto"
        "../proto/insert_op.proto"
        )

file(GLOB SRC_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "ge_api.cc"
        "ge_prof.cc"
        )

ge_protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})
ge_protobuf_generate(ge PROTO_HEADER_SRCS PROTO_HEADER_HDRS ${PROTO_HEADER_LIST})

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${GE_SOURCE_DIR}/ge)
include_directories(${GE_SOURCE_DIR}/inc)
include_directories(${GE_SOURCE_DIR}/inc/external)
include_directories(${GE_SOURCE_DIR}/inc/common)
include_directories(${GE_SOURCE_DIR}/inc/framework)
include_directories(${GE_SOURCE_DIR}/metadef/inc)
include_directories(${GE_SOURCE_DIR}/metadef/inc/external/graph)
include_directories(${GE_SOURCE_DIR}/metadef/inc/graph)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc/cce)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)

############ libge_client.so ################
add_library(ge_client SHARED ${SRC_LIST} ${PROTO_SRCS} ${PROTO_HEADER_HDRS})
target_compile_definitions(ge_client PRIVATE
        Werror
        PROTOBUF_INLINE_NOT_IN_HEADERS=0
        REUSE_MEMORY=1
        PLATFORM_CLOUD)
target_link_libraries(ge_client
        graph
        ge_compiler
        ge_common
        ${PROTOBUF_LIBRARY}
        ${register}
        ${c_sec}
        ${slog}
        ${mmpa}
        ${runtime}
        ${msprof}
        rt
        dl)
