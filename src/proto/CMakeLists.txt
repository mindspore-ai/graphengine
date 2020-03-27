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

cmake_minimum_required(VERSION 2.8)

project(GeProto[CXX])

set(CMAKE_CXX_STANDARD 11)

# add all proto files, generate corresponding .h and .cc files
file(GLOB_RECURSE PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "om.proto"
        "ge_ir.proto"
        "insert_op.proto"
        "task.proto"
        "fwk_adapter.proto"
        "op_mapping_info.proto"
        )

file(GLOB_RECURSE ONNX_PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "${onnx_INC}/onnx/onnx.proto"
        )

ge_protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})
ge_protobuf_generate(ge PROTO_ONNX_SRCS PROTO_ONNX_HDRS ${ONNX_PROTO_LIST})

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${GE_SOURCE_DIR}/src)
include_directories(${GE_SOURCE_DIR}/src/common)
include_directories(${GE_SOURCE_DIR}/src/common/graph)
include_directories(${GE_SOURCE_DIR}/inc)
include_directories(${GE_SOURCE_DIR}/inc/external)
include_directories(${GE_SOURCE_DIR}/inc/external/graph)
include_directories(${GE_SOURCE_DIR}/inc/graph)
include_directories(${GE_SOURCE_DIR}/inc/common)
include_directories(${GE_SOURCE_DIR}/third_party/securec/include)
include_directories(${GE_SOURCE_DIR}/third_party/json/include)
include_directories(${GE_SOURCE_DIR}/third_party/protobuf/src)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)

set(CMAKE_CXX_FLAGS "-Wno-unused-variable ${CMAKE_CXX_FLAGS}")

######### libproto_common.so #############
add_library(proto_common STATIC ${PROTO_SRCS} ${PROTO_ONNX_SRCS})
target_compile_definitions(proto_common PRIVATE
        DAVINCI_CLOUD)
target_link_libraries(proto_common
        ${PROTOBUF_LIBRARY}
        ${c_sec}
        ${slog}
        rt
        dl)
