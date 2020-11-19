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

# libgraph.so
# compiling proto files generates some warnings, use no-unused-variable to suppress them
set(CMAKE_CXX_FLAGS "-Wno-unused-variable ${CMAKE_CXX_FLAGS}")
# add all proto files, generate corresponding .h and .cc files
file(GLOB_RECURSE PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "../../proto/om.proto"
        "../../proto/ge_ir.proto"
        "../../proto/insert_op.proto"
        "../../proto/task.proto"
        "../../proto/fwk_adaper.proto"
        "../../proto/op_mapping_info.proto"
        "../../proto/dump_task.proto"
        )

file(GLOB_RECURSE ONNX_PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "${onnx_INC}/onnx/onnx.proto"
        )

ge_protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})
ge_protobuf_generate(ge PROTO_ONNX_SRCS PROTO_ONNX_HDRS ${ONNX_PROTO_LIST})

# need to remove dependencies on pb files later
file(GLOB SRC_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "*.cc"
        "utils/*.cc"
        "opsproto/*.cc"
        "detail/*.cc"
        "debug/*.cc"
        "option/*.cc"
        "transformer/src/*cc"
        )

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${GE_SOURCE_DIR})
include_directories(${GE_SOURCE_DIR}/src)
include_directories(${GE_SOURCE_DIR}/src/ge)
include_directories(${GE_SOURCE_DIR}/src/common)
include_directories(${GE_SOURCE_DIR}/src/common/graph)
include_directories(${GE_SOURCE_DIR}/inc)
include_directories(${GE_SOURCE_DIR}/inc/framework)
include_directories(${GE_SOURCE_DIR}/inc/external)
include_directories(${GE_SOURCE_DIR}/inc/external/graph)
include_directories(${GE_SOURCE_DIR}/inc/graph)
include_directories(${GE_SOURCE_DIR}/inc/common)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc/ops)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${GE_SOURCE_DIR}/build)

######### libgraph.so #############
add_library(graph SHARED ${SRC_LIST} ${PROTO_SRCS} ${PROTO_ONNX_SRCS})
target_compile_definitions(graph PRIVATE
        DAVINCI_CLOUD
        Werror)
target_link_libraries(graph PRIVATE
        ${PROTOBUF_LIBRARY}
        ${c_sec}
        ${slog}
        ${error_manager}
        rt
        dl)
