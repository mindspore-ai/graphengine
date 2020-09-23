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

# libge_local_engine.so
# add all proto files, generate corresponding .h and .cc files
file(GLOB PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "../../proto/task.proto"
        )

file(GLOB SRC_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "engine/ge_local_engine.cc"
        "ops_kernel_store/*.cc"
        "ops_kernel_store/op/*.cc"
        )

ge_protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${GE_SOURCE_DIR}/ge)
include_directories(${GE_SOURCE_DIR}/inc)
include_directories(${GE_SOURCE_DIR}/inc/external)
include_directories(${GE_SOURCE_DIR}/inc/framework)
include_directories(${GE_SOURCE_DIR}/metadef/inc)
include_directories(${GE_SOURCE_DIR}/metadef/inc/graph)
include_directories(${GE_SOURCE_DIR}/metadef/inc/external)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)

######### libge_local_engine.so #############
add_library(ge_local_engine SHARED ${SRC_LIST} ${PROTO_HDRS})
target_compile_definitions(ge_local_engine PRIVATE Werror)
target_link_libraries(ge_local_engine
        graph
        ${PROTOBUF_LIBRARY}
        ${register}
        ${c_sec}
        ${slog}
        ${runtime})
