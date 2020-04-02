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

# libge_memosy.a
file(GLOB_RECURSE SRC_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "memory_assigner.cc"
        "graph_mem_assigner.cc"
        "binary_block_mem_assigner.cc"
        "block_mem_assigner.cc"
        "hybrid_mem_assigner.cc"
        "max_block_mem_assigner.cc"
        "var_mem_assign_util.cc"
        )

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${GE_SOURCE_DIR}/src)
include_directories(${GE_SOURCE_DIR}/src/ge)
include_directories(${GE_SOURCE_DIR}/inc)
include_directories(${GE_SOURCE_DIR}/inc/external)
include_directories(${GE_SOURCE_DIR}/inc/external/graph)
include_directories(${GE_SOURCE_DIR}/inc/framework)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc)
include_directories(${GE_SOURCE_DIR}/third_party/securec/include)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)

######### libge_memory.a #############
add_library(ge_memory STATIC ${SRC_LIST})
target_compile_definitions(ge_memory PRIVATE
        Werror
        DAVINCI_CLOUD)
target_link_libraries(ge_memory
        graph
        ge_common
        ${PROTOBUF_LIBRARY}
        ${c_sec}
        ${slog}
        rt
        dl)
