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

# libge_runtime.so
# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${GE_SOURCE_DIR}/ge)
include_directories(${GE_SOURCE_DIR}/inc)
include_directories(${GE_SOURCE_DIR}/metadef/inc)
include_directories(${GE_SOURCE_DIR}/metadef/inc/graph)
include_directories(${GE_SOURCE_DIR}/metadef/inc/external)
include_directories(${GE_SOURCE_DIR}/inc/external)
include_directories(${GE_SOURCE_DIR}/inc/framework)
include_directories(${GE_SOURCE_DIR}/inc/framework/common)
include_directories(${GE_SOURCE_DIR}/inc/framework/ge_runtime)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc/cce)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)

######### libge_runtime.so #############
file(GLOB_RECURSE GE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "model_runner.cc"
        "runtime_model.cc"
        "output.cc"
        "task/*.cc"
        )

add_library(ge_runtime SHARED ${GE_SRC_LIST})
target_compile_definitions(ge_runtime PUBLIC
        PROTOBUF_INLINE_NOT_IN_HEADERS=0
        Werror)
target_link_libraries(ge_runtime
        graph
        ${slog}
        ${runtime}
        ${c_sec}
        rt
        dl
        )
