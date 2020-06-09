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

# libengine.so
file(GLOB SRC_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "*.cc"
        )

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${GE_SOURCE_DIR})
include_directories(${GE_SOURCE_DIR}/src)
include_directories(${GE_SOURCE_DIR}/src/ge)
include_directories(${GE_SOURCE_DIR}/inc)
include_directories(${GE_SOURCE_DIR}/inc/framework)
include_directories(${GE_SOURCE_DIR}/inc/framework/common)
include_directories(${GE_SOURCE_DIR}/inc/external)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${GE_SOURCE_DIR}/build)

######### libengine.so #############
add_library(engine SHARED ${SRC_LIST})
target_compile_definitions(engine PRIVATE
        REUSE_MEMORY=1
        PLATFORM_CLOUD
        PROTOBUF_INLINE_NOT_IN_HEADERS=0
        Werror)
target_link_libraries(engine
        ${slog}
        rt
        dl)
