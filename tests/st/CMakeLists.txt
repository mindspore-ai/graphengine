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

cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 11)
project(ge_st CXX C)

set(CMAKE_CXX_FLAGS "-O1 -fPIC -Wl,-unresolved-symbols=ignore-in-shared-libs")


file(GLOB_RECURSE RES50_TRAIN_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "resnet50/resnet50_train.cc"
        "resnet50/common.cc"
)

include_directories(${GE_SOURCE_DIR}/inc)
include_directories(${GE_SOURCE_DIR}/inc/graph)
include_directories(${GE_SOURCE_DIR}/inc/framework)
include_directories(${GE_SOURCE_DIR}/inc/external)
include_directories(${GE_SOURCE_DIR}/inc/external/ge)
include_directories(${GE_SOURCE_DIR}/inc/external/graph)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc/ops)
include_directories(/usr/local/HiAI/opp/op_proto/built-in/inc)

add_executable(st_resnet50_train ${RES50_TRAIN_SRCS})
target_link_libraries(st_resnet50_train
        ${PROTOBUF_LIBRARY}
        ge_client_train ge_memory
)