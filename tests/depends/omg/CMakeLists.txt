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

#cmake_minimum_required(VERSION 2.8)

project(OMG_CCE)

set(CMAKE_CXX_STANDARD 11)

include_directories(${GE_CODE_DIR}/third_party/fwkacllib/inc)
include_directories(${GE_CODE_DIR}/third_party/fwkacllib/inc/cce)
include_directories(${GE_CODE_DIR}/inc)
include_directories(${GE_CODE_DIR}/metadef/inc)
include_directories(${GE_CODE_DIR}/inc/framework)
include_directories(${GE_CODE_DIR}/metadef/inc/graph)
include_directories(${GE_CODE_DIR}/inc/external)
include_directories(${GE_CODE_DIR}/metadef/inc/external)
include_directories(${GE_CODE_DIR}/metadef/inc/external/graph)
include_directories(${GE_CODE_DIR}/ge)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
set(PROTO_LIST
    "${GE_CODE_DIR}/metadef/proto/om.proto"
    "${GE_CODE_DIR}/metadef/proto/task.proto"
)

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

set(SRCS
#   "${GE_CODE_DIR}/src/ge/common/util.cc"
    "src/omg_stub.cc"
)

add_library(omg_stub SHARED ${SRCS} ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(omg_stub PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--no-as-needed
    ascend_protobuf
    -Wl,--as-needed
    c_sec
    json
)
