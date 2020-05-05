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

project(STUB_CCE)

set(CMAKE_CXX_STANDARD 11)

include_directories(${GE_SOURCE_DIR}/inc)
include_directories(${GE_SOURCE_DIR}/inc/framework)
include_directories(${GE_SOURCE_DIR}/inc/graph)
include_directories(${GE_SOURCE_DIR}/inc/external)
include_directories(${GE_SOURCE_DIR}/inc/external/graph)
include_directories(${GE_SOURCE_DIR}/src/common)
include_directories(${GE_SOURCE_DIR}/src/common/graph)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc/cce)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc/ops)
include_directories(${GE_SOURCE_DIR}/third_party/securec/include)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${GE_SOURCE_DIR}/third_party/securec/include)
file(GLOB_RECURSE PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
    "${GE_SOURCE_DIR}/src/proto/om.proto"
    "${GE_SOURCE_DIR}/src/proto/ge_ir.proto"
    "${GE_SOURCE_DIR}/src/proto/task.proto"
)

ge_protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

file(GLOB_RECURSE SRCS RELATIVE ${CMAKE_CURRENT_LIST_DIR}
    "${GE_SOURCE_DIR}/src/common/graph/ge_attr_define.cc"
    "${GE_SOURCE_DIR}/src/common/graph/anchor.cc"
    "${GE_SOURCE_DIR}/src/common/graph/ge_attr_value.cc"
    "${GE_SOURCE_DIR}/src/common/graph/buffer.cc"
    "${GE_SOURCE_DIR}/src/common/graph/compute_graph.cc"
    "${GE_SOURCE_DIR}/src/common/graph/graph.cc"
    "${GE_SOURCE_DIR}/src/common/graph/model.cc"
    "${GE_SOURCE_DIR}/src/common/graph/model_serialize.cc"
    "${GE_SOURCE_DIR}/src/common/graph/node.cc"
    "${GE_SOURCE_DIR}/src/common/graph/op_desc.cc"
    "${GE_SOURCE_DIR}/src/common/graph/operator.cc"
    "${GE_SOURCE_DIR}/src/common/graph/operator_factory.cc"
    "${GE_SOURCE_DIR}/src/common/graph/operator_factory_impl.cc"
    "${GE_SOURCE_DIR}/src/common/graph/tensor.cc"
    "${GE_SOURCE_DIR}/src/common/graph/detail/attributes_holder.cc"
    "${GE_SOURCE_DIR}/src/common/graph/utils/anchor_utils.cc"
    "${GE_SOURCE_DIR}/src/common/graph/utils/graph_utils.cc"
    "${GE_SOURCE_DIR}/src/common/graph/utils/node_utils.cc"
    "${GE_SOURCE_DIR}/src/common/graph/utils/op_desc_utils.cc"
    "${GE_SOURCE_DIR}/src/common/graph/utils/type_utils.cc"
    "${GE_SOURCE_DIR}/src/common/graph/op_imp.cc"
    "${GE_SOURCE_DIR}/src/common/graph/shape_refiner.cc"
    "${GE_SOURCE_DIR}/src/common/graph/ge_tensor.cc"
    "${GE_SOURCE_DIR}/src/common/graph/opsproto/opsproto_manager.cc"
)
add_library(cce_ge_stub SHARED src/cce_stub.cc ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(cce_ge_stub protobuf::protobuf)

add_library(cce_stub SHARED ${SRCS} ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(cce_stub protobuf::protobuf)
