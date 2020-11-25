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

project(STUB_CCE)

set(CMAKE_CXX_STANDARD 11)

include_directories(${GE_CODE_DIR}/inc)
include_directories(${GE_CODE_DIR}/inc/framework)
include_directories(${GE_CODE_DIR}/metadef/inc/graph)
include_directories(${GE_CODE_DIR}/inc/external)
include_directories(${GE_CODE_DIR}/metadef/inc/external)
include_directories(${GE_CODE_DIR}/metadef/inc/external/graph)
include_directories(${GE_CODE_DIR}/metadef)
include_directories(${GE_CODE_DIR}/metadef/inc)
include_directories(${GE_CODE_DIR}/metadef/graph)
include_directories(${GE_CODE_DIR}/third_party/fwkacllib/inc)
include_directories(${GE_CODE_DIR}/third_party/fwkacllib/inc/cce)
include_directories(${GE_CODE_DIR}/third_party/fwkacllib/inc/ops)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
set(PROTO_LIST
    "${GE_CODE_DIR}/metadef/proto/om.proto"
    "${GE_CODE_DIR}/metadef/proto/ge_ir.proto"
    "${GE_CODE_DIR}/metadef/proto/task.proto"
)

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

set(SRCS
    "${GE_CODE_DIR}/metadef/graph/ge_attr_define.cc"
    "${GE_CODE_DIR}/metadef/graph/anchor.cc"
    "${GE_CODE_DIR}/metadef/graph/ge_attr_value.cc"
    "${GE_CODE_DIR}/metadef/graph/buffer.cc"
    "${GE_CODE_DIR}/metadef/graph/compute_graph.cc"
    "${GE_CODE_DIR}/metadef/graph/graph.cc"
    "${GE_CODE_DIR}/metadef/graph/model.cc"
    "${GE_CODE_DIR}/metadef/graph/model_serialize.cc"
    "${GE_CODE_DIR}/metadef/graph/node.cc"
    "${GE_CODE_DIR}/metadef/graph/op_desc.cc"
    "${GE_CODE_DIR}/metadef/graph/operator.cc"
    "${GE_CODE_DIR}/metadef/graph/operator_factory.cc"
    "${GE_CODE_DIR}/metadef/graph/operator_factory_impl.cc"
    "${GE_CODE_DIR}/metadef/graph/tensor.cc"
    "${GE_CODE_DIR}/metadef/graph/detail/attributes_holder.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/anchor_utils.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/graph_utils.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/node_utils.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/op_desc_utils.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/type_utils.cc"
    "${GE_CODE_DIR}/metadef/ops/op_imp.cpp"
    "${GE_CODE_DIR}/metadef/graph/shape_refiner.cc"
    "${GE_CODE_DIR}/metadef/graph/ge_tensor.cc"
    "${GE_CODE_DIR}/metadef/graph/opsproto/opsproto_manager.cc"
)
add_library(cce_ge_stub SHARED src/cce_stub.cc ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(cce_ge_stub
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--no-as-needed
    ascend_protobuf
    -Wl,--as-needed
    c_sec
)

add_library(cce_stub SHARED ${SRCS} ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(cce_stub PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--no-as-needed
    ascend_protobuf
    -Wl,--as-needed
    c_sec
)
