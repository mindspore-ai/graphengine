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

project(ut_libgraph)

set(CMAKE_CXX_STANDARD 11)

file(GLOB_RECURSE PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "${GE_SOURCE_DIR}/src/proto/om.proto"
        "${GE_SOURCE_DIR}/src/proto/ge_ir.proto"
        "${onnx_INC}/onnx/onnx.proto"
)

ge_protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

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
include_directories(${GE_SOURCE_DIR}/inc/ops)
include_directories(${GE_SOURCE_DIR}/third_party/securec/include)
include_directories(${GE_SOURCE_DIR}/third_party/json/include)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc/ops)
include_directories(${GE_SOURCE_DIR}/third_party/fwkacllib/inc)
include_directories(/usr/local/HiAI/opp/op_proto/built-in/inc)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)

file(GLOB_RECURSE UT_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "testcase/ge_graph/ge_anchor_utils_unittest.cc"
    "testcase/ge_graph/ge_def_type_unittest.cc"
    "testcase/ge_graph/ge_graph_anchor_unittest.cc"
    "testcase/ge_graph/ge_model_serialize_unittest.cc"
    "testcase/ge_graph/ge_node_unittest.cc"
    "testcase/ge_graph/ge_opdesc_unittest.cc"
    "testcase/ge_graph/ge_tensor_unittest.cc"
    "testcase/ge_graph/graph_builder_utils.cc"
    "testcase/ge_graph/ge_reg_unittest.cc"
    "testcase/ge_graph/ge_format_refiner_unittest.cc"
    "testcase/ge_graph/ge_opsproto_manager_unittest.cc"
    "testcase/ge_graph/ge_operator_unittest.cc"
    "testcase/ge_graph/ge_model_unittest.cc"
)

file(GLOB_RECURSE SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "${GE_SOURCE_DIR}/src/common/graph/option/ge_local_context.cc"
    "${GE_SOURCE_DIR}/src/common/graph/option/ge_context.cc"
    "${GE_SOURCE_DIR}/src/common/graph/anchor.cc"
    "${GE_SOURCE_DIR}/src/common/graph/ge_attr_value.cc"
    "${GE_SOURCE_DIR}/src/common/graph/attr_value.cc"
    "${GE_SOURCE_DIR}/src/common/graph/buffer.cc"
    "${GE_SOURCE_DIR}/src/common/graph/compute_graph.cc"
    "${GE_SOURCE_DIR}/src/common/graph/ge_attr_define.cc"
    "${GE_SOURCE_DIR}/src/common/graph/graph.cc"
    "${GE_SOURCE_DIR}/src/common/graph/model.cc"
    "${GE_SOURCE_DIR}/src/common/graph/model_serialize.cc"
    "${GE_SOURCE_DIR}/src/common/graph/node.cc"
    "${GE_SOURCE_DIR}/src/common/graph/op_desc.cc"
    "${GE_SOURCE_DIR}/src/common/graph/operator.cc"
    "${GE_SOURCE_DIR}/src/common/graph/operator_reg.cc"
    "${GE_SOURCE_DIR}/src/common/graph/operator_factory.cc"
    "${GE_SOURCE_DIR}/src/common/graph/operator_factory_impl.cc"
    "${GE_SOURCE_DIR}/src/common/graph/range_vistor.cc"
    "${GE_SOURCE_DIR}/src/common/graph/tensor.cc"
    "${GE_SOURCE_DIR}/src/common/graph/ge_tensor.cc"
    "${GE_SOURCE_DIR}/src/common/graph/shape_refiner.cc"
    "${GE_SOURCE_DIR}/src/common/graph/format_refiner.cc"
    "${GE_SOURCE_DIR}/src/common/graph/inference_context.cc"
    "${GE_SOURCE_DIR}/src/common/graph/detail/attributes_holder.cc"
    "${GE_SOURCE_DIR}/src/common/graph/utils/anchor_utils.cc"
    "${GE_SOURCE_DIR}/src/common/graph/utils/graph_utils.cc"
    "${GE_SOURCE_DIR}/src/common/graph/utils/node_utils.cc"
    "${GE_SOURCE_DIR}/src/common/graph/utils/op_desc_utils.cc"
    "${GE_SOURCE_DIR}/src/common/graph/utils/type_utils.cc"
    "${GE_SOURCE_DIR}/src/common/graph/utils/ge_ir_utils.cc"
    "${GE_SOURCE_DIR}/src/common/graph/utils/tensor_utils.cc"
    "${GE_SOURCE_DIR}/src/common/ops/op_imp.cc"
    "${GE_SOURCE_DIR}/src/common/graph/opsproto/opsproto_manager.cc"
)

add_executable(ut_libgraph ${UT_FILES} ${SRC_FILES} ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(ut_libgraph graphengine::gtest graphengine::gtest_main slog_stub ge_protobuf::protobuf ${c_sec} rt dl)
