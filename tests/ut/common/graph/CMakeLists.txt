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

set(PROTO_LIST
    "${GE_CODE_DIR}/metadef/proto/om.proto"
    "${GE_CODE_DIR}/metadef/proto/ge_ir.proto"
    "${GE_CODE_DIR}/metadef/proto/onnx/ge_onnx.proto"
)

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${GE_CODE_DIR})
include_directories(${GE_CODE_DIR}/metadef)
include_directories(${GE_CODE_DIR}/metadef/graph)
include_directories(${GE_CODE_DIR}/inc)
include_directories(${GE_CODE_DIR}/inc/external)
include_directories(${GE_CODE_DIR}/metadef/inc/external)
include_directories(${GE_CODE_DIR}/metadef/inc/external/graph)
include_directories(${GE_CODE_DIR}/metadef/inc)
include_directories(${GE_CODE_DIR}/metadef/inc/graph)
include_directories(${GE_CODE_DIR}/metadef/inc/common)
include_directories(${GE_CODE_DIR}/metadef/third_party)
include_directories(${GE_CODE_DIR}/metadef/third_party/transformer/inc)
include_directories(${GE_CODE_DIR}/third_party/fwkacllib/inc)
include_directories(${GE_CODE_DIR}/third_party/fwkacllib/inc/ops)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/ge/proto)

set(UT_FILES
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

set(SRC_FILES
    "${GE_CODE_DIR}/metadef/graph/option/ge_local_context.cc"
    "${GE_CODE_DIR}/metadef/graph/option/ge_context.cc"
    "${GE_CODE_DIR}/metadef/graph/anchor.cc"
    "${GE_CODE_DIR}/metadef/graph/ge_attr_value.cc"
    "${GE_CODE_DIR}/metadef/graph/attr_value.cc"
    "${GE_CODE_DIR}/metadef/graph/buffer.cc"
    "${GE_CODE_DIR}/metadef/graph/aligned_ptr.cc"
    "${GE_CODE_DIR}/metadef/graph/compute_graph.cc"
    "${GE_CODE_DIR}/metadef/graph/ge_attr_define.cc"
    "${GE_CODE_DIR}/metadef/graph/graph.cc"
    "${GE_CODE_DIR}/metadef/graph/gnode.cc"
    "${GE_CODE_DIR}/metadef/graph/ascend_string.cc"
    "${GE_CODE_DIR}/metadef/graph/model.cc"
    "${GE_CODE_DIR}/metadef/graph/model_serialize.cc"
    "${GE_CODE_DIR}/metadef/graph/node.cc"
    "${GE_CODE_DIR}/metadef/graph/op_desc.cc"
    "${GE_CODE_DIR}/metadef/graph/operator.cc"
    "${GE_CODE_DIR}/metadef/graph/operator_factory.cc"
    "${GE_CODE_DIR}/metadef/graph/operator_factory_impl.cc"
    "${GE_CODE_DIR}/metadef/graph/tensor.cc"
    "${GE_CODE_DIR}/metadef/graph/types.cc"
    "${GE_CODE_DIR}/metadef/graph/ge_tensor.cc"
    "${GE_CODE_DIR}/metadef/graph/shape_refiner.cc"
    "${GE_CODE_DIR}/metadef/graph/format_refiner.cc"
    "${GE_CODE_DIR}/metadef/graph/inference_context.cc"
    "${GE_CODE_DIR}/metadef/graph/detail/attributes_holder.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/anchor_utils.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/graph_utils.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/node_utils.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/op_desc_utils.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/type_utils.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/ge_ir_utils.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/tensor_utils.cc"
    "${GE_CODE_DIR}/metadef/ops/op_imp.cpp"
    "${GE_CODE_DIR}/metadef/graph/opsproto/opsproto_manager.cc"
    "${GE_CODE_DIR}/metadef/graph/runtime_inference_context.cc"
    "${GE_CODE_DIR}/metadef/graph/ref_relation.cc"
    "${GE_CODE_DIR}/metadef/third_party/transformer/src/transfer_shape_according_to_format.cc"
    "${GE_CODE_DIR}/metadef/third_party/transformer/src/axis_util.cc"
    "${GE_CODE_DIR}/metadef/third_party/transformer/src/expand_dimension.cc"
    "${GE_CODE_DIR}/metadef/graph/utils/transformer_utils.cc"
)

#add_executable(ut_libgraph ${UT_FILES} ${SRC_FILES} ${PROTO_SRCS} ${PROTO_HDRS})
add_executable(ut_libgraph ${UT_FILES} ${SRC_FILES} ${PROTO_SRCS} ${PROTO_HDRS})

target_compile_options(ut_libgraph PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
)

target_compile_definitions(ut_libgraph PRIVATE
    google=ascend_private
)

target_link_libraries(ut_libgraph
    $<BUILD_INTERFACE:intf_pub>
    gtest
    gtest_main
    slog_stub
    ascend_protobuf
    c_sec
    error_manager_stub
    mmpa_stub
    -lrt
    -ldl
    -lgcov
)
