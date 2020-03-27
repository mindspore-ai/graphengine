/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#define protected public
#define private public
#include "graph/passes/folding_kernel/expanddims_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/fp16_t.h"
#include "common/ge_inner_error_codes.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "inc/kernel_factory.h"
#undef protected
#undef private

using namespace std;
using namespace testing;
using namespace ge;
using namespace domi;

class UTEST_graph_passes_folding_kernel_expanddims_kernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  template <typename inner_data_type, DataType data_type, typename inner_dim_type, DataType dim_type>
  void TestExpandDims(vector<int64_t> &data_vec, vector<inner_dim_type> &dim_value_vec, vector<int64_t> &result) {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

    ge::OpDescPtr data_op_desc = make_shared<ge::OpDesc>("data", CONSTANTOP);
    int64_t dims_size = 1;
    for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
    vector<inner_data_type> data_value_vec(dims_size, 1);
    GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, data_type);
    GeTensorPtr data_tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                    data_value_vec.size() * sizeof(inner_data_type));
    OpDescUtils::SetWeights(data_op_desc, data_tensor);
    data_op_desc->AddOutputDesc(data_tensor_desc);
    NodePtr data_node = graph->AddNode(data_op_desc);
    data_node->Init();

    // add dim node
    ge::OpDescPtr dim_op_desc = make_shared<ge::OpDesc>("dim", CONSTANTOP);
    GeTensorDesc dim_tensor_desc(ge::GeShape(), FORMAT_NCHW, dim_type);
    GeTensorPtr dim_tensor = make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(),
                                                   dim_value_vec.size() * sizeof(inner_dim_type));
    OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
    dim_op_desc->AddOutputDesc(dim_tensor_desc);
    NodePtr dim_node = graph->AddNode(dim_op_desc);
    dim_node->Init();

    // add expanddims node
    OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Expanddims", EXPANDDIMS);
    expanddim_op_desc->AddInputDesc(data_tensor_desc);
    expanddim_op_desc->AddInputDesc(dim_tensor_desc);
    NodePtr op_node = graph->AddNode(expanddim_op_desc);
    op_node->Init();

    // add edge
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(dim_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(1));

    shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(EXPANDDIMS);
    Status status = kernel->Compute(op_node);
    EXPECT_EQ(ge::SUCCESS, status);
  }

  template <typename inner_data_type, DataType data_type, typename inner_dim_type, DataType dim_type, Format format>
  void TestInvalidExpandDims(vector<int64_t> &data_vec, vector<inner_dim_type> &dim_value_vec,
                             vector<int64_t> &result) {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

    ge::OpDescPtr data_op_desc = make_shared<ge::OpDesc>("data", CONSTANTOP);
    int64_t dims_size = 1;
    for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
    vector<inner_data_type> data_value_vec(dims_size, 1);
    GeTensorDesc data_tensor_desc(GeShape(data_vec), format, data_type);
    GeTensorPtr data_tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                    data_value_vec.size() * sizeof(inner_data_type));
    OpDescUtils::SetWeights(data_op_desc, data_tensor);
    data_op_desc->AddOutputDesc(data_tensor_desc);
    NodePtr data_node = graph->AddNode(data_op_desc);
    data_node->Init();

    // add dim node
    ge::OpDescPtr dim_op_desc = make_shared<ge::OpDesc>("dim", CONSTANTOP);
    GeTensorDesc dim_tensor_desc(ge::GeShape(), format, dim_type);
    GeTensorPtr dim_tensor = make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(),
                                                   dim_value_vec.size() * sizeof(inner_dim_type));
    OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
    dim_op_desc->AddOutputDesc(dim_tensor_desc);
    NodePtr dim_node = graph->AddNode(dim_op_desc);
    dim_node->Init();

    // add expanddims node
    OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Expanddims", EXPANDDIMS);
    expanddim_op_desc->AddInputDesc(data_tensor_desc);
    expanddim_op_desc->AddInputDesc(dim_tensor_desc);
    NodePtr op_node = graph->AddNode(expanddim_op_desc);
    op_node->Init();

    // add edge
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(dim_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(1));

    shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(EXPANDDIMS);
    Status status = kernel->Compute(op_node);
    EXPECT_NE(ge::SUCCESS, status);

    vector<ConstGeTensorPtr> input = {data_tensor};
    vector<GeTensorPtr> v_output;
    status = kernel->Compute(op_node->GetOpDesc(), input, v_output);
    EXPECT_EQ(NOT_CHANGED, status);
  }
};

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, int8_int32_case_0) {
  vector<int64_t> data_vec = {2, 3};
  vector<int32_t> dim_value_vec = {0};
  vector<int64_t> result = {1, 2, 3};
  TestExpandDims<int8_t, DT_INT8, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, int8_int32_case_1) {
  vector<int64_t> data_vec = {2, 3};
  vector<int32_t> dim_value_vec = {1};
  vector<int64_t> result = {2, 1, 3};
  TestExpandDims<int8_t, DT_INT8, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, int8_int32_case_2) {
  vector<int64_t> data_vec = {2, 3};
  vector<int32_t> dim_value_vec = {2};
  vector<int64_t> result = {2, 3, 1};
  TestExpandDims<int8_t, DT_INT8, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, int8_int32_negative_case_1) {
  vector<int64_t> data_vec = {2, 3};
  vector<int32_t> dim_value_vec = {-3};
  vector<int64_t> result = {1, 2, 3};
  TestExpandDims<int8_t, DT_INT8, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, int8_int32_negative_case_2) {
  vector<int64_t> data_vec = {2, 3};
  vector<int32_t> dim_value_vec = {-2};
  vector<int64_t> result = {2, 1, 3};
  TestExpandDims<int8_t, DT_INT8, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, int8_int32_negative_case_3) {
  vector<int64_t> data_vec = {2, 3};
  vector<int32_t> dim_value_vec = {-1};
  vector<int64_t> result = {2, 3, 1};
  TestExpandDims<int8_t, DT_INT8, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, int16_int32) {
  vector<int64_t> data_vec = {3};
  vector<int32_t> dim_value_vec = {-1};
  vector<int64_t> result = {3, 1};
  TestExpandDims<int16_t, DT_INT16, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, int32_int32) {
  vector<int64_t> data_vec = {3, 3, 3, 5, 6};
  vector<int32_t> dim_value_vec = {3};
  vector<int64_t> result = {3, 3, 3, 1, 5, 6};
  TestExpandDims<int32_t, DT_INT32, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, int64_int32) {
  vector<int64_t> data_vec = {6, 1, 12, 3, 4, 56, 7};
  vector<int32_t> dim_value_vec = {7};
  vector<int64_t> result = {6, 1, 12, 3, 4, 56, 7, 1};
  TestExpandDims<int64_t, DT_INT64, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, uint8_int32) {
  vector<int64_t> data_vec = {2, 3};
  vector<int32_t> dim_value_vec = {-1};
  vector<int64_t> result = {2, 3, 1};
  TestExpandDims<uint8_t, DT_UINT8, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, uint16_int32) {
  vector<int64_t> data_vec = {3};
  vector<int32_t> dim_value_vec = {-1};
  vector<int64_t> result = {3, 1};
  TestExpandDims<uint16_t, DT_UINT16, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, uint32_int32) {
  vector<int64_t> data_vec = {3, 3, 3, 5, 6};
  vector<int32_t> dim_value_vec = {3};
  vector<int64_t> result = {3, 3, 3, 1, 5, 6};
  TestExpandDims<uint32_t, DT_UINT32, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, uint64_int32) {
  vector<int64_t> data_vec = {6, 1, 12, 3, 4, 56, 7};
  vector<int32_t> dim_value_vec = {7};
  vector<int64_t> result = {6, 1, 12, 3, 4, 56, 7, 1};
  TestExpandDims<uint64_t, DT_UINT64, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, fp16_int32) {
  vector<int64_t> data_vec = {6, 1, 12, 3, 4, 56, 7};
  vector<int32_t> dim_value_vec = {7};
  vector<int64_t> result = {6, 1, 12, 3, 4, 56, 7, 1};
  TestExpandDims<fp16_t, DT_FLOAT16, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, float_int32) {
  vector<int64_t> data_vec = {1, 1, 1, 11, 1, 1, 1};
  vector<int32_t> dim_value_vec = {7};
  vector<int64_t> result = {1, 1, 1, 11, 1, 1, 1, 1};
  TestExpandDims<float, DT_FLOAT, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, double_int32) {
  vector<int64_t> data_vec = {1, 1, 1, 11, 1, 1, 1};
  vector<int32_t> dim_value_vec = {7};
  vector<int64_t> result = {1, 1, 1, 11, 1, 1, 1, 1};
  TestExpandDims<double, DT_DOUBLE, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, double_int64) {
  vector<int64_t> data_vec = {1, 1, 1, 11, 1, 1, 1};
  vector<int64_t> dim_value_vec = {7};
  vector<int64_t> result = {1, 1, 1, 11, 1, 1, 1, 1};
  TestExpandDims<double, DT_DOUBLE, int64_t, DT_INT64>(data_vec, dim_value_vec, result);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, bool_int64) {
  vector<int64_t> data_vec = {1, 1, 1, 11, 1, 1, 1};
  vector<int64_t> dim_value_vec = {7};
  vector<int64_t> result = {1, 1, 1, 11, 1, 1, 1, 1};

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr data_op_desc = make_shared<ge::OpDesc>("data", CONSTANTOP);
  int64_t dims_size = 1;
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<uint8_t> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_BOOL);
  GeTensorPtr data_tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                  data_value_vec.size() * sizeof(uint8_t));
  OpDescUtils::SetWeights(data_op_desc, data_tensor);
  data_op_desc->AddOutputDesc(data_tensor_desc);
  NodePtr data_node = graph->AddNode(data_op_desc);
  data_node->Init();

  // add dim node
  ge::OpDescPtr dim_op_desc = make_shared<ge::OpDesc>("dim", CONSTANTOP);
  GeTensorDesc dim_tensor_desc(ge::GeShape(), FORMAT_NCHW, DT_INT64);
  GeTensorPtr dim_tensor =
      make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(), dim_value_vec.size() * sizeof(int64_t));
  OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
  dim_op_desc->AddOutputDesc(dim_tensor_desc);
  NodePtr dim_node = graph->AddNode(dim_op_desc);
  dim_node->Init();

  // add expanddims node
  OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Expanddims", EXPANDDIMS);
  expanddim_op_desc->AddInputDesc(data_tensor_desc);
  expanddim_op_desc->AddInputDesc(dim_tensor_desc);
  NodePtr op_node = graph->AddNode(expanddim_op_desc);
  op_node->Init();

  // add edge
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(dim_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(1));

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(EXPANDDIMS);
  Status status = kernel->Compute(op_node);
  EXPECT_EQ(ge::SUCCESS, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, invalid_format) {
  vector<int64_t> data_vec = {2, 3, 4};
  vector<int32_t> dim_value_vec = {0};
  vector<int64_t> result = {1, 2, 3, 4};

  TestInvalidExpandDims<int32_t, DT_INT32, int32_t, DT_INT32, FORMAT_FRACTAL_Z>(data_vec, dim_value_vec, result);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, node_is_null) {
  NodePtr op_node = nullptr;
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(EXPANDDIMS);
  Status status = kernel->Compute(op_node);
  EXPECT_NE(domi::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, invalid_input_node_size) {
  vector<int64_t> data_vec = {1, 1, 1, 11, 1, 1, 1};
  vector<int64_t> dim_value_vec = {7};
  vector<int64_t> result = {1, 1, 1, 11, 1, 1, 1, 1};

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::OpDescPtr data_op_desc = make_shared<ge::OpDesc>("data", CONSTANTOP);
  int64_t dims_size = 1;
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<uint8_t> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_BOOL);
  GeTensorPtr data_tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                  data_value_vec.size() * sizeof(uint8_t));
  OpDescUtils::SetWeights(data_op_desc, data_tensor);
  data_op_desc->AddOutputDesc(data_tensor_desc);
  NodePtr data_node = graph->AddNode(data_op_desc);
  data_node->Init();

  // add dim node
  ge::OpDescPtr dim_op_desc = make_shared<ge::OpDesc>("dim", CONSTANTOP);
  GeTensorDesc dim_tensor_desc(ge::GeShape(), FORMAT_NCHW, DT_INT64);
  GeTensorPtr dim_tensor =
      make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(), dim_value_vec.size() * sizeof(int64_t));
  OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
  dim_op_desc->AddOutputDesc(dim_tensor_desc);
  NodePtr dim_node = graph->AddNode(dim_op_desc);
  dim_node->Init();

  // add expanddims node
  OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Expanddims", EXPANDDIMS);
  expanddim_op_desc->AddInputDesc(data_tensor_desc);
  expanddim_op_desc->AddInputDesc(dim_tensor_desc);
  NodePtr op_node = graph->AddNode(expanddim_op_desc);
  op_node->Init();

  // add edge
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(EXPANDDIMS);
  Status status = kernel->Compute(op_node);
  EXPECT_NE(ge::SUCCESS, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, dim_node_not_contain_weight) {
  vector<int64_t> data_vec = {1, 1, 1, 11, 1, 1, 1};
  vector<int32_t> dim_value_vec = {7};
  vector<int64_t> result = {1, 1, 1, 11, 1, 1, 1, 1};

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr data_op_desc = make_shared<ge::OpDesc>("data", CONSTANTOP);
  int64_t dims_size = 1;
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<int32_t> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_INT32);
  GeTensorPtr data_tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                  data_value_vec.size() * sizeof(int32_t));
  OpDescUtils::SetWeights(data_op_desc, data_tensor);
  data_op_desc->AddOutputDesc(data_tensor_desc);
  NodePtr data_node = graph->AddNode(data_op_desc);
  data_node->Init();

  // add dim node
  ge::OpDescPtr dim_op_desc = make_shared<ge::OpDesc>("dim", "dim");
  GeTensorDesc dim_tensor_desc(ge::GeShape(), FORMAT_NCHW, DT_INT32);
  GeTensorPtr dim_tensor =
      make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(), dim_value_vec.size() * sizeof(int32_t));
  OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
  dim_op_desc->AddOutputDesc(dim_tensor_desc);
  NodePtr dim_node = graph->AddNode(dim_op_desc);
  dim_node->Init();

  // add expanddims node
  OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Expanddims", EXPANDDIMS);
  expanddim_op_desc->AddInputDesc(data_tensor_desc);
  expanddim_op_desc->AddInputDesc(dim_tensor_desc);
  NodePtr op_node = graph->AddNode(expanddim_op_desc);
  op_node->Init();

  // add edge
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(dim_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(1));

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(EXPANDDIMS);
  Status status = kernel->Compute(op_node);
  EXPECT_NE(ge::SUCCESS, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, folding_int64_success) {
  vector<int64_t> data_vec = {1, 1, 1, 11, 1, 1, 1};
  vector<int64_t> dim_value_vec = {7};
  vector<int64_t> result = {1, 1, 1, 11, 1, 1, 1, 1};

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr data_op_desc = make_shared<ge::OpDesc>("data", CONSTANTOP);
  int64_t dims_size = 1;
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<uint8_t> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_BOOL);
  GeTensorPtr data_tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                  data_value_vec.size() * sizeof(uint8_t));
  OpDescUtils::SetWeights(data_op_desc, data_tensor);
  data_op_desc->AddOutputDesc(data_tensor_desc);
  NodePtr data_node = graph->AddNode(data_op_desc);
  data_node->Init();

  // add dim node
  ge::OpDescPtr dim_op_desc = make_shared<ge::OpDesc>("dim", CONSTANTOP);
  GeTensorDesc dim_tensor_desc(ge::GeShape(), FORMAT_NCHW, DT_INT64);
  GeTensorPtr dim_tensor =
      make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(), dim_value_vec.size() * sizeof(int64_t));
  OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
  dim_op_desc->AddOutputDesc(dim_tensor_desc);
  NodePtr dim_node = graph->AddNode(dim_op_desc);
  dim_node->Init();

  // add expanddims node
  OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Expanddims", EXPANDDIMS);
  expanddim_op_desc->AddInputDesc(data_tensor_desc);
  expanddim_op_desc->AddInputDesc(dim_tensor_desc);
  expanddim_op_desc->AddOutputDesc(data_tensor_desc);
  NodePtr op_node = graph->AddNode(expanddim_op_desc);
  op_node->Init();

  // add edge
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(dim_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(1));

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(EXPANDDIMS);

  vector<ConstGeTensorPtr> input = {data_tensor, dim_tensor};
  vector<GeTensorPtr> v_output;
  Status status = kernel->Compute(op_node->GetOpDesc(), input, v_output);
  EXPECT_EQ(ge::SUCCESS, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_expanddims_kernel, nullptr_opdesc_failed) {
  ge::OpDescPtr null_op_desc = nullptr;
  vector<ConstGeTensorPtr> input = {};
  vector<GeTensorPtr> v_output;
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(EXPANDDIMS);
  Status status = kernel->Compute(null_op_desc, input, v_output);
  EXPECT_EQ(ge::PARAM_INVALID, status);
}
