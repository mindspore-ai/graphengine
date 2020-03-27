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

#include "graph/passes/folding_kernel/reformat_kernel.h"

#include "common/debug/log.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "graph/passes/folding_kernel/kernel_utils.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "inc/kernel_factory.h"

using namespace domi;
using namespace testing;
using namespace ge;

class UTEST_graph_passes_folding_kernel_reformat_kernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UTEST_graph_passes_folding_kernel_reformat_kernel, compute_success) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("ReFormat", "ReFormat");

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);
  GeTensorDesc dims_tensor_desc_in(GeShape({1, 1, 1, 1}), FORMAT_ND, DT_FLOAT);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 = make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), 1 * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(REFORMAT);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(ge::SUCCESS, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_reformat_kernel, empty_input) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("ReFormat", "ReFormat");

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);
  GeTensorDesc dims_tensor_desc_in(GeShape({1, 1, 1, 1}), FORMAT_ND, DT_FLOAT);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 = make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), 1 * sizeof(float));

  vector<ConstGeTensorPtr> input = {};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(REFORMAT);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(ge::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_reformat_kernel, input_nullptr) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("ReFormat", "ReFormat");

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);
  GeTensorDesc dims_tensor_desc_in(GeShape({1, 1, 1, 1}), FORMAT_ND, DT_FLOAT);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 = make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), 1 * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(REFORMAT);
  ge::Status status = kernel->Compute(nullptr, input, v_output);

  EXPECT_EQ(ge::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_reformat_kernel, invalid_inputsize) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("ReFormat", "ReFormat");

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);
  GeTensorDesc dims_tensor_desc_in(GeShape({1, 1, 1, 1}), FORMAT_ND, DT_FLOAT);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 = make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), 1 * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(REFORMAT);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(ge::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_reformat_kernel, mismatch_shape) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("ReFormat", "ReFormat");

  GeTensorDesc dims_tensor_desc(GeShape({1}), FORMAT_ND, DT_FLOAT);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);
  GeTensorDesc dims_tensor_desc_in(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 = make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), 1 * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(REFORMAT);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_reformat_kernel, mismatch_dtype) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("ReFormat", "ReFormat");

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_ND, DT_FLOAT16);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);
  GeTensorDesc dims_tensor_desc_in(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 = make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), 1 * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(REFORMAT);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_reformat_kernel, mismatch_data_size) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("ReFormat", "ReFormat");

  GeTensorDesc dims_tensor_desc(GeShape({1, 2, 3, 4}), FORMAT_ND, DT_FLOAT);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);
  GeTensorDesc dims_tensor_desc_in(GeShape({1, 2, 3, 4}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 = make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), 1 * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(REFORMAT);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(NOT_CHANGED, status);
}
