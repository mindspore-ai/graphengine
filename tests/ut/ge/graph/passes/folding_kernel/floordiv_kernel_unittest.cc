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
#include "graph/passes/folding_kernel/floordiv_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/passes/folding_kernel/kernel_utils.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "inc/kernel_factory.h"
#undef protected
#undef private

using namespace std;
using namespace testing;
using namespace domi;
using namespace ge;

class UTEST_graph_passes_folding_kernel_floordiv_kernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UTEST_graph_passes_folding_kernel_floordiv_kernel, int32_vector_vector_success) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("floor_div", FLOORDIV);

  vector<int64_t> dims_vec_0 = {2, 3};
  vector<int32_t> data_vec_0 = {1, 6, 32, 9, 10, 7};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {2, 3};
  vector<int32_t> data_vec_1 = {2, 9, 9, 9, 9, 9};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(FLOORDIV);
  domi::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  int32_t *out_data = (int32_t *)v_output[0]->GetData().data();

  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(out_data[0], 0);
  EXPECT_EQ(out_data[1], 0);
  EXPECT_EQ(out_data[2], 3);
  EXPECT_EQ(out_data[3], 1);
  EXPECT_EQ(out_data[4], 1);
  EXPECT_EQ(out_data[5], 0);
  EXPECT_EQ(v_output[0]->GetData().size(), 24);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetShape().GetDims().size(), 2);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetShape().GetDim(0), 2);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetShape().GetDim(1), 3);
}

TEST_F(UTEST_graph_passes_folding_kernel_floordiv_kernel, int32_scale_vector_success) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("floor_div", FLOORDIV);

  vector<int64_t> dims_vec_0 = {2, 3};
  vector<int32_t> data_vec_0 = {1, 6, 32, 9, 10, 7};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {};
  vector<int32_t> data_vec_1 = {5};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_1, tensor_0};
  vector<GeTensorPtr> v_output;
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(FLOORDIV);
  domi::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  int32_t *out_data = (int32_t *)v_output[0]->GetData().data();

  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(out_data[0], 5);
  EXPECT_EQ(out_data[1], 0);
  EXPECT_EQ(out_data[2], 0);
  EXPECT_EQ(out_data[3], 0);
  EXPECT_EQ(out_data[4], 0);
  EXPECT_EQ(out_data[5], 0);
  EXPECT_EQ(v_output[0]->GetData().size(), 24);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetShape().GetDims().size(), 2);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetShape().GetDim(0), 2);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetShape().GetDim(1), 3);
}

TEST_F(UTEST_graph_passes_folding_kernel_floordiv_kernel, int32_vector_scale_success) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("floor_div", FLOORDIV);

  vector<int64_t> dims_vec_0 = {2, 3};
  vector<int32_t> data_vec_0 = {1, 6, 32, 9, 10, 7};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {};
  vector<int32_t> data_vec_1 = {-9};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(FLOORDIV);
  domi::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  int32_t *out_data = (int32_t *)v_output[0]->GetData().data();

  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(out_data[0], -1);
  EXPECT_EQ(out_data[1], -1);
  EXPECT_EQ(out_data[2], -4);
  EXPECT_EQ(out_data[3], -1);
  EXPECT_EQ(out_data[4], -2);
  EXPECT_EQ(out_data[5], -1);
  EXPECT_EQ(v_output[0]->GetData().size(), 24);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetShape().GetDims().size(), 2);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetShape().GetDim(0), 2);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetShape().GetDim(1), 3);
}

TEST_F(UTEST_graph_passes_folding_kernel_floordiv_kernel, int32_scale_scale_success) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("floor_div", FLOORDIV);

  vector<int64_t> dims_vec_0 = {};
  vector<int32_t> data_vec_0 = {-9};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {};
  vector<int32_t> data_vec_1 = {-5};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(FLOORDIV);
  domi::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  int32_t *out_data = (int32_t *)v_output[0]->GetData().data();

  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(out_data[0], 1);
  EXPECT_EQ(v_output[0]->GetData().size(), 4);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetShape().GetDims().size(), 0);
}

TEST_F(UTEST_graph_passes_folding_kernel_floordiv_kernel, float_vector_vector_success) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("floor_div", FLOORDIV);

  vector<int64_t> dims_vec_0 = {2, 3};
  vector<float> data_vec_0 = {1, 6, 32, 9, -10, -7};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<int64_t> dims_vec_1 = {2, 3};
  vector<float> data_vec_1 = {2, -9.9, -9.9, -9.9, -9.9, -9.9};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(FLOORDIV);
  domi::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  float *out_data = const_cast<float *>(reinterpret_cast<const float *>(v_output[0]->GetData().GetData()));

  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(v_output[0]->GetData().size(), 24);
  EXPECT_EQ(out_data[0], 0);
  EXPECT_EQ(out_data[1], -1);
  EXPECT_EQ(out_data[2], -4);
  EXPECT_EQ(out_data[3], -1);
  EXPECT_EQ(out_data[4], 1);
  EXPECT_EQ(out_data[5], 0);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetShape().GetDims().size(), 2);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetShape().GetDim(0), 2);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetShape().GetDim(1), 3);
}

TEST_F(UTEST_graph_passes_folding_kernel_floordiv_kernel, invalid_input_size_failed) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("floor_div", FLOORDIV);

  vector<int64_t> dims_vec_0 = {2, 3};
  vector<int32_t> data_vec_0 = {1, 6, 32, 9, 10, 7};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {6};
  vector<int32_t> data_vec_1 = {2, 9, 9, 9, 9, 9};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_1};
  vector<GeTensorPtr> v_output;
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(FLOORDIV);
  domi::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(domi::NOT_CHANGED, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_floordiv_kernel, invalid_dim_size_failed) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("floor_div", FLOORDIV);

  vector<int64_t> dims_vec_0 = {2, 3};
  vector<int32_t> data_vec_0 = {1, 6, 32, 9, 10, 7};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {6};
  vector<int32_t> data_vec_1 = {2, 9, 9, 9, 9, 9};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(FLOORDIV);
  domi::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(domi::NOT_CHANGED, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_floordiv_kernel, invalid_dim_i_failed) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("floor_div", FLOORDIV);

  vector<int64_t> dims_vec_0 = {2, 3};
  vector<int32_t> data_vec_0 = {1, 6, 32, 9, 10, 7};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {2, 1};
  vector<int32_t> data_vec_1 = {2, 9};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(FLOORDIV);
  domi::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(domi::NOT_CHANGED, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_floordiv_kernel, empty_data_failed) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("floor_div", FLOORDIV);

  vector<int64_t> dims_vec_0 = {2};
  vector<int32_t> data_vec_0 = {};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {2};
  vector<int32_t> data_vec_1 = {2, 9};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(FLOORDIV);
  domi::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(domi::NOT_CHANGED, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_floordiv_kernel, unmatched_data_type_failed) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("floor_div", FLOORDIV);

  vector<int64_t> dims_vec_0 = {2};
  vector<float> data_vec_0 = {3, 36};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<int64_t> dims_vec_1 = {2};
  vector<int32_t> data_vec_1 = {2, 9};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(FLOORDIV);
  domi::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(domi::NOT_CHANGED, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_floordiv_kernel, invalid_data_type_failed) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("floor_div", FLOORDIV);

  vector<int64_t> dims_vec_0 = {2};
  vector<double> data_vec_0 = {3, 36};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_DOUBLE);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(double));

  vector<int64_t> dims_vec_1 = {2};
  vector<int32_t> data_vec_1 = {2, 9};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(FLOORDIV);
  domi::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(domi::NOT_CHANGED, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_floordiv_kernel, zero_vector_vector_failed) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("floor_div", FLOORDIV);

  vector<int64_t> dims_vec_0 = {2, 3};
  vector<int32_t> data_vec_0 = {1, 6, 32, 9, 5, 3};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {2, 3};
  vector<int32_t> data_vec_1 = {2, 9, 9, 9, 0, 9};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(FLOORDIV);
  domi::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(domi::NOT_CHANGED, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_floordiv_kernel, zero_vector_scale_failed) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("floor_div", FLOORDIV);

  vector<int64_t> dims_vec_0 = {2, 3};
  vector<int32_t> data_vec_0 = {1, 6, 32, 9, 5, 3};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {};
  vector<int32_t> data_vec_1 = {0};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(FLOORDIV);
  domi::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(domi::NOT_CHANGED, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_floordiv_kernel, zero_scale_vector_failed) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("floor_div", FLOORDIV);

  vector<int64_t> dims_vec_0 = {2, 3};
  vector<int32_t> data_vec_0 = {1, 6, 32, 9, 0, 3};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {};
  vector<int32_t> data_vec_1 = {6};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_1, tensor_0};
  vector<GeTensorPtr> v_output;
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(FLOORDIV);
  domi::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(domi::NOT_CHANGED, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_floordiv_kernel, zero_scale_scale_failed) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("floor_div", FLOORDIV);

  vector<int64_t> dims_vec_0 = {};
  vector<int32_t> data_vec_0 = {0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {};
  vector<int32_t> data_vec_1 = {6};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_1, tensor_0};
  vector<GeTensorPtr> v_output;
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(FLOORDIV);
  domi::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(domi::NOT_CHANGED, status);
}
