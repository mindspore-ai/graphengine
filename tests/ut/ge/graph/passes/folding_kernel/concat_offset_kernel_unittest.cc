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
#include "graph/passes/folding_kernel/concat_offset_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/fp16_t.h"
#include "common/ge_inner_error_codes.h"
#include "common/op/attr_value_util.h"
#include "common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/passes/dimension_compute_pass.h"
#include "graph/passes/folding_kernel/kernel_utils.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "inc/kernel_factory.h"
#undef protected
#undef private

using namespace domi;
using namespace testing;
using namespace ge;

class UTEST_graph_passes_folding_kernel_concat_offset_kernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UTEST_graph_passes_folding_kernel_concat_offset_kernel, check_attr_fail) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("ConcatOffset", "ConcatOffset");

  vector<ConstGeTensorPtr> input = {};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(CONCATOFFSET);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_concat_offset_kernel, check_input_size) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("ConcatOffset", "ConcatOffset");
  AttrUtils::SetInt(op_desc_ptr, "N", 2);
  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(CONCATOFFSET);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_concat_offset_kernel, compute_success) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("ConcatOffset", "ConcatOffset");
  (void)AttrUtils::SetInt(op_desc_ptr, "N", 3);
  GeTensorDesc dims_tensor_desc(GeShape({0, 0, 0, 0}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {0};
  vector<int32_t> data_vec_0 = {0, 0, 0, 0};
  vector<int32_t> data_vec_1 = {1, 1, 1, 1};
  vector<int32_t> data_vec_2 = {1, 0, 0, 0};
  GeTensorDesc tensor_desc_0(GeShape({0}), FORMAT_ND, DT_INT32);
  GeTensorDesc tensor_desc_1(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_dim =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)dims_vec_0.data(), dims_vec_0.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_dim, tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(CONCATOFFSET);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(ge::SUCCESS, status);
}
