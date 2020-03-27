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
#include "graph/passes/folding_kernel/concat_v2_kernel.h"

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

using namespace testing;
using namespace ge;

class UTEST_graph_passes_folding_kernel_concat_v2_kernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UTEST_graph_passes_folding_kernel_concat_v2_kernel, check_param) {
  OpDescPtr op_desc_ptr = nullptr;

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT16);
  ConstGeTensorPtr tensor_0 = make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * 2);

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(CONCATV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  EXPECT_EQ(ge::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_concat_v2_kernel, check_input_size) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("Concat", "Concat");
  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(CONCATV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_concat_v2_kernel, check_1) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("Concat", "Concat");
  GeTensorDesc dims_tensor_desc(GeShape({0, 0, 0, 0}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {0, 0, 0, 0};
  vector<int32_t> data_vec_0 = {0, 0, 0, 0};
  vector<int32_t> data_vec_1 = {1, 1, 1, 1};
  vector<int32_t> data_vec_2 = {0, 0, 0, 0};
  GeTensorDesc tensor_desc_0(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc tensor_desc_1(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc tensor_desc_2(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(CONCATV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  EXPECT_EQ(ge::SUCCESS, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_concat_v2_kernel, check_int32_success) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("ConcatV2", "ConcatV2");
  vector<bool> is_input_const_vec = {true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  vector<int64_t> dims_vec_0 = {2, 3};
  vector<int32_t> data_vec_0 = {1, 2, 3, 4, 5, 6};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {2, 3};
  vector<int32_t> data_vec_1 = {7, 8, 9, 10, 11, 12};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_2 = {1};
  vector<int32_t> data_vec_2 = {0};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> v_output;

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(CONCATV2);
  Status status = kernel->Compute(op_desc_ptr, input, v_output);
  EXPECT_EQ(status, SUCCESS);

  GeTensorPtr out = v_output[0];
  vector<int32_t> data_y = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  EXPECT_EQ(out->GetData().size(), 48);
  size_t one_size = sizeof(int32_t);
  size_t out_nums = out->GetData().size() / one_size;
  for (size_t i = 0; i < out_nums; i++) {
    int32_t *one_val = (int32_t *)(out->GetData().data() + i * one_size);
    EXPECT_EQ(data_y[i], *one_val);
  }
}

TEST_F(UTEST_graph_passes_folding_kernel_concat_v2_kernel, check_int32_success_1) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("ConcatV2", CONCATV2);
  vector<bool> is_input_const_vec = {true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  vector<int64_t> dims_vec_0 = {2, 3};
  vector<int32_t> data_vec_0 = {1, 2, 3, 4, 5, 6};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {2, 3};
  vector<int32_t> data_vec_1 = {7, 8, 9, 10, 11, 12};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_2;
  vector<int32_t> data_vec_2 = {1};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> v_output;

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(CONCATV2);
  Status status = kernel->Compute(op_desc_ptr, input, v_output);
  EXPECT_EQ(status, SUCCESS);

  GeTensorPtr out = v_output[0];
  vector<int32_t> data_y = {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12};
  EXPECT_EQ(out->GetData().size(), 48);
  size_t one_size = sizeof(int32_t);
  size_t out_nums = out->GetData().size() / one_size;
  for (size_t i = 0; i < out_nums; i++) {
    int32_t *one_val = (int32_t *)(out->GetData().data() + i * one_size);
    EXPECT_EQ(data_y[i], *one_val);
  }
}

TEST_F(UTEST_graph_passes_folding_kernel_concat_v2_kernel, check_float_success) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("ConcatV2", CONCATV2);
  vector<bool> is_input_const_vec = {true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_FLOAT);

  vector<int64_t> dims_vec_0 = {2, 3};
  vector<float> data_vec_0 = {1.12, 2.12, 3.12, 4.12, 5.12, 6.12};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<int64_t> dims_vec_1 = {2, 3};
  vector<float> data_vec_1 = {7.12, 8.12, 9.12, 10.12, 11.12, 12.13};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(float));

  vector<int64_t> dims_vec_2;
  vector<int32_t> data_vec_2 = {0};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> v_output;

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(CONCATV2);
  Status status = kernel->Compute(op_desc_ptr, input, v_output);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(v_output[0]->GetData().size(), 48);  // 12*4

  GeTensorPtr out = v_output[0];
  vector<float> data_y = {1.12, 2.12, 3.12, 4.12, 5.12, 6.12, 7.12, 8.12, 9.12, 10.12, 11.12, 12.13};
  EXPECT_EQ(out->GetData().size(), 48);
  size_t one_size = sizeof(float);
  size_t out_nums = out->GetData().size() / one_size;
  for (size_t i = 0; i < out_nums; i++) {
    float *one_val = (float *)(out->GetData().data() + i * one_size);
    EXPECT_EQ(data_y[i], *one_val);
  }
}

TEST_F(UTEST_graph_passes_folding_kernel_concat_v2_kernel, check_not_change) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("ConcatV2", CONCATV2);
  vector<bool> is_input_const_vec = {true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_FLOAT);

  vector<int64_t> dims_vec_0 = {2, 3};
  vector<float> data_vec_0 = {1.12, 2.12, 3.12, 4.12, 5.12, 6.12};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_UNDEFINED);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<int64_t> dims_vec_1 = {2, 3};
  vector<int32_t> data_vec_1 = {7, 8, 9, 10, 11, 12};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_2;
  vector<int32_t> data_vec_2 = {0};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  ConstGeTensorPtr tensor_6 = nullptr;
  vector<ConstGeTensorPtr> input_less = {tensor_0, tensor_2};
  vector<ConstGeTensorPtr> input_diff = {tensor_0, tensor_1, tensor_2};
  vector<ConstGeTensorPtr> input_null = {tensor_0, tensor_6, tensor_2};
  vector<ConstGeTensorPtr> input_not_support = {tensor_0, tensor_0, tensor_2};
  vector<GeTensorPtr> v_output;

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(CONCATV2);
  Status status = kernel->Compute(op_desc_ptr, input_diff, v_output);
  EXPECT_EQ(status, NOT_CHANGED);
  EXPECT_EQ(v_output.size(), 0);

  status = kernel->Compute(op_desc_ptr, input_less, v_output);
  EXPECT_EQ(status, NOT_CHANGED);

  status = kernel->Compute(op_desc_ptr, input_null, v_output);
  EXPECT_EQ(status, NOT_CHANGED);

  status = kernel->Compute(op_desc_ptr, input_not_support, v_output);
  EXPECT_EQ(status, NOT_CHANGED);
}
