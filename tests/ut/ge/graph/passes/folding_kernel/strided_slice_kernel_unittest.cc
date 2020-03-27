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
#include "graph/passes/folding_kernel/strided_slice_kernel.h"

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

class UTEST_graph_passes_folding_kernel_strided_slice_kernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UTEST_graph_passes_folding_kernel_strided_slice_kernel, check_input_size) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("StridedSlice", "StridedSlice");
  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
}

TEST_F(UTEST_graph_passes_folding_kernel_strided_slice_kernel, test_2) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("StridedSlice", "StridedSlice");
  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  // EXPECT_EQ(domi::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_strided_slice_kernel, test_3_0) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  // EXPECT_EQ(domi::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_strided_slice_kernel, test_3_1) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  // EXPECT_EQ(domi::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_strided_slice_kernel, test_3_2) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  // EXPECT_EQ(domi::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_strided_slice_kernel, test_3_3) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  // EXPECT_EQ(domi::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_strided_slice_kernel, test_3_5) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  // EXPECT_EQ(domi::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_strided_slice_kernel, test_3_6) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  // EXPECT_EQ(domi::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_strided_slice_kernel, test_3_7) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 = nullptr;

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  // EXPECT_EQ(domi::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_strided_slice_kernel, test_3_8) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 1);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 1);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 = nullptr;

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  // EXPECT_EQ(domi::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_strided_slice_kernel, test_3_9) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, DT_FLOAT16);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT16);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT16);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  // EXPECT_EQ(domi::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_strided_slice_kernel, test_3_10) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 1);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  // EXPECT_EQ(domi::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_strided_slice_kernel, test_3_11) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 1);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  // EXPECT_EQ(domi::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_strided_slice_kernel, test_3_12) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 1);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  // EXPECT_EQ(domi::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_strided_slice_kernel, test_4) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  // EXPECT_EQ(domi::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_strided_slice_kernel, test_5) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  // EXPECT_EQ(domi::PARAM_INVALID, status);
}

TEST_F(UTEST_graph_passes_folding_kernel_strided_slice_kernel, test_7) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {10, 10, 10, 10};
  vector<int32_t> data_vec_0 = {3, 3, 3, 3};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  ge::Status status = kernel->Compute(op_desc_ptr, input, v_output);
  // EXPECT_EQ(domi::PARAM_INVALID, status);
}
