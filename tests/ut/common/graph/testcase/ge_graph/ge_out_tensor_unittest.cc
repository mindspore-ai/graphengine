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
#include <iostream>

#include "graph/ge_attr_value.h"
#include "graph/tensor.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/tensor_utils.h"

using namespace std;
using namespace ge;

class ge_out_test_tensor : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(ge_out_test_tensor, shape) {
  Shape a;
  EXPECT_EQ(a.GetDim(0), 0);
  EXPECT_EQ(a.GetShapeSize(), 0);
  EXPECT_EQ(a.SetDim(0, 0), GRAPH_FAILED);

  vector<int64_t> vec({1, 2, 3, 4});
  Shape b(vec);
  Shape c({1, 2, 3, 4});
  EXPECT_EQ(c.GetDimNum(), 4);
  EXPECT_EQ(c.GetDim(2), 3);
  EXPECT_EQ(c.GetDim(5), 0);
  EXPECT_EQ(c.SetDim(10, 0), GRAPH_FAILED);

  EXPECT_EQ(c.SetDim(2, 2), GRAPH_SUCCESS);
  EXPECT_EQ(c.GetDim(2), 2);
  vector<int64_t> vec1 = c.GetDims();
  EXPECT_EQ(c.GetDim(0), vec1[0]);
  EXPECT_EQ(c.GetDim(1), vec1[1]);
  EXPECT_EQ(c.GetDim(2), vec1[2]);
  EXPECT_EQ(c.GetDim(3), vec1[3]);

  EXPECT_EQ(c.GetShapeSize(), 16);
}

TEST_F(ge_out_test_tensor, tensorDesc) {
  TensorDesc a;
  Shape s({1, 2, 3, 4});
  TensorDesc b(s);
  Shape s1 = b.GetShape();
  EXPECT_EQ(s1.GetDim(0), s.GetDim(0));
  auto shape_m1 = b.GetShape();
  shape_m1.SetDim(0, 2);
  b.SetShape(shape_m1);
  EXPECT_EQ(b.GetShape().GetDim(0), 2);
  Shape s2({3, 2, 3, 4});
  b.SetShape(s2);
  EXPECT_EQ(b.GetShape().GetDim(0), 3);

  EXPECT_EQ(b.GetFormat(), FORMAT_NCHW);
  b.SetFormat(FORMAT_RESERVED);
  EXPECT_EQ(b.GetFormat(), FORMAT_RESERVED);

  EXPECT_EQ(b.GetDataType(), DT_FLOAT);
  b.SetDataType(DT_INT8);
  EXPECT_EQ(b.GetDataType(), DT_INT8);

  TensorDesc c;
  c.Update(Shape({1}), FORMAT_NCHW);
  c.Update(s, FORMAT_NCHW);
  c.SetSize(1);

  TensorDesc d;
  d = c;  // Clone;
  EXPECT_EQ(d.GetSize(), 1);
  d.SetSize(12);
  EXPECT_EQ(d.GetSize(), 12);

  TensorDesc e = c;
  EXPECT_EQ(e.GetSize(), 1);

  TensorDesc f = c;
  EXPECT_EQ(f.GetSize(), 1);
}

TEST_F(ge_out_test_tensor, tensor) {
  Shape s({1, 2, 3, 4});
  TensorDesc tensorDesc(s);
  std::vector<uint8_t> data({1, 2, 3, 4});
  Tensor a;
  Tensor b(tensorDesc);
  Tensor c(tensorDesc, data);
  Tensor d(tensorDesc, data.data(), data.size());

  ASSERT_EQ(a.GetSize(), 0);
  ASSERT_EQ(b.GetSize(), 0);
  ASSERT_EQ(c.GetSize(), data.size());
  ASSERT_EQ(d.GetSize(), data.size());
  EXPECT_EQ(c.GetData()[0], uint8_t(1));
  EXPECT_EQ(c.GetData()[1], uint8_t(2));
  EXPECT_EQ(d.GetData()[2], uint8_t(3));
  EXPECT_EQ(d.GetData()[3], uint8_t(4));
  EXPECT_EQ(d.GetTensorDesc().GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(b.GetTensorDesc().GetShape().GetDim(0), 1);
  EXPECT_EQ(c.GetTensorDesc().GetShape().GetDim(1), 2);
  EXPECT_EQ(d.GetTensorDesc().GetShape().GetDim(2), 3);

  Shape s1 = b.GetTensorDesc().GetShape();
  EXPECT_EQ(s1.GetDim(0), 1);
  EXPECT_EQ(b.GetTensorDesc().GetDataType(), DT_FLOAT);
  EXPECT_EQ(b.GetTensorDesc().GetFormat(), FORMAT_NCHW);

  auto tensorDesc_m1 = b.GetTensorDesc();
  tensorDesc_m1.SetDataType(DT_INT8);
  b.SetTensorDesc(tensorDesc_m1);
  EXPECT_EQ(b.GetTensorDesc().GetDataType(), DT_INT8);
  EXPECT_EQ(b.GetTensorDesc().GetFormat(), FORMAT_NCHW);

  EXPECT_EQ(b.GetTensorDesc().GetSize(), 0);
  auto tensorDesc_m2 = b.GetTensorDesc();
  tensorDesc_m2.SetFormat(FORMAT_NC1HWC0);
  tensorDesc_m2.SetSize(112);
  b.SetTensorDesc(tensorDesc_m2);
  EXPECT_EQ(b.GetTensorDesc().GetDataType(), DT_INT8);
  EXPECT_EQ(b.GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(b.GetTensorDesc().GetSize(), 112);

  auto data1 = c.GetData();
  c.SetData(data);
  c.SetData(data.data(), data.size());
  EXPECT_EQ(c.GetSize(), data.size());
  EXPECT_EQ(c.GetData()[0], uint8_t(1));
  EXPECT_EQ(c.GetData()[1], uint8_t(2));
  EXPECT_EQ(c.GetData()[2], uint8_t(3));
  EXPECT_EQ(c.GetData()[3], uint8_t(4));

  Tensor e(std::move(tensorDesc), std::move(data));
  EXPECT_EQ(e.GetSize(), data.size());
  EXPECT_EQ(e.GetData()[2], uint8_t(3));

  Tensor f = e.Clone();
  e.GetData()[2] = 5;
  EXPECT_EQ(e.GetData()[2], uint8_t(5));
  EXPECT_EQ(f.GetSize(), data.size());
  EXPECT_EQ(f.GetData()[2], uint8_t(3));
}

TEST_F(ge_out_test_tensor, test_shape_copy) {
  Shape shape;
  EXPECT_EQ(shape.GetDimNum(), 0);

  Shape shape2 = shape;
  EXPECT_EQ(shape2.GetDimNum(), 0);

  Shape shape3({1, 2, 3});
  shape2 = shape3;
  EXPECT_EQ(shape2.GetDimNum(), 3);
  EXPECT_EQ(shape3.GetDimNum(), 3);
}

TEST_F(ge_out_test_tensor, test_tensor_adapter_as_ge_tensor) {
  TensorDesc tensorDesc(Shape({2, 3, 4, 5}), FORMAT_NC1HWC0, DT_FLOAT16);
  tensorDesc.SetSize(120);
  vector<uint8_t> data = {3, 4, 5, 6, 7, 8};
  Tensor tensor(tensorDesc, data);

  GeTensor geTensor = TensorAdapter::AsGeTensor(tensor);
  EXPECT_EQ(geTensor.GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(geTensor.GetTensorDesc().GetDataType(), DT_FLOAT16);
  uint32_t size = 0;
  TensorUtils::GetSize(geTensor.GetTensorDesc(), size);
  EXPECT_EQ(size, 120);
  auto dims = geTensor.GetTensorDesc().GetShape().GetDims();
  ASSERT_EQ(dims.size(), 4);
  EXPECT_EQ(dims[0], 2);
  EXPECT_EQ(dims[3], 5);
  EXPECT_EQ(geTensor.GetData().GetSize(), 6);
  EXPECT_EQ(geTensor.GetData().GetData()[0], 3);
  EXPECT_EQ(geTensor.GetData().GetData()[5], 8);

  auto geTensorPtr = TensorAdapter::AsGeTensorPtr(tensor);
  EXPECT_EQ(geTensorPtr->GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(geTensorPtr->GetTensorDesc().GetDataType(), DT_FLOAT16);

  const Tensor tensor2 = tensor;
  const GeTensor geTensor2 = TensorAdapter::AsGeTensor(tensor2);
  EXPECT_EQ(geTensor2.GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(geTensor2.GetTensorDesc().GetDataType(), DT_FLOAT16);
  TensorUtils::GetSize(geTensor2.GetTensorDesc(), size);
  EXPECT_EQ(size, 120);
  auto dims2 = geTensor2.GetTensorDesc().GetShape().GetDims();
  ASSERT_EQ(dims2.size(), 4);
  EXPECT_EQ(dims2[0], 2);
  EXPECT_EQ(dims2[3], 5);
  EXPECT_EQ(geTensor2.GetData().GetSize(), 6);
  EXPECT_EQ(geTensor2.GetData().GetData()[0], 3);
  EXPECT_EQ(geTensor2.GetData().GetData()[5], 8);

  auto geTensorPtr2 = TensorAdapter::AsGeTensorPtr(tensor2);
  EXPECT_EQ(geTensorPtr2->GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(geTensorPtr2->GetTensorDesc().GetDataType(), DT_FLOAT16);

  // modify format
  geTensor.MutableTensorDesc().SetFormat(FORMAT_NC1C0HWPAD);
  EXPECT_EQ(geTensor.GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(tensor.GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(geTensor2.GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(tensor2.GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);

  EXPECT_EQ(geTensorPtr->GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(geTensorPtr2->GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);

  // modify datatype
  tensorDesc.SetDataType(DT_INT32);
  tensor.SetTensorDesc(tensorDesc);
  EXPECT_EQ(tensor.GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(geTensor.GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(tensor2.GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(geTensor2.GetTensorDesc().GetDataType(), DT_INT32);

  EXPECT_EQ(geTensorPtr->GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(geTensorPtr2->GetTensorDesc().GetDataType(), DT_INT32);
}

TEST_F(ge_out_test_tensor, test_tensor_adapter_as_tensor) {
  GeTensorDesc geTensorDesc(GeShape({2, 3, 4, 5}), FORMAT_NC1HWC0, DT_FLOAT16);
  TensorUtils::SetSize(geTensorDesc, 120);
  vector<uint8_t> data = {3, 4, 5, 6, 7, 8};
  GeTensor geTensor(geTensorDesc, data);

  Tensor tensor = TensorAdapter::AsTensor(geTensor);
  EXPECT_EQ(tensor.GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(tensor.GetTensorDesc().GetDataType(), DT_FLOAT16);
  EXPECT_EQ(tensor.GetTensorDesc().GetSize(), 120);

  auto dims = tensor.GetTensorDesc().GetShape().GetDims();
  ASSERT_EQ(dims.size(), 4);
  EXPECT_EQ(dims[0], 2);
  EXPECT_EQ(dims[3], 5);
  EXPECT_EQ(tensor.GetSize(), 6);
  EXPECT_EQ(tensor.GetData()[0], 3);
  EXPECT_EQ(tensor.GetData()[5], 8);

  const GeTensor geTensor2 = geTensor;
  const Tensor tensor2 = TensorAdapter::AsTensor(geTensor2);
  EXPECT_EQ(tensor2.GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(tensor2.GetTensorDesc().GetDataType(), DT_FLOAT16);
  EXPECT_EQ(tensor2.GetTensorDesc().GetSize(), 120);
  auto dims2 = tensor2.GetTensorDesc().GetShape().GetDims();
  ASSERT_EQ(dims2.size(), 4);
  EXPECT_EQ(dims2[0], 2);
  EXPECT_EQ(dims2[3], 5);
  EXPECT_EQ(tensor2.GetSize(), 6);
  EXPECT_EQ(tensor2.GetData()[0], 3);
  EXPECT_EQ(tensor2.GetData()[5], 8);

  // modify format
  geTensor.MutableTensorDesc().SetFormat(FORMAT_NC1C0HWPAD);
  EXPECT_EQ(geTensor.GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(tensor.GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(geTensor2.GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(tensor2.GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);

  // modify datatype
  auto tensorDesc = TensorAdapter::GeTensorDesc2TensorDesc(geTensorDesc);
  tensorDesc.SetDataType(DT_INT32);
  tensor.SetTensorDesc(tensorDesc);
  EXPECT_EQ(tensor.GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(geTensor.GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(tensor2.GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(geTensor2.GetTensorDesc().GetDataType(), DT_INT32);
}

TEST_F(ge_out_test_tensor, test_tensor_adapter_transfer2_ge_tensor) {
  TensorDesc tensorDesc(Shape({2, 3, 4, 5}), FORMAT_NC1HWC0, DT_FLOAT16);
  tensorDesc.SetSize(120);
  vector<uint8_t> data = {3, 4, 5, 6, 7, 8};
  Tensor tensor(tensorDesc, data);

  auto getTensorPtr = TensorAdapter::Tensor2GeTensor(tensor);

  EXPECT_EQ(getTensorPtr->GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(getTensorPtr->GetTensorDesc().GetDataType(), DT_FLOAT16);
  uint32_t size = 0;
  TensorUtils::GetSize(getTensorPtr->GetTensorDesc(), size);
  EXPECT_EQ(size, 120);
  auto dims = getTensorPtr->GetTensorDesc().GetShape().GetDims();
  ASSERT_EQ(dims.size(), 4);
  EXPECT_EQ(dims[0], 2);
  EXPECT_EQ(dims[3], 5);
  EXPECT_EQ(getTensorPtr->GetData().GetSize(), 6);
  EXPECT_EQ(getTensorPtr->GetData().GetData()[0], 3);
  EXPECT_EQ(getTensorPtr->GetData().GetData()[5], 8);

  // modify format
  getTensorPtr->MutableTensorDesc().SetFormat(FORMAT_NC1C0HWPAD);
  EXPECT_EQ(getTensorPtr->GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(tensor.GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);  // copy, not change

  // modify datatype
  tensorDesc.SetDataType(DT_INT32);
  tensor.SetTensorDesc(tensorDesc);
  EXPECT_EQ(tensor.GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(getTensorPtr->GetTensorDesc().GetDataType(), DT_FLOAT16);  // copy, not change
}

TEST_F(ge_out_test_tensor, test_tensor_adapter_transfer2_tensor) {
  GeTensorDesc geTensorDesc(GeShape({2, 3, 4, 5}), FORMAT_NC1HWC0, DT_FLOAT16);
  TensorUtils::SetSize(geTensorDesc, 120);
  vector<uint8_t> data = {3, 4, 5, 6, 7, 8};
  GeTensor geTensor(geTensorDesc, data);

  Tensor tensor = TensorAdapter::GeTensor2Tensor(std::make_shared<GeTensor>(geTensor));
  EXPECT_EQ(tensor.GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(tensor.GetTensorDesc().GetDataType(), DT_FLOAT16);
  EXPECT_EQ(tensor.GetTensorDesc().GetSize(), 120);

  auto dims = tensor.GetTensorDesc().GetShape().GetDims();
  ASSERT_EQ(dims.size(), 4);
  EXPECT_EQ(dims[0], 2);
  EXPECT_EQ(dims[3], 5);
  EXPECT_EQ(tensor.GetSize(), 6);
  EXPECT_EQ(tensor.GetData()[0], 3);
  EXPECT_EQ(tensor.GetData()[5], 8);

  // modify format
  geTensor.MutableTensorDesc().SetFormat(FORMAT_NC1C0HWPAD);
  EXPECT_EQ(geTensor.GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(tensor.GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);  // copy, not change

  // modify datatype
  auto tensorDesc = TensorAdapter::GeTensorDesc2TensorDesc(geTensorDesc);
  tensorDesc.SetDataType(DT_INT32);
  tensor.SetTensorDesc(tensorDesc);
  EXPECT_EQ(tensor.GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(geTensor.GetTensorDesc().GetDataType(), DT_FLOAT16);  // copy, not change
}
