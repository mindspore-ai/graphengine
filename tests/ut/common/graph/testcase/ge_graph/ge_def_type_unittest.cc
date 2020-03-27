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
#include "graph/ge_tensor.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

using namespace std;
using namespace ge;

class ge_test_def_type : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(ge_test_def_type, base) {
  CompressInfo com1;
  com1.set_blockrow(1);
  int32_t a = com1.blockrow;
  EXPECT_EQ(a, 1);

  QuantizeFactor qfactor;
  QuantizeFactorParams qParams;
}

TEST_F(ge_test_def_type, quant) {
  OpDescPtr descPtr1 = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(descPtr1->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(descPtr1->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(descPtr1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  EXPECT_EQ(OpDescUtils::HasQuantizeFactorParams(descPtr1), false);
  EXPECT_EQ(OpDescUtils::HasQuantizeFactorParams(*descPtr1), false);
  QuantizeFactorParams q1;
  EXPECT_EQ(q1.has_quantize_param(), false);
  QuantizeFactor *qf1 = q1.mutable_quantize_param();
  EXPECT_EQ(q1.has_quantize_param(), true);

  string s1 = "value1";
  q1.quantize_param.set_scale_value(s1.data(), s1.size());
  EXPECT_EQ(OpDescUtils::SetQuantizeFactorParams(descPtr1, q1), GRAPH_SUCCESS);
  QuantizeFactorParams q2;
  EXPECT_EQ(OpDescUtils::GetQuantizeFactorParams(descPtr1, q2), GRAPH_SUCCESS);
  string s2((char *)q2.quantize_param.scale_value.GetData(), q2.quantize_param.scale_value.GetSize());
  EXPECT_EQ(s2, "value1");

  float f[2] = {1, 2};
  string s(static_cast<char *>(static_cast<void *>(f)), 2 * sizeof(float));
  q1.quantize_param.set_scale_value(f, 2 * sizeof(float));
  EXPECT_EQ(OpDescUtils::SetQuantizeFactorParams(*descPtr1, q1), GRAPH_SUCCESS);
  QuantizeFactorParams q3;
  EXPECT_EQ(OpDescUtils::GetQuantizeFactorParams(*descPtr1, q3), GRAPH_SUCCESS);
  Buffer &b = q3.quantize_param.scale_value;
  float f1[2];
  memcpy(f1, b.GetData(), b.GetSize());
  EXPECT_EQ(f1[0], 1);
  EXPECT_EQ(f1[1], 2);
}
