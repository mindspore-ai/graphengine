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

#define protected public
#define private public
#include "graph/utils/op_desc_utils.h"

#include "debug/ge_op_types.h"
#include "graph/compute_graph.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/operator.h"
#undef protected
#undef private

using namespace std;
using namespace ge;

class ge_test_opdesc_utils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(ge_test_opdesc_utils, CreateOperatorFromDesc) {
  OpDescPtr descPtr = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(descPtr->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(descPtr->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(descPtr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);
  GeAttrValue test_attr = GeAttrValue::CreateFrom<GeAttrValue::INT>(1);
  descPtr->SetAttr("test_attr", std::move(test_attr));

  Operator oprt = OpDescUtils::CreateOperatorFromOpDesc(descPtr);

  GeAttrValue::INT out;
  oprt.GetAttr("test_attr", out);
  EXPECT_EQ(out, 1);

  TensorDesc inputDesc1 = oprt.GetInputDesc("x");
  EXPECT_TRUE(inputDesc1.GetShape().GetDimNum() == 4);
  EXPECT_TRUE(inputDesc1.GetShape().GetDim(0) == 1);
  EXPECT_TRUE(inputDesc1.GetShape().GetDim(1) == 16);
  EXPECT_TRUE(inputDesc1.GetShape().GetDim(2) == 16);
  EXPECT_TRUE(inputDesc1.GetShape().GetDim(3) == 16);

  TensorDesc inputDesc2 = oprt.GetInputDesc(1);
  EXPECT_TRUE(inputDesc2.GetShape().GetDimNum() == 4);
  EXPECT_TRUE(inputDesc2.GetShape().GetDim(0) == 1);
  EXPECT_TRUE(inputDesc2.GetShape().GetDim(1) == 1);
  EXPECT_TRUE(inputDesc2.GetShape().GetDim(2) == 1);
  EXPECT_TRUE(inputDesc2.GetShape().GetDim(3) == 1);

  OpDescPtr outPtr = OpDescUtils::GetOpDescFromOperator(oprt);
  EXPECT_TRUE(outPtr == descPtr);

  string name1 = outPtr->GetName();
  string name2 = oprt.GetName();
  EXPECT_TRUE(name1 == name2);
}

TEST_F(ge_test_opdesc_utils, ClearInputDesc) {
  OpDescPtr descPtr = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(descPtr->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(descPtr->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(descPtr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  OpDescPtr descPtr2 = std::make_shared<OpDesc>("name2", "type2");
  EXPECT_EQ(descPtr2->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(descPtr2->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(descPtr2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("name");
  NodePtr n1 = graphPtr->AddNode(descPtr);
  NodePtr n2 = graphPtr->AddNode(descPtr);
  EXPECT_TRUE(OpDescUtils::ClearInputDesc(n1));
  EXPECT_TRUE(OpDescUtils::ClearInputDesc(descPtr2, 0));
}

TEST_F(ge_test_opdesc_utils, MutableWeights) {
  OpDescPtr descPtr = std::make_shared<OpDesc>("name1", CONSTANT);
  EXPECT_EQ(descPtr->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(descPtr->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(descPtr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  OpDescPtr descPtr2 = std::make_shared<OpDesc>("name2", "type2");
  EXPECT_EQ(descPtr2->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(descPtr2->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(descPtr2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("name");
  NodePtr n1 = graphPtr->AddNode(descPtr);
  NodePtr n2 = graphPtr->AddNode(descPtr);

  float f[1] = {1.0};
  GeTensorDesc tensorDesc(GeShape({1}));
  GeTensorPtr tensor = std::make_shared<GeTensor>(tensorDesc, (const uint8_t *)f, 1 * sizeof(float));

  OpDescPtr nullOpDesc = nullptr;

  EXPECT_EQ(GRAPH_PARAM_INVALID, OpDescUtils::SetWeights(descPtr, nullptr));
  EXPECT_EQ(GRAPH_SUCCESS, OpDescUtils::SetWeights(descPtr, tensor));
  EXPECT_EQ(GRAPH_SUCCESS, OpDescUtils::SetWeights(*descPtr2.get(), tensor));
  EXPECT_EQ(GRAPH_FAILED, OpDescUtils::SetWeights(*descPtr2.get(), nullptr));

  EXPECT_NE(nullptr, OpDescUtils::MutableWeights(descPtr));
  EXPECT_NE(nullptr, OpDescUtils::MutableWeights(*descPtr.get()));

  EXPECT_EQ(nullptr, OpDescUtils::MutableWeights(nullOpDesc));

  EXPECT_EQ(nullptr, OpDescUtils::CreateOperatorFromOpDesc(descPtr));

  auto tensorVec = OpDescUtils::GetWeights(n1);
  EXPECT_NE(0, tensorVec.size());
}
