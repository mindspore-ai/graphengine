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
#include "graph/op_desc.h"

#include "graph/compute_graph.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/node.h"
#include "graph/operator_factory.h"
#include "utils/op_desc_utils.h"
#undef protected
#undef private

using namespace std;
using namespace ge;

class ge_test_opdesc : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(ge_test_opdesc, ge_test_opdesc_common) {
  string name = "Conv2d";
  string type = "Data";
  OpDescPtr opDesc = std::make_shared<OpDesc>(name, type);
  EXPECT_TRUE(opDesc);
  EXPECT_EQ(name, opDesc->GetName());
  EXPECT_EQ(type, opDesc->GetType());
  name = name + "_modify";
  type = type + "_modify";
  opDesc->SetName(name);
  opDesc->SetType(type);
  EXPECT_EQ(name, opDesc->GetName());
  EXPECT_EQ(type, opDesc->GetType());
}

TEST_F(ge_test_opdesc, ClearAllOutputDesc) {
  auto g = std::make_shared<ge::ComputeGraph>("Test");

  // creat node
  ::ge::OpDescPtr desc = std::make_shared<ge::OpDesc>("", "");
  desc->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW));
  desc->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW));
  desc->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  auto node = g->AddNode(desc);
  bool ret = OpDescUtils::ClearOutputDesc(node);
  EXPECT_EQ(true, ret);
}

TEST_F(ge_test_opdesc, ClearOutputDescByIndex) {
  auto g = std::make_shared<ge::ComputeGraph>("Test");

  // creat node
  ::ge::OpDescPtr desc = std::make_shared<ge::OpDesc>("", "");
  desc->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW));
  desc->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW));
  desc->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  desc->AddOutputDesc("z", GeTensorDesc(GeShape({1, 1, 8, 8}), FORMAT_NCHW));
  auto node = g->AddNode(desc);
  bool ret = OpDescUtils::ClearOutputDesc(desc, 1);
  EXPECT_EQ(true, ret);
}

TEST_F(ge_test_opdesc, ge_test_opdesc_inputs) {
  string name = "Conv2d";
  string type = "Data";
  OpDescPtr opDesc = std::make_shared<OpDesc>(name, type);
  EXPECT_TRUE(opDesc);
  GeTensorDesc tedesc1(GeShape({1, 2, 3, 4}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(GRAPH_SUCCESS, opDesc->AddInputDesc(tedesc1));
  GeTensorDesc tedesc2(GeShape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(GRAPH_SUCCESS, opDesc->AddInputDesc("w", tedesc2));
  GeTensorDesc tedesc3(GeShape({8, 9, 10, 11}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(GRAPH_SUCCESS, opDesc->AddInputDesc("w", tedesc3));
  EXPECT_EQ(GRAPH_SUCCESS, opDesc->AddInputDesc(1, tedesc3));
  EXPECT_EQ(GRAPH_SUCCESS, opDesc->AddInputDesc(2, tedesc3));

  GeTensorDesc tedesc4(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(opDesc->UpdateInputDesc(1, tedesc4), GRAPH_SUCCESS);
  EXPECT_EQ(opDesc->UpdateInputDesc(4, tedesc4), GRAPH_FAILED);
  EXPECT_EQ(opDesc->UpdateInputDesc("w", tedesc4), GRAPH_SUCCESS);
  EXPECT_EQ(opDesc->UpdateInputDesc("weight", tedesc4), GRAPH_FAILED);

  GeTensorDesc getTe1 = opDesc->GetInputDesc(1);
  GeTensorDesc getTe2 = opDesc->GetInputDesc(4);
  GeTensorDesc getTe4 = opDesc->GetInputDesc("w");
  GeTensorDesc getTe3 = opDesc->GetInputDesc("weight");

  EXPECT_EQ(opDesc->GetInputNameByIndex(1), "w");
  EXPECT_EQ(opDesc->GetInputNameByIndex(3), "");

  auto vistor_in = opDesc->GetAllInputsDesc();
  EXPECT_EQ(vistor_in.size(), 3);

  auto input_size = opDesc->GetInputsSize();
  EXPECT_EQ(input_size, 3);
}

TEST_F(ge_test_opdesc, ge_test_opdesc_outputs) {
  string name = "Conv2d";
  string type = "Data";
  OpDescPtr opDesc = std::make_shared<OpDesc>(name, type);
  EXPECT_TRUE(opDesc);
  GeTensorDesc tedesc1(GeShape({1, 2, 3, 4}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(GRAPH_SUCCESS, opDesc->AddOutputDesc(tedesc1));
  GeTensorDesc tedesc2(GeShape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(GRAPH_SUCCESS, opDesc->AddOutputDesc("w", tedesc2));
  GeTensorDesc tedesc3(GeShape({8, 9, 10, 11}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(GRAPH_FAILED, opDesc->AddOutputDesc("w", tedesc3));

  GeTensorDesc tedesc4(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(opDesc->UpdateOutputDesc(1, tedesc4), GRAPH_SUCCESS);
  EXPECT_EQ(opDesc->UpdateOutputDesc(4, tedesc4), GRAPH_FAILED);
  EXPECT_EQ(opDesc->UpdateOutputDesc("w", tedesc4), GRAPH_SUCCESS);
  EXPECT_EQ(opDesc->UpdateOutputDesc("weight", tedesc4), GRAPH_FAILED);

  GeTensorDesc getTe1 = opDesc->GetOutputDesc(1);
  GeTensorDesc getTe2 = opDesc->GetOutputDesc(4);
  GeTensorDesc getTe4 = opDesc->GetOutputDesc("w");
  GeTensorDesc getTe3 = opDesc->GetOutputDesc("weight");

  auto vistor_in = opDesc->GetAllOutputsDesc();
  EXPECT_EQ(vistor_in.size(), 2);
}

TEST_F(ge_test_opdesc, ge_test_opdesc_attrs) {
  string name = "Conv2d";
  string type = "Data";
  OpDescPtr opDesc = std::make_shared<OpDesc>(name, type);
  EXPECT_TRUE(opDesc);
  auto defautlAttrSize = opDesc->GetAllAttrs().size();

  static const string PAD = "pad";
  static const string BIAS = "bias";

  opDesc->SetAttr(PAD, GeAttrValue::CreateFrom<GeAttrValue::INT>(6));
  opDesc->SetAttr(BIAS, GeAttrValue::CreateFrom<GeAttrValue::INT>(0));

  GeAttrValue at;
  EXPECT_EQ(opDesc->GetAttr(PAD, at), GRAPH_SUCCESS);
  int getatt = -1;
  at.GetValue<GeAttrValue::INT>(getatt);
  EXPECT_EQ(getatt, 6);
  EXPECT_EQ(opDesc->GetAttr("xxx", at), GRAPH_FAILED);
  EXPECT_EQ(opDesc->GetAttr(BIAS, at), GRAPH_SUCCESS);
  EXPECT_EQ(opDesc->GetAttr("bia", at), GRAPH_FAILED);
  EXPECT_TRUE(opDesc->HasAttr(BIAS));
  EXPECT_FALSE(opDesc->HasAttr("xxx"));

  EXPECT_EQ(2, opDesc->GetAllAttrs().size() - defautlAttrSize);
  EXPECT_EQ(opDesc->DelAttr("xxx"), GRAPH_FAILED);
  EXPECT_EQ(opDesc->DelAttr(PAD), GRAPH_SUCCESS);
  EXPECT_EQ(1, opDesc->GetAllAttrs().size() - defautlAttrSize);
}

graphStatus InferFunctionStub(Operator &op) { return GRAPH_FAILED; }

TEST_F(ge_test_opdesc, ge_test_opdesc_call_infer_func_failed) {
  GeTensorDesc ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT16);
  auto addn_op_desc = std::make_shared<OpDesc>("AddN", "AddN");
  addn_op_desc->AddInputDesc(ge_tensor_desc);
  addn_op_desc->AddOutputDesc(ge_tensor_desc);
  addn_op_desc->AddInferFunc(InferFunctionStub);
  auto graph = std::make_shared<ComputeGraph>("test");
  auto addn_node = std::make_shared<Node>(addn_op_desc, graph);
  addn_node->Init();
  Operator op = OpDescUtils::CreateOperatorFromNode(addn_node);

  graphStatus ret = addn_op_desc->CallInferFunc(op);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

graphStatus InferFunctionSuccessStub(Operator &op) { return GRAPH_SUCCESS; }

TEST_F(ge_test_opdesc, ge_test_opdesc_call_infer_func_success) {
  auto addn_op_desc = std::make_shared<OpDesc>("AddN", "AddN");
  addn_op_desc->AddInferFunc(InferFunctionSuccessStub);
  auto graph = std::make_shared<ComputeGraph>("test");
  auto addn_node = std::make_shared<Node>(addn_op_desc, graph);
  addn_node->Init();
  Operator op = OpDescUtils::CreateOperatorFromNode(addn_node);

  graphStatus ret = addn_op_desc->CallInferFunc(op);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_opdesc, ge_test_opdesc_infer_shape_and_type) {
  auto addn_op_desc = std::make_shared<OpDesc>("name", "type");
  graphStatus ret = addn_op_desc->InferShapeAndType();
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_opdesc, default_infer_format_success) {
  auto addn_op_desc = std::make_shared<OpDesc>("name", "type");
  std::function<graphStatus(Operator &)> func = nullptr;
  addn_op_desc->AddInferFormatFunc(func);
  auto fun1 = addn_op_desc->GetInferFormatFunc();
  graphStatus ret = addn_op_desc->DefaultInferFormat();
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_opdesc, call_infer_format_func_success) {
  auto addn_op_desc = std::make_shared<OpDesc>("name", "type");
  Operator op;
  graphStatus ret = addn_op_desc->CallInferFormatFunc(op);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_opdesc, AddDynamicOutputDesc) {
  OpDescPtr descPtr = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(descPtr->AddDynamicOutputDesc("x", 1, false), GRAPH_SUCCESS);
  EXPECT_EQ(descPtr->AddDynamicOutputDesc("x1", 1, false), GRAPH_SUCCESS);
  EXPECT_EQ(descPtr->AddDynamicOutputDesc("x", 1, false), GRAPH_FAILED);

  OpDescPtr descPtr2 = std::make_shared<OpDesc>("name2", "type2");
  EXPECT_EQ(descPtr2->AddDynamicOutputDesc("x", 1), GRAPH_SUCCESS);
}
