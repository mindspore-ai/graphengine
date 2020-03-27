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
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#define private public
#define protected public
#include "graph/model_serialize.h"

#include "graph/detail/model_serialize_imp.h"
#include "graph/ge_attr_value.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#undef private
#undef protected

#include "proto/ge_ir.pb.h"

using namespace ge;
using namespace std;
using std::string;
using std::vector;

bool LinkEdge(NodePtr srcNode, int32_t srcIndex, NodePtr dstNode, int32_t dstIndex) {
  if (srcIndex >= 0) {
    auto srcAnchor = srcNode->GetOutDataAnchor(srcIndex);
    auto dstAnchor = dstNode->GetInDataAnchor(dstIndex);
    srcAnchor->LinkTo(dstAnchor);
  } else {
    auto srcAnchor = srcNode->GetOutControlAnchor();
    auto dstAnchor = dstNode->GetInControlAnchor();
    srcAnchor->LinkTo(dstAnchor);
  }
}

NodePtr CreateNode(OpDescPtr op, ComputeGraphPtr ownerGraph) { return ownerGraph->AddNode(op); }

void CompareShape(const vector<int64_t> &shape1, const vector<int64_t> &shape2) {
  EXPECT_EQ(shape1.size(), shape2.size());
  if (shape1.size() == shape2.size()) {
    for (int i = 0; i < shape1.size(); i++) {
      EXPECT_EQ(shape1[i], shape2[i]);
    }
  }
}

template <typename T>
void CompareList(const vector<T> &val1, const vector<T> &val2) {
  EXPECT_EQ(val1.size(), val2.size());
  if (val1.size() == val2.size()) {
    for (int i = 0; i < val1.size(); i++) {
      EXPECT_EQ(val1[i], val2[i]);
    }
  }
}

static bool NamedAttrsSimpleCmp(const GeAttrValue &left, const GeAttrValue &right) {
  GeAttrValue::NamedAttrs val1, val2;
  left.GetValue<GeAttrValue::NamedAttrs>(val1);
  right.GetValue<GeAttrValue::NamedAttrs>(val2);
  if (val1.GetName() != val2.GetName()) {
    return false;
  }
  auto attrs1 = val1.GetAllAttrs();
  auto attrs2 = val2.GetAllAttrs();
  if (attrs1.size() != attrs1.size()) {
    return false;
  }

  for (auto it : attrs1) {
    auto it2 = attrs2.find(it.first);
    if (it2 == attrs2.end()) {  // simple check
      return false;
    }
    if (it.second.GetValueType() != it2->second.GetValueType()) {
      return false;
    }
    switch (it.second.GetValueType()) {
      case GeAttrValue::VT_INT: {
        int64_t i1 = 0, i2 = 0;
        it.second.GetValue<GeAttrValue::INT>(i1);
        it2->second.GetValue<GeAttrValue::INT>(i2);
        if (i1 != i2) {
          return false;
        }
      }
      case GeAttrValue::VT_FLOAT: {
        GeAttrValue::FLOAT i1 = 0, i2 = 0;
        it.second.GetValue<GeAttrValue::FLOAT>(i1);
        it2->second.GetValue<GeAttrValue::FLOAT>(i2);
        if (i1 != i2) {
          return false;
        }
      }
      case GeAttrValue::VT_STRING: {
        string i1, i2;
        it.second.GetValue<GeAttrValue::STR>(i1);
        it2->second.GetValue<GeAttrValue::STR>(i2);
        if (i1 != i2) {
          return false;
        }
      }
      case GeAttrValue::VT_BOOL: {
        bool i1 = false, i2 = false;
        it.second.GetValue<GeAttrValue::BOOL>(i1);
        it2->second.GetValue<GeAttrValue::BOOL>(i2);
        if (i1 != i2) {
          return false;
        }
      }
    }
  }
  return true;
}

static GeAttrValue::NamedAttrs CreateNamedAttrs(const string &name, std::map<string, GeAttrValue> map) {
  GeAttrValue::NamedAttrs namedAttrs;
  namedAttrs.SetName(name);
  for (auto it : map) {
    namedAttrs.SetAttr(it.first, it.second);
  }
  return namedAttrs;
}

TEST(UTEST_ge_model_serialize, simple) {
  Model model("model_name", "custom version3.0");
  model.SetAttr("model_key1", GeAttrValue::CreateFrom<GeAttrValue::INT>(123));
  model.SetAttr("model_key2", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(456.78f));
  model.SetAttr("model_key3", GeAttrValue::CreateFrom<GeAttrValue::STR>("abcd"));
  model.SetAttr("model_key4", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
  model.SetAttr("model_key5", GeAttrValue::CreateFrom<GeAttrValue::LIST_FLOAT>({456.78f, 998.90f}));
  model.SetAttr("model_key6", GeAttrValue::CreateFrom<GeAttrValue::LIST_STR>({"abcd", "happy"}));
  model.SetAttr("model_key7", GeAttrValue::CreateFrom<GeAttrValue::BOOL>(false));
  model.SetAttr("model_key8", GeAttrValue::CreateFrom<GeAttrValue::LIST_BOOL>({true, false}));

  auto computeGraph = std::make_shared<ComputeGraph>("graph_name");

  // input
  auto inputOp = std::make_shared<OpDesc>("input", "Input");
  inputOp->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
  auto input = CreateNode(inputOp, computeGraph);
  // w1
  auto w1Op = std::make_shared<OpDesc>("w1", "ConstOp");
  w1Op->AddOutputDesc(GeTensorDesc(GeShape({12, 2, 64, 64, 16}), FORMAT_NC1HWC0, DT_FLOAT16));
  auto w1 = CreateNode(w1Op, computeGraph);

  // node1
  auto node1Op = std::make_shared<OpDesc>("node1", "Conv2D");
  node1Op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
  node1Op->AddInputDesc(GeTensorDesc(GeShape({12, 2, 64, 64, 16}), FORMAT_NC1HWC0, DT_FLOAT16));
  node1Op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
  auto node1 = CreateNode(node1Op, computeGraph);

  // Attr set
  node1Op->SetAttr("node_key1", GeAttrValue::CreateFrom<GeAttrValue::BYTES>(Buffer(10)));
  node1Op->SetAttr("node_key2", GeAttrValue::CreateFrom<GeAttrValue::LIST_BYTES>({Buffer(20), Buffer(30)}));
  auto namedAttrs1 = GeAttrValue::CreateFrom<GeAttrValue::NAMED_ATTRS>(
      CreateNamedAttrs("my_name", {{"int_val", GeAttrValue::CreateFrom<int64_t>(123)},
                                   {"str_val", GeAttrValue::CreateFrom<string>("abc")},
                                   {"float_val", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(345.345)}}));

  node1Op->SetAttr("node_key3", std::move(namedAttrs1));
  auto listNamedAttrs = GeAttrValue::CreateFrom<GeAttrValue::LIST_NAMED_ATTRS>(
      {CreateNamedAttrs("my_name", {{"int_val", GeAttrValue::CreateFrom<int64_t>(123)},
                                    {"float_val", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(345.345)}}),
       CreateNamedAttrs("my_name2", {{"str_val", GeAttrValue::CreateFrom<string>("abc")},
                                     {"float_val", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(345.345)}})});
  node1Op->SetAttr("node_key4", std::move(listNamedAttrs));
  // tensor
  auto tensorData1 = "qwertyui";
  auto tensor1 =
      std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT8), (uint8_t *)tensorData1, 8);
  auto tensorData2 = "asdfqwertyui";
  auto tensor2 =
      std::make_shared<GeTensor>(GeTensorDesc(GeShape({3, 2, 2}), FORMAT_ND, DT_UINT8), (uint8_t *)tensorData2, 12);
  auto tensorData3 = "ghjkasdfqwertyui";
  auto tensor3 =
      std::make_shared<GeTensor>(GeTensorDesc(GeShape({4, 2, 2}), FORMAT_ND, DT_UINT16), (uint8_t *)tensorData3, 16);
  node1Op->SetAttr("node_key5", GeAttrValue::CreateFrom<GeAttrValue::TENSOR>(tensor1));
  node1Op->SetAttr("node_key6", GeAttrValue::CreateFrom<GeAttrValue::LIST_TENSOR>({tensor2, tensor3}));

  auto tensorDesc = GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT16);
  TensorUtils::SetSize(tensorDesc, 100);
  node1Op->SetAttr("node_key7", GeAttrValue::CreateFrom<GeAttrValue::TENSOR_DESC>(tensorDesc));
  node1Op->SetAttr("node_key8", GeAttrValue::CreateFrom<GeAttrValue::LIST_TENSOR_DESC>(
                                    {GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT32),
                                     GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_UINT32),
                                     GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT64),
                                     GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_UINT64),
                                     GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_BOOL),
                                     GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_DOUBLE)}));

  LinkEdge(input, 0, node1, 0);
  LinkEdge(w1, 0, node1, 1);

  Graph graph = GraphUtils::CreateGraphFromComputeGraph(computeGraph);
  model.SetGraph(graph);

  Buffer buffer;
  ASSERT_EQ(model.Save(buffer), GRAPH_SUCCESS);
  EXPECT_TRUE(buffer.GetData() != nullptr);

  Model model2;
  ASSERT_EQ(Model::Load(buffer.GetData(), buffer.GetSize(), model2), GRAPH_SUCCESS);
  EXPECT_EQ(model2.GetName(), "model_name");
  GeAttrValue::INT modelVal1;
  AttrUtils::GetInt(&model2, "model_key1", modelVal1);
  EXPECT_EQ(modelVal1, 123);

  GeAttrValue::FLOAT modelVal2;
  AttrUtils::GetFloat(&model2, "model_key2", modelVal2);
  EXPECT_EQ(modelVal2, (float)456.78f);

  GeAttrValue::STR modelVal3;
  AttrUtils::GetStr(&model2, "model_key3", modelVal3);
  EXPECT_EQ(modelVal3, "abcd");

  GeAttrValue::LIST_INT modelVal4;
  AttrUtils::GetListInt(&model2, "model_key4", modelVal4);
  CompareList(modelVal4, {123, 456});

  GeAttrValue::LIST_FLOAT modelVal5;
  AttrUtils::GetListFloat(&model2, "model_key5", modelVal5);
  CompareList(modelVal5, {456.78f, 998.90f});

  GeAttrValue::LIST_STR modelVal6;
  AttrUtils::GetListStr(&model2, "model_key6", modelVal6);
  CompareList(modelVal6, {"abcd", "happy"});

  GeAttrValue::BOOL modelVal7;
  EXPECT_EQ(AttrUtils::GetBool(&model2, "model_key7", modelVal7), true);
  EXPECT_EQ(modelVal7, false);

  GeAttrValue::LIST_BOOL modelVal8;
  AttrUtils::GetListBool(&model2, "model_key8", modelVal8);
  CompareList(modelVal8, {true, false});

  auto graph2 = model2.GetGraph();
  const auto &s_graph = GraphUtils::GetComputeGraph(graph2);
  ASSERT_TRUE(s_graph != nullptr);
  auto s_nodes = s_graph->GetDirectNode();
  ASSERT_EQ(3, s_nodes.size());

  auto s_input = s_nodes.at(0);
  auto s_w1 = s_nodes.at(1);
  auto s_nod1 = s_nodes.at(2);
  {
    auto s_op = s_input->GetOpDesc();
    EXPECT_EQ(s_op->GetName(), "input");
    EXPECT_EQ(s_op->GetType(), "Input");
    auto s_input_descs = s_op->GetAllInputsDesc();
    ASSERT_EQ(s_input_descs.size(), 0);
    auto s_output_descs = s_op->GetAllOutputsDesc();
    ASSERT_EQ(s_output_descs.size(), 1);
    auto desc1 = s_output_descs.at(0);
    EXPECT_EQ(desc1.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(desc1.GetDataType(), DT_FLOAT);
    CompareShape(desc1.GetShape().GetDims(), vector<int64_t>{12, 32, 64, 64});

    auto outAnchor = s_input->GetOutDataAnchor(0);
    auto peerAnchors = outAnchor->GetPeerInDataAnchors();
    ASSERT_EQ(peerAnchors.size(), 1);
    auto peerAnchor = peerAnchors.at(0);
    ASSERT_EQ(peerAnchor->GetIdx(), 0);
    ASSERT_EQ(peerAnchor->GetOwnerNode(), s_nod1);
  }

  {
    auto s_op = s_w1->GetOpDesc();
    EXPECT_EQ(s_op->GetName(), "w1");
    EXPECT_EQ(s_op->GetType(), "ConstOp");
    auto s_input_descs = s_op->GetAllInputsDesc();
    ASSERT_EQ(s_input_descs.size(), 0);
    auto s_output_descs = s_op->GetAllOutputsDesc();
    ASSERT_EQ(s_output_descs.size(), 1);
    auto desc1 = s_output_descs.at(0);
    EXPECT_EQ(desc1.GetFormat(), FORMAT_NC1HWC0);
    EXPECT_EQ(desc1.GetDataType(), DT_FLOAT16);
    CompareShape(desc1.GetShape().GetDims(), vector<int64_t>{12, 2, 64, 64, 16});

    auto outAnchor = s_w1->GetOutDataAnchor(0);
    auto peerAnchors = outAnchor->GetPeerInDataAnchors();
    ASSERT_EQ(peerAnchors.size(), 1);
    auto peerAnchor = peerAnchors.at(0);
    ASSERT_EQ(peerAnchor->GetIdx(), 1);
    ASSERT_EQ(peerAnchor->GetOwnerNode(), s_nod1);
  }
  {
    auto s_op = s_nod1->GetOpDesc();
    EXPECT_EQ(s_op->GetName(), "node1");
    EXPECT_EQ(s_op->GetType(), "Conv2D");
    auto s_input_descs = s_op->GetAllInputsDesc();
    ASSERT_EQ(s_input_descs.size(), 2);

    auto desc1 = s_input_descs.at(0);
    EXPECT_EQ(desc1.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(desc1.GetDataType(), DT_FLOAT);
    CompareShape(desc1.GetShape().GetDims(), vector<int64_t>{12, 32, 64, 64});

    auto desc2 = s_input_descs.at(1);
    EXPECT_EQ(desc2.GetFormat(), FORMAT_NC1HWC0);
    EXPECT_EQ(desc2.GetDataType(), DT_FLOAT16);
    CompareShape(desc2.GetShape().GetDims(), vector<int64_t>{12, 2, 64, 64, 16});

    auto s_output_descs = s_op->GetAllOutputsDesc();
    ASSERT_EQ(s_output_descs.size(), 1);
    auto desc3 = s_output_descs.at(0);
    EXPECT_EQ(desc3.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(desc3.GetDataType(), DT_FLOAT);
    CompareShape(desc3.GetShape().GetDims(), vector<int64_t>{12, 32, 64, 64});

    auto outAnchor = s_nod1->GetOutDataAnchor(0);
    auto peerAnchors = outAnchor->GetPeerInDataAnchors();
    ASSERT_EQ(peerAnchors.size(), 0);

    // node attrs
    GeAttrValue::BYTES nodeVal1;
    AttrUtils::GetBytes(s_op, "node_key1", nodeVal1);
    ASSERT_EQ(nodeVal1.GetSize(), 10);

    GeAttrValue::LIST_BYTES nodeVal2;
    AttrUtils::GetListBytes(s_op, "node_key2", nodeVal2);
    ASSERT_EQ(nodeVal2.size(), 2);
    ASSERT_EQ(nodeVal2[0].GetSize(), 20);
    ASSERT_EQ(nodeVal2[1].GetSize(), 30);

    GeAttrValue s_namedAttrs;
    s_op->GetAttr("node_key3", s_namedAttrs);
    EXPECT_TRUE(NamedAttrsSimpleCmp(s_namedAttrs, namedAttrs1));

    GeAttrValue s_listNamedAttrs;
    s_op->GetAttr("node_key4", s_listNamedAttrs);
    EXPECT_TRUE(NamedAttrsSimpleCmp(s_listNamedAttrs, listNamedAttrs));

    ConstGeTensorPtr s_tensor;
    AttrUtils::GetTensor(s_op, "node_key5", s_tensor);
    ASSERT_TRUE(s_tensor != nullptr);
    string str((char *)s_tensor->GetData().data(), s_tensor->GetData().size());
    EXPECT_EQ(str, "qwertyui");

    vector<ConstGeTensorPtr> s_listTensor;
    AttrUtils::GetListTensor(s_op, "node_key6", s_listTensor);
    ASSERT_EQ(s_listTensor.size(), 2);
    string str2((char *)s_listTensor[0]->GetData().data(), s_listTensor[0]->GetData().size());
    EXPECT_EQ(str2, "asdfqwertyui");
    string str3((char *)s_listTensor[1]->GetData().data(), s_listTensor[1]->GetData().size());
    EXPECT_EQ(str3, "ghjkasdfqwertyui");

    GeTensorDesc s_tensorDesc;
    AttrUtils::GetTensorDesc(s_op, "node_key7", s_tensorDesc);
    EXPECT_EQ(s_tensorDesc.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(s_tensorDesc.GetDataType(), DT_INT16);
    uint32_t size = 0;
    TensorUtils::GetSize(s_tensorDesc, size);
    EXPECT_EQ(size, 100);

    vector<GeTensorDesc> s_listTensorDesc;
    AttrUtils::GetListTensorDesc(s_op, "node_key8", s_listTensorDesc);
    ASSERT_EQ(s_listTensorDesc.size(), 6);
    EXPECT_EQ(s_listTensorDesc[0].GetDataType(), DT_INT32);
    EXPECT_EQ(s_listTensorDesc[1].GetDataType(), DT_UINT32);
    EXPECT_EQ(s_listTensorDesc[2].GetDataType(), DT_INT64);
    EXPECT_EQ(s_listTensorDesc[3].GetDataType(), DT_UINT64);
    EXPECT_EQ(s_listTensorDesc[4].GetDataType(), DT_BOOL);
    EXPECT_EQ(s_listTensorDesc[5].GetDataType(), DT_DOUBLE);
  }
}

TEST(UTEST_ge_model_serialize, OpDesc) {
  // node1Op
  auto node1Op = std::make_shared<OpDesc>("node1", "Conv2D");
  node1Op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
  node1Op->AddInputDesc(GeTensorDesc(GeShape({12, 2, 64, 64, 16}), FORMAT_NC1HWC0, DT_FLOAT16));
  node1Op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

  // Attr set
  node1Op->SetAttr("node_key1", GeAttrValue::CreateFrom<GeAttrValue::BYTES>(Buffer(10)));
  node1Op->SetAttr("node_key2", GeAttrValue::CreateFrom<GeAttrValue::LIST_BYTES>({Buffer(20), Buffer(30)}));
  auto namedAttrs1 = GeAttrValue::CreateFrom<GeAttrValue::NAMED_ATTRS>(
      CreateNamedAttrs("my_name", {{"int_val", GeAttrValue::CreateFrom<int64_t>(123)},
                                   {"str_val", GeAttrValue::CreateFrom<string>("abc")},
                                   {"float_val", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(345.345)}}));

  node1Op->SetAttr("node_key3", std::move(namedAttrs1));
  auto listNamedAttrs = GeAttrValue::CreateFrom<GeAttrValue::LIST_NAMED_ATTRS>(
      {CreateNamedAttrs("my_name", {{"int_val", GeAttrValue::CreateFrom<int64_t>(123)},
                                    {"float_val", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(345.345)}}),
       CreateNamedAttrs("my_name2", {{"str_val", GeAttrValue::CreateFrom<string>("abc")},
                                     {"float_val", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(345.345)}})});
  node1Op->SetAttr("node_key4", std::move(listNamedAttrs));

  ModelSerialize modelSerialize;
  Buffer buffer = modelSerialize.SerializeOpDesc(node1Op);
  EXPECT_TRUE(buffer.GetData() != nullptr);

  auto s_op = modelSerialize.UnserializeOpDesc(buffer.GetData(), buffer.GetSize());
  ASSERT_TRUE(s_op != nullptr);

  {
    EXPECT_EQ(s_op->GetName(), "node1");
    EXPECT_EQ(s_op->GetType(), "Conv2D");
    auto s_input_descs = s_op->GetAllInputsDesc();
    ASSERT_EQ(s_input_descs.size(), 2);

    auto desc1 = s_input_descs.at(0);
    EXPECT_EQ(desc1.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(desc1.GetDataType(), DT_FLOAT);
    CompareShape(desc1.GetShape().GetDims(), vector<int64_t>{12, 32, 64, 64});

    auto desc2 = s_input_descs.at(1);
    EXPECT_EQ(desc2.GetFormat(), FORMAT_NC1HWC0);
    EXPECT_EQ(desc2.GetDataType(), DT_FLOAT16);
    CompareShape(desc2.GetShape().GetDims(), vector<int64_t>{12, 2, 64, 64, 16});

    auto s_output_descs = s_op->GetAllOutputsDesc();
    ASSERT_EQ(s_output_descs.size(), 1);
    auto desc3 = s_output_descs.at(0);
    EXPECT_EQ(desc3.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(desc3.GetDataType(), DT_FLOAT);
    CompareShape(desc3.GetShape().GetDims(), vector<int64_t>{12, 32, 64, 64});

    // node attrs
    GeAttrValue::BYTES nodeVal1;
    AttrUtils::GetBytes(s_op, "node_key1", nodeVal1);
    ASSERT_EQ(nodeVal1.GetSize(), 10);

    GeAttrValue::LIST_BYTES nodeVal2;
    AttrUtils::GetListBytes(s_op, "node_key2", nodeVal2);
    ASSERT_EQ(nodeVal2.size(), 2);
    ASSERT_EQ(nodeVal2[0].GetSize(), 20);
    ASSERT_EQ(nodeVal2[1].GetSize(), 30);

    GeAttrValue s_namedAttrs;
    s_op->GetAttr("node_key3", s_namedAttrs);
    EXPECT_TRUE(NamedAttrsSimpleCmp(s_namedAttrs, namedAttrs1));

    GeAttrValue s_listNamedAttrs;
    s_op->GetAttr("node_key4", s_listNamedAttrs);
    EXPECT_TRUE(NamedAttrsSimpleCmp(s_listNamedAttrs, listNamedAttrs));
  }
}

TEST(UTEST_ge_model_serialize, OpDescAsAttrValue) {
  // node1Op
  auto node1Op = std::make_shared<OpDesc>("node1", "Conv2D");
  node1Op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
  node1Op->AddInputDesc(GeTensorDesc(GeShape({12, 2, 64, 64, 16}), FORMAT_NC1HWC0, DT_FLOAT16));
  node1Op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

  // Attr set
  node1Op->SetAttr("node_key1", GeAttrValue::CreateFrom<GeAttrValue::BYTES>(Buffer(10)));
  node1Op->SetAttr("node_key2", GeAttrValue::CreateFrom<GeAttrValue::LIST_BYTES>({Buffer(20), Buffer(30)}));
  auto namedAttrs1 = GeAttrValue::CreateFrom<GeAttrValue::NAMED_ATTRS>(
      CreateNamedAttrs("my_name", {{"int_val", GeAttrValue::CreateFrom<int64_t>(123)},
                                   {"str_val", GeAttrValue::CreateFrom<string>("abc")},
                                   {"float_val", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(345.345)}}));

  node1Op->SetAttr("node_key3", std::move(namedAttrs1));
  auto listNamedAttrs = GeAttrValue::CreateFrom<GeAttrValue::LIST_NAMED_ATTRS>(
      {CreateNamedAttrs("my_name", {{"int_val", GeAttrValue::CreateFrom<int64_t>(123)},
                                    {"float_val", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(345.345)}}),
       CreateNamedAttrs("my_name2", {{"str_val", GeAttrValue::CreateFrom<string>("abc")},
                                     {"float_val", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(345.345)}})});
  node1Op->SetAttr("node_key4", std::move(listNamedAttrs));

  Model model;
  EXPECT_TRUE(AttrUtils::SetListOpDesc(&model, "my_key", vector<OpDescPtr>{node1Op}));
  EXPECT_TRUE(AttrUtils::SetListInt(&model, "my_key2", {123}));
  EXPECT_TRUE(AttrUtils::SetListBytes(&model, "my_key3", {Buffer(100)}));

  vector<OpDescPtr> opList;
  EXPECT_FALSE(AttrUtils::GetListOpDesc(&model, "my_error_key", opList));
  EXPECT_FALSE(AttrUtils::GetListOpDesc(&model, "my_key2", opList));

  EXPECT_TRUE(AttrUtils::GetListOpDesc(&model, "my_key", opList));

  ASSERT_TRUE(opList.size() > 0);
  auto s_op = opList[0];

  {
    EXPECT_EQ(s_op->GetName(), "node1");
    EXPECT_EQ(s_op->GetType(), "Conv2D");
    auto s_input_descs = s_op->GetAllInputsDesc();
    ASSERT_EQ(s_input_descs.size(), 2);

    auto desc1 = s_input_descs.at(0);
    EXPECT_EQ(desc1.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(desc1.GetDataType(), DT_FLOAT);
    CompareShape(desc1.GetShape().GetDims(), vector<int64_t>{12, 32, 64, 64});

    auto desc2 = s_input_descs.at(1);
    EXPECT_EQ(desc2.GetFormat(), FORMAT_NC1HWC0);
    EXPECT_EQ(desc2.GetDataType(), DT_FLOAT16);
    CompareShape(desc2.GetShape().GetDims(), vector<int64_t>{12, 2, 64, 64, 16});

    auto s_output_descs = s_op->GetAllOutputsDesc();
    ASSERT_EQ(s_output_descs.size(), 1);
    auto desc3 = s_output_descs.at(0);
    EXPECT_EQ(desc3.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(desc3.GetDataType(), DT_FLOAT);
    CompareShape(desc3.GetShape().GetDims(), vector<int64_t>{12, 32, 64, 64});

    // node attrs
    GeAttrValue::BYTES nodeVal1;
    AttrUtils::GetBytes(s_op, "node_key1", nodeVal1);
    ASSERT_EQ(nodeVal1.GetSize(), 10);

    GeAttrValue::LIST_BYTES nodeVal2;
    AttrUtils::GetListBytes(s_op, "node_key2", nodeVal2);
    ASSERT_EQ(nodeVal2.size(), 2);
    ASSERT_EQ(nodeVal2[0].GetSize(), 20);
    ASSERT_EQ(nodeVal2[1].GetSize(), 30);

    GeAttrValue s_namedAttrs;
    s_op->GetAttr("node_key3", s_namedAttrs);
    EXPECT_TRUE(NamedAttrsSimpleCmp(s_namedAttrs, namedAttrs1));

    GeAttrValue s_listNamedAttrs;
    s_op->GetAttr("node_key4", s_listNamedAttrs);
    EXPECT_TRUE(NamedAttrsSimpleCmp(s_listNamedAttrs, listNamedAttrs));
  }
}

TEST(UTEST_ge_model_serialize, test_subGraph) {
  Model model("model_name", "custom version3.0");
  {
    auto computeGraph = std::make_shared<ComputeGraph>("graph_name");
    // input
    auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
    inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto input = CreateNode(inputOp, computeGraph);
    Graph graph = GraphUtils::CreateGraphFromComputeGraph(computeGraph);
    model.SetGraph(graph);

    auto subComputeGraph = std::make_shared<ComputeGraph>("sub_graph");
    // input
    auto subGraphInputOp = std::make_shared<OpDesc>("sub_graph_test", "TestOp2");
    subGraphInputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto subGraphInput = CreateNode(subGraphInputOp, subComputeGraph);

    AttrUtils::SetGraph(inputOp, "sub_graph", subComputeGraph);
  }

  ModelSerialize serialize;
  auto buffer = serialize.SerializeModel(model);
  ASSERT_GE(buffer.GetSize(), 0);
  ASSERT_GE(serialize.GetSerializeModelSize(model), 0);

  auto model2 = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
  ASSERT_TRUE(model2.GetGraph().IsValid());
  auto graph2 = GraphUtils::GetComputeGraph(model2.GetGraph());
  EXPECT_EQ(graph2->GetName(), "graph_name");
  auto nodes2 = graph2->GetDirectNode();
  ASSERT_EQ(nodes2.size(), 1);
  auto node2 = nodes2.at(0);
  EXPECT_EQ(node2->GetName(), "test");
  auto node2Op = node2->GetOpDesc();
  EXPECT_EQ(node2Op->GetType(), "TestOp");
  auto node2InputDescs = node2Op->GetAllInputsDesc();
  ASSERT_EQ(node2InputDescs.size(), 1);
  auto node2InputDesc = node2InputDescs.at(0);

  ComputeGraphPtr subComputeGraph2;
  ASSERT_TRUE(AttrUtils::GetGraph(node2Op, "sub_graph", subComputeGraph2));
  EXPECT_EQ(subComputeGraph2->GetName(), "sub_graph");
  auto subNodes2 = subComputeGraph2->GetDirectNode();
  ASSERT_EQ(subNodes2.size(), 1);
  auto subNode2 = subNodes2.at(0);
  EXPECT_EQ(subNode2->GetName(), "sub_graph_test");
  ASSERT_EQ(subNode2->GetAllInDataAnchors().size(), 1);
  auto subNodeOp2 = subNode2->GetOpDesc();
  EXPECT_EQ(subNodeOp2->GetType(), "TestOp2");
  ASSERT_EQ(subNodeOp2->GetAllInputsDesc().size(), 1);
  auto subNode2InputDesc = subNodeOp2->GetAllInputsDesc().at(0);
  EXPECT_EQ(subNode2InputDesc.GetShape().GetDim(1), 32);
}

TEST(UTEST_ge_model_serialize, test_listSubGraph) {
  Model model("model_name", "custom version3.0");
  {
    auto computeGraph = std::make_shared<ComputeGraph>("graph_name");
    // input
    auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
    inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto input = CreateNode(inputOp, computeGraph);
    Graph graph = GraphUtils::CreateGraphFromComputeGraph(computeGraph);
    model.SetGraph(graph);

    auto subComputeGraph1 = std::make_shared<ComputeGraph>("sub_graph1");
    // input
    auto subGraphInputOp1 = std::make_shared<OpDesc>("sub_graph_test1", "TestOp2");
    subGraphInputOp1->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto subGraphInput1 = CreateNode(subGraphInputOp1, subComputeGraph1);

    auto subComputeGraph2 = std::make_shared<ComputeGraph>("sub_graph2");
    // input
    auto subGraphInputOp2 = std::make_shared<OpDesc>("sub_graph_test2", "TestOp2");
    subGraphInputOp2->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto subGraphInput2 = CreateNode(subGraphInputOp2, subComputeGraph2);

    AttrUtils::SetListGraph(inputOp, "sub_graph", vector<ComputeGraphPtr>{subComputeGraph1, subComputeGraph2});
  }

  ModelSerialize serialize;
  auto buffer = serialize.SerializeModel(model);
  ASSERT_GE(buffer.GetSize(), 0);

  auto model2 = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
  ASSERT_TRUE(model2.GetGraph().IsValid());
  auto graph2 = GraphUtils::GetComputeGraph(model2.GetGraph());
  EXPECT_EQ(graph2->GetName(), "graph_name");
  auto nodes2 = graph2->GetDirectNode();
  ASSERT_EQ(nodes2.size(), 1);
  auto node2 = nodes2.at(0);
  auto node2Op = node2->GetOpDesc();

  vector<ComputeGraphPtr> listSubComputeGraph;
  ASSERT_TRUE(AttrUtils::GetListGraph(node2Op, "sub_graph", listSubComputeGraph));
  ASSERT_EQ(listSubComputeGraph.size(), 2);

  EXPECT_EQ(listSubComputeGraph[0]->GetName(), "sub_graph1");
  EXPECT_EQ(listSubComputeGraph[1]->GetName(), "sub_graph2");

  auto subNodes21 = listSubComputeGraph[0]->GetDirectNode();
  ASSERT_EQ(subNodes21.size(), 1);
  auto subNode21 = subNodes21.at(0);
  EXPECT_EQ(subNode21->GetName(), "sub_graph_test1");

  auto subNodes22 = listSubComputeGraph[1]->GetDirectNode();
  ASSERT_EQ(subNodes22.size(), 1);
  auto subNode22 = subNodes22.at(0);
  EXPECT_EQ(subNode22->GetName(), "sub_graph_test2");
}

TEST(UTEST_ge_model_serialize, test_Format) {
  Model model("model_name", "custom version3.0");
  {
    auto computeGraph = std::make_shared<ComputeGraph>("graph_name");
    // input
    auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
    inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NHWC, DT_FLOAT));
    inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_ND, DT_FLOAT));
    inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NC1HWC0, DT_FLOAT));
    inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FRACTAL_Z, DT_FLOAT));
    inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NC1C0HWPAD, DT_FLOAT));
    inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NHWC1C0, DT_FLOAT));
    inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FSR_NCHW, DT_FLOAT));
    inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FRACTAL_DECONV, DT_FLOAT));
    inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_BN_WEIGHT, DT_FLOAT));
    inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_CHWN, DT_FLOAT));
    inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FILTER_HWCK, DT_FLOAT));
    inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FRACTAL_Z_C04, DT_FLOAT));
    auto input = CreateNode(inputOp, computeGraph);
    model.SetGraph(GraphUtils::CreateGraphFromComputeGraph(computeGraph));
  }
  ModelSerialize serialize;
  auto buffer = serialize.SerializeModel(model);
  ASSERT_GE(buffer.GetSize(), 0);
  auto model2 = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
  ASSERT_TRUE(model2.GetGraph().IsValid());

  auto graph = model2.GetGraph();
  ASSERT_TRUE(GraphUtils::GetComputeGraph(graph) != nullptr);
  ASSERT_EQ(GraphUtils::GetComputeGraph(graph)->GetDirectNode().size(), 1);

  auto op = GraphUtils::GetComputeGraph(graph)->GetDirectNode().at(0)->GetOpDesc();
  auto inputDescs = op->GetAllInputsDesc();
  ASSERT_EQ(inputDescs.size(), 13);
  EXPECT_EQ(inputDescs.at(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(inputDescs.at(1).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(inputDescs.at(2).GetFormat(), FORMAT_ND);
  EXPECT_EQ(inputDescs.at(3).GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(inputDescs.at(4).GetFormat(), FORMAT_FRACTAL_Z);
  EXPECT_EQ(inputDescs.at(5).GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(inputDescs.at(6).GetFormat(), FORMAT_NHWC1C0);
  EXPECT_EQ(inputDescs.at(7).GetFormat(), FORMAT_FSR_NCHW);
  EXPECT_EQ(inputDescs.at(8).GetFormat(), FORMAT_FRACTAL_DECONV);
  EXPECT_EQ(inputDescs.at(9).GetFormat(), FORMAT_BN_WEIGHT);
  EXPECT_EQ(inputDescs.at(10).GetFormat(), FORMAT_CHWN);
  EXPECT_EQ(inputDescs.at(11).GetFormat(), FORMAT_FILTER_HWCK);
  EXPECT_EQ(inputDescs.at(12).GetFormat(), FORMAT_FRACTAL_Z_C04);
}

TEST(UTEST_ge_model_serialize, test_ControlEdge) {
  Model model("model_name", "custom version3.0");
  {
    auto computeGraph = std::make_shared<ComputeGraph>("graph_name");
    // input
    auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
    inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto input = CreateNode(inputOp, computeGraph);
    // sink
    auto sinkOp = std::make_shared<OpDesc>("test2", "Sink");
    auto sink = CreateNode(sinkOp, computeGraph);
    LinkEdge(sink, -1, input, -1);

    // sink2
    auto sinkOp2 = std::make_shared<OpDesc>("test3", "Sink");
    auto sink2 = CreateNode(sinkOp2, computeGraph);
    LinkEdge(sink2, -1, input, -1);

    // dest
    auto destOp = std::make_shared<OpDesc>("test4", "Dest");
    auto dest = CreateNode(destOp, computeGraph);
    LinkEdge(input, -1, dest, -1);

    computeGraph->AddInputNode(sink);
    computeGraph->AddInputNode(sink2);
    computeGraph->AddOutputNode(dest);

    Graph graph = GraphUtils::CreateGraphFromComputeGraph(computeGraph);
    model.SetGraph(graph);
  }
  ModelSerialize serialize;
  auto buffer = serialize.SerializeModel(model);
  EXPECT_GE(buffer.GetSize(), 0);

  auto model2 = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
  ASSERT_TRUE(model2.GetGraph().IsValid());
  auto graph = GraphUtils::GetComputeGraph(model2.GetGraph());
  EXPECT_EQ(graph->GetName(), "graph_name");
  auto nodes = graph->GetDirectNode();
  ASSERT_EQ(nodes.size(), 4);

  auto node1 = nodes.at(0);
  auto sink = nodes.at(1);
  auto sink2 = nodes.at(2);
  auto dest = nodes.at(3);
  EXPECT_EQ(node1->GetName(), "test");
  EXPECT_EQ(sink->GetName(), "test2");
  ASSERT_EQ(node1->GetAllInDataAnchors().size(), 1);
  auto anchor1 = node1->GetAllInDataAnchors().at(0);
  EXPECT_EQ(anchor1->GetPeerAnchors().size(), 0);

  auto contorlInAnchor1 = node1->GetInControlAnchor();
  ASSERT_EQ(contorlInAnchor1->GetPeerAnchors().size(), 2);

  EXPECT_EQ(contorlInAnchor1->GetPeerAnchors().at(0)->GetOwnerNode(), sink);
  EXPECT_EQ(contorlInAnchor1->GetPeerAnchors().at(1)->GetOwnerNode(), sink2);

  auto contorlOutAnchor1 = node1->GetOutControlAnchor();
  ASSERT_EQ(contorlOutAnchor1->GetPeerAnchors().size(), 1);
  EXPECT_EQ(contorlOutAnchor1->GetPeerAnchors().at(0)->GetOwnerNode(), dest);

  auto inputNodes = graph->GetInputNodes();
  ASSERT_EQ(inputNodes.size(), 2);
  EXPECT_EQ(inputNodes.at(0), sink);
  EXPECT_EQ(inputNodes.at(1), sink2);

  auto outputNodes = graph->GetOutputNodes();
  ASSERT_EQ(outputNodes.size(), 1);
  EXPECT_EQ(outputNodes.at(0), dest);
}

TEST(UTEST_ge_model_serialize, test_SerializeGraph) {
  auto computeGraph = std::make_shared<ComputeGraph>("graph_name");
  {
    // input
    auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
    inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto input = CreateNode(inputOp, computeGraph);
    // sink
    auto sinkOp = std::make_shared<OpDesc>("test2", "Sink");
    auto sink = CreateNode(sinkOp, computeGraph);
    LinkEdge(sink, -1, input, -1);

    // sink2
    auto sinkOp2 = std::make_shared<OpDesc>("test3", "Sink");
    auto sink2 = CreateNode(sinkOp2, computeGraph);
    LinkEdge(sink2, -1, input, -1);

    // dest
    auto destOp = std::make_shared<OpDesc>("test4", "Dest");
    auto dest = CreateNode(destOp, computeGraph);
    LinkEdge(input, -1, dest, -1);

    computeGraph->AddInputNode(sink);
    computeGraph->AddInputNode(sink2);
    computeGraph->AddOutputNode(dest);
  }
  ModelSerialize serialize;
  auto buffer = serialize.SerializeGraph(computeGraph);
  EXPECT_GE(buffer.GetSize(), 0);

  auto graph = serialize.UnserializeGraph(buffer.GetData(), buffer.GetSize());
  ASSERT_TRUE(graph != nullptr);
  EXPECT_EQ(graph->GetName(), "graph_name");
  auto nodes = graph->GetDirectNode();
  ASSERT_EQ(nodes.size(), 4);

  auto node1 = nodes.at(0);
  auto sink = nodes.at(1);
  auto sink2 = nodes.at(2);
  auto dest = nodes.at(3);
  EXPECT_EQ(node1->GetName(), "test");
  EXPECT_EQ(sink->GetName(), "test2");
  ASSERT_EQ(node1->GetAllInDataAnchors().size(), 1);
  auto anchor1 = node1->GetAllInDataAnchors().at(0);
  EXPECT_EQ(anchor1->GetPeerAnchors().size(), 0);

  auto contorlInAnchor1 = node1->GetInControlAnchor();
  ASSERT_EQ(contorlInAnchor1->GetPeerAnchors().size(), 2);

  EXPECT_EQ(contorlInAnchor1->GetPeerAnchors().at(0)->GetOwnerNode(), sink);
  EXPECT_EQ(contorlInAnchor1->GetPeerAnchors().at(1)->GetOwnerNode(), sink2);

  auto contorlOutAnchor1 = node1->GetOutControlAnchor();
  ASSERT_EQ(contorlOutAnchor1->GetPeerAnchors().size(), 1);
  EXPECT_EQ(contorlOutAnchor1->GetPeerAnchors().at(0)->GetOwnerNode(), dest);

  auto inputNodes = graph->GetInputNodes();
  ASSERT_EQ(inputNodes.size(), 2);
  EXPECT_EQ(inputNodes.at(0), sink);
  EXPECT_EQ(inputNodes.at(1), sink2);

  auto outputNodes = graph->GetOutputNodes();
  ASSERT_EQ(outputNodes.size(), 1);
  EXPECT_EQ(outputNodes.at(0), dest);
}

TEST(UTEST_ge_model_serialize, test_invalid_Model) {
  {  // empty graph
    Model model("model_name", "custom version3.0");
    auto computeGraph = std::make_shared<ComputeGraph>("graph_name");

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    EXPECT_EQ(buffer.GetSize(), 0);
  }
}

TEST(UTEST_ge_model_serialize, test_invalid_Graph) {
  {  // empty graph

    ComputeGraphPtr graph = nullptr;

    ModelSerialize serialize;
    auto buffer = serialize.SerializeGraph(graph);
    EXPECT_EQ(buffer.GetSize(), 0);
  }
}

TEST(UTEST_ge_model_serialize, test_invalid_OpDesc) {
  {  // empty OpDesc
    OpDescPtr opDesc = nullptr;
    ModelSerialize serialize;
    auto buffer = serialize.SerializeOpDesc(opDesc);
    EXPECT_EQ(buffer.GetSize(), 0);
  }
}

TEST(UTEST_ge_model_serialize, test_invalid_TensorDesc) {
  {  // valid test
    Model model("model_name", "custom version3.0");
    auto computeGraph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
    inputOp->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto input = CreateNode(inputOp, computeGraph);
    Graph graph = GraphUtils::CreateGraphFromComputeGraph(computeGraph);
    model.SetGraph(graph);

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    EXPECT_GE(buffer.GetSize(), 0);
  }
  {  // invalid format
    Model model("model_name", "custom version3.0");
    auto computeGraph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
    inputOp->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_RESERVED, DT_FLOAT));  // invalid format
    auto input = CreateNode(inputOp, computeGraph);
    Graph graph = GraphUtils::CreateGraphFromComputeGraph(computeGraph);
    model.SetGraph(graph);

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    ASSERT_GE(buffer.GetSize(), 0);
    auto model2 = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
    ASSERT_TRUE(model2.IsValid());
    auto graphNew = GraphUtils::GetComputeGraph(model2.GetGraph());
    ASSERT_TRUE(graphNew != nullptr);
    auto nodeListNew = graphNew->GetAllNodes();
    ASSERT_EQ(nodeListNew.size(), 1);
    auto opDescNew = nodeListNew.at(0)->GetOpDesc();
    ASSERT_TRUE(opDescNew != nullptr);
    auto outputDescListNew = opDescNew->GetAllOutputsDesc();
    ASSERT_EQ(outputDescListNew.size(), 1);
    auto outputDescNew = outputDescListNew.at(0);
    EXPECT_EQ(outputDescNew.GetDataType(), DT_FLOAT);
    EXPECT_EQ(outputDescNew.GetFormat(), FORMAT_RESERVED);
  }
  {  // DT_UNDEFINED datatype
    Model model("model_name", "custom version3.0");
    auto computeGraph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
    inputOp->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_UNDEFINED));
    auto input = CreateNode(inputOp, computeGraph);
    Graph graph = GraphUtils::CreateGraphFromComputeGraph(computeGraph);
    model.SetGraph(graph);

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    ASSERT_GE(buffer.GetSize(), 0);
    auto model2 = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
    ASSERT_TRUE(model2.IsValid());
    auto graphNew = GraphUtils::GetComputeGraph(model2.GetGraph());
    ASSERT_TRUE(graphNew != nullptr);
    auto nodeListNew = graphNew->GetAllNodes();
    ASSERT_EQ(nodeListNew.size(), 1);
    auto opDescNew = nodeListNew.at(0)->GetOpDesc();
    ASSERT_TRUE(opDescNew != nullptr);
    auto outputDescListNew = opDescNew->GetAllOutputsDesc();
    ASSERT_EQ(outputDescListNew.size(), 1);
    auto outputDescNew = outputDescListNew.at(0);
    EXPECT_EQ(outputDescNew.GetDataType(), DT_UNDEFINED);
    EXPECT_EQ(outputDescNew.GetFormat(), FORMAT_NCHW);
  }
}

TEST(UTEST_ge_model_serialize, test_invalid_Attrs) {
  {  // valid test
    Model model("model_name", "custom version3.0");
    auto computeGraph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
    inputOp->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

    GeAttrValue::NamedAttrs namedAttrs;
    namedAttrs.SetAttr("key1", GeAttrValue::CreateFrom<GeAttrValue::INT>(10));
    AttrUtils::SetNamedAttrs(inputOp, "key", namedAttrs);

    auto input = CreateNode(inputOp, computeGraph);
    Graph graph = GraphUtils::CreateGraphFromComputeGraph(computeGraph);
    model.SetGraph(graph);

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    EXPECT_GE(buffer.GetSize(), 0);
  }
  {  // none type
    Model model("model_name", "custom version3.0");
    auto computeGraph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
    inputOp->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

    GeAttrValue::NamedAttrs namedAttrs;
    EXPECT_EQ(namedAttrs.SetAttr("key1", GeAttrValue()), GRAPH_FAILED);
  }
  {  // bytes attr len is 0
    Model model("model_name", "custom version3.0");
    auto computeGraph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
    inputOp->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

    GeAttrValue::NamedAttrs namedAttrs;
    namedAttrs.SetAttr("key1", GeAttrValue::CreateFrom<GeAttrValue::BYTES>(GeAttrValue::BYTES(0)));
    AttrUtils::SetNamedAttrs(inputOp, "key", namedAttrs);

    auto input = CreateNode(inputOp, computeGraph);
    Graph graph = GraphUtils::CreateGraphFromComputeGraph(computeGraph);
    model.SetGraph(graph);

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    EXPECT_GE(buffer.GetSize(), 0);

    auto model2 = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
    EXPECT_TRUE(model2.IsValid());
  }
  {  // invalid list bytes attr
    Model model("model_name", "custom version3.0");
    auto computeGraph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
    inputOp->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

    GeAttrValue::NamedAttrs namedAttrs;
    namedAttrs.SetAttr("key1", GeAttrValue::CreateFrom<GeAttrValue::LIST_BYTES>({GeAttrValue::BYTES(0)}));
    AttrUtils::SetNamedAttrs(inputOp, "key", namedAttrs);

    auto input = CreateNode(inputOp, computeGraph);
    Graph graph = GraphUtils::CreateGraphFromComputeGraph(computeGraph);
    model.SetGraph(graph);

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    EXPECT_GE(buffer.GetSize(), 0);
  }
  {  // invalid graph attr
    Model model("model_name", "custom version3.0");
    auto computeGraph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
    inputOp->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

    GeAttrValue::NamedAttrs namedAttrs;
    EXPECT_EQ(namedAttrs.SetAttr("key1", GeAttrValue::CreateFrom<GeAttrValue::GRAPH>(nullptr)), GRAPH_FAILED);
    GeAttrValue value;
    EXPECT_EQ(namedAttrs.GetAttr("key1", value), GRAPH_FAILED);
    EXPECT_TRUE(value.IsEmpty());
  }
  {  // invalid list graph attr
    Model model("model_name", "custom version3.0");
    auto computeGraph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
    inputOp->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

    GeAttrValue::NamedAttrs namedAttrs;
    EXPECT_EQ(namedAttrs.SetAttr("key1", GeAttrValue::CreateFrom<GeAttrValue::LIST_GRAPH>({nullptr})), GRAPH_FAILED);
    GeAttrValue value;
    EXPECT_EQ(namedAttrs.GetAttr("key1", value), GRAPH_FAILED);
    EXPECT_TRUE(value.IsEmpty());
  }
}

TEST(UTEST_ge_model_serialize, test_ModelSerializeImp_Invalid_Param) {
  ModelSerializeImp imp;
  EXPECT_FALSE(imp.SerializeModel(Model(), nullptr));
  EXPECT_FALSE(imp.SerializeGraph(nullptr, nullptr));
  EXPECT_FALSE(imp.SerializeNode(nullptr, nullptr));
  EXPECT_FALSE(imp.SerializeOpDesc(nullptr, nullptr));

  auto graph = std::make_shared<ComputeGraph>("test_graph");
  auto node = graph->AddNode(std::make_shared<OpDesc>());
  node->op_ = nullptr;
  proto::ModelDef modelDef;
  Model model;
  model.SetGraph(GraphUtils::CreateGraphFromComputeGraph(graph));
  EXPECT_FALSE(imp.SerializeModel(model, &modelDef));

  ModelSerialize serialize;
  EXPECT_EQ(serialize.GetSerializeModelSize(model), 0);
}

TEST(UTEST_ge_model_serialize, test_parse_node_false) {
  ModelSerializeImp imp;
  string node_index = "invalid_index";
  string node_name = "name";
  int32_t index = 1;
  EXPECT_EQ(imp.ParseNodeIndex(node_index, node_name, index), false);
}

TEST(UTEST_ge_model_serialize, test_invalid_tensor) {
  ModelSerializeImp imp;
  EXPECT_EQ(imp.SerializeTensor(nullptr, nullptr), false);

  try {
    ConstGeTensorPtr tensorPtr = std::make_shared<GeTensor>();
    EXPECT_EQ(imp.SerializeTensor(tensorPtr, nullptr), false);
  } catch (...) {
  }
}

TEST(UTEST_ge_model_unserialize, test_invalid_tensor) {
  ModelSerializeImp imp;
  EXPECT_EQ(imp.SerializeTensor(nullptr, nullptr), false);

  try {
    ConstGeTensorPtr tensorPtr = std::make_shared<GeTensor>();
    EXPECT_EQ(imp.SerializeTensor(tensorPtr, nullptr), false);
  } catch (...) {
  }
}

TEST(UTEST_ge_model_unserialize, test_invalid_TensorDesc) {
  {  // valid
    proto::ModelDef modeDeff;
    auto attrs = modeDeff.mutable_attr();

    proto::AttrDef *attrDef = &(*attrs)["key1"];
    auto tensorDescAttr = attrDef->mutable_td();
    tensorDescAttr->set_layout("NCHW");
    tensorDescAttr->set_dtype(proto::DataType::DT_INT8);

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
  }
  {  // invalid layout
    proto::ModelDef modeDeff;
    auto attrs = modeDeff.mutable_attr();

    proto::AttrDef *attrDef = &(*attrs)["key1"];
    auto tensorDescAttr = attrDef->mutable_td();
    tensorDescAttr->set_layout("InvalidLayout");
    tensorDescAttr->set_dtype(proto::DataType::DT_INT8);

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
    GeTensorDesc tensorDesc;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(model, "key1", tensorDesc));
    EXPECT_EQ(tensorDesc.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensorDesc.GetDataType(), DT_INT8);
  }
  {  // invalid datatype
    proto::ModelDef modeDeff;
    auto attrs = modeDeff.mutable_attr();

    proto::AttrDef *attrDef = &(*attrs)["key1"];
    auto tensorDescAttr = attrDef->mutable_td();  // tensor desc
    tensorDescAttr->set_layout("NHWC");
    tensorDescAttr->set_dtype((proto::DataType)100);

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
    GeTensorDesc tensorDesc;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(model, "key1", tensorDesc));
    EXPECT_EQ(tensorDesc.GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(tensorDesc.GetDataType(), DT_UNDEFINED);
  }
  {  // invalid datatype
    proto::ModelDef modeDeff;
    auto attrs = modeDeff.mutable_attr();

    proto::AttrDef *attrDef = &(*attrs)["key1"];
    auto tensorDescAttr = attrDef->mutable_t()->mutable_desc();  // tensor
    tensorDescAttr->set_layout("NHWC");
    tensorDescAttr->set_dtype((proto::DataType)100);

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
    ConstGeTensorPtr tensor;
    EXPECT_TRUE(AttrUtils::GetTensor(model, "key1", tensor));
    ASSERT_TRUE(tensor != nullptr);
    auto tensorDesc = tensor->GetTensorDesc();
    EXPECT_EQ(tensorDesc.GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(tensorDesc.GetDataType(), DT_UNDEFINED);
  }
  {  // invalid attrmap
    proto::ModelDef modeDeff;
    auto attrs = modeDeff.add_graph()->mutable_attr();  // graph attr

    proto::AttrDef *attrDef = &(*attrs)["key1"];
    auto tensorDescAttr = attrDef->mutable_t()->mutable_desc();  // tensor
    tensorDescAttr->set_layout("NCHW");
    tensorDescAttr->set_dtype(proto::DataType::DT_INT8);
    auto attrs1 = tensorDescAttr->mutable_attr();
    auto attr1 = (*attrs1)["key2"];  // empty attr

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    ConstGeTensorPtr tensor;
    EXPECT_TRUE(AttrUtils::GetTensor(graph, "key1", tensor));
    ASSERT_TRUE(tensor != nullptr);
    auto tensorDesc = tensor->GetTensorDesc();
    GeAttrValue attrValue;
    EXPECT_EQ(tensorDesc.GetAttr("key2", attrValue), GRAPH_SUCCESS);
    EXPECT_EQ(attrValue.GetValueType(), GeAttrValue::VT_NONE);
  }
  {  // invalid attrmap2
    proto::ModelDef modeDeff;
    auto attrs = modeDeff.add_graph()->add_op()->mutable_attr();  // node attr

    proto::AttrDef *attrDef = &(*attrs)["key1"];
    auto tensorDescAttr = attrDef->mutable_t()->mutable_desc();  // tensor
    tensorDescAttr->set_layout("NCHW");
    tensorDescAttr->set_dtype(proto::DataType::DT_INT8);
    auto attrs1 = tensorDescAttr->mutable_attr();
    auto attr1 = (*attrs1)["key2"].mutable_list();  // empty list attr

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    ConstGeTensorPtr tensor;
    EXPECT_TRUE(AttrUtils::GetTensor(nodes.at(0)->GetOpDesc(), "key1", tensor));
    ASSERT_TRUE(tensor != nullptr);
    auto tensorDesc = tensor->GetTensorDesc();
    GeAttrValue attrValue;
    EXPECT_EQ(tensorDesc.GetAttr("key2", attrValue), GRAPH_SUCCESS);
    EXPECT_EQ(attrValue.GetValueType(), GeAttrValue::VT_NONE);
  }
}
TEST(UTEST_ge_model_unserialize, test_invalid_Attr) {
  {  // invalid graph
    proto::ModelDef modeDeff;
    auto attrs = modeDeff.add_graph()->add_op()->mutable_attr();  // node attr

    proto::AttrDef *attrDef = &(*attrs)["key1"];
    auto graphAttr = attrDef->mutable_g();
    auto attrsOfGraph = graphAttr->mutable_attr();
    auto tensorVal = (*attrsOfGraph)["key2"].mutable_td();
    tensorVal->set_dtype(proto::DT_INT8);
    tensorVal->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    ComputeGraphPtr graphAttrNew;
    EXPECT_TRUE(AttrUtils::GetGraph(nodes.at(0)->GetOpDesc(), "key1", graphAttrNew));
    ASSERT_TRUE(graphAttrNew != nullptr);
    GeTensorDesc tensorDesc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(graphAttrNew, "key2", tensorDesc1));
    EXPECT_EQ(tensorDesc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensorDesc1.GetDataType(), DT_INT8);
  }
  {  // invalid list graph
    proto::ModelDef modeDeff;
    auto attrs = modeDeff.add_graph()->add_op()->mutable_attr();  // node attr

    proto::AttrDef *attrDef = &(*attrs)["key1"];
    attrDef->mutable_list()->set_val_type(ge::proto::AttrDef_ListValue_ListValueType_VT_LIST_GRAPH);
    auto graphAttr = attrDef->mutable_list()->add_g();
    auto attrsOfGraph = graphAttr->mutable_attr();
    auto tensorVal = (*attrsOfGraph)["key2"].mutable_td();
    tensorVal->set_dtype(proto::DT_INT8);
    tensorVal->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    vector<ComputeGraphPtr> graphListAttr;
    EXPECT_TRUE(AttrUtils::GetListGraph(nodes.at(0)->GetOpDesc(), "key1", graphListAttr));
    ASSERT_EQ(graphListAttr.size(), 1);
    ASSERT_TRUE(graphListAttr[0] != nullptr);
    GeTensorDesc tensorDesc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(graphListAttr[0], "key2", tensorDesc1));
    EXPECT_EQ(tensorDesc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensorDesc1.GetDataType(), DT_INT8);
  }
  {  // invalid namedAttrs
    proto::ModelDef modeDeff;
    auto attrs = modeDeff.add_graph()->add_op()->mutable_attr();  // node attr

    proto::AttrDef *attrDef = &(*attrs)["key1"];
    auto graphAttr = attrDef->mutable_func();
    auto attrsOfGraph = graphAttr->mutable_attr();
    auto tensorVal = (*attrsOfGraph)["key2"].mutable_td();
    tensorVal->set_dtype(proto::DT_INT8);
    tensorVal->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    GeAttrValue::NAMED_ATTRS namedAttrs;
    EXPECT_TRUE(AttrUtils::GetNamedAttrs(nodes.at(0)->GetOpDesc(), "key1", namedAttrs));
    GeTensorDesc tensorDesc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(namedAttrs, "key2", tensorDesc1));
    EXPECT_EQ(tensorDesc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensorDesc1.GetDataType(), DT_INT8);
  }
  {  // invalid list namedAttrs
    proto::ModelDef modeDeff;
    auto attrs = modeDeff.add_graph()->add_op()->mutable_attr();  // node attr

    proto::AttrDef *attrDef = &(*attrs)["key1"];
    attrDef->mutable_list()->set_val_type(ge::proto::AttrDef_ListValue_ListValueType_VT_LIST_NAMED_ATTRS);
    auto graphAttr = attrDef->mutable_list()->add_na();
    auto attrsOfGraph = graphAttr->mutable_attr();
    auto tensorVal = (*attrsOfGraph)["key2"].mutable_td();
    tensorVal->set_dtype(proto::DT_INT8);
    tensorVal->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    GeAttrValue::LIST_NAMED_ATTRS namedAttrs;
    EXPECT_TRUE(AttrUtils::GetListNamedAttrs(nodes.at(0)->GetOpDesc(), "key1", namedAttrs));
    ASSERT_EQ(namedAttrs.size(), 1);
    GeTensorDesc tensorDesc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(namedAttrs.at(0), "key2", tensorDesc1));
    EXPECT_EQ(tensorDesc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensorDesc1.GetDataType(), DT_INT8);
  }
  {  // invalid tensorDesc
    proto::ModelDef modeDeff;
    auto attrs = modeDeff.add_graph()->add_op()->mutable_attr();  // node attr

    proto::AttrDef *attrDef = &(*attrs)["key1"];
    auto graphAttr = attrDef->mutable_td();
    auto attrsOfGraph = graphAttr->mutable_attr();
    auto tensorVal = (*attrsOfGraph)["key2"].mutable_td();
    tensorVal->set_dtype(proto::DT_INT8);
    tensorVal->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    GeTensorDesc tensorDesc;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(nodes.at(0)->GetOpDesc(), "key1", tensorDesc));
    GeTensorDesc tensorDesc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(tensorDesc, "key2", tensorDesc1));
    EXPECT_EQ(tensorDesc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensorDesc1.GetDataType(), DT_INT8);
  }
  {  // invalid list tensorDesc
    proto::ModelDef modeDeff;
    auto attrs = modeDeff.add_graph()->add_op()->mutable_attr();  // node attr

    proto::AttrDef *attrDef = &(*attrs)["key1"];
    attrDef->mutable_list()->set_val_type(ge::proto::AttrDef_ListValue_ListValueType_VT_LIST_TENSOR_DESC);
    auto graphAttr = attrDef->mutable_list()->add_td();
    auto attrsOfGraph = graphAttr->mutable_attr();
    auto tensorVal = (*attrsOfGraph)["key2"].mutable_td();
    tensorVal->set_dtype(proto::DT_INT8);
    tensorVal->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    vector<GeTensorDesc> tensorDesc;
    EXPECT_TRUE(AttrUtils::GetListTensorDesc(nodes.at(0)->GetOpDesc(), "key1", tensorDesc));
    ASSERT_EQ(tensorDesc.size(), 1);
    GeTensorDesc tensorDesc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(tensorDesc.at(0), "key2", tensorDesc1));
    EXPECT_EQ(tensorDesc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensorDesc1.GetDataType(), DT_INT8);
  }
  {  // invalid tensor
    proto::ModelDef modeDeff;
    auto attrs = modeDeff.add_graph()->add_op()->mutable_attr();  // node attr

    proto::AttrDef *attrDef = &(*attrs)["key1"];
    auto graphAttr = attrDef->mutable_t()->mutable_desc();
    auto attrsOfGraph = graphAttr->mutable_attr();
    auto tensorVal = (*attrsOfGraph)["key2"].mutable_td();
    tensorVal->set_dtype(proto::DT_INT8);
    tensorVal->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    ConstGeTensorPtr tensor;
    EXPECT_TRUE(AttrUtils::GetTensor(nodes.at(0)->GetOpDesc(), "key1", tensor));
    GeTensorDesc tensorDesc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(tensor->GetTensorDesc(), "key2", tensorDesc1));
    EXPECT_EQ(tensorDesc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensorDesc1.GetDataType(), DT_INT8);
  }
  {  // invalid list tensor
    proto::ModelDef modeDeff;
    auto attrs = modeDeff.add_graph()->add_op()->mutable_attr();  // node attr

    proto::AttrDef *attrDef = &(*attrs)["key1"];
    attrDef->mutable_list()->set_val_type(ge::proto::AttrDef_ListValue_ListValueType_VT_LIST_TENSOR);
    auto graphAttr = attrDef->mutable_list()->add_t()->mutable_desc();
    auto attrsOfGraph = graphAttr->mutable_attr();
    auto tensorVal = (*attrsOfGraph)["key2"].mutable_td();
    tensorVal->set_dtype(proto::DT_INT8);
    tensorVal->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    vector<ConstGeTensorPtr> tensor;
    EXPECT_TRUE(AttrUtils::GetListTensor(nodes.at(0)->GetOpDesc(), "key1", tensor));
    ASSERT_EQ(tensor.size(), 1);
    GeTensorDesc tensorDesc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(tensor.at(0)->GetTensorDesc(), "key2", tensorDesc1));
    EXPECT_EQ(tensorDesc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensorDesc1.GetDataType(), DT_INT8);
  }
  {  // invalid list tensor
    proto::GraphDef graphDef;
    auto attrs = graphDef.add_op()->mutable_attr();  // node attr

    proto::AttrDef *attrDef = &(*attrs)["key1"];
    attrDef->mutable_list()->set_val_type(ge::proto::AttrDef_ListValue_ListValueType_VT_LIST_TENSOR);
    auto graphAttr = attrDef->mutable_list()->add_t()->mutable_desc();
    auto attrsOfGraph = graphAttr->mutable_attr();
    auto tensorVal = (*attrsOfGraph)["key2"].mutable_td();
    tensorVal->set_dtype(proto::DT_INT8);
    tensorVal->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Buffer buffer(graphDef.ByteSizeLong());
    graphDef.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    ModelSerialize serialize;
    auto graph = serialize.UnserializeGraph(buffer.GetData(), buffer.GetSize());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    vector<ConstGeTensorPtr> tensor;
    EXPECT_TRUE(AttrUtils::GetListTensor(nodes.at(0)->GetOpDesc(), "key1", tensor));
    ASSERT_EQ(tensor.size(), 1);
    GeTensorDesc tensorDesc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(tensor.at(0)->GetTensorDesc(), "key2", tensorDesc1));
    EXPECT_EQ(tensorDesc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensorDesc1.GetDataType(), DT_INT8);
  }
}

TEST(UTEST_ge_model_unserialize, test_invalid_InputOutput) {
  // model invalid node input
  {
    proto::ModelDef modelDef;
    auto opDef = modelDef.add_graph()->add_op();  // node attr
    opDef->add_input("invalidNodeName:0");

    Buffer buffer(modelDef.ByteSizeLong());
    modelDef.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    ModelSerialize serialize;
    auto model = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
    EXPECT_FALSE(model.IsValid());
  }
  // model invalid node control input
  {
    proto::ModelDef modelDef;
    auto opDef = modelDef.add_graph()->add_op();  // node attr
    opDef->add_input("invalidNodeName:-1");

    Buffer buffer(modelDef.ByteSizeLong());
    modelDef.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    ModelSerialize serialize;
    auto model = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
    EXPECT_FALSE(model.IsValid());
  }
  // model invalid graph input
  {
    proto::ModelDef modelDef;
    modelDef.add_graph()->add_input("invalidNodeName:0");

    Buffer buffer(modelDef.ByteSizeLong());
    modelDef.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    ModelSerialize serialize;
    auto model = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
    EXPECT_FALSE(model.IsValid());
  }
  // model invalid graph input
  {
    proto::ModelDef modelDef;
    modelDef.add_graph()->add_output("invalidNodeName:0");

    Buffer buffer(modelDef.ByteSizeLong());
    modelDef.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    ModelSerialize serialize;
    auto model = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
    EXPECT_FALSE(model.IsValid());
  }
  // graph invalid node input
  {
    proto::GraphDef graphDef;
    auto opDef = graphDef.add_op();  // node attr
    opDef->add_input("invalidNodeName:0");

    Buffer buffer(graphDef.ByteSizeLong());
    graphDef.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    ModelSerialize serialize;
    auto graph = serialize.UnserializeGraph(buffer.GetData(), buffer.GetSize());
    EXPECT_FALSE(graph != nullptr);
  }
  // graph invalid node control input
  {
    proto::GraphDef graphDef;
    auto opDef = graphDef.add_op();  // node attr
    opDef->add_input("invalidNodeName:-1");

    Buffer buffer(graphDef.ByteSizeLong());
    graphDef.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    ModelSerialize serialize;
    auto graph = serialize.UnserializeGraph(buffer.GetData(), buffer.GetSize());
    EXPECT_FALSE(graph != nullptr);
  }
  // graph invalid graph input
  {
    proto::GraphDef graphDef;
    graphDef.add_input("invalidNodeName:0");

    Buffer buffer(graphDef.ByteSizeLong());
    graphDef.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    ModelSerialize serialize;
    auto graph = serialize.UnserializeGraph(buffer.GetData(), buffer.GetSize());
    EXPECT_FALSE(graph != nullptr);
  }
  // graph invalid graph output
  {
    proto::GraphDef graphDef;
    graphDef.add_output("invalidNodeName:0");

    Buffer buffer(graphDef.ByteSizeLong());
    graphDef.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    ModelSerialize serialize;
    auto graph = serialize.UnserializeGraph(buffer.GetData(), buffer.GetSize());
    EXPECT_FALSE(graph != nullptr);
  }
  // model invalid node input anchor
  {
    proto::ModelDef modelDef;
    auto graphDef = modelDef.add_graph();
    auto nodeDef1 = graphDef->add_op();  // node attr
    nodeDef1->set_name("node1");

    auto nodeDef2 = graphDef->add_op();  // node attr
    nodeDef2->add_input("node1:0");

    Buffer buffer(modelDef.ByteSizeLong());
    modelDef.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    ModelSerialize serialize;
    auto model = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
    EXPECT_FALSE(model.IsValid());
  }
}

TEST(UTEST_ge_model_unserialize, test_invalid_CodeBuffer) {
  {
    char buffer[100] = "sdfasf";
    ModelSerialize serialize;
    auto graph = serialize.UnserializeGraph((uint8_t *)buffer, 100);
    EXPECT_FALSE(graph != nullptr);
  }
  {
    char buffer[100] = "sdfasf";
    ModelSerialize serialize;
    auto model = serialize.UnserializeModel((uint8_t *)buffer, 100);
    EXPECT_FALSE(model.IsValid());
  }
  {
    char buffer[100] = "sdfasf";
    ModelSerialize serialize;
    auto opDesc = serialize.UnserializeOpDesc((uint8_t *)buffer, 100);
    EXPECT_FALSE(opDesc != nullptr);
  }
  {
    ModelSerialize serialize;
    auto graph = serialize.UnserializeGraph((uint8_t *)nullptr, 100);
    EXPECT_FALSE(graph != nullptr);
  }
  {
    ModelSerialize serialize;
    auto model = serialize.UnserializeModel((uint8_t *)nullptr, 100);
    EXPECT_FALSE(model.IsValid());
  }
  {
    ModelSerialize serialize;
    auto opDesc = serialize.UnserializeOpDesc((uint8_t *)nullptr, 100);
    EXPECT_FALSE(opDesc != nullptr);
  }
}
