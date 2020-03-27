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
#include "graph/node.h"

#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#undef protected
#undef private

using namespace std;
using namespace ge;

class ge_test_node : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(ge_test_node, node) {
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
  NodePtr n3 = graphPtr->AddNode(descPtr);
  NodePtr n4 = graphPtr->AddNode(descPtr);

  EXPECT_EQ(n3->Init(), GRAPH_SUCCESS);
  EXPECT_EQ(n4->Init(), GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(n3->GetOutDataAnchor(0), n4->GetInDataAnchor(0)), GRAPH_SUCCESS);

  EXPECT_EQ(n3->GetOwnerComputeGraph(), graphPtr);
  EXPECT_EQ(n3->GetName(), "name1");
  EXPECT_EQ(n3->GetOpDesc(), descPtr);
  int i = 0;
  for (auto in : n3->GetAllOutDataAnchors()) {
    EXPECT_EQ(in->GetIdx(), i++);
  }
  i = 0;
  for (auto in : n3->GetAllInDataAnchors()) {
    EXPECT_EQ(in->GetIdx(), i++);
  }
  EXPECT_EQ(n3->GetInControlAnchor() != nullptr, true);
  EXPECT_EQ(n3->GetOutControlAnchor() != nullptr, true);

  for (auto innode : n4->GetInDataNodes()) {
    EXPECT_EQ(innode, n3);
  }

  for (auto outnode : n3->GetOutDataNodes()) {
    EXPECT_EQ(outnode, n4);
  }
}

TEST_F(ge_test_node, out_nodes) {
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
  NodePtr n3 = graphPtr->AddNode(descPtr);
  NodePtr n4 = graphPtr->AddNode(descPtr);

  EXPECT_EQ(GraphUtils::AddEdge(n1->GetOutDataAnchor(0), n2->GetInDataAnchor(0)), GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(n1->GetOutDataAnchor(0), n3->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(n1->GetOutControlAnchor(), n4->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(n2->GetOutDataAnchor(0), n4->GetInDataAnchor(0)), GRAPH_SUCCESS);
  EXPECT_EQ(n1->GetOutDataNodes().size(), 1);
  EXPECT_EQ(n1->GetOutDataNodes().at(0), n2);
  EXPECT_EQ(n1->GetOutControlNodes().size(), 2);
  EXPECT_EQ(n1->GetOutControlNodes().at(0), n3);
  EXPECT_EQ(n1->GetOutControlNodes().at(1), n4);
  EXPECT_EQ(n1->GetOutAllNodes().size(), 3);
  EXPECT_EQ(n1->GetOutAllNodes().at(0), n2);
  EXPECT_EQ(n1->GetOutAllNodes().at(1), n3);
  EXPECT_EQ(n1->GetOutAllNodes().at(2), n4);
  EXPECT_EQ(n4->GetInControlNodes().size(), 1);
  EXPECT_EQ(n4->GetInDataNodes().size(), 1);
  EXPECT_EQ(n4->GetInAllNodes().size(), 2);

  EXPECT_EQ(n1->GetOutDataNodesAndAnchors().size(), 1);
  EXPECT_EQ(n1->GetOutDataNodesAndAnchors().at(0).first, n2);
  EXPECT_EQ(n1->GetOutDataNodesAndAnchors().at(0).second, n2->GetAllInDataAnchors().at(0));
  EXPECT_EQ(n2->GetInDataNodesAndAnchors().size(), 1);
  EXPECT_EQ(n2->GetInDataNodesAndAnchors().at(0).first, n1);
  EXPECT_EQ(n2->GetInDataNodesAndAnchors().at(0).second, n1->GetAllOutDataAnchors().at(0));

  OutDataAnchorPtr a1;
  InControlAnchorPtr a2;
  EXPECT_EQ(NodeUtils::GetDataOutAnchorAndControlInAnchor(n1, a1, a2), GRAPH_SUCCESS);
  EXPECT_EQ(a1, n1->GetOutDataAnchor(0));
  EXPECT_EQ(a2, n3->GetInControlAnchor());

  a1 = nullptr;
  a2 = nullptr;
  EXPECT_EQ(NodeUtils::GetDataOutAnchorAndControlInAnchor(n4, a1, a2), GRAPH_FAILED);
}

TEST_F(ge_test_node, update_opdesc) {
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

  EXPECT_EQ(n1->UpdateOpDesc(descPtr2), GRAPH_SUCCESS);
}

TEST_F(ge_test_node, add_link_from) {
  OpDescPtr descPtr = std::make_shared<OpDesc>("name", "type");
  EXPECT_EQ(descPtr->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(descPtr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("name");
  NodePtr n1 = graphPtr->AddNode(descPtr);
  NodePtr n2 = graphPtr->AddNode(descPtr);
  EXPECT_EQ(n2->AddLinkFrom(n1), GRAPH_SUCCESS);
  EXPECT_EQ(n2->AddLinkFromForParse(n1), GRAPH_SUCCESS);
  NodePtr n3 = graphPtr->AddNode(descPtr);
  NodePtr n4 = graphPtr->AddNode(descPtr);
  NodePtr n5 = graphPtr->AddNode(descPtr);
  EXPECT_EQ(n3->AddLinkFrom("x", n4), GRAPH_SUCCESS);
  EXPECT_EQ(n3->AddLinkFrom(0, n5), GRAPH_SUCCESS);
  descPtr->input_name_idx_.insert(make_pair("__input1", 1));
  EXPECT_EQ(n2->AddLinkFrom(n1), GRAPH_SUCCESS);

  OpDescPtr descPtr1 = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(descPtr1->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  ComputeGraphPtr graphPtr1 = std::make_shared<ComputeGraph>("name1");
  NodePtr n7 = graphPtr1->AddNode(descPtr1);
  NodePtr n8 = graphPtr1->AddNode(descPtr1);
  EXPECT_EQ(n8->AddLinkFromForParse(n7), GRAPH_PARAM_INVALID);
}

TEST_F(ge_test_node, add_link_from_fail) {
  OpDescPtr descPtr = std::make_shared<OpDesc>("name1", "type1");
  ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("name");
  NodePtr n1 = graphPtr->AddNode(descPtr);

  NodePtr node_ptr = std::make_shared<Node>();
  EXPECT_EQ(n1->AddLinkFrom(node_ptr), GRAPH_PARAM_INVALID);
  EXPECT_EQ(n1->AddLinkFrom(1, node_ptr), GRAPH_PARAM_INVALID);
  EXPECT_EQ(n1->AddLinkFrom("test", node_ptr), GRAPH_PARAM_INVALID);
  EXPECT_EQ(n1->AddLinkFromForParse(node_ptr), GRAPH_PARAM_INVALID);
}

TEST_F(ge_test_node, verify_failed) {
  OpDescPtr descPtr = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(descPtr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("name");
  NodePtr n1 = graphPtr->AddNode(descPtr);

  EXPECT_EQ(n1->Verify(), GRAPH_SUCCESS);
}

TEST_F(ge_test_node, infer_origin_format_success) {
  OpDescPtr descPtr = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(descPtr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("name");
  NodePtr n1 = graphPtr->AddNode(descPtr);

  EXPECT_EQ(n1->InferOriginFormat(), GRAPH_SUCCESS);
}

TEST_F(ge_test_node, node_anchor_is_equal) {
  ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("name");
  OpDescPtr descPtrSrc = std::make_shared<OpDesc>("strNode", "type");
  EXPECT_EQ(descPtrSrc->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(descPtrSrc->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);
  NodePtr strNode = graphPtr->AddNode(descPtrSrc);

  OpDescPtr descPtrPeer = std::make_shared<OpDesc>("peerNode", "type");
  EXPECT_EQ(descPtrPeer->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(descPtrPeer->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);
  NodePtr peerNode = graphPtr->AddNode(descPtrPeer);
  EXPECT_EQ(peerNode->AddLinkFrom(strNode), GRAPH_SUCCESS);
  EXPECT_EQ(strNode->NodeAnchorIsEqual(strNode->GetOutAnchor(0), strNode->GetOutAnchor(0), 0), true);
}
