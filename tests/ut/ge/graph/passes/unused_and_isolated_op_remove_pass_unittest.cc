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

#include "graph/passes/unused_op_remove_pass.h"

#include <gtest/gtest.h>
#include "graph/passes/isolated_op_remove_pass.h"
#include "pass_manager.h"

using namespace ge;
using namespace domi;

class UTEST_graph_passes_unused_and_isolated_op_remove_pass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  NodePtr AddNode(ComputeGraphPtr graph, const string &name, const string &type, int32_t in_anchors_num = 1,
                  int32_t out_anchors_num = 1) {
    GeTensorDesc tensor_desc;
    OpDescPtr opdesc = make_shared<OpDesc>(name, type);
    for (int32_t i = 0; i < in_anchors_num; i++) {
      opdesc->AddInputDesc(tensor_desc);
    }
    for (int32_t i = 0; i < out_anchors_num; i++) {
      opdesc->AddOutputDesc(tensor_desc);
    }

    NodePtr node = graph->AddNode(opdesc);
    return node;
  }
};

TEST_F(UTEST_graph_passes_unused_and_isolated_op_remove_pass, transpose_and_reshape) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  NodePtr data_node = AddNode(graph, "DATA", DATA);
  NodePtr transpose_node = AddNode(graph, "transpose1", PERMUTE);
  NodePtr reshape_node = AddNode(graph, "reshape1", RESHAPE);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), transpose_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transpose_node->GetOutDataAnchor(0), reshape_node->GetInDataAnchor(0));

  ge::UnusedOpRemovePass unused_pass(FMK_TYPE_T);
  ge::IsolatedOpRemovePass isolate_pass;
  vector<GraphPass *> passes = {&unused_pass, &isolate_pass};
  domi::Status status = PassManager::Run(graph, passes);
  EXPECT_EQ(domi::SUCCESS, status);
  NodePtr found_node = graph->FindNode("transpose1");
  EXPECT_EQ(transpose_node, found_node);
}

TEST_F(UTEST_graph_passes_unused_and_isolated_op_remove_pass, transpose_and_squeeze) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  NodePtr data_node = AddNode(graph, "DATA", DATA);
  NodePtr transpose_node = AddNode(graph, "transpose1", PERMUTE);
  NodePtr squeeze_node = AddNode(graph, "squeeze1", SQUEEZE);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), transpose_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transpose_node->GetOutDataAnchor(0), squeeze_node->GetInDataAnchor(0));

  ge::UnusedOpRemovePass unused_pass(FMK_TYPE_T);
  ge::IsolatedOpRemovePass isolate_pass;
  vector<GraphPass *> passes = {&unused_pass, &isolate_pass};
  domi::Status status = PassManager::Run(graph, passes);
  EXPECT_EQ(domi::SUCCESS, status);
  NodePtr found_node = graph->FindNode("transpose1");
  EXPECT_EQ(transpose_node, found_node);
}

TEST_F(UTEST_graph_passes_unused_and_isolated_op_remove_pass, transpose_and_conv) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  NodePtr data_node = AddNode(graph, "DATA", DATA);

  NodePtr transpose_node = AddNode(graph, "transpose1", PERMUTE);
  vector<int64_t> order_list = {0, 2, 3, 1};
  AttrUtils::SetListInt(transpose_node->GetOpDesc(), PERMUTE_ATTR_ORDER, order_list);
  AttrUtils::SetInt(transpose_node->GetOpDesc(), ATTR_NAME_FORMAT, (int64_t)DT_FLOAT);

  NodePtr conv_node = AddNode(graph, "conv1", CONVOLUTION);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), transpose_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transpose_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));

  NodePtr conv2_node = AddNode(graph, "conv2", CONVOLUTION);
  GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), conv2_node->GetInDataAnchor(0));

  ge::UnusedOpRemovePass unused_pass(FMK_TYPE_T);
  ge::IsolatedOpRemovePass isolate_pass;
  vector<GraphPass *> passes = {&unused_pass, &isolate_pass};
  domi::Status status = PassManager::Run(graph, passes);
  EXPECT_EQ(domi::SUCCESS, status);
  NodePtr found_node0 = graph->FindNode("transpose1");
  // EXPECT_EQ(nullptr, found_node0);
  NodePtr found_node = graph->FindNode("conv1");
  EXPECT_EQ(conv_node, found_node);
}

TEST_F(UTEST_graph_passes_unused_and_isolated_op_remove_pass, transpose_and_conv3) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  NodePtr data_node = AddNode(graph, "DATA", DATA);

  NodePtr transpose_node = AddNode(graph, "transpose1", PERMUTE);
  vector<int64_t> order_list = {0, 1, 3, 2};
  AttrUtils::SetListInt(transpose_node->GetOpDesc(), PERMUTE_ATTR_ORDER, order_list);
  AttrUtils::SetInt(transpose_node->GetOpDesc(), ATTR_NAME_FORMAT, (int64_t)DT_FLOAT);

  NodePtr conv_node = AddNode(graph, "conv1", CONVOLUTION);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), transpose_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transpose_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));

  NodePtr conv2_node = AddNode(graph, "conv2", CONVOLUTION);
  GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), conv2_node->GetInDataAnchor(0));

  ge::UnusedOpRemovePass unused_pass(FMK_TYPE_T);
  ge::IsolatedOpRemovePass isolate_pass;
  vector<GraphPass *> passes = {&unused_pass, &isolate_pass};
  domi::Status status = PassManager::Run(graph, passes);
  EXPECT_EQ(domi::SUCCESS, status);
  NodePtr found_node0 = graph->FindNode("transpose1");
  EXPECT_EQ(transpose_node, found_node0);
  NodePtr found_node = graph->FindNode("conv1");
  EXPECT_EQ(conv_node, found_node);
}

TEST_F(UTEST_graph_passes_unused_and_isolated_op_remove_pass, cast_and_cast) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  NodePtr data_node = AddNode(graph, "DATA", DATA);
  NodePtr conv3_node = AddNode(graph, "cast3", CAST);
  NodePtr transpose_node = AddNode(graph, "cast1", CAST);
  NodePtr transpose_node_1 = AddNode(graph, "cast2", CAST);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv3_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(conv3_node->GetOutDataAnchor(0), transpose_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transpose_node->GetOutDataAnchor(0), transpose_node_1->GetInDataAnchor(0));

  ge::UnusedOpRemovePass unused_pass(FMK_TYPE_T);
  ge::IsolatedOpRemovePass isolate_pass;
  vector<GraphPass *> passes = {&unused_pass, &isolate_pass};
  domi::Status status = PassManager::Run(graph, passes);
  EXPECT_EQ(domi::SUCCESS, status);
}

TEST_F(UTEST_graph_passes_unused_and_isolated_op_remove_pass, RemoveParentNode) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<NodePtr> node_vec;

  NodePtr data_node = AddNode(graph, "DATA", DATA);
  NodePtr conv3_node = AddNode(graph, "cast3", CAST);
  NodePtr transpose_node = AddNode(graph, "cast1", CAST);
  NodePtr transpose_node_1 = AddNode(graph, "cast2", CAST);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv3_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(conv3_node->GetOutDataAnchor(0), transpose_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transpose_node->GetOutDataAnchor(0), transpose_node_1->GetInDataAnchor(0));

  ge::UnusedOpRemovePass unused_pass(FMK_TYPE_T);
  ge::IsolatedOpRemovePass isolate_pass;
  vector<GraphPass *> passes = {&unused_pass, &isolate_pass};
  domi::Status status = PassManager::Run(graph, passes);
  EXPECT_EQ(domi::SUCCESS, status);
}
