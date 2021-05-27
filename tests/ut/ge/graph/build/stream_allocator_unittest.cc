/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <string>
#include <vector>
#include <gtest/gtest.h>

#define protected public
#define private public
#include "graph/build/stream_allocator.h"
#undef protected
#undef private

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"

namespace ge {
class UtestStreamAllocator : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
 public:

  ///
  ///    A
  ///   / \
  ///  B   C
  ///  |   |
  ///  D  400
  ///  |   |
  ///  |   E
  ///   \ /
  ///    F
  ///
  void make_graph_active(const ComputeGraphPtr &graph) {
    const auto &a_desc = std::make_shared<OpDesc>("A", DATA);
    a_desc->AddInputDesc(GeTensorDesc());
    a_desc->AddOutputDesc(GeTensorDesc());
    a_desc->SetStreamId(0);
    const auto &a_node = graph->AddNode(a_desc);

    const auto &b_desc = std::make_shared<OpDesc>("B", "testa");
    b_desc->AddInputDesc(GeTensorDesc());
    b_desc->AddOutputDesc(GeTensorDesc());
    b_desc->SetStreamId(1);
    AttrUtils::SetListStr(b_desc, ATTR_NAME_ACTIVE_LABEL_LIST, {"1"});
    const auto &b_node = graph->AddNode(b_desc);

    const auto &c_desc = std::make_shared<OpDesc>("C", "testa");
    c_desc->AddInputDesc(GeTensorDesc());
    c_desc->AddOutputDesc(GeTensorDesc());
    c_desc->SetStreamId(2);
    AttrUtils::SetStr(c_desc, ATTR_NAME_STREAM_LABEL, "1");
    const auto &c_node = graph->AddNode(c_desc);

    const auto &d_desc = std::make_shared<OpDesc>("D", "testa");
    d_desc->AddInputDesc(GeTensorDesc());
    d_desc->AddOutputDesc(GeTensorDesc());
    d_desc->SetStreamId(1);
    const auto &d_node = graph->AddNode(d_desc);

    const auto &e_desc = std::make_shared<OpDesc>("E", "testa");
    e_desc->AddInputDesc(GeTensorDesc());
    e_desc->AddOutputDesc(GeTensorDesc());
    e_desc->SetStreamId(2);
    const auto &e_node = graph->AddNode(e_desc);

    const auto &f_desc = std::make_shared<OpDesc>("F", "testa");
    f_desc->AddInputDesc(GeTensorDesc());
    f_desc->AddInputDesc(GeTensorDesc());
    f_desc->AddOutputDesc(GeTensorDesc());
    f_desc->SetStreamId(2);
    const auto &f_node = graph->AddNode(f_desc);

    std::vector<NodePtr> node_list(400);
    for (int  i = 0; i < 400; i++) {
      const auto &op_desc = std::make_shared<OpDesc>("X", DATA);
      op_desc->AddInputDesc(GeTensorDesc());
      op_desc->AddOutputDesc(GeTensorDesc());
      op_desc->SetStreamId(2);
      node_list[i] = graph->AddNode(op_desc);
    }

    GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), b_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), c_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(b_node->GetOutDataAnchor(0), d_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(d_node->GetOutDataAnchor(0), f_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(c_node->GetOutDataAnchor(0), node_list[0]->GetInDataAnchor(0));
    for (uint32_t i = 0; i < 399; i++) {
      GraphUtils::AddEdge(node_list[i]->GetOutDataAnchor(0), node_list[i + 1]->GetInDataAnchor(0));
    }
    GraphUtils::AddEdge(node_list[399]->GetOutDataAnchor(0), e_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(e_node->GetOutDataAnchor(0), f_node->GetInDataAnchor(1));
  }
};

TEST_F(UtestStreamAllocator, test_split_streams_active) {
  const auto &graph = std::make_shared<ComputeGraph>("test_split_streams_active_graph");
  EXPECT_NE(graph, nullptr);
  make_graph_active(graph);

  StreamAllocator allocator(graph, Graph2SubGraphInfoList());
  allocator.stream_num_ = 3;
  EXPECT_EQ(allocator.SetActiveStreamsByLabel(), SUCCESS);
  std::vector<std::set<int64_t>> split_stream(3);
  EXPECT_EQ(allocator.SplitStreams(split_stream), SUCCESS);
  EXPECT_EQ(allocator.UpdateActiveStreams(split_stream), SUCCESS);
  EXPECT_EQ(allocator.SetActiveStreamsForLoop(), SUCCESS);
  EXPECT_EQ(allocator.specific_activated_streams_.count(3), 1);

  const auto &node_b = graph->FindNode("B");
  EXPECT_NE(node_b, nullptr);
  std::vector<uint32_t> active_stream_list;
  EXPECT_TRUE(AttrUtils::GetListInt(node_b->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list));
  EXPECT_EQ(active_stream_list.size(), 2);
  const auto &node_e = graph->FindNode("E");
  EXPECT_NE(node_e, nullptr);
  EXPECT_EQ(active_stream_list[0], node_e->GetOpDesc()->GetStreamId());
  EXPECT_EQ(active_stream_list[1], 3);
}

TEST_F(UtestStreamAllocator, test_update_active_streams_for_subgraph) {
  const auto &root_graph = std::make_shared<ComputeGraph>("test_update_active_streams_for_subgraph_root_graph");
  EXPECT_NE(root_graph, nullptr);
  root_graph->SetGraphUnknownFlag(false);
  const auto &sub_graph1 = std::make_shared<ComputeGraph>("test_update_active_streams_for_subgraph_sub_graph1");
  EXPECT_NE(sub_graph1, nullptr);
  root_graph->AddSubGraph(sub_graph1);
  const auto &sub_graph2 = std::make_shared<ComputeGraph>("test_update_active_streams_for_subgraph_sub_graph2");
  EXPECT_NE(sub_graph2, nullptr);
  root_graph->AddSubGraph(sub_graph2);

  const auto &case_desc = std::make_shared<OpDesc>("case", CASE);
  EXPECT_NE(case_desc, nullptr);
  EXPECT_EQ(case_desc->AddInputDesc(GeTensorDesc()), GRAPH_SUCCESS);
  EXPECT_EQ(case_desc->AddOutputDesc(GeTensorDesc()), GRAPH_SUCCESS);
  case_desc->AddSubgraphName("branch1");
  case_desc->SetSubgraphInstanceName(0, "test_update_active_streams_for_subgraph_sub_graph1");
  case_desc->AddSubgraphName("branch2");
  case_desc->SetSubgraphInstanceName(1, "test_update_active_streams_for_subgraph_sub_graph2");
  const auto &case_node = root_graph->AddNode(case_desc);
  EXPECT_NE(case_node, nullptr);
  sub_graph1->SetParentNode(case_node);
  sub_graph2->SetParentNode(case_node);

  const auto &active_desc1 = std::make_shared<OpDesc>("active1", STREAMACTIVE);
  EXPECT_NE(active_desc1, nullptr);
  EXPECT_TRUE(AttrUtils::SetListInt(active_desc1, ATTR_NAME_ACTIVE_STREAM_LIST, {0}));
  const auto &active_node1 = sub_graph1->AddNode(active_desc1);
  EXPECT_NE(active_node1, nullptr);

  const auto &active_desc2 = std::make_shared<OpDesc>("active2", STREAMACTIVE);
  EXPECT_NE(active_desc2, nullptr);
  EXPECT_TRUE(AttrUtils::SetListInt(active_desc2, ATTR_NAME_ACTIVE_STREAM_LIST, {1}));
  const auto &active_node2 = sub_graph2->AddNode(active_desc2);
  EXPECT_NE(active_node2, nullptr);

  StreamAllocator allocator(root_graph, Graph2SubGraphInfoList());
  allocator.node_split_stream_map_[active_node1] = 2;
  allocator.node_split_stream_map_[active_node2] = 3;
  allocator.split_ori_stream_map_[2] = 0;
  allocator.subgraph_first_active_node_map_[sub_graph1] = active_node1;
  allocator.subgraph_first_active_node_map_[sub_graph2] = active_node2;
  EXPECT_EQ(allocator.UpdateActiveStreamsForSubgraphs(), SUCCESS);
  std::vector<uint32_t> active_stream_list1;
  EXPECT_TRUE(AttrUtils::GetListInt(active_node1->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list1));
  EXPECT_EQ(active_stream_list1.size(), 1);
  EXPECT_EQ(active_stream_list1[0], 0);
  std::vector<uint32_t> active_stream_list2;
  EXPECT_TRUE(AttrUtils::GetListInt(active_node2->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list2));
  EXPECT_EQ(active_stream_list2.size(), 2);
  EXPECT_EQ(active_stream_list2[0], 1);
  EXPECT_EQ(active_stream_list2[1], 3);
  EXPECT_EQ(allocator.specific_activated_streams_.size(), 1);
  EXPECT_EQ(allocator.specific_activated_streams_.count(3), 1);
}
}
