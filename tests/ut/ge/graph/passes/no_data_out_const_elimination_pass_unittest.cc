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

#include "graph/passes/no_data_out_const_elimination_pass.h"

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <map>

#include "common/ge_inner_error_codes.h"
#include "graph/utils/graph_utils.h"

namespace ge {

class UtestNoDataOutConstEliminationPass : public testing::Test {
protected:
  void SetUp() {}
  void TearDown() {}

public:
  NodePtr MakeNode(const ComputeGraphPtr &graph, uint32_t in_num, uint32_t out_num, string name, string type) {
    GeTensorDesc test_desc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    auto op_desc = std::make_shared<OpDesc>(name, type);
    for (auto i = 0; i < in_num; ++i) {
      op_desc->AddInputDesc(test_desc);
    }
    for (auto i = 0; i < out_num; ++i) {
      op_desc->AddOutputDesc(test_desc);
    }
    return graph->AddNode(op_desc);
  }
};

/// graph with subgraph
///       const1
///          |(control)
///        const2
///          |
///        output
TEST_F(UtestNoDataOutConstEliminationPass, succ_graph1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto const_node1 = MakeNode(graph, 0, 1, "const_node1", "Const");
  auto const_node2 = MakeNode(graph, 1, 1, "const_node2", "Const");
  auto output_node = MakeNode(graph, 1, 0, "output_node", "NetOutput");
  GeTensorDesc tensor_desc(GeShape({1,3,224,224}), FORMAT_NCHW, DT_FLOAT);

  const_node1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  const_node2->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
  const_node2->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  output_node->GetOpDesc()->UpdateInputDesc(0, tensor_desc);

  GraphUtils::AddEdge(const_node1->GetOutControlAnchor(), const_node2->GetInControlAnchor());
  GraphUtils::AddEdge(const_node2->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));

  GEPass pass(graph);
  NamesToPass node_pass;
  NoDataOutConstEliminationPass no_data_out_const_elimination_pass;
  node_pass.emplace_back("NoDataOutConstEliminationPass", &no_data_out_const_elimination_pass);
  auto const1 = graph->FindNode("const_node1");
  EXPECT_NE(const1, nullptr);
  EXPECT_TRUE(const1->GetInDataNodes().empty());
  EXPECT_TRUE(const1->GetOutDataNodes().empty());
  EXPECT_EQ(pass.Run(node_pass), SUCCESS);
  // after pass, const1 will be delete
  const1 = graph->FindNode("const_node1");
  EXPECT_EQ(const1, nullptr);
}
}  // namespace ge
