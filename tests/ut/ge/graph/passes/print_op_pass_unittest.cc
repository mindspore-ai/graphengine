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

#include "graph/passes/print_op_pass.h"

#include <gtest/gtest.h>

#include "omg/omg_inner_types.h"
#include "utils/op_desc_utils.h"

using namespace domi;
namespace ge {
class UTEST_graph_passes_print_op_pass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
  void make_graph(ComputeGraphPtr graph, bool match = true, int flag = 0) {
    auto data = std::make_shared<OpDesc>("Data", DATA);
    GeTensorDesc tensorDescData(GeShape({1, 1, 1, 1}));
    data->AddInputDesc(tensorDescData);
    data->AddOutputDesc(tensorDescData);
    auto dataNode = graph->AddNode(data);

    auto data1 = std::make_shared<OpDesc>("Data", DATA);
    data1->AddInputDesc(tensorDescData);
    data1->AddOutputDesc(tensorDescData);
    auto dataNode1 = graph->AddNode(data1);

    auto printDesc = std::make_shared<OpDesc>("Print", "Print");
    printDesc->AddInputDesc(tensorDescData);
    printDesc->AddInputDesc(tensorDescData);
    printDesc->AddOutputDesc(tensorDescData);
    auto printNode = graph->AddNode(printDesc);

    auto retValDesc = std::make_shared<OpDesc>("RetVal", "RetVal");
    retValDesc->AddInputDesc(tensorDescData);
    retValDesc->AddOutputDesc(tensorDescData);
    auto retValNode = graph->AddNode(retValDesc);

    auto ret = GraphUtils::AddEdge(dataNode->GetOutDataAnchor(0), printNode->GetInDataAnchor(0));
    ret = GraphUtils::AddEdge(dataNode1->GetOutDataAnchor(0), printNode->GetInDataAnchor(1));
    ret = GraphUtils::AddEdge(printNode->GetOutDataAnchor(0), retValNode->GetInDataAnchor(0));
  }
};

TEST_F(UTEST_graph_passes_print_op_pass, apply_success) {
  GetContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_graph(graph);
  ge::PrintOpPass applyPass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("Test", &applyPass);
  GEPass pass(graph);
  domi::Status status = pass.Run(names_to_pass);
  EXPECT_EQ(domi::SUCCESS, status);
}

TEST_F(UTEST_graph_passes_print_op_pass, param_invalid) {
  ge::NodePtr node = nullptr;
  ge::PrintOpPass applyPass;
  domi::Status status = applyPass.Run(node);
  EXPECT_EQ(ge::PARAM_INVALID, status);
}
}  // namespace ge
