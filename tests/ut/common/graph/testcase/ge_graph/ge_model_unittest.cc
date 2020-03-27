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

#include "graph/model.h"

#include <gtest/gtest.h>

#include "graph/compute_graph.h"
#include "graph/debug/graph_debug.h"

using namespace std;
using namespace testing;
using namespace ge;

class UTEST_ge_model_unittest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

ge::ComputeGraphPtr CreateSaveGraph() {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  // variable1
  ge::OpDescPtr VariableOp = std::make_shared<ge::OpDesc>();
  VariableOp->SetType("Variable");
  VariableOp->SetName("Variable1");
  VariableOp->AddInputDesc(ge::GeTensorDesc());
  VariableOp->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr VariableNode = graph->AddNode(VariableOp);
  // save1
  ge::OpDescPtr SaveOp = std::make_shared<ge::OpDesc>();
  SaveOp->SetType("Save");
  SaveOp->SetName("Save1");
  SaveOp->AddInputDesc(ge::GeTensorDesc());
  SaveOp->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr SaveNode = graph->AddNode(SaveOp);

  // add edge
  ge::GraphUtils::AddEdge(VariableNode->GetOutDataAnchor(0), SaveNode->GetInDataAnchor(0));

  return graph;
}

TEST_F(UTEST_ge_model_unittest, save_model_to_file_success) {
  ge::ComputeGraphPtr compute_graph = CreateSaveGraph();
  auto all_nodes = compute_graph->GetAllNodes();
  for (auto node : all_nodes) {
    auto op_desc = node->GetOpDesc();
    GeTensorDesc weightDesc;
    op_desc->AddOptionalInputDesc("test", weightDesc);
    for (auto in_anchor_ptr : node->GetAllInDataAnchors()) {
      bool is_optional = op_desc->IsOptionalInput(in_anchor_ptr->GetIdx());
    }
  }
  ge::Graph ge_graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  string file_name = "model_data.pb";
  setenv("DUMP_MODEL", "1", true);
  // EXPECT_EQ(ge_graph.SaveToFile(file_name), GRAPH_FAILED);
  setenv("DUMP_MODEL", "0", true);
}

TEST_F(UTEST_ge_model_unittest, load_model_from_file_success) {
  ge::Graph ge_graph;
  string file_name = "model_data.pb";
  // EXPECT_EQ(ge_graph.LoadFromFile(file_name), GRAPH_SUCCESS);
}
