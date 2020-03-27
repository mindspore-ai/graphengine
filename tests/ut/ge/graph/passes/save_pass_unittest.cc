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

#include "graph/passes/save_pass.h"

#include <gtest/gtest.h>

#include "common/ge_inner_error_codes.h"
#include "ge/ge_api.h"
#include "graph/compute_graph.h"
#include "graph/debug/graph_debug.h"
#include "graph/manager/graph_manager.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/operator_reg.h"
#include "graph/utils/op_desc_utils.h"
#include "inc/pass_manager.h"
#include "init/gelib.h"
#include "opskernel_manager/ops_kernel_manager.h"

using namespace std;
using namespace testing;
using namespace ge;

class UTEST_graph_passes_save_pass : public testing::Test {
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

  vector<ge::NodePtr> targets{SaveNode};
  graph->SetGraphTargetNodesInfo(targets);

  // add edge
  ge::GraphUtils::AddEdge(VariableNode->GetOutDataAnchor(0), SaveNode->GetInDataAnchor(0));

  return graph;
}

TEST_F(UTEST_graph_passes_save_pass, cover_run_success) {
  ge::ComputeGraphPtr compute_graph = CreateSaveGraph();
  ge::PassManager pass_managers;
  pass_managers.AddPass(new (std::nothrow) SavePass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::SUCCESS);
}
