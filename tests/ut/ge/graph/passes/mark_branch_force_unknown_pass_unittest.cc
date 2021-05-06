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

#define protected public
#define private public
#include "graph/passes/mark_branch_force_unknown_pass.h"

#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/operator_factory.h"
#include "graph/operator_reg.h"
#include "graph_builder_utils.h"

using namespace std;
using namespace testing;
namespace ge {
class UtestMarkBranchForceUnknownPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

static NodePtr CreateNode(ComputeGraph &graph, const string &name, const string &type, int in_num, int out_num) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  op_desc->SetStreamId(0);
  static int32_t index = 0;
  op_desc->SetId(index++);

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  vector<int64_t> input_offset;
  for (int i = 0; i < in_num; i++) {
    op_desc->AddInputDesc(tensor);
    input_offset.emplace_back(1024);
  }
  op_desc->SetInputOffset(input_offset);

  vector<int64_t> output_offset;
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(1024);
  }
  op_desc->SetOutputOffset(output_offset);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  const auto stub_func = [](Operator &op) { return GRAPH_SUCCESS; };
  op_desc->AddInferFunc(stub_func);
  op_desc->AddInferFormatFunc(stub_func);
  op_desc->AddVerifierFunc(stub_func);

  return graph.AddNode(op_desc);
}

static void CreateLoopGraph(ComputeGraphPtr &graph, NodePtr &merge) {
/*******************************************************************************
 *      Exit         Identify
 *        \         /       \.
 *         \       /         \.
 *          Switch           Add
 *         /     |            |
 *        /      |            |
 *       /       |            |
 *  LoopCond     |            |
 *      \        |            |
 *       \       |            |
 *        \      |            |
 *       Less    |            |
 *          \    |       NextIteration
 *           \   |            |
 *            \  |            |
 *            Merge <---------|
 *              |
 *              |
 *            Enter
 ******************************************************************************/
  auto data1 = CreateNode(*graph, "data", DATA, 1, 1);
  auto enter1 = CreateNode(*graph, "enter", ENTER, 1, 1);
  auto merge1 = CreateNode(*graph, "merge", MERGE, 2, 2);
  auto less1 = CreateNode(*graph, "less", LESS, 2, 1);
  auto loop1 = CreateNode(*graph, "loopcond", LOOPCOND, 1, 1);
  auto switch1 = CreateNode(*graph, "switch", SWITCH, 2, 2);
  auto ident1 = CreateNode(*graph, "identity", IDENTITY, 1, 1);
  auto add1 = CreateNode(*graph, "add", ADD, 2, 1);
  auto next1 = CreateNode(*graph, "next", NEXTITERATION, 1, 1);
  auto exit1 = CreateNode(*graph, "exit", EXIT, 1, 1);
  auto value0 = CreateNode(*graph, "const", CONSTANT, 0, 1);
  auto value1 = CreateNode(*graph, "const", CONSTANT, 0, 1);
  auto output1 = CreateNode(*graph, "net_output", NETOUTPUT, 1, 1);

  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), enter1->GetInDataAnchor(0));
  GraphUtils::AddEdge(enter1->GetOutDataAnchor(0), merge1->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), less1->GetInDataAnchor(0));
  GraphUtils::AddEdge(value1->GetOutDataAnchor(0), less1->GetInDataAnchor(1));
  GraphUtils::AddEdge(less1->GetOutDataAnchor(0), loop1->GetInDataAnchor(0));

  GraphUtils::AddEdge(loop1->GetOutDataAnchor(0), switch1->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), switch1->GetInDataAnchor(1));

  GraphUtils::AddEdge(switch1->GetOutDataAnchor(0), exit1->GetInDataAnchor(0));
  GraphUtils::AddEdge(switch1->GetOutDataAnchor(1), ident1->GetInDataAnchor(0));

  GraphUtils::AddEdge(ident1->GetOutDataAnchor(0), add1->GetInDataAnchor(0));
  GraphUtils::AddEdge(value1->GetOutDataAnchor(0), add1->GetInDataAnchor(1));
  GraphUtils::AddEdge(add1->GetOutDataAnchor(0), next1->GetInDataAnchor(0));

  GraphUtils::AddEdge(next1->GetOutDataAnchor(0), merge1->GetInDataAnchor(1));
  GraphUtils::AddEdge(exit1->GetOutDataAnchor(0), output1->GetInDataAnchor(0));

  merge = merge1;
}

static void CreateCondGraph(ComputeGraphPtr &graph, NodePtr &merge) {
/*******************************************************************************
 *        NetOutput
 *            |
 *            |
 *          Merge
 *         /     \.
 *        /       \.
 *       /         \.
 *     Add          Sub
 *     |  \         |  \.
 *     |   \        |   \.
 *     |    \       |    Const
 *     |     \      |      \.
 *     |      \     |      Identify
 *     |       \    |        |
 *  Switch  Switch Switch  Switch
 *     |     |     |   |    |
 *     |     |     |   |    |
 *     x     y   Cond  z
 ******************************************************************************/
  auto data1 = CreateNode(*graph, "data_x", DATA, 1, 1);
  auto data2 = CreateNode(*graph, "data_y", DATA, 1, 1);
  auto data3 = CreateNode(*graph, "data_z", DATA, 1, 1);

  auto less1 = CreateNode(*graph, "less", LESS, 2, 1);

  auto switch1 = CreateNode(*graph, "switch_x", SWITCH, 2, 2);
  auto switch2 = CreateNode(*graph, "switch_y", SWITCH, 2, 2);
  auto switch3 = CreateNode(*graph, "switch_z", SWITCH, 2, 2);
  auto switch4 = CreateNode(*graph, "switch_i", SWITCH, 2, 2);

  auto add1 = CreateNode(*graph, "add", ADD, 2, 1);
  auto sub1 = CreateNode(*graph, "add", SUB, 2, 1);
  auto ident1 = CreateNode(*graph, "identity", IDENTITY, 1, 1);
  auto const1 = CreateNode(*graph, "const", CONSTANT, 0, 1);

  auto merge1 = CreateNode(*graph, "merge", MERGE, 2, 2);
  auto output1 = CreateNode(*graph, "net_output", NETOUTPUT, 1, 1);

  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), less1->GetInDataAnchor(0));
  GraphUtils::AddEdge(data2->GetOutDataAnchor(0), less1->GetInDataAnchor(1));

  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), switch1->GetInDataAnchor(0));
  GraphUtils::AddEdge(data2->GetOutDataAnchor(0), switch2->GetInDataAnchor(0));
  GraphUtils::AddEdge(data3->GetOutDataAnchor(0), switch3->GetInDataAnchor(0));
  GraphUtils::AddEdge(less1->GetOutDataAnchor(0), switch4->GetInDataAnchor(0));

  GraphUtils::AddEdge(less1->GetOutDataAnchor(0), switch1->GetInDataAnchor(1));
  GraphUtils::AddEdge(less1->GetOutDataAnchor(0), switch2->GetInDataAnchor(1));
  GraphUtils::AddEdge(less1->GetOutDataAnchor(0), switch3->GetInDataAnchor(1));
  GraphUtils::AddEdge(less1->GetOutDataAnchor(0), switch4->GetInDataAnchor(1));

  GraphUtils::AddEdge(switch1->GetOutDataAnchor(0), add1->GetInDataAnchor(0));
  GraphUtils::AddEdge(switch2->GetOutDataAnchor(0), add1->GetInDataAnchor(1));
  GraphUtils::AddEdge(switch3->GetOutDataAnchor(0), sub1->GetInDataAnchor(0));
  GraphUtils::AddEdge(switch4->GetOutDataAnchor(0), ident1->GetInDataAnchor(1));
  GraphUtils::AddEdge(ident1->GetOutControlAnchor(), const1->GetInControlAnchor());
  GraphUtils::AddEdge(const1->GetOutDataAnchor(0), sub1->GetInDataAnchor(1));

  GraphUtils::AddEdge(add1->GetOutDataAnchor(0), merge1->GetInDataAnchor(0));
  GraphUtils::AddEdge(sub1->GetOutDataAnchor(0), merge1->GetInDataAnchor(1));
  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), output1->GetInDataAnchor(0));

  merge = merge1;
}

TEST_F(UtestMarkBranchForceUnknownPass, skip_while_loop_merge) {
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr merge;
  CreateLoopGraph(graph, merge);

  AttrUtils::SetBool(merge->GetOpDesc(), ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);

  MarkBranchForceUnknownPass mark_force_unknown_pass;
  EXPECT_EQ(mark_force_unknown_pass.Run(graph), SUCCESS);   // skip LoopCond
}

TEST_F(UtestMarkBranchForceUnknownPass, skip_known_shape_merge) {
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr merge;
  CreateCondGraph(graph, merge);

  MarkBranchForceUnknownPass mark_force_unknown_pass;
  EXPECT_EQ(mark_force_unknown_pass.Run(graph), SUCCESS);   // skip known shape merge
}


TEST_F(UtestMarkBranchForceUnknownPass, mark_unknown_shape_merge) {
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr merge;
  CreateCondGraph(graph, merge);

  auto tensor_desc = merge->GetOpDesc()->GetOutputDesc(0);
  tensor_desc.SetShape(GeShape({-1}));  // Set for unknown.
  merge->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);

  MarkBranchForceUnknownPass mark_force_unknown_pass;
  EXPECT_EQ(mark_force_unknown_pass.Run(graph), SUCCESS);
}
}  // namespace ge
