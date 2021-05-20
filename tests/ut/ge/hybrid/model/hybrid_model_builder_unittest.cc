/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <gmock/gmock.h>

#include <vector>

#define private public
#define protected public
#include "hybrid/model/hybrid_model_builder.h"
#include "hybrid/node_executor/node_executor.h"

#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace hybrid;

class UtestHybridModelBuilder : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() { }
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

  return graph.AddNode(op_desc);
}

TEST_F(UtestHybridModelBuilder, normal_hybrid_model_build) {
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
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);

  auto enter1 = CreateNode(*graph, "enter", ENTER, 1, 1);
  auto merge1 = CreateNode(*graph, "merge", STREAMMERGE, 2, 2);
  auto less1 = CreateNode(*graph, "less", LESS, 2, 1);
  less1->GetOpDesc()->SetOpKernelLibName("AIcoreEngine");
  auto loop1 = CreateNode(*graph, "loopcond", LOOPCOND, 1, 1);
  auto switch_t = CreateNode(*graph, "switch_t", STREAMSWITCH, 2, 0);
  auto switch_f = CreateNode(*graph, "switch_f", STREAMSWITCH, 2, 0);
  auto ident1 = CreateNode(*graph, "identity", IDENTITY, 2, 1);
  auto add1 = CreateNode(*graph, "add", ADD, 2, 1);
  add1->GetOpDesc()->SetOpKernelLibName("AIcoreEngine");
  auto next1 = CreateNode(*graph, "next", NEXTITERATION, 1, 1);
  auto exit1 = CreateNode(*graph, "exit", EXIT, 1, 1);
  auto value0 = CreateNode(*graph, "const", CONSTANT, 0, 1);
  auto value1 = CreateNode(*graph, "const", CONSTANT, 0, 1);
  auto active1 = CreateNode(*graph, "active1", STREAMACTIVE, 0, 0);
  auto active2 = CreateNode(*graph, "active2", STREAMACTIVE, 0, 0);
  auto active3 = CreateNode(*graph, "active3", STREAMACTIVE, 0, 0);
  auto output1 = CreateNode(*graph, "net_output", NETOUTPUT, 1, 1);

  GraphUtils::AddEdge(enter1->GetOutDataAnchor(0), merge1->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), less1->GetInDataAnchor(0));
  GraphUtils::AddEdge(value1->GetOutDataAnchor(0), less1->GetInDataAnchor(1));
  GraphUtils::AddEdge(less1->GetOutDataAnchor(0), loop1->GetInDataAnchor(0));

  GraphUtils::AddEdge(loop1->GetOutDataAnchor(0), switch_t->GetInDataAnchor(0));
  GraphUtils::AddEdge(value1->GetOutDataAnchor(0), switch_t->GetInDataAnchor(1));
  GraphUtils::AddEdge(loop1->GetOutDataAnchor(0), switch_f->GetInDataAnchor(0));
  GraphUtils::AddEdge(value0->GetOutDataAnchor(0), switch_f->GetInDataAnchor(1));

  GraphUtils::AddEdge(switch_f->GetOutControlAnchor(), exit1->GetInControlAnchor());
  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), exit1->GetInDataAnchor(0));

  GraphUtils::AddEdge(switch_t->GetOutControlAnchor(), ident1->GetInControlAnchor());
  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), ident1->GetInDataAnchor(0));

  GraphUtils::AddEdge(ident1->GetOutDataAnchor(0), add1->GetInDataAnchor(0));
  GraphUtils::AddEdge(value1->GetOutDataAnchor(0), add1->GetInDataAnchor(1));
  GraphUtils::AddEdge(add1->GetOutDataAnchor(0), next1->GetInDataAnchor(0));

  GraphUtils::AddEdge(enter1->GetOutControlAnchor(), active1->GetInControlAnchor());
  GraphUtils::AddEdge(active1->GetOutControlAnchor(), merge1->GetInControlAnchor());

  GraphUtils::AddEdge(loop1->GetOutControlAnchor(), active2->GetInControlAnchor());
  GraphUtils::AddEdge(active2->GetOutControlAnchor(), switch_f->GetInControlAnchor());
  GraphUtils::AddEdge(active2->GetOutControlAnchor(), switch_t->GetInControlAnchor());

  GraphUtils::AddEdge(next1->GetOutControlAnchor(), active3->GetInControlAnchor());

  GraphUtils::AddEdge(exit1->GetOutDataAnchor(0), output1->GetInDataAnchor(0));
  AttrUtils::SetStr(merge1->GetOpDesc(), ATTR_NAME_NEXT_ITERATION, next1->GetName());

  AttrUtils::SetBool(enter1->GetOpDesc(), ATTR_NAME_INSERT_FP_PROFILILNG_TASK, true);
  AttrUtils::SetBool(output1->GetOpDesc(), ATTR_NAME_INSERT_BP_PROFILILNG_TASK, true);
  AttrUtils::SetBool(add1->GetOpDesc(), ATTR_NAME_INSERT_FP_PROFILILNG_TASK, true);
  AttrUtils::SetBool(add1->GetOpDesc(), ATTR_NAME_INSERT_BP_PROFILILNG_TASK, true);

  // Build -> IndexSpecialNodes --> stream_merge_op_nodes_
  // Build -> LoadGraph -> RelinkNextIteration
  // Build -> LoadGraph -> LoadDynamicSubgraph --> BuildNodeItem --> NodeItem::SetDataSend
  // Build -> LoadGraph -> LoadDynamicSubgraph --> BuildControlFlowGroup --> NodeItem::SetCtrlSend
  auto &engine_mapping = NodeExecutorManager::GetInstance().engine_mapping_;
  engine_mapping.emplace("AIcoreEngine", NodeExecutorManager::ExecutorType::AICORE);
  engine_mapping.emplace("DNN_VM_GE_LOCAL_OP_STORE", NodeExecutorManager::ExecutorType::GE_LOCAL);
  engine_mapping.emplace("aicpu_tf_kernel", NodeExecutorManager::ExecutorType::AICPU_TF);
  engine_mapping.emplace("aicpu_ascend_kernel", NodeExecutorManager::ExecutorType::AICPU_TF);
  engine_mapping.emplace("ops_kernel_info_hccl", NodeExecutorManager::ExecutorType::HCCL);
  engine_mapping.emplace("DNN_VM_RTS_OP_STORE", NodeExecutorManager::ExecutorType::RTS);
  engine_mapping.emplace("DNN_VM_HOST_CPU_OP_STORE", NodeExecutorManager::ExecutorType::HOST_CPU);

  auto &task_executor = NodeExecutorManager::GetInstance().executors_;
  task_executor.emplace(NodeExecutorManager::ExecutorType::AICORE, std::unique_ptr<NodeExecutor>(new NodeExecutor()));
  task_executor.emplace(NodeExecutorManager::ExecutorType::GE_LOCAL, std::unique_ptr<NodeExecutor>(new NodeExecutor()));
  task_executor.emplace(NodeExecutorManager::ExecutorType::AICPU_TF, std::unique_ptr<NodeExecutor>(new NodeExecutor()));
  task_executor.emplace(NodeExecutorManager::ExecutorType::HCCL, std::unique_ptr<NodeExecutor>(new NodeExecutor()));
  task_executor.emplace(NodeExecutorManager::ExecutorType::RTS, std::unique_ptr<NodeExecutor>(new NodeExecutor()));
  task_executor.emplace(NodeExecutorManager::ExecutorType::HOST_CPU, std::unique_ptr<NodeExecutor>(new NodeExecutor()));

  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);
  ASSERT_EQ(hybrid_model_builder.Build(), SUCCESS);
  engine_mapping.clear();
  task_executor.clear();
}

TEST_F(UtestHybridModelBuilder, create_called_invalid) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  auto node = CreateNode(*graph, "node", PARTITIONEDCALL, 1, 1);
  NodeItem node_item(node);

  ASSERT_EQ(hybrid_model_builder.CreateStreamActiveGroup(node, &node_item), INTERNAL_ERROR);
  ASSERT_EQ(hybrid_model_builder.CreateStreamSwitchGroup(node, &node_item), INTERNAL_ERROR);
  ASSERT_EQ(hybrid_model_builder.CreateNextIterationGroup(node, &node_item), INTERNAL_ERROR);
  ASSERT_EQ(hybrid_model_builder.CreateStreamSwitchNGroup(node, &node_item), INTERNAL_ERROR);
  ASSERT_EQ(hybrid_model_builder.CreateSwitchGroup(node, &node_item), INTERNAL_ERROR);

  ASSERT_EQ(hybrid_model_builder.CreateLabelSetGroup(node, &node_item), INTERNAL_ERROR);
  node_item.node_type = LABELSET;
  ASSERT_EQ(hybrid_model_builder.CreateLabelSetGroup(node, &node_item), UNSUPPORTED);

  ASSERT_EQ(hybrid_model_builder.CreateLabelGotoGroup(node, &node_item), INTERNAL_ERROR);
  node_item.node_type = LABELGOTO;
  ASSERT_EQ(hybrid_model_builder.CreateLabelGotoGroup(node, &node_item), UNSUPPORTED);

  ASSERT_EQ(hybrid_model_builder.CreateLabelSwitchGroup(node, &node_item), INTERNAL_ERROR);
  node_item.node_type = LABELSWITCH;
  ASSERT_EQ(hybrid_model_builder.CreateLabelSwitchGroup(node, &node_item), UNSUPPORTED);
}

TEST_F(UtestHybridModelBuilder, stream_switch_n_group) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  auto switch_n = CreateNode(*graph, "switch_n", STREAMSWITCHN, 1, 0);
  NodeItem node_item(switch_n);

  // no batch_num
  ASSERT_EQ(hybrid_model_builder.CreateStreamSwitchNGroup(switch_n, &node_item), INTERNAL_ERROR);

  uint32_t batch_num = 0;
  AttrUtils::SetInt(switch_n->GetOpDesc(), ATTR_NAME_BATCH_NUM, batch_num);
  ASSERT_EQ(hybrid_model_builder.CreateStreamSwitchNGroup(switch_n, &node_item), SUCCESS);

  batch_num = 3;
  AttrUtils::SetInt(switch_n->GetOpDesc(), ATTR_NAME_BATCH_NUM, batch_num);
  ASSERT_EQ(hybrid_model_builder.CreateStreamSwitchNGroup(switch_n, &node_item), SUCCESS);
}
} // namespace ge