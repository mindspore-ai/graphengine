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

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include "runtime/rt.h"

#define protected public
#define private public
#include "hybrid/model/hybrid_model.h"
#include "hybrid/node_executor/node_executor.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/hybrid_model_executor.h"
#include "hybrid/executor/worker/execution_engine.h"
#include "hybrid/executor/subgraph_executor.h"
#undef private
#undef protected

using namespace std;
using namespace testing;
using namespace ge;
using namespace hybrid;


class UtestExecutionEngine : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {
  }
};
namespace {
const int kIntBase = 10;
}
static ge::OpDescPtr CreateOpDesc(string name = "", string type = "") {
  auto op_desc = std::make_shared<ge::OpDesc>(name, type);
  op_desc->SetStreamId(0);
  op_desc->SetId(0);
  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetInputOffset({});
  op_desc->SetOutputOffset({});

  ge::AttrUtils::SetStr(op_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  bool support_dynamic = true;
  ge::AttrUtils::GetBool(op_desc, "support_dynamicshape", support_dynamic);
  return op_desc;
}

TEST_F(UtestExecutionEngine, ExecuteAsync_without_kernel_task) {
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);
  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(node, node_item);
  ASSERT_TRUE(node_item != nullptr);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphExecutionContext execution_context;
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_ = std::unique_ptr<GraphItem>(new(std::nothrow)GraphItem());
  execution_context.model = &hybrid_model;
  execution_context.profiling_level = 1;
  SubgraphContext subgraph_context(nullptr, &execution_context);

  NodeState node_state(*node_item, &subgraph_context);
  auto task_context = TaskContext::Create(&node_state, &execution_context, &subgraph_context);
  auto shared_task_context = std::shared_ptr<TaskContext>(task_context.release());
  node_state.SetTaskContext(shared_task_context);

  ExecutionEngine execution_engine;
  ASSERT_TRUE(node_state.GetTaskContext() != nullptr);

  std::function<void()> callback;
  SubgraphExecutor executor(hybrid_model.GetRootGraphItem(), &execution_context);
  executor.InitCallback(&node_state, callback);
  EXPECT_EQ(execution_engine.ExecuteAsync(node_state, node_state.GetTaskContext(), execution_context, callback), INTERNAL_ERROR);
}

TEST_F(UtestExecutionEngine, ExecuteAsync_without_callback_and_kernel_task) {
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);
  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(node, node_item);
  ASSERT_TRUE(node_item != nullptr);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphExecutionContext execution_context;
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_ = std::unique_ptr<GraphItem>(new(std::nothrow)GraphItem());
  execution_context.model = &hybrid_model;
  SubgraphContext subgraph_context(nullptr, &execution_context);

  NodeState node_state(*node_item, &subgraph_context);
  auto task_context = TaskContext::Create(&node_state, &execution_context, &subgraph_context);
  auto shared_task_context = std::shared_ptr<TaskContext>(task_context.release());
  node_state.SetTaskContext(shared_task_context);

  ExecutionEngine execution_engine;
  ASSERT_TRUE(node_state.GetTaskContext() != nullptr);

  std::function<void()> callback;
  SubgraphExecutor executor(hybrid_model.GetRootGraphItem(), &execution_context);
  executor.InitCallback(&node_state, callback);
  EXPECT_EQ(execution_engine.ExecuteAsync(node_state, node_state.GetTaskContext(), execution_context, callback), INTERNAL_ERROR);
}
