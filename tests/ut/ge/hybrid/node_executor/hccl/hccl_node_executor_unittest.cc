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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#define private public
#define protected public
#include "graph/runtime_inference_context.h"
#include "hybrid/executor/subgraph_context.h"
#include "hybrid/node_executor/hccl/hccl_node_executor.h"
#undef protected
#undef private

using namespace std;
using namespace testing;
namespace ge {
using namespace hybrid;

class UtestHcclNodeExecutor : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

static NodePtr CreateNode(ComputeGraph &graph, const string &name, const string &type, int in_num, int out_num) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  op_desc->SetStreamId(0);
  static int32_t index = 0;
  op_desc->SetId(index++);

  GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT64);
  TensorUtils::SetSize(tensor, 64);
  vector<int64_t> input_offset;
  for (int i = 0; i < in_num; i++) {
    op_desc->AddInputDesc(tensor);
    input_offset.emplace_back(i * 64);
  }
  op_desc->SetInputOffset(input_offset);

  vector<int64_t> output_offset;
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(in_num * 64 + i * 64);
  }
  op_desc->SetOutputOffset(output_offset);

  return graph.AddNode(op_desc);
}

TEST_F(UtestHcclNodeExecutor, test_rdmatask_extract_tensor) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = CreateNode(*graph, "hcom", HCOMREMOTEREAD, 0, 0);
  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphItem graph_item;
  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);

  auto node_state = subgraph_context.GetOrCreateNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  RuntimeInferenceContext::CreateContext(std::to_string(graph_context.context_id));
  RuntimeInferenceContext *ctx = nullptr;
  RuntimeInferenceContext::GetContext(std::to_string(graph_context.context_id), &ctx);

  Shape s({1, 3});
  TensorDesc tensor_desc(s);
  Tensor tensor(tensor_desc);
  std::vector<uint8_t> data = {1, 2, 3, 4};
  tensor.SetData(data);
  ctx->SetTensor(1, 0, tensor.Clone());

  auto unique_task_context = TaskContext::Create(node_state.get(), &graph_context, &subgraph_context);
  vector<HcomRemoteAccessAddrInfo> addr_infos;
  shared_ptr<RdmaNodeTask> task = MakeShared<RdmaNodeTask>();
  task->remote_index_ = {1, 0};
  ASSERT_EQ(task->ExtractTensor(*unique_task_context, addr_infos), PARAM_INVALID);

  Shape s2({1});
  TensorDesc tensor_desc2(s2);
  Tensor tensor2(tensor_desc2);
  ctx->SetTensor(1, 0, tensor2.Clone());
  task->ExtractTensor(*unique_task_context, addr_infos);
  ASSERT_EQ(task->ExtractTensor(*unique_task_context, addr_infos), PARAM_INVALID);
  RuntimeInferenceContext::DestroyContext(std::to_string(graph_context.context_id));
}
}  // namespace ge