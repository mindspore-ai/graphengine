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
#include "aicpu/common/aicpu_task_struct.h"
#include "hybrid/executor/subgraph_context.h"
#include "hybrid/node_executor/aicpu/aicpu_node_executor.h"
#undef protected
#undef private

using namespace std;
using namespace testing;

namespace {
struct AicpuTaskStruct {
  aicpu::AicpuParamHead head;
  uint64_t io_addrp[6];
}__attribute__((packed));
}  // namespace

namespace ge {
using namespace hybrid;

class UtestAicpuNodeExecutor : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

static NodePtr CreateNode(ComputeGraphPtr graph, const string &name, const string &type, int in_num, int out_num) {
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

  return graph->AddNode(op_desc);
}

TEST_F(UtestAicpuNodeExecutor, aicpu_tf_node_task) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(graph, "frameworkop", FRAMEWORK_OP_TYPE, 4, 2);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;
  node_item->is_dynamic = true;
  node_item->shape_inference_type = DEPEND_COMPUTE;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 4;
  graph_item.total_outputs_ = 2;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = std::unique_ptr<CallbackManager>(new CallbackManager());

  auto node_state = subgraph_context.GetOrCreateNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  for (int i=0; i<4; ++i) {
    uint64_t value_0 = 512;
    TensorValue in_tensor0(&value_0, sizeof(value_0));
    subgraph_context.SetInput(*node_item, 0, in_tensor0);
  }

  uint64_t value_0 = 512;
  TensorValue out_tensor0(&value_0, sizeof(value_0));
  subgraph_context.SetOutput(*node_item, 0, out_tensor0);

  uint64_t value_1 = 512;
  TensorValue out_tensor1(&value_1, sizeof(value_1));
  subgraph_context.SetOutput(*node_item, 1, out_tensor1);

  // task
  domi::TaskDef task_def;
  domi::KernelExDef *kernel_ex_def = task_def.mutable_kernel_ex();
  kernel_ex_def->set_kernel_ext_info_size(12);

  AicpuExtInfo aicpu_ext_info;
  aicpu_ext_info.infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
  aicpu_ext_info.infoLen = sizeof(int32_t);
  int32_t type = node_item->shape_inference_type;
  memcpy_s(aicpu_ext_info.infoMsg, sizeof(int32_t), &type, sizeof(int32_t));
  char *ext_mem = (char*)malloc(sizeof(AicpuExtInfo) + sizeof(int32_t));
  memcpy_s(ext_mem, sizeof(AicpuExtInfo) + sizeof(int32_t), &aicpu_ext_info, sizeof(AicpuExtInfo) + sizeof(int32_t));
  std::string ext_info(ext_mem, sizeof(AicpuExtInfo) + sizeof(int32_t));

  std::string *mutable_ext_info = kernel_ex_def->mutable_kernel_ext_info();
  (*mutable_ext_info) = ext_info;

  hybrid_model.task_defs_[node] = std::vector<domi::TaskDef>({task_def, task_def});

  AicpuTfNodeTask aicpu_tf_node_task(node_item, task_def);

  ASSERT_EQ(aicpu_tf_node_task.Init(hybrid_model), SUCCESS);
  ASSERT_EQ(aicpu_tf_node_task.LaunchTask(*node_state->GetTaskContext()), SUCCESS);

  AicpuTaskStruct args;
  args.head.length = sizeof(args);
  args.head.ioAddrNum = 6;

  domi::TaskDef task_def2;
  task_def2.set_type(RT_MODEL_TASK_ALL_KERNEL);
  task_def2.mutable_kernel()->set_args(reinterpret_cast<const char *>(&args), args.head.length);
  task_def2.mutable_kernel()->set_args_size(args.head.length);

  hybrid_model.task_defs_[node] = std::vector<domi::TaskDef>({task_def2});

  AicpuNodeTask aicpu_node_task(node_item, task_def);
  ASSERT_EQ(aicpu_node_task.Init(hybrid_model), FAILED);
  ASSERT_EQ(aicpu_node_task.LaunchTask(*node_state->GetTaskContext()), SUCCESS);


  //kernel_ex_def->set_allocated_kernel_ext_info(nullptr);

  free(ext_mem);

}

}  // namespace ge

