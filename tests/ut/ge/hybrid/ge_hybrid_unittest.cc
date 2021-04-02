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
#include "runtime/rt.h"

#define protected public
#define private public
#include "hybrid/model/hybrid_model_builder.h"
#include "hybrid/model/hybrid_model.h"
#include "model/ge_model.h"
#include "model/ge_root_model.h"
#include "hybrid/node_executor/aicore/aicore_op_task.h"
#include "framework/common/taskdown_common.h"
#include "framework/common/debug/log.h"
#include "graph/ge_context.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/hybrid_model_executor.h"
#include "hybrid/node_executor/aicore/aicore_task_builder.h"
#include "graph/load/model_manager/tbe_handle_store.h"
#include "graph/manager/graph_mem_allocator.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "graph/types.h"
#include "graph/utils/tensor_utils.h"

#undef private
#undef protected

using namespace std;
using namespace testing;
using namespace ge;
using namespace hybrid;


class UtestGeHybrid : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

static ge::OpDescPtr CreateOpDesc(string name = "", string type = "") {
  auto op_desc = std::make_shared<ge::OpDesc>(name, type);
  op_desc->SetStreamId(0);
  op_desc->SetId(0);

  op_desc->SetWorkspace({});
  ;
  op_desc->SetWorkspaceBytes({});
  op_desc->SetInputOffset({});
  op_desc->SetOutputOffset({});

  ge::AttrUtils::SetStr(op_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  bool support_dynamic = true;
  ge::AttrUtils::GetBool(op_desc, "support_dynamicshape", support_dynamic);
  return op_desc;
}

TEST_F(UtestGeHybrid, aicore_op_task_init_success) {
  // build aicore task
  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
  domi::TaskDef task_def;
  task_def.set_type(RT_MODEL_TASK_ALL_KERNEL);
  domi::KernelDefWithHandle *kernel_with_handle = task_def.mutable_kernel_with_handle();
  kernel_with_handle->set_original_kernel_key("");
  kernel_with_handle->set_node_info("");
  kernel_with_handle->set_block_dim(32);
  kernel_with_handle->set_args_size(64);
  string args(64, '1');
  kernel_with_handle->set_args(args.data(), 64);
  domi::KernelContext *context = kernel_with_handle->mutable_context();
  context->set_op_index(1);
  context->set_kernel_type(2);    // ccKernelType::TE
  uint16_t args_offset[9] = {0};
  context->set_args_offset(args_offset, 9 * sizeof(uint16_t));

  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/Add", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  std::string kernel_name("kernel/Add");
  AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", kernel_name);
  ASSERT_EQ(aicore_task->InitWithTaskDef(*op_desc.get(), task_def), SUCCESS);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  ASSERT_EQ(aicore_task->LaunchKernel(stream), SUCCESS);
  char *handle = "";
  aicore_task->handle_ = handle;
  aicore_task->tiling_key_ = 1;
  ASSERT_EQ(aicore_task->LaunchKernel(stream), SUCCESS);
}

TEST_F(UtestGeHybrid, task_update_tiling_info) {
  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
  aicore_task->is_single_op_ = true;
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  ge::AttrUtils::SetStr(op_desc, "compile_info_key", "key");
  ge::AttrUtils::SetStr(op_desc, "compile_info_json", "json");
  auto node = graph->AddNode(op_desc);
  optiling::OpRunInfo tiling_info;
  ASSERT_EQ(aicore_task->CalcTilingInfo(node, tiling_info), SUCCESS);
}

TEST_F(UtestGeHybrid, index_taskdefs_failed) {
  // build aicore task
  domi::ModelTaskDef model_task_def;

  std::shared_ptr<domi::ModelTaskDef> model_task_def_ptr = make_shared<domi::ModelTaskDef>(model_task_def);
  domi::TaskDef *task_def = model_task_def_ptr->add_task();
  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetModelTaskDef(model_task_def_ptr);

  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
  task_def->set_type(RT_MODEL_TASK_ALL_KERNEL);
  domi::KernelDefWithHandle *kernel_with_handle = task_def->mutable_kernel_with_handle();
  kernel_with_handle->set_original_kernel_key("");
  kernel_with_handle->set_node_info("");
  kernel_with_handle->set_block_dim(32);
  kernel_with_handle->set_args_size(64);
  string args(64, '1');
  kernel_with_handle->set_args(args.data(), 64);
  domi::KernelContext *context = kernel_with_handle->mutable_context();
  context->set_op_index(1);
  context->set_kernel_type(2);    // ccKernelType::TE
  uint16_t args_offset[9] = {0};
  context->set_args_offset(args_offset, 9 * sizeof(uint16_t));

  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/Add", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  std::string kernel_name("kernel/Add");
  AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", kernel_name);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  ASSERT_EQ(hybrid_model_builder.IndexTaskDefs(graph, ge_model), INTERNAL_ERROR);
}

TEST_F(UtestGeHybrid, parse_force_infershape_nodes) {
  const char *const kForceInfershape = "_force_infershape_when_running";
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Conv2D", "Conv2D");
  ge::AttrUtils::SetBool(op_desc, kForceInfershape, true);
  auto node = graph->AddNode(op_desc);
  std::unique_ptr<NodeItem> new_node;
  NodeItem::Create(node, new_node);
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);
  ASSERT_EQ(hybrid_model_builder.ParseForceInfershapeNodes(node, *new_node), SUCCESS);
}

TEST_F(UtestGeHybrid, index_taskdefs_success) {
  // build aicore task
  domi::ModelTaskDef model_task_def;

  std::shared_ptr<domi::ModelTaskDef> model_task_def_ptr = make_shared<domi::ModelTaskDef>(model_task_def);
  domi::TaskDef *task_def = model_task_def_ptr->add_task();
  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetModelTaskDef(model_task_def_ptr);

  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
  task_def->set_type(RT_MODEL_TASK_ALL_KERNEL);
  domi::KernelDefWithHandle *kernel_with_handle = task_def->mutable_kernel_with_handle();
  kernel_with_handle->set_original_kernel_key("");
  kernel_with_handle->set_node_info("");
  kernel_with_handle->set_block_dim(32);
  kernel_with_handle->set_args_size(64);
  string args(64, '1');
  kernel_with_handle->set_args(args.data(), 64);
  domi::KernelContext *context = kernel_with_handle->mutable_context();
  context->set_op_index(0);
  context->set_kernel_type(2);    // ccKernelType::TE
  uint16_t args_offset[9] = {0};
  context->set_args_offset(args_offset, 9 * sizeof(uint16_t));

  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/Add", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  std::string kernel_name("kernel/Add");
  AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", kernel_name);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  ASSERT_EQ(hybrid_model_builder.IndexTaskDefs(graph, ge_model), SUCCESS);
}

TEST_F(UtestGeHybrid, init_weight_success) {
  NpuMemoryAllocator::allocators_.emplace(make_pair(0, nullptr));
  // make graph with sub_graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("root_graph");
  OpDescPtr op_desc = CreateOpDesc("if", IF);
  NodePtr node = graph->AddNode(op_desc);
  // make sub graph
  ComputeGraphPtr sub_graph = std::make_shared<ComputeGraph>("if_sub_graph");
  OpDescPtr const_op_desc = CreateOpDesc("const", CONSTANT);
  vector<int64_t> dims_vec_0 = {2, 1, 4, 1, 2};
  vector<int32_t> data_vec_0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  (void)TensorUtils::SetRealDimCnt(tensor_desc_0, dims_vec_0.size());
  ConstGeTensorPtr constTensor_0 =
    std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)&data_vec_0[0], data_vec_0.size() * sizeof(int32_t));
  AttrUtils::SetTensor(const_op_desc, ge::ATTR_NAME_WEIGHTS, constTensor_0);
  const_op_desc->AddOutputDesc(tensor_desc_0);
  NodePtr const_node = sub_graph->AddNode(const_op_desc);
  graph->AddSubgraph("sub", sub_graph);

  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  //Buffer weight_buffer = Buffer(128,0);
  //ge_sub_model->SetWeight(weight_buffer);
  ge_root_model->SetSubgraphInstanceNameToModel("sub",ge_sub_model);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);
  auto ret = hybrid_model_builder.InitWeights();
  ASSERT_EQ(ret,SUCCESS);
  Buffer weight_buffer = Buffer(128,0);
  ge_sub_model->SetWeight(weight_buffer);
  ret = hybrid_model_builder.InitWeights();
  ASSERT_EQ(ret,PARAM_INVALID);
}

  TEST_F(UtestGeHybrid, hybrid_model_executor) {
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("abc");
  GeRootModelPtr root_model = MakeShared<ge::GeRootModel>(compute_graph);
  HybridModel model(root_model);
  HybridModel *model_ptr = &model;

  uint32_t device_id = 0;
  rtStream_t stream;
  HybridModelExecutor executor(model_ptr, device_id, stream);
  executor.Init();
}

TEST_F(UtestGeHybrid, unfold_subgraphs_success) {
ComputeGraphPtr merged_graph = nullptr;

ComputeGraphPtr sub_sub_graph1 = std::make_shared<ComputeGraph>("while_cond");
OpDescPtr sub_sub_graph_while_cond_data_op_desc = CreateOpDesc("cond_data", DATA);
NodePtr sub_sub_graph_while_cond_data_node = sub_sub_graph1->AddNode(sub_sub_graph_while_cond_data_op_desc);

ComputeGraphPtr sub_sub_graph2 = std::make_shared<ComputeGraph>("while_body");
/*OpDescPtr sub_sub_graph_while_body_const_op_desc = CreateOpDesc("body_const", CONSTANT);
NodePtr sub_sub_graph_while_body_const_node = sub_sub_graph2->AddNode(sub_sub_graph_while_body_const_op_desc);*/
OpDescPtr sub_sub_graph_while_body_data_op_desc = CreateOpDesc("body_data", DATA);
NodePtr sub_sub_graph_while_body_data_node = sub_sub_graph2->AddNode(sub_sub_graph_while_body_data_op_desc);
sub_sub_graph2->SetGraphUnknownFlag(true);
/*OpDescPtr sub_sub_graph_while_body_add_op_desc = CreateOpDesc("body_add", ADD);
NodePtr sub_sub_graph_while_body_add_node = sub_sub_graph2->AddNode(sub_sub_graph_while_body_add_node);
sub_sub_graph_while_body_add_node->AddLinkFrom(sub_sub_graph_while_body_data_node);
sub_sub_graph_while_body_add_node->AddLinkFrom(sub_sub_graph_while_body_const_node);*/

ComputeGraphPtr sub_graph = std::make_shared<ComputeGraph>("sub_graph");
OpDescPtr sub_graph_while_op_desc = CreateOpDesc("while", WHILE);
NodePtr sub_graph_while_node = sub_graph->AddNode(sub_graph_while_op_desc);
sub_graph->SetGraphUnknownFlag(true);
sub_graph_while_node->GetOpDesc()->AddSubgraphName("while_cond");
sub_graph_while_node->GetOpDesc()->AddSubgraphName("while_body");
sub_graph_while_node->GetOpDesc()->SetSubgraphInstanceName(0, "while_cond");
sub_graph_while_node->GetOpDesc()->SetSubgraphInstanceName(1, "while_body");

ComputeGraphPtr root_graph = std::make_shared<ComputeGraph>("root_graph");
auto partitioned_call_op_desc = MakeShared<OpDesc>("partitioned_call", PARTITIONEDCALL);
auto partitioned_call_node = root_graph->AddNode(partitioned_call_op_desc);
partitioned_call_node->GetOpDesc()->AddSubgraphName("sub_graph");
partitioned_call_node->GetOpDesc()->SetSubgraphInstanceName(0, "sub_graph");

root_graph->AddSubGraph(sub_sub_graph1);
root_graph->AddSubGraph(sub_sub_graph2);
sub_sub_graph1->SetParentGraph(root_graph);
sub_sub_graph2->SetParentGraph(root_graph);
sub_sub_graph1->SetParentNode(sub_graph_while_node);
sub_sub_graph2->SetParentNode(sub_graph_while_node);

root_graph->AddSubGraph(sub_graph);
sub_graph->SetParentNode(partitioned_call_node);
sub_graph->SetParentGraph(root_graph);

GeRootModelPtr root_model = MakeShared<ge::GeRootModel>(root_graph);
HybridModel hybrid_model(root_model);
HybridModelBuilder hybrid_model_builder(hybrid_model);

// subgraph num before unfold: 1
EXPECT_EQ(root_graph->GetAllSubgraphs().size(), 3);
// num of nodes in root_graph before unfold: 1, name: partitioned_call
EXPECT_EQ(root_graph->GetDirectNodesSize(), 1);
EXPECT_EQ(root_graph->GetDirectNode().at(0)->GetName(), "partitioned_call");
// two sub_sub_graphs: while cond & while body, their parent graph is "subgraph" before unfold
EXPECT_EQ(sub_sub_graph1->GetParentGraph()->GetName(), "root_graph");
EXPECT_EQ(sub_sub_graph1->GetParentGraph()->GetName(), "root_graph");
// node "cond_data" & "body_data" has owner compute graph "subgraph" before unfold
EXPECT_EQ(sub_graph_while_node->GetOwnerComputeGraph()->GetName(), "sub_graph");

// unfold success
EXPECT_EQ(hybrid_model_builder.UnfoldSubgraphs(root_graph, merged_graph), SUCCESS);

// subgraph num after unfold: 0
EXPECT_EQ(merged_graph->GetAllSubgraphs().size(), 2);
// num of nodes in MergedGraph after unfold: 1, name: while
EXPECT_EQ(merged_graph->GetDirectNodesSize(), 1);
EXPECT_EQ(merged_graph->GetDirectNode().at(0)->GetName(), "while");
// two sub_sub_graphs: while cond & while body, their parent graph is "MergedGraph" after unfold
EXPECT_EQ(sub_sub_graph1->GetParentGraph()->GetName(), "MergedGraph" );
EXPECT_EQ(sub_sub_graph1->GetParentGraph()->GetName(), "MergedGraph");
// node "cond_data" & "body_data" has owner compute graph "MergedGraph" before unfold
EXPECT_EQ(sub_graph_while_node->GetOwnerComputeGraph()->GetName(), "MergedGraph");
}
