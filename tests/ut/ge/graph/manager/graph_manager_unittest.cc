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
#include <memory>
#define protected public
#define private public
#include "graph/manager/graph_manager.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/davinci_model.h"
#define const
#include "common/helper/model_cache_helper.h"
#undef const
#include "init/gelib.h"
#undef private
#undef public

#include <pthread.h>
#include <algorithm>
#include <future>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <future>

#include "common/math/math_util.h"
#include "common/thread_pool.h"
#include "common/dump/dump_manager.h"
#include "analyzer/analyzer.h"
#include "graph/common/ge_call_wrapper.h"
#include "graph/common/local_context.h"
#include "graph/common/transop_util.h"
#include "graph/ge_context.h"
#include "graph/ge_global_options.h"
#include "graph/manager/util/rt_context_util.h"
#include "graph/partition/dynamic_shape_partition.h"
#include "graph/passes/enter_pass.h"
#include "graph/partition/stage_partition.h"
#include "graph/passes/addn_pass.h"
#include "graph/passes/bitcast_pass.h"
#include "graph/passes/assign_remove_pass.h"
#include "graph/passes/inplace_support_check_pass.h"
#include "graph/passes/atomic_addr_clean_pass.h"
#include "graph/passes/attach_stream_label_pass.h"
#include "graph/passes/cast_remove_pass.h"
#include "graph/passes/common_subexpression_elimination_pass.h"
#include "graph/passes/compile_nodes_pass.h"
#include "graph/passes/cond_remove_pass.h"
#include "graph/passes/constant_folding_pass.h"
#include "graph/passes/constant_fuse_same_pass.h"
#include "graph/passes/control_trigger_pass.h"
#include "graph/passes/ctrl_edge_transfer_pass.h"
#include "graph/passes/dimension_adjust_pass.h"
#include "graph/passes/dimension_compute_pass.h"
#include "graph/passes/flow_ctrl_pass.h"
#include "graph/passes/fuse_data_nodes_with_common_input_pass.h"
#include "graph/passes/identity_pass.h"
#include "graph/passes/input_output_connection_identify_pass.h"
#include "graph/passes/iterator_op_pass.h"
#include "graph/passes/link_gen_mask_nodes_pass.h"
#include "graph/passes/mark_graph_unknown_status_pass.h"
#include "graph/passes/merge_pass.h"
#include "graph/passes/merge_input_memcpy_pass.h"
#include "graph/passes/merge_to_stream_merge_pass.h"
#include "graph/passes/multi_batch_pass.h"
#include "graph/passes/next_iteration_pass.h"
#include "graph/passes/permute_pass.h"
#include "graph/passes/prune_pass.h"
#include "graph/passes/ref_identity_delete_op_pass.h"
#include "graph/passes/remove_same_const_pass.h"
#include "graph/passes/reshape_recovery_pass.h"
#include "graph/passes/reshape_remove_pass.h"
#include "graph/passes/same_transdata_breadth_fusion_pass.h"
#include "graph/passes/subgraph_pass.h"
#include "graph/passes/switch_data_edges_bypass.h"
#include "graph/passes/switch_dead_branch_elimination.h"
#include "graph/passes/switch_logic_remove_pass.h"
#include "graph/passes/switch_to_stream_switch_pass.h"
#include "graph/passes/transop_breadth_fusion_pass.h"
#include "graph/passes/transop_nearby_allreduce_fusion_pass.h"
#include "graph/passes/transop_symmetry_elimination_pass.h"
#include "graph/passes/transop_without_reshape_fusion_pass.h"
#include "graph/passes/transpose_transdata_pass.h"
#include "graph/passes/useless_control_out_remove_pass.h"
#include "graph/passes/variable_op_pass.h"
#include "graph/passes/variable_ref_delete_op_pass.h"
#include "graph/passes/variable_ref_useless_control_out_delete_pass.h"
#include "graph/passes/end_of_sequence_add_control_pass.h"
#include "graph/passes/subexpression_migration_pass.h"
#include "graph/passes/subgraph_const_migration_pass.h"
#include "graph/passes/unused_args_clean_pass.h"
#include "graph/passes/global_step_insert_pass.h"
#include "graph/passes/memcpy_addr_async_pass.h"
#include "graph/passes/hccl_continuous_memcpy_pass.h"
#include "graph/build/label_allocator.h"
#include "graph/utils/tensor_adapter.h"
#include "inc/pass_manager.h"
#include "ir_build/option_utils.h"
#include "graph/common/local_context.h"
#include "graph/common/omg_util.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "register/custom_pass_helper.h"
#include "graph/ops_stub.h"

using namespace std;
using namespace testing;
using namespace ge;
using namespace domi;

namespace {
const uint32_t kNotAdded = 0;
const uint32_t kStartAdd = 1;
const uint32_t kDoneAdded = 2;
}
class UtestGraphManagerTest : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

void CreateGraph(Graph &graph) {
  TensorDesc desc(ge::Shape({1, 3, 224, 224}));
  uint32_t size = desc.GetShape().GetShapeSize();
  desc.SetSize(size);
  auto data = op::Data("Data").set_attr_index(0);
  data.update_input_desc_data(desc);
  data.update_output_desc_out(desc);

  auto flatten = op::Flatten("Flatten").set_input_x(data, data.name_out_out());

  std::vector<Operator> inputs{data};
  std::vector<Operator> outputs{flatten};
  std::vector<Operator> targets{flatten};
  // Graph graph("test_graph");
  graph.SetInputs(inputs).SetOutputs(outputs).SetTargets(targets);
}

TEST_F(UtestGraphManagerTest, set_and_get_add_graph_flag) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.SetAddGraphCondition(graph_id, 1);
  uint32_t res = graph_manager.GetAddGraphCondition(graph_id);
  EXPECT_EQ(res, 1);
}

TEST_F(UtestGraphManagerTest, test_add_graph_1) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  // create graph
  Graph graph("test_graph");
  CreateGraph(graph);

  std::map<std::string, std::string> options;
  OmgContext context;
  Status status = graph_manager.AddGraph(graph_id, graph, options, context);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_add_graph_2) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_manager.SetAddGraphCondition(graph_id, kDoneAdded);
  Graph graph("test_graph");
  CreateGraph(graph);
  std::map<std::string, std::string> options;
  OmgContext context;
  Status status = graph_manager.AddGraph(graph_id, graph, options, context);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_add_graph_3) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  Graph graph("test_graph");
  CreateGraph(graph);

  std::map<std::string, std::string> options;
  OmgContext context;

  std::future<Status> fut1 = std::async(std::launch::async,
      &GraphManager::AddGraph, &graph_manager, graph_id, graph, options, context);
  std::future<Status> fut2 = std::async(std::launch::async,
      &GraphManager::AddGraph, &graph_manager, graph_id, graph, options, context);
  fut1.wait();
  fut2.wait();
  Status status1 = fut1.get();
  Status status2 = fut2.get();
  EXPECT_EQ(status1, ge::SUCCESS);
  EXPECT_EQ(status2, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_remove_graph_1) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  Status status = graph_manager.RemoveGraph(graph_id);
  EXPECT_EQ(status, ge::GE_GRAPH_GRAPH_NOT_EXIST);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetRunFlag(true);
  status = graph_manager.RemoveGraph(graph_id);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_remove_graph_2) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  Graph graph("test_graph");
  CreateGraph(graph);
  auto compute_graph = GraphUtils::GetComputeGraph(graph);
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(compute_graph);
  auto model_manager = ModelManager::GetInstance();
  auto listener = MakeShared<RunAsyncListener>();
  shared_ptr<DavinciModel> davinci_model1 = MakeShared<DavinciModel>(1, listener);
  davinci_model1->SetId(1);
  shared_ptr<DavinciModel> davinci_model2 = MakeShared<DavinciModel>(2, listener);
  davinci_model1->SetId(2);
  model_manager->InsertModel(1, davinci_model1);
  model_manager->InsertModel(2, davinci_model2);
  ge_root_model->SetModelId(1);
  ge_root_model->SetModelId(2);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_manager.AddGraphNode(graph_id, graph_node);
  Status status = graph_manager.RemoveGraph(graph_id);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_pre_run_thread) {
  
  GraphManager graph_manager;
  graph_manager.thread_run_flag_ = true;

  GraphId graph_id = 1;
  std::vector<ge::InputTensorInfo> input_tensor;
  uint64_t session_id = 0;
  ErrorMessage::Context error_context;
  GEThreadLocalContext context;
  RunAsyncCallback callback;
  // PreRunArgs args{graph_id, input_tensor, session_id, error_context, context, callback};
  bool ret = graph_manager.prerun_args_q_.Push({graph_id, input_tensor, session_id, error_context, context, callback});
  EXPECT_EQ(ret, true);

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_manager.PreRunThread(&graph_manager);
  // end with failed
}

TEST_F(UtestGraphManagerTest, test_pre_run_thread_2) {
  
  GraphManager graph_manager;
  graph_manager.thread_run_flag_ = true;

  GraphId graph_id = 1;
  GraphNodePtr graph_node_1 = MakeShared<ge::GraphNode>(graph_id);
  graph_manager.AddGraphNode(graph_id, graph_node_1);
  graph_manager.IncreaseGraphCount(graph_id);
  graph_manager.IncreaseGraphCount(graph_id);
  graph_node_1->SetBuildFlag(true);
  std::vector<ge::InputTensorInfo> input_tensor;
  uint64_t session_id = 0;
  ErrorMessage::Context error_context;
  GEThreadLocalContext context;
  RunAsyncCallback callback;
  // PreRunArgs args{graph_id, input_tensor, session_id, error_context, context, callback};
  bool ret = graph_manager.prerun_args_q_.Push({graph_id, input_tensor, session_id, error_context, context, callback});
  EXPECT_EQ(ret, true);
  graph_id = 2;
  GraphNodePtr graph_node_2 = MakeShared<ge::GraphNode>(graph_id);
  graph_manager.AddGraphNode(graph_id, graph_node_2);
  ret = graph_manager.prerun_args_q_.Push({graph_id, input_tensor, session_id, error_context, context, callback});
  EXPECT_EQ(ret, true);
  graph_manager.PreRunThread(&graph_manager);
  // end with failed
}

TEST_F(UtestGraphManagerTest, test_check_and_release_memory) {
  
  GraphManager graph_manager;
  GeModelPtr ge_model = make_shared<GeModel>();
  int64_t memory_size = 25 * 1024UL * 1024UL * 1024UL;
  int64_t weight_size = 25 * 1024UL * 1024UL * 1024UL;
  uint64_t session_id = 0;
  ge::AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, memory_size);
  ge::AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size);
  ge::AttrUtils::SetInt(ge_model, MODEL_ATTR_SESSION_ID, session_id);

  
  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_manager.IncreaseGraphCount(graph_id);
  graph_manager.IncreaseGraphCount(graph_id);

  auto model_manager = ModelManager::GetInstance();
  auto listener = MakeShared<RunAsyncListener>();
  shared_ptr<DavinciModel> davinci_model1 = MakeShared<DavinciModel>(1, listener);
  davinci_model1->SetId(1);
  shared_ptr<DavinciModel> davinci_model2 = MakeShared<DavinciModel>(2, listener);
  davinci_model1->SetId(2);
  model_manager->InsertModel(1, davinci_model1);
  model_manager->InsertModel(2, davinci_model2);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  bool is_dynamic_shape = false;
  (void)AttrUtils::GetBool(compute_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape);
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(compute_graph);
  ge_root_model->SetModelId(1);
  ge_root_model->SetModelId(2);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  Status status = graph_manager.CheckAndReleaseMemory(ge_model, graph_node);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_check_incre_build_and_pre_run_1) {
  // no need to build
  GraphId graph_id = 1;
  GraphManager graph_manager;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(compute_graph);
  GraphManager::PreRunArgs arg;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetBuildFlag(true);
  Status status = graph_manager.CheckIncreBuildAndPreRun(&graph_manager, arg, graph_node, ge_root_model);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_check_incre_build_and_pre_run_2) {
  // need build while buildflag is true, var format changed
  GraphId graph_id = 1;
  GraphManager graph_manager;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(compute_graph);
  GraphManager::PreRunArgs arg;
  arg.callback = [](Status, std::vector<ge::OutputTensorInfo> &) {};
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetBuildFlag(true);
  graph_node->Lock();
  graph_manager.var_acc_ctrl_.graph_ids_need_rebuild_.insert(graph_id);
  Status status = graph_manager.CheckIncreBuildAndPreRun(&graph_manager, arg, graph_node, ge_root_model);
  EXPECT_EQ(status, ge::PARAM_INVALID);
}

TEST_F(UtestGraphManagerTest, test_check_incre_build_and_pre_run_3) {
  // need build while buildflag is false, var format unchanged
  GraphId graph_id = 1;
  GraphManager graph_manager;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(compute_graph);
  GraphManager::PreRunArgs arg;
  arg.callback = [](Status, std::vector<ge::OutputTensorInfo> &) {};
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetBuildFlag(false);
  graph_node->Lock();
  Status status = graph_manager.CheckIncreBuildAndPreRun(&graph_manager, arg, graph_node, ge_root_model);
  EXPECT_NE(status, ge::SUCCESS);
}
