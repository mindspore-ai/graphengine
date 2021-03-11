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
#include <gmock/gmock.h>

#define private public
#define protected public
#include "graph/utils/graph_utils.h"
#include "common/profiling/profiling_manager.h"
#include "graph/load/model_manager/davinci_model.h"

using namespace std;

namespace ge {
extern OpDescPtr CreateOpDesc(string name, string type);

class DModelListener : public ModelListener {
 public:
  DModelListener(){};
  uint32_t OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t result, vector<OutputTensorInfo> &outputs) {
    return 0;
  }
};

shared_ptr<ModelListener> g_local_call_back(new DModelListener());

class UtestDavinciModel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

int32_t MsprofReport(uint32_t moduleId, uint32_t type, void *data, uint32_t len) {
  return 0;
}

TEST_F(UtestDavinciModel, init_success) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");
  ProfilingManager::Instance().is_load_profiling_ = true;

  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetGraph(GraphUtils::CreateGraphFromComputeGraph(graph));
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({1024});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }

  {
    OpDescPtr op_desc = CreateOpDesc("square", "Square");
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({1024});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 1

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(RT_MODEL_TASK_KERNEL);
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_op_index(op_desc->GetId());
    context->set_kernel_type(2);    // ccKernelType::TE
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({5120});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 2

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(RT_MODEL_TASK_MEMCPY_ASYNC);
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(5120);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({5120});
    op_desc->SetSrcName( { "memcpy" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  EXPECT_EQ(model.Assign(ge_model), SUCCESS);
  EXPECT_EQ(model.Init(), SUCCESS);

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 1);
  EXPECT_EQ(model.task_list_.size(), 2);

  OutputData output_data;
  vector<OutputTensorInfo> outputs;
  EXPECT_EQ(model.GenOutputTensorInfo(&output_data, outputs), SUCCESS);
  EXPECT_EQ(output_data.blobs.size(), 1);
  EXPECT_EQ(outputs.size(), 1);

  ProfilingManager::Instance().is_load_profiling_ = false;
}

TEST_F(UtestDavinciModel, init_data_op) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = make_shared<GeModel>();
  model.runtime_param_.mem_base = (uint8_t *)0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_input = CreateOpDesc("data", DATA);
  op_input->AddInputDesc(tensor);
  op_input->AddOutputDesc(tensor);
  op_input->SetInputOffset({1024});
  op_input->SetOutputOffset({1024});
  NodePtr node_input = graph->AddNode(op_input);

  OpDescPtr op_output = CreateOpDesc("output", NETOUTPUT);
  op_output->AddInputDesc(tensor);
  op_output->SetInputOffset({1024});
  op_output->SetSrcName( { "data" } );
  op_output->SetSrcIndex( { 0 } );
  NodePtr node_output = graph->AddNode(op_output);

  EXPECT_EQ(model.InitNodes(graph), SUCCESS);

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 1);
  EXPECT_EQ(model.op_list_.size(), 2);
}

TEST_F(UtestDavinciModel, init_data_op_subgraph) {
  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_base = (uint8_t *)0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_input = CreateOpDesc("data", DATA);
  op_input->AddInputDesc(tensor);
  op_input->AddOutputDesc(tensor);
  op_input->SetInputOffset({1024});
  op_input->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_input);

  uint32_t data_op_index = 0;
  map<uint32_t, OpDescPtr> data_by_index;
  set<const void *> input_outside_addrs;
  EXPECT_EQ(model.InitDataOp(nullptr, node, data_op_index, data_by_index, input_outside_addrs), SUCCESS);

  EXPECT_EQ(model.input_addrs_list_.size(), 0);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_EQ(data_op_index, 0);
  EXPECT_TRUE(data_by_index.empty());
}

TEST_F(UtestDavinciModel, init_netoutput_op_subgraph) {
  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_base = (uint8_t *)0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_output = CreateOpDesc("output", NETOUTPUT);
  op_output->AddInputDesc(tensor);
  op_output->SetInputOffset({1024});
  op_output->SetSrcName( { "data" } );
  op_output->SetSrcIndex( { 0 } );
  NodePtr node = graph->AddNode(op_output);

  std::vector<OpDescPtr> output_op_list;
  set<const void *> output_outside_addrs;
  EXPECT_EQ(model.InitNetOutput(nullptr, node, output_op_list, output_outside_addrs), SUCCESS);

  EXPECT_EQ(model.input_addrs_list_.size(), 0);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_TRUE(output_op_list.empty());
}

TEST_F(UtestDavinciModel, init_unknown) {
  DavinciModel model(0, nullptr);
  model.SetKnownNode(true);
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetGraph(GraphUtils::CreateGraphFromComputeGraph(graph));
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 5120000);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_input = CreateOpDesc("data", DATA);
  op_input->AddInputDesc(tensor);
  op_input->AddOutputDesc(tensor);
  op_input->SetInputOffset({1024});
  op_input->SetOutputOffset({1024});
  NodePtr node_input = graph->AddNode(op_input);    // op_index = 0

  OpDescPtr op_kernel = CreateOpDesc("square", "Square");
  op_kernel->AddInputDesc(tensor);
  op_kernel->AddOutputDesc(tensor);
  op_kernel->SetInputOffset({1024});
  op_kernel->SetOutputOffset({1024});
  NodePtr node_kernel = graph->AddNode(op_kernel);  // op_index = 1

  OpDescPtr op_memcpy = CreateOpDesc("memcpy", MEMCPYASYNC);
  op_memcpy->AddInputDesc(tensor);
  op_memcpy->AddOutputDesc(tensor);
  op_memcpy->SetInputOffset({1024});
  op_memcpy->SetOutputOffset({5120});
  NodePtr node_memcpy = graph->AddNode(op_memcpy);  // op_index = 2

  OpDescPtr op_output = CreateOpDesc("output", NETOUTPUT);
  op_output->AddInputDesc(tensor);
  op_output->SetInputOffset({5120});
  op_output->SetSrcName( { "memcpy" } );
  op_output->SetSrcIndex( { 0 } );
  NodePtr node_output = graph->AddNode(op_output);  // op_index = 3


  domi::TaskDef *task_def1 = model_task_def->add_task();
  task_def1->set_stream_id(0);
  task_def1->set_type(RT_MODEL_TASK_KERNEL);
  domi::KernelDef *kernel_def = task_def1->mutable_kernel();
  kernel_def->set_stub_func("stub_func");
  kernel_def->set_args_size(64);
  string args(64, '1');
  kernel_def->set_args(args.data(), 64);
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_op_index(1);
  context->set_kernel_type(2);    // ccKernelType::TE
  uint16_t args_offset[9] = {0};
  context->set_args_offset(args_offset, 9 * sizeof(uint16_t));

  domi::TaskDef *task_def2 = model_task_def->add_task();
  task_def2->set_stream_id(0);
  task_def2->set_type(RT_MODEL_TASK_MEMCPY_ASYNC);
  domi::MemcpyAsyncDef *memcpy_async = task_def2->mutable_memcpy_async();
  memcpy_async->set_src(1024);
  memcpy_async->set_dst(5120);
  memcpy_async->set_dst_max(512);
  memcpy_async->set_count(1);
  memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
  memcpy_async->set_op_index(2);

  EXPECT_EQ(model.Assign(ge_model), SUCCESS);
  ProfilingManager::Instance().is_load_profiling_ = true;
  EXPECT_EQ(model.Init(), SUCCESS);
  ProfilingManager::Instance().is_load_profiling_ = false;

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 1);
  EXPECT_EQ(model.task_list_.size(), 2);

  EXPECT_EQ(model.task_list_[0]->UpdateArgs(), SUCCESS);
  EXPECT_EQ(model.task_list_[1]->UpdateArgs(), SUCCESS);

  vector<string> out_shape_info;
  model.GetModelAttr(out_shape_info);

  vector<InputOutputDescInfo> input_descs;
  vector<InputOutputDescInfo> output_descs;
  EXPECT_EQ(model.GetInputOutputDescInfo(input_descs, output_descs), SUCCESS);

  int32_t virtual_addr = 0;
  const vector<void *> inputs = { &virtual_addr };
  const vector<void *> outputs = { &virtual_addr  };
  EXPECT_EQ(model.UpdateKnownNodeArgs(inputs, outputs), SUCCESS);
}

TEST_F(UtestDavinciModel, Init_variable_op) {
  DavinciModel model(0, g_local_call_back);
  model.ge_model_ = make_shared<GeModel>();
  model.runtime_param_.mem_base = (uint8_t *)0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr var1 = CreateOpDesc("var1", VARIABLE);
  var1->AddInputDesc(tensor);
  var1->AddOutputDesc(tensor);
  var1->SetInputOffset({1024});
  var1->SetOutputOffset({1024});
  AttrUtils::SetBool(var1, VAR_ATTR_VAR_IS_BROADCAST, true);
  graph->AddNode(var1);

  OpDescPtr var2 = CreateOpDesc(NODE_NAME_GLOBAL_STEP, VARIABLE);
  var2->AddInputDesc(tensor);
  var2->AddOutputDesc(tensor);
  var2->SetInputOffset({1024});
  var2->SetOutputOffset({1024});
  graph->AddNode(var2);

  EXPECT_EQ(model.InitNodes(graph), SUCCESS);

  EXPECT_EQ(model.ReturnNoOutput(1), SUCCESS);
  EXPECT_EQ(model.SyncVarData(), SUCCESS);

  OutputData output_data;
  EXPECT_FALSE(model.has_output_node_);
  EXPECT_EQ(model.CopyOutputData(1, output_data, RT_MEMCPY_DEVICE_TO_HOST), SUCCESS);

  EXPECT_EQ(model.ReturnResult(1, false, true, &output_data), INTERNAL_ERROR);
}

TEST_F(UtestDavinciModel, InitRealSizeAndShapeInfo_succ1) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = make_shared<GeModel>();
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  OpDescPtr op_output = CreateOpDesc("output_ascend_mbatch_batch_1", NETOUTPUT);
  op_output->AddInputDesc(tensor);
  op_output->SetInputOffset({1024});
  NodePtr node_output = graph->AddNode(op_output);
  EXPECT_EQ(model.InitRealSizeAndShapeInfo(graph, node_output), SUCCESS);
}

TEST_F(UtestDavinciModel, InitRealSizeAndShapeInfo_succ2) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr data1 = CreateOpDesc("data1", DATA);
  GeTensorDesc shape_desc(GeShape({4,3,224,224}), FORMAT_NCHW, DT_FLOAT);
  data1->AddInputDesc(shape_desc);
  data1->AddOutputDesc(shape_desc);
  NodePtr data1_node = graph->AddNode(data1);

  OpDescPtr case_node = CreateOpDesc("case1", CASE);
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  case_node->AddInputDesc(tensor);
  case_node->AddOutputDesc(tensor);
  NodePtr case1_node = graph->AddNode(case_node);

  OpDescPtr output = CreateOpDesc("output1", NETOUTPUT);
  output->AddInputDesc(tensor);
  output->SetSrcName( { "case1" } );
  output->SetSrcIndex( { 0 } );
  NodePtr output_node = graph->AddNode(output);

  GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), case1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(case1_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));

  (void)AttrUtils::SetStr(output_node->GetOpDesc(), ATTR_ALL_GEARS_INFO, "1;2;4;8");
  (void)AttrUtils::SetBool(case_node, ATTR_INSERT_BY_MBATCH, true);

  model.is_getnext_sink_dynamic_ = false;
  model.is_online_infer_dynamic_ = true;
  auto ret = model.InitRealSizeAndShapeInfo(graph, output_node);
  // GetGearAndRealOutShapeInfo without ATTR_NAME_DYNAMIC_OUTPUT_DIMS
  EXPECT_EQ(ret, SUCCESS);
  vector<string> dynamic_output_dims = {"0,0,1,1,0,2,2,0,4,3,0,8"};
  (void)AttrUtils::SetListStr(output_node->GetOpDesc(), ATTR_NAME_DYNAMIC_OUTPUT_DIMS, dynamic_output_dims);
  ret = model.InitRealSizeAndShapeInfo(graph, output_node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestDavinciModel, InitRealSizeAndShapeInfo_succ3) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  OpDescPtr data1 = CreateOpDesc("data1", DATA);
  GeTensorDesc shape_desc(GeShape({4,3,224,224}), FORMAT_NCHW, DT_FLOAT);
  data1->AddInputDesc(shape_desc);
  data1->AddOutputDesc(shape_desc);
  NodePtr data1_node = graph->AddNode(data1);

  OpDescPtr shape_node = CreateOpDesc("ascend_mbatch_get_dynamic_dims_node", GETDYNAMICDIMS);
  GeTensorDesc in_tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc out_tensor(GeShape({4,3}), FORMAT_NCHW, DT_FLOAT);
  shape_node->AddInputDesc(in_tensor);
  shape_node->AddOutputDesc(out_tensor);
  NodePtr get_dynamic_dims_node = graph->AddNode(shape_node);

  OpDescPtr output = CreateOpDesc("output1", NETOUTPUT);
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  output->AddInputDesc(tensor);
  output->SetSrcName( { "data1", "ascend_mbatch_get_dynamic_dims_node" } );
  output->SetSrcIndex( { 0, 1 } );
  NodePtr output_node = graph->AddNode(output);
  GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(get_dynamic_dims_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(1));

  (void)AttrUtils::SetStr(output_node->GetOpDesc(), ATTR_ALL_GEARS_INFO, "1,3;;4,3;,3");

  model.is_getnext_sink_dynamic_ = true;
  model.is_online_infer_dynamic_ = false;
  auto ret = model.InitRealSizeAndShapeInfo(graph, output_node);
  EXPECT_EQ(ret, SUCCESS);
  model.runtime_param_.mem_base = (uint8_t *)0x08000000;
  model.runtime_param_.mem_size = 4;
  ret = model.InitRealSizeAndShapeInfo(graph, output_node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestDavinciModel, init_data_aipp_info) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = make_shared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = (uint8_t *)0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_desc);

  GeAttrValue::NAMED_ATTRS aipp_attr;
  aipp_attr.SetAttr("aipp_mode", GeAttrValue::CreateFrom<GeAttrValue::INT>(domi::AippOpParams::dynamic));
  aipp_attr.SetAttr("related_input_rank", GeAttrValue::CreateFrom<GeAttrValue::INT>(0));
  aipp_attr.SetAttr("max_src_image_size", GeAttrValue::CreateFrom<GeAttrValue::INT>(2048));
  aipp_attr.SetAttr("support_rotation", GeAttrValue::CreateFrom<GeAttrValue::INT>(1));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(op_desc, ATTR_NAME_AIPP, aipp_attr));

  AippConfigInfo aipp_info;
  EXPECT_EQ(model.GetAippInfo(0, aipp_info), ACL_ERROR_GE_AIPP_NOT_EXIST);
  EXPECT_EQ(model.InitNodes(graph), SUCCESS);
  EXPECT_EQ(model.GetAippInfo(0, aipp_info), SUCCESS);
  EXPECT_EQ(aipp_info.aipp_mode, domi::AippOpParams::dynamic);

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_EQ(model.op_list_.size(), 1);
}

TEST_F(UtestDavinciModel, init_data_aipp_static) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = make_shared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = (uint8_t *)0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_desc);

  AttrUtils::SetStr(op_desc, ATTR_DATA_RELATED_AIPP_MODE, "static_aipp");

  InputAippType aipp_type;
  size_t aipp_index = 0;
  EXPECT_EQ(model.GetAippType(0, aipp_type, aipp_index), PARAM_INVALID);
  EXPECT_EQ(model.InitNodes(graph), SUCCESS);
  EXPECT_EQ(model.GetAippType(0, aipp_type, aipp_index), SUCCESS);
  EXPECT_EQ(aipp_type, DATA_WITH_STATIC_AIPP);
  EXPECT_EQ(aipp_index, 0xFFFFFFFFu);

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_EQ(model.op_list_.size(), 1);
}

TEST_F(UtestDavinciModel, init_data_aipp_dynamic) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = make_shared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = (uint8_t *)0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_desc);   // op_index 0
  AttrUtils::SetStr(op_desc, ATTR_DATA_RELATED_AIPP_MODE, "dynamic_aipp");
  AttrUtils::SetStr(op_desc, ATTR_DATA_AIPP_DATA_NAME_MAP, "releated_aipp");

  InputAippType aipp_type;
  size_t aipp_index = 0;
  EXPECT_EQ(model.GetAippType(0, aipp_type, aipp_index), PARAM_INVALID);
  EXPECT_EQ(model.InitNodes(graph), SUCCESS);
  EXPECT_EQ(model.GetAippType(0, aipp_type, aipp_index), SUCCESS);

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_EQ(model.op_list_.size(), 1);
}

TEST_F(UtestDavinciModel, init_data_aipp_releated) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = make_shared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = (uint8_t *)0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({1024});
    NodePtr node = graph->AddNode(op_desc);   // op_index 0
    AttrUtils::SetStr(op_desc, ATTR_DATA_RELATED_AIPP_MODE, "dynamic_aipp");
    AttrUtils::SetStr(op_desc, ATTR_DATA_AIPP_DATA_NAME_MAP, "releated_aipp");
  }
  {
    OpDescPtr op_desc = CreateOpDesc("releated_aipp", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({1024});
    NodePtr node = graph->AddNode(op_desc);   // op_index 1
  }

  InputAippType aipp_type;
  size_t aipp_index = 0;
  EXPECT_EQ(model.GetAippType(0, aipp_type, aipp_index), PARAM_INVALID);
  EXPECT_EQ(model.InitNodes(graph), SUCCESS);
  EXPECT_EQ(model.GetAippType(0, aipp_type, aipp_index), SUCCESS);
  EXPECT_EQ(aipp_type, DATA_WITH_DYNAMIC_AIPP);
  EXPECT_EQ(aipp_index, 1);

  EXPECT_EQ(model.input_addrs_list_.size(), 2);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_EQ(model.op_list_.size(), 2);
}

TEST_F(UtestDavinciModel, init_data_aipp_dynamic_conf) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = make_shared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = (uint8_t *)0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_desc);   // op_index 0
  AttrUtils::SetStr(op_desc, ATTR_DATA_RELATED_AIPP_MODE, "dynamic_aipp_conf");

  InputAippType aipp_type;
  size_t aipp_index = 0;
  EXPECT_EQ(model.GetAippType(0, aipp_type, aipp_index), PARAM_INVALID);
  EXPECT_EQ(model.InitNodes(graph), SUCCESS);
  EXPECT_EQ(model.GetAippType(0, aipp_type, aipp_index), SUCCESS);
  EXPECT_EQ(aipp_type, DYNAMIC_AIPP_NODE);
  EXPECT_EQ(aipp_index, 0xFFFFFFFFU);

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_EQ(model.op_list_.size(), 1);
}

TEST_F(UtestDavinciModel, init_data_aipp_dynamic_invalid) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = make_shared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = (uint8_t *)0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_desc);   // op_index 0
  AttrUtils::SetStr(op_desc, ATTR_DATA_RELATED_AIPP_MODE, "dynamic_aipp_invalid");

  InputAippType aipp_type;
  size_t aipp_index = 0;
  EXPECT_EQ(model.GetAippType(0, aipp_type, aipp_index), PARAM_INVALID);
  EXPECT_EQ(model.InitNodes(graph), ACL_ERROR_GE_AIPP_MODE_INVALID);

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_EQ(model.op_list_.size(), 1);
}

TEST_F(UtestDavinciModel, init_data_aipp_input_info_empty) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = make_shared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = (uint8_t *)0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_desc);   // op_index 0

  vector<string> inputs = {};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_AIPP_INPUTS, inputs);
  vector<string> outputs = {};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_AIPP_OUTPUTS, outputs);

  OriginInputInfo orig_input_info;
  EXPECT_EQ(model.GetOrigInputInfo(0, orig_input_info), ACL_ERROR_GE_AIPP_NOT_EXIST);
  EXPECT_EQ(model.InitNodes(graph), SUCCESS);
  EXPECT_EQ(model.GetOrigInputInfo(0, orig_input_info), SUCCESS);

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_EQ(model.op_list_.size(), 1);
}

TEST_F(UtestDavinciModel, init_data_aipp_input_info_normal) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = make_shared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = (uint8_t *)0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_desc);   // op_index 0

  vector<string> inputs = { "NCHW:DT_FLOAT:TensorName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(op_desc, ATTR_NAME_AIPP_INPUTS, inputs);
  vector<string> outputs = { "NCHW:DT_FLOAT:TensorName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(op_desc, ATTR_NAME_AIPP_OUTPUTS, outputs);

  OriginInputInfo orig_input_info;
  EXPECT_EQ(model.GetOrigInputInfo(0, orig_input_info), ACL_ERROR_GE_AIPP_NOT_EXIST);
  EXPECT_EQ(model.InitNodes(graph), SUCCESS);
  EXPECT_EQ(model.GetOrigInputInfo(0, orig_input_info), SUCCESS);

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_EQ(model.op_list_.size(), 1);
}

TEST_F(UtestDavinciModel, init_data_aipp_input_info_invalid) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = make_shared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = (uint8_t *)0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_desc);   // op_index 0

  vector<string> inputs = { "NCHW:DT_FLOAT:TensorName" };     // Invalid
  AttrUtils::SetListStr(op_desc, ATTR_NAME_AIPP_INPUTS, inputs);
  vector<string> outputs = { "NCHW:DT_FLOAT:TensorName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(op_desc, ATTR_NAME_AIPP_OUTPUTS, outputs);

  OriginInputInfo orig_input_info;
  EXPECT_EQ(model.GetOrigInputInfo(0, orig_input_info), ACL_ERROR_GE_AIPP_NOT_EXIST);
  EXPECT_EQ(model.InitNodes(graph), ACL_ERROR_GE_AIPP_MODE_INVALID);
  EXPECT_EQ(model.GetOrigInputInfo(0, orig_input_info), ACL_ERROR_GE_AIPP_NOT_EXIST);

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_EQ(model.op_list_.size(), 1);
}

TEST_F(UtestDavinciModel, init_data_aipp_input_dims_normal) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = make_shared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = (uint8_t *)0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_desc);   // op_index 0

  vector<string> inputs = { "NCHW:DT_FLOAT:TensorName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(op_desc, ATTR_NAME_AIPP_INPUTS, inputs);
  vector<string> outputs = { "NCHW:DT_FLOAT:TensorName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(op_desc, ATTR_NAME_AIPP_OUTPUTS, outputs);

  vector<InputOutputDims> input_dims;
  vector<InputOutputDims> output_dims;
  EXPECT_EQ(model.GetAllAippInputOutputDims(0, input_dims, output_dims), ACL_ERROR_GE_AIPP_NOT_EXIST);
  EXPECT_EQ(model.InitNodes(graph), SUCCESS);
  EXPECT_EQ(model.GetAllAippInputOutputDims(0, input_dims, output_dims), SUCCESS);
  EXPECT_EQ(input_dims.size(), 1);
  EXPECT_EQ(output_dims.size(), 1);

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_EQ(model.op_list_.size(), 1);
}

// test label_set_task Init
TEST_F(UtestDavinciModel, label_task_success) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetGraph(GraphUtils::CreateGraphFromComputeGraph(graph));
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT32);
  TensorUtils::SetSize(tensor, 64);

  {
    OpDescPtr op_desc = CreateOpDesc("label_switch", LABELSWITCHBYINDEX);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({1024});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 0
    EXPECT_TRUE(AttrUtils::SetListInt(op_desc, ATTR_NAME_LABEL_SWITCH_LIST, {0, 1}));

    domi::TaskDef *task_def1 = model_task_def->add_task();
    task_def1->set_stream_id(0);
    task_def1->set_type(RT_MODEL_TASK_STREAM_LABEL_SWITCH_BY_INDEX);
    domi::LabelSwitchByIndexDef *label_task_def = task_def1->mutable_label_switch_by_index();
    label_task_def->set_op_index(op_desc->GetId());
    label_task_def->set_label_max(2);
  }

  {
    OpDescPtr op_desc = CreateOpDesc("label_then", LABELSET);
    NodePtr node = graph->AddNode(op_desc);  // op_index = 1
    EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, 1));

    domi::TaskDef *task_def1 = model_task_def->add_task();
    task_def1->set_stream_id(0);
    task_def1->set_type(RT_MODEL_TASK_LABEL_SET);
    domi::LabelSetDef *label_task_def = task_def1->mutable_label_set();
    label_task_def->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("label_goto", LABELGOTOEX);
    NodePtr node = graph->AddNode(op_desc);      // op_index = 2
    EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, 2));

    domi::TaskDef *task_def2 = model_task_def->add_task();
    task_def2->set_stream_id(0);
    task_def2->set_type(RT_MODEL_TASK_STREAM_LABEL_GOTO);
    domi::LabelGotoExDef *label_task_def = task_def2->mutable_label_goto_ex();
    label_task_def->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("label_else", LABELSET);
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
    EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, 0));

    domi::TaskDef *task_def1 = model_task_def->add_task();
    task_def1->set_stream_id(0);
    task_def1->set_type(RT_MODEL_TASK_LABEL_SET);
    domi::LabelSetDef *label_task_def = task_def1->mutable_label_set();
    label_task_def->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("label_leave", LABELSET);
    NodePtr node = graph->AddNode(op_desc);  // op_index = 4
    EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, 2));

    domi::TaskDef *task_def1 = model_task_def->add_task();
    task_def1->set_stream_id(0);
    task_def1->set_type(RT_MODEL_TASK_LABEL_SET);
    domi::LabelSetDef *label_task_def = task_def1->mutable_label_set();
    label_task_def->set_op_index(op_desc->GetId());
  }

  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 3));
  EXPECT_EQ(model.Assign(ge_model), SUCCESS);
  EXPECT_EQ(model.Init(), SUCCESS);
  EXPECT_EQ(model.input_addrs_list_.size(), 0);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_EQ(model.task_list_.size(), 5);
}

TEST_F(UtestDavinciModel, LoadWithQueue_fail_with_diff_args) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = make_shared<GeModel>();
  model.input_queue_ids_.emplace_back(0);
  EXPECT_EQ(model.LoadWithQueue(), ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID);
  EXPECT_EQ(model.input_data_info_.size(), 0);
  ZeroCopyOffset zero_copy_offset;
  model.input_data_info_[0] = zero_copy_offset;
  model.output_queue_ids_.emplace_back(0);
  EXPECT_EQ(model.LoadWithQueue(), ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID);
  EXPECT_EQ(model.output_data_info_.size(), 0);
  model.output_data_info_[0] = zero_copy_offset;
  EXPECT_EQ(model.LoadWithQueue(), INTERNAL_ERROR);
  EXPECT_EQ(model.active_stream_list_.size(), 0);
}

TEST_F(UtestDavinciModel, Sink_model_profile) {
  ProfilingManager::Instance().prof_cb_.msprofReporterCallback = MsprofReport;
   ProfileInfo profile;
   profile.fusion_info.op_name = "relu";

  DavinciModel model(0, nullptr);
  model.profile_list_.emplace_back(profile);
  std::map<std::string, std::pair<uint32_t, uint32_t>> op_info;
  op_info["relu"] = std::pair<uint32_t, uint32_t>(1, 1);
  model.profiler_report_op_info_ = op_info;
  model.SinkModelProfile();
}

TEST_F(UtestDavinciModel, Sink_time_profile) {
  ProfilingManager::Instance().prof_cb_.msprofReporterCallback = MsprofReport;
  DavinciModel model(0, nullptr);
  InputData current_data;
  model.SinkTimeProfile(current_data);
}

class ClassTest {
public:
    virtual ~ClassTest() {}

    virtual int func0() {
        return 0;
    }
    virtual int func1(int a) {
        return a;
    }
    virtual int func2(int a, int b) {
        return a + b;
    }
    virtual int func3(int a, int b) const {
        return a - b;
    }
};

class MockTest : public ClassTest {
public:
    MOCK_METHOD0(func0, int());
    MOCK_METHOD1(func1, int(int a));
    MOCK_METHOD2(func2, int(int a, int b));

    MOCK_CONST_METHOD2(func3, int(int a, int b));
};

TEST_F(UtestDavinciModel, simple_test_gmock) {
    MockTest mock_stub;

    ON_CALL(mock_stub, func0()).WillByDefault(testing::Return(250));
    EXPECT_EQ(mock_stub.func0(), 250);
    EXPECT_EQ(mock_stub.func0(), 250);
    EXPECT_EQ(mock_stub.func0(), 250);

    EXPECT_CALL(mock_stub, func1(testing::_)).Times(2).WillOnce(testing::Return(1024)).WillOnce(testing::Return(250));
    EXPECT_EQ(mock_stub.func1(1), 1024);
    EXPECT_EQ(mock_stub.func1(1), 250);

    EXPECT_CALL(mock_stub, func2(testing::_, 5)).Times(3).WillRepeatedly(testing::Return(1023));
    EXPECT_EQ(mock_stub.func2(1, 5), 1023);
    EXPECT_EQ(mock_stub.func2(2, 5), 1023);
    EXPECT_EQ(mock_stub.func2(3, 5), 1023);
}
}  // namespace ge
