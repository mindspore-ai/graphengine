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
#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/types.h"

#define private public
#define protected public
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/model_serialize.h"
#include "graph/load/new_model_manager/davinci_model.h"
#include "graph/load/new_model_manager/model_output.h"
#include "common/properties_manager.h"
#include "common/op/ge_op_utils.h"
#include <cce/taskdown_api.h>
#include "runtime/dev.h"
#include "runtime/kernel.h"
#include "cce/fwk_adpt_struct.h"
#include "graph/load/new_model_manager/task_info/task_info_factory.h"
#include "graph/load/new_model_manager/task_info/task_info.h"
#include "graph/load/new_model_manager/task_info/stream_active_task_info.h"
#include "graph/load/new_model_manager/task_info/stream_switch_task_info.h"
#include "graph/load/new_model_manager/task_info/profiler_trace_task_info.h"
#include "graph/load/new_model_manager/task_info/memcpy_async_task_info.h"
#include "graph/load/new_model_manager/task_info/label_goto_task_info.h"
#include "graph/load/new_model_manager/task_info/label_set_task_info.h"
#include "graph/load/new_model_manager/task_info/kernel_ex_task_info.h"
#include "graph/load/new_model_manager/task_info/kernel_task_info.h"
#include "graph/load/new_model_manager/task_info/hccl_task_info.h"
#include "graph/load/new_model_manager/task_info/fusion_start_task_info.h"
#include "graph/load/new_model_manager/task_info/fusion_stop_task_info.h"
#include "graph/load/new_model_manager/task_info/event_record_task_info.h"
#include "graph/load/new_model_manager/task_info/event_wait_task_info.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/load/new_model_manager/model_manager.h"
#undef private
#undef protected

#include "new_op_test_utils.h"
#include "graph/debug/ge_attr_define.h"
using namespace std;
using namespace testing;
using domi::EventExDef;
using domi::KernelContext;
using domi::KernelDef;
using domi::LogTimeStampDef;
using domi::ModelTaskDef;
using domi::StreamActiveDef;
using domi::TaskDef;

namespace ge {
class UtestModelManagerDavinciModel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

class DModelListener : public ge::ModelListener {
 public:
  DModelListener(){};
  uint32_t OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t resultCode) {
    GELOGI("In Call back. OnComputeDone");
    return 0;
  }
};

shared_ptr<ge::ModelListener> g_label_call_back(new DModelListener());

static ge::OpDescPtr CreateOpDesc(string name = "", string type = "") {
  auto op_desc = std::make_shared<ge::OpDesc>(name, type);
  op_desc->SetStreamId(0);
  op_desc->SetId(0);

  ge::AttrUtils::SetFloat(op_desc, ge::ATTR_NAME_ALPHA, 0);
  ge::AttrUtils::SetFloat(op_desc, ge::ATTR_NAME_BETA, 0);

  op_desc->SetWorkspace({});
  ;
  op_desc->SetWorkspaceBytes({});
  op_desc->SetInputOffset({});
  op_desc->SetOutputOffset({});

  ge::AttrUtils::SetListStr(op_desc, ge::ATTR_NAME_WEIGHT_NAME, {});
  ge::AttrUtils::SetInt(op_desc, ge::POOLING_ATTR_MODE, 0);
  ge::AttrUtils::SetInt(op_desc, ge::POOLING_ATTR_PAD_MODE, 0);
  ge::AttrUtils::SetInt(op_desc, ge::POOLING_ATTR_DATA_MODE, 0);
  ge::AttrUtils::SetInt(op_desc, ge::POOLING_ATTR_CEIL_MODE, 0);
  ge::AttrUtils::SetInt(op_desc, ge::POOLING_ATTR_NAN_OPT, 0);
  ge::AttrUtils::SetListInt(op_desc, ge::POOLING_ATTR_WINDOW, {});
  ge::AttrUtils::SetListInt(op_desc, ge::POOLING_ATTR_PAD, {});
  ge::AttrUtils::SetListInt(op_desc, ge::POOLING_ATTR_STRIDE, {});
  ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_ACTIVE_STREAM_LIST, {1, 1});
  ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_STREAM_SWITCH_COND, 0);
  ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_FRAMEWORK_FWK_TYPE, FMK_TYPE_T);
  return op_desc;
}

// tset failed_rt_free_host
TEST_F(UtestModelManagerDavinciModel, failed_rt_free_host) {
  DavinciModel model(0, g_label_call_back);

  OutputData output_data;

  auto op_desc = CreateOpDesc("Pooling", "Pooling");
  op_desc->SetOutputOffset({1});
  op_desc->SetInputOffset({1});

  {
    ge::GeTensorDesc in_desc(ge::GeShape({1, 1, 1, 1}));
    ge::TensorUtils::SetSize(in_desc, 16);
    ge::TensorUtils::SetOutputTensor(in_desc, false);
    ge::TensorUtils::SetInputTensor(in_desc, true);
    op_desc->AddInputDesc(in_desc);
  }

  {
    ge::GeTensorDesc out_desc(ge::GeShape({1, 1, 1, 1}));
    ge::TensorUtils::SetSize(out_desc, 16);
    ge::TensorUtils::SetOutputTensor(out_desc, true);
    ge::TensorUtils::SetInputTensor(out_desc, false);
    op_desc->AddOutputDesc(out_desc);
  }
  ge::AttrUtils::SetInt(op_desc, ge::POOLING_ATTR_PAD_MODE, cce::CC_PADDING_DIRECTASSIGN);
  ge::AttrUtils::SetListInt(op_desc, ge::POOLING_ATTR_PAD, vector<int>({1, 1, 1, 1}));
  ge::AttrUtils::SetListInt(op_desc, ge::POOLING_ATTR_WINDOW, vector<int>({1, 1}));
  ge::AttrUtils::SetListInt(op_desc, ge::POOLING_ATTR_STRIDE, vector<int>({1, 1}));

  auto compute_graph = make_shared<ge::ComputeGraph>("g");
  auto node = compute_graph->AddNode(op_desc);

  OmeTestOpUtils::InitModel(model);

  model.data_op_list_.push_back(op_desc);

  EXPECT_EQ(ge::INTERNAL_ERROR, model.ReturnResult(1, false, false, &output_data));
}

// test modeldef_fail
TEST_F(UtestModelManagerDavinciModel, contruct_modeldef_createfail) {
  DavinciModel model(0, g_label_call_back);

  OmeTestOpUtils::InitModel(model);

  auto op_desc = CreateOpDesc("Pooling", "Pooling");
  op_desc->SetOutputOffset({1});
  op_desc->SetInputOffset({1});

  {
    ge::GeTensorDesc in_desc(ge::GeShape({1, 1, 1, 1}));
    ge::TensorUtils::SetSize(in_desc, 16);
    ge::TensorUtils::SetOutputTensor(in_desc, false);
    ge::TensorUtils::SetInputTensor(in_desc, true);
    op_desc->AddInputDesc(in_desc);
  }

  {
    ge::GeTensorDesc out_desc(ge::GeShape({1, 1, 1, 1}));
    ge::TensorUtils::SetSize(out_desc, 16);
    ge::TensorUtils::SetOutputTensor(out_desc, true);
    ge::TensorUtils::SetInputTensor(out_desc, false);
    op_desc->AddOutputDesc(out_desc);
  }
  ge::AttrUtils::SetInt(op_desc, ge::POOLING_ATTR_PAD_MODE, cce::CC_PADDING_DIRECTASSIGN);
  ge::AttrUtils::SetListInt(op_desc, ge::POOLING_ATTR_PAD, vector<int>({1, 1, 1, 1}));
  ge::AttrUtils::SetListInt(op_desc, ge::POOLING_ATTR_WINDOW, vector<int>({1, 1}));
  ge::AttrUtils::SetListInt(op_desc, ge::POOLING_ATTR_STRIDE, vector<int>({1, 1}));

  // EXPECT_EQ(ge::SUCCESS, model.Init());

  model.GetEventList();
}

// test CopyInputDataToModel
TEST_F(UtestModelManagerDavinciModel, copy_input_data_to_model_fail) {
  DavinciModel model(0, g_label_call_back);

  ge::InputData input_data;
  ge::DataBuffer data_buffer;
  data_buffer.data = new char[16];
  data_buffer.length = 16;
  input_data.index = 0;
  input_data.model_id = 1;
  input_data.blobs.push_back(data_buffer);

  model.op_list_.clear();
  //    EXPECT_EQ(ge::PARAM_INVALID, model.CopyInputDataToModel(input_data.blobs, 0));

  delete[](char *) data_buffer.data;
}

// test StreamNum
TEST_F(UtestModelManagerDavinciModel, streamnum_success) {
  DavinciModel *model = new DavinciModel(0, g_label_call_back);

  OmeTestOpUtils::InitModel(*model);
  // EXPECT_EQ(ge::SUCCESS, model->Init());

  EXPECT_EQ(0, model->StreamNum());
  EXPECT_EQ(ge::INTERNAL_ERROR, model->ModelRunStart());

  EXPECT_EQ(ge::SUCCESS, model->ModelRunStop());

  delete model;
}

// test EventNum
TEST_F(UtestModelManagerDavinciModel, eventnum_success) {
  DavinciModel *model = new DavinciModel(0, g_label_call_back);

  OmeTestOpUtils::InitModel(*model);

  // EXPECT_EQ(ge::SUCCESS, model->Init());

  EXPECT_EQ(0, model->EventNum());
  EXPECT_EQ(ge::INTERNAL_ERROR, model->ModelRunStart());

  EXPECT_EQ(ge::SUCCESS, model->ModelRunStop());

  delete model;
}

TEST_F(UtestModelManagerDavinciModel, handlelist_success) {
  DavinciModel *model = new DavinciModel(0, g_label_call_back);

  OmeTestOpUtils::InitModel(*model);

  // EXPECT_EQ(ge::SUCCESS, model->Init());

  EXPECT_EQ(ge::INTERNAL_ERROR, model->ModelRunStart());

  EXPECT_EQ(ge::SUCCESS, model->ModelRunStop());

  delete model;
}

// test GetEventList
TEST_F(UtestModelManagerDavinciModel, eventlist_success) {
  DavinciModel *model = new DavinciModel(0, g_label_call_back);

  OmeTestOpUtils::InitModel(*model);

  // EXPECT_EQ(ge::SUCCESS, model->Init());

  EXPECT_EQ(true, model->GetEventList().empty());
  EXPECT_EQ(ge::INTERNAL_ERROR, model->ModelRunStart());

  EXPECT_EQ(ge::SUCCESS, model->ModelRunStop());

  delete model;
}

// test rtMalloc
TEST_F(UtestModelManagerDavinciModel, failed_reset_device) {
  DavinciModel model(0, g_label_call_back);
  ge::OutputData output_data;
  ge::DataBuffer buf_data;
  rtMalloc(&buf_data.data, 128, RT_MEMORY_HBM);
  buf_data.length = 128;
  output_data.blobs.push_back(buf_data);
  EXPECT_EQ(ge::INTERNAL_ERROR, model.ReturnResult(1, true, false, &output_data));
  rtFree(buf_data.data);
}

// test priority
TEST_F(UtestModelManagerDavinciModel, init_not_support_priority) {
  int32_t priority = 8;
  DavinciModel model(priority, g_label_call_back);
  // EXPECT_EQ(ge::PARAM_INVALID, model.Init());
}

// test GetInputOutputDescInfo
TEST_F(UtestModelManagerDavinciModel, success_GetInputOutputDescInfo_without_netoutput) {
  DavinciModel model(0, g_label_call_back);

  auto op_desc = CreateOpDesc("Data", "Data");
  op_desc->SetOutputOffset({1});
  op_desc->SetInputOffset({1});
  op_desc->SetStreamId(0);

  {
    ge::GeTensorDesc in_desc(ge::GeShape({1, 1, 10, 10}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    ge::TensorUtils::SetOutputTensor(in_desc, false);
    ge::TensorUtils::SetInputTensor(in_desc, true);
    op_desc->AddInputDesc(in_desc);
  }

  {
    ge::GeTensorDesc out_desc(ge::GeShape({1, 1, 10, 10}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    ge::TensorUtils::SetOutputTensor(out_desc, true);
    ge::TensorUtils::SetInputTensor(out_desc, false);
    op_desc->AddOutputDesc(out_desc);
  }

  op_desc->SetSrcName({"Pooling1", "Pooling0"});
  op_desc->SetSrcIndex({0, 1});

  auto compute_graph = make_shared<ge::ComputeGraph>("g");
  auto node = compute_graph->AddNode(op_desc);

  model.data_op_list_.push_back(op_desc);
  model.output_data_info_[0] = {32, (void *)0x70002010};

  model.op_list_[0] = op_desc;

  model.output_op_list_.push_back(op_desc);

  vector<InputOutputDescInfo> input_shapes;
  vector<InputOutputDescInfo> output_shapes;
  EXPECT_EQ(ge::SUCCESS, model.GetInputOutputDescInfo(input_shapes, output_shapes));
}

TEST_F(UtestModelManagerDavinciModel, CopyTensorFromSrcVarNode_input_is_nullptr) {
  NodePtr src_node = nullptr;
  NodePtr dst_node = nullptr;
  DavinciModel model(0, g_label_call_back);
  Status ret = model.CopyTensorFromSrcVarNode(src_node, dst_node);
  EXPECT_EQ(FAILED, ret);
}

TEST_F(UtestModelManagerDavinciModel, CopyTensorFromSrcVarNode_success) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("Cast", "Cast");
  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc dims_tensor_desc_in(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  NodePtr src_node = graph->AddNode(op_desc_ptr);
  NodePtr dst_node = graph->AddNode(op_desc_ptr);
  DavinciModel model(0, g_label_call_back);
  Status ret = model.CopyTensorFromSrcVarNode(src_node, dst_node);
  // EXPECT_EQ(SUCCESS, ret);
}

TEST_F(UtestModelManagerDavinciModel, CopyVarData_graph_is_nullptr) {
  ge::ComputeGraphPtr graph = nullptr;
  DavinciModel model(0, g_label_call_back);
  Status ret = model.CopyVarData(graph);
  EXPECT_EQ(FAILED, ret);
}

TEST_F(UtestModelManagerDavinciModel, copy_var_data_success) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("Variable", "Variable");
  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc dims_tensor_desc_in(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT16);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  NodePtr src_node = graph->AddNode(op_desc_ptr);
  (void)ge::AttrUtils::SetStr(src_node->GetOpDesc(), "_copy_from_var_node", "abc");
  (void)ge::AttrUtils::SetBool(src_node->GetOpDesc(), "_copy_value", false);

  DavinciModel model(0, g_label_call_back);
  Status ret = model.CopyVarData(graph);
  // EXPECT_EQ(SUCCESS, ret);
}

TEST_F(UtestModelManagerDavinciModel, get_input_output_desc_info_without_data_op_list) {
  DavinciModel model(0, g_label_call_back);
  vector<InputOutputDescInfo> input_list;
  vector<InputOutputDescInfo> output_list;
  Status ret = model.GetInputOutputDescInfo(input_list, output_list);
  EXPECT_EQ(SUCCESS, ret);
}

// test GetInputOutputDescInfo
TEST_F(UtestModelManagerDavinciModel, success_get_input_output_descInfo_with_net_output) {
  DavinciModel model(0, g_label_call_back);

  auto op_desc = CreateOpDesc("Data", "Data");
  op_desc->SetOutputOffset({1});
  op_desc->SetInputOffset({1});
  op_desc->SetStreamId(0);

  {
    ge::GeTensorDesc in_desc(ge::GeShape({1, 1, 10, 10}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    ge::TensorUtils::SetOutputTensor(in_desc, false);
    ge::TensorUtils::SetInputTensor(in_desc, true);
    op_desc->AddInputDesc(in_desc);
  }

  {
    ge::GeTensorDesc out_desc(ge::GeShape({1, 1, 10, 10}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    ge::TensorUtils::SetOutputTensor(out_desc, true);
    ge::TensorUtils::SetOutputTensor(out_desc, true);
    ge::TensorUtils::SetInputTensor(out_desc, false);
    op_desc->AddOutputDesc(out_desc);
  }
  op_desc->SetSrcName({"Pooling1", "Pooling0"});
  op_desc->SetSrcIndex({0, 1});

  auto compute_graph = make_shared<ge::ComputeGraph>("g");
  auto data_node = compute_graph->AddNode(op_desc);

  model.data_op_list_.push_back(op_desc);

  op_desc->SetType("NetOutput");

  auto no_node = compute_graph->AddNode(op_desc);

  model.op_list_[0] = op_desc;

  model.output_op_list_.push_back(op_desc);
  model.output_data_info_[0] = {32, (void *)0x70002010};

  vector<InputOutputDescInfo> input_shapes;
  vector<InputOutputDescInfo> output_shapes;
  EXPECT_EQ(ge::SUCCESS, model.GetInputOutputDescInfo(input_shapes, output_shapes));
}

TEST_F(UtestModelManagerDavinciModel, success_get_input_output_desc_info_for_zero_copy_with_net_output) {
  DavinciModel model(0, g_label_call_back);

  auto op_desc = CreateOpDesc("Data", "Data");
  op_desc->SetOutputOffset({1});
  op_desc->SetInputOffset({1});
  op_desc->SetStreamId(0);

  {
    ge::GeTensorDesc in_desc(ge::GeShape({1, 1, 10, 10}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    ge::TensorUtils::SetOutputTensor(in_desc, false);
    ge::TensorUtils::SetInputTensor(in_desc, true);
    op_desc->AddInputDesc(in_desc);
  }

  {
    ge::GeTensorDesc out_desc(ge::GeShape({1, 1, 10, 10}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    ge::TensorUtils::SetOutputTensor(out_desc, true);
    ge::TensorUtils::SetOutputTensor(out_desc, true);
    ge::TensorUtils::SetInputTensor(out_desc, false);
    op_desc->AddOutputDesc(out_desc);
  }

  op_desc->SetSrcName({"Pooling1", "Pooling0"});
  op_desc->SetSrcIndex({0, 1});

  auto compute_graph = make_shared<ge::ComputeGraph>("g");
  auto data_node = compute_graph->AddNode(op_desc);

  model.data_op_list_.push_back(op_desc);

  op_desc->SetType("NetOutput");

  auto net_out_node = compute_graph->AddNode(op_desc);
  model.op_list_[0] = op_desc;

  model.output_op_list_.push_back(op_desc);
  model.output_data_info_[0] = {32, (void *)0x70002010};
  model.output_memory_size_list_.push_back(64);

  vector<InputOutputDescInfo> input_shapes;
  vector<InputOutputDescInfo> output_shapes;
  EXPECT_EQ(ge::SUCCESS, model.GetInputOutputDescInfoForZeroCopy(input_shapes, output_shapes));
}

TEST_F(UtestModelManagerDavinciModel, success_get_input_output_desc_info_dim_size_not4) {
  DavinciModel model(0, g_label_call_back);

  auto op_desc = CreateOpDesc("Data", "Data");
  op_desc->SetOutputOffset({1});
  op_desc->SetInputOffset({1});
  op_desc->SetStreamId(0);

  {
    ge::GeTensorDesc in_desc(ge::GeShape({1, 1, 10}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    ge::TensorUtils::SetOutputTensor(in_desc, false);
    ge::TensorUtils::SetInputTensor(in_desc, true);
    op_desc->AddInputDesc(in_desc);
  }

  {
    ge::GeTensorDesc out_desc(ge::GeShape({1, 1, 10}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    ge::TensorUtils::SetOutputTensor(out_desc, true);
    ge::TensorUtils::SetOutputTensor(out_desc, true);
    ge::TensorUtils::SetInputTensor(out_desc, false);
    op_desc->AddOutputDesc(out_desc);
  }

  op_desc->SetSrcName({"Pooling1", "Pooling0"});
  op_desc->SetSrcIndex({0, 1});

  auto compute_graph = make_shared<ge::ComputeGraph>("g");
  auto data_node = compute_graph->AddNode(op_desc);

  model.data_op_list_.push_back(op_desc);

  op_desc->SetType("NetOutput");

  auto net_out_node = compute_graph->AddNode(op_desc);
  model.op_list_[0] = op_desc;

  model.output_op_list_.push_back(op_desc);
  model.output_data_info_[0] = {32, (void *)0x70002010};

  vector<InputOutputDescInfo> input_shapes;
  vector<InputOutputDescInfo> output_shapes;
  EXPECT_EQ(ge::SUCCESS, model.GetInputOutputDescInfo(input_shapes, output_shapes));
}

// test GetLabelList
TEST_F(UtestModelManagerDavinciModel, get_label_list_success) {
  DavinciModel model(0, g_label_call_back);
  OmeTestOpUtils::InitModel(model);
  vector<rtLabel_t> label_list;
  model.label_list_ = label_list;
  EXPECT_EQ(label_list, model.GetLabelList());
}

// test GetInputListSize
TEST_F(UtestModelManagerDavinciModel, get_label_list_size_success) {
  DavinciModel model(0, g_label_call_back);
  OmeTestOpUtils::InitModel(model);
  vector<OpDescPtr> data_op_list;
  data_op_list.push_back(std::make_shared<OpDesc>());
  model.data_op_list_ = data_op_list;
}

// test GetFlowctrlOpList
TEST_F(UtestModelManagerDavinciModel, get_flow_ctrl_op_list_success) {
  DavinciModel model(0, g_label_call_back);
  OmeTestOpUtils::InitModel(model);
  std::map<uint32_t, uint32_t> flowctrl_op_index_internal_map;
  flowctrl_op_index_internal_map.insert(pair<uint32_t, uint32_t>(1, 1));
  model.flowctrl_op_index_internal_map_ = flowctrl_op_index_internal_map;
  // EXPECT_EQ(flowctrl_op_index_internal_map_, model.GetFlowctrlOpList());
}

// test SetFlowctrlOpList
TEST_F(UtestModelManagerDavinciModel, get_flow_ctrl_index_success) {
  DavinciModel model(0, g_label_call_back);
  OmeTestOpUtils::InitModel(model);
  EXPECT_EQ(0, model.GetFlowctrlIndex(0));
  EXPECT_EQ(1, model.GetFlowctrlIndex(0));
  EXPECT_EQ(0, model.GetFlowctrlIndex(1));
  EXPECT_EQ(1, model.GetFlowctrlIndex(1));
  EXPECT_EQ(2, model.GetFlowctrlIndex(0));
}

// test GetRegisterStub
TEST_F(UtestModelManagerDavinciModel, success_get_register_stub) {
  DavinciModel model(0, g_label_call_back);
  OmeTestOpUtils::InitModel(model);
  std::string binfile = "tvmbin";
  string ret = model.GetRegisterStub(binfile);
  EXPECT_EQ("tvmbin", ret);
  model.tvm_bin_kernel_.insert("tvmbin");
  ret = model.GetRegisterStub(binfile);
  EXPECT_EQ("tvmbin", ret);
}

// test InitTbeHandle
TEST_F(UtestModelManagerDavinciModel, success_init_tbe_handle) {
  DavinciModel model(0, g_label_call_back);
  OmeTestOpUtils::InitModel(model);
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>();
  Status ret = model.InitTbeHandle(op_desc);
  EXPECT_EQ(ge::INTERNAL_ERROR, ret);
}

// test InitTVMTask failed
TEST_F(UtestModelManagerDavinciModel, init_tvm_task_failed1) {
  DavinciModel model(0, g_label_call_back);
  uint16_t offset = 0;
  TaskDef *task_def = new TaskDef();
  KernelDef *kernel_def = task_def->mutable_kernel();
  map<uint32_t, OpDescPtr> op_list;
  model.op_list_ = op_list;

  KernelTaskInfo *kernel_task_info = new KernelTaskInfo();
  Status ret = kernel_task_info->InitTVMTask(&model, offset, kernel_def[0]);
  EXPECT_EQ(INTERNAL_ERROR, ret);
  task_def->clear_kernel();
  delete kernel_task_info;
  delete task_def;
}

TEST_F(UtestModelManagerDavinciModel, kernel_taskInfo_init_cce_task_failed1) {
  DavinciModel model(0, g_label_call_back);

  TaskDef *task_def = new TaskDef();
  KernelTaskInfo *kernel_task_info = new KernelTaskInfo();
  KernelDef *kernel_def = task_def->mutable_kernel();
  Status ret = kernel_task_info->InitCceTask(&model, kernel_def[0]);
  EXPECT_EQ(ge::INTERNAL_ERROR, ret);
  task_def->clear_kernel();
  delete kernel_task_info;
  delete task_def;
}

// test SetContext success
TEST_F(UtestModelManagerDavinciModel, success_kernel_taskInfo_init_set_context) {
  DavinciModel model(0, g_label_call_back);

  TaskDef *task_def = new TaskDef();
  KernelTaskInfo *kernel_task_info = new KernelTaskInfo();
  KernelDef *kernel_def = task_def->mutable_kernel();
  KernelContext *context = kernel_def->mutable_context();
  context->set_op_id(1);
  context->set_kernel_func_id(1);
  context->set_is_flowtable(true);
  context->set_args_count(1);
  context->set_args_offset("args111111", 10);

  Status ret = kernel_task_info->SetContext(kernel_def[0]);
  EXPECT_EQ(ge::SUCCESS, ret);

  ret = kernel_task_info->Release();
  EXPECT_EQ(ge::SUCCESS, ret);
  kernel_def->clear_context();
  task_def->clear_kernel();
  delete kernel_task_info;
  delete task_def;
}

// test SetContext failed
TEST_F(UtestModelManagerDavinciModel, kernel_taskInfo_init_set_context_failed1) {
  DavinciModel model(0, g_label_call_back);

  TaskDef *task_def = new TaskDef();
  KernelTaskInfo *kernel_task_info = new KernelTaskInfo();
  KernelDef *kernel_def = task_def->mutable_kernel();
  KernelContext *context = kernel_def->mutable_context();
  context->set_op_id(1);
  context->set_kernel_func_id(1);
  context->set_is_flowtable(true);
  context->set_args_count(0);
  Status ret = kernel_task_info->SetContext(kernel_def[0]);
  EXPECT_EQ(ge::INTERNAL_ERROR, ret);

  kernel_def->clear_context();
  task_def->clear_kernel();
  delete kernel_task_info;
  delete task_def;
}

TEST_F(UtestModelManagerDavinciModel, kernel_taskInfo_init_set_context_failed2) {
  DavinciModel model(0, g_label_call_back);

  TaskDef *task_def = new TaskDef();
  KernelTaskInfo *kernel_task_info = new KernelTaskInfo();
  KernelDef *kernel_def = task_def->mutable_kernel();
  KernelContext *context = kernel_def->mutable_context();
  context->set_op_id(1);
  context->set_kernel_func_id(1);
  context->set_is_flowtable(true);
  context->set_args_count(5);
  context->set_args_offset("\0\0");  // args_offset = 0

  Status ret = kernel_task_info->SetContext(kernel_def[0]);
  EXPECT_EQ(ge::PARAM_INVALID, ret);

  kernel_def->clear_context();
  task_def->clear_kernel();
  delete kernel_task_info;
  delete task_def;
}

// test success DistributeDumpTask
TEST_F(UtestModelManagerDavinciModel, success_distribute_dump_task) {
  DavinciModel model(0, g_label_call_back);
  TaskDef *task_def = new TaskDef();
  KernelTaskInfo *kernel_task_info = new KernelTaskInfo();
  KernelDef *kernel_def = task_def->mutable_kernel();

  kernel_def->set_stub_func("kerneltaskinfo");
  kernel_def->set_block_dim(10);
  kernel_def->set_args("args111111", 10);
  kernel_def->set_args_size(10);
  rtSmDesc_t l2CtrlInfo;
  l2CtrlInfo.data[0].L2_mirror_addr = 1024;
  kernel_def->set_sm_desc((void *)&l2CtrlInfo, sizeof(rtSmDesc_t));

  // for SetStream
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  std::vector<rtStream_t> stream_list;
  stream_list.push_back(stream);
  Status ret = kernel_task_info->SetStream(0, stream_list);
  EXPECT_EQ(SUCCESS, ret);

  ret = kernel_task_info->Release();
  EXPECT_EQ(SUCCESS, ret);
  rtStreamDestroy(stream);
  task_def->clear_kernel();
  delete kernel_task_info;
  delete task_def;
}

// test success GetTaskID
TEST_F(UtestModelManagerDavinciModel, success_get_task_id) {
  ModelTaskDef *model_task_def = new ModelTaskDef();
  TaskDef *task = model_task_def->add_task();
  task->set_type(RT_MODEL_TASK_KERNEL);
  TaskInfoPtr task_info = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task->type()));

  KernelTaskInfo *kernel_task_info = new KernelTaskInfo();
  uint32_t ret = task_info->GetTaskID();
  EXPECT_EQ(0, ret);
  ret = kernel_task_info->GetTaskID();
  EXPECT_EQ(0, ret);
  HcclTaskInfo *hccl_task_info = new HcclTaskInfo();
  ret = hccl_task_info->GetTaskID();
  EXPECT_EQ(0, ret);

  delete hccl_task_info;
  delete kernel_task_info;
  delete model_task_def;
}

// test StoreInputOutputTensor success
TEST_F(UtestModelManagerDavinciModel, success_store_input_output_tensor) {
  DavinciModel model(0, g_label_call_back);
  TaskDef *task_def = new TaskDef();
  KernelTaskInfo *kernel_task_info = new KernelTaskInfo();

  std::vector<void *> input_data_addrs;
  std::vector<void *> output_data_addrs;
  std::vector<::tagCcAICPUTensor> input_descs;
  std::vector<::tagCcAICPUTensor> output_descs;

  int test = 1;
  int *addr = &test;
  void *input;
  void *output;
  input = addr;
  output = addr;
  input_data_addrs.push_back(&input);
  output_data_addrs.push_back(output);

  tagCcAICPUTensor input_desc;
  tagCcAICPUTensor output_desc;
  input_descs.push_back(input_desc);
  output_descs.push_back(output_desc);

  Status ret = kernel_task_info->StoreInputOutputTensor(input_data_addrs, output_data_addrs, input_descs, output_descs);
  EXPECT_EQ(SUCCESS, ret);
  ret = kernel_task_info->Release();
  EXPECT_EQ(SUCCESS, ret);
  delete kernel_task_info;
  delete task_def;
}

// test init EventRecordTaskInfo
TEST_F(UtestModelManagerDavinciModel, success_event_record_task_init) {
  DavinciModel *model1 = nullptr;
  TaskDef *task_def1 = new TaskDef();
  EventRecordTaskInfo *eventRecordTaskInfo1 = new EventRecordTaskInfo();
  Status ret1 = eventRecordTaskInfo1->Init(task_def1[0], model1);
  EXPECT_EQ(PARAM_INVALID, ret1);

  delete eventRecordTaskInfo1;
  delete task_def1;
  delete model1;
  DavinciModel model(0, g_label_call_back);

  ModelTaskDef *model_task_info = new ModelTaskDef();
  TaskDef *task = model_task_info->add_task();
  task->set_type(RT_MODEL_TASK_EVENT_RECORD);
  TaskInfoPtr task_info = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task->type()));

  task->stream_id_ = 0;
  rtStream_t rt_stream;
  rtStreamCreate(&rt_stream, 1);
  vector<rtStream_t> stream_list;
  stream_list.push_back(rt_stream);
  model.stream_list_ = stream_list;

  task->set_event_id(1);
  model.runtime_param_.event_num = 1;
  Status ret = task_info->Init(task[0], &model);
  EXPECT_EQ(ge::INTERNAL_ERROR, ret);

  model.runtime_param_.event_num = 2;
  rtEvent_t event1;
  rtEvent_t event2;
  rtEventCreate(&event1);
  rtEventCreate(&event2);
  model.event_list_.push_back(event1);
  model.event_list_.push_back(event2);

  EventExDef *event_ex_def = task->mutable_event_ex();
  event_ex_def->set_event_type(1);

  ret = task_info->Init(task[0], &model);
  EXPECT_EQ(SUCCESS, ret);

  task->clear_event_ex();
  task_info->Release();
  delete model_task_info;
}

// test init EventWaitTaskInfo
TEST_F(UtestModelManagerDavinciModel, success_event_wait_task_init) {
  DavinciModel *model1 = nullptr;
  TaskDef *task_def1 = new TaskDef();
  EventWaitTaskInfo *event_wait_task_info1 = new EventWaitTaskInfo();
  Status ret1 = event_wait_task_info1->Init(task_def1[0], model1);
  EXPECT_EQ(PARAM_INVALID, ret1);

  delete event_wait_task_info1;
  delete task_def1;
  delete model1;
  DavinciModel model(0, g_label_call_back);

  ModelTaskDef *model_task_info = new ModelTaskDef();
  TaskDef *task = model_task_info->add_task();
  task->set_type(RT_MODEL_TASK_EVENT_WAIT);
  TaskInfoPtr task_info = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task->type()));

  task->stream_id_ = 0;
  rtStream_t rt_stream;
  rtStreamCreate(&rt_stream, 1);
  vector<rtStream_t> stream_list;
  stream_list.push_back(rt_stream);
  model.stream_list_ = stream_list;

  task->set_event_id(1);
  model.runtime_param_.event_num = 1;
  Status ret = task_info->Init(task[0], &model);
  EXPECT_EQ(ge::INTERNAL_ERROR, ret);

  model.runtime_param_.event_num = 2;
  rtEvent_t event1;
  rtEvent_t event2;
  rtEventCreate(&event1);
  rtEventCreate(&event2);
  model.event_list_.push_back(event1);
  model.event_list_.push_back(event2);

  EventExDef *event_ex_def = task->mutable_event_ex();
  event_ex_def->set_event_type(1);

  ret = task_info->Init(task[0], &model);
  EXPECT_EQ(SUCCESS, ret);

  task->clear_event_ex();
  task_info->Release();
  delete model_task_info;
}

// test fusion_start_task Init
TEST_F(UtestModelManagerDavinciModel, success_fusion_start_task_init) {
  DavinciModel *model1 = nullptr;
  TaskDef *task_def1 = new TaskDef();
  FusionStartTaskInfo *fusion_start_task_info1 = new FusionStartTaskInfo();
  Status ret1 = fusion_start_task_info1->Init(task_def1[0], model1);
  EXPECT_EQ(PARAM_INVALID, ret1);

  delete fusion_start_task_info1;
  delete task_def1;
  delete model1;
  DavinciModel model(0, g_label_call_back);
  TaskDef *task_def = new TaskDef();
  FusionStartTaskInfo *fusion_start_task_info = new FusionStartTaskInfo();
  task_def->set_stream_id(0);
  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);

  Status ret = fusion_start_task_info->Init(task_def[0], &model);
  EXPECT_EQ(SUCCESS, ret);
  delete fusion_start_task_info;
  delete task_def;
}

// test fusion_end_task Init
TEST_F(UtestModelManagerDavinciModel, success_fusion_end_task_rinit) {
  DavinciModel *model1 = nullptr;
  TaskDef *task_def1 = new TaskDef();
  FusionStopTaskInfo *fusion_stop_task_info1 = new FusionStopTaskInfo();
  Status ret1 = fusion_stop_task_info1->Init(task_def1[0], model1);
  EXPECT_EQ(PARAM_INVALID, ret1);

  delete fusion_stop_task_info1;
  delete task_def1;
  delete model1;
  DavinciModel model(0, g_label_call_back);
  TaskDef *task_def = new TaskDef();
  FusionStopTaskInfo *fusion_stop_task_info = new FusionStopTaskInfo();
  task_def->set_stream_id(0);
  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);

  Status ret = fusion_stop_task_info->Init(task_def[0], &model);
  EXPECT_EQ(SUCCESS, ret);
  delete fusion_stop_task_info;
  delete task_def;
}

// test kernel_ex_task_Release
TEST_F(UtestModelManagerDavinciModel, success_kernel_ex_task_release) {
  KernelExTaskInfo *kernel_ex_task_info = new KernelExTaskInfo();
  Status ret = kernel_ex_task_info->Release();
  EXPECT_EQ(SUCCESS, ret);

  delete kernel_ex_task_info;
}

// test hccl_Distribute
TEST_F(UtestModelManagerDavinciModel, success_Distribute7) {
  DavinciModel model(0, g_label_call_back);

  ModelTaskDef *model_task_def = new ModelTaskDef();
  TaskDef *task7 = model_task_def->add_task();
  task7->set_type(RT_MODEL_TASK_HCCL);
  TaskInfoPtr task_info7 = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task7->type()));
  Status ret = task_info7->Init(task7[0], &model);
  EXPECT_EQ(FAILED, ret);

  std::vector<TaskInfoPtr> task_list;
  task_list.push_back(task_info7);
  model.task_list_ = task_list;

  task_info7->Release();
  delete model_task_def;
}

// test hccl_GetPrivateDefByTaskDef
TEST_F(UtestModelManagerDavinciModel, success_hccl_get_private_def_by_task_def) {
  DavinciModel model(0, g_label_call_back);

  ModelTaskDef *model_task_def = new ModelTaskDef();
  TaskDef *task7 = model_task_def->add_task();
  task7->set_type(RT_MODEL_TASK_HCCL);
  // for SetStream
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);
  // for GetPrivateDefByTaskDef
  task7->set_ops_kernel_store_ptr(10);
  std::string value = "hccl_task";
  task7->set_private_def(value);

  TaskInfoPtr task_info7 = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task7->type()));
  // for Distribute
  Status ret = task_info7->Init(task7[0], &model);
  EXPECT_EQ(ge::PARAM_INVALID, ret);

  task_info7->Release();
  delete model_task_def;
}

// test hccl_task_TransToGETaskInfo
TEST_F(UtestModelManagerDavinciModel, success_hccl_trans_to_ge_task_info) {
  DavinciModel model(0, g_label_call_back);

  ModelTaskDef *model_task_def = new ModelTaskDef();
  TaskDef *task7 = model_task_def->add_task();
  // for type
  task7->set_type(RT_MODEL_TASK_HCCL);
  TaskInfoPtr task_info7 = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task7->type()));

  GETaskInfo ge_task;
  HcclTaskInfo *hccl_task_info = new HcclTaskInfo();
  hccl_task_info->TransToGETaskInfo(ge_task);

  delete hccl_task_info;
  delete model_task_def;
}

// test stream_active_task Init
TEST_F(UtestModelManagerDavinciModel, success_stream_active_task_init) {
  DavinciModel *model1 = nullptr;
  TaskDef *task_def1 = new TaskDef();
  StreamActiveTaskInfo *stream_active_task_info1 = new StreamActiveTaskInfo();
  Status ret1 = stream_active_task_info1->Init(task_def1[0], model1);
  EXPECT_EQ(PARAM_INVALID, ret1);
  delete stream_active_task_info1;
  delete task_def1;
  delete model1;

  DavinciModel model(0, g_label_call_back);
  TaskDef *task_def = new TaskDef();
  task_def->set_stream_id(0);
  rtStream_t stream1, stream2;
  rtStreamCreate(&stream1, 0);
  rtStreamCreate(&stream2, 0);
  model.stream_list_.push_back(stream1);

  StreamActiveTaskInfo *stream_active_task_info = new StreamActiveTaskInfo();

  StreamActiveDef *stream_active_def = task_def->mutable_stream_active();
  stream_active_def->set_op_index(0);
  stream_active_def->set_active_stream_id(0);

  std::map<uint32_t, uint32_t> flowctrl;
  flowctrl.insert(pair<uint32_t, uint32_t>(1, 1));
  model.flowctrl_op_index_internal_map_ = flowctrl;

  auto opDef = CreateOpDesc("", "");
  model.op_list_[0] = opDef;

  Status ret = stream_active_task_info->Init(task_def[0], &model);
  EXPECT_EQ(ge::INTERNAL_ERROR, ret);  // line 51

  model.stream_list_.push_back(stream2);
  ret = stream_active_task_info->Init(task_def[0], &model);
  EXPECT_EQ(SUCCESS, ret);

  task_def->clear_stream_active();
  delete stream_active_task_info;
  delete task_def;
}

// test label_set_task Init
TEST_F(UtestModelManagerDavinciModel, success_label_set_task_init) {
  DavinciModel *model1 = nullptr;
  TaskDef *task_def1 = new TaskDef();
  LabelSetTaskInfo *label_set_task_info1 = new LabelSetTaskInfo();
  Status ret1 = label_set_task_info1->Init(task_def1[0], model1);
  EXPECT_EQ(PARAM_INVALID, ret1);
  delete label_set_task_info1;
  delete task_def1;
  delete model1;

  DavinciModel model(0, g_label_call_back);
  TaskDef *task_def = new TaskDef();
  LabelSetTaskInfo *label_set_task_info = new LabelSetTaskInfo();
  task_def->set_stream_id(0);
  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);

  task_def->set_label_id(1);
  model.runtime_param_.batch_num = 0;
  Status ret = label_set_task_info->Init(task_def[0], &model);
  EXPECT_EQ(PARAM_INVALID, ret);

  task_def->clear_label_id();
  task_def->set_label_id(0);
  model.runtime_param_.batch_num = 1;
  rtLabel_t label;
  rtLabelCreate(&label);
  model.label_list_.push_back(label);

  ret = label_set_task_info->Init(task_def[0], &model);
  EXPECT_EQ(SUCCESS, ret);
  delete label_set_task_info;
  delete task_def;
}

// test label_goto_task init
TEST_F(UtestModelManagerDavinciModel, success_label_goto_task_init) {
  DavinciModel model(0, g_label_call_back);
  TaskDef *task_def = new TaskDef();
  LabelGotoTaskInfo *label_goto_task_info = new LabelGotoTaskInfo();
  task_def->set_stream_id(0);

  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);

  rtLabel_t label;
  rtLabelCreate(&label);
  model.label_list_.push_back(label);

  Status ret = label_goto_task_info->Init(task_def[0], &model);
  EXPECT_EQ(SUCCESS, ret);

  delete label_goto_task_info;
  delete task_def;
}

// test profiler_trace_task init
TEST_F(UtestModelManagerDavinciModel, success_profiler_trace_task_init) {
  DavinciModel *model1 = nullptr;
  TaskDef *task_def1 = new TaskDef();
  ProfilerTraceTaskInfo *profiler_trace_task_info1 = new ProfilerTraceTaskInfo();
  Status ret1 = profiler_trace_task_info1->Init(task_def1[0], model1);
  EXPECT_EQ(PARAM_INVALID, ret1);

  delete profiler_trace_task_info1;
  delete task_def1;
  delete model1;
  DavinciModel model(0, g_label_call_back);
  TaskDef *task_def = new TaskDef();
  task_def->set_stream_id(0);
  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);
  LogTimeStampDef *logTimeStampDef = task_def->mutable_log_timestamp();
  logTimeStampDef->set_logid(1);
  logTimeStampDef->set_notify(1);
  logTimeStampDef->set_flat(1);
  ProfilerTraceTaskInfo *profiler_trace_task_info = new ProfilerTraceTaskInfo();
  Status ret = profiler_trace_task_info->Init(task_def[0], &model);
  EXPECT_EQ(SUCCESS, ret);

  task_def->clear_log_timestamp();
  delete profiler_trace_task_info;
  delete task_def;
}

TEST_F(UtestModelManagerDavinciModel, profiling_model_success) {
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);

  DavinciModel model(0, g_label_call_back);
  model.model_id_ = 1;
  model.name_ = "test";
  model.version_ = 0x01;

  model.stream_list_.push_back(stream);

  ge::ModelData data;
  rtMallocHost(&data.model_data, 128);
  data.model_len = 128;

  ModelDef *model_def = new ModelDef();
  auto op_def = CreateOpDesc("", "Data");
  op_def->SetInputOffset({1});
  op_def->SetOutputOffset({100});

  ge::GeTensorDesc descin(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(descin, 4);
  op_def->AddInputDesc(descin);
  ge::GeTensorDesc desc_out(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  ge::TensorUtils::SetSize(desc_out, 32);
  op_def->AddInputDesc(desc_out);
  op_def->SetId(0);

  model.data_op_list_.push_back(op_def);
  model.op_list_[0] = op_def;

  auto opdef1 = CreateOpDesc("", "Relu");
  opdef1->SetInputOffset({1});
  opdef1->SetOutputOffset({100});

  ge::GeTensorDesc desc_in1(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(desc_in1, 4);
  opdef1->AddInputDesc(desc_in1);
  ge::GeTensorDesc desc_out1(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  ge::TensorUtils::SetSize(desc_out1, 32);
  opdef1->AddInputDesc(desc_out1);
  op_def->SetId(1);

  model.op_list_[1] = opdef1;

  auto opdef2 = CreateOpDesc("", "Relu");
  opdef2->SetInputOffset({1});
  opdef2->SetOutputOffset({100});

  ge::GeTensorDesc desc_in2(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(desc_in2, 4);
  opdef2->AddInputDesc(desc_in2);
  ge::GeTensorDesc desc_out2(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  ge::TensorUtils::SetSize(desc_out2, 32);
  opdef2->AddInputDesc(desc_out2);
  op_def->SetId(2);

  model.op_list_[2] = opdef2;

  auto opdef3 = CreateOpDesc("", "Relu");
  opdef3->SetInputOffset({1});
  opdef3->SetOutputOffset({100});

  ge::GeTensorDesc desc_in3(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(desc_in3, 4);
  opdef3->AddInputDesc(desc_in3);
  ge::GeTensorDesc desc_out3(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  ge::TensorUtils::SetSize(desc_out3, 32);
  opdef3->AddInputDesc(desc_out3);
  op_def->SetId(3);

  model.op_list_[3] = opdef3;

  auto opdef4 = CreateOpDesc("", "Relu");
  opdef4->SetInputOffset({1});
  opdef4->SetOutputOffset({100});

  ge::GeTensorDesc desc_in4(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(desc_in4, 4);
  opdef4->AddInputDesc(desc_in4);
  ge::GeTensorDesc desc_out4(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  ge::TensorUtils::SetSize(desc_out4, 32);
  opdef4->AddInputDesc(desc_out4);
  op_def->SetId(4);

  model.op_list_[4] = opdef4;

  ge::InputData input_data;
  ge::DataBuffer data_buffer;
  data_buffer.data = new char[4];
  data_buffer.length = 4;
  input_data.index = 0;
  input_data.model_id = 1;
  input_data.blobs.push_back(data_buffer);
  // model.SinkModelProfile(&model);

  rtFreeHost(data.model_data);
  // delete stream;
  delete[](char *) data_buffer.data;
  delete model_def;
}

TEST_F(UtestModelManagerDavinciModel, success_output_list_0) {
  DavinciModel model(0, g_label_call_back);

  uint32_t version = 0;
  uint64_t session_id = 0;
  uint32_t device_id = 0;
  uint64_t job_id = 0;
  Status ret = VarManager::Instance(session_id)->Init(version, session_id, device_id, job_id);
  EXPECT_EQ(ret, ge::SUCCESS);

  ret = model.ReturnNoOutput(1);
  EXPECT_EQ(ret, ge::SUCCESS);

  VarManagerPool::Instance().Destroy();
}

// test dyncbatch_distributeTask_SUCCESS
TEST_F(UtestModelManagerDavinciModel, dyncbatch_distribute_task_success) {
  DavinciModel model(0, g_label_call_back);

  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);

  rtLabel_t label = nullptr;
  rtLabelCreate(&label);
  model.label_list_.push_back(label);
  rtLabelCreate(&label);
  model.label_list_.push_back(label);
  rtLabelCreate(&label);
  model.label_list_.push_back(label);

  rtLabelDestroy(label);
  rtStreamDestroy(stream);
}

// test GetOutputDescInfo
TEST_F(UtestModelManagerDavinciModel, success_get_output_desc_info_with_netoutput) {
  setenv("GE_TRAIN", "1", true);
  DavinciModel model(0, g_label_call_back);

  auto op_desc = CreateOpDesc("Data", "Data");
  op_desc->SetOutputOffset({1});
  op_desc->SetInputOffset({1});
  op_desc->SetStreamId(0);

  {
    ge::GeTensorDesc in_desc(ge::GeShape({1, 1, 10, 10}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
    ge::TensorUtils::SetOutputTensor(in_desc, false);
    ge::TensorUtils::SetInputTensor(in_desc, true);
    op_desc->AddInputDesc(in_desc);
  }

  {
    ge::GeTensorDesc out_desc(ge::GeShape({1, 1, 10, 10}), ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorUtils::SetOutputTensor(out_desc, true);
    ge::TensorUtils::SetInputTensor(out_desc, false);
    op_desc->AddOutputDesc(out_desc);
  }

  op_desc->SetSrcName({"Pooling1", "Pooling0"});
  op_desc->SetSrcIndex({0, 1});

  auto compute_graph = make_shared<ge::ComputeGraph>("g");

  op_desc->SetType("NetOutput");

  auto net_out_node = compute_graph->AddNode(op_desc);
  model.op_list_[0] = op_desc;

  model.output_op_list_.push_back(op_desc);
  model.output_data_info_[0] = {32, (void *)0x70002010};
  model.output_memory_size_list_.push_back(64);

  vector<InputOutputDescInfo> output_shapes;
  vector<uint32_t> formats;
  EXPECT_EQ(ge::SUCCESS, model.GetOutputDescInfo(output_shapes, formats));

  setenv("GE_TRAIN", "0", true);
}

TEST_F(UtestModelManagerDavinciModel, device_runtime_success_Run) {
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);

  DavinciModel model(0, g_label_call_back);

  model.stream_list_.push_back(stream);
  auto model_def = make_shared<ge::Model>();

  auto op_def = CreateOpDesc("", "Data");

  auto compute_graph = make_shared<ge::ComputeGraph>("g");
  compute_graph->AddNode(op_def);

  model_def->SetGraph(ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph));

  model.data_op_list_.push_back(op_def);

  model.data_inputer_ = new DataInputer();

  model.ModelRunStart();

  OutputData output_data;
  ge::InputData input_data;

  ge::DataBuffer data_buffer;
  data_buffer.data = new char[16];
  data_buffer.length = 16;

  input_data.index = 0;
  input_data.model_id = 1;
  input_data.blobs.push_back(data_buffer);

  model.ModelRunStop();

  delete[](char *) data_buffer.data;
}

TEST_F(UtestModelManagerDavinciModel, run_failed) {
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);

  DavinciModel model(0, g_label_call_back);

  model.stream_list_.push_back(stream);
  auto model_def = make_shared<ge::Model>();

  auto op_def = CreateOpDesc("", "Data");

  auto compute_graph = make_shared<ge::ComputeGraph>("g");
  compute_graph->AddNode(op_def);

  model_def->SetGraph(ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph));

  model.data_op_list_.push_back(op_def);

  model.data_inputer_ = new DataInputer();

  model.ModelRunStart();

  OutputData output_data;
  ge::InputData input_data;

  ge::DataBuffer data_buffer;
  data_buffer.data = new char[16];
  data_buffer.length = 16;

  input_data.index = 0;
  input_data.model_id = 1;
  input_data.blobs.push_back(data_buffer);

  model.ModelRunStop();
  delete[](char *) data_buffer.data;
}

TEST_F(UtestModelManagerDavinciModel, run_failed01) {
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);

  DavinciModel model(0, g_label_call_back);

  model.stream_list_.push_back(stream);
  auto model_def = make_shared<ge::Model>();

  auto op_def = CreateOpDesc("", "Data");

  auto compute_graph = make_shared<ge::ComputeGraph>("g");
  compute_graph->AddNode(op_def);

  model_def->SetGraph(ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph));

  model.data_op_list_.push_back(op_def);

  model.data_inputer_ = nullptr;
  model.ModelRunStart();

  model.ModelRunStop();
}

TEST_F(UtestModelManagerDavinciModel, init_tbe_handle_fe_registered) {
  DavinciModel::tvm_bin_kernel_.clear();
  DavinciModel model(0, g_label_call_back);
  OpDescPtr op_desc = CreateOpDesc("MatMul", "MatMul");

  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/MatMul", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);

  std::string kernel_name("kernel/MatMul");
  AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", kernel_name);

  EXPECT_EQ(model.InitTbeHandle(op_desc), SUCCESS);
  EXPECT_EQ(model.InitTbeHandle(op_desc), SUCCESS);

  EXPECT_EQ(model.used_tbe_handle_map_.size(), 0);
  DavinciModel::tvm_bin_kernel_.clear();
}

TEST_F(UtestModelManagerDavinciModel, init_tbe_handle_ge_registered) {
  DavinciModel::tvm_bin_kernel_.clear();
  DavinciModel model(0, g_label_call_back);
  OpDescPtr op_desc = CreateOpDesc("MatMul", "MatMul");

  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/MatMul", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);

  std::string kernel_name("kernel/MatMul");
  AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", kernel_name);

  string session_graph_id;
  AttrUtils::GetStr(op_desc, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
  const char *bin_file_key = DavinciModel::GetRegisterStub(op_desc->GetName(), session_graph_id);
  model.used_tbe_handle_map_[bin_file_key] = 1;  // test first register.

  EXPECT_EQ(model.InitTbeHandle(op_desc), SUCCESS);
  EXPECT_EQ(model.InitTbeHandle(op_desc), SUCCESS);

  EXPECT_EQ(model.used_tbe_handle_map_.size(), 1);

  auto it = model.used_tbe_handle_map_.find(bin_file_key);
  EXPECT_NE(it, model.used_tbe_handle_map_.end());
  EXPECT_EQ(it->second, 3);
  DavinciModel::tvm_bin_kernel_.clear();
}
}  // namespace ge
