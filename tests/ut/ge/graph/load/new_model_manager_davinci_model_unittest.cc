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
using namespace domi;

namespace ge {
class TEST_model_manager_davinci_model : public testing::Test {
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

shared_ptr<ge::ModelListener> g_labelCallBack(new DModelListener());

static ge::OpDescPtr CreateOpDesc(string name = "", string type = "") {
  auto opDesc = std::make_shared<ge::OpDesc>(name, type);
  opDesc->SetStreamId(0);
  opDesc->SetId(0);

  ge::AttrUtils::SetFloat(opDesc, ge::ATTR_NAME_ALPHA, 0);
  ge::AttrUtils::SetFloat(opDesc, ge::ATTR_NAME_BETA, 0);

  opDesc->SetWorkspace({});
  ;
  opDesc->SetWorkspaceBytes({});
  opDesc->SetInputOffset({});
  opDesc->SetOutputOffset({});

  ge::AttrUtils::SetListStr(opDesc, ge::ATTR_NAME_WEIGHT_NAME, {});
  ge::AttrUtils::SetInt(opDesc, ge::POOLING_ATTR_MODE, 0);
  ge::AttrUtils::SetInt(opDesc, ge::POOLING_ATTR_PAD_MODE, 0);
  ge::AttrUtils::SetInt(opDesc, ge::POOLING_ATTR_DATA_MODE, 0);
  ge::AttrUtils::SetInt(opDesc, ge::POOLING_ATTR_CEIL_MODE, 0);
  ge::AttrUtils::SetInt(opDesc, ge::POOLING_ATTR_NAN_OPT, 0);
  ge::AttrUtils::SetListInt(opDesc, ge::POOLING_ATTR_WINDOW, {});
  ge::AttrUtils::SetListInt(opDesc, ge::POOLING_ATTR_PAD, {});
  ge::AttrUtils::SetListInt(opDesc, ge::POOLING_ATTR_STRIDE, {});
  ge::AttrUtils::SetListInt(opDesc, ge::ATTR_NAME_ACTIVE_STREAM_LIST, {1, 1});
  ge::AttrUtils::SetInt(opDesc, ge::ATTR_NAME_STREAM_SWITCH_COND, 0);
  ge::AttrUtils::SetInt(opDesc, ge::ATTR_NAME_FRAMEWORK_FWK_TYPE, FMK_TYPE_T);
  return opDesc;
}

// tset failed_rt_free_host
TEST_F(TEST_model_manager_davinci_model, failed_rt_free_host) {
  DavinciModel model(0, g_labelCallBack);

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

  auto computeGraph = make_shared<ge::ComputeGraph>("g");
  auto node = computeGraph->AddNode(op_desc);

  OmeTestOpUtils::InitModel(model);

  model.data_op_list_.push_back(op_desc);

  EXPECT_EQ(ge::INTERNAL_ERROR, model.ReturnResult(1, 1, false, false, &output_data));
}

// test modeldef_fail
TEST_F(TEST_model_manager_davinci_model, contruct_modeldef_createfail) {
  DavinciModel model(0, g_labelCallBack);

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
TEST_F(TEST_model_manager_davinci_model, copy_input_data_to_model_fail) {
  DavinciModel model(0, g_labelCallBack);

  ge::InputData inputdata;
  ge::DataBuffer databuffer;
  databuffer.data = new char[16];
  databuffer.length = 16;
  inputdata.index = 0;
  inputdata.model_id = 1;
  inputdata.blobs.push_back(databuffer);

  model.op_list_.clear();
  //    EXPECT_EQ(ge::PARAM_INVALID, model.CopyInputDataToModel(inputdata.blobs, 0));

  delete[](char *) databuffer.data;
}

// test StreamNum
TEST_F(TEST_model_manager_davinci_model, streamnum_success) {
  DavinciModel *model = new DavinciModel(0, g_labelCallBack);

  OmeTestOpUtils::InitModel(*model);
  // EXPECT_EQ(ge::SUCCESS, model->Init());

  EXPECT_EQ(0, model->StreamNum());
  EXPECT_EQ(ge::INTERNAL_ERROR, model->ModelRunStart());

  EXPECT_EQ(ge::SUCCESS, model->ModelRunStop());

  delete model;
}

// test EventNum
TEST_F(TEST_model_manager_davinci_model, eventnum_success) {
  DavinciModel *model = new DavinciModel(0, g_labelCallBack);

  OmeTestOpUtils::InitModel(*model);

  // EXPECT_EQ(ge::SUCCESS, model->Init());

  EXPECT_EQ(0, model->EventNum());
  EXPECT_EQ(ge::INTERNAL_ERROR, model->ModelRunStart());

  EXPECT_EQ(ge::SUCCESS, model->ModelRunStop());

  delete model;
}

TEST_F(TEST_model_manager_davinci_model, handlelist_success) {
  DavinciModel *model = new DavinciModel(0, g_labelCallBack);

  OmeTestOpUtils::InitModel(*model);

  // EXPECT_EQ(ge::SUCCESS, model->Init());

  EXPECT_EQ(ge::INTERNAL_ERROR, model->ModelRunStart());

  EXPECT_EQ(ge::SUCCESS, model->ModelRunStop());

  delete model;
}

// test GetEventList
TEST_F(TEST_model_manager_davinci_model, eventlist_success) {
  DavinciModel *model = new DavinciModel(0, g_labelCallBack);

  OmeTestOpUtils::InitModel(*model);

  // EXPECT_EQ(ge::SUCCESS, model->Init());

  EXPECT_EQ(true, model->GetEventList().empty());
  EXPECT_EQ(ge::INTERNAL_ERROR, model->ModelRunStart());

  EXPECT_EQ(ge::SUCCESS, model->ModelRunStop());

  delete model;
}

// test rtMalloc
TEST_F(TEST_model_manager_davinci_model, failed_reset_device) {
  DavinciModel model(0, g_labelCallBack);
  ge::OutputData output_data;
  ge::DataBuffer bufdata;
  rtMalloc(&bufdata.data, 128, RT_MEMORY_HBM);
  bufdata.length = 128;
  output_data.blobs.push_back(bufdata);
  EXPECT_EQ(ge::INTERNAL_ERROR, model.ReturnResult(1, 1, true, false, &output_data));
  rtFree(bufdata.data);
}

// test priority
TEST_F(TEST_model_manager_davinci_model, init_not_support_priority) {
  int32_t priority = 8;
  DavinciModel model(priority, g_labelCallBack);
  // EXPECT_EQ(ge::PARAM_INVALID, model.Init());
}

// test GetInputOutputDescInfo
TEST_F(TEST_model_manager_davinci_model, success_GetInputOutputDescInfo_without_netoutput) {
  DavinciModel model(0, g_labelCallBack);

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

  auto computeGraph = make_shared<ge::ComputeGraph>("g");
  auto node = computeGraph->AddNode(op_desc);

  model.data_op_list_.push_back(op_desc);
  model.output_size_list_.push_back(32);

  model.op_list_[0] = op_desc;

  model.output_op_list_.push_back(op_desc);

  vector<InputOutputDescInfo> input_shapes;
  vector<InputOutputDescInfo> output_shapes;
  EXPECT_EQ(ge::SUCCESS, model.GetInputOutputDescInfo(input_shapes, output_shapes));
}

TEST_F(TEST_model_manager_davinci_model, CopyTensorFromSrcVarNode_input_is_nullptr) {
  NodePtr src_node = nullptr;
  NodePtr dst_node = nullptr;
  DavinciModel model(0, g_labelCallBack);
  Status ret = model.CopyTensorFromSrcVarNode(src_node, dst_node);
  EXPECT_EQ(FAILED, ret);
}

TEST_F(TEST_model_manager_davinci_model, CopyTensorFromSrcVarNode_success) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("Cast", "Cast");
  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc dims_tensor_desc_in(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  NodePtr src_node = graph->AddNode(op_desc_ptr);
  NodePtr dst_node = graph->AddNode(op_desc_ptr);
  DavinciModel model(0, g_labelCallBack);
  Status ret = model.CopyTensorFromSrcVarNode(src_node, dst_node);
  // EXPECT_EQ(SUCCESS, ret);
}

TEST_F(TEST_model_manager_davinci_model, CopyVarData_graph_is_nullptr) {
  ge::ComputeGraphPtr graph = nullptr;
  DavinciModel model(0, g_labelCallBack);
  Status ret = model.CopyVarData(graph);
  EXPECT_EQ(FAILED, ret);
}

TEST_F(TEST_model_manager_davinci_model, CopyVarData_success) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("Variable", "Variable");
  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc dims_tensor_desc_in(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT16);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  NodePtr src_node = graph->AddNode(op_desc_ptr);
  (void)ge::AttrUtils::SetStr(src_node->GetOpDesc(), "_copy_from_var_node", "abc");
  (void)ge::AttrUtils::SetBool(src_node->GetOpDesc(), "_copy_value", false);

  DavinciModel model(0, g_labelCallBack);
  Status ret = model.CopyVarData(graph);
  // EXPECT_EQ(SUCCESS, ret);
}

TEST_F(TEST_model_manager_davinci_model, GetInputOutputDescInfo_without_data_op_list) {
  DavinciModel model(0, g_labelCallBack);
  vector<InputOutputDescInfo> input_list;
  vector<InputOutputDescInfo> output_list;
  Status ret = model.GetInputOutputDescInfo(input_list, output_list);
  EXPECT_EQ(SUCCESS, ret);
}

// test GetInputOutputDescInfo
TEST_F(TEST_model_manager_davinci_model, success_GetInputOutputDescInfo_with_netoutput) {
  DavinciModel model(0, g_labelCallBack);

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

  auto computeGraph = make_shared<ge::ComputeGraph>("g");
  auto data_node = computeGraph->AddNode(op_desc);

  model.data_op_list_.push_back(op_desc);

  op_desc->SetType("NetOutput");

  auto no_node = computeGraph->AddNode(op_desc);

  model.op_list_[0] = op_desc;

  model.output_op_list_.push_back(op_desc);
  model.output_size_list_.push_back(32);

  vector<InputOutputDescInfo> input_shapes;
  vector<InputOutputDescInfo> output_shapes;
  EXPECT_EQ(ge::SUCCESS, model.GetInputOutputDescInfo(input_shapes, output_shapes));
}

TEST_F(TEST_model_manager_davinci_model, success_GetInputOutputDescInfoForZeroCopy_with_netoutput) {
  DavinciModel model(0, g_labelCallBack);

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

  auto computeGraph = make_shared<ge::ComputeGraph>("g");
  auto data_node = computeGraph->AddNode(op_desc);

  model.data_op_list_.push_back(op_desc);

  op_desc->SetType("NetOutput");

  auto netout_node = computeGraph->AddNode(op_desc);
  model.op_list_[0] = op_desc;

  model.output_op_list_.push_back(op_desc);
  model.output_size_list_.push_back(32);
  model.output_memory_size_list_.push_back(64);

  vector<InputOutputDescInfo> input_shapes;
  vector<InputOutputDescInfo> output_shapes;
  EXPECT_EQ(ge::SUCCESS, model.GetInputOutputDescInfoForZeroCopy(input_shapes, output_shapes));
}

TEST_F(TEST_model_manager_davinci_model, success_GetInputOutputDescInfo_dim_size_not4) {
  DavinciModel model(0, g_labelCallBack);

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

  auto computeGraph = make_shared<ge::ComputeGraph>("g");
  auto data_node = computeGraph->AddNode(op_desc);

  model.data_op_list_.push_back(op_desc);

  op_desc->SetType("NetOutput");

  auto netout_node = computeGraph->AddNode(op_desc);
  model.op_list_[0] = op_desc;

  model.output_op_list_.push_back(op_desc);
  model.output_size_list_.push_back(32);

  vector<InputOutputDescInfo> input_shapes;
  vector<InputOutputDescInfo> output_shapes;
  EXPECT_EQ(ge::SUCCESS, model.GetInputOutputDescInfo(input_shapes, output_shapes));
}

// test GetLabelList
TEST_F(TEST_model_manager_davinci_model, GetLabelList_success) {
  DavinciModel model(0, g_labelCallBack);
  OmeTestOpUtils::InitModel(model);
  vector<rtLabel_t> label_list_;
  model.label_list_ = label_list_;
  EXPECT_EQ(label_list_, model.GetLabelList());
}

// test GetInputListSize
TEST_F(TEST_model_manager_davinci_model, GetInputListSize_success) {
  DavinciModel model(0, g_labelCallBack);
  OmeTestOpUtils::InitModel(model);
  vector<OpDescPtr> data_op_list_;
  data_op_list_.push_back(std::make_shared<OpDesc>());
  model.data_op_list_ = data_op_list_;
}

// test GetFlowctrlOpList
TEST_F(TEST_model_manager_davinci_model, GetFlowctrlOpList_success) {
  DavinciModel model(0, g_labelCallBack);
  OmeTestOpUtils::InitModel(model);
  std::map<uint32_t, uint32_t> flowctrl_op_index_internal_map_;
  flowctrl_op_index_internal_map_.insert(pair<uint32_t, uint32_t>(1, 1));
  model.flowctrl_op_index_internal_map_ = flowctrl_op_index_internal_map_;
  // EXPECT_EQ(flowctrl_op_index_internal_map_, model.GetFlowctrlOpList());
}

// test SetFlowctrlOpList
TEST_F(TEST_model_manager_davinci_model, GetFlowctrlIndex_success) {
  DavinciModel model(0, g_labelCallBack);
  OmeTestOpUtils::InitModel(model);
  EXPECT_EQ(0, model.GetFlowctrlIndex(0));
  EXPECT_EQ(1, model.GetFlowctrlIndex(0));
  EXPECT_EQ(0, model.GetFlowctrlIndex(1));
  EXPECT_EQ(1, model.GetFlowctrlIndex(1));
  EXPECT_EQ(2, model.GetFlowctrlIndex(0));
}

// test GetRegisterStub
TEST_F(TEST_model_manager_davinci_model, success_GetRegisterStub) {
  DavinciModel model(0, g_labelCallBack);
  OmeTestOpUtils::InitModel(model);
  std::string binfile = "tvmbin";
  string ret = model.GetRegisterStub(binfile);
  EXPECT_EQ("tvmbin", ret);
  model.tvm_bin_kernel_.insert("tvmbin");
  ret = model.GetRegisterStub(binfile);
  EXPECT_EQ("tvmbin", ret);
}

// test InitTbeHandle
TEST_F(TEST_model_manager_davinci_model, success_InitTbeHandle) {
  DavinciModel model(0, g_labelCallBack);
  OmeTestOpUtils::InitModel(model);
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>();
  Status ret = model.InitTbeHandle(op_desc);
  EXPECT_EQ(ge::INTERNAL_ERROR, ret);
}

// test InitTVMTask failed
TEST_F(TEST_model_manager_davinci_model, InitTVMTask_failed1) {
  DavinciModel model(0, g_labelCallBack);
  uint16_t offset = 0;
  TaskDef *taskDef = new TaskDef();
  KernelDef *kernelDef = taskDef->mutable_kernel();
  map<uint32_t, OpDescPtr> op_list;
  model.op_list_ = op_list;

  KernelTaskInfo *kernelTaskInfo = new KernelTaskInfo();
  Status ret = kernelTaskInfo->InitTVMTask(&model, offset, kernelDef[0]);
  EXPECT_EQ(INTERNAL_ERROR, ret);
  taskDef->clear_kernel();
  delete kernelTaskInfo;
  delete taskDef;
}

TEST_F(TEST_model_manager_davinci_model, KernelTaskInfo_InitCceTask_failed1) {
  DavinciModel model(0, g_labelCallBack);

  TaskDef *taskDef = new TaskDef();
  KernelTaskInfo *kernelTaskInfo = new KernelTaskInfo();
  KernelDef *kernelDef = taskDef->mutable_kernel();
  Status ret = kernelTaskInfo->InitCceTask(&model, kernelDef[0]);
  EXPECT_EQ(ge::INTERNAL_ERROR, ret);
  taskDef->clear_kernel();
  delete kernelTaskInfo;
  delete taskDef;
}

// test SetContext success
TEST_F(TEST_model_manager_davinci_model, success_KernelTaskInfo_SetContext) {
  DavinciModel model(0, g_labelCallBack);

  TaskDef *taskDef = new TaskDef();
  KernelTaskInfo *kernelTaskInfo = new KernelTaskInfo();
  KernelDef *kernelDef = taskDef->mutable_kernel();
  KernelContext *context = kernelDef->mutable_context();
  context->set_op_id(1);
  context->set_kernel_func_id(1);
  context->set_is_flowtable(true);
  context->set_args_count(1);
  context->set_args_offset("args111111", 10);

  Status ret = kernelTaskInfo->SetContext(kernelDef[0]);
  EXPECT_EQ(ge::SUCCESS, ret);

  ret = kernelTaskInfo->Release();
  EXPECT_EQ(ge::SUCCESS, ret);
  kernelDef->clear_context();
  taskDef->clear_kernel();
  delete kernelTaskInfo;
  delete taskDef;
}

// test SetContext failed
TEST_F(TEST_model_manager_davinci_model, KernelTaskInfo_SetContext_failed1) {
  DavinciModel model(0, g_labelCallBack);

  TaskDef *taskDef = new TaskDef();
  KernelTaskInfo *kernelTaskInfo = new KernelTaskInfo();
  KernelDef *kernelDef = taskDef->mutable_kernel();
  KernelContext *context = kernelDef->mutable_context();
  context->set_op_id(1);
  context->set_kernel_func_id(1);
  context->set_is_flowtable(true);
  context->set_args_count(0);
  Status ret = kernelTaskInfo->SetContext(kernelDef[0]);
  EXPECT_EQ(ge::INTERNAL_ERROR, ret);

  kernelDef->clear_context();
  taskDef->clear_kernel();
  delete kernelTaskInfo;
  delete taskDef;
}

TEST_F(TEST_model_manager_davinci_model, KernelTaskInfo_SetContext_failed2) {
  DavinciModel model(0, g_labelCallBack);

  TaskDef *taskDef = new TaskDef();
  KernelTaskInfo *kernelTaskInfo = new KernelTaskInfo();
  KernelDef *kernelDef = taskDef->mutable_kernel();
  KernelContext *context = kernelDef->mutable_context();
  context->set_op_id(1);
  context->set_kernel_func_id(1);
  context->set_is_flowtable(true);
  context->set_args_count(5);
  context->set_args_offset("\0\0");  // args_offset = 0

  Status ret = kernelTaskInfo->SetContext(kernelDef[0]);
  EXPECT_EQ(ge::PARAM_INVALID, ret);

  kernelDef->clear_context();
  taskDef->clear_kernel();
  delete kernelTaskInfo;
  delete taskDef;
}

// test success DistributeDumpTask
TEST_F(TEST_model_manager_davinci_model, success_DistributeDumpTask) {
  DavinciModel model(0, g_labelCallBack);
  TaskDef *taskDef = new TaskDef();
  KernelTaskInfo *kernelTaskInfo = new KernelTaskInfo();
  KernelDef *kernelDef = taskDef->mutable_kernel();

  kernelDef->set_stub_func("kerneltaskinfo");
  kernelDef->set_block_dim(10);
  kernelDef->set_args("args111111", 10);
  kernelDef->set_args_size(10);
  rtSmDesc_t l2CtrlInfo;
  l2CtrlInfo.data[0].L2_mirror_addr = 1024;
  kernelDef->set_sm_desc((void *)&l2CtrlInfo, sizeof(rtSmDesc_t));

  // for SetStream
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  std::vector<rtStream_t> stream_list;
  stream_list.push_back(stream);
  Status ret = kernelTaskInfo->SetStream(0, stream_list);
  EXPECT_EQ(SUCCESS, ret);

  ret = kernelTaskInfo->Release();
  EXPECT_EQ(SUCCESS, ret);
  rtStreamDestroy(stream);
  taskDef->clear_kernel();
  delete kernelTaskInfo;
  delete taskDef;
}

// test success GetTaskID
TEST_F(TEST_model_manager_davinci_model, success_GetTaskID) {
  ModelTaskDef *modelTaskDef = new ModelTaskDef();
  TaskDef *task = modelTaskDef->add_task();
  task->set_type(RT_MODEL_TASK_KERNEL);
  TaskInfoPtr task_info = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task->type()));

  KernelTaskInfo *kernelTaskInfo = new KernelTaskInfo();
  uint32_t ret = task_info->GetTaskID();
  EXPECT_EQ(0, ret);
  ret = kernelTaskInfo->GetTaskID();
  EXPECT_EQ(0, ret);
  HcclTaskInfo *hcclTaskInfo = new HcclTaskInfo();
  ret = hcclTaskInfo->GetTaskID();
  EXPECT_EQ(0, ret);

  delete hcclTaskInfo;
  delete kernelTaskInfo;
  delete modelTaskDef;
}

// test StoreInputOutputTensor success
TEST_F(TEST_model_manager_davinci_model, success_StoreInputOutputTensor) {
  DavinciModel model(0, g_labelCallBack);
  TaskDef *taskDef = new TaskDef();
  KernelTaskInfo *kernelTaskInfo = new KernelTaskInfo();

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

  Status ret = kernelTaskInfo->StoreInputOutputTensor(input_data_addrs, output_data_addrs, input_descs, output_descs);
  EXPECT_EQ(SUCCESS, ret);
  ret = kernelTaskInfo->Release();
  EXPECT_EQ(SUCCESS, ret);
  delete kernelTaskInfo;
  delete taskDef;
}

// test init EventRecordTaskInfo
TEST_F(TEST_model_manager_davinci_model, success_event_record_task_Init) {
  DavinciModel *model1 = nullptr;
  TaskDef *taskDef1 = new TaskDef();
  EventRecordTaskInfo *eventRecordTaskInfo1 = new EventRecordTaskInfo();
  Status ret1 = eventRecordTaskInfo1->Init(taskDef1[0], model1);
  EXPECT_EQ(PARAM_INVALID, ret1);

  delete eventRecordTaskInfo1;
  delete taskDef1;
  delete model1;
  DavinciModel model(0, g_labelCallBack);

  ModelTaskDef *modelTaskInfo = new ModelTaskDef();
  TaskDef *task = modelTaskInfo->add_task();
  task->set_type(RT_MODEL_TASK_EVENT_RECORD);
  TaskInfoPtr task_info = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task->type()));

  task->stream_id_ = 0;
  rtStream_t rtStream;
  rtStreamCreate(&rtStream, 1);
  vector<rtStream_t> stream_list;
  stream_list.push_back(rtStream);
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

  EventExDef *eventExDef = task->mutable_event_ex();
  eventExDef->set_event_type(1);

  ret = task_info->Init(task[0], &model);
  EXPECT_EQ(SUCCESS, ret);

  task->clear_event_ex();
  task_info->Release();
  delete modelTaskInfo;
}

// test init EventWaitTaskInfo
TEST_F(TEST_model_manager_davinci_model, success_event_wait_task_Init) {
  DavinciModel *model1 = nullptr;
  TaskDef *taskDef1 = new TaskDef();
  EventWaitTaskInfo *eventWaitTaskInfo1 = new EventWaitTaskInfo();
  Status ret1 = eventWaitTaskInfo1->Init(taskDef1[0], model1);
  EXPECT_EQ(PARAM_INVALID, ret1);

  delete eventWaitTaskInfo1;
  delete taskDef1;
  delete model1;
  DavinciModel model(0, g_labelCallBack);

  ModelTaskDef *modelTaskInfo = new ModelTaskDef();
  TaskDef *task = modelTaskInfo->add_task();
  task->set_type(RT_MODEL_TASK_EVENT_WAIT);
  TaskInfoPtr task_info = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task->type()));

  task->stream_id_ = 0;
  rtStream_t rtStream;
  rtStreamCreate(&rtStream, 1);
  vector<rtStream_t> stream_list;
  stream_list.push_back(rtStream);
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

  EventExDef *eventExDef = task->mutable_event_ex();
  eventExDef->set_event_type(1);

  ret = task_info->Init(task[0], &model);
  EXPECT_EQ(SUCCESS, ret);

  task->clear_event_ex();
  task_info->Release();
  delete modelTaskInfo;
}

// test fusion_start_task Init
TEST_F(TEST_model_manager_davinci_model, success_fusion_start_task_Init) {
  DavinciModel *model1 = nullptr;
  TaskDef *taskDef1 = new TaskDef();
  FusionStartTaskInfo *fusionStartTaskInfo1 = new FusionStartTaskInfo();
  Status ret1 = fusionStartTaskInfo1->Init(taskDef1[0], model1);
  EXPECT_EQ(PARAM_INVALID, ret1);

  delete fusionStartTaskInfo1;
  delete taskDef1;
  delete model1;
  DavinciModel model(0, g_labelCallBack);
  TaskDef *taskDef = new TaskDef();
  FusionStartTaskInfo *fusionStartTaskInfo = new FusionStartTaskInfo();
  taskDef->set_stream_id(0);
  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);

  Status ret = fusionStartTaskInfo->Init(taskDef[0], &model);
  EXPECT_EQ(SUCCESS, ret);
  delete fusionStartTaskInfo;
  delete taskDef;
}

// test fusion_end_task Init
TEST_F(TEST_model_manager_davinci_model, success_fusion_end_task_Init) {
  DavinciModel *model1 = nullptr;
  TaskDef *taskDef1 = new TaskDef();
  FusionStopTaskInfo *fusionStopTaskInfo1 = new FusionStopTaskInfo();
  Status ret1 = fusionStopTaskInfo1->Init(taskDef1[0], model1);
  EXPECT_EQ(PARAM_INVALID, ret1);

  delete fusionStopTaskInfo1;
  delete taskDef1;
  delete model1;
  DavinciModel model(0, g_labelCallBack);
  TaskDef *taskDef = new TaskDef();
  FusionStopTaskInfo *fusionStopTaskInfo = new FusionStopTaskInfo();
  taskDef->set_stream_id(0);
  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);

  Status ret = fusionStopTaskInfo->Init(taskDef[0], &model);
  EXPECT_EQ(SUCCESS, ret);
  delete fusionStopTaskInfo;
  delete taskDef;
}

// test kernel_ex_task_Release
TEST_F(TEST_model_manager_davinci_model, success_kernel_ex_task_Release) {
  KernelExTaskInfo *kernelExTaskInfo = new KernelExTaskInfo();
  Status ret = kernelExTaskInfo->Release();
  EXPECT_EQ(SUCCESS, ret);

  delete kernelExTaskInfo;
}

// test hccl_Distribute
TEST_F(TEST_model_manager_davinci_model, success_Distribute7) {
  DavinciModel model(0, g_labelCallBack);

  ModelTaskDef *modelTaskDef = new ModelTaskDef();
  TaskDef *task7 = modelTaskDef->add_task();
  task7->set_type(RT_MODEL_TASK_HCCL);
  TaskInfoPtr task_info7 = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task7->type()));
  Status ret = task_info7->Init(task7[0], &model);
  EXPECT_EQ(FAILED, ret);

  std::vector<TaskInfoPtr> task_list;
  task_list.push_back(task_info7);
  model.task_list_ = task_list;

  task_info7->Release();
  delete modelTaskDef;
}

// test hccl_GetPrivateDefByTaskDef
TEST_F(TEST_model_manager_davinci_model, success_hccl_GetPrivateDefByTaskDef) {
  DavinciModel model(0, g_labelCallBack);

  ModelTaskDef *modelTaskDef = new ModelTaskDef();
  TaskDef *task7 = modelTaskDef->add_task();
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
  delete modelTaskDef;
}

// test hccl_task_TransToGETaskInfo
TEST_F(TEST_model_manager_davinci_model, success_hccl_TransToGETaskInfo) {
  DavinciModel model(0, g_labelCallBack);

  ModelTaskDef *modelTaskDef = new ModelTaskDef();
  TaskDef *task7 = modelTaskDef->add_task();
  // for type
  task7->set_type(RT_MODEL_TASK_HCCL);
  TaskInfoPtr task_info7 = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task7->type()));

  GETaskInfo ge_task;
  HcclTaskInfo *hcclTaskInfo = new HcclTaskInfo();
  hcclTaskInfo->TransToGETaskInfo(ge_task);

  delete hcclTaskInfo;
  delete modelTaskDef;
}

// test stream_active_task Init
TEST_F(TEST_model_manager_davinci_model, success_stream_active_task_Init) {
  DavinciModel *model1 = nullptr;
  TaskDef *taskDef1 = new TaskDef();
  StreamActiveTaskInfo *streamActiveTaskInfo1 = new StreamActiveTaskInfo();
  Status ret1 = streamActiveTaskInfo1->Init(taskDef1[0], model1);
  EXPECT_EQ(PARAM_INVALID, ret1);
  delete streamActiveTaskInfo1;
  delete taskDef1;
  delete model1;

  DavinciModel model(0, g_labelCallBack);
  TaskDef *taskDef = new TaskDef();
  taskDef->set_stream_id(0);
  rtStream_t stream1, stream2;
  rtStreamCreate(&stream1, 0);
  rtStreamCreate(&stream2, 0);
  model.stream_list_.push_back(stream1);

  StreamActiveTaskInfo *streamActiveTaskInfo = new StreamActiveTaskInfo();

  StreamActiveDef *streamActiveDef = taskDef->mutable_stream_active();
  streamActiveDef->set_op_index(0);
  streamActiveDef->set_active_stream_id(0);

  std::map<uint32_t, uint32_t> flowctrl;
  flowctrl.insert(pair<uint32_t, uint32_t>(1, 1));
  model.flowctrl_op_index_internal_map_ = flowctrl;

  auto opDef = CreateOpDesc("", "");
  model.op_list_[0] = opDef;

  Status ret = streamActiveTaskInfo->Init(taskDef[0], &model);
  EXPECT_EQ(ge::INTERNAL_ERROR, ret);  // line 51

  model.stream_list_.push_back(stream2);
  ret = streamActiveTaskInfo->Init(taskDef[0], &model);
  EXPECT_EQ(SUCCESS, ret);

  taskDef->clear_stream_active();
  delete streamActiveTaskInfo;
  delete taskDef;
}

// test label_set_task Init
TEST_F(TEST_model_manager_davinci_model, success_label_set_task_Init) {
  DavinciModel *model1 = nullptr;
  TaskDef *taskDef1 = new TaskDef();
  LabelSetTaskInfo *labelSetTaskInfo1 = new LabelSetTaskInfo();
  Status ret1 = labelSetTaskInfo1->Init(taskDef1[0], model1);
  EXPECT_EQ(PARAM_INVALID, ret1);
  delete labelSetTaskInfo1;
  delete taskDef1;
  delete model1;

  DavinciModel model(0, g_labelCallBack);
  TaskDef *taskDef = new TaskDef();
  LabelSetTaskInfo *labelSetTaskInfo = new LabelSetTaskInfo();
  taskDef->set_stream_id(0);
  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);

  taskDef->set_label_id(1);
  model.runtime_param_.batch_num = 0;
  Status ret = labelSetTaskInfo->Init(taskDef[0], &model);
  EXPECT_EQ(PARAM_INVALID, ret);

  taskDef->clear_label_id();
  taskDef->set_label_id(0);
  model.runtime_param_.batch_num = 1;
  rtLabel_t label;
  rtLabelCreate(&label);
  model.label_list_.push_back(label);

  ret = labelSetTaskInfo->Init(taskDef[0], &model);
  EXPECT_EQ(SUCCESS, ret);
  delete labelSetTaskInfo;
  delete taskDef;
}

// test label_goto_task init
TEST_F(TEST_model_manager_davinci_model, success_label_goto_task_Init) {
  DavinciModel model(0, g_labelCallBack);
  TaskDef *taskDef = new TaskDef();
  LabelGotoTaskInfo *labelGotoTaskInfo = new LabelGotoTaskInfo();
  taskDef->set_stream_id(0);

  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);

  rtLabel_t label;
  rtLabelCreate(&label);
  model.label_list_.push_back(label);

  Status ret = labelGotoTaskInfo->Init(taskDef[0], &model);
  EXPECT_EQ(SUCCESS, ret);

  delete labelGotoTaskInfo;
  delete taskDef;
}

// test profiler_trace_task init
TEST_F(TEST_model_manager_davinci_model, success_profiler_trace_task_Init) {
  DavinciModel *model1 = nullptr;
  TaskDef *taskDef1 = new TaskDef();
  ProfilerTraceTaskInfo *profilerTraceTaskInfo1 = new ProfilerTraceTaskInfo();
  Status ret1 = profilerTraceTaskInfo1->Init(taskDef1[0], model1);
  EXPECT_EQ(PARAM_INVALID, ret1);

  delete profilerTraceTaskInfo1;
  delete taskDef1;
  delete model1;
  DavinciModel model(0, g_labelCallBack);
  TaskDef *taskDef = new TaskDef();
  taskDef->set_stream_id(0);
  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);
  LogTimeStampDef *logTimeStampDef = taskDef->mutable_log_timestamp();
  logTimeStampDef->set_logid(1);
  logTimeStampDef->set_notify(1);
  logTimeStampDef->set_flat(1);
  ProfilerTraceTaskInfo *profilerTraceTaskInfo = new ProfilerTraceTaskInfo();
  Status ret = profilerTraceTaskInfo->Init(taskDef[0], &model);
  EXPECT_EQ(SUCCESS, ret);

  taskDef->clear_log_timestamp();
  delete profilerTraceTaskInfo;
  delete taskDef;
}

TEST_F(TEST_model_manager_davinci_model, ProfilingModelSuccess) {
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);

  DavinciModel model(0, g_labelCallBack);
  model.model_id_ = 1;
  model.name_ = "test";
  model.version_ = 0x01;

  model.stream_list_.push_back(stream);

  ge::ModelData data;
  rtMallocHost(&data.model_data, 128);
  data.model_len = 128;

  ModelDef *modeldef = new ModelDef();
  auto opdef = CreateOpDesc("", "Data");
  opdef->SetInputOffset({1});
  opdef->SetOutputOffset({100});

  ge::GeTensorDesc descin(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(descin, 4);
  opdef->AddInputDesc(descin);
  ge::GeTensorDesc descout(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  ge::TensorUtils::SetSize(descout, 32);
  opdef->AddInputDesc(descout);
  opdef->SetId(0);

  model.data_op_list_.push_back(opdef);
  model.op_list_[0] = opdef;

  auto opdef1 = CreateOpDesc("", "Relu");
  opdef1->SetInputOffset({1});
  opdef1->SetOutputOffset({100});

  ge::GeTensorDesc descin1(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(descin1, 4);
  opdef1->AddInputDesc(descin1);
  ge::GeTensorDesc descout1(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  ge::TensorUtils::SetSize(descout1, 32);
  opdef1->AddInputDesc(descout1);
  opdef->SetId(1);

  model.op_list_[1] = opdef1;

  auto opdef2 = CreateOpDesc("", "Relu");
  opdef2->SetInputOffset({1});
  opdef2->SetOutputOffset({100});

  ge::GeTensorDesc descin2(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(descin2, 4);
  opdef2->AddInputDesc(descin2);
  ge::GeTensorDesc descout2(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  ge::TensorUtils::SetSize(descout2, 32);
  opdef2->AddInputDesc(descout2);
  opdef->SetId(2);

  model.op_list_[2] = opdef2;

  auto opdef3 = CreateOpDesc("", "Relu");
  opdef3->SetInputOffset({1});
  opdef3->SetOutputOffset({100});

  ge::GeTensorDesc descin3(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(descin3, 4);
  opdef3->AddInputDesc(descin3);
  ge::GeTensorDesc descout3(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  ge::TensorUtils::SetSize(descout3, 32);
  opdef3->AddInputDesc(descout3);
  opdef->SetId(3);

  model.op_list_[3] = opdef3;

  auto opdef4 = CreateOpDesc("", "Relu");
  opdef4->SetInputOffset({1});
  opdef4->SetOutputOffset({100});

  ge::GeTensorDesc descin4(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(descin4, 4);
  opdef4->AddInputDesc(descin4);
  ge::GeTensorDesc descout4(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  ge::TensorUtils::SetSize(descout4, 32);
  opdef4->AddInputDesc(descout4);
  opdef->SetId(4);

  model.op_list_[4] = opdef4;

  ge::InputData inputdata;
  ge::DataBuffer databuffer;
  databuffer.data = new char[4];
  databuffer.length = 4;
  inputdata.index = 0;
  inputdata.model_id = 1;
  inputdata.blobs.push_back(databuffer);
  // model.SinkModelProfile(&model);

  rtFreeHost(data.model_data);
  // delete stream;
  delete[](char *) databuffer.data;
  delete modeldef;
}

TEST_F(TEST_model_manager_davinci_model, success_output_list_0) {
  DavinciModel model(0, g_labelCallBack);

  uint32_t version = 0;
  uint64_t session_id = 0;
  uint32_t device_id = 0;
  uint64_t job_id = 0;
  Status ret = VarManager::Instance(session_id)->Init(version, session_id, device_id, job_id);
  EXPECT_EQ(ret, ge::SUCCESS);

  ret = model.ReturnNoOutput(1, 1);
  EXPECT_EQ(ret, ge::SUCCESS);

  VarManagerPool::Instance().Destory();
}

// test dyncbatch_distributeTask_SUCCESS
TEST_F(TEST_model_manager_davinci_model, dyncbatch_distributeTask_SUCCESS) {
  DavinciModel model(0, g_labelCallBack);

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
TEST_F(TEST_model_manager_davinci_model, success_GetOutputDescInfo_with_netoutput) {
  setenv("GE_TRAIN", "1", true);
  DavinciModel model(0, g_labelCallBack);

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

  auto computeGraph = make_shared<ge::ComputeGraph>("g");

  op_desc->SetType("NetOutput");

  auto netout_node = computeGraph->AddNode(op_desc);
  model.op_list_[0] = op_desc;

  model.output_op_list_.push_back(op_desc);
  model.output_size_list_.push_back(32);
  model.output_memory_size_list_.push_back(64);

  vector<InputOutputDescInfo> output_shapes;
  vector<uint32_t> formats;
  EXPECT_EQ(ge::SUCCESS, model.GetOutputDescInfo(output_shapes, formats));

  setenv("GE_TRAIN", "0", true);
}

TEST_F(TEST_model_manager_davinci_model, device_runtime_success_Run) {
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);

  DavinciModel model(0, g_labelCallBack);

  model.stream_list_.push_back(stream);
  auto model_def = make_shared<ge::Model>();

  auto opdef = CreateOpDesc("", "Data");

  auto computeGraph = make_shared<ge::ComputeGraph>("g");
  computeGraph->AddNode(opdef);

  model_def->SetGraph(ge::GraphUtils::CreateGraphFromComputeGraph(computeGraph));

  model.data_op_list_.push_back(opdef);

  model.data_inputer_ = new DataInputer();

  model.ModelRunStart();

  OutputData output_data;
  ge::InputData inputdata;

  ge::DataBuffer databuffer;
  databuffer.data = new char[16];
  databuffer.length = 16;

  inputdata.index = 0;
  inputdata.model_id = 1;
  inputdata.blobs.push_back(databuffer);

  model.ModelRunStop();

  delete[](char *) databuffer.data;
}

TEST_F(TEST_model_manager_davinci_model, Run_failed) {
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);

  DavinciModel model(0, g_labelCallBack);

  model.stream_list_.push_back(stream);
  auto model_def = make_shared<ge::Model>();

  auto opdef = CreateOpDesc("", "Data");

  auto computeGraph = make_shared<ge::ComputeGraph>("g");
  computeGraph->AddNode(opdef);

  model_def->SetGraph(ge::GraphUtils::CreateGraphFromComputeGraph(computeGraph));

  model.data_op_list_.push_back(opdef);

  model.data_inputer_ = new DataInputer();

  model.ModelRunStart();

  OutputData output_data;
  ge::InputData inputdata;

  ge::DataBuffer databuffer;
  databuffer.data = new char[16];
  databuffer.length = 16;

  inputdata.index = 0;
  inputdata.model_id = 1;
  inputdata.blobs.push_back(databuffer);

  model.ModelRunStop();
  delete[](char *) databuffer.data;
}

TEST_F(TEST_model_manager_davinci_model, Run_failed01) {
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);

  DavinciModel model(0, g_labelCallBack);

  model.stream_list_.push_back(stream);
  auto model_def = make_shared<ge::Model>();

  auto opdef = CreateOpDesc("", "Data");

  auto computeGraph = make_shared<ge::ComputeGraph>("g");
  computeGraph->AddNode(opdef);

  model_def->SetGraph(ge::GraphUtils::CreateGraphFromComputeGraph(computeGraph));

  model.data_op_list_.push_back(opdef);

  model.data_inputer_ = nullptr;
  model.ModelRunStart();

  model.ModelRunStop();
}

TEST_F(TEST_model_manager_davinci_model, InitTbeHandle_FE_Registered) {
  DavinciModel::tvm_bin_kernel_.clear();
  DavinciModel model(0, g_labelCallBack);
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

TEST_F(TEST_model_manager_davinci_model, InitTbeHandle_GE_Registered) {
  DavinciModel::tvm_bin_kernel_.clear();
  DavinciModel model(0, g_labelCallBack);
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
