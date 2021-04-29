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

#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "common/util.h"
#include "runtime/mem.h"
#include "common/util.h"
#include "omg/omg_inner_types.h"

#define private public
#define protected public
#include "executor/ge_executor.h"

#include "common/auth/file_saver.h"
#include "common/debug/log.h"
#include "common/properties_manager.h"
#include "common/types.h"
#include "graph/load/graph_loader.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/task_info/kernel_task_info.h"
#include "graph/load/model_manager/task_info/kernel_ex_task_info.h"
#include "ge/common/dump/dump_properties.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/utils/graph_utils.h"
#include "proto/ge_ir.pb.h"
#undef private
#undef protected

using namespace std;
namespace ge {
class UtestGeExecutor : public testing::Test {
 protected:
  static void InitModelDefault(ge::Model &model) {
    ge::AttrUtils::SetInt(&model, ATTR_MODEL_MEMORY_SIZE, 0);
    ge::AttrUtils::SetInt(&model, ATTR_MODEL_WEIGHT_SIZE, 0);
    ge::AttrUtils::SetInt(&model, ATTR_MODEL_STREAM_NUM, 0);
    ge::AttrUtils::SetInt(&model, ATTR_MODEL_EVENT_NUM, 0);
    ge::AttrUtils::SetStr(&model, ATTR_MODEL_TARGET_TYPE, "MINI");  // domi::MINI

    auto compute_graph = std::make_shared<ge::ComputeGraph>("graph");
    auto graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    model.SetGraph(graph);
  }

  void SetUp() {
    unsetenv("FMK_SYSMODE");
    unsetenv("FMK_DUMP_PATH");
    unsetenv("FMK_USE_FUSION");
    unsetenv("DAVINCI_TIMESTAT_ENABLE");
  }
};

class DModelListener : public ge::ModelListener {
 public:
  DModelListener() {
  };
  Status OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t resultCode,
                       std::vector<ge::OutputTensorInfo> &outputs) {
    GELOGI("In Call back. OnComputeDone");
    return SUCCESS;
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
  return op_desc;
}

TEST_F(UtestGeExecutor, load_data_from_file) {
  GeExecutor ge_executor;
  ge_executor.isInit_ = true;

  string test_smap = "/tmp/" + std::to_string(getpid()) + "_maps";
  string self_smap = "/proc/" + std::to_string(getpid()) + "/maps";
  string copy_smap = "cp -f " + self_smap + " " + test_smap;
  EXPECT_EQ(system(copy_smap.c_str()), 0);

  ModelData model_data;
  EXPECT_EQ(ge_executor.LoadDataFromFile(test_smap, model_data), SUCCESS);

  EXPECT_NE(model_data.model_data, nullptr);
  delete[] static_cast<char *>(model_data.model_data);
  model_data.model_data = nullptr;

  ge_executor.isInit_ = false;
}

/*
TEST_F(UtestGeExecutor, fail_UnloadModel_model_manager_stop_unload_error) {
  uint32_t model_id = 1;
  ge::GeExecutor ge_executor;
  ge_executor.isInit_ = true;
  ge::Status ret = ge_executor.UnloadModel(model_id);
  EXPECT_EQ(ge::PARAM_INVALID, ret);

  ge_executor.isInit_ = false;
  ret = ge_executor.UnloadModel(model_id);
  EXPECT_EQ(ge::GE_EXEC_NOT_INIT, ret);
}

TEST_F(UtestGeExecutor, fail_CommandHandle_model_manager_HandleCommand_error) {
  ge::Command cmd;
  ge::GeExecutor ge_executor;
  ge::Status ret = ge_executor.CommandHandle(cmd);
  EXPECT_EQ(ge::PARAM_INVALID, ret);
}
*/
TEST_F(UtestGeExecutor, InitFeatureMapAndP2PMem_failed) {
  DavinciModel model(0, g_label_call_back);
  model.is_feature_map_mem_has_inited_ = true;
  EXPECT_EQ(model.InitFeatureMapAndP2PMem(nullptr, 0), PARAM_INVALID);
}

TEST_F(UtestGeExecutor, kernel_InitDumpTask) {
  DavinciModel model(0, g_label_call_back);
  model.om_name_ = "testom";
  model.name_ = "test";
  OpDescPtr op_desc = CreateOpDesc("test", "test");

  std::map<std::string, std::set<std::string>> model_dump_properties_map;
  std::set<std::string> s;
  model_dump_properties_map[DUMP_ALL_MODEL] = s;
  DumpProperties dp;
  dp.model_dump_properties_map_ = model_dump_properties_map;
  model.SetDumpProperties(dp);

  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = op_desc;
  kernel_task_info.InitDumpTask(0);
}

TEST_F(UtestGeExecutor, kernel_ex_InitDumpTask) {
  DavinciModel model(0, g_label_call_back);
  model.om_name_ = "testom";
  model.name_ = "test";
  OpDescPtr op_desc = CreateOpDesc("test", "test");

  std::map<std::string, std::set<std::string>> model_dump_properties_map;
  std::set<std::string> s;
  model_dump_properties_map[DUMP_ALL_MODEL] = s;
  DumpProperties dp;
  dp.model_dump_properties_map_ = model_dump_properties_map;
  model.SetDumpProperties(dp);

  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.davinci_model_ = &model;
  kernel_ex_task_info.InitDumpTask(nullptr, op_desc);
}
}