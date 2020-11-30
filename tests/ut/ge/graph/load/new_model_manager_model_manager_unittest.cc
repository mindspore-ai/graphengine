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

#include <cce/compiler_stub.h>
#include "common/debug/log.h"
#include "common/model_parser/base.h"
#include "common/properties_manager.h"
#include "common/types.h"
#include "common/l2_cache_optimize.h"

#define private public
#define protected public
#include "graph/load/new_model_manager/model_manager.h"

#include "common/helper/om_file_helper.h"
#include "common/op/ge_op_utils.h"
#include "graph/load/graph_loader.h"
#include "graph/load/new_model_manager/davinci_model.h"
#include "graph/load/new_model_manager/davinci_model_parser.h"
#include "new_op_test_utils.h"
#undef private
#undef protected

using namespace std;
using namespace testing;

namespace ge {

const static std::string ENC_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

class UtestModelManagerModelManager : public testing::Test {
 protected:
  static Status LoadStub(const uint8_t *data, size_t len, ge::Model &model) {
    InitModelDefault(model);
    return ge::SUCCESS;
  }

  static void InitModelDefault(ge::Model &model) {
    ge::AttrUtils::SetInt(&model, ge::ATTR_MODEL_MEMORY_SIZE, 0);
    ge::AttrUtils::SetInt(&model, ge::ATTR_MODEL_WEIGHT_SIZE, 0);
    ge::AttrUtils::SetInt(&model, ge::ATTR_MODEL_STREAM_NUM, 0);
    ge::AttrUtils::SetInt(&model, ge::ATTR_MODEL_EVENT_NUM, 0);
    ge::AttrUtils::SetStr(&model, ge::ATTR_MODEL_TARGET_TYPE, "MINI");  // domi::MINI

    auto computeGraph = std::make_shared<ge::ComputeGraph>("graph");
    auto graph = ge::GraphUtils::CreateGraphFromComputeGraph(computeGraph);
    model.SetGraph(graph);
  }

  void SetUp() {}

  void TearDown() {}

  void GenUnencryptModelData(ge::ModelData &data) {
    const int model_len = 10;
    data.key;
    data.model_len = sizeof(ModelFileHeader) + model_len;
    data.model_data = new uint8_t[data.model_len];
    memset((uint8_t *)data.model_data + sizeof(ModelFileHeader), 10, model_len);

    ModelFileHeader *header = (ModelFileHeader *)data.model_data;
    header->magic = MODEL_FILE_MAGIC_NUM;
    header->version = MODEL_VERSION;
    header->is_encrypt = ModelEncryptType::UNENCRYPTED;
    header->length = model_len;
    header->is_checksum = ModelCheckType::CHECK;
  }

  void GenEncryptModelData(ge::ModelData &data) {
    const int model_len = 10;
    data.key = ENC_KEY;
    data.model_data = new uint8_t[data.model_len];
    uint8_t data_ori[model_len];
    memset(data_ori, 10, model_len);
    uint32_t out_len;
    ModelFileHeader *header = (ModelFileHeader *)data.model_data;
    header->magic = MODEL_FILE_MAGIC_NUM;
    header->version = MODEL_VERSION;
    header->is_encrypt = ModelEncryptType::ENCRYPTED;
    header->length = 10;  // encrypt_len;
  }

  void LoadStandardModelData(ge::ModelData &data) {
    static const std::string STANDARD_MODEL_DATA_PATH =
        "llt/framework/domi/ut/ome/test/data/standard_partition_model.txt";
    ge::proto::ModelDef model_def;
    ReadProtoFromText(STANDARD_MODEL_DATA_PATH.c_str(), &model_def);

    data.model_len = model_def.ByteSizeLong();
    data.model_data = new uint8_t[data.model_len];
    model_def.SerializePartialToArray(data.model_data, data.model_len);
  }
};

class DModelListener : public ge::ModelListener {
 public:
  DModelListener(){};
  uint32_t OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t resultCode) { return 0; }
};

shared_ptr<ModelListener> UTEST_CALL_BACK_FUN(new DModelListener());

TEST_F(UtestModelManagerModelManager, case_load_incorrect_param) {
  ModelManager mm;
  uint32_t model_id = 0;
  ge::ModelData model;
  EXPECT_EQ(ge::FAILED, mm.LoadModelOffline(model_id, model, nullptr, nullptr));
  ge::ModelData data;
  // Load allow listener is null
  EXPECT_EQ(ge::FAILED, mm.LoadModelOffline(model_id, data, nullptr, nullptr));
}

TEST_F(UtestModelManagerModelManager, case_load_model_len_too_short) {
  ModelManager mm;
  ge::ModelData data;
  data.model_len = 10;
  uint32_t model_id = 1;
  EXPECT_EQ(ge::FAILED, mm.LoadModelOffline(model_id, data, UTEST_CALL_BACK_FUN, nullptr));
}

TEST_F(UtestModelManagerModelManager, case_load_model_len_not_match) {
  ModelManager mm;
  ge::ModelData data;
  GenUnencryptModelData(data);
  data.model_len = sizeof(ModelFileHeader) + 1;
  uint32_t model_id = 1;
  EXPECT_EQ(ge::FAILED, mm.LoadModelOffline(model_id, data, UTEST_CALL_BACK_FUN, nullptr));
  delete[](uint8_t *) data.model_data;
}

TEST_F(UtestModelManagerModelManager, case_load_model_encypt_not_match) {
  ModelManager mm;
  ge::ModelData data;
  GenUnencryptModelData(data);
  data.key = ENC_KEY;
  uint32_t model_id = 1;
  EXPECT_EQ(ge::PARAM_INVALID, mm.LoadModelOffline(model_id, data, UTEST_CALL_BACK_FUN, nullptr));
  delete[](uint8_t *) data.model_data;
}

TEST_F(UtestModelManagerModelManager, case_load_model_encypt_type_unsupported) {
  ModelManager mm;
  ge::ModelData data;
  GenUnencryptModelData(data);
  ModelFileHeader *header = (ModelFileHeader *)data.model_data;
  header->is_encrypt = 255;
  uint32_t model_id = 1;
  EXPECT_EQ(ge::FAILED, mm.LoadModelOffline(model_id, data, UTEST_CALL_BACK_FUN, nullptr));
  delete[](uint8_t *) data.model_data;
}

shared_ptr<ModelListener> LabelCallBack(new DModelListener());

// test HandleCommand
TEST_F(UtestModelManagerModelManager, command_success1) {
  ModelManager manager;
  ge::Command cmd;

  cmd.cmd_type = "INFERENCE";
  EXPECT_EQ(ge::PARAM_INVALID, manager.HandleCommand(cmd));

  cmd.cmd_type = "NOT SUPPORT";
  EXPECT_EQ(ge::PARAM_INVALID, manager.HandleCommand(cmd));
}

TEST_F(UtestModelManagerModelManager, command_success2) {
  ModelManager manager;
  ge::Command cmd;

  cmd.cmd_type = "dump";
  cmd.cmd_params.push_back("status");
  cmd.cmd_params.push_back("on");
  cmd.cmd_params.push_back("model_name");
  cmd.cmd_params.push_back("test_model");
  cmd.cmd_params.push_back("path");
  cmd.cmd_params.push_back("/test");
  cmd.cmd_params.push_back("layer");
  cmd.cmd_params.push_back("layer1");

  EXPECT_EQ(ge::SUCCESS, manager.HandleCommand(cmd));
}

// test profile
TEST_F(UtestModelManagerModelManager, command_profile_success) {
  ModelManager manager;
  ge::Command cmd;
  cmd.cmd_type = "profile";

  cmd.cmd_params.push_back("ome");
  cmd.cmd_params.push_back("on");
  EXPECT_EQ(ge::SUCCESS, manager.HandleCommand(cmd));
  bool ome_profile_on = PropertiesManager::Instance().GetPropertyValue(OME_PROFILE) == "1";
  EXPECT_EQ(true, ome_profile_on);

  cmd.cmd_params.clear();
  cmd.cmd_params.push_back("ome");
  cmd.cmd_params.push_back("off");
  EXPECT_EQ(ge::SUCCESS, manager.HandleCommand(cmd));
  ome_profile_on = PropertiesManager::Instance().GetPropertyValue(OME_PROFILE) == "1";
  EXPECT_FALSE(ome_profile_on);

  cmd.cmd_params.clear();
  cmd.cmd_params.push_back("cce");
  cmd.cmd_params.push_back("on");
  EXPECT_EQ(ge::SUCCESS, manager.HandleCommand(cmd));
  bool cce_profile_on = PropertiesManager::Instance().GetPropertyValue(CCE_PROFILE) == "1";
  EXPECT_EQ(true, cce_profile_on);

  cmd.cmd_params.clear();
  cmd.cmd_params.push_back("cce");
  cmd.cmd_params.push_back("off");
  EXPECT_EQ(ge::SUCCESS, manager.HandleCommand(cmd));
  cce_profile_on = PropertiesManager::Instance().GetPropertyValue(CCE_PROFILE) == "1";
  EXPECT_FALSE(cce_profile_on);

  cmd.cmd_params.clear();
  cmd.cmd_params.push_back("runtime");
  cmd.cmd_params.push_back("on");
  EXPECT_EQ(ge::SUCCESS, manager.HandleCommand(cmd));
  bool rts_profile_on = PropertiesManager::Instance().GetPropertyValue(RTS_PROFILE) == "1";
  EXPECT_EQ(true, rts_profile_on);

  cmd.cmd_params.clear();
  cmd.cmd_params.push_back("runtime");
  cmd.cmd_params.push_back("off");
  EXPECT_EQ(ge::SUCCESS, manager.HandleCommand(cmd));
  rts_profile_on = PropertiesManager::Instance().GetPropertyValue(RTS_PROFILE) == "1";
  EXPECT_FALSE(rts_profile_on);

  cmd.cmd_params.clear();
  cmd.cmd_params.push_back("profiler_jobctx");
  cmd.cmd_params.push_back("jobctx");
  EXPECT_EQ(ge::SUCCESS, manager.HandleCommand(cmd));
  EXPECT_EQ("jobctx", PropertiesManager::Instance().GetPropertyValue(PROFILER_JOBCTX));

  cmd.cmd_params.clear();
  cmd.cmd_params.push_back("profiler_target_path");
  cmd.cmd_params.push_back("/test/target");
  EXPECT_EQ(ge::SUCCESS, manager.HandleCommand(cmd));
  EXPECT_EQ("/test/target", PropertiesManager::Instance().GetPropertyValue(PROFILER_TARGET_PATH));

  cmd.cmd_params.clear();
  cmd.cmd_params.push_back("RTS_PATH");
  cmd.cmd_params.push_back("/test/rts_path");
  EXPECT_EQ(ge::SUCCESS, manager.HandleCommand(cmd));
  EXPECT_EQ("/test/rts_path", PropertiesManager::Instance().GetPropertyValue(RTS_PROFILE_PATH));
}

// test acl profiling
TEST_F(UtestModelManagerModelManager, command_profiling) {
  ModelManager manager;
  ge::Command cmd;
  cmd.cmd_type = "profiling";

  cmd.cmd_params.push_back("config");
  cmd.cmd_params.push_back("on");
  EXPECT_EQ(ge::SUCCESS, manager.HandleCommand(cmd));
}

TEST_F(UtestModelManagerModelManager, command_profile_failed) {
  ModelManager manager;
  ge::Command cmd;
  cmd.cmd_type = "profile";

  cmd.cmd_params.push_back("ome");

  EXPECT_EQ(ge::PARAM_INVALID, manager.HandleCommand(cmd));
}

// test Start
TEST_F(UtestModelManagerModelManager, start_fail) {
  ModelManager manager;
  manager.model_map_[2] = nullptr;
  EXPECT_EQ(ge::PARAM_INVALID, manager.Start(2));
}

// test GetMaxUsedMemory
TEST_F(UtestModelManagerModelManager, get_max_used_memory_fail) {
  ModelManager manager;
  uint64_t max_size = 0;
  manager.model_map_[2] = nullptr;
  EXPECT_EQ(ge::PARAM_INVALID, manager.GetMaxUsedMemory(2, max_size));
}

// test GetInputOutputDescInfo
TEST_F(UtestModelManagerModelManager, get_input_output_desc_info_fail) {
  ModelManager manager;
  manager.model_map_[2] = nullptr;
  vector<InputOutputDescInfo> input_shape;
  vector<InputOutputDescInfo> output_shape;
  EXPECT_EQ(ge::PARAM_INVALID, manager.GetInputOutputDescInfo(2, input_shape, output_shape));
}


/*
// test GetInputOutputDescInfo fail
TEST_F(UtestModelManagerModelManager, get_input_output_desc_info_zero_copy_fail) {
  ModelManager manager;
  manager.model_map_[2] = nullptr;
  vector<InputOutputDescInfo> input_shape;
  vector<InputOutputDescInfo> output_shape;
  EXPECT_EQ(ge::PARAM_INVALID, manager.GetInputOutputDescInfoForZeroCopy(2, input_shape, output_shape));
}
*/

// test Stop
TEST_F(UtestModelManagerModelManager, stop_fail) {
  ModelManager manager;
  manager.model_map_[2] = nullptr;
  EXPECT_EQ(ge::PARAM_INVALID, manager.Stop(2));
}

// build input_data
TEST_F(UtestModelManagerModelManager, check_data_len_success) {
  shared_ptr<ModelListener> g_label_call_back(new DModelListener());
  DavinciModel model(0, g_label_call_back);
  ModelManager model_manager;
  ge::InputData input_data;
  ge::DataBuffer data_buffer;
  data_buffer.data = new char[51200];
  data_buffer.length = 51200;
  input_data.index = 0;
  input_data.model_id = 1;
  input_data.blobs.push_back(data_buffer);
  delete[](char *) data_buffer.data;
}

// test LoadModeldef
TEST_F(UtestModelManagerModelManager, destroy_aicpu_session) {
  ModelManager manager;
  manager.DestroyAicpuSession(0);

  manager.sess_ids_.insert(0);
  manager.DestroyAicpuSession(0);
}

}  // namespace ge
