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

#define private public
#define protected public
#include "graph/load/model_manager/model_manager.h"
#include "common/helper/om_file_helper.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/op/ge_op_utils.h"
#include "graph/load/graph_loader.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/ops_stub.h"

using namespace std;
using namespace testing;

namespace ge {

const static std::string ENC_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

class UtestModelManagerModelManager : public testing::Test {
 protected:
  static Status LoadStub(const uint8_t *data, size_t len, Model &model) {
    InitModelDefault(model);
    return SUCCESS;
  }

  static void InitModelDefault(Model &model) {
    AttrUtils::SetInt(&model, ATTR_MODEL_MEMORY_SIZE, 0);
    AttrUtils::SetInt(&model, ATTR_MODEL_WEIGHT_SIZE, 0);
    AttrUtils::SetInt(&model, ATTR_MODEL_STREAM_NUM, 0);
    AttrUtils::SetInt(&model, ATTR_MODEL_EVENT_NUM, 0);
    AttrUtils::SetStr(&model, ATTR_MODEL_TARGET_TYPE, "MINI");  // domi::MINI

    auto computeGraph = std::make_shared<ComputeGraph>("graph");
    auto graph = GraphUtils::CreateGraphFromComputeGraph(computeGraph);
    model.SetGraph(graph);
  }

  void SetUp() {}

  void TearDown() {}

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

  void GenUnencryptModelData(ModelData &data) {
    const int model_len = 10;
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

  void GenEncryptModelData(ModelData &data) {
    const int model_len = 10;
    data.key = ENC_KEY;
    data.model_data = new uint8_t[data.model_len];
    uint8_t data_ori[model_len];
    memset(data_ori, 10, model_len);
    ModelFileHeader *header = (ModelFileHeader *)data.model_data;
    header->magic = MODEL_FILE_MAGIC_NUM;
    header->version = MODEL_VERSION;
    header->is_encrypt = ModelEncryptType::ENCRYPTED;
    header->length = 10;  // encrypt_len;
  }

  void LoadStandardModelData(ModelData &data) {
    data.model_len = 512;
    data.model_data = new uint8_t[data.model_len];
    uint8_t *model_data = reinterpret_cast<uint8_t *>(data.model_data);

    uint32_t mem_offset = sizeof(ModelFileHeader);
    ModelPartitionTable *partition_table = reinterpret_cast<ModelPartitionTable *>(model_data + mem_offset);
    partition_table->num = PARTITION_SIZE;

    mem_offset += sizeof(ModelPartitionTable) + sizeof(ModelPartitionMemInfo) * 5;
    {
      Model model;
      ComputeGraphPtr graph = make_shared<ComputeGraph>("default");
      model.SetGraph(GraphUtils::CreateGraphFromComputeGraph(graph));
      model.SetVersion(123);

      Buffer buffer;
      model.Save(buffer);
      EXPECT_TRUE(mem_offset + buffer.GetSize() < 512);
      memcpy(model_data + mem_offset, buffer.GetData(), buffer.GetSize());

      ModelPartitionMemInfo &partition_info = partition_table->partition[0];
      partition_info.type = ModelPartitionType::MODEL_DEF;
      partition_info.mem_size = buffer.GetSize();
      mem_offset += buffer.GetSize();
    }

    {
      ModelPartitionMemInfo &partition_info = partition_table->partition[1];
      partition_info.type = ModelPartitionType::WEIGHTS_DATA;
      partition_info.mem_offset = mem_offset;
      partition_info.mem_size = 0;
    }

    {
      ModelPartitionMemInfo &partition_info = partition_table->partition[2];
      partition_info.type = ModelPartitionType::TASK_INFO;
      partition_info.mem_offset = mem_offset;
      partition_info.mem_size = 0;
    }

    {
      ModelPartitionMemInfo &partition_info = partition_table->partition[3];
      partition_info.type = ModelPartitionType::TBE_KERNELS;
      partition_info.mem_offset = mem_offset;
      partition_info.mem_size = 0;
    }

    {
      ModelPartitionMemInfo &partition_info = partition_table->partition[4];
      partition_info.type = ModelPartitionType::CUST_AICPU_KERNELS;
      partition_info.mem_offset = mem_offset;
      partition_info.mem_size = 0;
    }

    EXPECT_TRUE(mem_offset < 512);
    ModelFileHeader *header = new (data.model_data) ModelFileHeader;
    header->length = mem_offset - sizeof(ModelFileHeader);
    data.model_len = mem_offset;
  }
};

class DModelListener : public ModelListener {
 public:
  DModelListener(){};
  uint32_t OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t resultCode) { return 0; }
};

TEST_F(UtestModelManagerModelManager, case_is_need_hybrid_load) {
  ModelManager mm;
  uint32_t model_id = 0;
  ComputeGraphPtr root_graph = std::make_shared<ComputeGraph>("graph");
  ge::GeRootModel model;
  EXPECT_EQ(mm.IsNeedHybridLoad(model), false);
  model.SetRootGraph(root_graph);
  EXPECT_EQ(mm.IsNeedHybridLoad(model), false);
}

TEST_F(UtestModelManagerModelManager, case_load_incorrect_param) {
  ModelManager mm;
  uint32_t model_id = 0;
  ModelData data;
  // Load allow listener is null
  EXPECT_EQ(mm.LoadModelOffline(model_id, data, nullptr, nullptr), ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
}

TEST_F(UtestModelManagerModelManager, case_load_model_len_too_short) {
  ModelManager mm;
  ModelData data;
  data.model_len = 10;
  data.model_data = (void *)&data;
  uint32_t model_id = 1;
  EXPECT_EQ(mm.LoadModelOffline(model_id, data, nullptr, nullptr), ACL_ERROR_GE_PARAM_INVALID);
  data.model_data = nullptr;
}

TEST_F(UtestModelManagerModelManager, case_load_model_len_not_match) {
  ModelManager mm;
  ModelData data;
  GenUnencryptModelData(data);
  data.model_len = sizeof(ModelFileHeader) + 1;
  uint32_t model_id = 1;
  EXPECT_EQ(mm.LoadModelOffline(model_id, data, nullptr, nullptr), ACL_ERROR_GE_PARAM_INVALID);
  delete[](uint8_t *) data.model_data;
}

TEST_F(UtestModelManagerModelManager, case_load_model_encypt_not_match) {
  ModelManager mm;
  ModelData data;
  GenUnencryptModelData(data);
  data.key = ENC_KEY;
  uint32_t model_id = 1;
  EXPECT_EQ(mm.LoadModelOffline(model_id, data, nullptr, nullptr), ACL_ERROR_GE_PARAM_INVALID);
  delete[](uint8_t *) data.model_data;
}

TEST_F(UtestModelManagerModelManager, case_load_model_encypt_type_unsupported) {
  ModelManager mm;
  ModelData data;
  GenUnencryptModelData(data);
  ModelFileHeader *header = (ModelFileHeader *)data.model_data;
  header->is_encrypt = 255;
  uint32_t model_id = 1;
  EXPECT_EQ(mm.LoadModelOffline(model_id, data, nullptr, nullptr), ACL_ERROR_GE_PARAM_INVALID);
  delete[](uint8_t *) data.model_data;
}

TEST_F(UtestModelManagerModelManager, case_load_model_data_success) {
  ModelData data;
  LoadStandardModelData(data);

  uint32_t model_id = 1;
  ModelManager mm;
  EXPECT_EQ(mm.LoadModelOffline(model_id, data, nullptr, nullptr), SUCCESS);
  delete[](uint8_t *) data.model_data;
}

/*
shared_ptr<ModelListener> LabelCallBack(new DModelListener());

// test HandleCommand
TEST_F(UtestModelManagerModelManager, command_success1) {
  ModelManager manager;
  Command cmd;

  cmd.cmd_type = "INFERENCE";
  EXPECT_EQ(PARAM_INVALID, manager.HandleCommand(cmd));

  cmd.cmd_type = "NOT SUPPORT";
  EXPECT_EQ(PARAM_INVALID, manager.HandleCommand(cmd));
}

TEST_F(UtestModelManagerModelManager, command_success2) {
  ModelManager manager;
  Command cmd;

  cmd.cmd_type = "dump";
  cmd.cmd_params.push_back("status");
  cmd.cmd_params.push_back("on");
  cmd.cmd_params.push_back("model_name");
  cmd.cmd_params.push_back("test_model");
  cmd.cmd_params.push_back("path");
  cmd.cmd_params.push_back("/test");
  cmd.cmd_params.push_back("layer");
  cmd.cmd_params.push_back("layer1");

  EXPECT_EQ(SUCCESS, manager.HandleCommand(cmd));
}

// test profile
TEST_F(UtestModelManagerModelManager, command_profile_success) {
  ModelManager manager;
  Command cmd;
  cmd.cmd_type = "profile";

  cmd.cmd_params.push_back("ome");
  cmd.cmd_params.push_back("on");
  EXPECT_EQ(SUCCESS, manager.HandleCommand(cmd));
  bool ome_profile_on = PropertiesManager::Instance().GetPropertyValue(OME_PROFILE) == "1";
  EXPECT_EQ(true, ome_profile_on);

  cmd.cmd_params.clear();
  cmd.cmd_params.push_back("ome");
  cmd.cmd_params.push_back("off");
  EXPECT_EQ(SUCCESS, manager.HandleCommand(cmd));
  ome_profile_on = PropertiesManager::Instance().GetPropertyValue(OME_PROFILE) == "1";
  EXPECT_FALSE(ome_profile_on);

  cmd.cmd_params.clear();
  cmd.cmd_params.push_back("cce");
  cmd.cmd_params.push_back("on");
  EXPECT_EQ(SUCCESS, manager.HandleCommand(cmd));
  bool cce_profile_on = PropertiesManager::Instance().GetPropertyValue(CCE_PROFILE) == "1";
  EXPECT_EQ(true, cce_profile_on);

  cmd.cmd_params.clear();
  cmd.cmd_params.push_back("cce");
  cmd.cmd_params.push_back("off");
  EXPECT_EQ(SUCCESS, manager.HandleCommand(cmd));
  cce_profile_on = PropertiesManager::Instance().GetPropertyValue(CCE_PROFILE) == "1";
  EXPECT_FALSE(cce_profile_on);

  cmd.cmd_params.clear();
  cmd.cmd_params.push_back("runtime");
  cmd.cmd_params.push_back("on");
  EXPECT_EQ(SUCCESS, manager.HandleCommand(cmd));
  bool rts_profile_on = PropertiesManager::Instance().GetPropertyValue(RTS_PROFILE) == "1";
  EXPECT_EQ(true, rts_profile_on);

  cmd.cmd_params.clear();
  cmd.cmd_params.push_back("runtime");
  cmd.cmd_params.push_back("off");
  EXPECT_EQ(SUCCESS, manager.HandleCommand(cmd));
  rts_profile_on = PropertiesManager::Instance().GetPropertyValue(RTS_PROFILE) == "1";
  EXPECT_FALSE(rts_profile_on);

  cmd.cmd_params.clear();
  cmd.cmd_params.push_back("profiler_jobctx");
  cmd.cmd_params.push_back("jobctx");
  EXPECT_EQ(SUCCESS, manager.HandleCommand(cmd));
  EXPECT_EQ("jobctx", PropertiesManager::Instance().GetPropertyValue(PROFILER_JOBCTX));

  cmd.cmd_params.clear();
  cmd.cmd_params.push_back("profiler_target_path");
  cmd.cmd_params.push_back("/test/target");
  EXPECT_EQ(SUCCESS, manager.HandleCommand(cmd));
  EXPECT_EQ("/test/target", PropertiesManager::Instance().GetPropertyValue(PROFILER_TARGET_PATH));

  cmd.cmd_params.clear();
  cmd.cmd_params.push_back("RTS_PATH");
  cmd.cmd_params.push_back("/test/rts_path");
  EXPECT_EQ(SUCCESS, manager.HandleCommand(cmd));
  EXPECT_EQ("/test/rts_path", PropertiesManager::Instance().GetPropertyValue(RTS_PROFILE_PATH));
}

// test acl profiling
TEST_F(UtestModelManagerModelManager, command_profiling) {
  ModelManager manager;
  Command cmd;
  cmd.cmd_type = "profiling";

  cmd.cmd_params.push_back("config");
  cmd.cmd_params.push_back("on");
  EXPECT_EQ(SUCCESS, manager.HandleCommand(cmd));
}

TEST_F(UtestModelManagerModelManager, command_profile_failed) {
  ModelManager manager;
  Command cmd;
  cmd.cmd_type = "profile";

  cmd.cmd_params.push_back("ome");

  EXPECT_EQ(PARAM_INVALID, manager.HandleCommand(cmd));
}

// test Start
TEST_F(UtestModelManagerModelManager, start_fail) {
  ModelManager manager;
  manager.model_map_[2] = nullptr;
  EXPECT_EQ(PARAM_INVALID, manager.Start(2));
}

// test GetMaxUsedMemory
TEST_F(UtestModelManagerModelManager, get_max_used_memory_fail) {
  ModelManager manager;
  uint64_t max_size = 0;
  manager.model_map_[2] = nullptr;
  EXPECT_EQ(PARAM_INVALID, manager.GetMaxUsedMemory(2, max_size));
}

// test GetInputOutputDescInfo
TEST_F(UtestModelManagerModelManager, get_input_output_desc_info_fail) {
  ModelManager manager;
  manager.model_map_[2] = nullptr;
  vector<InputOutputDescInfo> input_shape;
  vector<InputOutputDescInfo> output_shape;
  EXPECT_EQ(PARAM_INVALID, manager.GetInputOutputDescInfo(2, input_shape, output_shape));
}


*//*
// test GetInputOutputDescInfo fail
TEST_F(UtestModelManagerModelManager, get_input_output_desc_info_zero_copy_fail) {
  ModelManager manager;
  manager.model_map_[2] = nullptr;
  vector<InputOutputDescInfo> input_shape;
  vector<InputOutputDescInfo> output_shape;
  EXPECT_EQ(PARAM_INVALID, manager.GetInputOutputDescInfoForZeroCopy(2, input_shape, output_shape));
}
*//*

// test Stop
TEST_F(UtestModelManagerModelManager, stop_fail) {
  ModelManager manager;
  manager.model_map_[2] = nullptr;
  EXPECT_EQ(PARAM_INVALID, manager.Stop(2));
}

// build input_data
TEST_F(UtestModelManagerModelManager, check_data_len_success) {
  shared_ptr<ModelListener> g_label_call_back(new DModelListener());
  DavinciModel model(0, g_label_call_back);
  ModelManager model_manager;
  InputData input_data;
  DataBuffer data_buffer;
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
}*/
// test DataInputTensor
TEST_F(UtestModelManagerModelManager, test_data_input_tensor) {
  shared_ptr<ModelListener> g_label_call_back(nullptr);
  auto model = std::make_shared<DavinciModel>(0, g_label_call_back);
  ModelManager mm;
  uint32_t model_id = 1;
  mm.model_map_[1] = model;
  mm.hybrid_model_map_[1] = std::make_shared<hybrid::HybridDavinciModel>();

  ge::Tensor input_tensor;
  vector<ge::Tensor> inputs;
  inputs.emplace_back(input_tensor);
  auto ret = mm.DataInputTensor(model_id,inputs);
  EXPECT_EQ(PARAM_INVALID, ret);    // HybridDavinciModel::impl_ is null.
}
}  // namespace ge
