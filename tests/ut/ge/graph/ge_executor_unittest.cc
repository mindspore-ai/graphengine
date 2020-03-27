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
#include "graph/load/new_model_manager/davinci_model.h"
#include "graph/load/new_model_manager/davinci_model_parser.h"
#include "graph/load/new_model_manager/model_manager.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/utils/graph_utils.h"
#include "proto/ge_ir.pb.h"
#undef private
#undef protected

using namespace std;
using namespace ge;

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

TEST_F(UtestGeExecutor, fail_UnloadModel_model_manager_stop_unload_error) {
  uint32_t model_id = 1;
  ge::GeExecutor ge_executor;
  ge_executor.is_init_ = true;
  ge::Status ret = ge_executor.UnloadModel(model_id);
  EXPECT_EQ(ge::PARAM_INVALID, ret);

  ge_executor.is_init_ = false;
  ret = ge_executor.UnloadModel(model_id);
  EXPECT_EQ(ge::GE_EXEC_NOT_INIT, ret);
}

TEST_F(UtestGeExecutor, fail_CommandHandle_model_manager_HandleCommand_error) {
  ge::Command cmd;
  ge::GeExecutor ge_executor;
  ge::Status ret = ge_executor.CommandHandle(cmd);
  EXPECT_EQ(ge::PARAM_INVALID, ret);
}
