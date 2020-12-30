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

#include <cce/compiler_stub.h>
#include <gtest/gtest.h>

#include "common/debug/log.h"
#include "common/l2_cache_optimize.h"
#include "common/model_parser/base.h"
#include "common/properties_manager.h"
#include "common/types.h"

#define private public
#define protected public
#include "common/helper/om_file_helper.h"
#include "common/op/ge_op_utils.h"
#include "graph/load/graph_loader.h"
#include "graph/load/new_model_manager/davinci_model.h"
#include "graph/load/new_model_manager/davinci_model_parser.h"
#include "graph/load/new_model_manager/model_manager.h"
//#include "new_op_test_utils.h"
#undef private
#undef protected

using namespace std;
using namespace testing;

namespace ge {

const static std::string ENC_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

class UtestModelManagerModelManagerAicpu : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestModelManagerModelManagerAicpu, checkAicpuOptype) {
  ModelManager model_manager;
  uint32_t model_id = 0;
  std::vector<std::string> aicpu_op_list;
  std::vector<std::string> aicpu_tf_list;
  aicpu_tf_list.emplace_back("FrameworkOp");
  aicpu_tf_list.emplace_back("Unique");

  model_manager.LaunchKernelCheckAicpuOp(aicpu_op_list, aicpu_tf_list);
  // Load allow listener is null
  // EXPECT_EQ(ge::FAILED, mm.LoadModelOffline(model_id, data, nullptr, nullptr));
}

TEST_F(UtestModelManagerModelManagerAicpu, DestroyAicpuKernel) {
  ModelManager model_manager;
  uint32_t model_id = 0;
  std::vector<std::string> aicpu_op_list;
  std::vector<std::string> aicpu_tf_list;
  aicpu_tf_list.emplace_back("FrameworkOp");
  aicpu_tf_list.emplace_back("Unique");

  model_manager.DestroyAicpuKernel(0,0,0);
  // Load allow listener is null
  // EXPECT_EQ(ge::FAILED, mm.LoadModelOffline(model_id, data, nullptr, nullptr));
}

}  // namespace ge
