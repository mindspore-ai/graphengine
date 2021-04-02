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
#define protected public
#define private public
#include "graph/load/model_manager/model_utils.h"
#include "graph/manager/graph_var_manager.h"

using namespace std;

namespace ge {
class UtestModelUtils : public testing::Test {
 protected:
  void TearDown() {}
};

// test ModelUtils::GetVarAddr
TEST_F(UtestModelUtils, get_var_addr_hbm) {
  uint8_t test = 2;
  uint8_t *pf = &test;
  RuntimeParam runtime_param;
  runtime_param.session_id = 0;
  runtime_param.logic_var_base = 0;
  runtime_param.var_base = pf;
  runtime_param.var_size = 16;

  int64_t offset = 8;
  EXPECT_EQ(VarManager::Instance(runtime_param.session_id)->Init(0, 0, 0, 0), SUCCESS);
  EXPECT_NE(VarManager::Instance(runtime_param.session_id)->var_resource_, nullptr);
  VarManager::Instance(runtime_param.session_id)->var_resource_->var_offset_map_[offset] = RT_MEMORY_HBM;
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  uint8_t *var_addr = nullptr;
  EXPECT_EQ(ModelUtils::GetVarAddr(runtime_param, op_desc, offset, var_addr), SUCCESS);
  EXPECT_EQ(runtime_param.var_base + offset - runtime_param.logic_var_base, var_addr);
  VarManager::Instance(runtime_param.session_id)->Destory();
}

TEST_F(UtestModelUtils, get_var_addr_rdma_hbm) {
  uint8_t test = 2;
  uint8_t *pf = &test;
  RuntimeParam runtime_param;
  runtime_param.session_id = 0;
  runtime_param.logic_var_base = 0;
  runtime_param.var_base = pf;

  int64_t offset = 8;
  EXPECT_EQ(VarManager::Instance(runtime_param.session_id)->Init(0, 0, 0, 0), SUCCESS);
  EXPECT_NE(VarManager::Instance(runtime_param.session_id)->var_resource_, nullptr);
  VarManager::Instance(runtime_param.session_id)->var_resource_->var_offset_map_[offset] = RT_MEMORY_RDMA_HBM;
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  uint8_t *var_addr = nullptr;
  EXPECT_EQ(ModelUtils::GetVarAddr(runtime_param, op_desc, offset, var_addr), SUCCESS);
  EXPECT_EQ(reinterpret_cast<uint8_t *>(offset), var_addr);
  VarManager::Instance(runtime_param.session_id)->Destory();
}

TEST_F(UtestModelUtils, get_var_addr_rdma_hbm_negative_offset) {
  uint8_t test = 2;
  uint8_t *pf = &test;
  RuntimeParam runtime_param;
  runtime_param.session_id = 0;
  runtime_param.logic_var_base = 0;
  runtime_param.var_base = pf;

  int64_t offset = -1;
  EXPECT_EQ(VarManager::Instance(runtime_param.session_id)->Init(0, 0, 0, 0), SUCCESS);
  EXPECT_NE(VarManager::Instance(runtime_param.session_id)->var_resource_, nullptr);
  VarManager::Instance(runtime_param.session_id)->var_resource_->var_offset_map_[offset] = RT_MEMORY_RDMA_HBM;
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  uint8_t *var_addr = nullptr;
  EXPECT_NE(ModelUtils::GetVarAddr(runtime_param, op_desc, offset, var_addr), SUCCESS);
  VarManager::Instance(runtime_param.session_id)->Destory();
}
}  // namespace ge
