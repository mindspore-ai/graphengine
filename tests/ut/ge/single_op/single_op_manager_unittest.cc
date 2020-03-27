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
#include <vector>

#include "cce/taskdown_common.hpp"
#include "runtime/rt.h"

#define protected public
#define private public
#include "single_op/single_op_manager.h"
#undef private
#undef protected

using namespace std;
using namespace testing;
using namespace ge;

class SingleOpManagerTest : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(SingleOpManagerTest, TestGetResource) {
  uintptr_t resource_id = 0x1;
  auto &instance = SingleOpManager::GetInstance();
  ASSERT_EQ(instance.TryGetResource(resource_id), nullptr);
  ASSERT_NE(instance.GetResource(resource_id), nullptr);
}

TEST_F(SingleOpManagerTest, TestGetOpFromModel) {
  auto stream = (rtStream_t)0x1;
  uintptr_t resource_id = 0x1;
  auto &instance = SingleOpManager::GetInstance();

  SingleOp *single_op = nullptr;
  ModelData model_data;
  string modelStr = "123456789";
  model_data.model_data = (void *)modelStr.c_str();
  model_data.model_len = modelStr.size();

  ASSERT_EQ(instance.GetOpFromModel("model", model_data, stream, &single_op), FAILED);
  ASSERT_EQ(instance.GetResource(resource_id)->GetOperator(model_data.model_data), nullptr);
}

TEST_F(SingleOpManagerTest, TestRelesaseResource) {
  auto stream = (rtStream_t)0x99;
  auto &instance = SingleOpManager::GetInstance();

  ASSERT_EQ(instance.ReleaseResource(stream), SUCCESS);
  instance.GetResource(0x99);
  ASSERT_EQ(instance.ReleaseResource(stream), SUCCESS);
}

TEST_F(SingleOpManagerTest, TestGetOpFromModelWithNullStream) {
  void *stream = nullptr;

  SingleOp *single_op = nullptr;
  ModelData model_data;
  string modelStr = "123456789";
  model_data.model_data = (void *)modelStr.c_str();
  model_data.model_len = modelStr.size();
  auto &instance = SingleOpManager::GetInstance();

  ASSERT_EQ(instance.GetOpFromModel("model", model_data, stream, &single_op), FAILED);
}

TEST_F(SingleOpManagerTest, GetResourceFailed) {
  auto stream = (rtStream_t)0x1;

  SingleOp *single_op = nullptr;
  ModelData model_data;
  string modelStr = "123456789";
  model_data.model_data = (void *)modelStr.c_str();
  model_data.model_len = modelStr.size();
  auto &instance = SingleOpManager::GetInstance();

  ASSERT_EQ(instance.GetOpFromModel("model", model_data, stream, &single_op), FAILED);
}