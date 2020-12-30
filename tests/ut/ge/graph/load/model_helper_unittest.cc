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
#include "framework/common/helper/model_helper.h"
#include "ge/model/ge_model.h"
#undef private
#undef protected

#include "proto/task.pb.h"

using namespace std;

namespace ge {
class UtestModelHelper : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestModelHelper, save_size_to_modeldef_failed)
{
  GeModelPtr ge_model = ge::MakeShared<ge::GeModel>();
  ModelHelper model_helper;
  EXPECT_EQ(ACL_ERROR_GE_MEMORY_ALLOCATION, model_helper.SaveSizeToModelDef(ge_model));
}

TEST_F(UtestModelHelper, save_size_to_modeldef)
{
  GeModelPtr ge_model = ge::MakeShared<ge::GeModel>();
  std::shared_ptr<domi::ModelTaskDef> task = ge::MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(task);
  ModelHelper model_helper;
  EXPECT_EQ(SUCCESS, model_helper.SaveSizeToModelDef(ge_model));
}
}  // namespace ge
