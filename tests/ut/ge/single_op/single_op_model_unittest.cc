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
#include "graph/load/new_model_manager/model_utils.h"
#include "graph/utils/graph_utils.h"
#include "runtime/rt.h"

#define protected public
#define private public
#include "single_op/single_op_model.h"
#include "single_op/task/tbe_task_builder.h"
#undef private
#undef protected

using namespace std;
using namespace testing;
using namespace ge;

class SingleOpModelTest : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(SingleOpModelTest, TestInitModel) {
  string modelDataStr = "123456789";
  SingleOpModel model("model", modelDataStr.c_str(), modelDataStr.size());
  ASSERT_EQ(model.InitModel(), FAILED);
}

void ParseOpModelParamsMock(ModelHelper &model_helper, SingleOpModelParam &param) {}

TEST_F(SingleOpModelTest, TestParseInputNode) {
  string modelDataStr = "123456789";
  SingleOpModel model("model", modelDataStr.c_str(), modelDataStr.size());
  auto op_desc = make_shared<OpDesc>("Data", "Data");

  ASSERT_EQ(model.ParseInputNode(op_desc), PARAM_INVALID);

  vector<int64_t> shape{1, 2, 3, 4};
  vector<int64_t> offsets{16};
  GeShape geShape(shape);
  GeTensorDesc desc(geShape);
  op_desc->AddOutputDesc(desc);
  op_desc->SetOutputOffset(offsets);
  ASSERT_EQ(model.ParseInputNode(op_desc), SUCCESS);

  op_desc->AddOutputDesc(desc);
  offsets.push_back(32);
  op_desc->SetOutputOffset(offsets);
  ASSERT_EQ(model.ParseInputNode(op_desc), PARAM_INVALID);
}

TEST_F(SingleOpModelTest, TestParseOutputNode) {
  string modelDataStr = "123456789";
  SingleOpModel model("model", modelDataStr.c_str(), modelDataStr.size());
  auto op_desc = make_shared<OpDesc>("NetOutput", "NetOutput");

  vector<int64_t> shape{1, 2, 3, 4};
  vector<int64_t> offsets{16};

  GeShape geShape(shape);
  GeTensorDesc desc(geShape);
  op_desc->AddInputDesc(desc);
  op_desc->SetInputOffset(offsets);
  op_desc->AddOutputDesc(desc);
  op_desc->SetOutputOffset(offsets);

  ASSERT_NO_THROW(model.ParseOutputNode(op_desc));
  ASSERT_NO_THROW(model.ParseOutputNode(op_desc));
}

TEST_F(SingleOpModelTest, TestSetInputsAndOutputs) {
  string modelDataStr = "123456789";
  SingleOpModel model("model", modelDataStr.c_str(), modelDataStr.size());
  model.input_offset_list_.push_back(0);
  model.input_sizes_.push_back(16);

  model.output_offset_list_.push_back(0);
  model.output_sizes_.push_back(16);

  SingleOp single_op;

  ASSERT_EQ(model.SetInputsAndOutputs(single_op), SUCCESS);
}

TEST_F(SingleOpModelTest, TestBuildKernelTask) {
  string modelDataStr = "123456789";
  SingleOpModel model("model", modelDataStr.c_str(), modelDataStr.size());
  model.input_offset_list_.push_back(0);
  model.input_sizes_.push_back(16);

  model.output_offset_list_.push_back(0);
  model.output_sizes_.push_back(16);

  auto op_desc = make_shared<OpDesc>("AddN", "AddN");
  vector<int64_t> shape{16, 16};
  GeShape geShape(shape);
  GeTensorDesc desc(geShape);
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);

  SingleOp single_op;
  domi::KernelDef kernel_def;
  kernel_def.mutable_context()->set_kernel_type(cce::ccKernelType::CCE_AI_CORE);
  OpTask *task = nullptr;
  ASSERT_EQ(model.BuildKernelTask(kernel_def, single_op, &task), UNSUPPORTED);

  kernel_def.mutable_context()->set_kernel_type(cce::ccKernelType::TE);
  ASSERT_EQ(model.BuildKernelTask(kernel_def, single_op, &task), INTERNAL_ERROR);

  model.op_list_[0] = op_desc;

  ASSERT_EQ(model.BuildKernelTask(kernel_def, single_op, &task), PARAM_INVALID);
  ASSERT_EQ(task, nullptr);
  delete task;
}

TEST_F(SingleOpModelTest, TestInit) {
  string modelDataStr = "123456789";
  SingleOpModel op_model("model", modelDataStr.c_str(), modelDataStr.size());
  ASSERT_EQ(op_model.Init(), FAILED);
}

TEST_F(SingleOpModelTest, TestParseArgTable) {
  string modelDataStr = "123456789";
  SingleOpModel op_model("model", modelDataStr.c_str(), modelDataStr.size());

  TbeOpTask task;
  SingleOp op;
  op.arg_table_.resize(2);

  auto *args = new uintptr_t[2];
  args[0] = 0x100000;
  args[1] = 0x200000;
  task.SetKernelArgs(args, 16, 1);

  op_model.model_params_.addr_mapping_[0x100000] = 1;
  op_model.ParseArgTable(&task, op);

  ASSERT_EQ(op.arg_table_[0].size(), 0);
  ASSERT_EQ(op.arg_table_[1].size(), 1);
  ASSERT_EQ(op.arg_table_[1].front(), &args[0]);
}
