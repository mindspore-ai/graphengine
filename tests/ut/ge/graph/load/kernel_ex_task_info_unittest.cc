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

#include "graph/load/model_manager/davinci_model.h"

#include "graph/load/model_manager/task_info/kernel_ex_task_info.h"
#include "cce/aicpu_engine_struct.h"

namespace ge {
extern OpDescPtr CreateOpDesc(string name, string type);

class UtestKernelExTaskInfo : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

// test kernel_ex_task_Release
TEST_F(UtestKernelExTaskInfo, success_kernel_ex_task_init) {
  domi::TaskDef task_def;
  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, nullptr), PARAM_INVALID);

  DavinciModel model(0, nullptr);
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model), FAILED);

  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);
  domi::KernelExDef *kernel_ex_def = task_def.mutable_kernel_ex();
  kernel_ex_def->set_op_index(1);
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model), INTERNAL_ERROR);

  kernel_ex_def->clear_op_index();
  kernel_ex_def->set_op_index(0);
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model), FAILED);

  kernel_ex_def->set_task_info("KernelEx");
  kernel_ex_def->set_task_info_size(1);
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model), FAILED);


  constexpr uint32_t arg_size = sizeof(STR_FWK_OP_KERNEL);
  string value1(arg_size, 'a');
  kernel_ex_def->set_args_size(arg_size);
  kernel_ex_def->set_args(value1);
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model), FAILED);


  task_def.clear_kernel_ex();
}

// test kernel_ex_task_Release
TEST_F(UtestKernelExTaskInfo, success_kernel_ex_task_release) {
  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_EQ(kernel_ex_task_info.Release(), SUCCESS);

  kernel_ex_task_info.kernel_buf_ = nullptr;
  rtMalloc(&kernel_ex_task_info.input_output_addr_, 64, RT_MEMORY_HBM);
  EXPECT_EQ(kernel_ex_task_info.Release(), SUCCESS);

  kernel_ex_task_info.input_output_addr_ = nullptr;
  rtMalloc(&kernel_ex_task_info.kernel_buf_, 64, RT_MEMORY_HBM);
  EXPECT_EQ(kernel_ex_task_info.Release(), SUCCESS);

  rtMalloc(&kernel_ex_task_info.kernel_buf_, 64, RT_MEMORY_HBM);
  rtMalloc(&kernel_ex_task_info.input_output_addr_, 64, RT_MEMORY_HBM);
  EXPECT_EQ(kernel_ex_task_info.Release(), SUCCESS);
}

// test kernel_ex_task_Release
TEST_F(UtestKernelExTaskInfo, success_kernel_ex_task_info_copy) {
  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_size = 10240;
  model.runtime_param_.mem_base = new uint8_t[model.runtime_param_.mem_size];

  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);

  domi::TaskDef task_def;
  KernelExTaskInfo kernel_ex_task_info;

  domi::KernelExDef *kernel_ex_def = task_def.mutable_kernel_ex();
  kernel_ex_def->set_task_info_size(150);
  kernel_ex_def->set_op_index(0);
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");

  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model), FAILED);  // workspace empty.

  model.op_list_[0]->SetWorkspace({1008});   // offset
  model.op_list_[0]->SetWorkspaceBytes({0});      // length
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model), FAILED);  // workspace addr is null.

  model.op_list_[0]->SetWorkspace({1208});   // offset
  model.op_list_[0]->SetWorkspaceBytes({10});     // length
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model), FAILED);  // workspace addr is small.

  model.op_list_[0]->SetWorkspace({1308});   // offset
  model.op_list_[0]->SetWorkspaceBytes({150});    // length
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model), SUCCESS);

  task_def.clear_kernel_ex();
  delete [] model.runtime_param_.mem_base;
  model.runtime_param_.mem_base = nullptr;
}

TEST_F(UtestKernelExTaskInfo, kernel_ex_task_info_calculate_args) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;
  domi::KernelExDef *kernel_ex_def = task_def.mutable_kernel_ex();
  kernel_ex_def->set_op_index(0);
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");

  AttrUtils::SetStr(model.op_list_[0], ATTR_DYNAMIC_SHAPE_FIXED_ADDR, "Hello Mr Tree");

  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_EQ(kernel_ex_task_info.CalculateArgs(task_def, &model), FAILED);
}

TEST_F(UtestKernelExTaskInfo, kernel_ex_task_ext_info) {
  const string ext_info = {1, 1, 1, 1, 0, 0, 0, 0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
}

TEST_F(UtestKernelExTaskInfo, parse_update_addr) {
  const string ext_info = {3,0,0,0,4,0,0,0,4,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
}

TEST_F(UtestKernelExTaskInfo, parse_topic_type_success_1) {
  const string ext_info = {7,0,0,0,4,0,0,0,0,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
}

TEST_F(UtestKernelExTaskInfo, parse_topic_type_success_2) {
  const string ext_info = {7,0,0,0,4,0,0,0,1,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
}

TEST_F(UtestKernelExTaskInfo, parse_topic_type_success_3) {
  const string ext_info = {7,0,0,0,4,0,0,0,2,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
}

TEST_F(UtestKernelExTaskInfo, parse_topic_type_success_4) {
  const string ext_info = {7,0,0,0,4,0,0,0,3,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
}

TEST_F(UtestKernelExTaskInfo, parse_topic_type_failed_1) {
  const string ext_info = {7,0,0,0,4,0,0,0,4,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_NE(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
}

TEST_F(UtestKernelExTaskInfo, parse_topic_type_failed_2) {
  const string ext_info = {7,0,0,0,2,0,0,0,2,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_NE(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
}
}  // namespace ge
