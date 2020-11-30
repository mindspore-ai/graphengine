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

class UtestSingleOpModel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestSingleOpModel, test_init_model) {
  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  ASSERT_EQ(model.InitModel(), FAILED);
}

void ParseOpModelParamsMock(ModelHelper &model_helper, SingleOpModelParam &param) {}

TEST_F(UtestSingleOpModel, test_parse_input_node) {
  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  auto op_desc = make_shared<OpDesc>("Data", "Data");

  ASSERT_EQ(model.ParseInputNode(op_desc), PARAM_INVALID);

  vector<int64_t> shape{1, 2, 3, 4};
  vector<int64_t> offsets{16};
  GeShape ge_shape(shape);
  GeTensorDesc desc(ge_shape);
  op_desc->AddOutputDesc(desc);
  op_desc->SetOutputOffset(offsets);
  ASSERT_EQ(model.ParseInputNode(op_desc), SUCCESS);

  op_desc->AddOutputDesc(desc);
  offsets.push_back(32);
  op_desc->SetOutputOffset(offsets);
  ASSERT_EQ(model.ParseInputNode(op_desc), PARAM_INVALID);
}

TEST_F(UtestSingleOpModel, test_parse_output_node) {
  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  auto op_desc = make_shared<OpDesc>("NetOutput", "NetOutput");

  vector<int64_t> shape{1, 2, 3, 4};
  vector<int64_t> offsets{16};

  GeShape ge_shape(shape);
  GeTensorDesc desc(ge_shape);
  op_desc->AddInputDesc(desc);
  op_desc->SetInputOffset(offsets);
  op_desc->AddOutputDesc(desc);
  op_desc->SetOutputOffset(offsets);

  ASSERT_NO_THROW(model.ParseOutputNode(op_desc));
  ASSERT_NO_THROW(model.ParseOutputNode(op_desc));
}

TEST_F(UtestSingleOpModel, test_set_inputs_and_outputs) {
  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  model.input_offset_list_.push_back(0);
  model.input_sizes_.push_back(16);

  model.output_offset_list_.push_back(0);
  model.output_sizes_.push_back(16);

  std::mutex stream_mu_;
  rtStream_t stream_ = nullptr;
  SingleOp single_op(&stream_mu_, stream_);

  ASSERT_EQ(model.SetInputsAndOutputs(single_op), SUCCESS);
}

TEST_F(UtestSingleOpModel, test_build_kernel_task) {
  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  model.input_offset_list_.push_back(0);
  model.input_sizes_.push_back(16);

  model.output_offset_list_.push_back(0);
  model.output_sizes_.push_back(16);

  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("AddN", "AddN");
  vector<int64_t> shape{16, 16};
  GeShape ge_shape(shape);
  GeTensorDesc desc(ge_shape);
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);
  auto node = graph->AddNode(op_desc);
  std::mutex stream_mu_;
  rtStream_t stream_ = nullptr;
  SingleOp single_op(&stream_mu_, stream_);

  domi::KernelDef kernel_def;
  kernel_def.mutable_context()->set_kernel_type(cce::ccKernelType::TE);
  TbeOpTask *task = nullptr;
  ASSERT_EQ(model.BuildKernelTask(kernel_def, &task), UNSUPPORTED);

  kernel_def.mutable_context()->set_kernel_type(cce::ccKernelType::TE);
  ASSERT_EQ(model.BuildKernelTask(kernel_def, &task), INTERNAL_ERROR);

  model.op_list_[0] = node;

  ASSERT_EQ(model.BuildKernelTask(kernel_def, &task), PARAM_INVALID);
  ASSERT_EQ(task, nullptr);
  delete task;
}

TEST_F(UtestSingleOpModel, test_init) {
  string model_data_str = "123456789";
  SingleOpModel op_model("model", model_data_str.c_str(), model_data_str.size());
  ASSERT_EQ(op_model.Init(), FAILED);
}

TEST_F(UtestSingleOpModel, test_parse_arg_table) {
  string model_data_str = "123456789";
  SingleOpModel op_model("model", model_data_str.c_str(), model_data_str.size());

  TbeOpTask task;
  OpDescPtr op_desc;
  std::mutex stream_mu_;
  rtStream_t stream_ = nullptr;
  SingleOp op(&stream_mu_, stream_);
  op.arg_table_.resize(2);

  auto args = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(uintptr_t) * 2]);
  auto *arg_base = (uintptr_t*)args.get();
  arg_base[0] = 0x100000;
  arg_base[1] = 0x200000;
  task.SetKernelArgs(std::move(args), 16, 1, op_desc);

  op_model.model_params_.addr_mapping_[0x100000] = 1;
  op_model.ParseArgTable(&task, op);

  ASSERT_EQ(op.arg_table_[0].size(), 0);
  ASSERT_EQ(op.arg_table_[1].size(), 1);
  ASSERT_EQ(op.arg_table_[1].front(), &arg_base[0]);
}
