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

#include "graph/load/new_model_manager/data_inputer.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/types.h"
#include "new_op_test_utils.h"

using namespace std;
using namespace testing;

namespace ge {

class UtestModelManagerDataInputer : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

/// InputDataWrapper
/// constructor
/// GetInput
TEST_F(UtestModelManagerDataInputer, inputdatawrapper_construct) {
  InputDataWrapper *input_data_wrapper = new InputDataWrapper();

  input_data_wrapper->GetInput();

  delete input_data_wrapper;
}

/// InputDataWrapper
/// Init func with correct input
TEST_F(UtestModelManagerDataInputer, success_inputdatawrapper_init) {
  InputDataWrapper *input_data_wrapper = new InputDataWrapper();
  ge::InputData input_data;
  ge::OutputData output_data;
  Status ret = input_data_wrapper->Init(input_data, output_data);

  EXPECT_EQ(ret, SUCCESS);

  delete input_data_wrapper;
  input_data_wrapper = NULL;
}

}  // namespace ge
