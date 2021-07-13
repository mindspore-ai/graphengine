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
#include <mutex>
#include <thread>
#include <vector>

#include "common/debug/log.h"
#include "common/helper/model_helper.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/op_desc.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"

#define protected public
#define private public
#include "graph/load/graph_loader.h"

#include "framework/common/ge_inner_error_codes.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/manager/graph_manager_utils.h"
#include "common/model/ge_model.h"
#undef private
#undef protected

using namespace testing;
namespace ge {

class UtestGraphGraphLoad : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGraphGraphLoad, load_graph_param_invalid1) {
  std::shared_ptr<GraphModelListener> graph_run_listener = nullptr;
  SubGraphInfo sub_graph1;
  ge::SubGraphInfoPtr sub_graph_ptr1 = std::make_shared<SubGraphInfo>(sub_graph1);
  ModelIdInfo model_Id_info;
  model_Id_info.model_id = 1;

  GeModelPtr ge_model_ptr = std::make_shared<GeModel>();
  sub_graph_ptr1->SetGeModelPtr(ge_model_ptr);

  std::vector<bool> input_flag;
  input_flag.push_back(false);
  sub_graph_ptr1->SetInputFlag(input_flag);

  ge::GraphLoader graph_load;
  EXPECT_EQ(GE_GRAPH_PARAM_NULLPTR, graph_load.LoadGraph(sub_graph_ptr1->ge_model_ptr_, graph_run_listener, model_Id_info));
  sub_graph_ptr1->SetModelIdInfo(model_Id_info);
}

TEST_F(UtestGraphGraphLoad, load_graph_param_invalid2) {
  std::mutex sync_run_mutex;
  std::condition_variable condition;
  std::shared_ptr<GraphModelListener> listener = std::make_shared<GraphModelListener>(sync_run_mutex, condition);

  SubGraphInfo sub_graph1;
  ge::SubGraphInfoPtr sub_graph_ptr1 = std::make_shared<SubGraphInfo>(sub_graph1);
  ModelIdInfo model_Id_info;
  model_Id_info.model_id = 1;

  GeModelPtr ge_model_ptr = std::make_shared<GeModel>();
  sub_graph_ptr1->SetGeModelPtr(ge_model_ptr);

  std::vector<bool> input_flag;
  input_flag.push_back(false);
  sub_graph_ptr1->SetInputFlag(input_flag);

  ge::GraphLoader graph_load;
  EXPECT_EQ(GE_GRAPH_PARAM_NULLPTR, graph_load.LoadGraph(sub_graph_ptr1->ge_model_ptr_, listener, model_Id_info));
  sub_graph_ptr1->SetModelIdInfo(model_Id_info);
}
}  // namespace ge
