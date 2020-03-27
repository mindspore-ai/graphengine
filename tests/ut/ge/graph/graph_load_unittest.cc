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
#include "graph/load/new_model_manager/davinci_model_parser.h"
#include "graph/op_desc.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"

#define protected public
#define private public
#include "graph/load/graph_loader.h"

#include "framework/common/ge_inner_error_codes.h"
#include "graph/load/new_model_manager/model_manager.h"
#include "graph/manager/graph_manager_utils.h"
#include "model/ge_model.h"
#undef private
#undef protected

using namespace domi;
using namespace testing;
namespace ge {

class UTEST_graph_graph_load : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UTEST_graph_graph_load, load_graph_param_invalid1) {
  std::shared_ptr<GraphModelListener> graphRunListener = nullptr;
  SubGraphInfo subGraph_1;
  ge::SubGraphInfoPtr subGraphPtr1 = std::make_shared<SubGraphInfo>(subGraph_1);
  ModelIdInfo modelIdInfo;
  modelIdInfo.model_id = 1;

  GeModelPtr geModelPtr = std::make_shared<GeModel>();
  subGraphPtr1->SetGeModelPtr(geModelPtr);

  std::vector<bool> inputFlag;
  inputFlag.push_back(false);
  subGraphPtr1->SetInputFlag(inputFlag);

  ge::GraphLoader graphLoad;
  EXPECT_EQ(GE_GRAPH_PARAM_NULLPTR, graphLoad.LoadGraph(subGraphPtr1->ge_model_ptr_, graphRunListener, modelIdInfo));
  subGraphPtr1->SetModelIdInfo(modelIdInfo);
}

TEST_F(UTEST_graph_graph_load, load_graph_param_invalid2) {
  std::mutex syncRunMutex;
  std::condition_variable condition;
  std::shared_ptr<GraphModelListener> listener = std::make_shared<GraphModelListener>();
  listener->mutex_ = &syncRunMutex;
  listener->condition_ = &condition;

  SubGraphInfo subGraph_1;
  ge::SubGraphInfoPtr subGraphPtr1 = std::make_shared<SubGraphInfo>(subGraph_1);
  ModelIdInfo modelIdInfo;
  modelIdInfo.model_id = 1;

  GeModelPtr geModelPtr = std::make_shared<GeModel>();
  subGraphPtr1->SetGeModelPtr(geModelPtr);

  std::vector<bool> inputFlag;
  inputFlag.push_back(false);
  subGraphPtr1->SetInputFlag(inputFlag);

  ge::GraphLoader graphLoad;
  EXPECT_EQ(GE_GRAPH_PARAM_NULLPTR, graphLoad.LoadGraph(subGraphPtr1->ge_model_ptr_, listener, modelIdInfo));
  subGraphPtr1->SetModelIdInfo(modelIdInfo);
}
}  // namespace ge
