/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <string>
#include <map>

#define protected public
#define private public
#include "common/ge/ge_util.h"
#include "proto/ge_ir.pb.h"
#include "inc/external/ge/ge_api.h"
#include "session/session_manager.h"
#undef protected
#undef private

using namespace std;

namespace ge {
class UtestGeApi : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestGeApi, run_graph_with_stream) {
  vector<Tensor> inputs;
  vector<Tensor> outputs;
  std::map<std::string, std::string> options;
  Session session(options);
  auto ret = session.RunGraphWithStreamAsync(10, nullptr, inputs, outputs);
  ASSERT_NE(ret, SUCCESS);
  SessionManager session_manager;
  session_manager.init_flag_ = true;
  ret = session_manager.RunGraphWithStreamAsync(10, 10, nullptr, inputs, outputs);
  ASSERT_NE(ret, SUCCESS);
  InnerSession inner_session(1, options);
  inner_session.init_flag_ = true;
  ret = inner_session.RunGraphWithStreamAsync(10, nullptr, inputs, outputs);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApi, build_graph_success) {
  vector<Tensor> inputs;
  std::map<std::string, std::string> options;
  Session session(options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApi, ge_initialize) {
  std::map<std::string, std::string> options = {
    {ge::MODIFY_MIXLIST, "/mixlist.json"}
  };
  auto ret = GEInitialize(options);
  ASSERT_NE(ret, SUCCESS);
}
}  // namespace ge
