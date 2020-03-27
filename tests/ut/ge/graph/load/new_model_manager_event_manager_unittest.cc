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

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/types.h"

#define private public
#include "graph/manager/model_manager/event_manager.h"
#undef private

using namespace domi;
using namespace std;
using namespace testing;

class UTEST_model_manager_event_manager : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

// test repeat initialize
TEST_F(UTEST_model_manager_event_manager, repeat_initialization) {
  ge::EventManager event_manager;
  size_t event_num = 1;
  event_manager.Init(event_num);
  Status ret = event_manager.Init(event_num);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UTEST_model_manager_event_manager, call_EventRecord_normal) {
  ge::EventManager event_manager;
  size_t event_num = 1;
  Status ret = event_manager.Init(event_num);
  EXPECT_EQ(SUCCESS, ret);
  EXPECT_NE(event_manager.event_list_.size(), 0);

  ret = event_manager.EventRecord(0, NULL);
  EXPECT_EQ(SUCCESS, ret);
}

// test load EventRecore when uninited
TEST_F(UTEST_model_manager_event_manager, call_EventRecord_while_unInited) {
  ge::EventManager event_manager;
  Status ret = event_manager.EventRecord(1, NULL);
  EXPECT_EQ(ge::INTERNAL_ERROR, ret);
}

// test with invalid param when load EventRecord
TEST_F(UTEST_model_manager_event_manager, call_EventRecord_with_invalid_param) {
  ge::EventManager event_manager;
  Status ret = event_manager.Init(1);
  EXPECT_EQ(SUCCESS, ret);
  ret = event_manager.EventRecord(1, NULL);
  EXPECT_EQ(ge::PARAM_INVALID, ret);
}

// test load EventElapsedTime when uninited
TEST_F(UTEST_model_manager_event_manager, call_EventElapsedTime_while_unInited) {
  ge::EventManager event_manager;
  float time = .0f;
  Status ret = event_manager.EventElapsedTime(1, 2, time);
  EXPECT_EQ(ge::INTERNAL_ERROR, ret);
}

// test with invalid param when load EventElapsedTime
TEST_F(UTEST_model_manager_event_manager, call_EventElapsedTime_with_invalid_param) {
  ge::EventManager *event_manager = new ge::EventManager;
  size_t event_num = 2;
  Status ret = event_manager->Init(event_num);
  EXPECT_EQ(SUCCESS, ret);
  float time = .0f;

  // normal load
  ret = event_manager->EventElapsedTime(0, 1, time);
  EXPECT_EQ(SUCCESS, ret);

  // startevent_idx overstep boundary
  ret = event_manager->EventElapsedTime(2, 1, time);
  EXPECT_EQ(ge::PARAM_INVALID, ret);

  // stopevent_idx overstep boundary
  ret = event_manager->EventElapsedTime(1, 2, time);
  EXPECT_EQ(ge::PARAM_INVALID, ret);

  // startevent_idx > stopevent_idx
  ret = event_manager->EventElapsedTime(1, 0, time);
  EXPECT_EQ(ge::PARAM_INVALID, ret);

  delete event_manager;
}
TEST_F(UTEST_model_manager_event_manager, call_GetEvent) {
  ge::EventManager event_manager;
  size_t event_num = 1;
  event_manager.Init(event_num);
  rtEvent_t event = nullptr;
  Status ret = event_manager.GetEvent(2, event);
  EXPECT_EQ(ge::PARAM_INVALID, ret);
  ret = event_manager.GetEvent(0, event);
  EXPECT_EQ(SUCCESS, ret);
}
