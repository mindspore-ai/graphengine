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

#include <bits/stdc++.h>
#include <dirent.h>
#include <gtest/gtest.h>
#include <fstream>
#include <map>
#include <string>

#define protected public
#define private public
#include "common/profiling/profiling_manager.h"
#undef protected
#undef private

using namespace ge;
using namespace std;

class UtestGeProfilinganager : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

class TestReporter : public Msprof::Engine::Reporter {
 public:
  TestReporter() {}
  ~TestReporter() {}

 public:
  int Report(const Msprof::Engine::ReporterData *data) { return 0; }

  int Flush() { return 0; }
};

class TestPluginIntf : public Msprof::Engine::PluginIntf {
 public:
  TestPluginIntf() {}
  ~TestPluginIntf() {}

 public:
  int Init(const Msprof::Engine::Reporter *reporter) { return 0; }

  int UnInit() { return 0; }
};

TEST_F(UtestGeProfilinganager, init_success) {
  setenv("PROFILING_MODE", "true", true);
  Options options;
  options.device_id = 0;
  options.job_id = "0";
  string profiling_config;

  ProfilingManager::Instance().SetProfilingConfig(profiling_config);

  Status ret = ProfilingManager::Instance().Init(options);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGeProfilinganager, start_profiling_success) {
  int32_t iter_num = 1;

  setenv("PROFILING_MODE", "true", true);
  setenv("PROFILING_OPTIONS", "training_trace", true);
  Options options;
  string profiling_config;

  ProfilingManager::Instance().SetProfilingConfig(profiling_config);

  Status ret = ProfilingManager::Instance().Init(options);
  EXPECT_EQ(ret, ge::SUCCESS);
  ret = ProfilingManager::Instance().StartProfiling(iter_num, 0);
  EXPECT_EQ(ret, ge::SUCCESS);

  setenv("PROFILING_OPTIONS", "op_trance", true);
  ret = ProfilingManager::Instance().Init(options);
  EXPECT_EQ(ret, ge::SUCCESS);
  ret = ProfilingManager::Instance().StartProfiling(iter_num, 0);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGeProfilinganager, stop_profiling_success) {
  int32_t iter_num = 1;
  Options options;

  TestReporter test_reporter;

  string profiling_config;
  ProfilingManager::Instance().SetProfilingConfig(profiling_config);

  Status ret = 0;
  setenv("PROFILING_OPTIONS", "op_trance", true);
  ret = ProfilingManager::Instance().Init(options);
  EXPECT_EQ(ret, ge::SUCCESS);
  ret = ProfilingManager::Instance().StartProfiling(iter_num, 0);
  EXPECT_EQ(ret, ge::SUCCESS);
  ProfilingManager::Instance().StopProfiling();
}

TEST_F(UtestGeProfilinganager, plugin_impl_success) {
  PluginImpl plugin_Impl("FMK");
  TestReporter test_reporter;
  Msprof::Engine::Reporter *reporter_ptr = &test_reporter;
  plugin_Impl.Init(reporter_ptr);
  plugin_Impl.UnInit();
}

TEST_F(UtestGeProfilinganager, profiling_engine_impl_success) {
  ProfilingEngineImpl profiling_engine_impl;

  Msprof::Engine::PluginIntf *plugin_ptr = new TestPluginIntf();
  profiling_engine_impl.ReleasePlugin(plugin_ptr);

  Msprof::Engine::PluginIntf *ptr = profiling_engine_impl.CreatePlugin();
  delete ptr;
  ptr = nullptr;
}

TEST_F(UtestGeProfilinganager, set_profilng_cfg_success) {
  string profiling_config = "profiling_mode: true";
  ProfilingManager::Instance().SetProfilingConfig(profiling_config);
}

TEST_F(UtestGeProfilinganager, init_from_cfg_success0) {
  Options options;
  string profiling_config =
      "{\"startCfg\":[{\"deviceID\":\"0\",\"features\":[{\"name\":\"op_trace\",\"conf\":\"2\"}]}]}";
  ProfilingManager::Instance().SetProfilingConfig(profiling_config);

  Status ret = ProfilingManager::Instance().Init(options);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGeProfilinganager, init_from_cfg_success1) {
  Options options;
  string profiling_config =
      "{\"startCfg\":[{\"deviceID\":\"0\",\"features\":[{\"name\":\"test_trace\"}],\"jobID\":\"1231231231\"}]}";
  ProfilingManager::Instance().SetProfilingConfig(profiling_config);

  Status ret = ProfilingManager::Instance().Init(options);
  EXPECT_EQ(ret, ge::SUCCESS);
}
