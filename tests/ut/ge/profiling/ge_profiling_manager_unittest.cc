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
#include "graph/ge_local_context.h"
#undef protected
#undef private

using namespace ge;
using namespace std;

class UtestGeProfilinganager : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

int32_t ReporterCallback(uint32_t moduleId, uint32_t type, void *data, uint32_t len) {
  return -1;
}

TEST_F(UtestGeProfilinganager, init_success) {
  setenv("PROFILING_MODE", "true", true);
  Options options;
  options.device_id = 0;
  options.job_id = "0";
  options.profiling_mode = "1";
  options.profiling_options = R"({"result_path":"/data/profiling","training_trace":"on","task_trace":"on","aicpu_trace":"on","fp_point":"Data_0","bp_point":"addn","ai_core_metrics":"ResourceConflictRatio"})";


  struct MsprofGeOptions prof_conf = {{ 0 }};

  Status ret = ProfilingManager::Instance().InitFromOptions(options, prof_conf);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGeProfilinganager, ParseOptions) {
  setenv("PROFILING_MODE", "true", true);
  Options options;
  options.device_id = 0;
  options.job_id = "0";
  options.profiling_mode = "1";
  options.profiling_options = R"({"result_path":"/data/profiling","training_trace":"on","task_trace":"on","aicpu_trace":"on","fp_point":"Data_0","bp_point":"addn","ai_core_metrics":"ResourceConflictRatio"})";

  struct MsprofGeOptions prof_conf = {{ 0 }};
  Status ret = ProfilingManager::Instance().ParseOptions(options.profiling_options);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(ProfilingManager::Instance().is_training_trace_, true);
  EXPECT_EQ(ProfilingManager::Instance().fp_point_, "Data_0");
  EXPECT_EQ(ProfilingManager::Instance().bp_point_, "addn");
}

TEST_F(UtestGeProfilinganager, plungin_init_) {
  ProfilingManager::Instance().prof_cb_.msprofReporterCallback = ReporterCallback;

  Status ret = ProfilingManager::Instance().PluginInit();
  EXPECT_EQ(ret, INTERNAL_ERROR);
  ProfilingManager::Instance().prof_cb_.msprofReporterCallback = nullptr;
}

TEST_F(UtestGeProfilinganager, report_data_) {
  std::string data = "ge is better than tensorflow.";
  std::string tag_name = "fmk";
  ProfilingManager::Instance().ReportData(0, data, tag_name);
}

TEST_F(UtestGeProfilinganager, get_fp_bp_point_) {
  map<std::string, string> options_map = {
    {OPTION_EXEC_PROFILING_OPTIONS,
    R"({"result_path":"/data/profiling","training_trace":"on","task_trace":"on","aicpu_trace":"on","fp_point":"Data_0","bp_point":"addn","ai_core_metrics":"ResourceConflictRatio"})"}};
  GEThreadLocalContext &context = GetThreadLocalContext();
  context.SetGraphOption(options_map);

  std::string fp_point;
  std::string bp_point;
  ProfilingManager::Instance().GetFpBpPoint(fp_point, bp_point);
  EXPECT_EQ(fp_point, "Data_0");
  EXPECT_EQ(bp_point, "addn");
}

TEST_F(UtestGeProfilinganager, get_fp_bp_point_empty) {
  // fp bp empty
  map<std::string, string> options_map = {
    { OPTION_EXEC_PROFILING_OPTIONS,
      R"({"result_path":"/data/profiling","training_trace":"on","task_trace":"on","aicpu_trace":"on","ai_core_metrics":"ResourceConflictRatio"})"}};
  GEThreadLocalContext &context = GetThreadLocalContext();
  context.SetGraphOption(options_map);
  std::string fp_point = "fp";
  std::string bp_point = "bp";
  ProfilingManager::Instance().bp_point_ = "";
  ProfilingManager::Instance().fp_point_ = "";
  ProfilingManager::Instance().GetFpBpPoint(fp_point, bp_point);
  EXPECT_EQ(fp_point, "");
  EXPECT_EQ(bp_point, "");
}