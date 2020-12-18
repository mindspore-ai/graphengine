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

#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"
#include "graph/debug/ge_log.h"

namespace fe {

FusionStatisticRecorder::FusionStatisticRecorder(){};

FusionStatisticRecorder::~FusionStatisticRecorder(){};

FusionStatisticRecorder &FusionStatisticRecorder::Instance() {
  static FusionStatisticRecorder fusion_statistic_recoder;
  return fusion_statistic_recoder;
}

void FusionStatisticRecorder::UpdateGraphFusionMatchTimes(FusionInfo &fusion_info) {
  std::lock_guard<std::recursive_mutex> lock_guard(mutex_);
  if (fusion_info.GetMatchTimes() != 0) {
    std::string session_and_graph_id = std::to_string(fusion_info.GetSessionId()) + "_" + fusion_info.GetGraphId();
    graph_fusion_info_map_[session_and_graph_id][fusion_info.GetPassName()].AddMatchTimes(fusion_info.GetMatchTimes());
    GELOGD("session %d graph %s pass %s match_times value: %d", fusion_info.GetSessionId(),
           fusion_info.GetGraphId().c_str(), fusion_info.GetPassName().c_str(),
           graph_fusion_info_map_[session_and_graph_id][fusion_info.GetPassName()].GetMatchTimes());
  }
}

void FusionStatisticRecorder::UpdateGraphFusionEffectTimes(FusionInfo &fusion_info) {
  std::lock_guard<std::recursive_mutex> lock_guard(mutex_);
  if (fusion_info.GetEffectTimes() != 0) {
    std::string session_and_graph_id = std::to_string(fusion_info.GetSessionId()) + "_" + fusion_info.GetGraphId();
    graph_fusion_info_map_[session_and_graph_id][fusion_info.GetPassName()].AddEffectTimes(
        fusion_info.GetEffectTimes());
    GELOGD("session %d graph %s pass %s effect_times value: %d", fusion_info.GetSessionId(),
           fusion_info.GetGraphId().c_str(), fusion_info.GetPassName().c_str(),
           graph_fusion_info_map_[session_and_graph_id][fusion_info.GetPassName()].GetEffectTimes());
  }
}

void FusionStatisticRecorder::UpdateBufferFusionMatchTimes(FusionInfo &fusion_info) {
  std::lock_guard<std::recursive_mutex> lock_guard(mutex_);
  if (fusion_info.GetMatchTimes() != 0) {
    std::string session_and_graph_id = std::to_string(fusion_info.GetSessionId()) + "_" + fusion_info.GetGraphId();
    buffer_fusion_info_map_[session_and_graph_id][fusion_info.GetPassName()].AddMatchTimes(fusion_info.GetMatchTimes());
    GELOGD("ub session %d graph %s pass %s match_times value: %d", fusion_info.GetSessionId(),
           fusion_info.GetGraphId().c_str(), fusion_info.GetPassName().c_str(),
           buffer_fusion_info_map_[session_and_graph_id][fusion_info.GetPassName()].GetMatchTimes());
  }
}

void FusionStatisticRecorder::UpdateBufferFusionEffectTimes(FusionInfo &fusion_info) {
  std::lock_guard<std::recursive_mutex> lock_guard(mutex_);
  if (fusion_info.GetEffectTimes() != 0) {
    std::string session_and_graph_id = std::to_string(fusion_info.GetSessionId()) + "_" + fusion_info.GetGraphId();
    buffer_fusion_info_map_[session_and_graph_id][fusion_info.GetPassName()].AddEffectTimes(
        fusion_info.GetEffectTimes());
    GELOGD("ub session %d graph %s pass %s effect_times value: %d", fusion_info.GetSessionId(),
           fusion_info.GetGraphId().c_str(), fusion_info.GetPassName().c_str(),
           buffer_fusion_info_map_[session_and_graph_id][fusion_info.GetPassName()].GetEffectTimes());
  }
}

void FusionStatisticRecorder::GetAndClearFusionInfo(const std::string &session_graph_id,
                                                    std::map<std::string, FusionInfo> &graph_fusion_info_map,
                                                    std::map<std::string, FusionInfo> &buffer_fusion_info_map) {
  std::lock_guard<std::recursive_mutex> lock_guard(mutex_);
  GELOGD("start to get graph map size %d", graph_fusion_info_map_.size());
  GELOGD("start to get ub graph map size %d", buffer_fusion_info_map_.size());
  GetFusionInfo(session_graph_id, graph_fusion_info_map, buffer_fusion_info_map);
  ClearFusionInfo(session_graph_id);
}

void FusionStatisticRecorder::GetFusionInfo(const std::string &session_graph_id,
                                            std::map<std::string, FusionInfo> &graph_fusion_info_map,
                                            std::map<std::string, FusionInfo> &buffer_fusion_info_map) {
  if (graph_fusion_info_map_.find(session_graph_id) != graph_fusion_info_map_.end()) {
    graph_fusion_info_map = graph_fusion_info_map_[session_graph_id];
  }
  if (buffer_fusion_info_map_.find(session_graph_id) != buffer_fusion_info_map_.end()) {
    buffer_fusion_info_map = buffer_fusion_info_map_[session_graph_id];
  }
}

void FusionStatisticRecorder::ClearFusionInfo(std::string session_graph_id) {
  if (graph_fusion_info_map_.find(session_graph_id) != graph_fusion_info_map_.end()) {
    graph_fusion_info_map_.erase(session_graph_id);
  }
  if (buffer_fusion_info_map_.find(session_graph_id) != buffer_fusion_info_map_.end()) {
    buffer_fusion_info_map_.erase(session_graph_id);
  }
}

FusionInfo::FusionInfo(uint64_t session_id, std::string graph_id, std::string pass_name, int32_t match_times,
                       int32_t effect_times)
    : session_id_(session_id),
      graph_id_(std::move(graph_id)),
      pass_name_(std::move(pass_name)),
      match_times_(match_times),
      effect_times_(effect_times) {}

FusionInfo::~FusionInfo() {}

void FusionInfo::AddMatchTimes(int32_t match_times) { this->match_times_ += match_times; }

void FusionInfo::AddEffectTimes(int32_t effect_times) { this->effect_times_ += effect_times; }

int32_t FusionInfo::GetMatchTimes() { return match_times_; }

int32_t FusionInfo::GetEffectTimes() { return effect_times_; }

std::string FusionInfo::GetGraphId() { return graph_id_; }

std::string FusionInfo::GetPassName() { return pass_name_; }

uint64_t FusionInfo::GetSessionId() { return session_id_; }

void FusionInfo::SetMatchTimes(int32_t match_times) { this->match_times_ = match_times; }

void FusionInfo::SetEffectTimes(int32_t effect_times) { this->effect_times_ = effect_times; }
}
