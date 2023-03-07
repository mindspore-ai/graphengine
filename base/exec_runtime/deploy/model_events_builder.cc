/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#include "model_events_builder.h"
#include <sstream>
#include "nlohmann/json.hpp"
#include "ge/ge_api_error_codes.h"
#include "ge/ge_api_types.h"
#include "common/ge_inner_error_codes.h"
#include "common/debug/ge_log.h"
#include "common/checker.h"
#include "graph/def_types.h"

#define USED_BY_JSON __attribute__((unused)) static
namespace ge {
struct EventNodeRankId {
  int32_t model_rank_id;
  int32_t logic_rank_id;
};

struct ExecRankMap {
  std::vector<EventNodeRankId> event_node_rank_ids;
};

namespace {
using Json = nlohmann::json;
Status StringToJson(const std::string &json_str, Json &json) {
  std::stringstream ss;
  ss << json_str;
  try {
    ss >> json;
  } catch (const nlohmann::json::exception &e) {
    GELOGE(PARAM_INVALID, "Failed to init json object, err = %s, json_str = %s", e.what(), json_str.c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}
template<typename T>
void GetValue(const Json &j, const std::string &key, T &value) {
  value = j.at(key).template get<T>();
}
template<typename T>
std::string ToJsonString(const T &obj) {
  try {
    const Json j = obj;
    return j.dump();
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to dump object, err = %s", e.what());
    return "";
  }
}

template<typename T>
Status ParseFromJson(const std::string &type, const std::string &json_str, T &value) {
  Json json;
  GE_ASSERT_SUCCESS(StringToJson(json_str, json));
  try {
    value = json.get<T>();
  } catch (const Json::exception &e) {
    GELOGE(PARAM_INVALID,
           "Failed to parse json object, type = %s, err = %s, json_str = %s",
           type.c_str(),
           e.what(),
           json_str.c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}
}

/**************************************
  {
  "model_events": [{
      "model_instance_name":"graph1_instance1",
      "rank_id":0,
      "event_nodes": [
        {
          "type":"HcomSend",
          "group_name":"HcclWorldGroup",
          "event_name":"PartitionedCall0:sr_tag_10001",
          "group_tag":10001,
          "model_peer_rank":1,
          "logic_peer_rank":1
        }
      ]
    }]
  }
**************************************/
const ModelEvents *ge::ModelEventsBuilder::Build(int32_t rank_id) {
  if (event_table_.empty()) {
    return nullptr;
  }
  if (models_events_.empty()) {
    Json json;
    GE_ASSERT_SUCCESS(ParseFromJson("model_events", event_table_, models_events_),
                      "[Call][ParseFromJson] failed, rank_id = %d", rank_id);
  }
  int32_t id = 0;
  for (const auto &models_event : models_events_) {
    if (rank_id == models_event.rank_id) {
      return &(models_events_[id]);
    }
    id++;
  }
  GELOGD("Can not find rank_id = %d in event_table = %s", event_table_.c_str());
  return nullptr;
}

USED_BY_JSON void from_json(const Json &j, ModelEvents &model_events) {
  GetValue(j, "model_instance_name", model_events.model_instance_name);
  GetValue(j, "rank_id", model_events.rank_id);
  GetValue(j, "event_nodes", model_events.event_nodes);
}

USED_BY_JSON void from_json(const Json &j, EventNode &event_node) {
  GetValue(j, "type", event_node.type);
  GetValue(j, "group_name", event_node.group_name);
  GetValue(j, "group_tag", event_node.group_tag);
  GetValue(j, "event_name", event_node.event_name);
  GetValue(j, "model_peer_rank", event_node.model_peer_rank);
  GetValue(j, "logic_peer_rank", event_node.logic_peer_rank);
}

USED_BY_JSON void to_json(Json &j, const EventNodeRankId &event_node_rank_id) {
  j = Json();
  j["model_rank_id"] = event_node_rank_id.model_rank_id;
  j["logic_rank_id"] = event_node_rank_id.logic_rank_id;
}

USED_BY_JSON void to_json(Json &j, const ExecRankMap &exec_rank_map) {
  j = Json();
  j["rank_map"] = exec_rank_map.event_node_rank_ids;
}

std::string HcomExecUtils::ToJson(const std::vector<EventNode> &event_nodes) {
  ExecRankMap exec_rank_map;
  for (const auto &event_node : event_nodes) {
    EventNodeRankId event_node_rank_id = {event_node.model_peer_rank, event_node.logic_peer_rank};
    exec_rank_map.event_node_rank_ids.emplace_back(event_node_rank_id);
  }
  return ToJsonString(exec_rank_map);
}
}
