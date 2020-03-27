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

#include "common/properties_manager.h"

#include <climits>
#include <cstdio>
#include <fstream>

#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_types.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "common/util.h"

namespace ge {
PropertiesManager::PropertiesManager() : is_inited_(false), delimiter("=") {}
PropertiesManager::~PropertiesManager() {}

// singleton
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY PropertiesManager &PropertiesManager::Instance() {
  static PropertiesManager instance;
  return instance;
}

// Initialize property configuration
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool PropertiesManager::Init(const std::string &file_path) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (is_inited_) {
    GELOGW("Already inited, will be initialized again");
    properties_map_.clear();
    is_inited_ = false;
    return is_inited_;
  }

  if (!LoadFileContent(file_path)) {
    return false;
  }

  is_inited_ = true;
  return is_inited_;
}

// Load file contents
bool PropertiesManager::LoadFileContent(const std::string &file_path) {
  // Normalize the path
  string resolved_file_path = RealPath(file_path.c_str());
  if (resolved_file_path.empty()) {
    DOMI_LOGE("Invalid input file path [%s], make sure that the file path is correct.", file_path.c_str());
    return false;
  }
  std::ifstream fs(resolved_file_path, std::ifstream::in);

  if (!fs.is_open()) {
    GELOGW("Open %s failed.", file_path.c_str());
    return false;
  }

  std::string line;

  while (getline(fs, line)) {  // line not with \n
    if (!ParseLine(line)) {
      GELOGW("Parse line failed. content is [%s].", line.c_str());
      fs.close();
      return false;
    }
  }

  fs.close();  // close the file

  GELOGI("LoadFileContent success.");
  return true;
}

// Parsing the command line
bool PropertiesManager::ParseLine(const std::string &line) {
  std::string temp = Trim(line);
  // Comment or newline returns true directly
  if (temp.find_first_of('#') == 0 || *(temp.c_str()) == '\n') {
    return true;
  }

  if (!temp.empty()) {
    std::string::size_type pos = temp.find_first_of(delimiter);  // Must be divided by "="
    if (pos == std::string::npos) {
      GELOGW("Incorrect line [%s]", line.c_str());
      return false;
    }

    std::string map_key = Trim(temp.substr(0, pos));
    std::string value = Trim(temp.substr(pos + 1));
    if (map_key.empty() || value.empty()) {
      GELOGW("Map_key or value empty. %s", line.c_str());
      return false;
    }

    properties_map_[map_key] = value;
  }

  return true;
}

// Remove the space and tab before and after the string
std::string PropertiesManager::Trim(const std::string &str) {
  if (str.empty()) {
    return str;
  }

  std::string::size_type start = str.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return str;
  }

  std::string::size_type end = str.find_last_not_of(" \t\r\n") + 1;
  return str.substr(start, end);
}

// Get property value, if not found, return ""
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY std::string PropertiesManager::GetPropertyValue(
    const std::string &map_key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = properties_map_.find(map_key);
  if (properties_map_.end() != iter) {
    return iter->second;
  }

  return "";
}

// Set property value
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void PropertiesManager::SetPropertyValue(const std::string &map_key,
                                                                                          const std::string &value) {
  std::lock_guard<std::mutex> lock(mutex_);
  properties_map_[map_key] = value;
}

// return properties_map_
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY std::map<std::string, std::string>
PropertiesManager::GetPropertyMap() {
  std::lock_guard<std::mutex> lock(mutex_);
  return properties_map_;
}

// Set separator
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void PropertiesManager::SetPropertyDelimiter(const std::string &de) {
  delimiter = de;
}

// The following is the new dump scenario of the fusion operator
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void PropertiesManager::AddDumpPropertyValue(
    const std::string &model, const std::set<std::string> &layers) {
  for (const std::string &layer : layers) {
    GELOGI("This model %s config to dump layer %s", model.c_str(), layer.c_str());
  }

  std::lock_guard<std::mutex> lock(dump_mutex_);
  model_dump_properties_map_[model] = layers;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void PropertiesManager::DeleteDumpPropertyValue(
    const std::string &model) {
  std::lock_guard<std::mutex> lock(dump_mutex_);
  auto iter = model_dump_properties_map_.find(model);
  if (iter != model_dump_properties_map_.end()) {
    model_dump_properties_map_.erase(iter);
  }
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void PropertiesManager::ClearDumpPropertyValue() {
  std::lock_guard<std::mutex> lock(dump_mutex_);
  model_dump_properties_map_.clear();
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY std::set<std::string> PropertiesManager::GetDumpPropertyValue(
    const std::string &model) {
  std::lock_guard<std::mutex> lock(dump_mutex_);
  auto iter = model_dump_properties_map_.find(model);
  if (iter != model_dump_properties_map_.end()) {
    return iter->second;
  }
  return {};
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool PropertiesManager::IsLayerNeedDump(const std::string &model,
                                                                                         const std::string &op_name) {
  std::lock_guard<std::mutex> lock(dump_mutex_);
  // if dump all
  if (model_dump_properties_map_.find(ge::DUMP_ALL_MODEL) != model_dump_properties_map_.end()) {
    return true;
  }

  // if this model need dump
  auto model_iter = model_dump_properties_map_.find(model);
  if (model_iter != model_dump_properties_map_.end()) {
    // if no dump layer info, dump all layer in this model
    if (model_iter->second.empty()) {
      return true;
    }

    return model_iter->second.find(op_name) != model_iter->second.end();
  }

  GELOGD("Model %s is not seated to be dump.", model.c_str());
  return false;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool PropertiesManager::QueryModelDumpStatus(
    const std::string &model) {
  std::lock_guard<std::mutex> lock(dump_mutex_);
  auto iter = model_dump_properties_map_.find(model);
  if (iter != model_dump_properties_map_.end()) {
    return true;
  } else if (model_dump_properties_map_.find(ge::DUMP_ALL_MODEL) != model_dump_properties_map_.end()) {
    return true;
  }
  return false;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void PropertiesManager::SetDumpOutputModel(
    const std::string &output_mode) {
  std::lock_guard<std::mutex> lock(dump_mutex_);
  this->output_mode_ = output_mode;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY std::string PropertiesManager::GetDumpOutputModel() {
  std::lock_guard<std::mutex> lock(dump_mutex_);
  return this->output_mode_;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void PropertiesManager::SetDumpOutputPath(
    const std::string &output_path) {
  std::lock_guard<std::mutex> lock(dump_mutex_);
  this->output_path_ = output_path;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY std::string PropertiesManager::GetDumpOutputPath() {
  std::lock_guard<std::mutex> lock(dump_mutex_);
  return this->output_path_;
}
}  // namespace ge
