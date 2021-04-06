/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "common/dump/dump_properties.h"

#include <cstdio>
#include <string>

#include "common/ge/ge_util.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_types.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/utils/attr_utils.h"

namespace {
const std::string kEnableFlag = "1";
const std::string kDumpStatusOpen = "on";
const uint32_t kAicoreOverflow = (0x1 << 0);
const uint32_t kAtomicOverflow = (0x1 << 1);
const uint32_t kAllOverflow = (kAicoreOverflow | kAtomicOverflow);
}  // namespace
namespace ge {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY DumpProperties::DumpProperties(const DumpProperties &other) {
  CopyFrom(other);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY DumpProperties &DumpProperties::operator=(
  const DumpProperties &other) {
  CopyFrom(other);
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void DumpProperties::InitByOptions() {
  enable_dump_.clear();
  enable_dump_debug_.clear();
  dump_path_.clear();
  dump_step_.clear();
  dump_mode_.clear();
  is_op_debug_ = false;
  op_debug_mode_ = 0;

  std::string enable_dump;
  (void)GetContext().GetOption(OPTION_EXEC_ENABLE_DUMP, enable_dump);
  enable_dump_ = enable_dump;

  std::string enable_dump_debug;
  (void)GetContext().GetOption(OPTION_EXEC_ENABLE_DUMP_DEBUG, enable_dump_debug);
  enable_dump_debug_ = enable_dump_debug;

  if ((enable_dump_ == kEnableFlag) || (enable_dump_debug_ == kEnableFlag)) {
    std::string dump_path;
    if (GetContext().GetOption(OPTION_EXEC_DUMP_PATH, dump_path) == GRAPH_SUCCESS) {
      if (!dump_path.empty() && dump_path[dump_path.size() - 1] != '/') {
        dump_path = dump_path + "/";
      }
      dump_path = dump_path + CurrentTimeInStr() + "/";
      GELOGI("Get dump path %s successfully", dump_path.c_str());
      SetDumpPath(dump_path);
    } else {
      GELOGW("Dump path is not set");
    }
  }

  if (enable_dump_ == kEnableFlag) {
    std::string dump_step;
    if (GetContext().GetOption(OPTION_EXEC_DUMP_STEP, dump_step) == GRAPH_SUCCESS) {
      GELOGI("Get dump step %s successfully", dump_step.c_str());
      SetDumpStep(dump_step);
    }
    string dump_mode;
    if (GetContext().GetOption(OPTION_EXEC_DUMP_MODE, dump_mode) == GRAPH_SUCCESS) {
      GELOGI("Get dump mode %s successfully", dump_mode.c_str());
      SetDumpMode(dump_mode);
    }
    AddPropertyValue(DUMP_ALL_MODEL, {});
  }

  SetDumpDebugOptions();
}

// The following is the new dump scenario of the fusion operator
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void DumpProperties::AddPropertyValue(
  const std::string &model, const std::set<std::string> &layers) {
  for (const std::string &layer : layers) {
    GELOGI("This model %s config to dump layer %s", model.c_str(), layer.c_str());
  }

  model_dump_properties_map_[model] = layers;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void DumpProperties::DeletePropertyValue(const std::string &model) {
  auto iter = model_dump_properties_map_.find(model);
  if (iter != model_dump_properties_map_.end()) {
    model_dump_properties_map_.erase(iter);
  }
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void DumpProperties::ClearDumpPropertyValue() {
  model_dump_properties_map_.clear();
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void DumpProperties::ClearDumpInfo() {
  enable_dump_.clear();
  enable_dump_debug_.clear();
  dump_path_.clear();
  dump_step_.clear();
  dump_mode_.clear();
  dump_op_switch_.clear();
  dump_status_.clear();
  is_op_debug_ = false;
  op_debug_mode_ = 0;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY std::set<std::string> DumpProperties::GetAllDumpModel() const {
  std::set<std::string> model_list;
  for (auto &iter : model_dump_properties_map_) {
    model_list.insert(iter.first);
  }

  return model_list;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY std::set<std::string> DumpProperties::GetPropertyValue(
  const std::string &model) const {
  auto iter = model_dump_properties_map_.find(model);
  if (iter != model_dump_properties_map_.end()) {
    return iter->second;
  }
  return {};
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool DumpProperties::IsLayerNeedDump(
  const std::string &model, const std::string &om_name, const std::string &op_name) const {
  // if dump all
  GELOGD("model name is %s om name is %s op is %s in layer need dump", model.c_str(), om_name.c_str(), op_name.c_str());
  if (model_dump_properties_map_.find(DUMP_ALL_MODEL) != model_dump_properties_map_.end()) {
    return true;
  }

  // if this model need dump
  auto om_name_iter = model_dump_properties_map_.find(om_name);
  auto model_name_iter = model_dump_properties_map_.find(model);
  if (om_name_iter != model_dump_properties_map_.end() || model_name_iter != model_dump_properties_map_.end()) {
    // if no dump layer info, dump all layer in this model
    auto model_iter = om_name_iter != model_dump_properties_map_.end() ? om_name_iter : model_name_iter;
    if (model_iter->second.empty()) {
      return true;
    }

    return model_iter->second.find(op_name) != model_iter->second.end();
  }

  GELOGD("Model %s is not seated to be dump.", model.c_str());
  return false;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void DumpProperties::SetDumpPath(const std::string &path) {
  dump_path_ = path;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY const std::string &DumpProperties::GetDumpPath() const {
  return dump_path_;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void DumpProperties::SetDumpStep(const std::string &step) {
  dump_step_ = step;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY const std::string &DumpProperties::GetDumpStep() const {
  return dump_step_;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void DumpProperties::SetDumpMode(const std::string &mode) {
  dump_mode_ = mode;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY const std::string &DumpProperties::GetDumpMode() const {
  return dump_mode_;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void DumpProperties::SetDumpStatus(const std::string &status) {
  dump_status_ = status;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY const std::string &DumpProperties::GetDumpStatus() const {
  return dump_status_;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void DumpProperties::SetDumpOpSwitch(
  const std::string &dump_op_switch) {
  dump_op_switch_ = dump_op_switch;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY const std::string &DumpProperties::GetDumpOpSwitch() const {
  return dump_op_switch_;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool DumpProperties::IsSingleOpNeedDump() const {
  if (dump_op_switch_ == kDumpStatusOpen) {
    return true;
  }
  return false;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool DumpProperties::IsDumpOpen() const {
  if (enable_dump_ == kEnableFlag || dump_status_ == kDumpStatusOpen) {
    return true;
  }
  return false;
}

void DumpProperties::CopyFrom(const DumpProperties &other) {
  if (&other != this) {
    enable_dump_ = other.enable_dump_;
    enable_dump_debug_ = other.enable_dump_debug_;
    dump_path_ = other.dump_path_;
    dump_step_ = other.dump_step_;
    dump_mode_ = other.dump_mode_;
    dump_status_ = other.dump_status_;
    dump_op_switch_ = other.dump_op_switch_;

    model_dump_properties_map_ = other.model_dump_properties_map_;
    is_op_debug_ = other.is_op_debug_;
    op_debug_mode_ = other.op_debug_mode_;
  }
}

void DumpProperties::SetDumpDebugOptions() {
  if (enable_dump_debug_ == kEnableFlag) {
    std::string dump_debug_mode;
    if (GetContext().GetOption(OPTION_EXEC_DUMP_DEBUG_MODE, dump_debug_mode) == GRAPH_SUCCESS) {
      GELOGD("Get dump debug mode %s successfully", dump_debug_mode.c_str());
    } else {
      GELOGW("Dump debug mode is not set.");
      return;
    }

    if (dump_debug_mode == OP_DEBUG_AICORE) {
      GELOGD("ge.exec.dumpDebugMode=aicore_overflow, op debug is open.");
      is_op_debug_ = true;
      op_debug_mode_ = kAicoreOverflow;
    } else if (dump_debug_mode == OP_DEBUG_ATOMIC) {
      GELOGD("ge.exec.dumpDebugMode=atomic_overflow, op debug is open.");
      is_op_debug_ = true;
      op_debug_mode_ = kAtomicOverflow;
    } else if (dump_debug_mode == OP_DEBUG_ALL) {
      GELOGD("ge.exec.dumpDebugMode=all, op debug is open.");
      is_op_debug_ = true;
      op_debug_mode_ = kAllOverflow;
    } else {
      GELOGW("ge.exec.dumpDebugMode is invalid.");
    }
  } else {
    GELOGI("ge.exec.enableDumpDebug is false or is not set.");
  }
}
}  // namespace ge
