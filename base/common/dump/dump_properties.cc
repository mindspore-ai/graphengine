/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#include <string>
#include <regex>

#include "common/plugin/ge_util.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_types.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/utils/attr_utils.h"
#include "mmpa/mmpa_api.h"
#include "runtime/config.h"
#include "exec_runtime/execution_runtime.h"
#include "common/global_variables/diagnose_switch.h"

namespace {
const std::string kEnableFlag = "1";
const std::string kDumpStatusOpen = "on";
const uint32_t kAicoreOverflow = 0x1U; // (0x1U << 0U)
const uint32_t kAtomicOverflow = 0x2U; // (0x1U << 1U)
const uint32_t kAllOverflow = kAicoreOverflow | kAtomicOverflow;
}  // namespace
namespace ge {
Status DumpProperties::CheckDumpStep(const std::string &dump_step) const {
  const std::string modified_dum_step = dump_step + "|";
  std::smatch result;
  const std::regex pattern(R"((\d{1,}-\d{1,}\||\d{1,}\|)+)");
  if (regex_match(modified_dum_step, result, pattern)) {
    auto match_vecs = StringUtils::Split(result.str(), '|');
    // StringUtils::Split: There will emplace a "" when the last character of string is delimiter
    if ((!match_vecs.empty()) && (match_vecs.back().empty())) {
      match_vecs.pop_back();
    }
    if (match_vecs.empty()) {
      REPORT_CALL_ERROR("E19999", "Split may get fatal exception, dump_step:%s.", dump_step.c_str());
      GELOGE(FAILED, "[Check][Param] failed. Split may get fatal exception, ge.exec.dumpStep:%s.", dump_step.c_str());
      return FAILED;
    }
    // 100 is the max sets of dump steps.
    if (match_vecs.size() > 100U) {
      REPORT_INPUT_ERROR("E10001",
                         std::vector<std::string>({"parameter", "value", "reason"}),
                         std::vector<std::string>({"ge.exec.dumpStep", dump_step.c_str(),
                             " is not supported, only support dump <= 100 sets of data"}));
      GELOGE(PARAM_INVALID, "[Check][Param] get dump_step value:%s, "
             "dump_step only support dump <= 100 sets of data.", dump_step.c_str());
      return PARAM_INVALID;
    }
    for (const auto &match_vec : match_vecs) {
      auto vec_after_split = StringUtils::Split(match_vec, '-');
      // StringUtils::Split: There will emplace a "" when the last character of string is delimiter
      if ((!vec_after_split.empty()) && (vec_after_split.back().empty())) {
        vec_after_split.pop_back();
      }
      if (match_vecs.empty()) {
        REPORT_CALL_ERROR("E19999", "Split may get fatal exception.");
        GELOGE(FAILED, "[Check][Param] failed, split may get fatal exception.");
        return FAILED;
      }
      if (vec_after_split.size() > 1U) {
        if (std::strtol(vec_after_split[0U].c_str(), nullptr, 10 /* base int */) >=
            std::strtol(vec_after_split[1U].c_str(), nullptr, 10 /* base int */)) {
          REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                             std::vector<std::string>({"ge.exec.dumpStep", dump_step.c_str(), " is not supported."
                                 "in range steps, the first step is >= second step, correct example:'0|5|10-20"}));
          GELOGE(PARAM_INVALID, "[Check][Param] get dump_step value:%s, "
          "in range steps, the first step is >= second step, correct example:'0|5|10-20'", dump_step.c_str());
          return PARAM_INVALID;
        }
      }
    }
  } else {
    REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                       std::vector<std::string>({"ge.exec.dumpStep", dump_step.c_str(),
                           " is not supported, correct example:'0|5|10|50-100."}));
    GELOGE(PARAM_INVALID, "[Check][Param] get dump_step value:%s, "
    "dump_step std::string style is error, correct example:'0|5|10|50-100.'", dump_step.c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status DumpProperties::CheckDumpMode(const std::string &dump_mode) const {
  const std::set<std::string> dump_mode_list = {"input", "output", "all"};
  const std::set<std::string>::const_iterator &iter = dump_mode_list.find(dump_mode);
  if (iter == dump_mode_list.end()) {
    REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                       std::vector<std::string>({"ge.exec.dumpMode", dump_mode.c_str(),
                           " is not supported, should be one of the following:[input, output, all]"}));
    GELOGE(PARAM_INVALID, "[Check][Param] the dump_debug_mode:%s, is is not supported,"
           "should be one of the following:[input, output, all].", dump_mode.c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status DumpProperties::CheckDumpPathValid(const std::string &input) const {
  if (mmIsDir(input.c_str()) != EN_OK) {
    REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                       std::vector<std::string>({"ge.exec.dumpPath", input.c_str(), " is not a directory."}));
    GELOGE(PARAM_INVALID, "[Check][Param] the path:%s, is not directory.", input.c_str());
    return PARAM_INVALID;
  }
  char_t trusted_path[MMPA_MAX_PATH] = {};
  if (mmRealPath(input.c_str(), &trusted_path[0], MMPA_MAX_PATH) != EN_OK) {
    REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                       std::vector<std::string>({"ge.exec.dumpPath", input.c_str(), " dumpPath invalid."}));
    GELOGE(PARAM_INVALID, "[Check][Param] the dumpPath:%s, is invalid.", input.c_str());
    return PARAM_INVALID;
  }
  const auto access_flag = static_cast<int32_t>(static_cast<uint32_t>(M_R_OK) | static_cast<uint32_t>(M_W_OK));
  if (mmAccess2(&trusted_path[0], access_flag) != EN_OK) {
    REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                       std::vector<std::string>({"ge.exec.dumpPath", input.c_str(),
                           " does't have read, write permissions."}));
    GELOGE(PARAM_INVALID, "[Check][Param] the path:%s, does't have read, write permissions.", input.c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status DumpProperties::CheckEnableDump(const std::string &input) const {
  std::set<std::string> enable_dump_option_list = {"1", "0"};
  const std::set<std::string>::const_iterator it = enable_dump_option_list.find(input);
  if (it == enable_dump_option_list.end()) {
    REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                       std::vector<std::string>({"ge.exec.enableDump", input.c_str(), " only support 1 or 0."}));
    GELOGE(PARAM_INVALID, "[Check][Param] Not support ge.exec.enableDump or ge.exec.enableDumpDebug format:%s, "
           "only support 1 or 0.", input.c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

DumpProperties::DumpProperties() {
  int32_t is_heterogeneous = 0;
  (void) rtGetIsHeterogenous(&is_heterogeneous);
  is_need_dump_path_ = (is_heterogeneous != ExecutionRuntime::kRuntimeTypeHeterogeneous);
}

DumpProperties::DumpProperties(const DumpProperties &dump) {
  CopyFrom(dump);
}

DumpProperties &DumpProperties::operator=(const DumpProperties &dump) {
  CopyFrom(dump);
  return *this;
}

Status DumpProperties::SetDumpOptions() {
  if (enable_dump_ == kEnableFlag) {
    std::string dump_step;
    if ((GetContext().GetOption(OPTION_EXEC_DUMP_STEP, dump_step) == GRAPH_SUCCESS) && (!dump_step.empty())) {
      GE_CHK_STATUS_RET(CheckDumpStep(dump_step), "[Check][dump_step] failed.");
      GELOGI("Get dump step %s successfully", dump_step.c_str());
      SetDumpStep(dump_step);
    }
    std::string dump_mode = "output";
    if (GetContext().GetOption(OPTION_EXEC_DUMP_MODE, dump_mode) == GRAPH_SUCCESS) {
      GELOGI("Get dump mode %s successfully", dump_mode.c_str());
      GE_CHK_STATUS_RET(CheckDumpMode(dump_mode), "[Check][dump_mode] failed.");
      SetDumpMode(dump_mode);
    }
    std::string dump_data = "tensor";
    if (GetContext().GetOption(OPTION_EXEC_DUMP_DATA, dump_data) == GRAPH_SUCCESS) {
      GELOGI("Get dump data %s successfully", dump_data.c_str());
      SetDumpData(dump_data);
    }
    std::string dump_layers;
    if ((GetContext().GetOption(OPTION_EXEC_DUMP_LAYER, dump_layers) == GRAPH_SUCCESS) && (!dump_layers.empty())) {
      GELOGI("Get dump layer %s successfully", dump_layers.c_str());
      SetDumpList(dump_layers);
    } else {
      AddPropertyValue(DUMP_ALL_MODEL, {});
    }
    diagnoseSwitch::EnableDataDump();
  }
  return SUCCESS;
}

Status DumpProperties::InitByOptions() {
  ClearDumpInfo();

  std::string enable_dump = std::to_string(0);
  (void)GetContext().GetOption(OPTION_EXEC_ENABLE_DUMP, enable_dump);
  enable_dump_ = enable_dump;
  if (!enable_dump_.empty()) {
    GE_CHK_STATUS_RET(CheckEnableDump(enable_dump_), "[Check][enable_dump] failed.");
  }

  std::string enable_dump_debug = std::to_string(0);
  (void)GetContext().GetOption(OPTION_EXEC_ENABLE_DUMP_DEBUG, enable_dump_debug);
  enable_dump_debug_ = enable_dump_debug;
  if (!enable_dump_debug_.empty()) {
    GE_CHK_STATUS_RET(CheckEnableDump(enable_dump_debug_), "[Check][enable_dump_debug] failed.");
  }
  if ((enable_dump_ == kEnableFlag) && (enable_dump_debug_ == kEnableFlag)) {
    REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                       std::vector<std::string>({"ge.exec.enableDump and ge.exec.enableDumpDebug",
                           enable_dump_ + ", " + enable_dump_debug,
                           "ge.exec.enableDump and ge.exec.enableDumpDebug cannot be set to 1 at the same time."}));
    GELOGE(FAILED, "ge.exec.enableDump and ge.exec.enableDumpDebug cannot be both set to 1 at the same time.");
    return FAILED;
  }
  if (is_need_dump_path_ && ((enable_dump_ == kEnableFlag) || (enable_dump_debug_ == kEnableFlag))) {
    std::string dump_path;
    if (GetContext().GetOption(OPTION_EXEC_DUMP_PATH, dump_path) == GRAPH_SUCCESS) {
      GE_CHK_STATUS_RET(CheckDumpPathValid(dump_path), "Check dump path failed.");
      if ((!dump_path.empty()) && (dump_path[dump_path.size() - 1U] != '/')) {
        dump_path = dump_path + "/";
      }
      dump_path = dump_path + CurrentTimeInStr() + "/";
      GELOGI("Get dump path %s successfully", dump_path.c_str());
      SetDumpPath(dump_path);
    } else {
      REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                         std::vector<std::string>({"ge.exec.dumpPath", dump_path, "ge.exec.dumpPath is not set."}));
      GELOGE(FAILED, "[Check][dump_path] failed. Dump path is not set.");
      return FAILED;
    }
  }

  GE_CHK_STATUS_RET(SetDumpOptions(), "SetDumpOptions failed.");

  GE_CHK_STATUS_RET(SetDumpDebugOptions(), "SetDumpDebugOptions failed.");

  return SUCCESS;
}

void DumpProperties::SetDumpList(const std::string &layers) {
  std::set<std::string> dump_layers;
  std::istringstream ss(layers);
  std::string layer;
  while (ss >> layer) {
    dump_layers.insert(layer);
  }
  AddPropertyValue(DUMP_LAYER_OP_MODEL, dump_layers);
}

// The following is the new dump scenario of the fusion operator
void DumpProperties::AddPropertyValue(const std::string &model, const std::set<std::string> &layers) {
  for (const std::string &layer : layers) {
    GELOGI("This model %s config to dump layer %s", model.c_str(), layer.c_str());
  }

  model_dump_properties_map_[model] = layers;
}

void DumpProperties::DeletePropertyValue(const std::string &model) {
  const std::map<std::string, std::set<std::string>>::const_iterator iter = model_dump_properties_map_.find(model);
  if (iter != model_dump_properties_map_.end()) {
    (void)model_dump_properties_map_.erase(iter);
  }
}

void DumpProperties::ClearDumpPropertyValue() {
  model_dump_properties_map_.clear();
}

void DumpProperties::ClearDumpInfo() {
  enable_dump_.clear();
  enable_dump_debug_.clear();
  dump_path_.clear();
  dump_step_.clear();
  dump_mode_.clear();
  dump_op_switch_.clear();
  dump_status_.clear();
  dump_data_.clear();
  is_train_op_debug_ = false;
  is_infer_op_debug_ = false;
  op_debug_mode_ = 0U;
}

std::set<std::string> DumpProperties::GetAllDumpModel() const {
  std::set<std::string> model_list;
  for (auto &iter : model_dump_properties_map_) {
    (void)model_list.insert(iter.first);
  }

  return model_list;
}

std::set<std::string> DumpProperties::GetPropertyValue(const std::string &model) const {
  const auto iter = model_dump_properties_map_.find(model);
  if (iter != model_dump_properties_map_.end()) {
    return iter->second;
  }
  return {};
}

bool DumpProperties::IsLayerNeedDump(const std::string &model, const std::string &om_name,
                                     const std::string &op_name) const {
  // if dump all
  GELOGD("model name is %s om name is %s op is %s in layer need dump", model.c_str(), om_name.c_str(), op_name.c_str());
  if (model_dump_properties_map_.find(DUMP_ALL_MODEL) != model_dump_properties_map_.end()) {
    return true;
  }
  // if dump layer need dump
  if (model_dump_properties_map_.find(DUMP_LAYER_OP_MODEL) != model_dump_properties_map_.end()) {
    const auto dump_name_iter = model_dump_properties_map_.find(DUMP_LAYER_OP_MODEL);
    if (dump_name_iter->second.empty()) {
      return true;
    }
    return dump_name_iter->second.find(op_name) != dump_name_iter->second.end();
  }
  // if this model need dump
  const auto om_name_iter = model_dump_properties_map_.find(om_name);
  const auto model_name_iter = model_dump_properties_map_.find(model);
  if ((om_name_iter != model_dump_properties_map_.end()) || (model_name_iter != model_dump_properties_map_.end())) {
    // if no dump layer info, dump all layer in this model
    const auto model_iter = (om_name_iter != model_dump_properties_map_.end()) ? om_name_iter : model_name_iter;
    if (model_iter->second.empty()) {
      return true;
    }

    return model_iter->second.find(op_name) != model_iter->second.end();
  }

  GELOGD("Model %s is not seated to be dump", model.c_str());
  return false;
}

void DumpProperties::SetDumpPath(const std::string &path) {
  dump_path_ = path;
}

const std::string &DumpProperties::GetDumpPath() const {
  return dump_path_;
}

void DumpProperties::SetDumpStep(const std::string &step) {
  dump_step_ = step;
}

const std::string &DumpProperties::GetDumpStep() const {
  return dump_step_;
}

void DumpProperties::SetDumpMode(const std::string &mode) {
  dump_mode_ = mode;
}

const std::string &DumpProperties::GetDumpMode() const {
  return dump_mode_;
}

void DumpProperties::SetDumpData(const std::string &data) {
  dump_data_ = data;
}

const std::string &DumpProperties::GetDumpData() const {
  return dump_data_;
}

void DumpProperties::SetDumpStatus(const std::string &status) {
  dump_status_ = status;
}

void DumpProperties::InitInferOpDebug() {
  is_infer_op_debug_ = true;
}

void DumpProperties::SetOpDebugMode(const uint32_t &op_debug_mode) {
  op_debug_mode_ = op_debug_mode;
}

void DumpProperties::SetDumpOpSwitch(const std::string &dump_op_switch) {
  dump_op_switch_ = dump_op_switch;
}

bool DumpProperties::IsSingleOpNeedDump() const {
  if (dump_op_switch_ == kDumpStatusOpen) {
    return true;
  }
  return false;
}

bool DumpProperties::IsNeedDumpPath() const {
  return is_need_dump_path_;
}

bool DumpProperties::IsDumpOpen() const {
  if ((enable_dump_ == kEnableFlag) || (dump_status_ == kDumpStatusOpen)) {
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
    dump_data_ = other.dump_data_;

    model_dump_properties_map_ = other.model_dump_properties_map_;
    is_train_op_debug_ = other.is_train_op_debug_;
    is_infer_op_debug_ = other.is_infer_op_debug_;
    op_debug_mode_ = other.op_debug_mode_;
  }
}

Status DumpProperties::SetDumpDebugOptions() {
  if (enable_dump_debug_ == kEnableFlag) {
    std::string dump_debug_mode;
    if (GetContext().GetOption(OPTION_EXEC_DUMP_DEBUG_MODE, dump_debug_mode) == GRAPH_SUCCESS) {
      GELOGD("Get ge.exec.dumpDebugMode %s successfully.", dump_debug_mode.c_str());
    } else {
      GELOGW("ge.exec.dumpDebugMode is not set.");
      return SUCCESS;
    }

    if (dump_debug_mode == OP_DEBUG_AICORE) {
      GELOGD("ge.exec.dumpDebugMode=aicore_overflow, op debug is open.");
      is_train_op_debug_ = true;
      op_debug_mode_ = kAicoreOverflow;
    } else if (dump_debug_mode == OP_DEBUG_ATOMIC) {
      GELOGD("ge.exec.dumpDebugMode=atomic_overflow, op debug is open.");
      is_train_op_debug_ = true;
      op_debug_mode_ = kAtomicOverflow;
    } else if (dump_debug_mode == OP_DEBUG_ALL) {
      GELOGD("ge.exec.dumpDebugMode=all, op debug is open.");
      is_train_op_debug_ = true;
      op_debug_mode_ = kAllOverflow;
    } else {
      REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                         std::vector<std::string>({"ge.exec.dumpDebugMode", dump_debug_mode,
                             "ge.exec.dumpDebugMode is invalid."}));
      GELOGE(PARAM_INVALID, "[Set][DumpDebugOptions] failed, ge.exec.dumpDebugMode is invalid.");
      return PARAM_INVALID;
    }
  } else {
    GELOGI("ge.exec.enableDumpDebug is false or is not set");
  }
  return SUCCESS;
}
}  // namespace ge
