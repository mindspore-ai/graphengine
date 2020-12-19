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

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "graph/debug/ge_log.h"
#include "register/graph_optimizer/graph_fusion/fusion_pattern.h"

namespace fe {

#define FE_PATTERN_ERROR_RETURN_IF(condition, ...) \
  do {                                             \
    if (condition) {                               \
      SetError();                                  \
      GELOGW(__VA_ARGS__);                         \
      return *this;                                \
    }                                              \
  } while (0)

FusionPattern::FusionPattern(string name) : name_(name), output_(nullptr), has_error_(false) {}

FusionPattern::~FusionPattern() {}

/**
 * @ingroup fe
 * @brief set pattern name
 */
FusionPattern &FusionPattern::SetName(const string &name) {
  name_ = name;
  return *this;
}

/**
 * @ingroup fe
 * @brief add Op description with unknown number of args
 */
FusionPattern &FusionPattern::AddOpDesc(const string &id, const initializer_list<string> &types) {
  return AddOpDesc(id, vector<string>(types));
}

/**
 * @ingroup fe
 * @brief add Op description with vector
 */
FusionPattern &FusionPattern::AddOpDesc(const string &id, const vector<string> &types) {
  FE_PATTERN_ERROR_RETURN_IF(id.empty(), "ID cannot be empty.");

  FE_PATTERN_ERROR_RETURN_IF(GetOpDesc(id) != nullptr, "ID already exists. (id:%s)", id.c_str());

  std::shared_ptr<OpDesc> op(new (std::nothrow) OpDesc());
  FE_PATTERN_ERROR_RETURN_IF(op == nullptr, "new an object failed.");

  op->id = id;
  op->types = types;
  op->repeatable = false;
  op->is_output = false;
  ops_.push_back(op);
  op_map_[id] = op;

  return *this;
}

/**
 * @ingroup fe
 * @brief set input Ops with unknown number of args
 */
FusionPattern &FusionPattern::SetInputs(const string &id, const initializer_list<string> &input_ids) {
  return SetInputs(id, vector<string>(input_ids));
}

/**
 * @ingroup fe
 * @brief set input Ops with vector
 */
FusionPattern &FusionPattern::SetInputs(const string &id, const vector<string> &input_ids) {
  FE_PATTERN_ERROR_RETURN_IF(id.empty(), "Id cannot be empty.");
  std::shared_ptr<FusionPattern::OpDesc> op_desc = GetOpDesc(id);
  FE_PATTERN_ERROR_RETURN_IF(op_desc == nullptr, "Id does not exist. (id:%s)", id.c_str());

  op_desc->inputs.clear();

  for (const string &input_id : input_ids) {
    std::shared_ptr<FusionPattern::OpDesc> input_op_desc = GetOpDesc(input_id);
    FE_PATTERN_ERROR_RETURN_IF(input_op_desc == nullptr, "Id does not exist. (id:%s)", input_id.c_str());
    op_desc->inputs.push_back(input_op_desc);
  }

  return *this;
}

/**
 * @ingroup fe
 * @brief set output Op
 */
FusionPattern &FusionPattern::SetOutput(const string &id) {
  FE_PATTERN_ERROR_RETURN_IF(id.empty(), "Id cannot be empty.");
  std::shared_ptr<FusionPattern::OpDesc> op_desc = GetOpDesc(id);
  FE_PATTERN_ERROR_RETURN_IF(op_desc == nullptr, "Id does not exist. (id:%s)", id.c_str());

  op_desc->is_output = true;

  return *this;
}

/**
 * @ingroup fe
 * @brief build pattern and check if error exists
 */
bool FusionPattern::Build() {
  if (has_error_) {
    return false;
  }

  // check whether output node already exists
  for (const std::shared_ptr<OpDesc> op : ops_) {
    if (op->is_output) {
      if (output_ != nullptr) {
        SetError();
        GELOGW("Multiple outputs are not supported. (id:%s)", op->id.c_str());
        break;
      }
      output_ = op;
    }
  }

  if (output_ == nullptr) {
    SetError();
    GELOGW("Output must be set value.");
  }

  return !has_error_;
}

/**
 * @ingroup fe
 * @brief get pattern name
 */
const string &FusionPattern::GetName() const { return name_; }
/**
 * @ingroup fe
 * @brief get the OpDesc of input Ops (const)
 */
const vector<std::shared_ptr<FusionPattern::OpDesc>> *FusionPattern::GetInputs(
    const std::shared_ptr<FusionPattern::OpDesc> op_desc) {
  if (op_desc == nullptr) {
    return nullptr;
  }
  return &(op_desc->inputs);
}

/**
 * @ingroup fe
 * @brief get the OpDesc of output Op
 */
const std::shared_ptr<FusionPattern::OpDesc> FusionPattern::GetOutput() const { return output_; }

/**
 * @ingroup fe
 * @brief print pattern
 */
void FusionPattern::Dump() const {
  std::ostringstream oss;
  oss << std::endl << "Pattern (" << name_ << "):" << std::endl;
  for (const auto &op : ops_) {
    oss << "  " << op->id << ": {";
    for (const string &type : op->types) {
      oss << type << ", ";
    }
    oss << "} {";
    for (const auto &input : op->inputs) {
      oss << input->id << ", ";
    }
    oss << "}";

    if (op->is_output) {
      oss << " [output]";
    }

    oss << std::endl;
  }
  GELOGD("%s", oss.str().c_str());
}

/**
 * @ingroup fe
 * @brief get OpDesc based on ID, return nullptr if failed
 */
std::shared_ptr<FusionPattern::OpDesc> FusionPattern::GetOpDesc(const string &id) const {
  auto it = op_map_.find(id);
  if (it != op_map_.end()) {
    return it->second;
  }
  return nullptr;
}

/**
 * @ingroup fe
 * @brief record error
 */
void FusionPattern::SetError() { has_error_ = true; }
}
