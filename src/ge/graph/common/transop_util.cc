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

#include "graph/common/transop_util.h"

#include "common/types.h"

namespace {
const int kInvalidTransopDataIndex = -1;
}  // namespace

namespace ge {
TransOpUtil::TransOpUtil() {
  transop_index_map_ = {{TRANSDATA, 0}, {TRANSPOSE, 0}, {TRANSPOSED, 0}, {RESHAPE, 0},
                        {REFORMAT, 0},  {CAST, 0},      {SQUEEZE, 0},    {EXPANDDIMS, 0}};
}

TransOpUtil::~TransOpUtil() {}

TransOpUtil &TransOpUtil::Instance() {
  static TransOpUtil inst;
  return inst;
}

bool TransOpUtil::IsTransOp(const NodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  return IsTransOp(node->GetType());
}

bool TransOpUtil::IsTransOp(const std::string &type) {
  return Instance().transop_index_map_.find(type) != Instance().transop_index_map_.end();
}

int TransOpUtil::GetTransOpDataIndex(const NodePtr &node) {
  if (node == nullptr) {
    return kInvalidTransopDataIndex;
  }
  return GetTransOpDataIndex(node->GetType());
}

int TransOpUtil::GetTransOpDataIndex(const std::string &type) {
  auto it = Instance().transop_index_map_.find(type);
  if (it != Instance().transop_index_map_.end()) {
    return it->second;
  }
  return kInvalidTransopDataIndex;
}
}  // namespace ge
