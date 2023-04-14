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

#ifndef GE_GRAPH_COMMON_TRANSOP_UTIL_H_
#define GE_GRAPH_COMMON_TRANSOP_UTIL_H_

#include <string>
#include <unordered_map>

#include "graph/node.h"

namespace ge {
class TransOpUtil {
 public:
  static bool IsTransOp(const NodePtr &node);

  static bool IsTransOp(const std::string &type);

  static int32_t GetTransOpDataIndex(const NodePtr &node);

  static int32_t GetTransOpDataIndex(const std::string &type);

  static bool CheckPrecisionLoss(const NodePtr &src_node);

  static std::string TransopMapToString();

 private:
  TransOpUtil();

  ~TransOpUtil() = default;

  static TransOpUtil &Instance();

  std::map<std::string, int32_t> transop_index_map_;
};
}  // namespace ge

#endif  // GE_GRAPH_COMMON_TRANSOP_UTIL_H_
