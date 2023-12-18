/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023
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

#ifndef GE_COMMON_OP_TILING_PARSE_CONTEXT_MANAGER_H_
#define GE_COMMON_OP_TILING_PARSE_CONTEXT_MANAGER_H_

#include <string>
#include <mutex>
#include <memory>
#include <unordered_map>
#include "exe_graph/runtime/kernel_run_context_builder.h"

namespace optiling {
using ParseContextHolderPtr = std::shared_ptr<gert::KernelContextHolder>;
class ParseContextCacheManager {
 public:
  ParseContextCacheManager(const ParseContextCacheManager &) = delete;
  ParseContextCacheManager &operator=(const ParseContextCacheManager &)& = delete;
  static ParseContextCacheManager &Instance();

  ParseContextHolderPtr GetParseContext(const std::string &compile_key) const;
  void AddParseContext(const std::string &compile_key, const ParseContextHolderPtr &parse_context);

 private:
  ParseContextCacheManager() = default;
  ~ParseContextCacheManager() = default;
  mutable std::mutex parse_context_mutex_;
  std::unordered_map<std::string, ParseContextHolderPtr> compile_keys_to_parse_contexts_;
};
}  // namespace optiling
#endif  // GE_COMMON_OP_TILING_PARSE_CONTEXT_MANAGER_H_
