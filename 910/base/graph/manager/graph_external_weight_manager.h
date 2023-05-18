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

#ifndef GE_GRAPH_EXTERNAL_WEIGHT_MANAGER_H_
#define GE_GRAPH_EXTERNAL_WEIGHT_MANAGER_H_

#include <map>
#include <set>
#include <mutex>
#include <vector>
#include "external/ge/ge_api_types.h"

namespace ge {
class ExternalWeightManager {
 public:
  explicit ExternalWeightManager(const uint64_t session_id);

  ~ExternalWeightManager() = default;

  bool CanReuseExternalWeight(std::string &file_name, const uint8_t *const data, const size_t data_length);

  bool IsWeightExist(std::string &file_name);

  bool IsWeightLoaded(const std::string &file_name);

  void SetWeightLoaded(const std::string &file_name);

  void Finalize() noexcept;

 private:
  static Status CheckFilesSame(const std::string &file_name,
                               const uint8_t *const data,
                               const size_t data_length,
                               bool &is_content_same);

  std::mutex mutex_;
  uint64_t session_id_;
  std::set<std::string> loaded_external_weight_files;
  std::map<size_t, std::vector<std::string>> hash_to_files_;
  std::map<std::string, std::string> file_to_exist_file_;
};

using ExternalWeightManagerPtr = std::shared_ptr<ExternalWeightManager>;

class ExternalWeightManagerPool {
 public:
  static ExternalWeightManagerPool &Instance();

  ~ExternalWeightManagerPool();

  ExternalWeightManagerPtr GetManager(const uint64_t session_id);

  void RemoveManager(const uint64_t session_id);

  void Destroy() noexcept;

 private:
  ExternalWeightManagerPool() = default;
  std::mutex mutex_;
  std::map<uint64_t, ExternalWeightManagerPtr> session_id_to_manager_;
};
}  // namespace ge
#endif  // GE_GRAPH_EXTERNAL_WEIGHT_MANAGER_H_