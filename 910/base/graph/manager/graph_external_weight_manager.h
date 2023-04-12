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

namespace ge {
class ExternalWeightManager {
 public:
  static ExternalWeightManager &GetInstance();

  ~ExternalWeightManager();

  bool CanReuseExternalWeight(std::string &file_name, const uint8_t *data, const size_t data_length);

  bool IsWeightExist(std::string &file_name);

  void RemoveExternalWeightFile(const std::string &file_name);

  std::set<std::string> &GetExternalWeightFiles();

  void Destroy() noexcept;

 private:
  ExternalWeightManager() = default;

  static bool CheckFilesSame(const std::string &exist_file_name,
                             const uint8_t *data,
                             const size_t data_length);

  std::mutex mutex_;
  std::set<std::string> external_weight_files_;
  std::map<size_t, std::vector<std::string>> hash_to_files_;
  std::map<std::string, std::string> file_to_exist_file_;
};
}  // namespace ge
#endif  // GE_GRAPH_EXTERNAL_WEIGHT_MANAGER_H_
