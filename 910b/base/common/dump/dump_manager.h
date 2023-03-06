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

#ifndef GE_COMMON_DUMP_DUMP_MANAGER_H_
#define GE_COMMON_DUMP_DUMP_MANAGER_H_

#include <mutex>

#include "common/dump/dump_properties.h"
#include "framework/common/ge_types.h"

namespace ge {
class DumpManager {
 public:
  static DumpManager &GetInstance();

  Status SetDumpConf(const DumpConfig &dump_config);
  const DumpProperties &GetDumpProperties(const uint64_t session_id);
  const std::map<uint64_t, DumpProperties> &GetDumpPropertiesMap() const { return dump_properties_map_; }
  void AddDumpProperties(const uint64_t session_id, const DumpProperties &dump_properties);
  void RemoveDumpProperties(const uint64_t session_id);
  static bool GetCfgFromOption(const std::map<std::string, std::string> &options_all, DumpConfig &dump_cfg);

 private:
  bool NeedDoDump(const DumpConfig &dump_config, DumpProperties &dump_properties);
  void SetDumpDebugConf(const DumpConfig &dump_config, DumpProperties &dump_properties) const;
  Status SetDumpPath(const DumpConfig &dump_config, DumpProperties &dump_properties) const;
  Status SetNormalDumpConf(const DumpConfig &dump_config, DumpProperties &dump_properties);
  void SetDumpList(const DumpConfig &dump_config, DumpProperties &dump_properties) const;
  std::mutex mutex_;
  std::map<uint64_t, DumpProperties> dump_properties_map_;
  // enable dump from acl with session_id 0
  std::map<uint64_t, DumpProperties> infer_dump_properties_map_;
};
}  // namespace ge
#endif  // GE_COMMON_DUMP_DUMP_MANAGER_H_
