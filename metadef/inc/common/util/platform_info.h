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

#ifndef PLATFORM_INFO_H
#define PLATFORM_INFO_H

#include <map>
#include <string>
#include <vector>
#include "platform_info_def.h"

using std::map;
using std::vector;
using std::string;

namespace fe {
class PlatformInfoManager {
 public:
  PlatformInfoManager(const PlatformInfoManager &) = delete;
  PlatformInfoManager &operator=(const PlatformInfoManager &) = delete;

  static PlatformInfoManager &Instance();
  uint32_t InitializePlatformInfo();
  uint32_t Finalize();

  uint32_t GetPlatformInfo(const string SoCVersion, PlatformInfo &platform_info, OptionalInfo &opti_compilation_info);

  uint32_t GetPlatformInfoWithOutSocVersion(PlatformInfo &platform_info, OptionalInfo &opti_compilation_info);

  void SetOptionalCompilationInfo(OptionalInfo &opti_compilation_info);

 private:
  PlatformInfoManager();
  ~PlatformInfoManager();

  uint32_t LoadIniFile(string ini_file_real_path);

  void Trim(string &str);

  uint32_t LoadConfigFile(string real_path);

  string RealPath(const std::string &path);

  string GetSoFilePath();

  void ParseVersion(map<string, string> &version_map, string &soc_version, PlatformInfo &platform_info_temp);

  void ParseSocInfo(map<string, string> &soc_info_map, PlatformInfo &platform_info_temp);

  void ParseCubeOfAICoreSpec(map<string, string> &ai_core_spec_map, PlatformInfo &platform_info_temp);

  void ParseBufferOfAICoreSpec(map<string, string> &ai_core_spec_map, PlatformInfo &platform_info_temp);

  void ParseUBOfAICoreSpec(map<string, string> &ai_core_spec_map, PlatformInfo &platform_info_temp);

  void ParseUnzipOfAICoreSpec(map<string, string> &ai_core_spec_map, PlatformInfo &platform_info_temp);

  void ParseAICoreSpec(map<string, string> &ai_core_spec_map, PlatformInfo &platform_info_temp);

  void ParseBufferOfAICoreMemoryRates(map<string, string> &ai_core_memory_rates_map, PlatformInfo &platform_info_temp);

  void ParseAICoreMemoryRates(map<string, string> &ai_core_memory_rates_map, PlatformInfo &platform_info_temp);

  void ParseUBOfAICoreMemoryRates(map<string, string> &ai_core_memory_rates_map, PlatformInfo &platform_info_temp);

  void ParseAICoreintrinsicDtypeMap(map<string, string> &ai_coreintrinsic_dtype_map, PlatformInfo &platform_info_temp);

  void ParseVectorCoreSpec(map<string, string> &vector_core_spec_map, PlatformInfo &platform_info_temp);

  void ParseVectorCoreMemoryRates(map<string, string> &vector_core_memory_rates_map, PlatformInfo &platform_info_temp);

  void ParseCPUCache(map<string, string> &CPUCacheMap, PlatformInfo &platform_info_temp);

  void ParseVectorCoreintrinsicDtypeMap(map<string, string> &vector_coreintrinsic_dtype_map,
                                        PlatformInfo &platform_info_temp);

  uint32_t ParsePlatformInfoFromStrToStruct(map<string, map<string, string>> &content_info_map, string &soc_version,
                                            PlatformInfo &platform_info_temp);

  uint32_t AssemblePlatformInfoVector(map<string, map<string, string>> &content_info_map);

 private:
  bool init_flag_;
  map<string, PlatformInfo> platform_info_map_;
  OptionalInfo opti_compilation_info_;
};
}  // namespace fe
#endif
