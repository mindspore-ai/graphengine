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

#ifndef PLATFORM_INFO_DEF_H
#define PLATFORM_INFO_DEF_H

#include <map>
#include <string>
#include <vector>

using std::map;
using std::string;
using std::vector;

namespace fe {
enum MemoryType { DDR = 0, HBM };

enum L2Type { Cache = 0, Buff };

typedef struct tagStrInfo {
  string aicVersion;
  string ccecAICVersion;
  string ccecAIVVersion;
  string isSupportAIcpuCompiler;
} StrInfo;

typedef struct tagSoCInfo {
  uint32_t aiCoreCnt;
  uint32_t vectorCoreCnt;
  uint32_t aiCpuCnt;
  MemoryType memoryType;
  uint64_t memorySize;
  L2Type l2Type;
  uint64_t l2Size;
  uint32_t l2PageNum;
} SoCInfo;

typedef struct tagAiCoreSpec {
  double cubeFreq;
  uint64_t cubeMSize;
  uint64_t cubeNSize;
  uint64_t cubeKSize;
  uint64_t vecCalcSize;
  uint64_t l0ASize;
  uint64_t l0BSize;
  uint64_t l0CSize;
  uint64_t l1Size;
  uint64_t smaskBuffer;
  uint64_t ubSize;
  uint64_t ubblockSize;
  uint64_t ubbankSize;
  uint64_t ubbankNum;
  uint64_t ubburstInOneBlock;
  uint64_t ubbankGroupNum;
  uint32_t unzipEngines;
  uint32_t unzipMaxRatios;
  uint32_t unzipChannels;
  uint8_t unzipIsTight;
} AiCoreSpec;

typedef struct tagAiCoreMemoryRates {
  double ddrRate;
  double ddrReadRate;
  double ddrWriteRate;
  double l2Rate;
  double l2ReadRate;
  double l2WriteRate;
  double l1ToL0ARate;
  double l1ToL0BRate;
  double l1ToUBRate;
  double l0CToUBRate;
  double ubToL2Rate;
  double ubToDdrRate;
  double ubToL1Rate;
} AiCoreMemoryRates;

typedef struct tagVectorCoreSpec {
  double vecFreq;
  uint64_t vecCalcSize;
  uint64_t smaskBuffer;
  uint64_t ubSize;
  uint64_t ubblockSize;
  uint64_t ubbankSize;
  uint64_t ubbankNum;
  uint64_t ubburstInOneBlock;
  uint64_t ubbankGroupNum;
  uint64_t vectorRegSize;
  uint64_t predicateRegSize;
  uint64_t addressRegSize;
} VectorCoreSpec;

typedef struct tagVectorCoreMemoryRates {
  double ddrRate;
  double ddrReadRate;
  double ddrWriteRate;
  double l2Rate;
  double l2ReadRate;
  double l2WriteRate;
  double ubToL2Rate;
  double ubToDdrRate;
} VectorCoreMemoryRates;

typedef struct tagCPUCache {
  uint32_t AICPUSyncBySW;
  uint32_t TSCPUSyncBySW;
} CPUCache;

typedef struct tagPlatformInfo {
  StrInfo strInfo;
  SoCInfo socInfo;
  AiCoreSpec aiCoreSpec;
  AiCoreMemoryRates aiCoreMemoryRates;
  map<string, vector<string>> aiCoreIntrinsicDtypeMap;
  VectorCoreSpec vectorCoreSpec;
  VectorCoreMemoryRates vectorCoreMemoryRates;
  CPUCache cpucache;
  map<string, vector<string>> vectorCoreIntrinsicDtypeMap;
} PlatformInfo;

typedef struct tagOptionalInfo {
  string socVersion;
  string coreType;
  uint32_t aiCoreNum;
  string l1FusionFlag;
} OptionalInfo;
}  // namespace fe
#endif
