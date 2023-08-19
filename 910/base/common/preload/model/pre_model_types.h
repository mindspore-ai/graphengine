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
#ifndef GE_COMMON_PRELOAD_PRE_MODEL_TYPES_H_
#define GE_COMMON_PRELOAD_PRE_MODEL_TYPES_H_
#include <string>
#include <map>
#include "framework/common/taskdown_common.h"

namespace ge {
constexpr uint32_t DEFAULT_INFO_VALUE_ZERO = 0U;
constexpr uint32_t DEFAULT_INFO_VALUE_ONE = 1U;
constexpr uint32_t DEFAULT_INFO_VALUE_EIGHT = 8U;
constexpr uint32_t kAlignBy4B = 4U;

enum class NanoTaskDescType : uint16_t { NANO_AI_CORE, NANO_AI_CPU, NANO_PLACE_HOLDER, NANO_RESEVERD };

enum class NanoTaskPreStatus : uint16_t { NANO_PRE_DISABLE, NANO_PRE_ENABLE };

enum class NanoTaskSoftUserStatus : uint16_t { NANO_SOFTUSER_DEFAULT, NANO_SOFTUSER_HOSTFUNC };

enum class EngineType : uint32_t { kDefaultEngine, kNanoEngine };
const std::string kPreEngineAiCore = "aicore_engine";
const std::string kPreEngineAiCpu = "aicpu_engine";
const std::string kPreEngineNano = "nano_engine";
const std::string kPreEngineDefault = "default_engine";

// nano engine
const std::string kPreEngineNanoAiCore = "nano_aicore_engine";
const std::string kPreEngineNanoAiCpu = "nano_aicpu_engine";

const std::map<uint32_t, std::string> kKernelTypeToEngineName = {
    {static_cast<uint32_t>(ccKernelType::TE), kPreEngineAiCore},
    {static_cast<uint32_t>(ccKernelType::AI_CPU), kPreEngineAiCpu},
    {static_cast<uint32_t>(ccKernelType::CUST_AI_CPU), kPreEngineAiCpu}};
const std::map<uint32_t, std::string> kKernelTypeToNanoEngineName = {
    {static_cast<uint32_t>(ccKernelType::TE), kPreEngineNanoAiCore},
    {static_cast<uint32_t>(ccKernelType::AI_CPU), kPreEngineNanoAiCpu}};
const std::map<uint32_t, std::string> kTaskTypeToEngineName = {
    {static_cast<uint32_t>(RT_MODEL_TASK_EVENT_RECORD), kPreEngineDefault},
    {static_cast<uint32_t>(RT_MODEL_TASK_EVENT_WAIT), kPreEngineDefault},
    {static_cast<uint32_t>(RT_MODEL_TASK_STREAM_SWITCH), kPreEngineDefault},
    {static_cast<uint32_t>(RT_MODEL_TASK_STREAM_ACTIVE), kPreEngineDefault},
    {static_cast<uint32_t>(RT_MODEL_TASK_STREAM_LABEL_SWITCH_BY_INDEX), kPreEngineDefault},
    {static_cast<uint32_t>(RT_MODEL_TASK_STREAM_LABEL_GOTO), kPreEngineDefault},
    {static_cast<uint32_t>(RT_MODEL_TASK_LABEL_SET), kPreEngineDefault},
    {static_cast<uint32_t>(RT_MODEL_TASK_LABEL_SWITCH), kPreEngineDefault},
    {static_cast<uint32_t>(RT_MODEL_TASK_LABEL_GOTO), kPreEngineDefault},
    {static_cast<uint32_t>(RT_MODEL_TASK_MEMCPY_ASYNC), kPreEngineDefault},
    {static_cast<uint32_t>(RT_MODEL_TASK_MEMCPY_ADDR_ASYNC), kPreEngineDefault},
    {static_cast<uint32_t>(RT_MODEL_TASK_FUSION_START), kPreEngineDefault},
    {static_cast<uint32_t>(RT_MODEL_TASK_FUSION_END), kPreEngineDefault},
    {static_cast<uint32_t>(RT_MODEL_TASK_MODEL_END_GRAPH), kPreEngineDefault}};
}  // namespace ge
#endif  // GE_COMMON_PRELOAD_PRE_MODEL_UTILS_H_