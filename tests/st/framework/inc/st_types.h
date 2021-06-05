/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef GRAPHENGINE_ST_TYPES_H
#define GRAPHENGINE_ST_TYPES_H
#include <map>
namespace ge {
namespace st {
const std::string kAicoreLibName = "AiCoreLib";
const std::string kVectorLibName = "VectorLib";
const std::string kAicpuLibName = "AicpuLib";
const std::string kAicpuAscendLibName = "AicpuAscendLib";
const std::string kHcclLibName = "HcclLib";
const std::string kRTSLibName = "RTSLib";
const std::map<std::string, std::string> kStubEngine2KernelLib = {
  {"AIcoreEngine", "AiCoreLib"}, {"VectorEngine", "VectorLib"},
  {"DNN_VM_AICPU", "AicpuLib"},  {"DNN_VM_AICPU_ASCEND", "AicpuAscendLib"},
  {"DNN_HCCL", "HcclLib"},       {"DNN_VM_RTS", "RTSLib"}};
}  // namespace st
}  // namespace ge
#endif  // GRAPHENGINE_ST_TYPES_H
