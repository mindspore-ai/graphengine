/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include "diagnose_switch.h"

namespace ge {
namespace {
SingleDiagnoseSwitch profiling_switch_;
SingleDiagnoseSwitch dumper_switch_;
}  // namespace

namespace diagnoseSwitch {
SingleDiagnoseSwitch &MutableProfiling() {
  return profiling_switch_;
}

const SingleDiagnoseSwitch &GetProfiling() {
  return profiling_switch_;
}

SingleDiagnoseSwitch &MutableDumper() {
  return dumper_switch_;
}
const SingleDiagnoseSwitch &GetDumper() {
  return dumper_switch_;
}

void EnableDataDump() {
  dumper_switch_.SetEnableFlag(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kDataDump}));
}

void EnableExceptionDump() {
  dumper_switch_.SetEnableFlag(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kExceptionDump}));
}

void EnableGeHostProfiling() {
  profiling_switch_.SetEnableFlag(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>({gert::ProfilingType::kGeHost}));
}

void EnableDeviceProfiling() {
  profiling_switch_.SetEnableFlag(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>({gert::ProfilingType::kDevice}));
}

void EnableCannHostProfiling() {
  profiling_switch_.SetEnableFlag(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>({gert::ProfilingType::kCannHost}));
}

void EnableProfiling(const std::vector<gert::ProfilingType> &prof_type) {
  profiling_switch_.SetEnableFlag(gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>(prof_type));
}

void DisableProfiling() {
  profiling_switch_.SetEnableFlag(0UL);
}

void DisableDumper() {
  dumper_switch_.SetEnableFlag(0UL);
}
}
}  // namespace ge