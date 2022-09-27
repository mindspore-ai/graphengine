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

#ifndef AIR_CXX_INC_FRAMEWORK_RUNTIME_SUBSCRIBER_BUILT_IN_SUBSCRIBER_DEFINITIONS_H_
#define AIR_CXX_INC_FRAMEWORK_RUNTIME_SUBSCRIBER_BUILT_IN_SUBSCRIBER_DEFINITIONS_H_
#include <type_traits>
#include <vector>
#include "graph/gnode.h"
#include "common/ge_types.h"
#include "framework/common/ge_visibility.h"
namespace ge {
class GeRootModel;
}
namespace gert {
constexpr size_t kProfilingDataCap = 10UL * 1024UL * 1024UL;
constexpr size_t kInitSize = 10UL * 1024UL;
constexpr size_t kModelStrIdx = 0UL;
constexpr size_t kExecuteStrIdx = 1UL;
constexpr size_t kRegStartIdx = 2UL;
constexpr size_t kDouble = 2UL;
constexpr size_t kInitSubscriberSize = 2UL;

enum class BuiltInSubscriberType { kProfiling, kDumper, kNum };

enum class ProfilingType {
  kCannHost,  // 打开Host侧调度的profiling
  kDevice,
  kGeHost,  // 打开GE Host侧调度的profiling
  kNum,
  kAll = kNum
};
static_assert(static_cast<size_t>(ProfilingType::kNum) < sizeof(uint64_t) * 8,
              "The max num of profiling type must less than the width of uint64");

enum class DumpType { kDataDump, kExceptionDump, kNum, kAll = kNum };
static_assert(static_cast<size_t>(DumpType::kNum) < sizeof(uint64_t) * 8,
              "The max num of dumper type must less than the width of uint64");
class ModelV2Executor;
struct SubscriberExtendInfo {
  ModelV2Executor *executor;
  ge::ComputeGraphPtr exe_graph;
  ge::ModelData model_data;
  std::shared_ptr<ge::GeRootModel> root_model;
};

class VISIBILITY_EXPORT BuiltInSubscriberUtil {
 public:
  template <typename T,
            typename std::enable_if<(std::is_same<T, ProfilingType>::value) || (std::is_same<T, DumpType>::value),
                                    int>::type = 0>
  constexpr static uint64_t EnableBit(T et) {
    return 1UL << static_cast<size_t>(et);
  }

  template <typename T,
            typename std::enable_if<(std::is_same<T, ProfilingType>::value) || (std::is_same<T, DumpType>::value),
                                    int>::type = 0>
  static uint64_t BuildEnableFlags(const std::vector<T> &enable_types) {
    uint64_t flag = 0UL;
    for (auto et : enable_types) {
      if (et == T::kAll) {
        return EnableBit(T::kNum) - 1UL;
      }
      flag |= EnableBit(et);
    }
    return flag;
  }
};
}  // namespace gert
#endif  // AIR_CXX_INC_FRAMEWORK_RUNTIME_SUBSCRIBER_BUILT_IN_SUBSCRIBER_DEFINITIONS_H_
