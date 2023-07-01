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
#ifndef AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_GLOBAL_DUMPER_H_
#define AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_GLOBAL_DUMPER_H_
#include <mutex>

#include "built_in_subscriber_definitions.h"
#include "framework/common/ge_visibility.h"
#include "graph/compute_graph.h"
#include "runtime/base.h"

namespace ge {
class ExceptionDumper;
}
namespace gert {
// for global info for exception_dump and global switch
class VISIBILITY_EXPORT GlobalDumper {
 public:
  GlobalDumper(const GlobalDumper &) = delete;
  GlobalDumper(GlobalDumper &&) = delete;
  GlobalDumper &operator=(const GlobalDumper &) = delete;
  GlobalDumper &operator==(GlobalDumper &&) = delete;
  static GlobalDumper *GetInstance() {
    static GlobalDumper global_dumper;
    return &global_dumper;
  }

  static void OnGlobalDumperSwitch(void *ins, uint64_t enable_flags);

  void AddExceptionInfo(const rtExceptionInfo &exception_data) const;

  ge::ExceptionDumper *MutableExceptionDumper() const;

  void SetEnableFlags(const uint64_t enable_flags) {
    enable_flags_ = enable_flags;
  }

  uint64_t GetEnableFlags() const {
      return enable_flags_;
  };

  bool IsEnable(DumpType dump_type) const {
    return static_cast<bool>(enable_flags_ & BuiltInSubscriberUtil::EnableBit<DumpType>(dump_type));
  }

  std::set<ge::ExceptionDumper *> &MutableInnerExceptionDumpers() {
    return exception_dumpers_;
  }

 private:
  GlobalDumper();
  uint64_t enable_flags_{0UL};
  // each davinci model has own exception dumper
  std::set<ge::ExceptionDumper *> exception_dumpers_{};
};
}  // namespace gert
#endif
