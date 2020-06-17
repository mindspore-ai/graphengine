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

#ifndef GE_HYBRID_KERNEL_AICORE_OP_TASK_H_
#define GE_HYBRID_KERNEL_AICORE_OP_TASK_H_

#include <memory>
#include "common/ge_inner_error_codes.h"
#include "runtime/stream.h"
namespace ge {
namespace hybrid {
class AiCoreOpTask {
 public:
  AiCoreOpTask() = default;
  ~AiCoreOpTask() = default;
  Status LaunchKernel(rtStream_t stream);

 private:
  friend class AiCoreTaskBuilder;
  friend class AiCoreNodeTask;
  std::string stub_name_;
  void *stub_func_ = nullptr;
  std::unique_ptr<uint8_t[]> args_ = nullptr;
  uint32_t args_size_ = 0;
  uint32_t block_dim_ = 1;
  uint16_t offset_ = 0;
};

}  // namespace hybrid
}  // namespace ge
#endif  // GE_HYBRID_KERNEL_AICORE_OP_TASK_H_
