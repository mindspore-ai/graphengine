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

#ifndef GE_SINGLE_OP_SINGLE_OP_MANAGER_H_
#define GE_SINGLE_OP_SINGLE_OP_MANAGER_H_

#include <mutex>
#include <unordered_map>
#include <string>

#include "single_op/single_op_model.h"
#include "single_op/stream_resource.h"

namespace ge {
class SingleOpManager {
 public:
  ~SingleOpManager();

  static SingleOpManager &GetInstance() {
    static SingleOpManager instance;
    return instance;
  }

  Status GetOpFromModel(const std::string &key, const ge::ModelData &model_data, void *stream, SingleOp **single_op);

  Status ReleaseResource(void *stream);

 private:
  StreamResource *GetResource(uintptr_t resource_id);
  StreamResource *TryGetResource(uintptr_t resource_id);

  std::mutex mutex_;
  std::unordered_map<uintptr_t, StreamResource *> stream_resources_;
};
}  // namespace ge

#endif  // GE_SINGLE_OP_SINGLE_OP_MANAGER_H_
