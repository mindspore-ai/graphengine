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

#ifndef AIR_CXX_STREAM_EXECUTOR_H
#define AIR_CXX_STREAM_EXECUTOR_H
#include <map>
#include <memory>
#include <mutex>
#include "runtime/base.h"
#include "common/checker.h"
#include "model_v2_executor.h"
namespace gert {
// do not expose the Builder class definition to external api
class ModelV2ExecutorBuilder;
class VISIBILITY_EXPORT StreamExecutor {
 public:
  explicit StreamExecutor(ModelV2ExecutorBuilder *builder);
  StreamExecutor(const StreamExecutor &) = delete;
  StreamExecutor &operator=(const StreamExecutor &) = delete;
  StreamExecutor(StreamExecutor &&) = delete;
  StreamExecutor &operator=(StreamExecutor &&) = delete;
  ~StreamExecutor();
  ModelV2Executor *GetOrCreateLoaded(rtStream_t stream, const ModelExecuteArg &arg) {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto &iter = streams_to_executor_.find(stream);
    if (iter != streams_to_executor_.cend()) {
      return iter->second.get();
    }
    return CreateAndLoad(stream, arg);
  }
  ge::graphStatus Erase(rtStream_t stream);

 private:
  ModelV2Executor *CreateAndLoad(rtStream_t stream, const ModelExecuteArg &arg);

 private:
  std::mutex mutex_;
  ModelV2ExecutorBuilder *builder_;
  std::map<rtStream_t, std::unique_ptr<ModelV2Executor>> streams_to_executor_;
};
}  // namespace gert
#endif  // AIR_CXX_STREAM_EXECUTOR_H
