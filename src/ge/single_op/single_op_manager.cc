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

#include "single_op/single_op_manager.h"

#include <mutex>
#include <string>

#include "runtime/dev.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY
SingleOpManager::~SingleOpManager() {
  for (auto &it : stream_resources_) {
    delete it.second;
    it.second = nullptr;
  }
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY
Status SingleOpManager::GetOpFromModel(const std::string &model_name,
                                       const ModelData &model_data,
                                       void *stream,
                                       SingleOp **single_op) {
  if (single_op == nullptr) {
    GELOGE(PARAM_INVALID, "single op is null");
    return PARAM_INVALID;
  }
  uintptr_t resource_id;
  // runtime uses NULL to denote a default stream for each device
  if (stream == nullptr) {
    // use device id as resource key instead
    int32_t dev_id = 0;
    auto rt_err = rtGetDevice(&dev_id);
    if (rt_err != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Get current device id failed. ret = %d", static_cast<int>(rt_err));
      return RT_FAILED;
    }

    GELOGI("GetOpFromModel with default stream. device id = %d", dev_id);
    resource_id = static_cast<uintptr_t>(dev_id);
  } else {
    resource_id = reinterpret_cast<uintptr_t>(stream);
  }

  GELOGI("GetOpFromModel in. model name = %s, resource id = 0x%lx",
         model_name.c_str(),
         static_cast<uint64_t>(resource_id));

  StreamResource *res = GetResource(resource_id);
  if (res == nullptr) {
      GELOGE(MEMALLOC_FAILED, "GetResource failed");
      return MEMALLOC_FAILED;
  }

  SingleOp *op = res->GetOperator(model_data.model_data);
  if (op != nullptr) {
    GELOGD("Got operator from stream cache");
    *single_op = op;
    return SUCCESS;
  }

  SingleOpModel model(model_name, model_data.model_data, model_data.model_len);
  auto ret = model.Init();
  if (ret != SUCCESS) {
    GELOGE(ret, "Init model failed. model = %s, ret = %u", model_name.c_str(), ret);
    return ret;
  }

  auto *new_op = new(std::nothrow)SingleOp();
  if (new_op == nullptr) {
    GELOGE(MEMALLOC_FAILED, "new SingleOp failed");
    return MEMALLOC_FAILED;
  }

  GELOGI("To build operator: %s", model_name.c_str());
  ret = model.BuildOp(*res, *new_op);
  if (ret != SUCCESS) {
    GELOGE(ret, "Build op failed. op = %s, resource id = 0x%lx, ret = %u",
           model_name.c_str(),
           static_cast<uint64_t>(resource_id),
           ret);
    delete new_op;
    new_op = nullptr;
    return ret;
  }

  // stream is nullable
  new_op->SetStream(stream);
  res->CacheOperator(model_data.model_data, new_op);
  *single_op = new_op;
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY
Status SingleOpManager::ReleaseResource(void *stream) {
  auto resource_id = reinterpret_cast<uintptr_t>(stream);
  GELOGI("ReleaseResource in. resource id = 0x%lx", static_cast<uint64_t>(resource_id));
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = stream_resources_.find(resource_id);
  if (it == stream_resources_.end()) {
    return SUCCESS;
  }
  delete it->second;
  it->second = nullptr;
  (void)stream_resources_.erase(it);
  return SUCCESS;
}

StreamResource *SingleOpManager::GetResource(uintptr_t resource_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = stream_resources_.find(resource_id);
  StreamResource *res = nullptr;
  if (it == stream_resources_.end()) {
    res = new (std::nothrow)StreamResource();
    if (res != nullptr) {
      stream_resources_.emplace(resource_id, res);
    }
  } else {
    res = it->second;
  }

  return res;
}

StreamResource *SingleOpManager::TryGetResource(uintptr_t resource_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = stream_resources_.find(resource_id);
  if (it == stream_resources_.end()) {
    return nullptr;
  }

  return it->second;
}
}  // namespace ge
