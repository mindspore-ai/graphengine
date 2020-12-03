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

#include <memory>
#include "hybrid_davinci_model.h"
#include "hybrid/model/hybrid_model.h"
#include "hybrid/executor/hybrid_model_async_executor.h"
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {
class HybridDavinciModel::Impl {
 public:
  explicit Impl(GeRootModelPtr ge_model) : model_(std::move(ge_model)), executor_(&model_) {
  }

  ~Impl() {
    NodeExecutorManager::GetInstance().FinalizeExecutors();
  }

  Status Init() {
    GE_CHK_STATUS_RET(NodeExecutorManager::GetInstance().EnsureInitialized(), "Failed to initialize executors");
    GE_CHK_STATUS_RET(model_.Init(), "Failed to init model.")
    GE_CHK_STATUS_RET(executor_.Init(), "Failed to init model executor.")
    return SUCCESS;
  }

  Status Execute(const std::vector<DataBuffer> &inputs,
                 const std::vector<GeTensorDesc> &input_desc,
                 std::vector<DataBuffer> &outputs,
                 std::vector<GeTensorDesc> &output_desc,
                 rtStream_t stream) {
    return executor_.Execute(inputs, input_desc, outputs, output_desc);
  }

  Status Execute(const vector<GeTensor> &inputs, vector<GeTensor> &outputs) {
    return executor_.Execute(inputs, outputs);
  }

  Status ModelRunStart() {
    return executor_.Start(listener_);
  }

  Status ModelRunStop() {
    return executor_.Stop();
  }

  Status EnqueueData(const std::shared_ptr<InputDataWrapper> &data) {
    return executor_.EnqueueData(data);
  }

  void SetListener(const shared_ptr<ModelListener> &listener) {
    listener_ = listener;
  }

  void SetModelId(uint32_t model_id) {
    executor_.SetModelId(model_id);
    model_.SetModelId(model_id);
  }

  void SetDeviceId(uint32_t device_id) {
    model_.SetDeviceId(device_id);
    executor_.SetDeviceId(device_id);
  }

  uint64_t GetSessionId() {
    return model_.GetSessionId();
  }

  Status GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type) {
    return model_.GetDynamicBatchInfo(batch_info, dynamic_type);
  }

  void GetUserDesignateShapeOrder(std::vector<std::string> &user_input_shape_order) {
    model_.GetUserDesignateShapeOrder(user_input_shape_order);
  }

  void GetModelAttr(std::vector<std::string> &dynamic_output_shape_info) {
    model_.GetModelAttr(dynamic_output_shape_info);
  }

  Status GetInputOutputDescInfo(vector<InputOutputDescInfo> &input_desc,
                                vector<InputOutputDescInfo> &output_desc,
                                std::vector<uint32_t> &input_formats,
                                std::vector<uint32_t> &output_formats) {
    return model_.GetInputOutputDescInfo(input_desc, output_desc, input_formats, output_formats);
  }

  void SetModelDescVersion(bool is_new_model_desc) {
    model_.SetModelDescVersion(is_new_model_desc);
  }

 private:
  std::shared_ptr<ModelListener> listener_;
  HybridModel model_;
  HybridModelAsyncExecutor executor_;
};

HybridDavinciModel::~HybridDavinciModel() {
  delete impl_;
}

unique_ptr<HybridDavinciModel> HybridDavinciModel::Create(const GeRootModelPtr &ge_root_model) {
  auto instance = unique_ptr<HybridDavinciModel>(new (std::nothrow)HybridDavinciModel());
  if (instance != nullptr) {
    instance->impl_ = new (std::nothrow) HybridDavinciModel::Impl(ge_root_model);
    if (instance->impl_ != nullptr) {
      return instance;
    }
  }

  return nullptr;
}

Status HybridDavinciModel::Init() {
  GE_CHECK_NOTNULL(impl_);
  return impl_->Init();
}

Status HybridDavinciModel::Execute(const std::vector<DataBuffer> &inputs,
                                   const std::vector<GeTensorDesc> &input_desc,
                                   std::vector<DataBuffer> &outputs,
                                   std::vector<GeTensorDesc> &output_desc, rtStream_t stream) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->Execute(inputs, input_desc, outputs, output_desc, stream);
}

Status HybridDavinciModel::Execute(const vector<GeTensor> &inputs, vector<GeTensor> &outputs) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->Execute(inputs, outputs);
}

Status HybridDavinciModel::ModelRunStart() {
  GE_CHECK_NOTNULL(impl_);
  return impl_->ModelRunStart();
}

Status HybridDavinciModel::ModelRunStop() {
  GE_CHECK_NOTNULL(impl_);
  return impl_->ModelRunStop();
}

Status HybridDavinciModel::EnqueueData(const shared_ptr<InputDataWrapper> &data) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->EnqueueData(data);
}

void HybridDavinciModel::SetListener(const shared_ptr<ModelListener> &listener) {
  if (impl_ != nullptr) {
    impl_->SetListener(listener);
  }
}

void HybridDavinciModel::SetModelId(uint32_t model_id) {
  if (impl_ != nullptr) {
    impl_->SetModelId(model_id);
  }
}

void HybridDavinciModel::SetDeviceId(uint32_t device_id) {
  if (impl_ != nullptr) {
    impl_->SetDeviceId(device_id);
  }
}

Status HybridDavinciModel::GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->GetDynamicBatchInfo(batch_info, dynamic_type);
}

void HybridDavinciModel::GetUserDesignateShapeOrder(std::vector<std::string> &user_input_shape_order) {
  if (impl_ != nullptr) {
    impl_->GetUserDesignateShapeOrder(user_input_shape_order);
  }
}

void HybridDavinciModel::GetModelAttr(std::vector<std::string> &dynamic_output_shape_info) {
  if (impl_ != nullptr) {
    impl_->GetModelAttr(dynamic_output_shape_info);
  }
}

Status HybridDavinciModel::GetInputOutputDescInfo(vector<InputOutputDescInfo> &input_desc,
                                                  vector<InputOutputDescInfo> &output_desc,
                                                  std::vector<uint32_t> &input_formats,
                                                  std::vector<uint32_t> &output_formats) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->GetInputOutputDescInfo(input_desc, output_desc, input_formats, output_formats);
}

void HybridDavinciModel::SetModelDescVersion(bool is_new_model_desc) {
  if (impl_ != nullptr) {
    impl_->SetModelDescVersion(is_new_model_desc);
  }
}

uint64_t HybridDavinciModel::GetSessionId() {
  GE_CHECK_NOTNULL(impl_);
  return impl_->GetSessionId();
}
}  // namespace hybrid
}  // namespace ge
