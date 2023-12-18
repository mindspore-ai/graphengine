/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef AIR_BASE_COMMON_PRELOAD_MODEL_TASK_DESC_PARTITION_H_
#define AIR_BASE_COMMON_PRELOAD_MODEL_TASK_DESC_PARTITION_H_
#include "common/preload/partition/model_partition_base.h"

namespace ge {
class ModelTaskDescPartition : public ModelPartitionBase {
 public:
  ModelTaskDescPartition() = default;
  virtual ~ModelTaskDescPartition() override = default;
  virtual Status Init(const GeModelPtr &ge_model, const uint8_t type = 0U) override;
};
}  // namespace ge
#endif