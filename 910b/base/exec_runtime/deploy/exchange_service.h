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
#ifndef BASE_EXEC_RUNTIME_DEPLOY_EXCHANGE_SERVICE_H_
#define BASE_EXEC_RUNTIME_DEPLOY_EXCHANGE_SERVICE_H_

#include <functional>
#include <memory>
#include "external/ge/ge_api_error_codes.h"
#include "graph/ge_tensor.h"
#include "framework/common/debug/ge_log.h"
#include "runtime/rt_mem_queue.h"

namespace ge {
constexpr uint8_t kNullDataFlagBit = 1U;
constexpr size_t kMaxUserDataSize = 64U;

struct MemQueueAttr {
  uint32_t depth;
  uint32_t work_mode;
  bool is_client = false;
  bool overwrite = false;
};
/// Interfaces for data exchange operations
class ExchangeService {
 public:
  using FillFunc = std::function<Status(void *buffer, size_t size)>;
  struct MsgInfo {
    uint64_t trans_id;
    uint16_t version;
    uint16_t msg_type;
    int32_t ret_code;
    uint64_t start_time;
    uint64_t end_time;
    uint32_t flags;
    uint8_t data_flag;  // 0 bit is null data flag, 1 is null data, 0 is not null data
    char rsv[27];
  };
  struct ControlInfo {
    bool end_of_sequence_flag = false;
    int32_t timeout = 0;
    size_t skip_size = 0U;
    MsgInfo *msg_info = nullptr;
    bool is_shared_input = false;
    int8_t user_data[kMaxUserDataSize] = {};
    bool is_proxy_q = false;
  };
  struct BuffInfo {
    void *addr;
    size_t len;
  };
  ExchangeService() = default;
  ExchangeService(const ExchangeService &) = delete;
  ExchangeService &operator=(const ExchangeService &) = delete;
  virtual ~ExchangeService() = default;

  Status CreateQueue(const int32_t device_id,
                     const std::string &name,
                     const uint32_t depth,
                     const uint32_t work_mode,
                     uint32_t &queue_id) {
    MemQueueAttr mem_queue_attr{};
    mem_queue_attr.depth = depth;
    mem_queue_attr.work_mode = work_mode;
    mem_queue_attr.overwrite = false;
    return CreateQueue(device_id, name, mem_queue_attr, queue_id);
  }

  virtual Status CreateQueue(const int32_t device_id,
                             const std::string &name,
                             const MemQueueAttr &mem_queue_attr,
                             uint32_t &queue_id) = 0;
  virtual Status DestroyQueue(const int32_t device_id, const uint32_t queue_id) = 0;
  virtual Status Enqueue(const int32_t device_id, const uint32_t queue_id, const void *const data,
                         const size_t size, const ControlInfo &control_info) = 0;
  virtual Status Enqueue(int32_t device_id, uint32_t queue_id, size_t size, rtMbufPtr_t m_buf,
                         const ControlInfo &control_info) = 0;
  virtual Status Enqueue(const int32_t device_id, const uint32_t queue_id, const size_t size,
                         const FillFunc &fill_func, const ControlInfo &control_info) = 0;
  virtual Status Enqueue(const int32_t device_id, const uint32_t queue_id, const std::vector<BuffInfo> &buffs,
                         const ControlInfo &control_info) = 0;
  virtual Status Peek(const int32_t device_id, const uint32_t queue_id, size_t &size) = 0;
  virtual Status Dequeue(const int32_t device_id, const uint32_t queue_id, void *const data, const size_t size,
                         ControlInfo &control_info) = 0;
  virtual Status DequeueMbufTensor(const int32_t device_id, const uint32_t queue_id,
                                   std::shared_ptr<AlignedPtr> &aligned_ptr, const size_t size,
                                   ControlInfo &control_info) = 0;
  virtual Status DequeueTensor(const int32_t device_id, const uint32_t queue_id, GeTensor &tensor,
                               ControlInfo &control_info) = 0;
};
}
#endif  // BASE_EXEC_RUNTIME_DEPLOY_EXCHANGE_SERVICE_H_
