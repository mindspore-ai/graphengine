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

#include "graph/load/new_model_manager/zero_copy_task.h"

#include "graph/load/new_model_manager/model_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"

namespace ge {
const char *const kDefaultBatchLable = "Batch_default";

ZeroCopyTask::ZeroCopyTask(const string &name, uint8_t *args, size_t size)
    : name_(name), args_addr_(args), args_size_(size), is_updated_(false) {}

ZeroCopyTask::~ZeroCopyTask() { args_addr_ = nullptr; }

/**
 * @ingroup ge
 * @brief Set Task zero copy addr info.
 * @param [in] addr: task addr value.
 * @param [in] offset: saved offset in task args.
 * @return: 0 SUCCESS / others FAILED
 */
Status ZeroCopyTask::SetTaskArgsOffset(uintptr_t addr, size_t offset) {
  if (offset + sizeof(uintptr_t) > args_size_) {
    GELOGE(FAILED, "[ZCPY] %s set task args failed, args size: %zu, offset: %zu", name_.c_str(), args_size_, offset);
    return FAILED;  // unexpected error, need fix.
  }

  auto it = task_addr_offset_.find(addr);
  if (it == task_addr_offset_.end()) {
    task_addr_offset_[addr] = {offset};
  } else {
    it->second.push_back(offset);
  }

  GELOGI("[ZCPY] %s set task, addr: 0x%lx, args: %p, size: %zu, offset: %zu", name_.c_str(), addr, args_addr_,
         args_size_, offset);
  return SUCCESS;
}

/**
 * @ingroup ge
 * @brief Save orignal data of task args.
 * @param [in] info: task args orignal data.
 * @param [in] size: args size.
 * @return: void
 */
void ZeroCopyTask::SetOriginalArgs(const void *info, size_t size) {
  GE_CHECK_NOTNULL_JUST_RETURN(info);
  const uint8_t *data = static_cast<const uint8_t *>(info);
  args_info_.assign(data, data + size);

  GELOGI("[ZCPY] %s set info, args: %p, args size: %zu, info size: %zu", name_.c_str(), args_addr_, args_size_, size);
}

/**
 * @ingroup ge
 * @brief Check is dynamic batch node.
 * @param [in] addr: virtual address value from Op.
 * @param [in] data: data buffer from user.
 * @param [in] batch_addrs: dynamic batch addr info.
 * @param [in] batch_label: batch label.
 * @return: true / false
 */
bool ZeroCopyTask::CheckDynamicBatch(const map<string, set<uintptr_t>> &batch_addrs, const string &batch_label,
                                     uintptr_t addr) {
  // Used for dynamic batch / resolution scene
  set<uintptr_t> dynamic_input_addrs;
  auto dynamic_input_iter = batch_addrs.find(batch_label);
  if (dynamic_input_iter != batch_addrs.end()) {
    dynamic_input_addrs = dynamic_input_iter->second;
  }

  set<uintptr_t> fix_input_addrs;
  auto fix_input_iter = batch_addrs.find(kDefaultBatchLable);
  if (fix_input_iter != batch_addrs.end()) {
    fix_input_addrs = fix_input_iter->second;
  }

  if (fix_input_addrs.empty()) {
    if (!dynamic_input_addrs.empty() && dynamic_input_addrs.find(addr) == dynamic_input_addrs.end()) {
      return false;
    }
  } else {
    if (!dynamic_input_addrs.empty() && dynamic_input_addrs.find(addr) == dynamic_input_addrs.end() &&
        fix_input_addrs.find(addr) == fix_input_addrs.end()) {
      return false;
    }
  }

  return true;
}

/**
 * @ingroup ge
 * @brief Set user data addr to Task param.
 * @param [in] addr: virtual address value from Op.
 * @param [in] data: data buffer from user.
 * @param [in] batch_addrs: dynamic batch addr info.
 * @param [in] batch_label: batch label.
 * @return: void
 */
Status ZeroCopyTask::UpdateTaskParam(uintptr_t addr, const DataBuffer &data,
                                     const map<string, set<uintptr_t>> &batch_addrs, const string &batch_label) {
  for (auto pair : task_addr_offset_) {
    if (pair.first != addr) {
      continue;
    }

    uint8_t *args_info = args_info_.data();
    for (auto offset : pair.second) {
      if (!CheckDynamicBatch(batch_addrs, batch_label, reinterpret_cast<uintptr_t>(args_addr_ + offset))) {
        continue;
      }

      auto dst_addr = static_cast<uint8_t *>(data.data);
      auto dst_size = static_cast<uint64_t>(data.length);
      if (ModelUtils::ConvertVirtualAddressToPhysical(dst_addr, dst_size, dst_addr) != SUCCESS) {
        GELOGE(FAILED, "[ZCPY] Convert virtual address to physical for dst_addr failed.");
        return FAILED;
      }

      GELOGI("[ZCPY] %s update task, args: %p, size: %zu, offset: %zu, addr: 0x%lx, length: %u", name_.c_str(),
             args_addr_, args_size_, offset, addr, data.length);
      *(uintptr_t *)(args_info + offset) = reinterpret_cast<uintptr_t>(dst_addr);
      is_updated_ = true;
    }
  }

  return SUCCESS;
}

/**
 * @ingroup ge
 * @brief Update task param to device.
 * @param [in] stream: Stream for asychronous update.
 * @return: 0 SUCCESS / others FAILED
 */
Status ZeroCopyTask::DistributeParam(rtStream_t stream) {
  if (!is_updated_) {
    return SUCCESS;
  }

  is_updated_ = false;
  GE_CHECK_NOTNULL(args_addr_);
  rtError_t rt_err = RT_ERROR_NONE;
  if (stream != nullptr) {
    rt_err =
      rtMemcpyAsync(args_addr_, args_size_, args_info_.data(), args_info_.size(), RT_MEMCPY_HOST_TO_DEVICE_EX, stream);
  } else {
    __builtin_prefetch(args_addr_);
    rt_err = rtMemcpy(args_addr_, args_size_, args_info_.data(), args_info_.size(), RT_MEMCPY_HOST_TO_DEVICE);
  }

  if (rt_err != RT_ERROR_NONE) {
    GELOGE(FAILED, "[ZCPY] %s distribute task param failed, error=0x%x", name_.c_str(), rt_err);
    return FAILED;
  }

  GELOGI("[ZCPY] %s refresh task args success, args: %p, size: %zu, args_info_: %p, length: %zu", name_.c_str(),
         args_addr_, args_size_, args_info_.data(), args_info_.size());
  return SUCCESS;
}
}  // namespace ge
