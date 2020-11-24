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

#include "graph/load/new_model_manager/zero_copy_offset.h"

#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/load/new_model_manager/model_utils.h"
#include "graph/load/new_model_manager/zero_copy_task.h"

namespace ge {
namespace {
const uint32_t kDataIndex = 0;
}  // namespace

ZeroCopyOffset::ZeroCopyOffset() {}

ZeroCopyOffset::~ZeroCopyOffset() {}

Status ZeroCopyOffset::InitInputDataInfo(const vector<int64_t> &output_size_list,
                                         const vector<void *> &virtual_addr_list, const OpDescPtr &op_desc,
                                         bool &fusion_flag) {
  GELOGI("[ZCPY] Start to InitInputDataInfo of %s, total_data_size is %ld, virtual_addr is %p",
         op_desc->GetName().c_str(), output_size_list[kDataIndex], virtual_addr_list[kDataIndex]);
  if (output_size_list.empty() || virtual_addr_list.empty() || (output_size_list.size() != virtual_addr_list.size())) {
    GELOGE(PARAM_INVALID, "Data[%s] init failed: Output size is %zu, Output addr is %zu", op_desc->GetName().c_str(),
           output_size_list.size(), virtual_addr_list.size());
    return PARAM_INVALID;
  }

  basic_addr_ = virtual_addr_list[kDataIndex];
  (void)ge::AttrUtils::GetListInt(op_desc, ATTR_ZERO_COPY_BASIC_OFFSET, zero_copy_basic_offset_);
  (void)ge::AttrUtils::GetListInt(op_desc, ATTR_ZERO_COPY_RELATIVE_OFFSET, zero_copy_relative_offset_);
  GE_CHK_BOOL_EXEC(zero_copy_basic_offset_.size() == zero_copy_relative_offset_.size(), return PARAM_INVALID,
                   "basic_offset_size should be equal to relative_offset_size");
  GELOGI("[ZCPY] zero_copy_basic_offset size is %zu", zero_copy_basic_offset_.size());

  int64_t virtual_addr_offset = op_desc->GetOutputOffset().at(kDataIndex);
  GELOGI("virtual_addr_offset is %ld.", virtual_addr_offset);
  IsL2Fusion(zero_copy_basic_offset_, virtual_addr_offset, fusion_flag);

  uint32_t out_count = 0;
  data_size_ = output_size_list[kDataIndex];
  if (!fusion_flag) {
    GELOGI("[ZCPY] %s not set l2_fusion.", op_desc->GetName().c_str());
    out_count++;
    data_info_.emplace_back(output_size_list[kDataIndex], virtual_addr_list[kDataIndex]);
    relative_offset_.emplace_back(0);
    GELOGI("[ZCPY] %s size is %ld, virtual_addr is %p.", op_desc->GetName().c_str(), output_size_list[kDataIndex],
           virtual_addr_list[kDataIndex]);
  } else {
    GELOGI("[ZCPY] set l2_fusion for %s.", op_desc->GetName().c_str());
    for (size_t index = 0; index < zero_copy_basic_offset_.size(); ++index) {
      if (zero_copy_basic_offset_.at(index) == virtual_addr_offset) {
        out_count++;
        uint64_t out_offset =
          reinterpret_cast<uint64_t>(virtual_addr_list[kDataIndex]) + zero_copy_relative_offset_.at(index);
        int64_t real_data_size = ModelUtils::GetOutputSize(op_desc).at(kDataIndex);
        data_info_.emplace_back(real_data_size, reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(out_offset)));
        relative_offset_.emplace_back(zero_copy_relative_offset_.at(index));
        GELOGI("[ZCPY] virtual_addr: %p has been l2-fusion to %lu, need copy data_size is %ld.", basic_addr_,
               out_offset, real_data_size);
      }
    }
  }
  data_count_ = out_count;
  return SUCCESS;
}

Status ZeroCopyOffset::InitOutputDataInfo(const vector<int64_t> &input_size_list,
                                          const vector<void *> &virtual_addr_list, const OpDescPtr &op_desc,
                                          const size_t &idx, bool &fusion_flag) {
  GELOGI("[ZCPY] Start to InitOutputDataInfo of %s.", op_desc->GetName().c_str());
  int64_t size = input_size_list[idx];
  auto tensor_desc = op_desc->GetInputDescPtr(idx);
  GE_CHECK_NOTNULL(tensor_desc);
  if (TensorUtils::GetTensorSizeInBytes(*tensor_desc, size) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "GetTensorSizeInBytes failed!");
    return FAILED;
  }

  GELOGI("Tensor data size: GetSize=%ld, GetTensorSizeInBytes=%ld", input_size_list[idx], size);

  basic_addr_ = virtual_addr_list[idx];
  (void)ge::AttrUtils::GetListInt(op_desc, ATTR_ZERO_COPY_BASIC_OFFSET, zero_copy_basic_offset_);
  (void)ge::AttrUtils::GetListInt(op_desc, ATTR_ZERO_COPY_RELATIVE_OFFSET, zero_copy_relative_offset_);
  GE_CHK_BOOL_EXEC(zero_copy_basic_offset_.size() == zero_copy_relative_offset_.size(), return PARAM_INVALID,
                   "basic_offset_size should be equal to relative_offset_size");
  int64_t virtual_addr_offset = op_desc->GetInputOffset().at(idx);
  GELOGI("virtual_addr_offset is %ld.", virtual_addr_offset);
  IsL2Fusion(zero_copy_basic_offset_, virtual_addr_offset, fusion_flag);

  uint32_t in_count = 0;
  data_size_ = size;
  if (!fusion_flag) {
    GELOGI("[ZCPY] %s not set l2-fusion.", op_desc->GetName().c_str());
    in_count++;
    data_info_.emplace_back(size, virtual_addr_list[idx]);
    // op_desc not set l2fusion when fusion_flag is false
    relative_offset_.emplace_back(0);
    GELOGI("[ZCPY] %s size is %ld, virtual_addr is %p.", op_desc->GetName().c_str(), size, virtual_addr_list[idx]);
  } else {
    GELOGI("[ZCPY] set l2-fusion for %s.", op_desc->GetName().c_str());
    for (size_t index = 0; index < zero_copy_basic_offset_.size(); ++index) {
      if (zero_copy_basic_offset_.at(index) == virtual_addr_offset) {
        in_count++;
        uint64_t in_offset = reinterpret_cast<uint64_t>(virtual_addr_list[idx]) + zero_copy_relative_offset_.at(index);
        int64_t real_data_size = ModelUtils::GetInputSize(op_desc).at(idx);
        data_info_.emplace_back(real_data_size, reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(in_offset)));
        relative_offset_.emplace_back(zero_copy_relative_offset_.at(index));
        GELOGI("[ZCPY] virtual_addr: %p has been l2-fusion from %lu, need copy data_size is %ld.", basic_addr_,
               in_offset, real_data_size);
      }
    }
  }
  data_count_ = in_count;
  return SUCCESS;
}

void ZeroCopyOffset::IsL2Fusion(const vector<int64_t> &fusion_basic_addrs, const int64_t &tensor_offset,
                                bool &fusion_flag) {
  for (size_t fusion_count = 0; fusion_count < fusion_basic_addrs.size(); ++fusion_count) {
    if (fusion_basic_addrs.at(fusion_count) == tensor_offset) {
      fusion_flag = true;
      break;
    }
  }
}

void ZeroCopyOffset::SetInputOutsideAddrs(const vector<int64_t> &output_offset_list, void *addr, const size_t &index,
                                          bool fusion_flag, std::vector<void *> &real_virtual_addrs) {
  GELOGI("[ZCPY] Start to SetInputOutsideAddrs for virtual_addr %p.", addr);
  uint32_t out_count = 0;
  if (!fusion_flag) {
    GELOGI("[ZCPY] not set l2-fusion for virtual_adr %p.", addr);
    out_count++;
    std::map<const void *, std::vector<void *>> addr_mapping;
    addr_mapping[addr] = {};
    outside_addrs_.emplace_back(addr_mapping);
    real_virtual_addrs.emplace_back(addr);
  } else {
    GELOGI("[ZCPY] set l2-fusion for virtual_addr %p.", addr);
    int64_t output_offset = output_offset_list.at(index);
    for (size_t i = 0; i < zero_copy_basic_offset_.size(); ++i) {
      if (zero_copy_basic_offset_.at(i) == output_offset) {
        out_count++;
        void *virtual_addr =
          reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(addr) + zero_copy_relative_offset_.at(i));
        std::map<const void *, std::vector<void *>> addr_mapping;
        addr_mapping[virtual_addr] = {};
        outside_addrs_.emplace_back(addr_mapping);
        real_virtual_addrs.emplace_back(virtual_addr);
        GELOGI("[ZCPY] virtual_addr %p has been fusion to virtual_addr %p.", addr, virtual_addr);
      }
    }
  }
  addr_count_ = out_count;
}

void ZeroCopyOffset::SetOutputOutsideAddrs(const int64_t &input_offset, const bool &fusion_flag, void *addr,
                                           std::vector<void *> &tensor_addrs) {
  GELOGI("[ZCPY] Start to SetOutputOutsideAddrs for virtual_addr %p.", addr);
  uint32_t out_count = 0;
  if (!fusion_flag) {
    GELOGI("[ZCPY] not set l2-fusion for virtual_addr %p.", addr);
    out_count++;
    std::map<const void *, std::vector<void *>> addr_mapping;
    addr_mapping[addr] = {};
    outside_addrs_.emplace_back(addr_mapping);
    tensor_addrs.emplace_back(addr);
  } else {
    GELOGI("[ZCPY] set l2-fusion for virtual_addr %p.", addr);
    for (size_t i = 0; i < zero_copy_basic_offset_.size(); ++i) {
      if (zero_copy_basic_offset_.at(i) == input_offset) {
        out_count++;
        void *virtual_addr =
          reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(addr) + zero_copy_relative_offset_.at(i));
        std::map<const void *, std::vector<void *>> addr_mapping;
        addr_mapping[virtual_addr] = {};
        outside_addrs_.emplace_back(addr_mapping);
        tensor_addrs.emplace_back(virtual_addr);
        GELOGI("[ZCPY] virtual_addr %p has been fusion to virtual_addr %p.", addr, virtual_addr);
      }
    }
  }
  addr_count_ = out_count;
}

bool ZeroCopyOffset::SetOutsideAddrsValue(ZeroCopyTask &zero_copy_task, void *outside_addr, void *args, size_t offset) {
  const auto addr_val = reinterpret_cast<uintptr_t>(outside_addr);
  bool set_batch_label_flag = false;
  for (uint32_t out_count = 0; out_count < GetAddrCount(); ++out_count) {
    auto &addrs_mapping_list = GetOutsideAddrs();
    auto args_addrs = addrs_mapping_list[out_count].find(outside_addr);
    if (args_addrs != addrs_mapping_list[out_count].end()) {
      GE_CHK_STATUS(zero_copy_task.SetTaskArgsOffset(addr_val, offset), "Input args invalid.");
      void *args_val = static_cast<uint8_t *>(args) + offset;
      args_addrs->second.push_back(args_val);
      GELOGI("[ZCPY] set copy input: virtual_addr: 0x%lx, task_addr: %p, args: %p, offset: %zu.", addr_val, args_val,
             args, offset);
      set_batch_label_flag = true;
    }
  }
  return set_batch_label_flag;
}

}  // namespace ge
