/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#ifndef GE_COMMON_PRELOAD_PRE_MODEL_UTILS_H_
#define GE_COMMON_PRELOAD_PRE_MODEL_UTILS_H_

#include "common/model/ge_root_model.h"
#include "common/math/math_util.h"
#include "runtime/rt_preload_task.h"

namespace ge {
struct ArgOffset {
  bool need_refresh;
  uint64_t offset;
};

// TLV args的定义
struct KernelArgsParam {
  uint8_t type;
  ArgOffset offset;
  uint64_t para;
};

struct ZeroCopyParam {
  string batch_label;
  std::vector<uint64_t> offsets;
};

// args分区描述
struct KernelArgsInfo {
  std::shared_ptr<uint8_t> kernel_args_data;
  uint64_t kernel_args_data_size;
};

// args分区 tlv描述
struct KernelArgsDescInfo {
  std::vector<KernelArgsParam> kernel_args_desc_data;
  std::vector<ZeroCopyParam> zero_copy_data;
  // 针对其他的tlv描述， 可以继续在此结构体添加
};

struct PreTaskDescInfo {
  rtCompilerPartinfo_t seq_info;
  KernelArgsInfo kernel_args_info;
  KernelArgsDescInfo kernel_args_desc_info;
};

#pragma pack(push)
#pragma pack(1)
enum WeightType { PREFETCH_EVERYTIME = 0, PREFETCH_ALL = 1 };
struct ModelDescInfo {
  uint32_t task_num;
  uint64_t workspace_size;
  uint64_t weight_size;
  enum WeightType weight_type;
  bool profile_enable = false;
  bool model_interrupt = false;
};

#pragma pack(pop)

struct PreMemInfo {
  int64_t memory_size = 0;
  int64_t logic_memory_base = 0;
  uint8_t *memory_base = nullptr;
  uint64_t memory_type = RT_MEMORY_HBM;
  std::string memory_key;
};

struct PreRuntimeParam {
  uint64_t mem_size = 0UL;
  uint64_t logic_mem_base = 0UL;
  uint64_t weight_size = 0UL;
  uint64_t logic_weight_base = 0UL;
  int64_t zero_copy_size = 0L;
  std::map<uint64_t, PreMemInfo> memory_infos;
  uint32_t stream_num = 0U;
  uint32_t event_num = 0U;
  uint32_t label_num = 0U;
};

class PreModelUtils {
 public:
  struct NodeMemInfo {
    NodeMemInfo(const uint64_t mem_type, const ConstOpDescPtr &op_desc, const size_t index, const std::string &io_type,
                const int64_t size, const int64_t logical_offset)
        : mem_type_(mem_type),
          op_desc_(op_desc),
          index_(index),
          io_type_(io_type),
          size_(size),
          logical_offset_(logical_offset) {}
    uint64_t mem_type_;
    ConstOpDescPtr op_desc_;
    size_t index_;
    std::string io_type_;
    const int64_t size_;
    const int64_t logical_offset_;
  };
  PreModelUtils() = default;
  ~PreModelUtils() = default;
  static std::vector<std::pair<uint64_t, uint32_t>> GetInputDataAddrOffset(const PreRuntimeParam &model_param,
                                                                           const ConstOpDescPtr &op_desc,
                                                                           std::vector<KernelArgsParam> &args_param,
                                                                           std::vector<uint64_t> &args_offset_values);
  static std::vector<std::pair<uint64_t, uint32_t>> GetOutputDataAddrOffset(const PreRuntimeParam &model_param,
                                                                            const ConstOpDescPtr &op_desc,
                                                                            std::vector<KernelArgsParam> &args_param,
                                                                            std::vector<uint64_t> &args_offset_values);
  static std::vector<std::pair<uint64_t, uint32_t>> GetWorkspaceDataAddrOffset(
      const PreRuntimeParam &model_param, const ConstOpDescPtr &op_desc, std::vector<KernelArgsParam> &args_param,
      std::vector<uint64_t> &args_offset_values);
  static void InitRuntimeParams(const GeModelPtr &ge_model, PreRuntimeParam &runtime_param);
  static std::vector<int64_t> GetInputSize(const ConstOpDescPtr &op_desc);
  static std::vector<int64_t> GetOutputSize(const ConstOpDescPtr &op_desc);
  static std::vector<int64_t> GetWorkspaceSize(const ConstOpDescPtr &op_desc);
  static std::vector<int64_t> GetWeightSize(const ConstOpDescPtr &op_desc);

 private:
  static Status RefreshAddressByMemType(const PreRuntimeParam &model_param, const NodeMemInfo &node_mem_info,
                                        KernelArgsParam &arg_param);
  static void RefreshData(const KernelArgsParam &arg_param, std::vector<KernelArgsParam> &args_param,
                          std::vector<uint64_t> &args_offset_values,
                          std::vector<std::pair<uint64_t, uint32_t>> &v_input_data_addr);
  static bool ValidateMemRange(const ConstOpDescPtr &op_desc, const uint64_t total_size, const int64_t offset,
                               const int64_t size);
  static std::vector<PreMemInfo> GetAllMemoryTypeSize(const GeModelPtr &ge_model);
};
}  // namespace ge
#endif  // GE_COMMON_PRELOAD_PRE_MODEL_UTILS_H_