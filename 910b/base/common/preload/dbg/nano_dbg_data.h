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

#ifndef AIR_BASE_COMMON_MODEL_NANO_DBG_DATA_H_
#define AIR_BASE_COMMON_MODEL_NANO_DBG_DATA_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include "external/ge/ge_api_types.h"
#include "common/model/ge_root_model.h"

#include "proto/op_mapping.pb.h"
#include "proto/task.pb.h"

namespace ge {
constexpr size_t kAddrLength = sizeof(uint64_t);
constexpr size_t kDumpL1FusionOpMByteSize = 2U * 1042U * 1024U;

struct NanoDbgOutputDesc {
  int32_t data_type;
  int32_t format;
  toolkit::aicpu::dump::AddressType addr_type;
  int32_t original_index;
  int32_t original_data_type;
  int32_t original_format;
  uint64_t addr;
  uint64_t offset;
  uint64_t size;
  std::vector<int64_t> shape_dims;
  std::vector<int64_t> original_shape_dims;
  string original_name;
};

struct NanoDbgInputDesc {
  int32_t data_type;
  int32_t format;
  toolkit::aicpu::dump::AddressType addr_type;
  uint64_t addr;
  uint64_t offset;
  uint64_t size;
  std::vector<int64_t> shape_dims;
  std::vector<int64_t> original_shape_dims;
};

struct NanoDbgWorkspaceDesc {
  toolkit::aicpu::dump::Workspace::SpaceType type;
  uint64_t data_addr;
  uint64_t size;
};

struct NanoDbgBufferDesc {
  toolkit::aicpu::dump::BufferType type;
  uint64_t addr;
  uint64_t size;
};

struct NanoDbgMemInfoDesc {
  uint64_t input_mem_size;
  uint64_t output_mem_size;
  uint64_t weight_mem_size;
  uint64_t workspace_mem_size;
  uint64_t total_mem_size;
};

struct NanoDbgOpDesc {
  uint32_t task_id;
  uint32_t stream_id;
  uint32_t logic_stream_id;
  string op_name;
  string op_type;
  toolkit::aicpu::dump::Task::TaskType task_type;
  uint32_t block_dim;
  std::vector<string> original_op_names;
  bool datadump_is_multiop;
  string L1_fusion_sub_graph_no;
  std::vector<NanoDbgInputDesc> input_list;
  std::vector<NanoDbgOutputDesc> output_list;
  std::vector<NanoDbgWorkspaceDesc> workspace_list;
  std::vector<NanoDbgBufferDesc> buffer_list;
  std::vector<NanoDbgMemInfoDesc> mem_info_list;
};

class NanoDbgData {
 public:
  explicit NanoDbgData(const GeModelPtr &ge_model, const std::unordered_map<int64_t, uint32_t> zerocopy_info);
  ~NanoDbgData() = default;
  NanoDbgData &operator=(const NanoDbgData &dbg) & = delete;
  NanoDbgData(const NanoDbgData &dbg) = delete;

  Status Init();

  const void *GetDbgData() const {
    return static_cast<const void *>(buff_.data());
  }

  uint64_t GetDbgDataSize() const {
    return static_cast<uint64_t>(buff_size_ - des_size_);
  }

 private:
  void InitNodes();
  OpDescPtr GetOpByIndex(const uint32_t index) const {
    if (op_map_.find(index) == op_map_.end()) {
      return nullptr;
    }
    return op_map_.at(index);
  }

  Status InitDbgData();
  Status AddDbgOp(const domi::TaskDef &task_def);
  Status GenDbgInput(const GeTensorDesc &tensor_desc, NanoDbgInputDesc &dbg_input) const;
  Status AddDbgInput(const OpDescPtr &op_desc, NanoDbgOpDesc &dbg_op, const uint32_t &op_index);
  Status GenDbgOutput(const GeTensorDesc &tensor_desc, NanoDbgOutputDesc &dbg_output) const;
  Status AddDbgOutput(const OpDescPtr &op_desc, NanoDbgOpDesc &dbg_op, const uint32_t &op_index);
  Status AddDbgWorkspace(const OpDescPtr &op_desc, NanoDbgOpDesc &dbg_op) const;
  Status AddDbgBuffer(NanoDbgOpDesc &dbg_op);
  Status AddDbgMemInfo(const OpDescPtr &op_desc, NanoDbgOpDesc &dbg_op) const;
  void GenMemType(const int64_t id, const ge::NodePtr &node);
  void SaveMemType(std::map<int64_t, std::vector<toolkit::aicpu::dump::AddressType>> &mem_types, const int64_t id,
                   const ge::NodePtr &node) const;

  Status InitDbgTlv();
  void GenOpOriNameLen(const std::vector<string> &name_list);
  void GenInputDescLen(const std::vector<NanoDbgInputDesc> &input_list);
  void GenOutputDescLen(const std::vector<NanoDbgOutputDesc> &output_list);
  void GenWorkspaceDescLen(const std::vector<NanoDbgWorkspaceDesc> &workspace_list);
  void GenDbgPartitionLen();
  Status SaveDbgHead();
  Status SaveDbgStrTlv(const string &str, const uint32_t type);
  Status SaveDbgVecTlv(const std::vector<string> &vec, const uint32_t type);
  Status SaveDbgVecTlv(const std::vector<int64_t> &vec, const uint32_t type);
  Status SaveDbgMemInfoTlv(const std::vector<NanoDbgMemInfoDesc> &mem_info_list);
  Status SaveDbgBufTlv(const std::vector<NanoDbgBufferDesc> &buffer_list);
  Status SaveDbgInputDescTlv(const std::vector<NanoDbgInputDesc> &input_list);
  Status SaveDbgOutputDescTlv(const std::vector<NanoDbgOutputDesc> &output_list);
  Status SaveDbgWorkspaceDescTlv(const std::vector<NanoDbgWorkspaceDesc> &workspace_list);
  Status SaveDbgL1Tlv();
  Status SaveDbgPartition();

  bool need_generate_op_buffer_ = false;
  GeModelPtr ge_model_;
  std::map<uint32_t, OpDescPtr> op_map_;
  std::vector<NanoDbgOpDesc> op_list_;
  string model_name_;
  uint8_t *des_addr_ = nullptr;
  size_t des_size_ = 0U;
  size_t buff_size_ = 0U;
  std::vector<uint8_t> buff_;
  std::unordered_map<int64_t, uint32_t> zerocopy_info_;

  std::map<int64_t, std::vector<toolkit::aicpu::dump::AddressType>> output_mem_types_;
  std::map<int64_t, std::vector<toolkit::aicpu::dump::AddressType>> input_mem_types_;
};
} // namespace ge
#endif  // AIR_BASE_COMMON_MODEL_NANO_DBG_DATA_H_