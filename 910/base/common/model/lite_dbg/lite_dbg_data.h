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

#ifndef AIR_BASE_COMMON_MODEL_LITE_DBG_DATA_H_
#define AIR_BASE_COMMON_MODEL_LITE_DBG_DATA_H_

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
#include "framework/common/tlv/lite_dbg_desc.h"

using Status = domi::Status;
namespace ge {
struct LiteDbgOutputDesc {
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

struct LiteDbgInputDesc {
  int32_t data_type;
  int32_t format;
  toolkit::aicpu::dump::AddressType addr_type;
  uint64_t addr;
  uint64_t offset;
  uint64_t size;
  std::vector<int64_t> shape_dims;
  std::vector<int64_t> original_shape_dims;
};

struct LiteDbgWorkspaceDesc {
  toolkit::aicpu::dump::Workspace::SpaceType type;
  uint64_t data_addr;
  uint64_t size;
};

struct LiteDbgBufferDesc {
  toolkit::aicpu::dump::BufferType type;
  uint64_t addr;
  uint64_t size;
};

struct LiteDbgMemInfoDesc {
  uint64_t input_mem_size;
  uint64_t output_mem_size;
  uint64_t weight_mem_size;
  uint64_t workspace_mem_size;
  uint64_t total_mem_size;
};

struct LiteDbgOpDesc {
  uint32_t task_id;
  uint32_t stream_id;
  uint32_t logic_stream_id;
  string op_name;
  string op_type;
  toolkit::aicpu::dump::Task::TaskType task_type;
  uint32_t block_dim;
  string original_op_names;
  bool datadump_is_multiop;
  string L1_fusion_sub_graph_no;
  std::vector<LiteDbgInputDesc> input_list;
  std::vector<LiteDbgOutputDesc> output_list;
  std::vector<LiteDbgWorkspaceDesc> workspace_list;
  std::vector<LiteDbgBufferDesc> buffer_list;
  std::vector<LiteDbgMemInfoDesc> mem_info_list;
};

class LiteDbgData {
 public:
  enum {
    kAddrLength = sizeof(uint64_t),
    kDumpL1FusionOpMByteSize = 2U * 1042U * 1024U
  };

  explicit LiteDbgData(const GeModelPtr &ge_model);
  ~LiteDbgData() = default;
  LiteDbgData &operator=(const LiteDbgData &dbg) = delete;
  LiteDbgData(const LiteDbgData &dbg) = delete;

  Status Init();

  const void *GetDbgData() {
    return static_cast<const void *>(buff_.get());
  }

  uint64_t GetDbgDataSize() {
    return static_cast<uint64_t>(buff_size_ - des_size_);
  }

 private:
  void InitNodes();
  OpDescPtr GetOpByIndex(uint32_t index) const {
    if (op_map_.find(index) == op_map_.end()) {
      return nullptr;
    }
    return op_map_.at(index);
  }

  Status InitDbgData();
  Status AddDbgOp(const domi::TaskDef &task_def);
  Status GenDbgInput(const GeTensorDesc &tensor_desc, LiteDbgInputDesc &dbg_input);
  Status AddDbgInput(const OpDescPtr &op_desc, LiteDbgOpDesc &dbg_op);
  Status GenDbgOutput(const GeTensorDesc &tensor_desc, LiteDbgOutputDesc &dbg_output);
  Status AddDbgOutput(const OpDescPtr &op_desc, LiteDbgOpDesc &dbg_op);
  Status AddDbgWorkspace(const OpDescPtr &op_desc, LiteDbgOpDesc &dbg_op);
  Status AddDbgBuffer(LiteDbgOpDesc &dbg_op);
  Status AddDbgMemInfo(const OpDescPtr &op_desc, LiteDbgOpDesc &dbg_op);

  Status InitDbgTlv();
  void GenInputDescLen(std::vector<LiteDbgInputDesc> &input_list);
  void GenOutputDescLen(std::vector<LiteDbgOutputDesc> &output_list);
  void GenWorkspaceDescLen(std::vector<LiteDbgWorkspaceDesc> &workspace_list);
  void GenDbgPartitionLen();
  Status SaveDbgHead();
  Status SaveDbgStrTlv(string &str, uint16_t type);
  Status SaveDbgVecTlv(std::vector<int64_t> &vec, uint16_t type);
  Status SaveDbgMemInfoTlv(std::vector<LiteDbgMemInfoDesc> &mem_info_list);
  Status SaveDbgBufTlv(std::vector<LiteDbgBufferDesc> &buffer_list);
  Status SaveDbgInputDescTlv(std::vector<LiteDbgInputDesc> &input_list);
  Status SaveDbgOutputDescTlv(std::vector<LiteDbgOutputDesc> &output_list);
  Status SaveDbgWorkspaceDescTlv(std::vector<LiteDbgWorkspaceDesc> &workspace_list);
  Status SaveDbgL1Tlv();
  Status SaveDbgPartition();

  bool need_generate_op_buffer_ = false;
  GeModelPtr ge_model_;
  std::map<uint32_t, OpDescPtr> op_map_;
  std::vector<LiteDbgOpDesc> op_list_;
  string model_name_;
  uint8_t *des_addr_ = nullptr;
  size_t des_size_ = 0U;
  size_t buff_size_ = 0U;
  std::shared_ptr<uint8_t> buff_ = nullptr;
};
} // namespace ge
#endif  // AIR_BASE_COMMON_MODEL_LITE_DBG_DATA_H_