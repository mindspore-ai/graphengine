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

#ifndef INC_FRAMEWORK_COMMON_HELPER_OM_FILE_HELPER_H_
#define INC_FRAMEWORK_COMMON_HELPER_OM_FILE_HELPER_H_

#include <string>
#include <vector>

#include "external/ge/ge_ir_build.h"
#include "framework/common/types.h"
#include "framework/common/ge_types.h"
#include "graph/model.h"

namespace ge {
struct ModelPartition {
  ModelPartitionType type;
  const uint8_t *data = nullptr;
  uint64_t size = 0UL;
};

struct OmFileContext {
  std::vector<ModelPartition> partition_datas_;
  std::vector<char_t> partition_table_;
  uint64_t model_data_len_ = 0UL;
};

class GE_FUNC_VISIBILITY OmFileLoadHelper {
 public:
  Status Init(const ModelData &model);

  Status Init(uint8_t *const model_data, const uint32_t model_data_size);

  Status Init(uint8_t *const model_data, const uint32_t model_data_size, const uint32_t model_num);

  Status Init(uint8_t *const model_data, const uint64_t model_data_size,
              const ModelFileHeader *file_header = nullptr);

  Status Init(uint8_t *const model_data,
              const uint64_t model_data_size,
              const uint32_t model_num,
              const ModelFileHeader *file_header = nullptr);

  Status GetModelPartition(const ModelPartitionType type, ModelPartition &partition);

  Status GetModelPartition(const ModelPartitionType type, ModelPartition &partition, const size_t model_index) const;

  const std::vector<ModelPartition> &GetModelPartitions(const size_t model_index) const;

  bool IsPartitionTableNumValid(const uint32_t partition_num, const uint32_t increase_partition_num) const;

  Status CheckModelCompatibility(const Model &model) const;

  OmFileContext context_;

  std::vector<OmFileContext> model_contexts_;

 private:
  Status LoadModelPartitionTable(const uint8_t *const model_data, const uint64_t model_data_size,
                                 const size_t model_index, size_t &mem_offset,
                                 const ModelFileHeader *file_header = nullptr);

  Status LoadModelPartitionTable(const uint8_t *const model_data,
                                 const uint64_t model_data_size,
                                 const uint32_t model_num,
                                 const ModelFileHeader *file_header = nullptr);

  bool is_inited_{false};
};

class GE_FUNC_VISIBILITY OmFileSaveHelper {
 public:
  ModelFileHeader &GetModelFileHeader() { return model_header_; }

  ModelPartitionTable *GetPartitionTable();

  Status AddPartition(const ModelPartition &partition);

  Status AddPartition(const ModelPartition &partition, const size_t cur_index);

  Status SaveModel(const char_t *const output_file, ModelBufferData &model, const bool is_offline = true);

  ModelPartitionTable *GetPartitionTable(const size_t cur_ctx_index);

 private:
  ModelFileHeader model_header_;
  std::vector<OmFileContext> model_contexts_;
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_HELPER_OM_FILE_HELPER_H_
