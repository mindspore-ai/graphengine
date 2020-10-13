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

#ifndef INC_FRAMEWORK_COMMON_HELPER_OM_FILE_HELPER_H_
#define INC_FRAMEWORK_COMMON_HELPER_OM_FILE_HELPER_H_

#include <string>
#include <vector>

#include "external/ge/ge_ir_build.h"
#include "framework/common/fmk_types.h"
#include "framework/common/types.h"
#include "framework/common/ge_types.h"

using ProcParam = struct PROC_PARAM;
using std::string;
using std::vector;

namespace ge {
struct ModelPartition {
  ModelPartitionType type;
  uint8_t* data = 0;
  uint32_t size = 0;
};

struct OmFileContext {
  std::vector<ModelPartition> partition_datas_;
  std::vector<char> partition_table_;
  uint32_t model_data_len_;
};

struct SaveParam {
  int32_t encode_mode;
  std::string ek_file;
  std::string cert_file;
  std::string hw_key_file;
  std::string pri_key_file;
  std::string model_name;
};

class OmFileLoadHelper {
 public:
  Status Init(const ge::ModelData &model);

  Status Init(uint8_t *model_data, const uint32_t model_data_size);

  Status GetModelPartition(ModelPartitionType type, ModelPartition &partition);

  OmFileContext context_;

 private:
  Status CheckModelValid(const ge::ModelData &model) const;

  Status LoadModelPartitionTable(uint8_t *model_data, const uint32_t model_data_size);

  bool is_inited_{false};
};

class OmFileSaveHelper {
 public:
  ModelFileHeader &GetModelFileHeader() { return model_header_; }

  uint32_t GetModelDataSize() const { return context_.model_data_len_; }

  ModelPartitionTable *GetPartitionTable();

  Status AddPartition(ModelPartition &partition);

  const std::vector<ModelPartition> &GetModelPartitions() const;

  Status SaveModel(const SaveParam &save_param, const char *target_file,
                   ge::ModelBufferData& model, bool is_offline = true);

  Status SaveModelToFile(const char *output_file, ge::ModelBufferData &model, bool is_offline = true);

  ModelFileHeader model_header_;
  OmFileContext context_;
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_HELPER_OM_FILE_HELPER_H_
