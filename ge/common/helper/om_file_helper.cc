/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "framework/common/helper/om_file_helper.h"

#include <string>
#include <vector>
#include "common/math/math_util.h"
#include "common/auth/file_saver.h"
#include "framework/common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/util.h"

using std::string;

namespace {
const int32_t kOptionalNum = 2;
}
namespace ge {
// For Load
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status OmFileLoadHelper::Init(const ge::ModelData &model) {
  if (CheckModelValid(model) != SUCCESS) {
    return FAILED;
  }
  uint32_t model_data_size = model.model_len - sizeof(ModelFileHeader);
  uint8_t *model_data = static_cast<uint8_t *>(model.model_data) + sizeof(ModelFileHeader);
  Status ret = Init(model_data, model_data_size);
  return ret;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status OmFileLoadHelper::Init(uint8_t *model_data,
                                                                               const uint32_t model_data_size) {
  Status status = LoadModelPartitionTable(model_data, model_data_size);
  if (status != SUCCESS) {
    return status;
  }
  is_inited_ = true;
  return SUCCESS;
}

// Use both
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status OmFileLoadHelper::GetModelPartition(ModelPartitionType type,
                                                                                            ModelPartition &partition) {
  if (!is_inited_) {
    GELOGE(PARAM_INVALID, "OmFileLoadHelper has not been initialized!");
    return PARAM_INVALID;
  }

  bool found = false;
  for (ModelPartition &part : context_.partition_datas_) {
    if (part.type == type) {
      partition = part;
      found = true;
      break;
    }
  }

  if (!found) {
    if (type != ModelPartitionType::TBE_KERNELS && type != ModelPartitionType::WEIGHTS_DATA &&
        type != ModelPartitionType::CUST_AICPU_KERNELS) {
      GELOGE(FAILED, "GetModelPartition:type:%d is not in partition_datas!", static_cast<int>(type));
      return FAILED;
    }
  }
  return SUCCESS;
}

Status OmFileLoadHelper::CheckModelValid(const ge::ModelData &model) const {
  // Parameter validity check
  if (model.model_data == nullptr) {
    GELOGE(PARAM_INVALID, "Model_data must not be null!");
    return PARAM_INVALID;
  }

  // Model length too small
  if (model.model_len < (sizeof(ModelFileHeader) + sizeof(ModelPartitionTable))) {
    GELOGE(PARAM_INVALID,
        "Invalid model. length[%u] < sizeof(ModelFileHeader)[%zu] + sizeof(ModelPartitionTable)[%zu].",
        model.model_len, sizeof(ModelFileHeader), sizeof(ModelPartitionTable));
    return PARAM_INVALID;
  }

  // Get file header
  auto model_header = reinterpret_cast<ModelFileHeader *>(model.model_data);
  // Determine whether the file length and magic number match
  if ((model_header->length != model.model_len - sizeof(ModelFileHeader)) ||
      (MODEL_FILE_MAGIC_NUM != model_header->magic)) {
    GELOGE(PARAM_INVALID,
           "Invalid model. file_header->length[%u] + sizeof(ModelFileHeader)[%zu] != model->model_len[%u] || "
            "MODEL_FILE_MAGIC_NUM[%u] != file_header->magic[%u]",
           model_header->length, sizeof(ModelFileHeader), model.model_len, MODEL_FILE_MAGIC_NUM, model_header->magic);
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status OmFileLoadHelper::LoadModelPartitionTable(uint8_t *model_data, const uint32_t model_data_size) {
  if (model_data == nullptr) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_ADDR_INVALID, "Param model_data must not be null!");
    return ACL_ERROR_GE_EXEC_MODEL_ADDR_INVALID;
  }
  // Init partition table
  auto partition_table = reinterpret_cast<ModelPartitionTable *>(model_data);
  // Davinici model partition include graph-info  weight-info  task-info  tbe-kernel :
  // Original model partition include graph-info
  if ((partition_table->num != PARTITION_SIZE) && (partition_table->num != (PARTITION_SIZE - 1)) &&
      (partition_table->num != (PARTITION_SIZE - kOptionalNum)) && (partition_table->num != 1)) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_PARTITION_NUM_INVALID, "Invalid partition_table->num:%u", partition_table->num);
    return ACL_ERROR_GE_EXEC_MODEL_PARTITION_NUM_INVALID;
  }
  size_t mem_offset = SIZE_OF_MODEL_PARTITION_TABLE(*partition_table);
  GELOGI("ModelPartitionTable num :%u, ModelFileHeader length :%zu, ModelPartitionTable length :%zu",
         partition_table->num, sizeof(ModelFileHeader), mem_offset);
  if (model_data_size <= mem_offset) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID, "invalid model data, partition_table->num:%u, model data size %u",
           partition_table->num, model_data_size);
    return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
  }
  for (uint32_t i = 0; i < partition_table->num; i++) {
    ModelPartition partition;
    partition.size = partition_table->partition[i].mem_size;
    partition.data = model_data + mem_offset;
    partition.type = partition_table->partition[i].type;
    context_.partition_datas_.push_back(partition);

    if (partition.size > model_data_size || mem_offset > model_data_size - partition.size) {
      GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID, "The partition size %zu is greater than the model data size %u.",
             partition.size + mem_offset, model_data_size);
      return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
    }
    mem_offset += partition.size;
    GELOGI("Partition, type:%d, size:%u", static_cast<int>(partition.type), partition.size);
  }
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY const std::vector<ModelPartition>
  &OmFileSaveHelper::GetModelPartitions() const {
  return context_.partition_datas_;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ModelPartitionTable *OmFileSaveHelper::GetPartitionTable() {
  auto partition_size = static_cast<uint32_t>(context_.partition_datas_.size());
  // Build ModelPartitionTable, flex array
  context_.partition_table_.clear();
  context_.partition_table_.resize(sizeof(ModelPartitionTable) + sizeof(ModelPartitionMemInfo) * partition_size, 0);

  auto partition_table = reinterpret_cast<ModelPartitionTable *>(context_.partition_table_.data());
  partition_table->num = partition_size;

  uint32_t mem_offset = 0;
  for (uint32_t i = 0; i < partition_size; i++) {
    ModelPartition partition = context_.partition_datas_[i];
    partition_table->partition[i] = {partition.type, mem_offset, partition.size};
    mem_offset += partition.size;
    GELOGI("Partition, type:%d, size:%u", static_cast<int>(partition.type), partition.size);
  }
  return partition_table;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status OmFileSaveHelper::AddPartition(ModelPartition &partition) {
  if (ge::CheckUint32AddOverflow(context_.model_data_len_, partition.size) != SUCCESS) {
    GELOGE(FAILED, "UINT32 %u and %u addition can result in overflow!", context_.model_data_len_, partition.size);
    return FAILED;
  }
  context_.partition_datas_.push_back(partition);
  context_.model_data_len_ += partition.size;
  return SUCCESS;
}

Status OmFileSaveHelper::SaveModel(const SaveParam &save_param, const char *output_file, ModelBufferData &model,
                                   bool is_offline) {
  (void)save_param.cert_file;
  (void)save_param.ek_file;
  (void)save_param.encode_mode;
  (void)save_param.hw_key_file;
  (void)save_param.pri_key_file;
  Status ret = SaveModelToFile(output_file, model, is_offline);
  if (ret == SUCCESS) {
    GELOGI("Generate model with encrypt.");
  }
  return ret;
}

Status OmFileSaveHelper::SaveModelToFile(const char *output_file, ModelBufferData &model, bool is_offline) {
#if !defined(NONSUPPORT_SAVE_TO_FILE)
  uint32_t model_data_len = context_.model_data_len_;
  if (model_data_len == 0) {
    GELOGE(domi::PARAM_INVALID, "Model data len error! should not be 0");
    return domi::PARAM_INVALID;
  }

  ModelPartitionTable *partition_table = GetPartitionTable();
  if (partition_table == nullptr) {
    GELOGE(ge::GE_GRAPH_SAVE_FAILED, "SaveModelToFile execute failed: partition_table is NULL.");
    return ge::GE_GRAPH_SAVE_FAILED;
  }
  uint32_t size_of_table = SIZE_OF_MODEL_PARTITION_TABLE(*partition_table);
  FMK_UINT32_ADDCHECK(size_of_table, model_data_len)
  model_header_.length = size_of_table + model_data_len;

  GELOGI("Sizeof(ModelFileHeader):%zu,sizeof(ModelPartitionTable):%u, model_data_len:%u, model_total_len:%zu",
         sizeof(ModelFileHeader), size_of_table, model_data_len, model_header_.length + sizeof(ModelFileHeader));

  std::vector<ModelPartition> partition_datas = context_.partition_datas_;
  Status ret;
  if (is_offline) {
    ret = FileSaver::SaveToFile(output_file, model_header_, *partition_table, partition_datas);
  } else {
    ret = FileSaver::SaveToBuffWithFileHeader(model_header_, *partition_table, partition_datas, model);
  }
  if (ret == SUCCESS) {
    GELOGI("Save model success without encrypt.");
  }
  return ret;
#else
  return SUCCESS;
#endif
}
}  // namespace ge
