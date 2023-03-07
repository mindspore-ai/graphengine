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

#include "framework/common/helper/om_file_helper.h"

#include "common/helper/model_parser_base.h"
#include "common/helper/file_saver.h"
#include "common/math/math_util.h"

namespace {
const uint32_t kOptionalNum = 2U;
}
namespace ge {
// For Load
Status OmFileLoadHelper::Init(const ModelData &model) {
  uint64_t model_len = 0UL;
  uint8_t *model_data = nullptr;
  GE_CHK_STATUS_RET_NOLOG(ModelParserBase::ParseModelContent(model, model_data, model_len));
  ModelFileHeader *file_header = PtrToPtr<void, ModelFileHeader>(model.model_data);
  return Init(model_data, model_len, file_header);
}

Status OmFileLoadHelper::Init(uint8_t *const model_data, const uint32_t model_data_size) {
  return Init(model_data, static_cast<uint64_t>(model_data_size), nullptr);
}

Status OmFileLoadHelper::Init(uint8_t *const model_data, const uint32_t model_data_size, const uint32_t model_num) {
  return Init(model_data, static_cast<uint64_t>(model_data_size), model_num, nullptr);
}

Status OmFileLoadHelper::Init(uint8_t *const model_data,
                              const uint64_t model_data_size,
                              const ModelFileHeader *file_header) {
  size_t mem_offset = 0U;
  const Status status = LoadModelPartitionTable(model_data, model_data_size, 0U, mem_offset, file_header);
  if (status != SUCCESS) {
    return status;
  }
  is_inited_ = true;
  return SUCCESS;
}

Status OmFileLoadHelper::Init(uint8_t *const model_data,
                              const uint64_t model_data_size,
                              const uint32_t model_num,
                              const ModelFileHeader *file_header) {
  const Status status = LoadModelPartitionTable(model_data, model_data_size, model_num, file_header);
  if (status != SUCCESS) {
    return status;
  }
  is_inited_ = true;
  return SUCCESS;
}

// Use both
Status OmFileLoadHelper::GetModelPartition(const ModelPartitionType type, ModelPartition &partition) {
  return GetModelPartition(type, partition, 0U);
}

Status OmFileLoadHelper::GetModelPartition(const ModelPartitionType type,
                                           ModelPartition &partition, const size_t model_index) const {
  if (!is_inited_) {
    GELOGE(PARAM_INVALID, "OmFileLoadHelper has not been initialized!");
    return PARAM_INVALID;
  }
  if (model_index >= model_contexts_.size()) {
    GELOGE(PARAM_INVALID, "cur index : %zu, model_contexts size:%zu", model_index, model_contexts_.size());
    return PARAM_INVALID;
  }
  const auto &cur_ctx = model_contexts_[model_index];
  for (const ModelPartition &part : cur_ctx.partition_datas_) {
    if (part.type == type) {
      partition = part;
      return SUCCESS;
    }
  }

  if ((type != ModelPartitionType::TBE_KERNELS) && (type != ModelPartitionType::WEIGHTS_DATA) &&
      (type != ModelPartitionType::CUST_AICPU_KERNELS) && (type != ModelPartitionType::SO_BINS)) {
    GELOGE(FAILED, "GetModelPartition:type:%d is not in partition_datas!", static_cast<int32_t>(type));
    return FAILED;
  }
  return SUCCESS;
}

const std::vector<ModelPartition> &OmFileLoadHelper::GetModelPartitions(const size_t model_index) const {
  if (model_index >= model_contexts_.size()) {
    GELOGE(PARAM_INVALID, "cur index : %zu, model_contexts size:%zu", model_index, model_contexts_.size());
    static const std::vector<ModelPartition> kEmptyVec;
    return kEmptyVec;
  }
  return model_contexts_[model_index].partition_datas_;
}

static Status ConvertToModelPartitionTable(const TinyModelPartitionTable *tiny_table,
                                           std::unique_ptr<uint8_t[]> &model_partition_table_holder) {
  size_t total_size = sizeof(ModelPartitionTable) + sizeof(ModelPartitionMemInfo) * tiny_table->num;
  model_partition_table_holder = std::unique_ptr<uint8_t[]>(new(std::nothrow) uint8_t[total_size]);
  if (model_partition_table_holder == nullptr) {
    GELOGE(FAILED, "malloc failed for size %zu", total_size);
    return FAILED;
  }
  auto table = reinterpret_cast<ModelPartitionTable *>(model_partition_table_holder.get());
  table->num = tiny_table->num;
  for (size_t i = 0U; i < table->num; ++i) {
    table->partition[i].type = tiny_table->partition[i].type;
    table->partition[i].mem_offset = static_cast<uint64_t>(tiny_table->partition[i].mem_offset);
    table->partition[i].mem_size = static_cast<uint64_t>(tiny_table->partition[i].mem_size);
  }
  return SUCCESS;
}

bool OmFileLoadHelper::IsPartitionTableNumValid(const uint32_t partition_num,
                                                const uint32_t increase_partition_num) const {
  if ((partition_num != (PARTITION_SIZE + increase_partition_num)) &&
      (partition_num != (PARTITION_SIZE - 1U + increase_partition_num)) &&
      (partition_num != (PARTITION_SIZE - kOptionalNum + increase_partition_num)) &&
      (partition_num != (1U + increase_partition_num))) {
      GELOGW("Invalid partition_table->num:%u", partition_num);
      return false;
  }
  return true;
}

Status OmFileLoadHelper::LoadModelPartitionTable(const uint8_t *const model_data,
                                                 const uint64_t model_data_size,
                                                 const size_t model_index,
                                                 size_t &mem_offset,
                                                 const ModelFileHeader *file_header) {
  if (model_data == nullptr) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_ADDR_INVALID, "Param model_data must not be null!");
    return ACL_ERROR_GE_EXEC_MODEL_ADDR_INVALID;
  }

  if ((model_data_size < mem_offset) || (model_data_size - mem_offset <= sizeof(ModelPartitionTable))) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID,
           "The partition table size %zu is greater than model data size %lu",
           mem_offset + sizeof(ModelPartitionTable), model_data_size);
    return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
  }

  const bool is_flow_model = (file_header != nullptr) && (file_header->modeltype == MODEL_TYPE_FLOW_MODEL);
  // Init partition table
  ModelPartitionTable *partition_table = nullptr;
  std::unique_ptr<uint8_t[]> model_partition_table_holder = nullptr;
  size_t partition_table_size = 0U;
  if (is_flow_model || ((file_header != nullptr) && (file_header->model_length != 0UL))) {
    partition_table = PtrToPtr<void, ModelPartitionTable>(ValueToPtr(PtrToValue(model_data) + mem_offset));
    partition_table_size = SizeOfModelPartitionTable(*partition_table);
  } else {
    TinyModelPartitionTable * const tiny_partition_table =
        PtrToPtr<void, TinyModelPartitionTable>(ValueToPtr(PtrToValue(model_data) + mem_offset));
    if (!(IsPartitionTableNumValid(tiny_partition_table->num, 0U) ||
        IsPartitionTableNumValid(tiny_partition_table->num, 1U))) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Invalid tiny_partition_table->num:%u", tiny_partition_table->num);
      return ACL_ERROR_GE_PARAM_INVALID;
    }
    partition_table_size = SizeOfTinyModelPartitionTable(*tiny_partition_table);
    GE_CHK_STATUS_RET_NOLOG(ConvertToModelPartitionTable(tiny_partition_table, model_partition_table_holder));
    partition_table = reinterpret_cast<ModelPartitionTable *>(model_partition_table_holder.get());
  }

  if (is_flow_model) {
    constexpr uint32_t kMaxFlowModelPartitionNum = 4096U;
    if (partition_table->num > kMaxFlowModelPartitionNum) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Invalid flow model partition_table->num:%u, range[0, %u]",
             partition_table->num, kMaxFlowModelPartitionNum);
      return ACL_ERROR_GE_PARAM_INVALID;
    }
  } else {
    // Davinici model partition include graph-info  weight-info  task-info  tbe-kernel :
    // Original model partition include graph-info
    if (!(IsPartitionTableNumValid(partition_table->num, 0U) || IsPartitionTableNumValid(partition_table->num, 1U))) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Invalid partition_table->num:%u", partition_table->num);
      return ACL_ERROR_GE_PARAM_INVALID;
    }
  }
  mem_offset += partition_table_size;
  GELOGD("Cur model index:%zu, ModelPartitionTable num:%u, ModelFileHeader size:%zu, ModelPartitionTable size:%zu",
         model_index, partition_table->num, sizeof(ModelFileHeader), partition_table_size);
  if (model_data_size <= mem_offset) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID, "invalid model data, partition_table->num:%u, data size %lu",
           partition_table->num, model_data_size);
    return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
  }

  if (model_index != model_contexts_.size()) {
    GELOGE(FAILED, "cur index is %zu make model_contexts_ overflow", model_index);
    return FAILED;
  }
  model_contexts_.push_back(OmFileContext{});
  for (uint32_t i = 0U; i < partition_table->num; i++) {
    ModelPartition partition;
    partition.size = partition_table->partition[i].mem_size;
    partition.data = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(model_data) + mem_offset));
    partition.type = partition_table->partition[i].type;
    model_contexts_[model_index].partition_datas_.push_back(partition);
    if ((partition.size > model_data_size) || (mem_offset > static_cast<size_t>(model_data_size - partition.size))) {
      GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID,
             "The partition size (%lu + %zu) is greater than the model data size %lu.",
             partition.size, mem_offset, model_data_size);
      return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
    }
    if (CheckUint64AddOverflow(mem_offset, partition.size) != SUCCESS) {
      GELOGE(FAILED, "UINT64 %zu and %lu addition can result in overflow!", mem_offset, partition.size);
      return FAILED;
    }
    mem_offset += partition.size;
    GELOGD("type:%d, size:%lu, index:%zu", static_cast<int32_t>(partition.type), partition.size, model_index);
  }
  return SUCCESS;
}

Status OmFileLoadHelper::LoadModelPartitionTable(const uint8_t *const model_data, const uint64_t model_data_size,
                                                 const uint32_t model_num, const ModelFileHeader *file_header) {
  if (model_data == nullptr) {
    GELOGE(PARAM_INVALID, "Param model_data must not be null!");
    return PARAM_INVALID;
  }

  size_t cur_offset = 0U;
  for (size_t index = 0U; index < static_cast<size_t>(model_num); ++index) {
    GE_CHK_STATUS_RET_NOLOG(LoadModelPartitionTable(model_data, model_data_size, index, cur_offset, file_header));
  }
  if (cur_offset != model_data_size) {
    GELOGE(FAILED, "do not get the complete model, read end offset:%zu, all size:%lu", cur_offset, model_data_size);
    return FAILED;
  }
  return SUCCESS;
}

uint64_t OmFileSaveHelper::GetModelDataSize() const {
  return model_contexts_.empty() ? 0U : model_contexts_[0U].model_data_len_;
}

ModelPartitionTable *OmFileSaveHelper::GetPartitionTable() {
  return GetPartitionTable(0U);
}

ModelPartitionTable *OmFileSaveHelper::GetPartitionTable(const size_t cur_ctx_index) {
  auto &cur_ctx = model_contexts_[cur_ctx_index];
  const auto partition_size = static_cast<uint64_t>(cur_ctx.partition_datas_.size());
  // Build ModelPartitionTable, flex array
  cur_ctx.partition_table_.resize(sizeof(ModelPartitionTable) + (sizeof(ModelPartitionMemInfo) * partition_size),
                                  static_cast<char_t>(0));

  auto const partition_table = PtrToPtr<char_t, ModelPartitionTable>(cur_ctx.partition_table_.data());
  partition_table->num = static_cast<uint32_t>(partition_size);

  uint64_t mem_offset = 0UL;
  for (size_t i = 0U; i < static_cast<size_t>(partition_size); i++) {
    const ModelPartition partition = cur_ctx.partition_datas_[i];
    partition_table->partition[i] = {partition.type, mem_offset, partition.size};
    if (CheckUint64AddOverflow(mem_offset, partition.size) != SUCCESS) {
      GELOGE(FAILED, "UINT64 %lu and %lu addition can result in overflow!", mem_offset, partition.size);
      return nullptr;
    }
    mem_offset += partition.size;
    GELOGD("Partition, type:%d, size:%lu", static_cast<int32_t>(partition.type), partition.size);
  }
  return partition_table;
}

Status OmFileSaveHelper::AddPartition(const ModelPartition &partition) {
  return AddPartition(partition, 0U);
}

Status OmFileSaveHelper::AddPartition(const ModelPartition &partition, const size_t cur_index) {
  if (cur_index >= model_contexts_.size()) {
    if (cur_index != model_contexts_.size()) {
      GELOGE(FAILED, "cur index is %zu make model_contexts_ overflow", cur_index);
      return FAILED;
    }
    OmFileContext tmp_ctx;
    tmp_ctx.model_data_len_ += partition.size;
    tmp_ctx.partition_datas_.push_back(partition);
    model_contexts_.push_back(tmp_ctx);
  } else {
    auto &cur_ctx = model_contexts_[cur_index];
    if (CheckUint64AddOverflow(cur_ctx.model_data_len_, partition.size) != SUCCESS) {
      GELOGE(FAILED, "UINT64 %lu and %lu addition can result in overflow!", cur_ctx.model_data_len_, partition.size);
      return FAILED;
    }
    cur_ctx.model_data_len_ += partition.size;
    cur_ctx.partition_datas_.push_back(partition);
  }
  return SUCCESS;
}

Status OmFileSaveHelper::SaveModel(const SaveParam &save_param, const char_t *const output_file, ModelBufferData &model,
                                   const bool is_offline) {
  (void)save_param.cert_file;
  (void)save_param.ek_file;
  (void)save_param.encode_mode;
  (void)save_param.hw_key_file;
  (void)save_param.pri_key_file;
  const Status ret = SaveModelToFile(output_file, model, is_offline);
  if (ret == SUCCESS) {
    GELOGD("Generate model with encrypt.");
  }
  return ret;
}

Status OmFileSaveHelper::SaveModelToFile(const char_t *const output_file, ModelBufferData &model,
                                         const bool is_offline) {
  if (model_contexts_.empty()) {
    GELOGE(FAILED, "mode contexts empty");
    return FAILED;
  }
  const auto &cur_ctx = model_contexts_[0U];
  if (cur_ctx.model_data_len_ == 0U) {
    GELOGE(PARAM_INVALID, "Model data len error! should not be 0");
    return PARAM_INVALID;
  }

  ModelPartitionTable *const partition_table = GetPartitionTable();
  GE_CHECK_NOTNULL(partition_table);
  const uint64_t size_of_table = SizeOfModelPartitionTable(*partition_table);
  FMK_UINT64_ADDCHECK(size_of_table, cur_ctx.model_data_len_)

  model_header_.model_length = size_of_table + cur_ctx.model_data_len_;

  GELOGD("Sizeof(ModelFileHeader):%zu,sizeof(ModelPartitionTable):%lu, model_data_len:%lu, model_total_len:%zu",
         sizeof(ModelFileHeader), size_of_table, cur_ctx.model_data_len_,
         static_cast<size_t>(model_header_.model_length + sizeof(ModelFileHeader)));

  Status ret;
  if (is_offline) {
    ret = FileSaver::SaveToFile(output_file, model_header_, *partition_table, cur_ctx.partition_datas_);
  } else {
    ret = FileSaver::SaveToBuffWithFileHeader(model_header_, *partition_table, cur_ctx.partition_datas_, model);
  }
  if (ret == SUCCESS) {
    GELOGD("Save model success without encrypt.");
  }
  return ret;
}

Status OmFileSaveHelper::SaveRootModel(const SaveParam &save_param, const char_t *const output_file,
                                       ModelBufferData &model, const bool is_offline) {
  (void)save_param.cert_file;
  (void)save_param.ek_file;
  (void)save_param.encode_mode;
  (void)save_param.hw_key_file;
  (void)save_param.pri_key_file;

  std::vector<ModelPartitionTable *> model_partition_tabels;
  std::vector<std::vector<ModelPartition>> all_model_partitions;
  for (size_t ctx_index = 0U; ctx_index < model_contexts_.size(); ++ctx_index) {
    auto &cur_ctx = model_contexts_[ctx_index];
    const uint64_t cur_model_data_len = cur_ctx.model_data_len_;
    if (cur_model_data_len == 0U) {
      GELOGE(PARAM_INVALID, "Model data len error! should not be 0");
      return PARAM_INVALID;
    }

    ModelPartitionTable *const tmp_table = GetPartitionTable(ctx_index);
    GE_CHECK_NOTNULL(tmp_table);
    const uint64_t size_of_table = (SizeOfModelPartitionTable(*tmp_table));
    FMK_UINT64_ADDCHECK(size_of_table, cur_model_data_len)
    FMK_UINT64_ADDCHECK(size_of_table + cur_model_data_len, model_header_.model_length)

    model_header_.model_length += size_of_table + cur_model_data_len;
    model_partition_tabels.push_back(tmp_table);
    all_model_partitions.push_back(cur_ctx.partition_datas_);
    GELOGD("sizeof(ModelPartitionTable):%lu, cur_model_data_len:%lu, cur_context_index:%zu", size_of_table,
           cur_model_data_len, ctx_index);
  }
  Status ret;
  if (is_offline) {
    ret = FileSaver::SaveToFile(output_file, model_header_, model_partition_tabels, all_model_partitions);
  } else {
    ret = FileSaver::SaveToBuffWithFileHeader(model_header_, model_partition_tabels, all_model_partitions, model);
  }
  if (ret == SUCCESS) {
    GELOGD("Save model success without encrypt.");
  }
  return ret;
}
}  // namespace ge
