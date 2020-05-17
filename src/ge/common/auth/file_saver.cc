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

#include "common/auth/file_saver.h"

#include <fcntl.h>

#include <securec.h>
#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "common/math/math_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/util.h"

using domi::CreateDirectory;
using domi::ModelEncryptType;
using ge::ModelBufferData;

namespace {
const int kFileOpSuccess = 0;
}  //  namespace

namespace ge {
Status FileSaver::OpenFile(int32_t &fd, const std::string &file_path) {
  if (CheckPath(file_path) != SUCCESS) {
    GELOGE(FAILED, "Check output file failed.");
    return FAILED;
  }

  char real_path[PATH_MAX] = {0};
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(file_path.length() >= PATH_MAX, return FAILED, "File path is longer than PATH_MAX!");
  GE_IF_BOOL_EXEC(realpath(file_path.c_str(), real_path) == nullptr,
                  GELOGI("File %s is not exit, it will be created.", file_path.c_str()));
  // Open file
  mode_t mode = S_IRUSR | S_IWUSR;
  fd = mmOpen2(real_path, O_RDWR | O_CREAT | O_TRUNC, mode);
  if (fd == EN_INVALID_PARAM || fd == EN_ERROR) {
    // -1: Failed to open file; - 2: Illegal parameter
    GELOGE(FAILED, "Open file failed. mmpa_errno = %d, %s", fd, strerror(errno));
    return FAILED;
  }
  return SUCCESS;
}

Status FileSaver::WriteData(const void *data, uint32_t size, int32_t fd) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(size == 0 || data == nullptr, return PARAM_INVALID);

  // Write data
  int32_t write_count = mmWrite(fd, const_cast<void *>(data), size);
  // -1: Failed to write to file; - 2: Illegal parameter
  if (write_count == EN_INVALID_PARAM || write_count == EN_ERROR) {
    GELOGE(FAILED, "Write data failed. mmpa_errorno = %d, %s", write_count, strerror(errno));
    return FAILED;
  }

  return SUCCESS;
}

Status FileSaver::SaveWithFileHeader(const std::string &file_path, const ModelFileHeader &file_header, const void *data,
                                     int len) {
  if (data == nullptr || len <= 0) {
    GELOGE(FAILED, "Model_data is null or the length[%d] less than 1.", len);
    return FAILED;
  }

  // Open file
  int32_t fd = 0;
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(OpenFile(fd, file_path) != SUCCESS, return FAILED, "OpenFile FAILED");

  Status ret = SUCCESS;
  do {
    // Write file header
    GE_CHK_BOOL_EXEC(WriteData(static_cast<const void *>(&file_header), sizeof(ModelFileHeader), fd) == SUCCESS,
                     ret = FAILED;
                     break, "WriteData FAILED");
    // write data
    GE_CHK_BOOL_EXEC(WriteData(data, static_cast<uint32_t>(len), fd) == SUCCESS, ret = FAILED, "WriteData FAILED");
  } while (0);
  // Close file
  if (mmClose(fd) != 0) {  // mmClose 0: success
    GELOGE(FAILED, "Close file failed.");
    ret = FAILED;
  }
  return ret;
}

Status FileSaver::SaveWithFileHeader(const std::string &file_path, const ModelFileHeader &file_header,
                                     ModelPartitionTable &model_partition_table,

                                     const std::vector<ModelPartition> &partition_datas) {
  GE_CHK_BOOL_RET_STATUS(
    !partition_datas.empty() && model_partition_table.num != 0 && model_partition_table.num == partition_datas.size(),
    FAILED, "Invalid param:partition data size is (%u), model_partition_table.num is (%zu).", model_partition_table.num,
    partition_datas.size());
  // Open file
  int32_t fd = 0;
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(OpenFile(fd, file_path) != SUCCESS, return FAILED);
  Status ret = SUCCESS;
  do {
    // Write file header
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      WriteData(static_cast<const void *>(&file_header), sizeof(ModelFileHeader), fd) != SUCCESS, ret = FAILED; break);
    // Write model partition table
    uint32_t table_size = static_cast<uint32_t>(SIZE_OF_MODEL_PARTITION_TABLE(model_partition_table));
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      WriteData(static_cast<const void *>(&model_partition_table), table_size, fd) != SUCCESS, ret = FAILED; break);
    // Write partition data
    for (const auto &partitionData : partition_datas) {
      GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        WriteData(static_cast<const void *>(partitionData.data), partitionData.size, fd) != SUCCESS, ret = FAILED;
        break);
    }
  } while (0);
  // Close file
  GE_CHK_BOOL_RET_STATUS(mmClose(fd) == EN_OK, FAILED, "Close file failed.");
  return ret;
}

Status FileSaver::SaveToBuffWithFileHeader(const ModelFileHeader &file_header,
                                           ModelPartitionTable &model_partition_table,
                                           const std::vector<ModelPartition> &partitionDatas,
                                           ge::ModelBufferData &model) {
  GE_CHK_BOOL_RET_STATUS(
    !partitionDatas.empty() && model_partition_table.num != 0 && model_partition_table.num == partitionDatas.size(),
    FAILED, "Invalid param:partition data size is (%u), model_partition_table.num is (%zu).", model_partition_table.num,
    partitionDatas.size());
  uint32_t model_header_size = sizeof(ModelFileHeader);
  uint32_t table_size = static_cast<uint32_t>(SIZE_OF_MODEL_PARTITION_TABLE(model_partition_table));
  uint32_t total_size = model_header_size + table_size;

  for (const auto &partitionData : partitionDatas) {
    auto ret = ge::CheckUint32AddOverflow(total_size, partitionData.size);
    GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED, "add uint32 overflow!");
    total_size = total_size + partitionData.size;
  }
  auto buff = reinterpret_cast<uint8_t *>(malloc(total_size));
  GE_CHK_BOOL_RET_STATUS(buff != nullptr, FAILED, "malloc failed!");
  GE_PRINT_DYNAMIC_MEMORY(malloc, "file buffer.", total_size)
  model.data.reset(buff, [](uint8_t *buff) {
    GELOGD("Free online model memory.");
    free(buff);
    buff = nullptr;
  });
  model.length = total_size;
  uint32_t left_space = total_size;
  auto ret_mem1 = memcpy_s(buff, left_space, reinterpret_cast<void *>(const_cast<ModelFileHeader *>(&file_header)),
                           model_header_size);
  GE_CHK_BOOL_RET_STATUS(ret_mem1 == 0, FAILED, "memcpy_s failed!");
  buff += model_header_size;
  left_space -= model_header_size;
  auto ret_mem2 = memcpy_s(buff, left_space, reinterpret_cast<void *>(&model_partition_table), table_size);
  GE_CHK_BOOL_RET_STATUS(ret_mem2 == 0, FAILED, "memcpy_s failed!");
  buff += table_size;
  left_space -= table_size;
  for (const auto &partitionData : partitionDatas) {
    auto ret_mem3 = memcpy_s(buff, left_space, reinterpret_cast<void *>(const_cast<uint8_t *>(partitionData.data)),
                             partitionData.size);
    GE_CHK_BOOL_RET_STATUS(ret_mem3 == 0, FAILED, "memcpy failed!");
    buff += partitionData.size;
    left_space -= partitionData.size;
  }
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status FileSaver::CheckPath(const std::string &file_path) {
  // Determine file path length
  if (file_path.size() >= PATH_MAX) {
    GELOGE(FAILED, "Path is too long:%zu", file_path.size());
    return FAILED;
  }

  // Find the last separator
  int path_split_pos = static_cast<int>(file_path.size() - 1);
  for (; path_split_pos >= 0; path_split_pos--) {
    if (file_path[path_split_pos] == '\\' || file_path[path_split_pos] == '/') {
      break;
    }
  }

  if (path_split_pos == 0) {
    return SUCCESS;
  }

  // If there is a path before the file name, create the path
  if (path_split_pos != -1) {
    if (CreateDirectory(std::string(file_path).substr(0, static_cast<size_t>(path_split_pos))) != kFileOpSuccess) {
      GELOGE(FAILED, "CreateDirectory failed, file path:%s.", file_path.c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status
FileSaver::SaveToFile(const string &file_path, const ge::ModelData &model, const ModelFileHeader *model_file_header) {
  if (file_path.empty() || model.model_data == nullptr || model.model_len == 0) {
    GELOGE(FAILED, "Incorrected input param. file_path.empty() || model.model_data == nullptr || model.model_len == 0");
    return FAILED;
  }

  ModelFileHeader file_header;

  int32_t copy_header_ret = 0;
  GE_IF_BOOL_EXEC(model_file_header != nullptr, copy_header_ret = memcpy_s(&file_header, sizeof(ModelFileHeader),
                                                                           model_file_header, sizeof(ModelFileHeader)));
  GE_CHK_BOOL_RET_STATUS(copy_header_ret == 0, FAILED, "Copy ModelFileHeader failed, memcpy_s return: %d",
                         copy_header_ret);

  file_header.length = model.model_len;
  file_header.is_encrypt = ModelEncryptType::UNENCRYPTED;

  const Status ret = SaveWithFileHeader(file_path, file_header, model.model_data, file_header.length);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Save file failed, file_path:%s, file header len:%u.", file_path.c_str(), file_header.length);
    return FAILED;
  }

  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status
FileSaver::SaveToFile(const string &file_path, ModelFileHeader &file_header, ModelPartitionTable &model_partition_table,
                      const std::vector<ModelPartition> &partition_datas) {
  file_header.is_encrypt = ModelEncryptType::UNENCRYPTED;

  const Status ret = SaveWithFileHeader(file_path, file_header, model_partition_table, partition_datas);
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED, "save file failed, file_path:%s, file header len:%u.",
                         file_path.c_str(), file_header.length);
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status FileSaver::SaveToFile(const string &file_path, const void *data,
                                                                              int len) {
  if (data == nullptr || len <= 0) {
    GELOGE(FAILED, "Model_data is null or the length[%d] less than 1.", len);
    return FAILED;
  }

  // Open file
  int32_t fd = 0;
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(OpenFile(fd, file_path) != SUCCESS, return FAILED, "OpenFile FAILED");

  Status ret = SUCCESS;

  // write data
  GE_CHK_BOOL_EXEC(SUCCESS == WriteData(data, (uint32_t)len, fd), ret = FAILED, "WriteData FAILED");

  // Close file
  if (mmClose(fd) != 0) {  // mmClose 0: success
    GELOGE(FAILED, "Close file failed.");
    ret = FAILED;
  }
  return ret;
}
}  // namespace ge
