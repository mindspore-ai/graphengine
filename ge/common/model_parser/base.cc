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

#include "common/model_parser/base.h"
#include "common/helper/model_helper.h"
#include <securec.h>
#include <fstream>
#include <memory>
#include <string>

#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/util.h"

namespace ge {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ModelParserBase::ModelParserBase() {}
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ModelParserBase::~ModelParserBase() {}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status ModelParserBase::LoadFromFile(const char *model_path,
                                                                                      const char *key, int32_t priority,
                                                                                      ge::ModelData &model_data) {
  std::string real_path = RealPath(model_path);
  if (real_path.empty()) {
    GELOGE(GE_EXEC_MODEL_PATH_INVALID, "Model file path '%s' is invalid", model_path);
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  if (GetFileLength(model_path) == -1) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID, "File size not valid, file: %s.", model_path);
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  std::ifstream fs(real_path.c_str(), std::ifstream::binary);
  if (!fs.is_open()) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID, "Open file: %s failed, error: %s", model_path, strerror(errno));
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  // get length of file:
  (void)fs.seekg(0, std::ifstream::end);
  uint32_t len = static_cast<uint32_t>(fs.tellg());

  GE_CHECK_GE(len, 1);

  (void)fs.seekg(0, std::ifstream::beg);

  char *data = new (std::nothrow) char[len];
  if (data == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "Load model From file failed, bad memory allocation occur. (need:%u)", len);
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  // read data as a block:
  (void)fs.read(data, len);
  ModelHelper model_helper;
  model_helper.GetBaseNameFromFileName(model_path, model_data.om_name);
  // Set the model data parameter
  model_data.model_data = data;
  model_data.model_len = len;
  model_data.priority = priority;
  model_data.key = (key == nullptr) ? "" : key;

  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status ModelParserBase::ParseModelContent(const ge::ModelData &model,
                                                                                           uint8_t *&model_data,
                                                                                           uint32_t &model_len) {
  // Parameter validity check
  GE_CHECK_NOTNULL(model.model_data);

  // Model length too small
  GE_CHK_BOOL_RET_STATUS(model.model_len >= sizeof(ModelFileHeader), ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID,
                         "Invalid model. Model data size %u must be greater than or equal to %zu.", model.model_len,
                         sizeof(ModelFileHeader));
  // Get file header
  auto file_header = reinterpret_cast<ModelFileHeader *>(model.model_data);
  // Determine whether the file length and magic number match
  GE_CHK_BOOL_RET_STATUS(
    file_header->length == model.model_len - sizeof(ModelFileHeader) && file_header->magic == MODEL_FILE_MAGIC_NUM,
    ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID,
    "Invalid model. file_header->length[%u] + sizeof(ModelFileHeader)[%zu] != model->model_len[%u] || "
    "MODEL_FILE_MAGIC_NUM[%u] != file_header->magic[%u]",
    file_header->length, sizeof(ModelFileHeader), model.model_len, MODEL_FILE_MAGIC_NUM, file_header->magic);

  Status res = SUCCESS;

  // Get data address
  uint8_t *data = reinterpret_cast<uint8_t *>(model.model_data) + sizeof(ModelFileHeader);
  if (file_header->is_encrypt == ModelEncryptType::UNENCRYPTED) {  // Unencrypted model
    GE_CHK_BOOL_RET_STATUS(model.key.empty(), ACL_ERROR_GE_EXEC_MODEL_NOT_SUPPORT_ENCRYPTION,
                           "Invalid param. model is unencrypted, but key is not empty.");

    model_data = data;
    model_len = file_header->length;
    GELOGD("Model_len is %u, model_file_head_len is %zu.", model_len, sizeof(ModelFileHeader));
  } else {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_NOT_SUPPORT_ENCRYPTION, "Invalid model. ModelEncryptType not supported.");
    res = ACL_ERROR_GE_EXEC_MODEL_NOT_SUPPORT_ENCRYPTION;
  }

  return res;
}
}  // namespace ge
