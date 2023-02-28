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

#include "common/helper/model_parser_base.h"

#include <fstream>
#include <string>
#include <cstring>

#include "securec.h"
#include "framework/common/helper/model_helper.h"
#include "mmpa/mmpa_api.h"
#include "graph/def_types.h"

namespace {
const size_t kMaxErrorStringLen = 128U;
const uint32_t kStatiOmFileModelNum = 1U;
}  // namespace

namespace ge {
Status ModelParserBase::LoadFromFile(const char_t *const model_path, const int32_t priority, ModelData &model_data) {
  const std::string real_path = RealPath(model_path);
  if (real_path.empty()) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID, "[Check][Param]Model file path %s is invalid",
           model_path);
    REPORT_CALL_ERROR("E19999", "Model file path %s is invalid", model_path);
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  if (GetFileLength(model_path) == -1) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID, "[Check][Param]File size not valid, file %s",
           model_path);
    REPORT_INNER_ERROR("E19999", "File size not valid, file %s", model_path);
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  std::ifstream fs(real_path.c_str(), std::ifstream::binary);
  if (!fs.is_open()) {
    char_t err_buf[kMaxErrorStringLen + 1U]{};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStringLen);
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID, "[Open][File]Failed, file %s, error %s",
           model_path, err_msg);
    REPORT_CALL_ERROR("E19999", "Open file %s failed, error %s", model_path, err_msg);
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  // get length of file:
  (void)fs.seekg(0, std::ifstream::end);
  const uint64_t len = fs.tellg();

  GE_CHECK_GE(len, 1U);

  (void)fs.seekg(0, std::ifstream::beg);
  char_t *const data = new (std::nothrow) char_t[len];
  if (data == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Load][ModelFromFile]Failed, "
           "bad memory allocation occur(need %" PRIu64 "), file %s", len, model_path);
    REPORT_CALL_ERROR("E19999", "Load model from file %s failed, "
                      "bad memory allocation occur(need %" PRIu64 ")", model_path, len);
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  // read data as a block:
  (void)fs.read(data, static_cast<std::streamsize>(len));
  const ModelHelper model_helper;
  (void)model_helper.GetBaseNameFromFileName(model_path, model_data.om_name);
  // Set the model data parameter
  model_data.model_data = data;
  model_data.model_len = len;
  model_data.priority = priority;

  return SUCCESS;
}

Status ModelParserBase::ParseModelContent(const ModelData &model, uint8_t *&model_data, uint64_t &model_len) {
  // Parameter validity check
  GE_CHECK_NOTNULL(model.model_data);

  // Model length too small
  GE_CHK_BOOL_EXEC(model.model_len >= sizeof(ModelFileHeader),
                   REPORT_INPUT_ERROR("E10003", std::vector<std::string>({"parameter", "value", "reason"}),
                                      std::vector<std::string>({"om", model.om_name.c_str(), "invalid om file"}));
                   GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID,
                          "[Check][Param] Invalid model. Model data size %" PRIu64 " must be"
                          "greater than or equal to %zu.",
                          model.model_len, sizeof(ModelFileHeader));
                   return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
  // Get file header
  const auto file_header = static_cast<ModelFileHeader *>(model.model_data);
  // Determine whether the file length and magic number match
  model_len =
      (file_header->model_length == 0UL) ? static_cast<uint64_t>(file_header->length) : file_header->model_length;
  GE_CHK_BOOL_EXEC((model_len == (model.model_len - sizeof(ModelFileHeader))) &&
                   (file_header->magic == MODEL_FILE_MAGIC_NUM),
                   REPORT_INPUT_ERROR("E10003", std::vector<std::string>({"parameter", "value", "reason"}),
                                      std::vector<std::string>({"om", model.om_name.c_str(), "invalid om file"}));
                   GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID,
                          "[Check][Param] Invalid model, file_header->(model)length[%" PRIu64 "]"
                          " + sizeof(ModelFileHeader)[%zu] != model->model_len[%" PRIu64 "]"
                          "|| MODEL_FILE_MAGIC_NUM[%u] != file_header->magic[%u]",
                          model_len, sizeof(ModelFileHeader), model.model_len,
                          MODEL_FILE_MAGIC_NUM, file_header->magic);
                   return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);

  // Get data address
  model_data = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(model.model_data) + sizeof(ModelFileHeader)));
  GELOGD("Model_len is %" PRIu64 ", model_file_head_len is %zu.", model_len, sizeof(ModelFileHeader));

  return SUCCESS;
}

bool ModelParserBase::IsDynamicModel(const ModelFileHeader &file_header) {
  return (file_header.version >= ge::MODEL_VERSION) && (file_header.model_num > kStatiOmFileModelNum);
}
}  // namespace ge
