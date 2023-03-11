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

#include "common/helper/model_saver.h"

#include <securec.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cstring>

#include "framework/common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "common/util/error_manager/error_manager.h"

namespace {
const size_t kMaxErrStrLength = 128U;
}  //  namespace

namespace ge {
const int32_t kInteval = 2;

Status ModelSaver::SaveJsonToFile(const char_t *const file_path, const Json &model) {
  Status ret = SUCCESS;
  if ((file_path == nullptr) || (CheckPathValid(file_path) != SUCCESS)) {
    GELOGE(FAILED, "[Check][OutputFile]Failed, file %s", file_path);
    REPORT_CALL_ERROR("E19999", "Output file %s check invalid", file_path);
    return FAILED;
  }
  std::string model_str;
  try {
    model_str = model.dump(kInteval, ' ', false, Json::error_handler_t::ignore);
  } catch (std::exception &e) {
    REPORT_INNER_ERROR("E19999", "Failed to convert JSON to string, reason: %s, savefile:%s.", e.what(), file_path);
    GELOGE(FAILED, "[Convert][File]Failed to convert JSON to string, file %s, reason %s",
           file_path, e.what());
    return FAILED;
  } catch (...) {
    REPORT_INNER_ERROR("E19999", "Failed to convert JSON to string, savefile:%s.", file_path);
    GELOGE(FAILED, "[Convert][File]Failed to convert JSON to string, file %s", file_path);
    return FAILED;
  }

  char_t file_real_path[MMPA_MAX_PATH]{};
  GE_IF_BOOL_EXEC(mmRealPath(file_path, &file_real_path[0], MMPA_MAX_PATH) != EN_OK,
                  GELOGI("File %s does not exit, it will be created.", file_path));

  // Open file
  const mmMode_t open_mode = static_cast<mmMode_t>(static_cast<uint32_t>(M_IRUSR) | static_cast<uint32_t>(M_IWUSR));
  char_t err_buf[kMaxErrStrLength + 1UL] = {};
  const int32_t fd = mmOpen2(&file_real_path[0],
                             static_cast<int32_t>(static_cast<uint32_t>(M_RDWR) |
                                                  static_cast<uint32_t>(M_CREAT) |
                                                  static_cast<uint32_t>(O_TRUNC)), open_mode);
  if ((fd == EN_ERROR) || (fd == EN_INVALID_PARAM)) {
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrStrLength);
    ErrorManager::GetInstance().ATCReportErrMessage("E19001", {"file", "errmsg"}, {file_path, err_msg});
    GELOGE(FAILED, "[Open][File]Failed, file %s, errmsg %s", file_path, err_msg);
    return FAILED;
  }
  const char_t *const model_char = model_str.c_str();
  const uint32_t len = static_cast<uint32_t>(model_str.length());
  // Write data to file
  const mmSsize_t mmpa_ret = mmWrite(fd, const_cast<char_t *>(model_char), len);
  if ((mmpa_ret == EN_ERROR) || (mmpa_ret == EN_INVALID_PARAM)) {
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrStrLength);
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E19004", {"file", "errmsg"}, {file_path, err_msg});
    // Need to both print the error info of mmWrite and mmClose, so return ret after mmClose
    GELOGE(FAILED, "[Write][Data]To file %s failed. errno %ld, errmsg %s",
           file_path, mmpa_ret, err_msg);
    ret = FAILED;
  }
  // Close file
  if (mmClose(fd) != EN_OK) {
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrStrLength);
    GELOGE(FAILED, "[Close][File]Failed, file %s, errmsg %s", file_path, err_msg);
    REPORT_CALL_ERROR("E19999", "Close file %s failed, errmsg %s", file_path, err_msg);
    ret = FAILED;
  }
  return ret;
}
}  // namespace ge
