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

#include "common/model_saver.h"

#include <fcntl.h>
#include <securec.h>
#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include <string>

#include "framework/common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "common/util/error_manager/error_manager.h"

namespace ge {
const uint32_t kInteval = 2;

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status ModelSaver::SaveJsonToFile(const char *file_path,
                                                                                   const Json &model) {
  Status ret = SUCCESS;
  if (file_path == nullptr || SUCCESS != CheckPath(file_path)) {
    GELOGE(FAILED, "Check output file failed.");
    return FAILED;
  }
  std::string model_str;
  try {
    model_str = model.dump(kInteval, ' ', false, Json::error_handler_t::ignore);
  } catch (std::exception &e) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19007", {"exception"}, {e.what()});
    GELOGE(FAILED, "Failed to convert JSON to string, reason: %s.", e.what());
    return FAILED;
  } catch (...) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19008");
    GELOGE(FAILED, "Failed to convert JSON to string.");
    return FAILED;
  }

  char real_path[PATH_MAX] = {0};
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(strlen(file_path) >= PATH_MAX, return FAILED, "file path is too long!");
  GE_IF_BOOL_EXEC(realpath(file_path, real_path) == nullptr,
                  GELOGI("File %s does not exit, it will be created.", file_path));

  // Open file
  mode_t mode = S_IRUSR | S_IWUSR;
  int32_t fd = mmOpen2(real_path, O_RDWR | O_CREAT | O_TRUNC, mode);
  if (fd == EN_ERROR || fd == EN_INVALID_PARAM) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19001", {"file", "errmsg"}, {file_path, strerror(errno)});
    GELOGE(FAILED, "Open file[%s] failed. %s", file_path, strerror(errno));
    return FAILED;
  }
  const char *model_char = model_str.c_str();
  uint32_t len = static_cast<uint32_t>(model_str.length());
  // Write data to file
  mmSsize_t mmpa_ret = mmWrite(fd, const_cast<void *>((const void *)model_char), len);
  if (mmpa_ret == EN_ERROR || mmpa_ret == EN_INVALID_PARAM) {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E19004", {"file", "errmsg"}, {file_path, strerror(errno)});
    // Need to both print the error info of mmWrite and mmClose, so return ret after mmClose
    GELOGE(FAILED, "Write to file failed. errno = %d, %s", mmpa_ret, strerror(errno));
    ret = FAILED;
  }
  // Close file
  if (mmClose(fd) != EN_OK) {
    GELOGE(FAILED, "Close file failed.");
    ret = FAILED;
  }
  return ret;
}
}  // namespace ge
