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

#include "common/debug/memory_dumper.h"

#include <string>

#include "framework/common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "framework/common/ge_inner_error_codes.h"
#include "external/graph/types.h"

namespace {
const int32_t kInvalidFd = (-1);
const size_t kMaxErrorStringLength = 128U;
}  // namespace

namespace ge {
MemoryDumper::MemoryDumper() : fd_(kInvalidFd) {}

MemoryDumper::~MemoryDumper() { Close(); }

// Dump the data to the file
Status MemoryDumper::DumpToFile(const char_t *const filename, void *const data, const int64_t len) {
#ifdef FMK_SUPPORT_DUMP
  GE_CHECK_NOTNULL(filename);
  GE_CHECK_NOTNULL(data);
  if (len == 0) {
    GELOGE(FAILED, "[Check][Param]Failed, data length is 0.");
    REPORT_INNER_ERROR("E19999", "Check param failed, data length is 0.");
    return PARAM_INVALID;
  }

  // Open the file
  const int32_t fd = OpenFile(filename);
  if (fd == kInvalidFd) {
    GELOGE(FAILED, "[Open][File]Failed, filename:%s.", filename);
    REPORT_INNER_ERROR("E19999", "Opne file failed, filename:%s.", filename);
    return FAILED;
  }

  // Write the data to the file
  Status ret = SUCCESS;
  const int32_t mmpa_ret = static_cast<int32_t>(mmWrite(fd, data, static_cast<uint32_t>(len)));
  // mmWrite return -1:Failed to write data to file；return -2:Invalid parameter
  if ((mmpa_ret == EN_ERROR) || (mmpa_ret == EN_INVALID_PARAM)) {
    char_t err_buf[kMaxErrorStringLength + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStringLength);
    GELOGE(FAILED, "[Write][Data]Failed, errno:%d, errmsg:%s", mmpa_ret, err_msg);
    REPORT_INNER_ERROR("E19999", "Write data failed, errno:%d, errmsg:%s.",
                       mmpa_ret, err_msg);
    ret = FAILED;
  }

  // Close the file
  if (mmClose(fd) != EN_OK) {  // mmClose return 0: success
    char_t err_buf[kMaxErrorStringLength + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStringLength);
    GELOGE(FAILED, "[Close][File]Failed, error_code:%u, filename:%s errmsg:%s.", ret, filename, err_msg);
    REPORT_INNER_ERROR("E19999", "Close file failed, error_code:%u, filename:%s errmsg:%s.",
                       ret, filename, err_msg);
    ret = FAILED;
  }

  return ret;
#else
  GELOGW("need to define FMK_SUPPORT_DUMP for dump op input and output.");
  return SUCCESS;
#endif
}

// Close file
void MemoryDumper::Close() noexcept {
  // Close file
  if ((fd_ != kInvalidFd) && (mmClose(fd_) != EN_OK)) {
    char_t err_buf[kMaxErrorStringLength + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStringLength);
    GELOGW("Close file failed, errmsg:%s.", err_msg);
  }
  fd_ = kInvalidFd;
}

// Open file
int32_t MemoryDumper::OpenFile(const std::string &filename) {
  // Find the last separator
  int32_t path_split_pos = static_cast<int32_t>(filename.size());
  path_split_pos--;
  for (; path_split_pos >= 0; path_split_pos--) {
    GE_IF_BOOL_EXEC((filename[static_cast<size_t>(path_split_pos)] == '\\') ||
                    (filename[static_cast<size_t>(path_split_pos)] == '/'), break;)
  }
  // Get the absolute path
  std::string real_path;
  char_t tmp_path[MMPA_MAX_PATH] = {};
  GE_IF_BOOL_EXEC(
    path_split_pos != -1, const std::string prefix_path = filename.substr(0U, static_cast<size_t>(path_split_pos));
    const std::string last_path = filename.substr(static_cast<size_t>(path_split_pos), filename.size() - 1U);
    if (prefix_path.length() >= static_cast<size_t>(MMPA_MAX_PATH)) {
      GELOGE(FAILED, "Prefix path is too long!");
      return kInvalidFd;
    }
    if (mmRealPath(prefix_path.c_str(), &tmp_path[0], MMPA_MAX_PATH) != EN_OK) {
      char_t err_buf[kMaxErrorStringLength + 1U] = {};
      const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStringLength);
      GELOGE(ge::FAILED, "Dir %s does not exit, errmsg:%s.", prefix_path.c_str(), err_msg);
      return kInvalidFd;
    }
    real_path = std::string(tmp_path) + last_path;)
  GE_IF_BOOL_EXEC(
    (path_split_pos == -1) || (path_split_pos == 0),
    if (filename.size() >= static_cast<size_t>(MMPA_MAX_PATH)) {
      GELOGE(FAILED, "Prefix path is too long!");
      return kInvalidFd;
    }

    GE_IF_BOOL_EXEC(mmRealPath(filename.c_str(), &tmp_path[0], MMPA_MAX_PATH) != EN_OK,
                    GELOGI("File %s does not exit, it will be created.", filename.c_str()));
    real_path = std::string(tmp_path));

  // Open file, only the current user can read and write, to avoid malicious application access
  // Using the O_EXCL, if the file already exists,return failed to avoid privilege escalation vulnerability.
  const mmMode_t mode = static_cast<uint32_t>(M_IRUSR) | static_cast<uint32_t>(M_IWUSR);
  const int32_t flag = static_cast<int32_t>(static_cast<uint32_t>(M_RDWR) |
                                            static_cast<uint32_t>(M_CREAT) |
                                            static_cast<uint32_t>(M_APPEND));

  const int32_t fd = mmOpen2(real_path.c_str(), flag, mode);
  if ((fd == EN_ERROR) || (fd == EN_INVALID_PARAM)) {
    char_t err_buf[kMaxErrorStringLength + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStringLength);
    GELOGE(static_cast<uint32_t>(kInvalidFd), "[Open][File]Failed. errno:%d, errmsg:%s, filename:%s.",
           fd, err_msg, filename.c_str());
    return kInvalidFd;
  }
  return fd;
}
}  // namespace ge
