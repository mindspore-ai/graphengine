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

#include "common/debug/memory_dumper.h"

#include <string>

#include "framework/common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "framework/common/ge_inner_error_codes.h"

using std::string;

namespace {
const int kInvalidFd = (-1);
}  // namespace

namespace ge {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY MemoryDumper::MemoryDumper() : fd_(kInvalidFd) {}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY MemoryDumper::~MemoryDumper() { Close(); }

// Dump the data to the file
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status MemoryDumper::DumpToFile(const char *filename, void *data,
                                                                                 int64_t len) {
#ifdef FMK_SUPPORT_DUMP
  GE_CHECK_NOTNULL(filename);
  GE_CHECK_NOTNULL(data);
  if (len == 0) {
    GELOGE(FAILED, "len is 0.");
    return PARAM_INVALID;
  }

  // Open the file
  int fd = OpenFile(filename);
  if (fd == kInvalidFd) {
    GELOGE(FAILED, "Open file failed.");
    return FAILED;
  }

  // Write the data to the file
  Status ret = SUCCESS;
  int32_t mmpa_ret = mmWrite(fd, data, len);
  // mmWrite return -1:Failed to write data to file；return -2:Invalid parameter
  if (mmpa_ret == EN_ERROR || mmpa_ret == EN_INVALID_PARAM) {
    GELOGE(FAILED, "Write to file failed. errno = %d, %s", mmpa_ret, strerror(errno));
    ret = FAILED;
  }

  // Close the file
  if (mmClose(fd) != EN_OK) {  // mmClose return 0: success
    GELOGE(FAILED, "Close file failed.");
    ret = FAILED;
  }

  return ret;
#else
  GELOGW("need to define FMK_SUPPORT_DUMP for dump op input and output.");
  return SUCCESS;
#endif
}

// Open file
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status MemoryDumper::Open(const char *filename) {
  GE_CHK_BOOL_RET_STATUS(filename != nullptr, FAILED, "Incorrect parameter. filename is nullptr");

  // Try to remove file first for reduce the close time by overwriting way
  // (The process of file closing will be about 100~200ms slower per file when written by overwriting way)
  // If remove file failed, then try to open it with overwriting way
  int ret = remove(filename);
  // If remove file failed, print the warning log
  if (ret != 0) {
    GELOGW("Remove file failed.");
  }

  fd_ = OpenFile(filename);
  if (fd_ == kInvalidFd) {
    GELOGE(FAILED, "Open %s failed.", filename);
    return FAILED;
  }

  return SUCCESS;
}

// Dump the data to file
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status MemoryDumper::Dump(void *data, uint32_t len) const {
  GE_CHK_BOOL_RET_STATUS(data != nullptr, FAILED, "Incorrect parameter. data is nullptr");

#ifdef FMK_SUPPORT_DUMP
  int32_t mmpa_ret = mmWrite(fd_, data, len);
  // mmWrite return -1:failed to write data to file；return -2:invalid parameter
  if (mmpa_ret == EN_ERROR || mmpa_ret == EN_INVALID_PARAM) {
    GELOGE(FAILED, "Write to file failed. errno = %d, %s", mmpa_ret, strerror(errno));
    return FAILED;
  }

  return SUCCESS;
#else
  GELOGW("need to define FMK_SUPPORT_DUMP for dump op input and output.");
  return SUCCESS;
#endif
}

// Close file
void MemoryDumper::Close() noexcept {
  // Close file
  if (fd_ != kInvalidFd && mmClose(fd_) != EN_OK) {
    GELOGW("Close file failed.");
  }
  fd_ = kInvalidFd;
}

// Open file
int MemoryDumper::OpenFile(const char *filename) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(filename == nullptr, return kInvalidFd, "Incorrect parameter. filename is nullptr");

  // Find the last separator
  int path_split_pos = static_cast<int>(strlen(filename) - 1);
  for (; path_split_pos >= 0; path_split_pos--) {
    GE_IF_BOOL_EXEC(filename[path_split_pos] == '\\' || filename[path_split_pos] == '/', break;)
  }
  // Get the absolute path
  string real_path;
  char tmp_path[MMPA_MAX_PATH] = {0};
  GE_IF_BOOL_EXEC(-1 != path_split_pos, string prefix_path = std::string(filename).substr(0, path_split_pos);
                  string last_path = std::string(filename).substr(path_split_pos, strlen(filename) - 1);
                  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(prefix_path.length() >= MMPA_MAX_PATH, return kInvalidFd,
                                                 "Prefix path is too long!");
                  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(mmRealPath(prefix_path.c_str(), tmp_path, MMPA_MAX_PATH) != EN_OK,
                                                 return kInvalidFd, "Dir %s does not exit.", prefix_path.c_str());
                  real_path = std::string(tmp_path) + last_path;)
  GE_IF_BOOL_EXEC(
    path_split_pos == -1 || path_split_pos == 0,
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(strlen(filename) >= MMPA_MAX_PATH, return kInvalidFd, "Prefix path is too long!");
    GE_IF_BOOL_EXEC(mmRealPath(filename, tmp_path, MMPA_MAX_PATH) != EN_OK,
                    GELOGI("File %s does not exit, it will be created.", filename));
    real_path = std::string(tmp_path);)

  // Open file, only the current user can read and write, to avoid malicious application access
  // Using the O_EXCL, if the file already exists,return failed to avoid privilege escalation vulnerability.
  mmMode_t mode = M_IRUSR | M_IWUSR;

  int32_t fd = mmOpen2(real_path.c_str(), M_RDWR | M_CREAT | O_TRUNC, mode);
  if (fd == EN_ERROR || fd == EN_INVALID_PARAM) {
    GELOGE(kInvalidFd, "open file failed. errno = %d, %s", fd, strerror(errno));
    return kInvalidFd;
  }
  return fd;
}
}  // namespace ge
