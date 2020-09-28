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

#include "framework/common/util.h"

#include <fcntl.h>
#include <sys/stat.h>

#include <regex.h>
#include <unistd.h>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include "common/util/error_manager/error_manager.h"
#include "external/ge/ge_api_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/fmk_types.h"
#include "framework/common/ge_inner_error_codes.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "mmpa/mmpa_api.h"

using google::protobuf::io::CodedInputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::ZeroCopyInputStream;

namespace {
/*
 * kProtoReadBytesLimit and kWarningThreshold are real arguments of CodedInputStream::SetTotalBytesLimit.
 * In order to prevent integer overflow and excessive memory allocation during protobuf processing,
 * it is necessary to limit the length of proto message (call SetTotalBytesLimit function).
 * In theory, the minimum message length that causes an integer overflow is 512MB, and the default is 64MB.
 * If the limit of warning_threshold is exceeded, the exception information will be printed in stderr.
 * If such an exception is encountered during operation,
 * the proto file can be divided into several small files or the limit value can be increased.
 */
const int kProtoReadBytesLimit = INT_MAX;     // Max size of 2 GB minus 1 byte.
const int kWarningThreshold = 536870912 * 2;  // 536870912 represent 512M

/// The maximum length of the file.
/// Based on the security coding specification and the current actual (protobuf) model size, it is determined as 2G-1
const int kMaxFileSizeLimit = INT_MAX;
const int kMaxBuffSize = 256;
const char *const kPathValidReason = "The path can only contain 'a-z' 'A-Z' '0-9' '-' '.' '_' and chinese character";
constexpr uint32_t kMaxConfigFileByte = 10 * 1024 * 1024;
}  // namespace

namespace ge {
static bool ReadProtoFromCodedInputStream(CodedInputStream &coded_stream, Message *proto) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(proto == nullptr, return false, "incorrect parameter. nullptr == proto");

  coded_stream.SetTotalBytesLimit(kProtoReadBytesLimit, kWarningThreshold);
  return proto->ParseFromCodedStream(&coded_stream);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool ReadProtoFromBinaryFile(const char *file, Message *proto) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((file == nullptr || proto == nullptr), return false,
                                 "Input parameter file or proto is nullptr!");

  std::string real_path = RealPath(file);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(real_path.empty(), return false, "pb file path '%s' not valid", file);

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(GetFileLength(real_path) == -1, return false, "file size not valid.");

  std::ifstream fs(real_path, std::ifstream::in | std::ifstream::binary);
  if (!fs.is_open()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19001", {"file", "errmsg"}, {file, "ifstream is_open failed"});
    GELOGE(ge::FAILED, "Open real path[%s] failed.", file);
    return false;
  }

  google::protobuf::io::IstreamInputStream istream(&fs);
  google::protobuf::io::CodedInputStream coded_stream(&istream);

  bool ret = ReadProtoFromCodedInputStream(coded_stream, proto);

  fs.close();

  if (!ret) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19005", {"file"}, {file});
    GELOGE(ge::FAILED, "Parse file[%s] failed.", file);
    return ret;
  }

  return ret;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool ReadProtoFromArray(const void *data, int size, Message *proto) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((proto == nullptr || data == nullptr || size == 0), return false,
                                 "incorrect parameter. proto is nullptr || data is nullptr || size is 0");

  google::protobuf::io::CodedInputStream coded_stream(reinterpret_cast<uint8_t *>(const_cast<void *>(data)), size);
  return ReadProtoFromCodedInputStream(coded_stream, proto);
}

// Get file length
long GetFileLength(const std::string &input_file) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(input_file.empty(), return -1, "input_file path is null.");

  std::string real_path = RealPath(input_file.c_str());

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(real_path.empty(), return -1, "input_file path '%s' not valid", input_file.c_str());
  unsigned long long file_length = 0;
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
    mmGetFileSize(input_file.c_str(), &file_length) != EN_OK,
    ErrorManager::GetInstance().ATCReportErrMessage("E19001", {"file", "errmsg"}, {input_file, strerror(errno)});
    return -1, "Open file[%s] failed. %s", input_file.c_str(), strerror(errno));

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((file_length == 0),
                                 ErrorManager::GetInstance().ATCReportErrMessage("E19015", {"filepath"}, {input_file});
                                 return -1, "File[%s] size is 0, not valid.", input_file.c_str());

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
    file_length > kMaxFileSizeLimit, ErrorManager::GetInstance().ATCReportErrMessage(
                                       "E19016", {"filepath", "filesize", "maxlen"},
                                       {input_file, std::to_string(file_length), std::to_string(kMaxFileSizeLimit)});
    return -1, "File[%s] size %lld is out of limit: %d.", input_file.c_str(), file_length, kMaxFileSizeLimit);
  return static_cast<long>(file_length);
}

/** @ingroup domi_common
 *  @brief Read all data from binary file
 *  @param [in] file_name  File path
 *  @param [out] buffer  The address of the output memory, which needs to be released by the caller
 *  @param [out] length  Output memory size
 *  @return false fail
 *  @return true success
 */
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool ReadBytesFromBinaryFile(const char *file_name, char **buffer,
                                                                              int &length) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((file_name == nullptr), return false, "incorrect parameter. file is nullptr");
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((buffer == nullptr), return false, "incorrect parameter. buffer is nullptr");

  std::string real_path = RealPath(file_name);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(real_path.empty(), return false, "file path '%s' not valid", file_name);

  std::ifstream file(real_path.c_str(), std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    GELOGE(ge::FAILED, "Read file %s failed.", file_name);
    return false;
  }

  length = static_cast<int>(file.tellg());

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((length <= 0), file.close(); return false, "file length <= 0");

  file.seekg(0, std::ios::beg);

  *buffer = new (std::nothrow) char[length]();
  GE_CHK_BOOL_TRUE_EXEC_RET_STATUS(*buffer == nullptr, false, file.close(), "new an object failed.");

  file.read(*buffer, length);
  file.close();
  return true;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool ReadBytesFromBinaryFile(const char *file_name,
                                                                              std::vector<char> &buffer) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((file_name == nullptr), return false, "incorrect parameter. file path is null");

  std::string real_path = RealPath(file_name);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(real_path.empty(), return false, "file path '%s' not valid", file_name);

  std::ifstream file(real_path.c_str(), std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    GELOGE(ge::FAILED, "Read file %s failed.", file_name);
    return false;
  }

  std::streamsize size = file.tellg();

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((size <= 0), file.close(); return false, "file length <= 0, not valid.");
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(size > kMaxFileSizeLimit, file.close();
                                 return false, "file size %ld is out of limit: %d.", size, kMaxFileSizeLimit);

  file.seekg(0, std::ios::beg);  // [no need to check value]

  buffer.resize(static_cast<uint64_t>(size));  // [no need to check value]
  file.read(&buffer[0], size);                 // [no need to check value]
  file.close();
  GELOGI("Read size:%ld", size);
  return true;
}

/**
 *  @ingroup domi_common
 *  @brief Create directory, support to create multi-level directory
 *  @param [in] directory_path  Path, can be multi-level directory
 *  @return -1 fail
 *  @return 0 success
 */
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY int CreateDirectory(const std::string &directory_path) {
  GE_CHK_BOOL_EXEC(!directory_path.empty(), return -1, "directory path is empty.");
  auto dir_path_len = directory_path.length();
  if (dir_path_len >= PATH_MAX) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19002", {"filepath", "size"},
                                                    {directory_path, std::to_string(PATH_MAX)});
    GELOGW("Path[%s] len is too long, it must be less than %d", directory_path.c_str(), PATH_MAX);
    return -1;
  }
  char tmp_dir_path[PATH_MAX] = {0};
  for (size_t i = 0; i < dir_path_len; i++) {
    tmp_dir_path[i] = directory_path[i];
    if ((tmp_dir_path[i] == '\\') || (tmp_dir_path[i] == '/')) {
      if (access(tmp_dir_path, F_OK) != 0) {
        int32_t ret = mmMkdir(tmp_dir_path, S_IRUSR | S_IWUSR | S_IXUSR);  // 700
        if (ret != 0) {
          if (errno != EEXIST) {
            ErrorManager::GetInstance().ATCReportErrMessage("E19006", {"path"}, {directory_path});
            GELOGW("Can not create directory %s. Make sure the directory exists and writable.", directory_path.c_str());
            return ret;
          }
        }
      }
    }
  }
  int32_t ret = mmMkdir(const_cast<char *>(directory_path.c_str()), S_IRUSR | S_IWUSR | S_IXUSR);  // 700
  if (ret != 0) {
    if (errno != EEXIST) {
      ErrorManager::GetInstance().ATCReportErrMessage("E19006", {"path"}, {directory_path});
      GELOGW("Can not create directory %s. Make sure the directory exists and writable.", directory_path.c_str());
      return ret;
    }
  }
  return 0;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY std::string CurrentTimeInStr() {
  std::time_t now = std::time(nullptr);
  std::tm *ptm = std::localtime(&now);
  if (ptm == nullptr) {
    GELOGE(ge::FAILED, "Localtime failed.");
    return "";
  }

  const int kTimeBufferLen = 32;
  char buffer[kTimeBufferLen + 1] = {0};
  // format: 20171122042550
  std::strftime(buffer, kTimeBufferLen, "%Y%m%d%H%M%S", ptm);
  return std::string(buffer);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool ReadProtoFromText(const char *file,
                                                                        google::protobuf::Message *message) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((file == nullptr || message == nullptr), return false,
                                 "incorrect parameter. nullptr == file || nullptr == message");

  std::string real_path = RealPath(file);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(real_path.empty(), ErrorManager::GetInstance().ATCReportErrMessage(
                                                      "E19000", {"path", "errmsg"}, {file, strerror(errno)});
                                 return false, "Path[%s]'s realpath is empty, errmsg[%s]", file, strerror(errno));

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(GetFileLength(real_path) == -1, return false, "file size not valid.");

  std::ifstream fs(real_path.c_str(), std::ifstream::in);

  if (!fs.is_open()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19017", {"realpth", "protofile"}, {real_path, file});
    GELOGE(ge::FAILED, "Fail to open proto file real path is '%s' when orginal file path is '%s'.", real_path.c_str(),
           file);
    return false;
  }

  google::protobuf::io::IstreamInputStream input(&fs);
  bool ret = google::protobuf::TextFormat::Parse(&input, message);
  GE_IF_BOOL_EXEC(!ret, ErrorManager::GetInstance().ATCReportErrMessage("E19018", {"protofile"}, {file});
                  GELOGE(ret,
                         "Parse file[%s] through [google::protobuf::TextFormat::Parse] failed, "
                         "please check whether the file is a valid protobuf format file.",
                         file));
  fs.close();

  return ret;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool ReadProtoFromMem(const char *data, int size,
                                                                       google::protobuf::Message *message) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((data == nullptr || message == nullptr), return false,
                                 "incorrect parameter. data is nullptr || message is nullptr");
  std::string str(data, static_cast<size_t>(size));
  std::istringstream fs(str);

  google::protobuf::io::IstreamInputStream input(&fs);
  bool ret = google::protobuf::TextFormat::Parse(&input, message);
  GE_IF_BOOL_EXEC(
    !ret, GELOGE(ret, "Call [google::protobuf::TextFormat::Parse] func ret fail, please check your text file."));

  return ret;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY uint64_t GetCurrentTimestap() {
  struct timeval tv {};
  int ret = gettimeofday(&tv, nullptr);
  GE_LOGE_IF(ret != 0, "Func gettimeofday may failed: ret=%d", ret);
  auto total_use_time = tv.tv_usec + tv.tv_sec * 1000000;  // 1000000: seconds to microseconds
  return static_cast<uint64_t>(total_use_time);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY uint32_t GetCurrentSecondTimestap() {
  struct timeval tv {};
  int ret = gettimeofday(&tv, nullptr);
  GE_LOGE_IF(ret != 0, "Func gettimeofday may failed: ret=%d", ret);
  auto total_use_time = tv.tv_sec;  // seconds
  return static_cast<uint32_t>(total_use_time);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool CheckInt64MulOverflow(int64_t a, int64_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (INT64_MAX / b)) {
        return false;
      }
    } else {
      if (b < (INT64_MIN / a)) {
        return false;
      }
    }
  } else {
    if (b > 0) {
      if (a < (INT64_MIN / b)) {
        return false;
      }
    } else {
      if ((a != 0) && (b < (INT64_MAX / a))) {
        return false;
      }
    }
  }
  return true;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY std::string RealPath(const char *path) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(path == nullptr, return "", "path pointer is NULL.");
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
    strlen(path) >= PATH_MAX,
    ErrorManager::GetInstance().ATCReportErrMessage("E19002", {"filepath", "size"}, {path, std::to_string(PATH_MAX)});
    return "", "Path[%s] len is too long, it must be less than %d", path, PATH_MAX);

  // Nullptr is returned when the path does not exist or there is no permission
  // Return absolute path when path is accessible
  std::string res;
  char resolved_path[PATH_MAX] = {0};
  if (realpath(path, resolved_path) != nullptr) {
    res = resolved_path;
  }

  return res;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool CheckInputPathValid(const std::string &file_path,
                                                                          const std::string &atc_param) {
  // The specified path is empty
  std::map<std::string, std::string> args_map;
  if (file_path.empty()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10004", {"parameter"}, {atc_param});
    GELOGW("Input parameter %s is empty.", file_path.c_str());
    return false;
  }
  std::string real_path = RealPath(file_path.c_str());
  // Unable to get absolute path (does not exist or does not have permission to access)
  if (real_path.empty()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19000", {"path", "errmsg"}, {file_path, strerror(errno)});
    GELOGW("Path[%s]'s realpath is empty, errmsg[%s]", file_path.c_str(), strerror(errno));
    return false;
  }

  // A regular matching expression to verify the validity of the input file path
  // Path section: Support upper and lower case letters, numbers dots(.) chinese and underscores
  // File name section: Support upper and lower case letters, numbers, underscores chinese and dots(.)
  std::string mode = "^[\u4e00-\u9fa5A-Za-z0-9./_-]+$";

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
    !ValidateStr(real_path, mode),
    ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                    {atc_param, real_path, kPathValidReason});
    return false, "Invalid value for %s[%s], %s.", atc_param.c_str(), real_path.c_str(), kPathValidReason);

  // The absolute path points to a file that is not readable
  if (access(real_path.c_str(), R_OK) != 0) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19003", {"file", "errmsg"}, {file_path.c_str(), strerror(errno)});
    GELOGW("Read file[%s] failed, errmsg[%s]", file_path.c_str(), strerror(errno));
    return false;
  }

  return true;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool CheckOutputPathValid(const std::string &file_path,
                                                                           const std::string &atc_param) {
  // The specified path is empty
  if (file_path.empty()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10004", {"parameter"}, {atc_param});
    GELOGW("Input parameter's value is empty.");
    return false;
  }

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
    strlen(file_path.c_str()) >= PATH_MAX, ErrorManager::GetInstance().ATCReportErrMessage(
                                             "E19002", {"filepath", "size"}, {file_path, std::to_string(PATH_MAX)});
    return "", "Path[%s] len is too long, it must be less than %d", file_path.c_str(), PATH_MAX);

  // A regular matching expression to verify the validity of the input file path
  // Path section: Support upper and lower case letters, numbers dots(.) chinese and underscores
  // File name section: Support upper and lower case letters, numbers, underscores chinese and dots(.)
  std::string mode = "^[\u4e00-\u9fa5A-Za-z0-9./_-]+$";

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
    !ValidateStr(file_path, mode),
    ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                    {atc_param, file_path, kPathValidReason});
    return false, "Invalid value for %s[%s], %s.", atc_param.c_str(), file_path.c_str(), kPathValidReason);

  std::string real_path = RealPath(file_path.c_str());
  // Can get absolute path (file exists)
  if (!real_path.empty()) {
    // File is not readable or writable
    if (access(real_path.c_str(), W_OK | F_OK) != 0) {
      ErrorManager::GetInstance().ATCReportErrMessage("E19004", {"file", "errmsg"}, {real_path, strerror(errno)});
      GELOGW("Write file[%s] failed, errmsg[%s]", real_path.c_str(), strerror(errno));
      return false;
    }
  } else {
    // Find the last separator
    int path_split_pos = static_cast<int>(file_path.size() - 1);
    for (; path_split_pos >= 0; path_split_pos--) {
      if (file_path[path_split_pos] == '\\' || file_path[path_split_pos] == '/') {
        break;
      }
    }
    if (path_split_pos == 0) {
      return true;
    }
    if (path_split_pos != -1) {
      std::string prefix_path = std::string(file_path).substr(0, static_cast<size_t>(path_split_pos));
      // Determine whether the specified path is valid by creating the path
      if (CreateDirectory(prefix_path) != 0) {
        ErrorManager::GetInstance().ATCReportErrMessage("E19006", {"path"}, {file_path});
        GELOGW("Can not create directory[%s].", file_path.c_str());
        return false;
      }
    }
  }

  return true;
}

FMK_FUNC_HOST_VISIBILITY bool ValidateStr(const std::string &str, const std::string &mode) {
  char ebuff[kMaxBuffSize];
  regex_t reg;
  int cflags = REG_EXTENDED | REG_NOSUB;
  int ret = regcomp(&reg, mode.c_str(), cflags);
  if (ret) {
    regerror(ret, &reg, ebuff, kMaxBuffSize);
    GELOGW("regcomp failed, reason: %s", ebuff);
    regfree(&reg);
    return true;
  }

  ret = regexec(&reg, str.c_str(), 0, nullptr, 0);
  if (ret) {
    regerror(ret, &reg, ebuff, kMaxBuffSize);
    GELOGE(ge::PARAM_INVALID, "regexec failed, reason: %s", ebuff);
    regfree(&reg);
    return false;
  }

  regfree(&reg);
  return true;
}

FMK_FUNC_HOST_VISIBILITY bool IsValidFile(const char *file_path) {
  if (file_path == nullptr) {
    GELOGE(PARAM_INVALID, "Config path is null.");
    return false;
  }
  if (!CheckInputPathValid(file_path)) {
    GELOGE(PARAM_INVALID, "Config path is invalid: %s", file_path);
    return false;
  }
  // Normalize the path
  std::string resolved_file_path = RealPath(file_path);
  if (resolved_file_path.empty()) {
    GELOGE(PARAM_INVALID, "Invalid input file path [%s], make sure that the file path is correct.", file_path);
    return false;
  }

  mmStat_t stat = {0};
  int32_t ret = mmStatGet(resolved_file_path.c_str(), &stat);
  if (ret != EN_OK) {
    GELOGE(PARAM_INVALID, "cannot get config file status, which path is %s, maybe not exist, return %d, errcode %d",
           resolved_file_path.c_str(), ret, mmGetErrorCode());
    return false;
  }
  if ((stat.st_mode & S_IFMT) != S_IFREG) {
    GELOGE(PARAM_INVALID, "config file is not a common file, which path is %s, mode is %u", resolved_file_path.c_str(),
           stat.st_mode);
    return false;
  }
  if (stat.st_size > kMaxConfigFileByte) {
    GELOGE(PARAM_INVALID, "config file %s size[%ld] is larger than max config file Bytes[%u]",
           resolved_file_path.c_str(), stat.st_size, kMaxConfigFileByte);
    return false;
  }
  return true;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status CheckPath(const char *path, size_t length) {
  if (path == nullptr) {
    GELOGE(PARAM_INVALID, "Config path is invalid.");
    return PARAM_INVALID;
  }

  if (strlen(path) != length) {
    GELOGE(PARAM_INVALID, "Path is invalid or length of config path is not equal to given length.");
    return PARAM_INVALID;
  }

  if (length == 0 || length > MMPA_MAX_PATH) {
    GELOGE(PARAM_INVALID, "Length of config path is invalid.");
    return PARAM_INVALID;
  }

  INT32 is_dir = mmIsDir(path);
  if (is_dir != EN_OK) {
    GELOGE(PATH_INVALID, "Open directory %s failed, maybe it is not exit or not a dir", path);
    return PATH_INVALID;
  }

  if (mmAccess2(path, M_R_OK) != EN_OK) {
    GELOGE(PATH_INVALID, "Read path[%s] failed, errmsg[%s]", path, strerror(errno));
    return PATH_INVALID;
  }
  return SUCCESS;
}
}  //  namespace ge
