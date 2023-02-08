/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include "common/proto_util.h"

#include <fstream>

#include "mmpa/mmpa_api.h"
#include "graph/def_types.h"
#include "framework/common/util.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

namespace ge {
namespace {
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::ZeroCopyInputStream;

/*
 * kProtoReadBytesLimit and kWarningThreshold are real arguments of CodedInputStream::SetTotalBytesLimit.
 * In order to prevent integer overflow and excessive memory allocation during protobuf processing,
 * it is necessary to limit the length of proto message (call SetTotalBytesLimit function).
 * In theory, the minimum message length that causes an integer overflow is 512MB, and the default is 64MB.
 * If the limit of warning_threshold is exceeded, the exception information will be printed in stderr.
 * If such an exception is encountered during operation,
 * the proto file can be divided into several small files or the limit value can be increased.
 */
const int32_t kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

const size_t kMaxErrorStrLength = 128U;
}  // namespace

static bool ReadProtoFromCodedInputStream(CodedInputStream &coded_stream, google::protobuf::Message *const proto) {
  if (proto == nullptr) {
    GELOGE(FAILED, "incorrect parameter. nullptr == proto");
    return false;
  }

  coded_stream.SetTotalBytesLimit(kProtoReadBytesLimit);
  return proto->ParseFromCodedStream(&coded_stream);
}

bool ReadProtoFromArray(const void *const data, const int32_t size, google::protobuf::Message *const proto) {
  if ((proto == nullptr) || (data == nullptr) || (size == 0)) {
    GELOGE(FAILED, "incorrect parameter. proto is nullptr || data is nullptr || size is 0");
    return false;
  }

  google::protobuf::io::CodedInputStream coded_stream(PtrToPtr<void, uint8_t>(const_cast<void *>(data)), size);
  return ReadProtoFromCodedInputStream(coded_stream, proto);
}

bool ReadProtoFromText(const char_t *const file, google::protobuf::Message *const message) {
  if ((file == nullptr) || (message == nullptr)) {
    GELOGE(FAILED, "incorrect parameter. nullptr == file || nullptr == message");
    return false;
  }

  const std::string real_path = RealPath(file);
  if (real_path.empty()) {
    char_t err_buf[kMaxErrorStrLength + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLength);
    ErrorManager::GetInstance().ATCReportErrMessage("E19000", {"path", "errmsg"}, {file, err_msg});
    GELOGE(FAILED, "[Do][RealPath]Path[%s]'s realpath is empty, errmsg[%s]", file, err_msg);
    return false;
  }

  if (GetFileLength(real_path) == -1) {
    GELOGE(FAILED, "file size not valid.");
    return false;
  }

  std::ifstream fs(real_path.c_str(), std::ifstream::in);
  if (!fs.is_open()) {
    REPORT_INNER_ERROR("E19999", "open file:%s failed", real_path.c_str());
    GELOGE(ge::FAILED, "[Open][ProtoFile]Failed, real path %s, orginal file path %s",
           real_path.c_str(), file);
    return false;
  }

  google::protobuf::io::IstreamInputStream input(&fs);
  const bool ret = google::protobuf::TextFormat::Parse(&input, message);
  GE_IF_BOOL_EXEC(!ret, ErrorManager::GetInstance().ATCReportErrMessage("E19018", {"protofile"}, {file});
                  GELOGE(static_cast<uint32_t>(ret), "[Parse][File]Through"
                        "[google::protobuf::TextFormat::Parse] failed, file %s", file));
  fs.close();

  return ret;
}
}  //  namespace ge
