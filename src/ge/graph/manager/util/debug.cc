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

#include "graph/manager/util/debug.h"

#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"

using google::protobuf::Message;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::FileOutputStream;

namespace ge {
Debug::Debug() = default;

Debug::~Debug() = default;

void Debug::DumpProto(const Message &proto, const char *file) {
  int fd = open(file, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  if (fd == -1) {
    GELOGW("Write %s failed", file);
    return;
  }
  auto output = ge::MakeShared<FileOutputStream>(fd);
  if (output == nullptr) {
    GELOGW("create output failed.");
    if (close(fd) != 0) {
      GELOGW("close fd failed.");
    }
    return;
  }
  bool ret = google::protobuf::TextFormat::Print(proto, output.get());
  if (!ret) {
    GELOGW("dump proto failed.");
  }
  if (close(fd) != 0) {
    GELOGW("close fd failed.");
  }
}

Status Debug::DumpDevMem(const char *file, const void *addr, uint32_t size) {
  uint8_t *host_addr = nullptr;
  rtError_t ret = rtMallocHost(reinterpret_cast<void **>(&host_addr), size);
  if (ret != RT_ERROR_NONE) {
    GELOGE(FAILED, "Call rt api rtMallocHost failed.");
    return FAILED;
  }
  GE_MAKE_GUARD_RTMEM(host_addr);
  ret = rtMemcpy(host_addr, size, addr, size, RT_MEMCPY_DEVICE_TO_HOST);
  if (ret != RT_ERROR_NONE) {
    GELOGE(FAILED, "Call rt api rtMemcpy failed, ret: 0x%X", ret);
    return FAILED;
  }

  GE_CHK_STATUS_RET(MemoryDumper::DumpToFile(file, host_addr, size));
  return SUCCESS;
}
}  // namespace ge
