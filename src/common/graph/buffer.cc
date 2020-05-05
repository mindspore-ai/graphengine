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

#include "graph/buffer.h"
#include "proto/ge_ir.pb.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
Buffer::Buffer() {
  data_.InitDefault();
  if (data_.GetProtoMsg()) {
    buffer_ = data_.GetProtoMsg()->mutable_bt();
  }
}

Buffer::Buffer(const Buffer &other) {
  // Share data
  data_ = other.data_;
  buffer_ = other.buffer_;
}

Buffer::Buffer(std::size_t buffer_size, std::uint8_t default_val) : Buffer() {  // default
  auto proto_msg = data_.GetProtoMsg();
  if (proto_msg != nullptr) {
    try {
      proto_msg->set_bt(std::string(buffer_size, default_val));
      buffer_ = proto_msg->mutable_bt();
    } catch (std::bad_alloc &e) {
      GELOGE(MEMALLOC_FAILED, "Failed to alloc buffer memory, buffer size %zu", buffer_size);
      buffer_ = nullptr;
    }
  }
}

Buffer Buffer::CopyFrom(const std::uint8_t *data, std::size_t buffer_size) {
  Buffer buffer;
  auto proto_msg = buffer.data_.GetProtoMsg();
  if (proto_msg != nullptr && data != nullptr) {
    try {
      proto_msg->set_bt(data, buffer_size);
      buffer.buffer_ = proto_msg->mutable_bt();
    } catch (std::bad_alloc &e) {
      GELOGE(MEMALLOC_FAILED, "Failed to alloc buffer memory, buffer size %zu", buffer_size);
      buffer.buffer_ = nullptr;
    }
  }
  return buffer;
}

Buffer::Buffer(const std::shared_ptr<google::protobuf::Message> &proto_owner, proto::AttrDef *buffer)
    : data_(proto_owner, buffer) {
  if (data_.GetProtoMsg() != nullptr) {
    buffer_ = data_.GetProtoMsg()->mutable_bt();
  }
}

Buffer::Buffer(const std::shared_ptr<google::protobuf::Message> &proto_owner, std::string *buffer)
    : data_(proto_owner, nullptr) {
  buffer_ = buffer;
}

Buffer &Buffer::operator=(const Buffer &other) {
  if (&other != this) {
    // Share data
    data_ = other.data_;
    buffer_ = other.buffer_;
  }
  return *this;
}

const std::uint8_t *Buffer::GetData() const {
  if (buffer_ != nullptr) {
    return (const std::uint8_t *)buffer_->data();
  }
  return nullptr;
}

std::uint8_t *Buffer::GetData() {
  if (buffer_ != nullptr && !buffer_->empty()) {
    // Avoid copy on write
    (void)(*buffer_)[0];
    return reinterpret_cast<uint8_t *>(const_cast<char *>(buffer_->data()));
  }
  return nullptr;
}

std::size_t Buffer::GetSize() const {
  if (buffer_ != nullptr) {
    return buffer_->size();
  }
  return 0;
}

void Buffer::ClearBuffer() {
  if (buffer_ != nullptr) {
    buffer_->clear();
  }
}
}  // namespace ge
