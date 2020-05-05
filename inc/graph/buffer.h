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

#ifndef INC_GRAPH_BUFFER_H_
#define INC_GRAPH_BUFFER_H_

#include <graph/types.h>
#include <memory>
#include <string>
#include <vector>
#include "detail/attributes_holder.h"

namespace ge {
#ifdef HOST_VISIBILITY
#define GE_FUNC_HOST_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_HOST_VISIBILITY
#endif
#ifdef DEV_VISIBILITY
#define GE_FUNC_DEV_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_DEV_VISIBILITY
#endif

using std::shared_ptr;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Buffer {
 public:
  Buffer();
  Buffer(const Buffer &other);

  explicit Buffer(std::size_t bufferSize, std::uint8_t defualtVal = 0);

  ~Buffer() = default;

  Buffer &operator=(const Buffer &other);

  static Buffer CopyFrom(const std::uint8_t *data, std::size_t bufferSize);

  const std::uint8_t *GetData() const;
  std::uint8_t *GetData();
  std::size_t GetSize() const;
  void ClearBuffer();

  // For compatibility
  inline const std::uint8_t *data() const { return GetData(); }
  inline std::uint8_t *data() { return GetData(); }
  inline std::size_t size() const { return GetSize(); }
  inline void clear() { return ClearBuffer(); }
  uint8_t operator[](size_t index) const {
    if (buffer_ != nullptr && index < buffer_->size()) {
      return (uint8_t)(*buffer_)[index];
    }
    return 0xff;
  }

 private:
  GeIrProtoHelper<proto::AttrDef> data_;
  std::string *buffer_ = nullptr;

  // Create from protobuf obj
  Buffer(const ProtoMsgOwner &protoOnwer, proto::AttrDef *buffer);
  Buffer(const ProtoMsgOwner &protoOnwer, std::string *buffer);

  friend class GeAttrValueImp;
  friend class GeTensor;
};
}  // namespace ge
#endif  // INC_GRAPH_BUFFER_H_
