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

#ifndef INC_FRAMEWORK_COMMON_GE_MODEL_INOUT_INFO_H_
#define INC_FRAMEWORK_COMMON_GE_MODEL_INOUT_INFO_H_

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <securec.h>

#include "framework/common/ge_model_inout_types.h"
#include "external/ge/ge_api_types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/def_types.h"
#include "common/checker.h"

namespace ge {
class ModelIntroduction;
class BaseTlvBlock {
public:
  virtual size_t Size() = 0;
  virtual bool Serilize(uint8_t ** const addr, size_t &left_size) = 0;
  virtual bool NeedSave() = 0;
};

class ModelTensorDesc {
public:
  friend class ModelIntroduction;
  size_t Size() {
    const size_t size = static_cast<size_t>(sizeof(base_info) + base_info.name_len + base_info.dims_len +
                    base_info.dimsV2_len + base_info.shape_range_len);
    return size;
  }
  bool Serilize(uint8_t ** const addr, size_t &left_size) {
    GE_ASSERT_NOTNULL(addr, "input param is invalid, addr is null.");
    GE_ASSERT_NOTNULL(*addr, "addr ptr is null.");
    errno_t ret = memcpy_s(*addr, left_size, static_cast<const void *>(&base_info),
                           sizeof(ModelTensorDescBaseInfo));
    GE_ASSERT_EOK(ret, "serilize ModelTensorDesc::ModelTensorDescBaseInfo failed");
    *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr) + sizeof(ModelTensorDescBaseInfo)));
    left_size -= sizeof(ModelTensorDescBaseInfo);

    if ((name.data() != nullptr) && (base_info.name_len != 0U)) {
      ret = memcpy_s(*addr, left_size, static_cast<const void *>(name.data()), static_cast<size_t>(base_info.name_len));
      GE_ASSERT_EOK(ret, "serilize ModelTensorDesc::name failed");
      *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr) + base_info.name_len));
      left_size -= base_info.name_len;
    }
    
    if ((dims.data() != nullptr) && (base_info.dims_len != 0U)) {
      ret = memcpy_s(*addr, left_size, static_cast<void *>(dims.data()), static_cast<size_t>(base_info.dims_len));
      GE_ASSERT_EOK(ret, "serilize ModelTensorDesc::dims failed");
      *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr) + base_info.dims_len));
      left_size -= base_info.dims_len;
    }
  
    if ((dimsV2.data() != nullptr) && (base_info.dimsV2_len != 0U)) {
      ret = memcpy_s(*addr, left_size, static_cast<void *>(dimsV2.data()),
                     static_cast<size_t>(base_info.dimsV2_len));
      if (ret != EOK) {
        GELOGE(FAILED, "serilize ModelTensorDesc::dimsVe failed");
        return false;
      }
      *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr) + base_info.dimsV2_len));
      left_size -= base_info.dimsV2_len;
    }
  
    if ((shape_range.data() != nullptr) && (base_info.shape_range_len != 0U)) {
      ret = memcpy_s(*addr, left_size, static_cast<void *>(shape_range.data()),
                     static_cast<size_t>(base_info.shape_range_len));
      if (ret != EOK) {
        GELOGE(FAILED, "serilize ModelTensorDesc::dimsVe failed");
        return false;
      }
      *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr) + base_info.shape_range_len));
      left_size -= base_info.shape_range_len;
    }
    return true;
  }
private:
  ModelTensorDescBaseInfo base_info;
  std::string name;
  std::vector<int64_t> dims;
  std::vector<int64_t> dimsV2;
  std::vector<std::pair<int64_t, int64_t>> shape_range;
};

class ModelTensorDescsValue : public BaseTlvBlock {
public:
  friend class ModelIntroduction;
  size_t Size() override {
    size_t size = 0U;
    for (auto &desc : descs) {
      size += desc.Size();
    }
    size += sizeof(uint32_t);
    return size;
  }
  bool Serilize(uint8_t ** const addr, size_t &left_size) override {
    if (addr == nullptr) {
      GELOGE(PARAM_INVALID, "input param is invalid, addr is null.");
      return false;
    }
    GE_ASSERT_NOTNULL(*addr, "addr ptr is null.");
    const errno_t ret = memcpy_s(*addr, left_size, static_cast<void *>(&tensor_desc_size), sizeof(uint32_t));
    if (ret != EOK) {
      GELOGE(FAILED, "serilize ModelTensorDescsValue::tensor_desc_size failed");
      return false;
    }
    *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr) + sizeof(uint32_t)));
    left_size -= sizeof(uint32_t);
  
    for (auto desc : descs) {
      if (!desc.Serilize(addr, left_size)) {
        GELOGE(FAILED, "save ModelTensorDesc failed.");
        return false;
      }
    }
    return true;
  }
  bool NeedSave() override {
    return tensor_desc_size > 0U;
  }
private:
  uint32_t tensor_desc_size = 0U;
  std::vector<ModelTensorDesc> descs;
};

class vecIntValue : public BaseTlvBlock {
public:
  friend class ModelIntroduction;
  size_t Size() override {
    const size_t size = vec_size * sizeof(int64_t) + sizeof(uint32_t);
    return size;
  }
  bool Serilize(uint8_t ** const addr, size_t &left_size) override {
    if (addr == nullptr) {
      GELOGE(PARAM_INVALID, "input param is invalid, addr  is null.");
      return false;
    }
    GE_ASSERT_NOTNULL(*addr, "addr ptr is null.");

    errno_t ret = memcpy_s(*addr, left_size, static_cast<void *>(&vec_size), sizeof(uint32_t));
    if (ret != EOK) {
      GELOGE(FAILED, "serilize vecIntValue::vec_size failed");
      return false;
    }
    *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr) + sizeof(uint32_t)));
    left_size -= sizeof(uint32_t);

    if ((value.data() != nullptr) && (vec_size != 0U)) {
      ret = memcpy_s(*addr, left_size, static_cast<void *>(value.data()), sizeof(uint64_t) * vec_size);
      if (ret != EOK) {
        GELOGE(FAILED, "serilize vecIntValue::value failed");
        return false;
      }
      *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr)
                                      + static_cast<uint64_t>(sizeof(uint64_t) * vec_size)));
      left_size -= sizeof(uint64_t) * vec_size;
    }
  
    return true;
  }
  bool NeedSave() override {
    return vec_size > 0U;
  }
private:
  uint32_t vec_size = 0U;
  std::vector<int64_t> value;
};

class vecIntIntValue : public BaseTlvBlock {
public:
  friend class ModelIntroduction;
  size_t Size() override {
    size_t size = 0U;
    for (auto &num : vec_part_size) {
      size += num * sizeof(int64_t);
    }
    size += vec_size * sizeof(uint32_t);
    size += sizeof(uint32_t);
    return size;
  }
  bool Serilize(uint8_t ** const addr, size_t &left_size) override {
    if (addr == nullptr) {
      GELOGE(PARAM_INVALID, "input param is invalid, addr is null.");
      return false;
    }
    GE_ASSERT_NOTNULL(*addr, "addr ptr is null.");

    errno_t ret = memcpy_s(*addr, left_size, static_cast<void *>(&vec_size), sizeof(uint32_t));
    if (ret != EOK) {
      GELOGE(FAILED, "serilize vecIntIntValue::vec_size failed");
      return false;
    }
    *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr) + sizeof(uint32_t)));
    left_size -= sizeof(uint32_t);
  
    if ((vec_part_size.data() != nullptr) && (vec_size != 0U)) {
      ret = memcpy_s(*addr, left_size, static_cast<void *>(vec_part_size.data()), sizeof(uint32_t) * vec_size);
      if (ret != EOK) {
        GELOGE(FAILED, "serilize vecIntIntValue::vec_part_size failed");
        return false;
      }
      *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr)
                                      + static_cast<uint64_t>(sizeof(uint32_t) * vec_size)));
      left_size -= sizeof(uint32_t) * vec_size;
    }
  
    for (size_t i = 0; i < vec_size; ++i) {
      if ((value[i].data() != nullptr) && (vec_part_size[i] != 0U)) {
        ret = memcpy_s(*addr, left_size, static_cast<void *>(value[i].data()), sizeof(int64_t) * vec_part_size[i]);
        if (ret != EOK) {
          GELOGE(FAILED, "serilize vecIntIntValue::value failed");
          return false;
        }
        *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr)
                                        + static_cast<uint64_t>(sizeof(int64_t) * vec_part_size[i])));
        left_size -= sizeof(int64_t) * vec_part_size[i];
      }
    }
    return true;
  }
  bool NeedSave() override {
    return vec_size > 0U;
  }
private:
  uint32_t vec_size = 0U;
  std::vector<uint32_t> vec_part_size;
  std::vector<std::vector<int64_t>> value;
};

class vecStrValue : public BaseTlvBlock {
public:
  friend class ModelIntroduction;
  size_t Size() override {
    size_t size = 0U;
    for (const auto &num : str_len) {
      size += num * sizeof(char);
    }
    size += vec_size * sizeof(uint32_t);
    size += sizeof(uint32_t);
    return size;
  }
  bool Serilize(uint8_t ** const addr, size_t &left_size) override {
    if (addr == nullptr) {
      GELOGE(PARAM_INVALID, "input param is invalid, addr is null.");
      return false;
    }
    GE_ASSERT_NOTNULL(*addr, "addr ptr is null.");

    errno_t ret = memcpy_s(*addr, left_size, &vec_size, sizeof(uint32_t));
    if (ret != EOK) {
      GELOGE(FAILED, "serilize vecStrValue::vec_size failed");
      return false;
    }
    *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr) + static_cast<uint64_t>(sizeof(uint32_t))));
    left_size -= sizeof(uint32_t);
  
    if ((str_len.data() != nullptr) && (vec_size != 0U)) {
      ret = memcpy_s(*addr, left_size, static_cast<void *>(str_len.data()), sizeof(uint32_t) * vec_size);
      if (ret != EOK) {
        GELOGE(FAILED, "serilize vecStrValue::str_len failed");
        return false;
      }
      *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr)
                                      + static_cast<uint64_t>(sizeof(uint32_t) * vec_size)));
      left_size -= sizeof(uint32_t) * vec_size;
    }
  
    for (size_t i = 0; i < vec_size; ++i) {
      if ((value[i].data() != nullptr) && (str_len[i] != 0U)) {
        ret = memcpy_s(*addr, left_size, static_cast<const void *>(value[i].data()), sizeof(char) * str_len[i]);
        if (ret != EOK) {
          GELOGE(FAILED, "serilize vecStrValue::value failed");
          return false;
        }
        *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr)
                                        + static_cast<uint64_t>(sizeof(char) * str_len[i])));
        left_size -= sizeof(char) * str_len[i];
      }
    }
    return true;
  }
  bool NeedSave() override {
    return vec_size > 0U;
  }
private:
  uint32_t vec_size = 0U;
  std::vector<uint32_t> str_len;
  std::vector<std::string> value;
};
}  // namespace ge
#endif