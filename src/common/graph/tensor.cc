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

#include "external/graph/tensor.h"
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ge_tensor.h"
#include "securec.h"
#include "utils/attr_utils.h"
#include "utils/tensor_adapter.h"
#include "utils/tensor_utils.h"
#include "utils/type_utils.h"

namespace {
/// Extra 8 bytes store pointer of string
/// Extra 1 byte store '\0'
const int EXTRA_STORE_POINTER_FOR_STRING = 8;
const int EXTRA_STORE_POINTER_FOR_STRING_AND_END_SYMBOL = 9;
const int64_t UNKNOWN_DIM_SIZE = -1;
}  // namespace

namespace ge {
// If not overflow return true
static bool Int64MulNotOverflow(int64_t a, int64_t b) {
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

class TensorDescImpl {
 public:
  TensorDescImpl() = default;
  ~TensorDescImpl() = default;
  TensorDescImpl(const Shape &shape, Format format, DataType dt) : shape_(shape), format_(format), data_type_(dt) {}

  Shape shape_;
  std::vector<std::pair<int64_t, int64_t>> range_;
  Format format_ = FORMAT_ND;
  Format origin_format_ = FORMAT_ND;
  DataType data_type_ = DT_FLOAT;
  Shape origin_shape_;
  int64_t size_ = 0;
  int64_t real_dim_cnt_ = 0;
  std::string name_;
};

class TensorImpl {
 public:
  TensorImpl() = default;
  ~TensorImpl() = default;

  explicit TensorImpl(const TensorDesc &tensor_desc) : ge_tensor(TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc)) {}
  TensorImpl(const TensorDesc &tensor_desc, const std::vector<uint8_t> &data)
      : ge_tensor(TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc), data) {}
  TensorImpl(const TensorDesc &tensor_desc, const uint8_t *data, size_t size)
      : ge_tensor(TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc), data, size) {}
  TensorImpl(TensorDesc &&tensor_desc, std::vector<uint8_t> &&data)
      : ge_tensor(TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc), std::move(data)) {}

  GeTensor ge_tensor;
};

class ShapeImpl {
 public:
  ShapeImpl() = default;
  ~ShapeImpl() = default;
  explicit ShapeImpl(const std::vector<int64_t> &dims) {
    bool is_unknown_dim_num = false;
    for (const auto &dim : dims) {
      if (dim == UNKNOWN_DIM_NUM) {
        is_unknown_dim_num = true;
        break;
      }
    }
    dims_ = is_unknown_dim_num ? std::vector<int64_t>({UNKNOWN_DIM_NUM}) : dims;
  }

  std::vector<int64_t> dims_;
};

Shape::Shape() { impl_ = ComGraphMakeShared<ShapeImpl>(); }

Shape::Shape(const std::vector<int64_t> &dims) { impl_ = ComGraphMakeShared<ShapeImpl>(dims); }

size_t Shape::GetDimNum() const {
  if (impl_ != nullptr) {
    for (auto i : impl_->dims_) {
      if (i == UNKNOWN_DIM_NUM) {
        return 0;
      }
    }
    return impl_->dims_.size();
  }
  return 0;
}

int64_t Shape::GetDim(size_t idx) const {
  if (impl_ != nullptr) {
    if (idx >= impl_->dims_.size()) {
      return 0;
    }
    return impl_->dims_[idx];
  }
  return 0;
}

graphStatus Shape::SetDim(size_t idx, int64_t value) {
  if (impl_ != nullptr) {
    if (idx >= impl_->dims_.size()) {
      return GRAPH_FAILED;
    }
    impl_->dims_[idx] = value;
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

std::vector<int64_t> Shape::GetDims() const {
  vector<int64_t> dims;
  if (impl_ != nullptr) {
    return impl_->dims_;
  }
  return dims;
}

int64_t Shape::GetShapeSize() const {
  if (impl_ != nullptr) {
    if (impl_->dims_.empty()) {
      return 0;
    }
    int64_t size = 1;
    for (auto i : impl_->dims_) {
      if (i == UNKNOWN_DIM_NUM || i == UNKNOWN_DIM) {
        return UNKNOWN_DIM_SIZE;
      }

      if (!Int64MulNotOverflow(size, i)) {
        GELOGE(GRAPH_FAILED, "mul overflow: %ld, %ld", size, i);
        size = 0;
        return size;
      }
      size *= i;
    }
    return size;
  }
  return 0;
}

TensorDesc::TensorDesc() {
  impl = ComGraphMakeShared<TensorDescImpl>();  // lint !e665
}

TensorDesc::TensorDesc(Shape shape, Format format, DataType dt) {
  impl = ComGraphMakeShared<TensorDescImpl>(shape, format, dt);  // lint !e665
  SetRealDimCnt(shape.GetDimNum());
}

TensorDesc::TensorDesc(const TensorDesc &desc) {
  // Copy
  impl = ComGraphMakeShared<TensorDescImpl>();  // lint !e665
  if (desc.impl != nullptr && impl != nullptr) {
    *impl = *desc.impl;
  }
}

TensorDesc::TensorDesc(TensorDesc &&desc) {
  // Move
  impl = std::move(desc.impl);
}

TensorDesc &TensorDesc::operator=(const TensorDesc &desc) {
  // Copy
  if (&desc != this) {
    impl = ComGraphMakeShared<TensorDescImpl>();
    if (desc.impl != nullptr && impl != nullptr) {
      *impl = *desc.impl;
    }
  }
  return *this;
}

TensorDesc &TensorDesc::operator=(TensorDesc &&desc) {
  if (&desc != this) {
    impl = std::move(desc.impl);
  }
  return *this;
}

void TensorDesc::Update(const Shape &shape, Format format, DataType dt) {
  if (impl != nullptr) {
    impl->shape_ = shape;
    impl->format_ = format;
    impl->data_type_ = dt;
  }
}

Shape TensorDesc::GetShape() const {
  if (impl != nullptr) {
    return impl->shape_;
  }
  return Shape();
}

void TensorDesc::SetShape(const Shape &shape) {
  if (impl != nullptr) {
    impl->shape_ = shape;
  }
}

// set shape with -2, it stand for unknown shape
graphStatus TensorDesc::SetUnknownDimNumShape() {
  if (impl != nullptr) {
    impl->shape_ = Shape({UNKNOWN_DIM_NUM});
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "Set unknown shape failed,because no impl class!");
  return GRAPH_FAILED;
}

// for unknown shape
graphStatus TensorDesc::SetShapeRange(const std::vector<std::pair<int64_t, int64_t>> &range) {
  if (impl != nullptr) {
    impl->range_ = range;
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "SetShapeRange failed!impl is nullptr!");
  return GRAPH_FAILED;
}
graphStatus TensorDesc::GetShapeRange(std::vector<std::pair<int64_t, int64_t>> &range) const {
  if (impl != nullptr) {
    range = impl->range_;
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "impl is nullptr!");
  return GRAPH_FAILED;
}

Shape TensorDesc::GetOriginShape() const {
  if (impl != nullptr) {
    return impl->origin_shape_;
  }
  return Shape();
}

void TensorDesc::SetOriginShape(const Shape &origin_shape) {
  if (impl != nullptr) {
    impl->origin_shape_ = origin_shape;
  }
}

Format TensorDesc::GetFormat() const {
  if (impl != nullptr) {
    return impl->format_;
  }
  return FORMAT_RESERVED;
}

void TensorDesc::SetFormat(Format format) {
  if (impl != nullptr) {
    impl->format_ = format;
  }
}

Format TensorDesc::GetOriginFormat() const {
  if (impl != nullptr) {
    return impl->origin_format_;
  }
  return FORMAT_RESERVED;
}

void TensorDesc::SetOriginFormat(Format origin_format) {
  if (impl != nullptr) {
    impl->origin_format_ = origin_format;
  }
}

DataType TensorDesc::GetDataType() const {
  if (impl != nullptr) {
    return impl->data_type_;
  }
  return DT_UNDEFINED;
}

void TensorDesc::SetDataType(DataType dt) {
  if (impl != nullptr) {
    impl->data_type_ = dt;
  }
}

void TensorDesc::SetSize(int64_t size) {
  if (impl != nullptr) {
    impl->size_ = size;
  }
}

int64_t TensorDesc::GetSize() const {
  if (impl != nullptr) {
    return impl->size_;
  }
  return 0;
}

void TensorDesc::SetRealDimCnt(const int64_t real_dim_cnt) {
  if (impl != nullptr) {
    impl->real_dim_cnt_ = real_dim_cnt;
  }
}

int64_t TensorDesc::GetRealDimCnt() const {
  if (impl != nullptr) {
    return impl->real_dim_cnt_;
  }
  return 0;
}

std::string TensorDesc::GetName() const {
  if (impl != nullptr) {
    return impl->name_;
  }
  return "";
}

void TensorDesc::SetName(const std::string &name) {
  if (impl != nullptr) {
    impl->name_ = name;
  }
}

Tensor::Tensor() { impl = ComGraphMakeShared<TensorImpl>(); }

Tensor::Tensor(const TensorDesc &tensor_desc) {
  impl = ComGraphMakeShared<TensorImpl>(tensor_desc);  // lint !e665
}

Tensor::Tensor(const TensorDesc &tensor_desc, const std::vector<uint8_t> &data) {
  uint64_t shape_size = tensor_desc.GetShape().GetShapeSize();
  DataType data_type = tensor_desc.GetDataType();
  uint32_t type_length;
  bool ret = TypeUtils::GetDataTypeLength(data_type, type_length);
  if (!ret) {
    GELOGW("datatype %d is not found.", data_type);
  }

  auto data_size = data.size();
  if (ret && (shape_size || (data_size != type_length))) {
    if (type_length != 0 && UINT64_MAX / type_length < shape_size) {
      GELOGW("mul overflow: %lu, %u", shape_size, type_length);
    } else {
      if (shape_size * type_length != data_size) {
        GELOGW("tensor length not equal: shape_byte_size=%lu, data_size=%zu, dt_type=%s.", shape_size * type_length,
               data_size, TypeUtils::DataTypeToSerialString(data_type).c_str());
      }
    }
  }
  impl = ComGraphMakeShared<TensorImpl>(tensor_desc, data);  // lint !e665
}

Tensor::Tensor(const TensorDesc &tensor_desc, const uint8_t *data, size_t size) {
  uint64_t shape_size = tensor_desc.GetShape().GetShapeSize();
  DataType data_type = tensor_desc.GetDataType();
  uint32_t type_length;
  bool ret = TypeUtils::GetDataTypeLength(data_type, type_length);
  if (!ret) {
    GELOGW("datatype %d is not found.", data_type);
  }
  if (ret && (shape_size || (size != type_length))) {
    if (type_length != 0 && UINT64_MAX / type_length < shape_size) {
      GELOGW("mul overflow: %lu, %u", shape_size, type_length);
    } else {
      if (shape_size * type_length != size) {
        GELOGW("tensor length not equal: shape_byte_size=%lu, data_size=%zu, dt_type=%s.", shape_size * type_length,
               size, TypeUtils::DataTypeToSerialString(data_type).c_str());
      }
    }
  }

  impl = ComGraphMakeShared<TensorImpl>(tensor_desc, data, size);  // lint !e665
}

Tensor::Tensor(TensorDesc &&tensor_desc, std::vector<uint8_t> &&data) {
  uint64_t shape_size = tensor_desc.GetShape().GetShapeSize();
  DataType data_type = tensor_desc.GetDataType();
  uint32_t type_length;
  bool ret = TypeUtils::GetDataTypeLength(data_type, type_length);
  if (!ret) {
    GELOGW("datatype %d is not found.", data_type);
  }

  auto data_size = data.size();
  if (ret && (shape_size || (data_size != type_length))) {
    if (type_length != 0 && UINT64_MAX / type_length < shape_size) {
      GELOGW("mul overflow: %lu, %u", shape_size, type_length);
    } else {
      if (shape_size * type_length != data_size) {
        GELOGW("tensor length not equal: shape_byte_size=%lu, data_size=%zu, dt_type=%s.", shape_size * type_length,
               data_size, TypeUtils::DataTypeToSerialString(data_type).c_str());
      }
    }
  }
  impl = ComGraphMakeShared<TensorImpl>(std::move(tensor_desc), std::move(data));  // lint !e665
}

TensorDesc Tensor::GetTensorDesc() const {
  if (impl != nullptr) {
    return TensorAdapter::GeTensorDesc2TensorDesc(impl->ge_tensor.MutableTensorDesc());
  }
  return TensorDesc();
}

graphStatus Tensor::SetTensorDesc(const TensorDesc &tensor_desc) {
  if (impl != nullptr) {
    impl->ge_tensor.SetTensorDesc(TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc));
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

const uint8_t *Tensor::GetData() const {
  if (impl != nullptr) {
    return impl->ge_tensor.GetData().data();
  }
  return nullptr;
}

uint8_t *Tensor::GetData() {
  if (impl != nullptr) {
    return impl->ge_tensor.MutableData().data();
  }
  return nullptr;
}

size_t Tensor::GetSize() const {
  if (impl != nullptr) {
    return impl->ge_tensor.GetData().size();
  }
  return 0;
}

graphStatus Tensor::SetData(std::vector<uint8_t> &&data) {
  if (impl != nullptr) {
    (void)impl->ge_tensor.SetData(data);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

graphStatus Tensor::SetData(const std::vector<uint8_t> &data) {
  if (impl != nullptr) {
    (void)impl->ge_tensor.SetData(data);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

graphStatus Tensor::SetData(const uint8_t *data, size_t size) {
  if (impl != nullptr) {
    (void)impl->ge_tensor.SetData(data, size);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

graphStatus Tensor::SetData(const std::string &data) {
  if (impl != nullptr && (!data.empty())) {
    /// Extra 8 bytes store pointer of string
    /// Extra 1 byte store '\0'
    size_t total_size = data.size() + EXTRA_STORE_POINTER_FOR_STRING_AND_END_SYMBOL;
    std::unique_ptr<char[]> buff(new (std::nothrow) char[total_size]());
    if (buff == nullptr) {
      GELOGE(GRAPH_FAILED, "allocate string raw data buff failed");
      return GRAPH_FAILED;
    }
    uint64_t *p = reinterpret_cast<uint64_t *>(buff.get());
    // Front 8 bytes store pointer of string
    char *raw_data = buff.get() + EXTRA_STORE_POINTER_FOR_STRING;
    p[0] = reinterpret_cast<uintptr_t>(raw_data);
    int32_t memcpy_ret = memcpy_s(raw_data, total_size - EXTRA_STORE_POINTER_FOR_STRING, data.c_str(), data.size() + 1);
    GE_CHK_BOOL_RET_STATUS(memcpy_ret == EOK, GRAPH_FAILED, "copy data failed");
    (void)impl->ge_tensor.SetData(reinterpret_cast<const uint8_t *>(buff.get()), total_size);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
graphStatus Tensor::SetData(const std::vector<std::string> &data) {
  if (impl != nullptr) {
    if (data.empty()) {
      GELOGE(GRAPH_FAILED, "there is no data, please check the input variable");
      return GRAPH_FAILED;
    }
    size_t total_size = 0;
    for (auto str : data) {
      /// Extra 8 bytes store pointer of each string
      /// Extra 1 byte store '\0'
      total_size += (str.size() + EXTRA_STORE_POINTER_FOR_STRING_AND_END_SYMBOL);
    }
    std::unique_ptr<char[]> buff(new (std::nothrow) char[total_size]);
    if (buff == nullptr) {
      GELOGE(GRAPH_FAILED, "allocate string raw data buff failed");
      return GRAPH_FAILED;
    }
    uint64_t *p = reinterpret_cast<uint64_t *>(buff.get());
    // Front some bytes store pointer of each string
    char *raw_data = buff.get() + data.size() * sizeof(uint64_t);
    uint64_t ptr_size = data.size() * sizeof(uint64_t);
    for (size_t i = 0; i < data.size(); ++i) {
      p[i] = reinterpret_cast<uintptr_t>(raw_data);
      if (total_size < ptr_size) {
        GELOGE(GRAPH_FAILED, "Subtraction invalid, total_size: %zu, ptr_size: %lu", total_size, ptr_size);
        return GRAPH_FAILED;
      }
      int32_t memcpy_ret = memcpy_s(raw_data, total_size - ptr_size, data[i].c_str(), data[i].size() + 1);
      GE_CHK_BOOL_RET_STATUS(memcpy_ret == EOK, GRAPH_FAILED, "copy data failed");
      raw_data += (data[i].size() + 1);
      ptr_size += (data[i].size() + 1);
    }

    (void)impl->ge_tensor.SetData(reinterpret_cast<const uint8_t *>(buff.get()), total_size);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

graphStatus Tensor::IsValid() {
  uint64_t shape_size = GetTensorDesc().GetShape().GetShapeSize();
  DataType data_type = GetTensorDesc().GetDataType();
  uint32_t type_length;
  bool ret = TypeUtils::GetDataTypeLength(data_type, type_length);
  if (!ret) {
    GELOGW("datatype %d is not found.", data_type);
    return GRAPH_SUCCESS;
  }

  size_t data_size = GetSize();
  if (data_type != DT_STRING) {
    if (shape_size || (data_size != type_length)) {
      if (type_length != 0 && UINT64_MAX / type_length < shape_size) {
        GELOGW("mul overflow: %lu, %u", shape_size, type_length);
      } else {
        if (shape_size * type_length != data_size) {
          GELOGW("tensor length not equal: shape_byte_size=%lu, data_size=%zu, dt_type=%s.", shape_size * type_length,
                 data_size, TypeUtils::DataTypeToSerialString(data_type).c_str());
          return GRAPH_FAILED;
        }
      }
    }
  }

  return GRAPH_SUCCESS;
}

Tensor Tensor::Clone() const {
  Tensor tensor;
  if (impl != nullptr && tensor.impl != nullptr) {
    tensor.impl->ge_tensor = impl->ge_tensor.Clone();
  }
  return tensor;
}

GeTensorDesc TensorAdapter::TensorDesc2GeTensorDesc(const TensorDesc &tensor_desc) {
  GeTensorDesc ge_tensor_desc(GeShape(tensor_desc.GetShape().GetDims()), tensor_desc.GetFormat(),
                              tensor_desc.GetDataType());
  ge_tensor_desc.SetOriginShape(GeShape(tensor_desc.GetOriginShape().GetDims()));
  ge_tensor_desc.SetOriginFormat(tensor_desc.GetOriginFormat());
  ge_tensor_desc.SetName(tensor_desc.GetName());
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  auto status = tensor_desc.GetShapeRange(shape_range);
  if (status != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Get shape range failed!");
    return ge_tensor_desc;
  }
  status = ge_tensor_desc.SetShapeRange(shape_range);
  if (status != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Set shape range failed!");
    return ge_tensor_desc;
  }
  auto size = tensor_desc.GetSize();
  TensorUtils::SetSize(ge_tensor_desc, size);

  auto real_dim_cnt = static_cast<uint32_t>(tensor_desc.GetRealDimCnt());
  TensorUtils::SetRealDimCnt(ge_tensor_desc, real_dim_cnt);
  return ge_tensor_desc;
}

TensorDesc TensorAdapter::GeTensorDesc2TensorDesc(const GeTensorDesc &ge_tensor_desc) {
  TensorDesc tensor_desc(Shape(ge_tensor_desc.GetShape().GetDims()), ge_tensor_desc.GetFormat(),
                         ge_tensor_desc.GetDataType());
  tensor_desc.SetOriginShape(Shape(ge_tensor_desc.GetOriginShape().GetDims()));
  tensor_desc.SetOriginFormat(ge_tensor_desc.GetOriginFormat());
  tensor_desc.SetName(ge_tensor_desc.GetName());
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  auto status = ge_tensor_desc.GetShapeRange(shape_range);
  if (status != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Get shape range failed!");
    return tensor_desc;
  }
  status = tensor_desc.SetShapeRange(shape_range);
  if (status != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Set shape range failed!");
    return tensor_desc;
  }
  int64_t size = 0;
  (void)TensorUtils::GetSize(ge_tensor_desc, size);
  tensor_desc.SetSize(size);

  uint32_t real_dim_cnt = 0;
  (void)TensorUtils::GetRealDimCnt(ge_tensor_desc, real_dim_cnt);
  tensor_desc.SetRealDimCnt(real_dim_cnt);
  return tensor_desc;
}

GeTensorPtr TensorAdapter::Tensor2GeTensor(const Tensor &tensor) {
  GeTensorPtr ge_tensor;
  if (tensor.impl != nullptr) {
    ge_tensor = ComGraphMakeShared<GeTensor>(tensor.impl->ge_tensor.Clone());  // lint !e665
  }
  return ge_tensor;
}

Tensor TensorAdapter::GeTensor2Tensor(const ConstGeTensorPtr &ge_tensor) {
  Tensor tensor;
  if (ge_tensor != nullptr && tensor.impl != nullptr) {
    tensor.impl->ge_tensor = ge_tensor->Clone();
  }
  return tensor;
}

ConstGeTensorPtr TensorAdapter::AsGeTensorPtr(const Tensor &tensor) {
  GeTensorPtr ge_tensor;
  if (tensor.impl != nullptr) {
    ge_tensor = ComGraphMakeShared<GeTensor>(tensor.impl->ge_tensor);  // lint !e665
  }
  return ge_tensor;
}

GeTensorPtr TensorAdapter::AsGeTensorPtr(Tensor &tensor) {
  GeTensorPtr ge_tensor;
  if (tensor.impl != nullptr) {
    ge_tensor = ComGraphMakeShared<GeTensor>(tensor.impl->ge_tensor);  // lint !e665
  }
  return ge_tensor;
}

const GeTensor TensorAdapter::AsGeTensor(const Tensor &tensor) {
  if (tensor.impl != nullptr) {
    return tensor.impl->ge_tensor;
  }
  return GeTensor();
}

GeTensor TensorAdapter::AsGeTensor(Tensor &tensor) {
  if (tensor.impl != nullptr) {
    return tensor.impl->ge_tensor;
  }
  return GeTensor();
}

const Tensor TensorAdapter::AsTensor(const GeTensor &ge_tensor) {
  Tensor tensor;
  if (tensor.impl != nullptr) {
    tensor.impl->ge_tensor = ge_tensor;
  }
  return tensor;
}

Tensor TensorAdapter::AsTensor(GeTensor &ge_tensor) {
  Tensor tensor;
  if (tensor.impl != nullptr) {
    tensor.impl->ge_tensor = ge_tensor;
  }
  return tensor;
}
}  // namespace ge
