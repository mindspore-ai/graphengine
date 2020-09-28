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

#include "external/graph/inference_context.h"
#include "debug/ge_util.h"

namespace ge {
class ShapeAndTypeImpl {
 public:
  ShapeAndTypeImpl() = default;
  ~ShapeAndTypeImpl() = default;

  ShapeAndTypeImpl(const Shape &shape, DataType data_type) : shape_(shape), data_type_(data_type) {}

  Shape shape_;
  DataType data_type_ = DT_UNDEFINED;
};

class InferenceContextImpl {
 public:
  InferenceContextImpl() = default;
  ~InferenceContextImpl() = default;

  // For deliver to op in pair, help to support dynamic shape
  std::vector<std::string> marks_;
  std::vector<std::vector<ShapeAndType>> input_handle_shapes_and_types_;
  std::vector<std::vector<ShapeAndType>> output_handle_shapes_and_types_;
};

ShapeAndType::ShapeAndType() { shape_and_type_impl_ = ComGraphMakeShared<ShapeAndTypeImpl>(); }

ShapeAndType::ShapeAndType(const Shape &shape, DataType data_type) {
  shape_and_type_impl_ = ComGraphMakeShared<ShapeAndTypeImpl>(shape, data_type);
}

void ShapeAndType::SetShape(const Shape &shape) {
  if (shape_and_type_impl_ != nullptr) {
    shape_and_type_impl_->shape_ = shape;
  }
}

void ShapeAndType::SetType(DataType data_type) {
  if (shape_and_type_impl_ != nullptr) {
    shape_and_type_impl_->data_type_ = data_type;
  }
}

Shape ShapeAndType::GetShape() const {
  if (shape_and_type_impl_ != nullptr) {
    return shape_and_type_impl_->shape_;
  }
  return Shape();
}

DataType ShapeAndType::GetDataType() const {
  if (shape_and_type_impl_ != nullptr) {
    return shape_and_type_impl_->data_type_;
  }
  return DT_UNDEFINED;
}

InferenceContext::InferenceContext(std::unique_ptr<InferenceContextImpl> &impl) {
  inference_context_impl_ = std::move(impl);
}

std::unique_ptr<InferenceContext> InferenceContext::Create() {
  std::unique_ptr<InferenceContextImpl> impl =
    std::unique_ptr<InferenceContextImpl>(new (std::nothrow) InferenceContextImpl());
  if (impl == nullptr) {
    return nullptr;
  }

  return std::unique_ptr<InferenceContext>(new (std::nothrow) InferenceContext(impl));
}

void InferenceContext::SetInputHandleShapesAndTypes(std::vector<std::vector<ShapeAndType>> &&shapes_and_types) {
  inference_context_impl_->input_handle_shapes_and_types_.swap(shapes_and_types);
}

const std::vector<std::vector<ShapeAndType>> &InferenceContext::GetInputHandleShapesAndTypes() const {
  return inference_context_impl_->input_handle_shapes_and_types_;
}

const std::vector<std::vector<ShapeAndType>> &InferenceContext::GetOutputHandleShapesAndTypes() const {
  return inference_context_impl_->output_handle_shapes_and_types_;
}

void InferenceContext::SetOutputHandleShapesAndTypes(const std::vector<std::vector<ShapeAndType>> &shapes_and_types) {
  inference_context_impl_->output_handle_shapes_and_types_ = shapes_and_types;
}

void InferenceContext::SetOutputHandleShapesAndTypes(std::vector<std::vector<ShapeAndType>> &&shapes_and_types) {
  inference_context_impl_->output_handle_shapes_and_types_.swap(shapes_and_types);
}

void InferenceContext::SetMarks(const std::vector<std::string> &marks) { inference_context_impl_->marks_ = marks; }

const std::vector<std::string> &InferenceContext::GetMarks() const { return inference_context_impl_->marks_; }
}  // namespace ge
