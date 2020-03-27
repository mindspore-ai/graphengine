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

namespace ge {
ShapeAndType::ShapeAndType(const Shape &shape, DataType data_type) : shape_(shape), data_type_(data_type) {}

void ShapeAndType::SetShape(const Shape &shape) { shape_ = shape; }

void ShapeAndType::SetType(DataType data_type) { data_type_ = data_type; }

const Shape &ShapeAndType::GetShape() const { return shape_; }

DataType ShapeAndType::GetDataType() const { return data_type_; }

void InferenceContext::SetInputHandleShapesAndTypes(std::vector<std::vector<ShapeAndType>> &&shapes_and_types) {
  input_handle_shapes_and_types_.swap(shapes_and_types);
}

const std::vector<std::vector<ShapeAndType>> &InferenceContext::GetInputHandleShapesAndTypes() const {
  return input_handle_shapes_and_types_;
}

const std::vector<std::vector<ShapeAndType>> &InferenceContext::GetOutputHandleShapesAndTypes() const {
  return output_handle_shapes_and_types_;
}

void InferenceContext::SetOutputHandleShapesAndTypes(const std::vector<std::vector<ShapeAndType>> &shapes_and_types) {
  output_handle_shapes_and_types_ = shapes_and_types;
}

void InferenceContext::SetOutputHandleShapesAndTypes(std::vector<std::vector<ShapeAndType>> &&shapes_and_types) {
  output_handle_shapes_and_types_.swap(shapes_and_types);
}

void InferenceContext::SetMarks(const std::vector<std::string> &marks) { marks_ = marks; }

const std::vector<std::string> &InferenceContext::GetMarks() const { return marks_; }
}  // namespace ge
