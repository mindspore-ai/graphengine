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

#ifndef AIR_CXX_INC_FRAMEWORK_RUNTIME_MODEL_DESC_H_
#define AIR_CXX_INC_FRAMEWORK_RUNTIME_MODEL_DESC_H_
#include "common/ge_types.h"
#include "common/ge_visibility.h"

#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_space_registry.h"

namespace gert {
class VISIBILITY_EXPORT ShapeRange {
 public:
  ShapeRange() = default;
  ShapeRange(const Shape &min_shape, const Shape &max_shape);
  bool operator==(const ShapeRange &other) const;

  const Shape &GetMin() const;
  const Shape &GetMax() const;
  Shape &MutableMin();
  Shape &MutableMax();

 private:
  Shape min_;
  Shape max_;
};

class VISIBILITY_EXPORT ModelIoDesc {
 public:
  const ge::char_t *GetName() const;
  int32_t GetDataType() const;
  ge::Format GetStorageFormat() const;
  ge::Format GetOriginFormat() const;
  int64_t GetSize() const;
  const Shape &GetStorageShape() const;
  const Shape &GetOriginShape() const;
  const ShapeRange &GetOriginShapeRange() const;
  const ShapeRange &GetStorageShapeRange() const;
  std::vector<std::pair<int64_t, int64_t>> GetOriginShapeRangeVector() const;
  std::vector<std::pair<int64_t, int64_t>> GetStorageShapeRangeVector() const;
  bool IsShapeInRange(const Shape &shape) const;
  bool IsOriginShapeInRange(const Shape &shape) const;

  void SetName(const ge::char_t *name);
  void SetDataType(int32_t data_type);
  void SetStorageFormat(ge::Format format);
  void SetOriginFormat(ge::Format format);
  Shape &MutableStorageShape();
  Shape &MutableOriginShape();
  ShapeRange &MutableOriginShapeRange();
  ShapeRange &MutableStorageShapeRange();

  const Shape &GetAippShape() const;
  Shape &MutableAippShape();

 private:
  const ge::char_t *name_;
  int32_t data_type_;
  StorageFormat format_;
  StorageShape shape_;
  ShapeRange storage_shape_range_;
  ShapeRange origin_shape_range_;
  // use for aipp
  Shape aipp_shape_;
};

class VISIBILITY_EXPORT ModelDesc {
 public:
  static size_t CalcSize(size_t input_num, size_t output_num);
  const ModelIoDesc *GetInputDesc(size_t index) const;
  const ModelIoDesc *GetAllInputsDesc(size_t &input_num) const;

  const ModelIoDesc *GetOutputDesc(size_t index) const;
  const ModelIoDesc *GetAllOutputsDesc(size_t &output_num) const;

  size_t GetInputNum() const;
  size_t GetOutputNum() const;

  ModelIoDesc *MutableInputDesc(size_t index);
  ModelIoDesc *MutableOutputDesc(size_t index);
  ModelIoDesc *AllMutableIoDesc(size_t &input_num, size_t &output_num);
  void SetInputNum(size_t input_num);
  void SetOutputNum(size_t output_num);
  void SetSpaceRegistry(gert::OpImplSpaceRegistry *space_registry);
  gert::OpImplSpaceRegistry *GetSpaceRegistry() const;
  void SetFileConstantWeightDir(const ge::char_t *file_constant_weight_dir);
  const ge::char_t *GetFileConstantWeightDir() const;

  ge::graphStatus GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type) const;
  ge::graphStatus GetUserDesignateShapeOrder(std::vector<std::string> &user_designate_shape_order) const;
  ge::graphStatus GetModelAttrs(std::vector<std::string> &attrs) const;

 private:
  gert::OpImplSpaceRegistry *space_registry_;
  const ge::char_t *file_constant_weight_dir_;
  size_t input_num_;
  size_t output_num_;
  ContinuousVector model_io_descs_;
  // do not add any variable here
};
static_assert(std::is_trivial<ModelDesc>::value, "The class ModelDesc must be a POD");
} // namespace gert

#endif  // AIR_CXX_INC_FRAMEWORK_RUNTIME_MODEL_DESC_H_