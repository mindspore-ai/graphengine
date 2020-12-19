/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "transformer_utils.h"

#include "external/ge/ge_api_types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils.h"

namespace ge {
bool NodeShapeTransUtils::CatchFormatAndShape() {
  auto inputs = op_desc_->MutableAllInputName();
  auto outputs = op_desc_->MutableAllOutputName();

  for (auto &ele : inputs) {
    auto tensor_desc_input = op_desc_->MutableInputDesc(ele.first);
    if (tensor_desc_input == nullptr) {
      continue;
    }
    auto format = tensor_desc_input->GetFormat();
    auto ori_format = tensor_desc_input->GetOriginFormat();
    if (format == ori_format) {
      GELOGD("Node is %s, input tensor name is %s. ori format: %s, format: %s is same! No need to catch format&shape!",
             op_desc_->GetName().c_str(), ele.first.c_str(), TypeUtils::FormatToSerialString(ori_format).c_str(),
             TypeUtils::FormatToSerialString(format).c_str());
      continue;
    }
    map_format_in_.insert(std::pair<std::string, Format>(ele.first, format));
    map_ori_format_in_.insert(std::pair<std::string, Format>(ele.first, ori_format));
    map_dtype_in_.insert(std::pair<std::string, DataType>(ele.first, tensor_desc_input->GetDataType()));
    tensor_desc_input->SetFormat(ori_format);
    tensor_desc_input->SetShape(tensor_desc_input->GetOriginShape());
  }

  for (auto &ele : outputs) {
    auto tensor_desc_output = op_desc_->MutableOutputDesc(ele.first);
    if (tensor_desc_output == nullptr) {
      continue;
    }
    auto format = tensor_desc_output->GetFormat();
    auto ori_format = tensor_desc_output->GetOriginFormat();
    if (format == ori_format) {
      GELOGD("Node is %s, output tensor name is %s. ori format: %s, format: %s is same! No need to catch format&shape!",
             op_desc_->GetName().c_str(), ele.first.c_str(), TypeUtils::FormatToSerialString(ori_format).c_str(),
             TypeUtils::FormatToSerialString(format).c_str());
      continue;
    }
    map_format_out_.insert(std::pair<std::string, Format>(ele.first, format));
    map_ori_format_out_.insert(std::pair<std::string, Format>(ele.first, ori_format));
    map_dtype_out_.insert(std::pair<std::string, DataType>(ele.first, tensor_desc_output->GetDataType()));

    if (format == ori_format) {
      continue;
    }
    tensor_desc_output->SetFormat(ori_format);
  }

  return true;
}

bool NodeShapeTransUtils::UpdateFormatAndShape() {
  auto inputs = op_desc_->MutableAllInputName();
  auto outputs = op_desc_->MutableAllOutputName();

  for (auto &ele : inputs) {
    auto tensor_desc_input = op_desc_->MutableInputDesc(ele.first);
    if (tensor_desc_input == nullptr) {
      continue;
    }
    // if can not find saved info, it says format and origin format is same when catched
    if (map_format_in_.find(ele.first) == map_format_in_.end()) {
      GELOGD("Node is [%s], input tensor name [%s] is not been catched.Skip update action for it!",
             op_desc_->GetName().c_str(), ele.first.c_str());
      tensor_desc_input->SetOriginFormat(tensor_desc_input->GetFormat());
      tensor_desc_input->SetOriginShape(tensor_desc_input->GetShape());
      continue;
    }
    auto ori_format = tensor_desc_input->GetFormat();
    auto ori_shape = tensor_desc_input->GetShape();
    auto curr_format = map_format_in_[ele.first];
    if (ori_format == curr_format) {
      continue;
    }
    std::unique_ptr<common::transformer::ShapeTransferAccordingToFormat> shape_transfer(new(std::nothrow)
      common::transformer::ShapeTransferAccordingToFormat());
    if (shape_transfer == nullptr) {
      GELOGE(GRAPH_FAILED, "Memory alloc failed");
      return false;
    }
    std::vector<int64_t> ori_shape_dims = ori_shape.GetDims();
    std::vector<int64_t> out_dims;
    ge::DataType dtype =  map_dtype_in_[ele.first];
    common::transformer::ShapeAndFormat shape_and_format_info {ori_shape_dims, out_dims, ori_format, curr_format, dtype,
                                                               common::transformer::EN_IMPL_CUSTOM_TBE};
    shape_transfer->GetShapeAccordingToFormat(shape_and_format_info);
    tensor_desc_input->SetFormat(curr_format);
    tensor_desc_input->SetShape(GeShape(out_dims));
  }

  for (auto &ele : outputs) {
    auto tensor_desc_output = op_desc_->MutableOutputDesc(ele.first);
    if (tensor_desc_output == nullptr) {
      continue;
    }
    // if can not find saved info, it says format and origin format is same when catched
    if (map_ori_format_out_.find(ele.first) == map_ori_format_out_.end()) {
      GELOGD("Node is [%s], input tensor name [%s] is not been catched.Skip update action for it!",
             op_desc_->GetName().c_str(), ele.first.c_str());
      tensor_desc_output->SetOriginFormat(tensor_desc_output->GetFormat());
      tensor_desc_output->SetOriginShape(tensor_desc_output->GetShape());
      continue;
    }
    auto ori_shape = tensor_desc_output->GetShape();
    auto curr_format = tensor_desc_output->GetFormat();
    if (curr_format != map_ori_format_out_[ele.first]) {
      GELOGE(GRAPH_FAILED, "Node is %s, out tensor name is %s. format: %s, recorded origin format: %s is not same",
             op_desc_->GetName().c_str(), ele.first.c_str(), TypeUtils::FormatToSerialString(curr_format).c_str(),
             TypeUtils::FormatToSerialString(map_ori_format_out_[ele.first]).c_str());
      return GRAPH_FAILED;
    }
    tensor_desc_output->SetOriginShape(ori_shape);
    auto saved_format = map_format_out_[ele.first];
    if (curr_format == saved_format) {
      GELOGD("Nodeis %s, out tensor name is %s. ori format: %s, recorded format: %s is same! No need to transfer",
             op_desc_->GetName().c_str(), ele.first.c_str(), TypeUtils::FormatToSerialString(curr_format).c_str(),
             TypeUtils::FormatToSerialString(saved_format).c_str());
      continue;
    }
    tensor_desc_output->SetFormat(saved_format);
    std::unique_ptr<common::transformer::ShapeTransferAccordingToFormat> shape_transfer(new(std::nothrow)
      common::transformer::ShapeTransferAccordingToFormat());
    if (shape_transfer == nullptr) {
      GELOGE(GRAPH_FAILED, "Memory alloc failed");
      return false;
    }
    std::vector<int64_t> ori_shape_dims = ori_shape.GetDims();
    std::vector<int64_t> out_dims;
    ge::DataType dtype =  tensor_desc_output->GetDataType();
    common::transformer::ShapeAndFormat shape_and_format_info {ori_shape_dims, out_dims, curr_format, saved_format,
                                                               dtype, common::transformer::EN_IMPL_CUSTOM_TBE};
    shape_transfer->GetShapeAccordingToFormat(shape_and_format_info);
    tensor_desc_output->SetShape(GeShape(out_dims));
    GELOGD("Node is %s, out tensor name is %s. Update format and shape success，ori format: %s, format: %s",
        op_desc_->GetName().c_str(), ele.first.c_str(), TypeUtils::FormatToSerialString(curr_format).c_str(),
        TypeUtils::FormatToSerialString(saved_format).c_str());
  }
  GELOGD("Node is %s. Update format and shape success", op_desc_->GetName().c_str());
  return true;
}
} // namespace ge