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

#include "multi_batch_options.h"

#include "framework/common/debug/ge_log.h"
#include "framework/omg/omg_inner_types.h"
#include "framework/common/util.h"
#include "framework/common/string_util.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph/ge_context.h"
#include "graph/common/local_context.h"

namespace ge {
namespace multibatch {
constexpr int kDecimal = 10;
constexpr uint8_t kMaxShapesCount = 100;
constexpr uint8_t kMinShapesCount = 2;

void ParseDynamicSize(string dynamic_size, vector<vector<int64_t>> &shapes) {
  std::vector<std::string> shape_strs = ge::StringUtils::Split(dynamic_size, ';');
  for (const auto &shape_str : shape_strs) {
    if (shape_str.empty()) {
      continue;
    }
    std::vector<int64_t> shape;
    std::vector<std::string> dims = ge::StringUtils::Split(shape_str, ',');
    for (const auto &dim : dims) {
      if (dim.empty()) {
        continue;
      }
      shape.emplace_back(std::strtol(dim.c_str(), nullptr, kDecimal));
    }
    if (!shape.empty()) {
      shapes.emplace_back(shape);
    }
  }
}

///
/// @ingroup ge
/// @brief Init Dynamic Param from Options.
/// @param [out] std::vector<std::vector<int64_t>> &shapes: Result for Params.
/// @return true: Configed for Multi batch / false: Not configed for Multi batch.
///
bool InitDynamicParams(vector<vector<int64_t>> &shapes) {
  if (!GetLocalOmgContext().dynamic_batch_size.empty()) {
    GELOGD("Found dynamic batch option, value %s", GetLocalOmgContext().dynamic_batch_size.c_str());
    std::vector<std::string> dims = ge::StringUtils::Split(GetLocalOmgContext().dynamic_batch_size, ',');
    for (const auto &dim : dims) {
      if (dim.empty()) {
        continue;
      }
      shapes.emplace_back(std::vector<int64_t>({std::strtol(dim.c_str(), nullptr, kDecimal)}));
      GELOGI("Found dynamic batch, shape %s", formats::JoinToString(*shapes.rbegin()).c_str());
    }
  }

  if (!GetLocalOmgContext().dynamic_image_size.empty()) {
    GELOGD("Found dynamic image size option, value %s", GetLocalOmgContext().dynamic_image_size.c_str());
    ParseDynamicSize(GetLocalOmgContext().dynamic_image_size, shapes);

    for (const auto &shape : shapes) {
      GELOGI("Found dynamic image size, shape %s", formats::JoinToString(shape).c_str());
    }
  }

  if (!GetLocalOmgContext().dynamic_dims.empty()) {
    GELOGD("Found dynamic dims option, value %s", GetLocalOmgContext().dynamic_dims.c_str());
    ParseDynamicSize(GetLocalOmgContext().dynamic_dims, shapes);

    for (const auto &shape : shapes) {
      GELOGI("Found dynamic dims, shape %s", formats::JoinToString(shape).c_str());
    }
  }

  return !shapes.empty();
}

///
/// @ingroup ge
/// @brief parse each data's own dynamic dims.
/// @param [out] map<string, vector<vector<int64_t>>> &data_to_dynamic_info: key:data_name. value:dynamic dims.
/// @return true: Configed for Multi batch / false: Not configed for Multi batch.
///
Status ParserDataToDynmaicInfo(const vector<vector<int64_t>> &shapes,
                               vector<pair<string, vector<int64_t>>> &data_name_and_shape,
                               map<string, vector<vector<int64_t>>> &data_to_dynamic_info) {
  size_t cur_data_index = 0;
  for (size_t index = 0; index < data_name_and_shape.size(); ++index) {
    auto &cur_item = data_name_and_shape[index];
    auto &data_name = cur_item.first;
    auto &data_shape = cur_item.second;
    auto dynamic_dims_num =
      std::count_if(data_shape.begin(), data_shape.end(), [&data_shape](int64_t dim) { return dim < 0; });
    vector<vector<int64_t>> dynamic_info;
    for (auto &dynamic_gear_info : shapes) {
      vector<int64_t> one_gear;
      if (dynamic_gear_info.size() == static_cast<size_t>(dynamic_dims_num)) {
        one_gear = dynamic_gear_info;
      } else if (dynamic_gear_info.size() > static_cast<size_t>(dynamic_dims_num)) {
        auto tmp_index = cur_data_index;
        for (size_t i = 0; i < static_cast<size_t>(dynamic_dims_num); ++i) {
          if (tmp_index >= dynamic_gear_info.size()) {
            GELOGE(PARAM_INVALID, "Data: %s shape: %s make dynamic dims overflow", data_name.c_str(),
                   formats::JoinToString(data_shape).c_str());
            return FAILED;
          }
          one_gear.push_back(dynamic_gear_info[tmp_index++]);
        }
      } else {
        GELOGE(PARAM_INVALID, "Dynamic dims num of data: %s shape: %s can not be more than one gear dynamic info size",
               data_name.c_str(), formats::JoinToString(data_shape).c_str());
        return FAILED;
      }
      dynamic_info.push_back(one_gear);
    }
    cur_data_index += dynamic_dims_num;
    data_to_dynamic_info[data_name] = dynamic_info;
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Check Dynamic Param is invalid.
/// @param [in] const vector<vector<int64_t>> &shapes: Params for check.
/// @return SUCCESS: valid / PARAM_INVALID: invalid.
///
Status CheckDynamicParams(const vector<vector<int64_t>> &shapes) {
  if (shapes.size() < kMinShapesCount) {
    ErrorManager::GetInstance().ATCReportErrMessage(
      "E10035", {"shapesize", "minshapesize"}, {std::to_string(shapes.size()), std::to_string(kMinShapesCount - 1)});
    GELOGE(PARAM_INVALID,
           "Input parameter[--dynamic_batch_size, --dynamic_image_size or --dynamic_dims]'s "
           "value size [%zu] must be greater than [%zu].",
           shapes.size(), kMinShapesCount - 1);
    return PARAM_INVALID;
  }
  if (shapes.size() > kMaxShapesCount) {
    ErrorManager::GetInstance().ATCReportErrMessage(
      "E10036", {"shapesize", "maxshapesize"}, {std::to_string(shapes.size()), std::to_string(kMaxShapesCount + 1)});
    GELOGE(PARAM_INVALID,
           "Input parameter[--dynamic_batch_size, --dynamic_image_size or --dynamic_dims]'s "
           "value size [%zu] must be less than [%zu].",
           shapes.size(), kMaxShapesCount + 1);
    return PARAM_INVALID;
  }
  std::set<std::vector<int64_t>> shapes_set;
  size_t shape_size = shapes.at(0).size();
  for (auto &shape : shapes) {
    if (shape_size != shape.size()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10037", {"shapesize1", "shapesize2"},
                                                      {std::to_string(shape_size), std::to_string(shape.size())});
      GELOGE(PARAM_INVALID,
             "Input parameter[--dynamic_batch_size, --dynamic_image_size or --dynamic_dims]'s "
             "value size must be same, first group's size is %zu and another's is %zu.",
             shape_size, shape.size());
      return PARAM_INVALID;
    }
    for (auto dim : shape) {
      if (dim <= 0) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10038", {"dim"}, {std::to_string(dim)});
        GELOGE(PARAM_INVALID, "Invalid dim %ld, all dims must be greater than 0", dim);
        return PARAM_INVALID;
      }
    }
    shapes_set.insert(shape);
  }
  if (shapes_set.size() != shapes.size()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10039");
    GELOGE(PARAM_INVALID,
           "Input parameter[--dynamic_batch_size, --dynamic_image_size or --dynamic_dims] exist duplicate shapes.");
    return PARAM_INVALID;
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get GeShape from configed shape.
/// @param [in] const std::vector<int64_t> &batch_shape: Configed shape.
/// @param [out] GeShape &data_shape: GeShape for configed shape.
/// @return SUCCESS / PARAM_INVALID
///
Status CalcShape(const std::vector<int64_t> &batch_shape, GeShape &data_shape) {
  size_t batch_shape_index = 0;
  for (size_t i = 0; i < data_shape.GetDimNum(); ++i) {
    if (data_shape.GetDim(i) < 0) {
      if (batch_shape_index >= batch_shape.size()) {
        ErrorManager::GetInstance().ATCReportErrMessage(
          "E19012", {"function", "reason"},
          {"CalcShape", "the batch shape count " + std::to_string(batch_shape.size()) +
                          " does not match the data shape " + data_shape.ToString()});
        GELOGE(PARAM_INVALID,
               "Failed to calc tensor shape, the batch shape count %zu, does not match the data shape %s",
               batch_shape.size(), data_shape.ToString().c_str());
        return PARAM_INVALID;
      }
      data_shape.SetDim(i, batch_shape[batch_shape_index++]);
    }
  }
  if (batch_shape_index != batch_shape.size()) {
    ErrorManager::GetInstance().ATCReportErrMessage(
      "E19012", {"function", "reason"},
      {"CalcShape", "the batch shape count " + std::to_string(batch_shape.size()) + " does not match the data shape " +
                      data_shape.ToString()});
    GELOGE(PARAM_INVALID, "Failed to calc tensor shape, the batch shape count %zu, does not match the data shape %s",
           batch_shape.size(), data_shape.ToString().c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Set mbatch_dynamic_type on node.
/// @param [in] const OpDescPtr &op_desc: Node for set attribute.
/// @return 0: SUCCESS / others: INTERNAL_ERROR
///
Status StampDynamicType(const OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(op_desc);
  int32_t dynamic_type = static_cast<int32_t>(FIXED);
  if (!GetLocalOmgContext().dynamic_batch_size.empty()) {
    dynamic_type = static_cast<int32_t>(DYNAMIC_BATCH);
  }
  if (!GetLocalOmgContext().dynamic_image_size.empty()) {
    dynamic_type = static_cast<int32_t>(DYNAMIC_IMAGE);
  }
  if (!GetLocalOmgContext().dynamic_dims.empty()) {
    dynamic_type = static_cast<int32_t>(DYNAMIC_DIMS);
  }
  if (!AttrUtils::SetInt(op_desc, ATTR_DYNAMIC_TYPE, dynamic_type)) {
    GELOGE(INTERNAL_ERROR, "Failed to add dynamic type attr for node %s", op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}
}  // namespace multibatch
}  // namespace ge
