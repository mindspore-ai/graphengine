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
#ifndef GE_COMMON_SINGLE_OP_PARSER_H
#define GE_COMMON_SINGLE_OP_PARSER_H

#include <vector>
#include <string>

#include <nlohmann/json.hpp>
#include "external/ge/ge_api_types.h"
#include "external/ge/ge_api_error_codes.h"
#include "external/graph/types.h"
#include "graph/ge_attr_value.h"
#include "graph/op_desc.h"
#include "common/fp16_t.h"

namespace ge {
struct SingleOpTensorDesc {
  std::string name;
  std::vector<int64_t> dims;
  std::vector<int64_t> ori_dims;
  std::vector<std::vector<int64_t>> dim_ranges;
  ge::Format format = ge::FORMAT_RESERVED;
  ge::Format ori_format = ge::FORMAT_RESERVED;
  ge::DataType type = ge::DT_UNDEFINED;
  std::string dynamic_input_name;
  bool is_const = false;
  std::shared_ptr<uint8_t> const_value;
  std::uint64_t const_value_size;
  bool is_valid = true;
};

struct SingleOpAttr {
  std::string name;
  std::string type;
  ge::GeAttrValue value;
};

struct SingleOpDesc {
  std::string op;
  std::string name;
  std::vector<SingleOpTensorDesc> input_desc;
  std::vector<SingleOpTensorDesc> output_desc;
  std::vector<SingleOpAttr> attrs;
  int32_t compile_flag = 0;
};

struct SingleOpBuildParam {
  ge::OpDescPtr op_desc;
  std::vector<ge::GeTensor> inputs;
  std::vector<ge::GeTensor> outputs;
  std::string file_name;
  int32_t compile_flag = 0;
};

void TransConstValue(const std::string &type_str, const nlohmann::json &j, SingleOpTensorDesc &desc);

void from_json(const nlohmann::json &j, fp16_t &fp16);

void from_json(const nlohmann::json &j, SingleOpTensorDesc &desc);

void from_json(const nlohmann::json &j, SingleOpAttr &attr);

void from_json(const nlohmann::json &j, SingleOpDesc &desc);

class SingleOpParser {
 public:
  static Status ParseSingleOpList(const std::string &file, std::vector<SingleOpBuildParam> &op_list);

 private:
  static Status ReadJsonFile(const std::string &file, nlohmann::json &json_obj);
  static bool Validate(const SingleOpDesc &op_desc);
  static std::unique_ptr<OpDesc> CreateOpDesc(const std::string &name, const std::string &op_type);
  static Status ConvertToBuildParam(int32_t index, const SingleOpDesc &single_op_desc, SingleOpBuildParam &build_param);
  static Status UpdateDynamicTensorName(std::vector<SingleOpTensorDesc> &desc);
  static Status VerifyOpInputOutputSizeByIr(const OpDesc &current_op_desc);
  static Status SetShapeRange(const std::string &op_name,
                              const SingleOpTensorDesc &tensor_desc,
                              GeTensorDesc &ge_tensor_desc);
};
}  // namespace ge

#endif  // GE_COMMON_SINGLE_OP_PARSER_H
