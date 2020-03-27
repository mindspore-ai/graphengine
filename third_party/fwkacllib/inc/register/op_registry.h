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

#ifndef INC_REGISTER_OP_REGISTRY_H_
#define INC_REGISTER_OP_REGISTRY_H_

#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "register/register.h"

namespace domi {
class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY OpRegistry {
 public:
  static OpRegistry *Instance();

  std::vector<OpRegistrationData> registrationDatas;

  bool Register(const OpRegistrationData &reg_data);

  domi::ImplyType GetImplyType(const std::string &op_type);

  void GetOpTypeByImplyType(std::vector<std::string> &vec_op_type, const domi::ImplyType &imply_type);

  void GetFormats(const std::string &op_type, std::vector<domi::tagDomiTensorFormat> &input_format_vector,
                  std::vector<domi::tagDomiTensorFormat> &output_format_vector);

  void GetWeightFormats(const std::string &op_type, std::vector<domi::tagDomiTensorFormat> &format_vector);

  domi::ParseParamFunc GetParseParamFunc(const std::string &op_type);

  domi::InferShapeFunc GetInferShapeFunc(const std::string &op_type);

  domi::InferShapeFuncV2 GetInferShapeFuncV2(const std::string &op_type);

  domi::GetWorkspaceSizeFunc GetGetWorkspaceSizeFunc(const std::string &op_type);

  domi::UpdateOpDescFunc GetUpdateOpDescFunc(const std::string &op_type);

  domi::BuildTeBinFunc GetBuildTeBinFunc(const std::string &op_type);

  domi::ImplyType GetImplyTypeByOriOpType(const std::string &ori_optype);

  void GetSupportedInputFormats(const std::string &opType, std::vector<std::vector<ge::Format>> &suportedInputFormats);

  void GetSupportedOutputFormats(const std::string &opType,
                                 std::vector<std::vector<ge::Format>> &supportedOutputFormats);

  void GetSupportedInputTypes(const std::string &opType,
                              std::vector<std::vector<ge::DataType>> &suportedInputDataTypes);
  void GetSupportedInputTypesByOriginOpType(const std::string &opType,
                                            std::vector<std::vector<ge::DataType>> &suportedInputDataTypes);

  void GetSupportedOutputTypes(const std::string &opType,
                               std::vector<std::vector<ge::DataType>> &supportedOutputDataTypes);
  void GetSupportedOutputTypesByOriginOpType(const std::string &opType,
                                             std::vector<std::vector<ge::DataType>> &supportedOutputDataTypes);

  void GetLimitedInputTensorDescs(const std::string &opType,
                                  std::vector<std::vector<ge::TensorDescInfo>> &inputLimitedTensorDescs);
  void GetLimitedInputTensorDescsByOriginOpType(const std::string &opType,
                                                std::vector<std::vector<ge::TensorDescInfo>> &inputLimitedTensorDescs);

  void GetLimitedOutputTensorDescs(const std::string &opType,
                                   std::vector<std::vector<ge::TensorDescInfo>> &outputLimitedTensorDescs);
  void GetLimitedOutputTensorDescsByOriginOpType(
      const std::string &opType, std::vector<std::vector<ge::TensorDescInfo>> &outputLimitedTensorDescs);

  const std::vector<MoveInputToAttrStu> &GetConstInputToAttr(const std::string &ori_optype) const;

 private:
  std::unordered_map<std::string, std::set<std::string>> op_ori_optype_map_;
  std::unordered_map<std::string, domi::ImplyType> op_run_mode_map_;
  std::unordered_map<std::string, std::vector<domi::tagDomiTensorFormat>> op_input_formats_map_;
  std::unordered_map<std::string, std::vector<domi::tagDomiTensorFormat>> op_output_formats_map_;
  std::unordered_map<std::string, std::vector<domi::tagDomiTensorFormat>> op_weight_formats_map_;
  std::unordered_map<std::string, ParseParamFunc> opParseParamsFnMap_;
  std::unordered_map<std::string, InferShapeFunc> opInferShapeFnMap_;
  std::unordered_map<std::string, InferShapeFuncV2> opInferShapeFnMapV2_;
  std::unordered_map<std::string, GetWorkspaceSizeFunc> opGetWorkspaceSizeFnMap_;
  std::unordered_map<std::string, UpdateOpDescFunc> opUpdateOpDescFnMap_;
  std::unordered_map<std::string, BuildTeBinFunc> opBuildTeBinFnMap_;
  std::unordered_map<std::string, std::vector<MoveInputToAttrStu>> opConstInputToAttrMap_;

  std::unordered_map<std::string, std::vector<std::vector<ge::Format>>> opInputSupportedFormats_;
  std::unordered_map<std::string, std::vector<std::vector<ge::Format>>> opOutputSupportedFormats_;
  std::unordered_map<std::string, std::vector<std::vector<ge::DataType>>> opInputSupportedDataTypes_;
  std::unordered_map<std::string, std::vector<std::vector<ge::DataType>>> opOutputSupportedDataTypes_;
  std::unordered_map<std::string, std::vector<std::vector<ge::TensorDescInfo>>> opInputLimitedTensorDescs_;
  std::unordered_map<std::string, std::vector<std::vector<ge::TensorDescInfo>>> opOutputLimitedTensorDescs_;

  std::unordered_map<std::string, std::string> originOpType2OmOpType_;
};
}  // namespace domi

#endif  // INC_REGISTER_OP_REGISTRY_H_
