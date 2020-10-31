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

#ifndef INC_GRAPH_OPERATOR_FACTORY_IMPL_H_
#define INC_GRAPH_OPERATOR_FACTORY_IMPL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "graph/operator_factory.h"

namespace ge {
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OperatorFactoryImpl {
 public:
  static Operator CreateOperator(const std::string &operator_name, const std::string &operator_type);

  static graphStatus GetOpsTypeList(std::vector<std::string> &all_ops);

  static bool IsExistOp(const string &operator_type);

  static InferShapeFunc GetInferShapeFunc(const std::string &operator_type);

  static InferFormatFunc GetInferFormatFunc(const std::string &operator_type);

  static VerifyFunc GetVerifyFunc(const std::string &operator_type);

  static graphStatus RegisterOperatorCreator(const std::string &operator_type, OpCreator const &op_creator);

  static graphStatus RegisterInferShapeFunc(const std::string &operator_type, InferShapeFunc const infer_shape_func);

  static graphStatus RegisterInferFormatFunc(const std::string &operator_type, InferFormatFunc const infer_format_func);

  static graphStatus RegisterVerifyFunc(const std::string &operator_type, VerifyFunc const verify_func);

  static shared_ptr<std::map<string, OpCreator>> operator_creators_;
  static shared_ptr<std::map<string, InferShapeFunc>> operator_infershape_funcs_;
  static shared_ptr<std::map<string, InferFormatFunc>> operator_inferformat_funcs_;
  static shared_ptr<std::map<string, VerifyFunc>> operator_verify_funcs_;
};
}  // namespace ge

#endif  // INC_GRAPH_OPERATOR_FACTORY_IMPL_H_
