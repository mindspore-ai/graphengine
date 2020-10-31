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

#ifndef INC_EXTERNAL_GRAPH_OPERATOR_FACTORY_H_
#define INC_EXTERNAL_GRAPH_OPERATOR_FACTORY_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "./operator.h"
#include "./ge_error_codes.h"

namespace ge {
using OpCreator = std::function<Operator(const std::string &)>;
using InferShapeFunc = std::function<graphStatus(Operator &)>;
using InferFormatFunc = std::function<graphStatus(Operator &)>;
using VerifyFunc = std::function<graphStatus(Operator &)>;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OperatorFactory {
 public:
  static Operator CreateOperator(const std::string &operator_name, const std::string &operator_type);

  static graphStatus GetOpsTypeList(std::vector<std::string> &all_ops);

  static bool IsExistOp(const string &operator_type);
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OperatorCreatorRegister {
 public:
  OperatorCreatorRegister(const string &operator_type, OpCreator const &op_creator);
  ~OperatorCreatorRegister() = default;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InferShapeFuncRegister {
 public:
  InferShapeFuncRegister(const std::string &operator_type, const InferShapeFunc &infer_shape_func);
  ~InferShapeFuncRegister() = default;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InferFormatFuncRegister {
 public:
  InferFormatFuncRegister(const std::string &operator_type, const InferFormatFunc &infer_format_func);
  ~InferFormatFuncRegister() = default;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY VerifyFuncRegister {
 public:
  VerifyFuncRegister(const std::string &operator_type, const VerifyFunc &verify_func);
  ~VerifyFuncRegister() = default;
};
}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_OPERATOR_FACTORY_H_
