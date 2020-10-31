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

#include "graph/operator_factory_impl.h"
#include "debug/ge_log.h"

namespace ge {
Operator OperatorFactory::CreateOperator(const std::string &operator_name, const std::string &operator_type) {
  return OperatorFactoryImpl::CreateOperator(operator_name, operator_type);
}

graphStatus OperatorFactory::GetOpsTypeList(std::vector<std::string> &all_ops) {
  return OperatorFactoryImpl::GetOpsTypeList(all_ops);
}

bool OperatorFactory::IsExistOp(const string &operator_type) { return OperatorFactoryImpl::IsExistOp(operator_type); }

OperatorCreatorRegister::OperatorCreatorRegister(const string &operator_type, OpCreator const &op_creator) {
  (void)OperatorFactoryImpl::RegisterOperatorCreator(operator_type, op_creator);
}

InferShapeFuncRegister::InferShapeFuncRegister(const std::string &operator_type,
                                               const InferShapeFunc &infer_shape_func) {
  (void)OperatorFactoryImpl::RegisterInferShapeFunc(operator_type, infer_shape_func);
}

InferFormatFuncRegister::InferFormatFuncRegister(const std::string &operator_type,
                                                 const InferFormatFunc &infer_format_func) {
  (void)OperatorFactoryImpl::RegisterInferFormatFunc(operator_type, infer_format_func);
}

VerifyFuncRegister::VerifyFuncRegister(const std::string &operator_type, const VerifyFunc &verify_func) {
  (void)OperatorFactoryImpl::RegisterVerifyFunc(operator_type, verify_func);
}
}  // namespace ge
