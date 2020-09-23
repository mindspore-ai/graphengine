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
#include "framework/common/debug/ge_log.h"

namespace ge {
shared_ptr<std::map<string, OpCreator>> OperatorFactoryImpl::operator_creators_;
shared_ptr<std::map<string, InferShapeFunc>> OperatorFactoryImpl::operator_infershape_funcs_;
shared_ptr<std::map<string, InferFormatFunc>> OperatorFactoryImpl::operator_inferformat_funcs_;
shared_ptr<std::map<string, VerifyFunc>> OperatorFactoryImpl::operator_verify_funcs_;

Operator OperatorFactoryImpl::CreateOperator(const std::string &operator_name, const std::string &operator_type) {
  if (operator_creators_ == nullptr) {
    return Operator();
  }
  auto it = operator_creators_->find(operator_type);
  if (it == operator_creators_->end()) {
    GELOGW("no OpProto of [%s] registered", operator_type.c_str());
    return Operator();
  }
  return it->second(operator_name);
}

graphStatus OperatorFactoryImpl::GetOpsTypeList(std::vector<std::string> &all_ops) {
  all_ops.clear();
  if (operator_creators_ != nullptr) {
    for (auto it = operator_creators_->begin(); it != operator_creators_->end(); ++it) {
      all_ops.emplace_back(it->first);
    }
  } else {
    GELOGE(GRAPH_FAILED, "no operator creators found");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

bool OperatorFactoryImpl::IsExistOp(const string &operator_type) {
  if (operator_creators_ == nullptr) {
    return false;
  }
  auto it = operator_creators_->find(operator_type);
  if (it == operator_creators_->end()) {
    return false;
  }
  return true;
}

InferShapeFunc OperatorFactoryImpl::GetInferShapeFunc(const std::string &operator_type) {
  if (operator_infershape_funcs_ == nullptr) {
    return nullptr;
  }
  auto it = operator_infershape_funcs_->find(operator_type);
  if (it == operator_infershape_funcs_->end()) {
    return nullptr;
  }
  return it->second;
}

InferFormatFunc OperatorFactoryImpl::GetInferFormatFunc(const std::string &operator_type) {
  if (operator_inferformat_funcs_ == nullptr) {
    GELOGI("operator_inferformat_funcs_ is null");
    return nullptr;
  }
  auto it = operator_inferformat_funcs_->find(operator_type);
  if (it == operator_inferformat_funcs_->end()) {
    return nullptr;
  }
  return it->second;
}

VerifyFunc OperatorFactoryImpl::GetVerifyFunc(const std::string &operator_type) {
  if (operator_verify_funcs_ == nullptr) {
    return nullptr;
  }
  auto it = operator_verify_funcs_->find(operator_type);
  if (it == operator_verify_funcs_->end()) {
    return nullptr;
  }
  return it->second;
}

graphStatus OperatorFactoryImpl::RegisterOperatorCreator(const string &operator_type, OpCreator const &op_creator) {
  if (operator_creators_ == nullptr) {
    operator_creators_.reset(new (std::nothrow) std::map<string, OpCreator>());
  }
  auto it = operator_creators_->find(operator_type);
  if (it != operator_creators_->end()) {
    return GRAPH_FAILED;
  }
  (void)operator_creators_->emplace(operator_type, op_creator);
  return GRAPH_SUCCESS;
}

graphStatus OperatorFactoryImpl::RegisterInferShapeFunc(const std::string &operator_type,
                                                        InferShapeFunc const infer_shape_func) {
  if (operator_infershape_funcs_ == nullptr) {
    GELOGI("operator_infershape_funcs_ init");
    operator_infershape_funcs_.reset(new (std::nothrow) std::map<string, InferShapeFunc>());
  }
  auto it = operator_infershape_funcs_->find(operator_type);
  if (it != operator_infershape_funcs_->end()) {
    return GRAPH_FAILED;
  }
  (void)operator_infershape_funcs_->emplace(operator_type, infer_shape_func);
  return GRAPH_SUCCESS;
}

graphStatus OperatorFactoryImpl::RegisterInferFormatFunc(const std::string &operator_type,
                                                         InferFormatFunc const infer_format_func) {
  if (operator_inferformat_funcs_ == nullptr) {
    GELOGI("operator_inferformat_funcs_ init");
    operator_inferformat_funcs_.reset(new (std::nothrow) std::map<string, InferFormatFunc>());
  }
  auto it = operator_inferformat_funcs_->find(operator_type);
  if (it != operator_inferformat_funcs_->end()) {
    return GRAPH_FAILED;
  }
  (void)operator_inferformat_funcs_->emplace(operator_type, infer_format_func);
  return GRAPH_SUCCESS;
}

graphStatus OperatorFactoryImpl::RegisterVerifyFunc(const std::string &operator_type, VerifyFunc const verify_func) {
  if (operator_verify_funcs_ == nullptr) {
    GELOGI("operator_verify_funcs_ init");
    operator_verify_funcs_.reset(new (std::nothrow) std::map<string, VerifyFunc>());
  }
  auto it = operator_verify_funcs_->find(operator_type);
  if (it != operator_verify_funcs_->end()) {
    return GRAPH_FAILED;
  }
  (void)operator_verify_funcs_->emplace(operator_type, verify_func);
  return GRAPH_SUCCESS;
}
}  // namespace ge
