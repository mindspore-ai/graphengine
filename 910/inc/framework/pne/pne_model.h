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

#ifndef INC_FRAMEWORK_PNE_MODEL_H_
#define INC_FRAMEWORK_PNE_MODEL_H_

#include <map>
#include <string>
#include <vector>

#include "graph/compute_graph.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "framework/engine/dnnengine.h"
#include "external/ge/ge_ir_build.h"

namespace ge {
const std::string PNE_ID_NPU = "NPU";
const std::string PNE_ID_CPU = "HOST_CPU";
const std::string PNE_ID_UDF = "UDF";

struct ModelRelation;
struct ModelDeployResource;
class PneModel {
 public:
  PneModel() = default;
  explicit PneModel(const ComputeGraphPtr &root_graph) : root_graph_(root_graph){};
  virtual ~PneModel() = default;
  PneModel(const PneModel &other) = delete;
  PneModel &operator=(const PneModel &other) = delete;

 public:
  inline Status AddSubModel(const shared_ptr<PneModel> &submodel, std::string type = "") {
    if (submodel == nullptr) {
      GELOGE(INTERNAL_ERROR, "submodel is nullptr, type = %s", type.c_str());
      return INTERNAL_ERROR;
    }
    submodel->SetModelType(type);
    if (!submodels_.emplace(submodel->GetModelName(), submodel).second) {
      GELOGE(INTERNAL_ERROR, "submodel already exist, name = %s, type = %s", submodel->GetModelName().c_str(),
             type.c_str());
      return INTERNAL_ERROR;
    }
    return SUCCESS;
  }

  inline const std::shared_ptr<PneModel> GetSubmodel(const std::string &name) const {
    const auto &it = submodels_.find(name);
    if (it == submodels_.end()) {
      return nullptr;
    }
    return it->second;
  }

  inline const std::map<std::string, std::shared_ptr<PneModel>> &GetSubmodels() const {
    return submodels_;
  }

  inline void SetModelType(const std::string &type) {
    model_type_ = type;
  }

  inline const std::string &GetModelType() const {
    return model_type_;
  }

  inline void SetModelName(const std::string &model_name) {
    model_name_ = model_name;
  }

  inline const std::string &GetModelName() const {
    return model_name_;
  }

  inline void SetRootGraph(const ComputeGraphPtr graph) {
    root_graph_ = graph;
  }

  inline const ComputeGraphPtr &GetRootGraph() const {
    return root_graph_;
  }

  inline void SetModelRelation(std::shared_ptr<ModelRelation> model_relation) {
    model_relation_ = std::move(model_relation);
  }

  inline const std::shared_ptr<ModelRelation> GetModelRelation() const {
    return model_relation_;
  }

  inline void SetDeployResource(std::shared_ptr<ModelDeployResource> deploy_resource) {
    deploy_resource_ = std::move(deploy_resource);
  }

  inline const std::shared_ptr<ModelDeployResource> GetDeployResource() const {
    return deploy_resource_;
  }

 public:
  virtual Status SerializeModel(ModelBufferData &model_buff) = 0;

  virtual Status UnSerializeModel(const ModelBufferData &model_buff) = 0;

  virtual void SetModelId(const uint32_t model_id) {
    model_id_ = model_id;
  }

  virtual uint32_t GetModelId() const {
    return model_id_;
  }

  virtual std::string GetLogicDeviceId() const {
    return "";
  }

 private:
  std::map<std::string, std::shared_ptr<PneModel>> submodels_;
  std::shared_ptr<ModelRelation> model_relation_;
  std::shared_ptr<ModelDeployResource> deploy_resource_;
  ComputeGraphPtr root_graph_ = nullptr;
  std::string model_name_;
  std::string model_type_;
  uint32_t model_id_ = INVALID_MODEL_ID;
};

using PneModelPtr = std::shared_ptr<PneModel>;
}  // namespace ge

#endif  // INC_FRAMEWORK_PNE_MODEL_H_
