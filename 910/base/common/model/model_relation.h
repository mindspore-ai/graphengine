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
#ifndef GE_COMMON_MODEL_MODEL_RELATION_H_
#define GE_COMMON_MODEL_MODEL_RELATION_H_
#include <map>
#include <vector>
#include "external/graph/gnode.h"
#include "graph/compute_graph.h"
#include "external/ge/ge_api_error_codes.h"
#include "external/ge/ge_api_types.h"

namespace ge {
/*lint -e148*/
struct ModelRelation {
  struct QueueDef {
    std::string name;
    uint32_t depth = 0U;
    std::string enqueue_policy = "FIFO";
    bool is_control_ = false;
  };

  struct InvokedModelQueueInfo {
    std::vector<std::string> input_queue_names;
    std::vector<std::string> output_queue_names;
  };

  struct ModelQueueInfo {
    std::string model_name;
    std::vector<std::string> input_queue_names;
    std::vector<std::string> output_queue_names;
    std::vector<std::string> external_output_queue_names;  // created by others
    std::vector<std::string> external_input_queue_names;
    std::vector<std::string> invoke_model_keys;
  };

  std::vector<QueueDef> queue_defs;
  // key: model_instance_name
  std::map<std::string, ModelQueueInfo> submodel_queue_infos;
  // key: invoke model key
  std::map<std::string, InvokedModelQueueInfo> invoked_model_queue_infos;
  ModelQueueInfo root_model_queue_info;
};

class ModelRelationBuilder {
 public:
  Status BuildFromRootGraph(const ComputeGraph &root_graph, std::unique_ptr<ModelRelation> &model_relation);
  Status BuildForSingleModel(const ComputeGraph &root_graph, ModelRelation &model_relation);
  virtual ~ModelRelationBuilder() = default;

 protected:
  Status CreateQueueDef(const GeTensorDesc &tensor_desc, const std::string &queue_name);
  static Status GetInputQueueNames(const NodePtr &node,
                                   const std::map<NodePtr, std::map<int32_t, std::string>> &paired_inputs,
                                   std::vector<std::string> &input_queue_names);
  Status CreateQueueForDataNode(const Node &node, const std::string &prefix, std::string &queue_name);

  ModelRelation model_relation_;

 private:
  Status DoBuild(const ComputeGraph &root_graph);
  Status DoBuildForData(const NodePtr &node, std::map<NodePtr, std::map<int32_t, std::string>> &paired_inputs,
                        const ComputeGraph &root_graph);
  Status DoBuildForPartitionedCall(const NodePtr &node, std::map<NodePtr,
                                   std::map<int32_t, std::string>> &paired_inputs);
  Status DoBuildForNetOutput(const NodePtr &node, const std::map<NodePtr,
                             std::map<int32_t, std::string>> &paired_inputs);
  Status CreateEmptyModelRelation(const OpDesc &op_desc);
  Status GetOrCreateModelQueueInfo(const OpDesc &op_desc, ModelRelation::ModelQueueInfo *&model_queue_info);
  Status CheckDataNode(const NodePtr &node, bool &create_relation_flag) const;
  Status CheckNetOutputNode(const NodePtr &node, bool &create_relation_flag) const;

  std::map<std::string, ModelRelation::QueueDef> queue_defs_;
};

class ModelRelationReader {
 public:
  explicit ModelRelationReader(const ModelRelation &model_relation);
  ~ModelRelationReader() = default;

  Status Initialize();

  Status BatchGetQueueDefs(const std::vector<std::string> &queue_names,
                           std::vector<const ModelRelation::QueueDef *> &queue_defs) const;

  const ModelRelation::InvokedModelQueueInfo *GetInvokedModelQueueInfo(const std::string &invoke_key) const;

  const std::vector<const ModelRelation::QueueDef *> &GetInputQueueDefs() const {
    return input_queue_defs_;
  }
  const std::vector<const ModelRelation::QueueDef *> &GetOutputQueueDefs() const {
    return output_queue_defs_;
  }

  const ModelRelation::QueueDef *GetQueueDef(const std::string &queue_name) const;

  const ModelRelation::ModelQueueInfo *GetSubmodelQueueInfo(const std::string &model_name) const;

 private:
  const ModelRelation &model_relation_;
  std::map<std::string, const ModelRelation::QueueDef *> queue_defs_;
  std::vector<const ModelRelation::QueueDef *> input_queue_defs_;
  std::vector<const ModelRelation::QueueDef *> output_queue_defs_;
};
}  // namespace ge

#endif  // GE_COMMON_MODEL_MODEL_RELATION_H_
