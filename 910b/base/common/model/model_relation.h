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
#include "endpoint.h"
#include "external/graph/gnode.h"
#include "graph/compute_graph.h"
#include "external/ge/ge_api_error_codes.h"
#include "external/ge/ge_api_types.h"
#include "common/checker.h"

namespace ge {
/*lint -e148*/
struct ModelRelation {
  struct InvokedModelQueueInfo {
    std::vector<std::string> input_queue_names;
    std::vector<std::string> output_queue_names;
  };

  struct ModelEndpointInfo {
    std::string model_name;
    std::vector<std::string> input_endpoint_names;
    std::vector<std::string> output_endpoint_names;
    std::vector<std::string> external_output_queue_names;  // created by others
    std::vector<std::string> external_input_queue_names;
    std::vector<std::string> p2p_input_node_names;
    std::vector<std::string> p2p_output_node_names;
    std::vector<std::string> invoke_model_keys;
    bool IsEmpty() const {
      return input_endpoint_names.empty() && output_endpoint_names.empty() &&
             external_output_queue_names.empty() && external_input_queue_names.empty() &&
             p2p_input_node_names.empty() && p2p_output_node_names.empty() &&
             invoke_model_keys.empty();
    }
  };

  std::vector<Endpoint> endpoints;
  // key: model_instance_name
  std::map<std::string, ModelEndpointInfo> submodel_endpoint_infos;
  // key: invoke model key
  std::map<std::string, InvokedModelQueueInfo> invoked_model_queue_infos;
  ModelEndpointInfo root_model_endpoint_info;
  bool IsEmpty() const {
    return endpoints.empty() && submodel_endpoint_infos.empty() &&
           invoked_model_queue_infos.empty() && root_model_endpoint_info.IsEmpty();
  }
};

class ModelRelationBuilder {
 public:
  Status BuildFromRootGraph(const ComputeGraph &root_graph, std::unique_ptr<ModelRelation> &model_relation);
  Status BuildForSingleModel(const ComputeGraph &root_graph, ModelRelation &model_relation);
  static Status SetLogicPeerRankForEndpoints(
      const std::map<std::string, std::vector<uint32_t>> &device_id_and_engine_to_rank_ids,
      const std::unique_ptr<ModelRelation> &model_relation);
  virtual ~ModelRelationBuilder() = default;

 protected:
  Status CreateQueueDef(const GeTensorDesc &tensor_desc, const std::string &queue_name);
  static Status GetInputQueueNames(const NodePtr &node,
                                   const std::map<NodePtr, std::map<int32_t, std::string>> &paired_inputs,
                                   std::vector<std::string> &input_queue_names);
  Status CreateQueueForDataNode(const Node &node, const std::string &prefix,
                                std::string &queue_name, const bool inner_node_flag = false);

  ModelRelation model_relation_;

 private:
  Status DoBuild(const ComputeGraph &root_graph);
  Status DoBuildForData(const NodePtr &node, std::map<NodePtr, std::map<int32_t, std::string>> &paired_inputs,
                        const ComputeGraph &root_graph);
  Status DoBuildForPartitionedCall(const ComputeGraph &subgraph,
                                   const NodePtr &node, std::map<NodePtr,
                                   std::map<int32_t, std::string>> &paired_inputs);
  Status DoBuildForNetOutput(const NodePtr &node, const std::map<NodePtr,
                             std::map<int32_t, std::string>> &paired_inputs);
  bool CheckInnerNode(const NodePtr &node) const;
  Status GetOrCreateModelEndpointInfo(const OpDesc &op_desc, ModelRelation::ModelEndpointInfo *&model_endpoint_info);
  ModelRelation::ModelEndpointInfo *GetOrCreateModelEndpointInfo(const std::string &model_name);
  Status CheckNetOutputNode(const NodePtr &node) const;
  Status CreateExternalEndpointInfo(const ComputeGraph &subgraph,
                                    ModelRelation::ModelEndpointInfo *&model_endpoint_info);
  Status CreateP2pNode(const std::string &p2p_node_name, const Endpoint &p2p_node);
  Status AddP2pNodeByGraphRecursively(
      const ComputeGraphPtr &graph, const std::string &model_name,
      const std::map<std::string, std::map<std::string, std::vector<Endpoint>>> &all_endpoints_by_graph,
      const int32_t depth = 16);
  Status AddP2pNodeByOneGraph(
      const std::map<std::string, std::map<std::string, std::vector<Endpoint>>> &all_endpoints_by_graph,
      const ComputeGraphPtr &subgraph, const std::string &model_name);
  Status GetP2pNodeAndSetEndpoint(
      const NodePtr &node,
      const std::map<std::string, std::map<std::string, std::vector<Endpoint>>> &all_endpoints_by_graph);
  std::map<std::string, Endpoint> endpoints_;
};

class ModelRelationReader {
 public:
  explicit ModelRelationReader(const ModelRelation &model_relation);
  ~ModelRelationReader() = default;

  Status Initialize();

  Status BatchGetEndpoints(const vector<std::string> &endpoint_names,
                           vector<const Endpoint *> &endpoints) const;

  const ModelRelation::InvokedModelQueueInfo *GetInvokedModelQueueInfo(const std::string &invoke_key) const;

  const Endpoint *GetEndpoint(const std::string &queue_name) const;

  const ModelRelation::ModelEndpointInfo *GetSubmodelQueueInfo(const std::string &model_name) const;
  static void LogDebugString(const ModelRelation &model_relation);

 private:
  const ModelRelation &model_relation_;
  std::map<std::string, const Endpoint *> endpoints_;
  std::vector<const Endpoint *> input_endpoints_;
  std::vector<const Endpoint *> output_endpoints_;
};
}  // namespace ge

#endif  // GE_COMMON_MODEL_MODEL_RELATION_H_
