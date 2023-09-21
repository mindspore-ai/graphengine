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
#ifndef BASE_EXEC_RUNTIME_DEPLOY_DEPLOY_PLANNER_H_
#define BASE_EXEC_RUNTIME_DEPLOY_DEPLOY_PLANNER_H_
#include <atomic>
#include <map>
#include <set>
#include <vector>
#include "common/model/ge_root_model.h"
#include "common/model/model_deploy_resource.h"

namespace ge {
/**
 * Deploy plan for GeRootModel
 */
class DeployPlan {
 public:
  class DeviceInfo {
   public:
    DeviceInfo() = default;
    DeviceInfo(const int32_t type, const int32_t node_id, const int32_t device_id) noexcept;
    DeviceInfo(const int32_t type, const int32_t node_id, const int32_t device_id,
               const int32_t phy_device_id) noexcept;
    static const DeviceInfo &ExternalDevice();
    static bool IsExternal(const DeviceInfo &device_info);
    int32_t GetType() const;
    int32_t GetNodeId() const;
    int32_t GetDeviceId() const;
    int32_t GetPhyDeviceId() const;
    const std::string &GetKey() const;
    const std::string &GetDesc() const;

   private:
    std::string key_ = "1_0_0(-1)";
    int32_t type_ = static_cast<int32_t>(CPU);
    int32_t node_id_ = 0;
    int32_t device_id_ = 0;
    int32_t phy_device_id_ = -1;
  };

  struct QueueInfo {
    DeviceInfo device_info;
    uint32_t depth = 2U; // minimal queue depth
    int32_t ref_index = -1;
    std::string name;
    std::string model_instance_name;
    std::string enqueue_policy;
    bool owned = true;
    bool is_control = false;
    int32_t fusion_offset = 0;
    uint32_t instance_num;
    uint32_t instance_idx;
    bool is_local = false;
  };

  struct InvokedModelQueueInfo {
    std::vector<int32_t> feed_queue_indices;
    std::vector<int32_t> fetch_queue_indices;
  };

  struct RankInfo {
    bool deploy_with_rank = false;
    uint32_t rank_id;
    std::string hcom_cluster_name;
  };

  struct EventInfo {
    std::string name;
    std::string model_instance_name;
    std::string group_name;
    std::string event_type;
    int32_t tag;
    int32_t logic_peer_rank;
    int32_t peer_rank;
  };

  enum class ProcessMode {
    kProcess,
    kThread
  };

  enum class LoadMode {
    kLoadWithQ,
    kLoadWithEvent,
    kLoadOnline,
    kLoadWithSyncEvent
  };

  struct LoadInfo {
    ProcessMode process_mode = ProcessMode::kProcess;
    LoadMode load_mode = LoadMode::kLoadWithQ;
  };

  struct SubmodelInfo {
    int32_t process_id = 0;
    DeviceInfo device_info;
    DeviceInfo queue_device_info;
    RankInfo rank_info;
    LoadInfo load_info;
    PneModelPtr model;
    std::vector<int32_t> input_queue_indices;
    std::vector<int32_t> control_input_queue_indices;
    std::vector<int32_t> output_queue_indices;
    std::vector<int32_t> control_output_queue_indices;
    std::map<std::string, std::string> attrs;
    // sync event info, to be unify to input_queue_indices/output_queue_indices
    std::vector<std::shared_ptr<EventInfo>> event_infos;
    // key:invoke key
    std::map<std::string, InvokedModelQueueInfo> invoked_model_queue_infos;
  };

  /// Get QueueInfo by queue_index
  /// @param queue_index      queue index
  /// @param queue_info       queue info
  /// @return                 SUCCESS if got successfully, otherwise returns appropriate error code
  Status GetQueueInfo(const int32_t queue_index, const DeployPlan::QueueInfo *&queue_info) const;

  /// getters and setters
  const std::vector<QueueInfo> &GetQueueInfoList() const;
  const std::vector<QueueInfo> &GetGroupEntryInfoList() const;
  const std::vector<std::pair<int32_t, int32_t>> &GetQueueBindings() const;
  const std::vector<int32_t> &GetInputQueueIndices() const;
  const std::vector<int32_t> &GetControlInputQueueIndices() const;
  const std::vector<int32_t> &GetControlOutputQueueIndices() const;
  const DeployPlan::DeviceInfo &GetRootModelQueueDeviceInfo() const;
  std::vector<int32_t> GetAllInputQueueIndices() const;
  const std::vector<int32_t> &GetOutputQueueIndices() const;
  const std::map<std::string, SubmodelInfo> &GetSubmodels() const;
  std::map<std::string, SubmodelInfo> &MutableSubmodels();
  const std::map<int32_t, std::vector<int32_t>> &GetGroups() const;
  bool IsGroupEndpoint(const int32_t queue_index) const;
  const std::vector<HcomCommGroup> &GetCommGroups(const std::string &hcom_cluster_name) const;
  void AddCommGroup(const std::string &hcom_cluster_name, const HcomCommGroup &comm_group);
  void AddHcomRankTable(const std::string &name, const std::string &rank_table);
  const std::string &GetHcomRankTable(const std::string &hcom_cluster_name) const;
  const std::map<std::string, DeployPlan::DeviceInfo> &GetFlowSendTagNameToLocalDeviceInfo() const;
  const std::map<std::string, DeployPlan::DeviceInfo> &GetFlowRecvTagNameToLocalDeviceInfo() const;

 private:
  friend class DeployPlannerBase;
  std::string model_name_;
  std::vector<QueueInfo> queues_;
  std::vector<EventInfo> events_;
  std::vector<std::pair<int32_t, int32_t>> queue_bindings_;
  std::map<int32_t, int32_t> dst_to_src_bindings_;
  SubmodelInfo root_model_info_;
  // key: model_instance_name
  std::map<std::string, SubmodelInfo> submodels_;
  // key is group queue index, value is sub queue index list
  std::map<int32_t, std::vector<int32_t>> groups_;
  std::map<std::string, int32_t> groups_key_to_idx_;
  std::vector<QueueInfo> group_entries_;
  std::map<std::string, std::vector<HcomCommGroup>> cluster_name_to_comm_groups_;
  std::map<std::string, std::string> cluster_name_to_rank_table_;
  std::map<std::string, DeployPlan::DeviceInfo> flow_send_tag_name_to_local_device_info_;
  std::map<std::string, DeployPlan::DeviceInfo> flow_recv_tag_name_to_local_device_info_;
};

class DeployPlannerBase {
 public:
  DeployPlannerBase() = default;
  GE_DELETE_ASSIGN_AND_COPY(DeployPlannerBase);
  virtual ~DeployPlannerBase() = default;

  /// Build DeployPlan
  /// @param deploy_plan      output DeployPlan
  /// @return                 SUCCESS if built successfully, otherwise returns appropriate error code
  Status BuildPlan(DeployPlan &deploy_plan);

  struct ModelQueueIndex {
    std::string model_name;
    // if not empty, means model is invoked by others.
    std::string invoke_key;
    int32_t id;
    bool operator < (const ModelQueueIndex &other) const {
      if (model_name != other.model_name) {
        return model_name < other.model_name;
      } else if (invoke_key != other.invoke_key) {
        return invoke_key < other.invoke_key;
      } else {
        return id < other.id;
      }
    }
  };

  struct InputGroupAttr {
    uint32_t instance_num;
    uint32_t instance_idx;
  };

 protected:
  virtual Status PrepareModelsAndRelation(ModelRelation &model_relation) = 0;
  virtual void SelectHeadAndTailDevice(DeployPlan::DeviceInfo &device_info) {
    device_info = DeployPlan::DeviceInfo();
  }
  virtual bool NeedProxyQ(const std::string &model_instance_name) = 0;
  DeployPlan::SubmodelInfo &MutableSubmodelInfo(const std::string &name);
  static Status ValidateModelAndRelation(const std::map<std::string, PneModelPtr> &models,
                                         const ModelRelation &model_relation);
  bool IsHeadOrTail(const std::string &name) const;
  Status CreateEndpointInfo(const DeployPlan::QueueInfo &queue_info);
  Status CreateEndpointInfo(const DeployPlan::QueueInfo &queue_info, int32_t &queue_idx);
  Status CreateGroupEntry(const DeployPlan::QueueInfo &queue_info, int32_t &entry_index);
  Status CreateGroupInfo(const DeployPlan::QueueInfo &queue_info,
                         const std::vector<int32_t> &grouped_indices,
                         int32_t &group_index);
  void AddEndpointBindings(int32_t src_index, int32_t dst_index, bool skip_if_dst_exists = true);
  static Status ConvertToTagInfo(const Endpoint &endpoint, std::vector<DeployPlan::QueueInfo> &tag_infos);
  void SetFlowSends(const std::map<std::string, std::vector<DeployPlan::QueueInfo>> &flow_sends);
  void SetFlowRecvs(const std::map<std::string, std::vector<DeployPlan::QueueInfo>> &flow_recvs);

  static std::atomic<int64_t> plan_id_gen_;
  DeployPlan deploy_plan_;

 private:
  Status Initialize();
  // methods for parsing model relation
  Status ParseModelRelation();
  void UpdateForInputControlIo();
  void UpdateForOutputControlIo();
  void UpdateRelationForControlIo();
  Status AssignEnqueueQueues();
  Status ResolveEnqueueFusion();
  Status ResolveDequeueFusion(int32_t src_endpoint_idx, int32_t dst_endpoint_idx);
  Status ResolveInputsPlacement(const std::string &model_instance_name,
                                const ModelRelation::ModelEndpointInfo &model_endpoint_info);
  Status ResolveModelFusion(const std::string &model_instance_name,
                            const ModelRelation::ModelEndpointInfo &model_endpoint_info);
  bool CanBeFused(const std::string &fusion_name, const std::string &endpoint_name);
  void UpdateFusionOffset(int32_t src_index, int32_t dst_index);
  void MarkMultiDeployedModels();
  Status ResolveDataFlows();
  Status ResolveModelInputs(const std::string &model_instance_name,
                            const ModelRelation::ModelEndpointInfo &model_endpoint_info);
  void LogDataFlow() const;
  Status ResolveReusableQueues();
  Status AssignDequeueQueues();
  Status BindRemoteOutputGroupToInput();
  Status BindOutputToRemoteInputs();
  void UpdateDeployPlan();
  Status CreateEndpointInfo(std::shared_ptr<DeployPlan::EventInfo> event_info, const std::string &model_instance_name);
  Status CreateOutputQueueDefs(const std::string &model_instance_name,
                               const std::vector<std::string> &queue_names,
                               const bool is_owned = true);
  Status CreateInputOutputEventDefs(const std::string &model_instance_name,
                                    const std::vector<std::string> &endpoint_names);
  Status CreateFeedEndpoints(const std::string &model_instance_name,
                             const std::vector<std::string> &queue_names,
                             const std::string &invoke_key);
  Status GetOrCreateInputEndpoint(const ModelQueueIndex &model_queue_index,
                                  const DeployPlan::QueueInfo &queue_info,
                                  int32_t &endpoint_index);
  bool IsOneToMany(const int32_t src_endpoint_idx);
  bool IsManyToOne(const int32_t dst_endpoint_idx);
  void AddInputGroups(const int32_t dst_endpoint_idx,
                      const int32_t src_tag_idx,
                      const InputGroupAttr &input_group_attr);
  Status CreateTags(const int32_t src_endpoint_idx,
                    const int32_t dst_endpoint_idx,
                    const ModelQueueIndex &model_queue_loc,
                    const DeployPlan::QueueInfo &queue_info,
                    const InputGroupAttr &input_group_attr);
  Status CreateOutputTags(const int32_t src_endpoint_idx,
                          const DeployPlan::QueueInfo &dst_queue_info,
                          int32_t &src_tag_idx,
                          int32_t &dst_tag_idx);
  std::vector<std::string> ToEndpointDescs(const std::vector<int32_t> &endpoint_indices,
                                           const bool is_group_entry = false) const;
  std::string ToEndpointDesc(const int32_t endpoint_indices, const bool is_group_entry = false) const;
  static void BuildEventInfo(Endpoint &endpoint, const std::string &model_instance_name,
                             std::shared_ptr<DeployPlan::EventInfo> event_info);
  DeployPlan::QueueInfo BuildQueueInfo(const Endpoint &queue_def,
                                       const std::string &model_instance_name, bool is_proxy_q = false);
  std::string GetEndpointFullName(const DeployPlan::QueueInfo &endpoint_info, const ModelQueueIndex &model_queue_index);
  const std::string &GetSubmodelType(const std::string &name);
  bool CheckAndAddRelation(const int32_t src_endpoint_idx, const int32_t dst_endpoint_idx);
  Status ResolveExternalFlowEndpoints();
  static bool IsExternalEndpoint(const DeployPlan::QueueInfo &endpoint_info);
  Status CreateAndBindGroup(const DeployPlan::QueueInfo &group_info,
                            const std::vector<int32_t> &grouped_indices,
                            const int32_t dst_endpoint_index,
                            bool skip_if_dst_exists = true);

  ModelRelation model_relation_;
  std::unique_ptr<ModelRelationReader> relation_reader_;
  std::map<std::string, std::vector<int32_t>> src_endpoint_indices_;
  // {key: src_endpoint_index, value: {key: model_and_in_queue, value: queue_infos}
  std::map<int32_t, std::map<ModelQueueIndex, std::vector<DeployPlan::QueueInfo>>> endpoint_pairs_;
  // {key：dst endpoint name, value: src_endpoint_index set}
  std::map<std::string, std::set<int32_t>> relation_dst_to_src_;
  std::set<int32_t> reusable_queue_indices_;
  std::map<std::pair<ModelQueueIndex, std::string>, int32_t> input_endpoint_indices_;
  // {key: src_endpoint_index, value: {key: model_and_in_queue, value: dst_endpoint_index}
  std::map<int32_t, std::map<std::string, int32_t>> dequeue_ref_indices_;
  // for creating outgoing group, entries are ordered by device key
  std::map<int32_t, std::map<ModelQueueIndex, std::map<std::string, int32_t>>> output_groups_;
  std::map<int32_t, std::map<std::string, std::pair<int32_t, int32_t>>> output_tags_;
  std::map<std::string, std::set<std::string>> dequeue_placements_;
  std::set<std::string> disable_fusion_queues_;
  // for creating incoming group
  std::map<int32_t, std::vector<int32_t>> input_groups_;
  std::map<int32_t, InputGroupAttr> input_groups_attr_;
  // for unifying input/output queues
  ModelRelation::ModelEndpointInfo head_model_queue_info_;
  ModelRelation::ModelEndpointInfo tail_model_queue_info_;
  DeployPlan::SubmodelInfo head_model_info_;
  DeployPlan::SubmodelInfo tail_model_info_;
  DeployPlan::SubmodelInfo external_model_info_;
  std::map<std::string, std::vector<DeployPlan::QueueInfo>> flow_sends_;
  std::map<std::string, std::vector<DeployPlan::QueueInfo>> flow_recvs_;
  static std::atomic<int64_t> endpoint_name_id_gen_;
  std::map<std::string, std::string> short_names_;
  std::map<std::string, std::string> instance_to_model_name_;
  std::map<std::string, std::set<std::string>> deploy_to_devlist_;
  std::set<std::string> relations_;  // key: src_endpoint_index_to_dst_endpoint_index
  std::set<std::string> no_group_endpoint_names_;
};

class ModelRelationFlattener {
 public:
  explicit ModelRelationFlattener(PneModelPtr root_model);
  Status Flatten(ModelRelation &flattened_model_relation, std::map<std::string, PneModelPtr> &name_to_models);
  static Status Flatten(const PneModelPtr &root_model);
 private:
  Status FlattenSubmodel(const ModelRelation::ModelEndpointInfo &parent_model_queue_info,
                         const PneModelPtr &pne_model,
                         const int32_t depth);
  void MergeEndpoints(const std::map<std::string, std::string> &name_refs,
                      const std::vector<Endpoint> &endpoints);
  static void ReplaceQueueNames(const std::map<std::string, std::string> &name_refs, std::vector<std::string> &names);
  static std::map<std::string, std::string> BuildNameRefs(
      const ModelRelation::ModelEndpointInfo &parent_model_queue_info,
      const ModelRelation::ModelEndpointInfo &root_model_queue_info);
  static Status CheckConsistency(const ModelRelation::ModelEndpointInfo &parent_model_queue_info,
                                 const ModelRelation::ModelEndpointInfo &root_model_queue_info);

  static bool NeedFlatten(const PneModelPtr &root_model);

  PneModelPtr root_model_;
  ModelRelation flattened_model_relation_;
  std::map<std::string, PneModelPtr> leaf_models_;
  int32_t max_depth_ = 16;
};

class DeployPlanner : public DeployPlannerBase {
 public:
  explicit DeployPlanner(const PneModelPtr &root_model);
  ~DeployPlanner() override = default;

 protected:
  Status PrepareModelsAndRelation(ModelRelation &model_relation) override;
  bool NeedProxyQ(const std::string&) override {
    return false;
  }

 private:
  const PneModelPtr root_model_;
};
}  // namespace ge
#endif  // BASE_EXEC_RUNTIME_DEPLOY_DEPLOY_PLANNER_H_
