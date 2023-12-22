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

// model id, src endpoint index, group id, groups(entry index, dst endpoint index)
using DynamicSchedIndex = std::map<int32_t, std::map<int32_t, std::map<int32_t,
  std::vector<std::pair<int32_t, int32_t>>>>>;

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
               const int32_t proxy_device_id) noexcept;
    static const DeviceInfo &ExternalDevice();
    static bool IsExternal(const DeviceInfo &device_info);
    bool WithProxy() const;
    int32_t GetType() const;
    int32_t GetNodeId() const;
    int32_t GetDeviceId() const;
    int32_t GetProxyDeviceId() const;
    const std::string &GetKey() const;
    const std::string &GetDesc() const;
    int32_t GetHcomDeviceId() const;
    void SetHcomDeviceId(int32_t hcom_device_id);
    int32_t GetOsId() const;
    void SetOsId(int32_t os_id);

   private:
    std::string key_ = "1_0_0";
    std::string desc_ = "1_0_0(-1)";
    int32_t type_ = static_cast<int32_t>(CPU);
    int32_t node_id_ = 0;
    int32_t device_id_ = 0;
    int32_t proxy_device_id_ = -1;
    int32_t hcom_device_id_ = 0;
    int32_t os_id_ = 0;
  };

  enum class QueueAction {
    kDefault,
    kControl,
    kStatus,
    kSched
  };

  struct QueueInfo {
    DeviceInfo device_info;
    uint32_t depth = 2U; // minimal queue depth
    int32_t ref_index = -1;
    std::string name;
    std::string model_instance_name;
    std::string enqueue_policy;
    bool owned = true;
    QueueAction queue_action;
    int32_t fusion_offset = 0;
    uint32_t instance_num;
    uint32_t instance_idx;
    uint32_t model_id = 0U;
    bool is_dummy = false;
    int32_t process_id = 0;
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
    std::vector<int32_t> status_input_queue_indices;
    std::vector<int32_t> status_output_queue_indices;
    std::vector<int32_t> sched_input_queue_indices;
    std::vector<int32_t> sched_output_queue_indices;
    bool is_head = false;
  };

  class DynamicSchedPlan {
  public:
    const std::vector<int32_t> &GetStatusOutputQueueIndices() const;
    const std::vector<int32_t> &GetSchedOutputQueueIndices() const;
    const std::vector<int32_t> &GetSchedInputQueueIndices() const;
    const std::map<int32_t, int32_t> &GetDatagwRequestBindings() const;
    const std::map<int32_t, int32_t> &GetEntryBindings() const;
    const DynamicSchedIndex &GetModelIndexInfo() const;
    const std::map<std::string, uint32_t> &GetModelInstanceNum() const;

  private:
    friend class DeployPlannerBase;
    std::map<int32_t, int32_t> datagw_request_bindings_; // sched output-->datagw_input
    std::map<int32_t, int32_t> entry_to_dst_index_; // group entry-->dst endpoint index
    DynamicSchedIndex model_index_info_;
    SubmodelInfo root_model_info_;
    std::map<std::string, uint32_t> submodels_id_;
    std::map<std::string, std::vector<int32_t>> src_endpoint_indices_;
    // {key: src_endpoint_index, value: {key: model_and_in_queue, value: queue_infos}
    std::map<int32_t, std::map<ModelQueueIndex, std::vector<DeployPlan::QueueInfo>>> endpoint_pairs_;
    std::map<std::string, uint32_t> model_instances_num_;
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
  const DynamicSchedPlan &GetDynamicSchedPlan() const;
  void SetIsDynamicSched(const bool is_dynamic_sched);
  const bool &GetIsDynamicSched() const;

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
  DynamicSchedPlan dynamic_sched_plan_;
  bool is_dynamic_sched_ = false;
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

  struct InputGroupAttr {
    uint32_t instance_num;
    uint32_t instance_idx;
  };

 protected:
  virtual Status PrepareModelsAndRelation(ModelRelation &model_relation) = 0;
  virtual void SelectHeadAndTailDevice(DeployPlan::DeviceInfo &device_info) {
    device_info = DeployPlan::DeviceInfo();
  }
  DeployPlan::SubmodelInfo &MutableSubmodelInfo(const std::string &name);
  static Status ValidateModelAndRelation(const std::map<std::string, PneModelPtr> &models,
                                         const ModelRelation &model_relation);
  bool IsHeadOrTail(const std::string &name) const;
  Status CreateEndpointInfo(const DeployPlan::QueueInfo &queue_info);
  Status CreateEndpointInfo(const DeployPlan::QueueInfo &queue_info, int32_t &queue_idx);
  Status CreateGroupEntry(const DeployPlan::QueueInfo &queue_info, int32_t &entry_index);
  Status CreateGroupRefEntry(const DeployPlan::QueueInfo &queue_info,
                             int32_t endpoint_index,
                             int32_t &entry_index);
  Status CreateGroupQueueEntry(const DeployPlan::QueueInfo &queue_info,
                               int32_t &queue_index,
                               int32_t &entry_index);
  Status CreateGroupInfo(const DeployPlan::QueueInfo &queue_info,
                         const std::vector<int32_t> &grouped_indices,
                         int32_t &group_index);
  void AddEndpointBindings(int32_t src_index, int32_t dst_index, bool skip_if_dst_exists = true);
  static Status ConvertToTagInfo(const Endpoint &endpoint, std::vector<DeployPlan::QueueInfo> &tag_infos);
  void SetFlowSends(const std::map<std::string, std::vector<DeployPlan::QueueInfo>> &flow_sends);
  void SetFlowRecvs(const std::map<std::string, std::vector<DeployPlan::QueueInfo>> &flow_recvs);
  const bool &GetIsDynamicSched() const;

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
  Status ResolveModelDynamicInputs(const std::string &model_instance_name,
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
  void AddInputGroups(const int32_t dst_endpoint_idx,
                      const int32_t src_tag_idx,
                      const InputGroupAttr &input_group_attr);
  std::vector<std::string> ToEndpointDescs(const std::vector<int32_t> &endpoint_indices,
                                           const bool is_group_entry = false) const;
  std::string ToEndpointDesc(const int32_t endpoint_indices, const bool is_group_entry = false) const;
  static void BuildEventInfo(Endpoint &endpoint, const std::string &model_instance_name,
                             std::shared_ptr<DeployPlan::EventInfo> event_info);
  DeployPlan::QueueInfo BuildQueueInfo(const Endpoint &queue_def,
                                       const std::string &model_instance_name);
  std::string GetEndpointFullName(const DeployPlan::QueueInfo &endpoint_info, const ModelQueueIndex &model_queue_index);
  const std::string &GetSubmodelType(const std::string &name);
  bool CheckAndAddRelation(const int32_t src_endpoint_idx, const int32_t dst_endpoint_idx);
  bool IsOutputMultiConnected(const int32_t src_endpoint_idx);
  bool IsInputMultiConnected(const int32_t dst_endpoint_idx);
  bool IsMultiDeployed(const std::string &model_instance_name) const;
  bool CheckSkipBinding(const std::string &src_model_instance_name,
                        const std::string &dst_model_instance_name) const;
  static bool CanConnectWithQ(const DeployPlan::DeviceInfo &src_device_info,
                              const DeployPlan::DeviceInfo &dst_device_info);
  Status GetOrCreateMappingTagPairEntry(const int32_t endpoint_idx,
                                        const DeployPlan::QueueInfo &mapping_queue_info,
                                        std::pair<int32_t, int32_t> &tag_pair);
  void GenTagEntityPair(int32_t endpoint_idx,
                        const DeployPlan::QueueInfo &mapping_queue_info,
                        std::pair<DeployPlan::QueueInfo, DeployPlan::QueueInfo> &entity_pair);
  Status GetOrCreateMappingEntry(const int32_t endpoint_idx,
                                 const DeployPlan::QueueInfo &mapping_queue_info,
                                 int32_t &mapping_idx);
  Status PrepareDiffNodeRelation(const int32_t src_endpoint_idx,
                                 const int32_t dst_endpoint_idx,
                                 const ModelQueueIndex &model_queue_loc,
                                 const DeployPlan::QueueInfo &queue_info,
                                 const InputGroupAttr &input_group_attr);
  Status PrepareSameNodeRelation(const int32_t src_endpoint_idx,
                                  const int32_t dst_endpoint_idx,
                                  const ModelQueueIndex &model_queue_loc,
                                  const DeployPlan::QueueInfo &queue_info,
                                  const InputGroupAttr &input_group_attr);
  Status PrepareQueuesRelation(const int32_t src_endpoint_idx,
                               const int32_t dst_endpoint_idx,
                               const ModelQueueIndex &model_queue_loc,
                               const DeployPlan::QueueInfo &queue_info,
                               const InputGroupAttr &input_group_attr);
  Status PrepareRelations(const int32_t src_endpoint_idx,
                          const int32_t dst_endpoint_idx,
                          const ModelQueueIndex &model_queue_loc,
                          const DeployPlan::QueueInfo &queue_info,
                          const InputGroupAttr &input_group_attr);
  Status ResolveExternalFlowEndpoints();
  static bool IsExternalEndpoint(const DeployPlan::QueueInfo &endpoint_info);
  Status CreateAndBindGroup(const DeployPlan::QueueInfo &group_info,
                            const std::vector<int32_t> &group_entry_index,
                            const int32_t dst_endpoint_index,
                            const bool skip_if_dst_exists = true);

  // dynamic sched deploy build
  Status AssignDynamicSchedDequeueQueues();
  Status AssignDynamicSchedDequeueQueue(const DeployPlan::QueueInfo &queue_info,
                                        const ModelQueueIndex &model_queue_loc,
                                        const int32_t &src_endpoint_idx);
  Status CreateDynamicSchedOutputQueueDefs(const std::string &model_instance_name,
                                           const std::vector<std::string> &queue_names,
                                           const bool is_owned = true);
  Status CreateDynamicSchedTags(const int32_t src_endpoint_idx,
                                const int32_t dst_endpoint_idx,
                                const DeployPlan::QueueInfo &queue_info);
  void GenerateDynamicSchedModelId();
  Status AssignDynamicSchedEnqueueQueues();
  void UpdateRelationForDynamicSched();
  Status DynamicSchedBindGroup2Queue(const int32_t src_idx,
                                     const int32_t dst_idx,
                                     int32_t &group_index);
  Status DynamicSchedBindQueue2Group(const int32_t src_idx,
                                     const int32_t dst_idx,
                                     int32_t &group_index);
  void UpdateDynamicSchedDeployPlan();
  Status BuildDynamicSchedInfo();
  Status SetHeadNodeInfo();

  ModelRelation model_relation_;
  std::unique_ptr<ModelRelationReader> relation_reader_;
  std::map<std::string, std::vector<int32_t>> src_endpoint_indices_;
  // {key: src_endpoint_index, value: {key: model_and_in_queue, value: queue_infos}
  std::map<int32_t, std::map<ModelQueueIndex, std::vector<DeployPlan::QueueInfo>>> endpoint_pairs_;
  // {key：dst endpoint name, value: src_endpoint_index set}
  std::map<std::string, std::set<int32_t>> relation_dst_to_src_;
  std::set<int32_t> reusable_queue_indices_;
  std::map<std::tuple<ModelQueueIndex, std::string, int32_t>, int32_t> input_endpoint_indices_;
  // {key: src_endpoint_index, value: {key: model_and_in_queue, value: dst_endpoint_index}
  std::map<int32_t, std::map<std::string, int32_t>> dequeue_ref_indices_;
  // for creating outgoing group, entries are ordered by device_key_process_id
  std::map<int32_t, std::map<ModelQueueIndex, std::map<std::string, int32_t>>> output_groups_;
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
  std::map<std::string, std::vector<std::pair<std::string, DeployPlan::DeviceInfo>>> model_deploy_locations_;
  // key:pair(endpoint index,device_key) value:mapping endpoint index
  std::map<std::pair<int32_t, std::string>, int32_t> endpoint_device_mapping_;
  // key:pair(endpoint index,device_key) value:mapping src and dst tag index
  std::map<std::pair<int32_t, std::string>, std::pair<int32_t, int32_t>> endpoint_device_tags_mapping_;
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

 private:
  const PneModelPtr root_model_;
};
}  // namespace ge
#endif  // BASE_EXEC_RUNTIME_DEPLOY_DEPLOY_PLANNER_H_
