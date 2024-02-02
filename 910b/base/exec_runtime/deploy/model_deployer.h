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
#ifndef BASE_RUNTIME_DEPLOY_MODEL_DEPLOYER_H_
#define BASE_RUNTIME_DEPLOY_MODEL_DEPLOYER_H_

#include <map>
#include <string>
#include <vector>
#include "framework/pne/flow_model.h"
#include "common/plugin/ge_util.h"
#include "common/model/model_relation.h"
#include "external/ge/ge_api_error_codes.h"
#include "exec_runtime/deploy/deploy_planner.h"
#include "exec_runtime/deploy/exchange_service.h"

namespace ge {
constexpr uint32_t kCallbackStartRedeploy = 1U;
constexpr uint32_t kCallbackDynamicSched = 2U;
constexpr uint32_t kCallbackFailedRedeploy = 3U;
constexpr uint32_t kCallbackRedeployDone = 4U;
struct DeployResult {
  uint32_t model_id;
  std::vector<DeployQueueAttr> input_queue_attrs;
  std::vector<DeployQueueAttr> output_queue_attrs;
  std::vector<DeployQueueAttr> control_input_queue_attrs;
  std::vector<DeployQueueAttr> control_output_queue_attrs;
  std::vector<Endpoint> event_relations;
  std::function<Status(void)> dev_stat_callback;
  size_t replica_num = 1U;
  std::string input_model_name;
  bool deploy_with_flow = true;
  std::vector<DeployQueueAttr> status_output_queue_attrs;
  std::vector<DeployQueueAttr> sched_input_queue_attrs;
  std::vector<DeployQueueAttr> sched_output_queue_attrs;
  DeployPlan::DynamicSchedIndex model_index_info;
  std::map<int32_t, int32_t> datagw_request_bindings;
  bool is_dynamic_sched = false;
  DeployPlan::AbnormalStatusCallbackInfo *abnormal_status_callback_info = nullptr;
};

class ModelDeployer {
 public:
  ModelDeployer() = default;
  GE_DELETE_ASSIGN_AND_COPY(ModelDeployer);
  virtual ~ModelDeployer() = default;

  /// Deploy model to devices
  /// @param model                models to deploy
  /// @param model_relation       relation among the models, can be nullptr iff models contains single model
  /// @param input_queue_ids      queue id of inputs
  /// @param output_queue_ids     queue id of outputs
  /// @param deploy_result        deploy result
  /// @return                     SUCCESS if deployed successfully, otherwise returns appropriate error code
  virtual Status DeployModel(const FlowModelPtr &flow_model,
                             const std::vector<uint32_t> &input_queue_ids,
                             const std::vector<uint32_t> &output_queue_ids,
                             DeployResult &deploy_result) = 0;

  /// Undeploy model
  /// @param model_id             id of the deployed model
  /// @return                     SUCCESS if undeployed successfully, otherwise returns appropriate error code
  virtual Status Undeploy(const uint32_t model_id) = 0;

  /// Get local device node mesh index
  /// @return                     empty means not support
  virtual Status GetDeviceMeshIndex(const int32_t, std::vector<int32_t> &)  { return UNSUPPORTED; };

  /// Get valid logic device id str
  virtual Status GetValidLogicDeviceId(std::string &) { return UNSUPPORTED; };
};
}  // namespace ge

#endif  // BASE_RUNTIME_DEPLOY_MODEL_DEPLOYER_H_