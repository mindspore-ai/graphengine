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

#include "graph/optimize/graph_optimize.h"

#include <utility>

#include "cce/optimizer/fusion_engine.h"
#include "framework/common/debug/ge_log.h"
#include "graph/anchor.h"
#include "graph/passes/dimension_adjust_pass.h"
#include "graph/utils/graph_utils.h"
#include "inc/pass_manager.h"
#include "init/gelib.h"
#include "opskernel_manager/ops_kernel_manager.h"

using ge::ComputeGraph;
using ge::OpDesc;

namespace {
const char *const kVectorEngine = "VectorEngine";
const char *const kAicoreEngine = "AIcoreEngine";
}  // namespace

namespace ge {
GraphOptimize::GraphOptimize()
    : optimize_type_(domi::FrameworkType::FMK_TYPE_T),
      cal_config_(""),
      insert_op_config_(""),
      parse_out_node_(""),
      core_type_(kAicoreEngine),
      graph_context_(nullptr) {}

void AddNodeInputProperty(ComputeGraphPtr &compute_graph) {
  if (compute_graph == nullptr) {
    GELOGE(GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL, "[AddNodeInputProperty]: compute_graph is nullptr.");
    return;
  }
  for (ge::NodePtr &node : compute_graph->GetDirectNode()) {
    auto node_op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, GELOGW("node_op_desc is nullptr!"); return);
    auto in_control_anchor = node->GetInControlAnchor();
    vector<string> src_name_list;
    vector<string> input_name_list;
    vector<int64_t> src_index_list;
    GE_IF_BOOL_EXEC(
        in_control_anchor != nullptr, string src_name_temp; for (auto &out_control_anchor
                                                                 : in_control_anchor->GetPeerOutControlAnchors()) {
          ge::NodePtr src_node = out_control_anchor->GetOwnerNode();
          GE_IF_BOOL_EXEC(src_node == nullptr, GELOGW("src_node is nullptr!"); continue);
          src_name_temp = src_name_temp == "" ? src_node->GetName() : src_name_temp + ":" + src_node->GetName();
        } GE_IF_BOOL_EXEC(src_name_temp != "", src_name_list.emplace_back(src_name_temp);
                          node_op_desc->SetSrcName(src_name_list);))

    for (auto &in_data_anchor : node->GetAllInDataAnchors()) {
      auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
      GE_IF_BOOL_EXEC(peer_out_anchor == nullptr,
                      GELOGW("peer_out_anchor is nullptr! node: %s", node->GetName().c_str());
                      continue);

      ge::NodePtr src_node = peer_out_anchor->GetOwnerNode();
      src_name_list = node_op_desc->GetSrcName();
      src_index_list = node_op_desc->GetSrcIndex();
      src_name_list.emplace_back(src_node->GetName());
      src_index_list.emplace_back(peer_out_anchor->GetIdx());
      node_op_desc->SetSrcName(src_name_list);
      node_op_desc->SetSrcIndex(src_index_list);
      GE_IF_BOOL_EXEC(!(node_op_desc->GetType() == NETOUTPUT && GetContext().type == domi::FMK_TYPE_T),
                      ge::NodePtr peer_owner_node = peer_out_anchor->GetOwnerNode();
                      input_name_list = node_op_desc->GetInputName(); input_name_list.emplace_back(
                          peer_owner_node->GetName() +
                          (peer_out_anchor->GetIdx() == 0 ? "" : ": " + to_string(peer_out_anchor->GetIdx())));
                      node_op_desc->SetInputName(input_name_list);)
    }
  }
}

Status GraphOptimize::OptimizeSubGraph(ComputeGraphPtr &compute_graph, const std::string &engine_name) {
  if (compute_graph == nullptr) {
    GELOGE(GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL, "[OptimizeSubGraph]: compute_graph is nullptr.");
    return GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL;
  }

  Status ret = SUCCESS;
  vector<GraphOptimizerPtr> graph_optimizer;

  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "GraphOptimzer: GE is  not initialized");
    return GE_CLI_GE_NOT_INITIALIZED;
  }

  if (instance_ptr->DNNEngineManagerObj().IsEngineRegistered(engine_name)) {
    instance_ptr->OpsKernelManagerObj().GetGraphOptimizerByEngine(engine_name, graph_optimizer);
    AddNodeInputProperty(compute_graph);

    if (compute_graph->GetDirectNode().size() == 0) {
      GELOGW("[OptimizeSubGraph] compute_graph do not has any node.");
      return SUCCESS;
    }

    for (auto iter = graph_optimizer.begin(); iter != graph_optimizer.end(); ++iter) {
      ret = (*iter)->OptimizeFusedGraph(*(compute_graph));
      if (ret != SUCCESS) {
        GELOGE(ret, "[OptimizeSubGraph][OptimizeFusedGraph]: graph optimize failed, ret:%d", ret);
        return ret;
      }
    }
  } else {
    GELOGI("Engine: %s is not registered. do nothing in subGraph Optimize by ATC.", engine_name.c_str());
  }

  return ret;
}

Status GraphOptimize::OptimizeOriginalGraph(ComputeGraphPtr &compute_graph) {
  if (compute_graph == nullptr) {
    GELOGE(GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL, "[OptimizeOriginalGraph]: compute_graph is nullptr.");
    return GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL;
  }

  Status ret = SUCCESS;
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "OptimizeOriginalGraph failed.");
    return GE_CLI_GE_NOT_INITIALIZED;
  }

  std::map<string, GraphOptimizerPtr> graph_optimizer = instance_ptr->OpsKernelManagerObj().GetAllGraphOptimizerObjs();
  GELOGI("optimize by opskernel in original graph optimize phase. num of graph_optimizer is %lu.",
         graph_optimizer.size());
  string exclude_core_Type = (core_type_ == kVectorEngine) ? kAicoreEngine : kVectorEngine;
  GELOGD("[OptimizeOriginalGraph]: engine type will exclude: %s", exclude_core_Type.c_str());
  if (graph_optimizer.size() != 0) {
    for (auto iter = graph_optimizer.begin(); iter != graph_optimizer.end(); ++iter) {
      if (iter->first == exclude_core_Type) {
        continue;
      }
      ret = (iter->second)->OptimizeOriginalGraph(*compute_graph);
      if (ret != SUCCESS) {
        GELOGE(ret, "[OptimizeOriginalGraph]: graph optimize failed, ret:%d", ret);
        return ret;
      }
    }
  }
  return ret;
}

Status GraphOptimize::OptimizeOriginalGraphForQuantize(ComputeGraphPtr &compute_graph) {
  if (compute_graph == nullptr) {
    GELOGE(GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL, "[OptimizeOriginalGraph]: compute_graph is nullptr.");
    return GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL;
  }

  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "OptimizeOriginalGraph failed.");
    return GE_CLI_GE_NOT_INITIALIZED;
  }

  std::map<string, GraphOptimizerPtr> graph_optimizer = instance_ptr->OpsKernelManagerObj().GetAllGraphOptimizerObjs();
  GELOGI("optimize by opskernel in original graph optimize quantize phase. num of graph_optimizer is %zu.",
         graph_optimizer.size());
  Status ret = SUCCESS;
  string exclude_core_Type = (core_type_ == kVectorEngine) ? kAicoreEngine : kVectorEngine;
  GELOGD("[OptimizeOriginalGraphForQuantize]: engine type will exclude: %s", exclude_core_Type.c_str());
  if (graph_optimizer.size() != 0) {
    for (auto iter = graph_optimizer.begin(); iter != graph_optimizer.end(); ++iter) {
      if (iter->first == exclude_core_Type || iter->second == nullptr) {
        continue;
      }
      ret = iter->second->OptimizeGraphPrepare(*compute_graph);
      if (ret != SUCCESS) {
        GELOGE(ret, "[OptimizeOriginalGraphForQuantize]: graph optimize failed, ret:%u", ret);
        return ret;
      }
    }
  }
  return ret;
}

Status GraphOptimize::SetOptions(const ge::GraphManagerOptions &options) {
  if (options.framework_type >= static_cast<int32_t>(domi::FrameworkType::FMK_TYPE_RESERVED)) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "Optimize Type %d invalid.", options.framework_type);
    return GE_GRAPH_OPTIONS_INVALID;
  }
  optimize_type_ = static_cast<domi::FrameworkType>(options.framework_type);
  cal_config_ = options.calibration_conf_file;
  insert_op_config_ = options.insert_op_file;
  train_graph_flag_ = options.train_graph_flag;
  local_fmk_op_flag_ = options.local_fmk_op_flag;
  func_bin_path_ = options.func_bin_path;
  core_type_ = options.core_type;
  return SUCCESS;
}

void GraphOptimize::TranFrameOp(ComputeGraphPtr &compute_graph) {
  GE_CHECK_NOTNULL_JUST_RETURN(compute_graph);
  vector<string> local_framework_op_vec = {
    "TensorDataset", "QueueDataset", "DeviceQueueDataset", "ParallelMapDataset", "BatchDatasetV2",
    "IteratorV2",    "MakeIterator", "IteratorGetNext",    "FilterDataset",      "MapAndBatchDatasetV2"};
  for (auto &nodePtr : compute_graph->GetAllNodes()) {
    OpDescPtr op = nodePtr->GetOpDesc();
    GE_IF_BOOL_EXEC(op == nullptr, GELOGW("op is nullptr!"); continue);
    // fwkop black-white sheet
    vector<string>::iterator iter =
        std::find(local_framework_op_vec.begin(), local_framework_op_vec.end(), op->GetType());
    if (iter != local_framework_op_vec.end()) {
      // set - original_type
      if (!AttrUtils::SetStr(op, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, op->GetType())) {
        GELOGW("TranFrameOp SetStr ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE failed");
      }
      // set - framework_type
      // [No need to verify return value]
      op->SetType("FrameworkOp");
      if (!AttrUtils::SetInt(op, ATTR_NAME_FRAMEWORK_FWK_TYPE, domi::FrameworkType::FMK_TYPE_T)) {
        GELOGW("TranFrameOp SetInt ATTR_NAME_FRAMEWORK_FWK_TYPE failed");
      }
    }
  }
}
}  // namespace ge
