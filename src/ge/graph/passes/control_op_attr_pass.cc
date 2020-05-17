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

#include "graph/passes/control_op_attr_pass.h"

#include <memory>
#include <string>
#include <vector>

#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "graph/common/omg_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "init/gelib.h"

using domi::ATTR_NAME_STREAM_LABEL;

using domi::STREAMACTIVE;
using domi::STREAMSWITCH;
using domi::STREAMSWITCHN;

namespace {
const uint32_t kMaxNodeNum = 350;
}  // namespace

namespace ge {
///
/// @brief Pass for Switch & Active Op attr
/// @param [in] graph
/// @return Status
///
Status ControlOpAttrPass::Run(ComputeGraphPtr graph) {
  GELOGD("ControlOpAttrPass Enter");

  if (AcquireEngineInfo() != SUCCESS) {
    GELOGE(FAILED, "AcquireEngineInfo fail.");
    return FAILED;
  }

  if (HandleStreamLabel(graph) != SUCCESS) {
    GELOGE(FAILED, "HandleStreamLabel fail.");
    return FAILED;
  }

  if (HandleSwitchNodes(graph) != SUCCESS) {
    GELOGE(FAILED, "HandleSwitchNodes fail.");
    return FAILED;
  }

  GELOGD("ControlOpAttrPass Leave");
  return SUCCESS;
}

///
/// @brief acquire engine info
/// @return Status
///
Status ControlOpAttrPass::AcquireEngineInfo() {
  auto gelib = GELib::GetInstance();
  if (gelib == nullptr) {
    GELOGE(INTERNAL_ERROR, "Get GELib instance failed.");
    return INTERNAL_ERROR;
  }

  const map<string, SchedulerConf> &scheduler_confs = gelib->DNNEngineManagerObj().GetSchedulers();
  for (const auto &item : scheduler_confs) {
    const SchedulerConf &scheduler = item.second;
    for (const auto &engine_pair : scheduler.cal_engines) {
      EngineConfPtr engine_conf = engine_pair.second;
      if (engine_conf != nullptr) {
        engine_confs_[engine_pair.first] = engine_conf;
      }
    }
  }

  return SUCCESS;
}

///
/// @brief Handle stream label
/// @param [in] graph
/// @return Status
///
Status ControlOpAttrPass::HandleStreamLabel(const ComputeGraphPtr &graph) {
  std::string stream_label;
  for (auto &node : graph->GetDirectNode()) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    const std::string type = op_desc->GetType();
    if ((type == STREAMSWITCH) || (type == STREAMSWITCHN)) {
      switch_nodes_.emplace_back(node);
    }

    if (!AttrUtils::GetStr(op_desc, ATTR_NAME_STREAM_LABEL, stream_label)) {
      continue;
    }

    auto num_iter = stream_label_num_.find(stream_label);
    if (num_iter == stream_label_num_.end()) {
      stream_label_num_[stream_label] = 1;
    } else {
      num_iter->second++;
    }

    bool independent = false;
    const std::string engine_name = op_desc->GetOpEngineName();
    if (!engine_name.empty()) {
      auto engine_conf_iter = engine_confs_.find(engine_name);
      bool exist_flag = (engine_conf_iter == engine_confs_.end()) || (engine_conf_iter->second == nullptr);
      if (exist_flag) {
        GELOGE(INTERNAL_ERROR, "Engine conf of node %s not found (engine name: %s).", op_desc->GetName().c_str(),
               engine_name.c_str());
        return INTERNAL_ERROR;
      }
      independent = engine_conf_iter->second->independent;
    }

    auto flag_iter = label_flag_.find(stream_label);
    if (flag_iter == label_flag_.end()) {
      label_flag_[stream_label] = independent ? std::make_pair(false, true) : std::make_pair(true, false);
    } else if (flag_iter->second.first && flag_iter->second.second) {
      continue;
    } else {
      bool &flag = (independent ? flag_iter->second.second : flag_iter->second.first);
      flag = true;
    }
  }

  return SUCCESS;
}

///
/// @brief Handle Switch Op
/// @param [in] graph
/// @return Status
///
Status ControlOpAttrPass::HandleSwitchNodes(ComputeGraphPtr &graph) {
  for (auto &switch_node : switch_nodes_) {
    GE_CHECK_NOTNULL(switch_node);
    std::vector<std::string> ori_active_label_list;
    OpDescPtr switch_desc = switch_node->GetOpDesc();
    GE_CHECK_NOTNULL(switch_desc);
    if (!AttrUtils::GetListStr(switch_desc, ATTR_NAME_ACTIVE_LABEL_LIST, ori_active_label_list) ||
        ori_active_label_list.empty()) {
      GELOGE(INTERNAL_ERROR, "active label of switch %s is null", switch_node->GetName().c_str());
      return INTERNAL_ERROR;
    }

    std::vector<std::string> active_label_list;
    std::vector<NodePtr> active_nodes;
    size_t label_num = ori_active_label_list.size();
    for (size_t i = 0; i < label_num; i++) {
      const std::string active_label = ori_active_label_list[i];
      if (!CheckNeedActiveNode(active_label)) {
        active_label_list.emplace_back(active_label);
        continue;
      }

      std::string name = switch_node->GetName() + "_" + STREAMACTIVE;
      if (label_num > 0) {
        name = name + "_" + std::to_string(i);
      }
      GELOGI("Create StreamActive op:%s.", name.c_str());
      OpDescPtr active_op_desc = MakeShared<OpDesc>(name, STREAMACTIVE);
      if (active_op_desc == nullptr) {
        GELOGE(FAILED, "Create node %s fail.", name.c_str());
        return FAILED;
      }
      NodePtr active_node = graph->AddNode(active_op_desc);
      if (active_node == nullptr) {
        GELOGE(FAILED, "Create StreamActive node fail.");
        return FAILED;
      }

      for (NodePtr &node : switch_node->GetOutControlNodes()) {
        std::string stream_label;
        OpDescPtr op_desc = node->GetOpDesc();
        GE_CHECK_NOTNULL(op_desc);
        (void)AttrUtils::GetStr(op_desc, ATTR_NAME_STREAM_LABEL, stream_label);
        if (stream_label != active_label) {
          continue;
        }
        GE_CHK_STATUS_RET(GraphUtils::RemoveEdge(switch_node->GetOutControlAnchor(), node->GetInControlAnchor()),
                          "remove edge failed");
        GE_CHK_STATUS_RET(GraphUtils::AddEdge(active_node->GetOutControlAnchor(), node->GetInControlAnchor()),
                          "add edge failed");
      }

      GE_CHK_STATUS_RET(SetSwitchBranchNodeLabel(active_node, name), "set switch branch node label failed");
      GE_CHK_STATUS_RET(SetStreamLabel(active_node, name), "set stream label failed");
      GE_CHK_STATUS_RET(SetActiveLabelList(active_node, {active_label}), "set active label list failed");

      active_nodes.emplace_back(active_node);
      active_label_list.emplace_back(name);
    }

    GE_CHK_STATUS_RET(SetActiveLabelList(switch_node, {active_label_list}), "set active label list failed");

    if (active_nodes.empty()) {
      continue;
    }

    if (!switch_node->GetOutAllNodes().empty()) {
      GELOGE(FAILED, "Exist out_node holds stream_label beyond the range of active_label_list, switch_node:%s.",
             switch_desc->GetName().c_str());
      return FAILED;
    }
    for (auto &active_node : active_nodes) {
      GE_CHK_STATUS_RET(GraphUtils::AddEdge(switch_node->GetOutControlAnchor(), active_node->GetInControlAnchor()),
                        "add edge failed");
    }
  }

  return SUCCESS;
}

///
/// @brief Check if insert active node
/// @param [in] stream_label
/// @return bool
///
bool ControlOpAttrPass::CheckNeedActiveNode(const std::string &stream_label) {
  if (stream_label_num_[stream_label] > kMaxNodeNum) {
    return true;
  }

  auto iter = label_flag_.find(stream_label);
  if (iter == label_flag_.end()) {
    GELOGE(INTERNAL_ERROR, "not find label %s", stream_label.c_str());
    return false;
  }
  if (iter->second.first && iter->second.second) {
    return true;
  }

  return false;
}
}  // namespace ge
