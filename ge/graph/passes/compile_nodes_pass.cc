/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "graph/passes/compile_nodes_pass.h"

#include <utility>
#include <vector>

#include "common/ge/ge_util.h"
#include "common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/common/ge_call_wrapper.h"
#include "graph/op_desc.h"

using domi::ImplyType;

namespace {
const char *const kAICPUEngineName = "DNN_VM_AICPU";
const char *const kAICPUKernelLibName = "aicpu_tf_kernel";
}  // namespace

namespace ge {
graphStatus CompileNodesPass::Run(ComputeGraphPtr graph) {
  GE_TIMESTAMP_START(CompileNodesPass);
  GELOGI("[CompileNodesPass]: optimize begin.");
  if (graph == nullptr) {
    return GRAPH_SUCCESS;
  }
  std::shared_ptr<GELib> instance = ge::GELib::GetInstance();
  if (instance == nullptr || !instance->InitFlag()) {
    GELOGE(ge::GE_CLI_GE_NOT_INITIALIZED, "Run CompileNodesPass failed.");
    return ge::GE_CLI_GE_NOT_INITIALIZED;
  }
  std::unordered_map<string, vector<NodePtr>> kernel_to_compile_nodes;
  for (auto &node : graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    auto node_need_compile = false;
    (void)ge::AttrUtils::GetBool(op_desc, ATTR_NEED_COMPILE, node_need_compile);
    if (!node_need_compile) {
      continue;
    }
    // collect all supported compile node
    string kernel_lib_name;
    auto ret = GetSupportedKernel(node, instance, kernel_lib_name);
    if (ret == GRAPH_SUCCESS) {
      auto iter = kernel_to_compile_nodes.find(kernel_lib_name);
      if (iter != kernel_to_compile_nodes.end()) {
        iter->second.emplace_back(node);
      } else {
        std::vector<NodePtr> node_vec{node};
        kernel_to_compile_nodes.insert(std::make_pair(kernel_lib_name, node_vec));
      }
    } else {
      GELOGE(GRAPH_FAILED, "Get node:%s, type:%s supported kernel failed.", node->GetName().c_str(),
             node->GetType().c_str());
      return GRAPH_FAILED;
    }
  }
  // compile node follow different kernel, currently only TBE kernel
  auto result = CompileNodes(instance, kernel_to_compile_nodes);
  if (result != GRAPH_SUCCESS) {
    GELOGE(result, "Compile op failed.");
    return result;
  }
  GELOGI("[CompileNodesPass]: Optimize success.");
  GE_TIMESTAMP_EVENT_END(CompileNodesPass, "OptimizeStage2::ControlAttrOptimize::CompileNodesPass");
  return GRAPH_SUCCESS;
}

graphStatus CompileNodesPass::GetSupportedKernel(const NodePtr &node, const std::shared_ptr<GELib> instance,
                                                 string &kernel_lib_name) {
  auto op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(ge::GE_GRAPH_PARAM_NULLPTR, "Get op %s opdesc failed", node->GetName().c_str());
    return ge::GE_GRAPH_PARAM_NULLPTR;
  }
  // reset op kernel lib, find supported kernel
  kernel_lib_name = op_desc->GetOpKernelLibName();
  if (kernel_lib_name.empty()) {
    (void)instance->DNNEngineManagerObj().GetDNNEngineName(node);
    kernel_lib_name = op_desc->GetOpKernelLibName();
    if (kernel_lib_name.empty()) {
      GELOGE(GRAPH_FAILED, "Get node:%s, type:%s kernel lib failed.", node->GetName().c_str(),
             op_desc->GetType().c_str());
      return GRAPH_FAILED;
    }
  }
  OpsKernelInfoStorePtr kernel_info = instance->OpsKernelManagerObj().GetOpsKernelInfoStore(kernel_lib_name);
  if (kernel_info == nullptr) {
    GELOGE(ge::GE_GRAPH_PARAM_NULLPTR, "Get op %s ops kernel info store failed", node->GetName().c_str());
    return ge::GE_GRAPH_PARAM_NULLPTR;
  }
  // begin accuracy supported check
  if (!CheckAccuracySupport(kernel_info, instance, op_desc)) {
    // if check accuracy support failed , try to go to aicpu engine
    string aicpu_kernel_lib_name = kAICPUKernelLibName;
    OpsKernelInfoStorePtr aicpu_kernel_info =
      instance->OpsKernelManagerObj().GetOpsKernelInfoStore(aicpu_kernel_lib_name);
    if (aicpu_kernel_info == nullptr) {
      GELOGE(ge::GE_GRAPH_PARAM_NULLPTR, "Get aicpu kernel info store failed.");
      return ge::GE_GRAPH_PARAM_NULLPTR;
    }
    if (!CheckAccuracySupport(aicpu_kernel_info, instance, op_desc)) {
      GELOGE(GRAPH_FAILED, "AICPU engine does not support node:%s, type:%s , get kernel lib failed.",
             node->GetName().c_str(), op_desc->GetType().c_str());
      return GRAPH_FAILED;
    }
    kernel_lib_name = kAICPUKernelLibName;
  }
  return GRAPH_SUCCESS;
}

bool CompileNodesPass::CheckAccuracySupport(const OpsKernelInfoStorePtr &kernel_info,
                                            const std::shared_ptr<GELib> instance, OpDescPtr &op_desc) {
  auto ge_desc = MakeShared<ge::OpDescPtr>(op_desc);
  if (ge_desc == nullptr) {
    GELOGE(GE_GRAPH_MEMORY_ALLOC_FAILED, "Fail to malloc op desc.");
    return false;
  }
  string reason;
  if (!(kernel_info->CheckAccuracySupported(*ge_desc, reason, true))) {
    GELOGW("Check Accuracy Supported return not support, node name is %s, reason: %s. Try to go to AICPU engine.",
           op_desc->GetName().c_str(), reason.c_str());
    return false;
  }
  return true;
}

graphStatus CompileNodesPass::CompileNodes(const std::shared_ptr<GELib> instance,
                                           std::unordered_map<string, vector<NodePtr>> &kernel_to_compile_nodes) {
  // compile nodes, if kernel is aicpu, check support and set engine info.
  OpsKernelInfoStorePtr kernel_info;
  for (auto &kernel_nodes : kernel_to_compile_nodes) {
    kernel_info = instance->OpsKernelManagerObj().GetOpsKernelInfoStore(kernel_nodes.first);
    if (kernel_info == nullptr) {
      GELOGE(ge::GE_GRAPH_PARAM_NULLPTR, "Get op %s ops kernel info store failed", kernel_nodes.first.c_str());
      return ge::GE_GRAPH_PARAM_NULLPTR;
    }
    string reason;
    if (kernel_nodes.first == kAICPUKernelLibName) {
      for (auto node : kernel_nodes.second) {
        // this node will go to aicpu engine ,no need compile
        node->GetOpDesc()->SetOpEngineName(kAICPUEngineName);
        node->GetOpDesc()->SetOpKernelLibName(kAICPUKernelLibName);
        AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(ImplyType::AI_CPU));
      }
      continue;
    }
    auto ret = kernel_info->CompileOp(kernel_nodes.second);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(ret, "Compile op failed, kernel name is %s", kernel_nodes.first.c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}
}  // namespace ge
