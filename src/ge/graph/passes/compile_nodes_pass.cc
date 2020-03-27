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

#include "graph/passes/compile_nodes_pass.h"

#include <utility>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "common/ge_inner_error_codes.h"
#include "common/ge/ge_util.h"
#include "graph/op_desc.h"
#include "graph/debug/ge_attr_define.h"

namespace {
const char *const kAICPUEngineName = "DNN_VM_AICPU";
const char *const kAICPUKernelLibName = "aicpu_kernel";
}  // namespace

namespace ge {
graphStatus CompileNodesPass::CompileOp(NodePtr node,
                                        const std::shared_ptr<GELib> &instance,
                                        const string &kernel_lib_name) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(instance);
  OpsKernelInfoStorePtr kernel_info = instance->OpsKernelManagerObj().GetOpsKernelInfoStore(kernel_lib_name);
  if (kernel_info == nullptr) {
    GELOGE(ge::GE_GRAPH_PARAM_NULLPTR, "Get op %s ops kernel info store failed", node->GetName().c_str());
    return ge::GE_GRAPH_PARAM_NULLPTR;
  }

  // check if support
  auto op_desc = node->GetOpDesc();
  auto ge_desc = MakeShared<ge::OpDescPtr>(op_desc);
  if (ge_desc == nullptr) {
    GELOGE(GE_GRAPH_MEMORY_ALLOC_FAILED, "Fail to malloc op desc.");
    return FAILED;
  }
  string reason;
  if (!(kernel_info->CheckAccuracySupported(*ge_desc, reason, true))) {
    GELOGW("Check Accuracy Supported failed, go to aicpu engine, node name is %s, reason: %s", node->GetName().c_str(),
           reason.c_str());
    op_desc->SetOpEngineName(kAICPUEngineName);
    op_desc->SetOpKernelLibName(kAICPUKernelLibName);
  } else {
    // TBE compile op
    vector<ge::NodePtr> node_vec = {node};
    auto ret = kernel_info->CompileOp(node_vec);
    if (ret != ge::SUCCESS) {
      GELOGE(ret, "Compile single op failed, node name is %s", node->GetName().c_str());
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus CompileNodesPass::CompileNode(const NodePtr &node, const std::shared_ptr<GELib> &instance) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(instance);
  auto op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(ge::GE_GRAPH_PARAM_NULLPTR, "Get op %s opdesc failed", node->GetName().c_str());
    return ge::GE_GRAPH_PARAM_NULLPTR;
  }
  string kernel_lib_name = op_desc->GetOpKernelLibName();
  if (kernel_lib_name.empty()) {
    // reset op kernel lib
    (void)instance->DNNEngineManagerObj().GetDNNEngineName(op_desc);
    kernel_lib_name = op_desc->GetOpKernelLibName();
    if (kernel_lib_name.empty()) {
      GELOGE(GRAPH_FAILED, "Get node:%s, type:%s kernel lib failed.", node->GetName().c_str(),
             op_desc->GetType().c_str());
      return GRAPH_FAILED;
    }
  }

  return CompileOp(node, instance, kernel_lib_name);
}

graphStatus CompileNodesPass::Run(ComputeGraphPtr graph) {
  GELOGI("[CompileNodesPass]: optimize begin.");
  if (graph == nullptr) {
    return GRAPH_SUCCESS;
  }

  std::shared_ptr<GELib> instance = ge::GELib::GetInstance();
  if (instance == nullptr || !instance->InitFlag()) {
    GELOGE(ge::GE_CLI_GE_NOT_INITIALIZED, "Run CompileNodesPass failed.");
    return ge::GE_CLI_GE_NOT_INITIALIZED;
  }

  for (auto &node : graph->GetAllNodes()) {
    if (node == nullptr) {
      continue;
    }
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    auto node_need_compile = false;
    (void) ge::AttrUtils::GetBool(op_desc, ATTR_NEED_COMPILE, node_need_compile);
    if (!node_need_compile) {
      continue;
    }

    auto ret = CompileNode(node, instance);
    if (ret != GRAPH_SUCCESS) {
      return ret;
    }
  }

  GELOGI("[CompileNodesPass]: Optimize success.");
  return GRAPH_SUCCESS;
}
}  // namespace ge
