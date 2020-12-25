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

#include "graph/passes/dynamic_single_op_reset_shape_pass.h"
#include "common/ge_inner_error_codes.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
namespace {
const int64_t kDynamicShapeDim = -2;
const char *const kEngineNameAiCpu = "DNN_VM_AICPU_ASCEND";
const char *const kEngineNameAiCpuTf = "DNN_VM_AICPU";
}  // namespace
Status DynamicSingleOpResetShapePass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);

  std::shared_ptr<GELib> instance = ge::GELib::GetInstance();
  if (instance == nullptr || !instance->InitFlag()) {
    GELOGE(ge::GE_CLI_GE_NOT_INITIALIZED, "Run CompileNodesPass failed.");
    return ge::GE_CLI_GE_NOT_INITIALIZED;
  }

  // pass if graph has not aicpu node.
  bool is_not_aicpu = false;
  if (CheckAllAicpuNodes(graph, is_not_aicpu) != SUCCESS) {
    GELOGE(ge::GE_CLI_GE_NOT_INITIALIZED, "Check if graph has not aicpu node failed.");
    return ge::GE_CLI_GE_NOT_INITIALIZED;
  }
  if (is_not_aicpu) {
    GELOGI("The graph [%s] has not aicpu node, whose aicpu nodes would not be reset dynamic shape",
           graph->GetName().c_str());
    return SUCCESS;
  }

  for (const auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    // pass input and output node
    if (node->GetType() == DATA || node->GetType() == CONSTANT || node->GetType() == CONSTANTOP ||
        node->GetType() == NETOUTPUT) {
      continue;
    }

    // pass node without attr: ATTR_DYNAMIC_SHAPE_SINGLE_AICPU
    bool single_aicpu_unknown = false;
    if (!AttrUtils::GetBool(node->GetOpDesc(), ATTR_DYNAMIC_SHAPE_SINGLE_AICPU, single_aicpu_unknown) ||
        !single_aicpu_unknown) {
      continue;
    }

    // reset aicpu shape to unknown shape
    auto op_desc = node->GetOpDesc();
    if (ResetOpShape(op_desc) != SUCCESS) {
      GELOGE(ge::GE_CLI_GE_NOT_INITIALIZED, "Reset node[%s] dynamic shapr failed.", node->GetName().c_str());
      return ge::GE_CLI_GE_NOT_INITIALIZED;
    }
    GELOGD("Reset dynamic aicpu node [%s] shape success!", node->GetName().c_str());
  }

  GELOGD("Reset dynamic aicpu nodes shape of graph [%s] success!", graph->GetName().c_str());
  return SUCCESS;
}

Status DynamicSingleOpResetShapePass::CheckAllAicpuNodes(const ComputeGraphPtr &graph, bool &is_not_aicpu) {
  is_not_aicpu = false;
  for (const auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    // pass input and output node
    if (node->GetType() == DATA || node->GetType() == CONSTANT || node->GetType() == CONSTANTOP ||
        node->GetType() == NETOUTPUT) {
      continue;
    }

    // find if there are aicpu nodes.
    auto op_desc = node->GetOpDesc();
    string engine_name = op_desc->GetOpEngineName();
    if (engine_name.empty()) {
      GELOGE(GRAPH_FAILED, "Get engine failed of node[%s].", node->GetName().c_str());
      return GRAPH_FAILED;
    }
    if (engine_name != kEngineNameAiCpu && engine_name != kEngineNameAiCpuTf) {
      is_not_aicpu = true;
      return SUCCESS;
    }
  }
  return SUCCESS;
}

bool DynamicSingleOpResetShapePass::CheckIfConstInput(const GeTensorDescPtr &input_tensor_desc) {
  bool is_const = false;
  (void)AttrUtils::GetBool(input_tensor_desc, CONST_ATTR_NAME_INPUT, is_const);
  return is_const;
}

Status DynamicSingleOpResetShapePass::ResetOpShape(OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(op_desc);
  std::vector<int64_t> dynamic_shape_dims = {kDynamicShapeDim};
  GeShape dynamic_shape(dynamic_shape_dims);
  bool reset_shape_flag = false;
  if (ResetInputTensorShape(op_desc, dynamic_shape, reset_shape_flag) == SUCCESS && reset_shape_flag) {
    (void)ResetOutputTensorShape(op_desc, dynamic_shape);
  }
  return SUCCESS;
}

Status DynamicSingleOpResetShapePass::ResetInputTensorShape(OpDescPtr &op_desc, const GeShape &dynamic_shape,
                                                            bool &reset_shape_flag) {
  reset_shape_flag = false;
  GE_CHECK_NOTNULL(op_desc);
  for (size_t i = 0; i < op_desc->GetAllInputsDesc().size(); i++) {
    auto input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    GE_CHECK_NOTNULL(input_desc);
    // pass scalar input desc
    auto dims_ori = input_desc->GetShape().GetDims();
    if (dims_ori.size() == 0) {
      continue;
    }
    // pass const input
    if (CheckIfConstInput(input_desc)) {
      continue;
    }
    reset_shape_flag = true;
    input_desc->SetShape(dynamic_shape);
  }
  return SUCCESS;
}

Status DynamicSingleOpResetShapePass::ResetOutputTensorShape(OpDescPtr &op_desc, const GeShape &dynamic_shape) {
  GE_CHECK_NOTNULL(op_desc);
  for (size_t i = 0; i < op_desc->GetAllOutputsDesc().size(); i++) {
    auto output_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(i));
    GE_CHECK_NOTNULL(output_desc);
    // pass scalar input desc
    auto output_dims_ori = output_desc->GetShape().GetDims();
    if (output_dims_ori.size() == 0) {
      continue;
    }
    output_desc->SetShape(dynamic_shape);
  }
  return SUCCESS;
}
}  // namespace ge