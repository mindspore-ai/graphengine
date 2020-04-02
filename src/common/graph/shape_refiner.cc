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

#include "graph/shape_refiner.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "debug/ge_log.h"
#include "debug/ge_op_types.h"
#include "external/graph/operator.h"
#include "external/graph/operator_factory.h"
#include "framework/common/debug/ge_log.h"
#include "graph/compute_graph.h"
#include "utils/node_utils.h"
#include "utils/op_desc_utils.h"
#include "utils/tensor_utils.h"
#include "utils/type_utils.h"

namespace ge {
void ShapeRefiner::PrintInOutTensorShape(const ge::NodePtr &node, const std::string &phase) {
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "node is null");
    return;
  }
  ge::OpDescPtr op_desc = node->GetOpDesc();
  GE_IF_BOOL_EXEC(op_desc == nullptr, GELOGE(GRAPH_FAILED, "op_desc is null."); return );
  std::string str;
  if (!op_desc->GetAllInputsDescPtr().empty()) {
    std::string input_desc_str = "input shape: ";
    for (const auto &input_desc : op_desc->GetAllInputsDescPtr()) {
      input_desc_str += "[";
      for (int64_t dim : input_desc->GetShape().GetDims()) {
        input_desc_str += std::to_string(dim) + " ";
      }
      input_desc_str += "]";
      input_desc_str += ":" + TypeUtils::DataTypeToSerialString(input_desc->GetDataType()) + ":" +
                        TypeUtils::FormatToSerialString(input_desc->GetFormat()) + " ";
    }
    str += input_desc_str;
  }

  if (!op_desc->GetAllOutputsDescPtr().empty()) {
    std::string output_desc_str = "output shape: ";
    for (const auto &output_desc : op_desc->GetAllOutputsDescPtr()) {
      if (output_desc == nullptr) {
        continue;
      }
      output_desc_str += "[";
      for (int64_t dim : output_desc->GetShape().GetDims()) {
        output_desc_str += std::to_string(dim) + " ";
      }
      output_desc_str += "]";
      output_desc_str += ":" + TypeUtils::DataTypeToSerialString(output_desc->GetDataType()) + ":" +
                         TypeUtils::FormatToSerialString(output_desc->GetFormat()) + " ";
    }
    str += output_desc_str;
  }
  GELOGD("Shape dump [%s], Node name: [%s]. %s", phase.c_str(), node->GetName().c_str(), str.c_str());
}

graphStatus ShapeRefiner::InferShapeAndType(const ConstNodePtr &node, Operator &op) {
  GE_IF_BOOL_EXEC(node == nullptr, GELOGE(GRAPH_FAILED, "node is null."); return GRAPH_FAILED);
  auto op_desc = node->GetOpDesc();
  GE_IF_BOOL_EXEC(op_desc == nullptr, GELOGE(GRAPH_FAILED, "op_desc is null."); return GRAPH_FAILED);
  const auto &op_type = op_desc->GetType();

  // Get infer func and execute
  graphStatus ret = op_desc->CallInferFunc(op);
  if (ret == GRAPH_PARAM_INVALID) {
    // Op ir no infer func, try to get infer func from operator factory
    auto node_op = ge::OperatorFactory::CreateOperator("node_op", op_desc->GetType());
    if (node_op.IsEmpty()) {
      GELOGW("get op from OperatorFactory fail. opType: %s", op_type.c_str());
      return ret;
    }

    GELOGD("get op from OperatorFactory success. opType: %s", op_type.c_str());
    auto temp_op_desc = ge::OpDescUtils::GetOpDescFromOperator(node_op);
    if (temp_op_desc == nullptr) {
      GELOGE(GRAPH_FAILED, "temp op desc is null");
      return GRAPH_FAILED;
    }
    if (!op_desc->UpdateInputName(temp_op_desc->GetAllInputName())) {
      GELOGW("InferShapeAndType UpdateInputName failed");
      for (const auto &out_desc : op_desc->GetAllOutputsDescPtr()) {
        if (out_desc != nullptr && out_desc->GetShape().GetDims().empty()) {
          break;
        }
        return GRAPH_SUCCESS;
      }
    }
    if (!op_desc->UpdateOutputName(temp_op_desc->GetAllOutputName())) {
      GELOGW("InferShapeAndType UpdateOutputName failed");
    }
    op_desc->AddInferFunc(temp_op_desc->GetInferFunc());
    ret = op_desc->CallInferFunc(op);
    GELOGI("op CallInferFunc second. ret: %u", ret);
  }
  return ret;
}

InferenceContextPtr CreateInferenceContext(const std::unordered_map<NodePtr, InferenceContextPtr> &context_map,
                                           const NodePtr &node) {
  if (node == nullptr) {
    GELOGE(GRAPH_FAILED, "node is null");
    return nullptr;
  }
  InferenceContextPtr inference_context = std::shared_ptr<InferenceContext>(InferenceContext::Create());
  if (inference_context == nullptr) {
    GELOGE(GRAPH_FAILED, "Failed to alloc InferenceContext");
    return nullptr;
  }

  auto all_in_data_anchors = node->GetAllInDataAnchors();
  std::vector<std::vector<ShapeAndType>> input_shapes_and_types(all_in_data_anchors.size());
  std::vector<std::string> marks;

  bool has_input_shapes_and_types = false;
  for (const auto &in_anchor : all_in_data_anchors) {
    const auto &out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      continue;
    }

    auto input_node = out_anchor->GetOwnerNode();
    if (input_node == nullptr) {
      continue;
    }

    auto iter = context_map.find(input_node);
    if (iter != context_map.end()) {
      const auto &src_context = iter->second;
      GE_IF_BOOL_EXEC(src_context == nullptr, GELOGE(GRAPH_FAILED, "src_context is null."); return nullptr);
      GELOGD("node:%s get %ld marks from node:%s", node->GetName().c_str(), src_context->GetMarks().size(),
             input_node->GetName().c_str());
      for (auto mark : src_context->GetMarks()) {
        marks.push_back(mark);
      }
      auto output_idx = out_anchor->GetIdx();
      auto input_idx = in_anchor->GetIdx();
      auto output_shape_and_type = src_context->GetOutputHandleShapesAndTypes();
      if (output_idx < static_cast<int>(output_shape_and_type.size())) {
        GELOGI("Add shape and type from %s:%d to %s:%d", input_node->GetName().c_str(), output_idx,
               node->GetName().c_str(), input_idx);
        input_shapes_and_types[input_idx] = output_shape_and_type[output_idx];
        has_input_shapes_and_types = true;
      } else {
        GELOGI("[%s] Output out of range. index = %d, size = %zu", node->GetName().c_str(), output_idx,
               output_shape_and_type.size());
      }
    }
  }

  if (has_input_shapes_and_types) {
    inference_context->SetInputHandleShapesAndTypes(std::move(input_shapes_and_types));
  }
  inference_context->SetMarks(marks);

  return inference_context;
}

namespace {
std::unordered_map<NodePtr, InferenceContextPtr> context_map;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ShapeRefiner::InferShapeAndType(const NodePtr &node) {
  GE_IF_BOOL_EXEC(node == nullptr, GELOGE(GRAPH_FAILED, "node is null."); return GRAPH_FAILED);
  if (node->Verify() != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Verifying %s failed.", node->GetName().c_str());
    return GRAPH_FAILED;
  }

  auto inference_context = CreateInferenceContext(context_map, node);
  if (inference_context == nullptr) {
    GELOGE(GRAPH_FAILED, "inference context is null");
    return GRAPH_FAILED;
  }

  GELOGD("create context for node:%s, marks %zu", node->GetName().c_str(), inference_context->GetMarks().size());

  PrintInOutTensorShape(node, "before_infershape");

  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  op.SetInferenceContext(inference_context);
  graphStatus status = InferShapeAndType(node, op);
  if (status == GRAPH_PARAM_INVALID || status == GRAPH_SUCCESS) {
    (void)ge::NodeUtils::UpdatePeerNodeInputDesc(node);
  } else {
    GELOGE(GRAPH_FAILED, "%s call infer function failed.", node->GetName().c_str());
    return GRAPH_FAILED;
  }

  auto ctx_after_infer = op.GetInferenceContext();
  if (ctx_after_infer != nullptr) {
    GELOGD("[%s] after infershape. mark:%zu", node->GetName().c_str(), ctx_after_infer->GetMarks().size());
    if (!ctx_after_infer->GetOutputHandleShapesAndTypes().empty() || !ctx_after_infer->GetMarks().empty()) {
      GELOGD("[%s] set inference context after. mark:%zu", node->GetName().c_str(), ctx_after_infer->GetMarks().size());
      (void)context_map.emplace(node, ctx_after_infer);
    }
  }

  PrintInOutTensorShape(node, "after_infershape");

  return GRAPH_SUCCESS;
}
}  // namespace ge
