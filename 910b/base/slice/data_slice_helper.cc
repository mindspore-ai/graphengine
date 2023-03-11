/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "slice/data_slice_helper.h"
#include "slice/data_slice_factory.h"
#include "graph/operator_factory_impl.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/debug/ge_log.h"
#include "slice/data_slice_adapter.h"

namespace ge {
Status DataSliceHelper::SetInputSlice(OpDescPtr &op, const AxisTypeInfo &slice_info, DataSliceType &input_slice)
{
  if (input_slice.size() == slice_info.GetRelateInputs().size()) {
    for (size_t tensor_slice_idx = 0; tensor_slice_idx < input_slice.size(); tensor_slice_idx++) {
      int64_t tensor_idx = slice_info.GetRelateInputs()[tensor_slice_idx].first;
      size_t input_size = op->GetAllInputsSize();
      if (tensor_idx >= static_cast<int64_t>(input_size)) {
        GELOGE(FAILED, "[DataSlice][Status] node %s cannot find cut tensor index.", op->GetName().c_str());
        return FAILED;
      }
      GeTensorDesc tensor_desc = op->GetInputDesc(slice_info.GetRelateInputs()[tensor_slice_idx].first);
      (void)AttrUtils::SetListListInt(tensor_desc, ATTR_NAME_DATA_SLICE, input_slice[tensor_slice_idx]);
      op->UpdateInputDesc(slice_info.GetRelateInputs()[tensor_slice_idx].first, tensor_desc);
    }
  }
  return SUCCESS;
}

// infer input slice by output slice
Status DataSliceHelper::InferAxisSlice(OpDescPtr &op, const AxisTypeInfo &slice_info)
{
  DataSliceType output_slice;
  for (const auto &tensor_slice : slice_info.GetRelateOutputs()) {
    GeTensorDesc tensor_desc = op->GetOutputDesc(tensor_slice.first);
    std::vector<std::vector<int64_t>> infer_range_vec_res;
    (void)AttrUtils::GetListListInt(tensor_desc, ATTR_NAME_DATA_SLICE, infer_range_vec_res);
    output_slice.emplace_back(infer_range_vec_res);
  }
  // call register to get special op infer func
  auto node_slice_infer_ptr = OperatorFactoryImpl::GetInferAxisSliceFunc(op->GetType());
  if (node_slice_infer_ptr != nullptr) {
    GELOGD("[DataSlice][Status] special node %s start infer axis slice", op->GetName().c_str());
    DataSliceType input_slice;
    Operator op_proxy = OpDescUtils::CreateOperatorFromOpDesc(op);
    const graphStatus ret = static_cast<graphStatus>(node_slice_infer_ptr(op_proxy,
        slice_info, output_slice, input_slice));
    if (ret != GRAPH_SUCCESS) {
      GELOGE(FAILED, "[DataSlice][Status]special node %s infer axis slice failed", op->GetName().c_str());
      return FAILED;
    }
    op_proxy.BreakConnect();
    if (SetInputSlice(op, slice_info, input_slice) != SUCCESS) {
      GELOGE(FAILED, "[DataSlice][Status]special node %s set axis slice failed", op->GetName().c_str());
      return FAILED;
    }
    return SUCCESS;
  }
  // call data slice factory to get axis infer func
  auto data_slice_infer_ptr = DataSliceFactory::GetInstance()->GetClassByAxisType(slice_info.GetAxisType());
  if (data_slice_infer_ptr != nullptr) {
    DataSliceType input_slice;
    Operator op_proxy = OpDescUtils::CreateOperatorFromOpDesc(op);
    if (data_slice_infer_ptr->InferAxisSlice(op_proxy, slice_info, output_slice, input_slice) != SUCCESS) {
      GELOGE(FAILED, "[DataSlice][Check] node: %s InferAxisSlice failed", op->GetName().c_str());
      return FAILED;
    }
    op_proxy.BreakConnect();
    if (SetInputSlice(op, slice_info, input_slice) != SUCCESS) {
      GELOGE(FAILED, "[DataSlice][Status] node %s set axis slice failed", op->GetName().c_str());
      return FAILED;
    }
    return SUCCESS;
  }
  return FAILED;
}

// get op axis slice info
Status DataSliceHelper::GetSliceInfo(OpDescPtr &op, std::vector<AxisTypeInfo> &axis_type_vec)
{
  // call register to get axis slice info
  auto axis_slice_info_ptr = OperatorFactoryImpl::GetInferAxisTypeInfoFunc(op->GetType());
  if (axis_slice_info_ptr == nullptr) {
    GELOGW("[DataSlice][Check] node: %s has no axis slice func.", op->GetName().c_str());
    return FAILED;
  }
  GELOGD("[DataSlice][Status] node %s get axis type info.", op->GetName().c_str());
  Operator op_proxy = OpDescUtils::CreateOperatorFromOpDesc(op);
  const graphStatus ret = static_cast<graphStatus>(axis_slice_info_ptr(op_proxy, axis_type_vec));
  if (ret != GRAPH_SUCCESS) {
    GEEVENT("[DataSlice][Status] node %s get axis slice failed", op->GetName().c_str());
    return FAILED;
  }
  op_proxy.BreakConnect();
  return SUCCESS;
}

Status DataSliceHelper::GetSliceInfo(const NodePtr &node, std::vector<AxisTypeInfo> &axis_type_vec)
{
  // call register to get axis slice info
  auto axis_slice_info_ptr = OperatorFactoryImpl::GetInferAxisTypeInfoFunc(node->GetType());
  if (axis_slice_info_ptr == nullptr) {
    GELOGW("[DataSlice][Check] node: %s has no axis slice func.", node->GetName().c_str());
    return FAILED;
  }
  GELOGD("[DataSlice][Status] node %s get axis type info.", node->GetName().c_str());
  Operator op_proxy = OpDescUtils::CreateOperatorFromNode(node);
  const graphStatus ret = static_cast<graphStatus>(axis_slice_info_ptr(op_proxy, axis_type_vec));
  if (ret != GRAPH_SUCCESS) {
    GEEVENT("[DataSlice][Status] node %s get axis slice failed", node->GetName().c_str());
    return FAILED;
  }
  op_proxy.BreakConnect();
  return SUCCESS;
}

Status DataSliceHelper::InferDavinciSpecialOpSlice(OpDescPtr &op, const AxisTypeInfo &slice_info,
    const InferAxisSliceFunc &node_slice_infer_ptr)
{
  Operator op_proxy;
  DataSliceType ori_input_slice;
  DataSliceType ori_output_slice;
  if (DataSliceAdapter::GetOriOutputSlice(op, slice_info, ori_output_slice) != SUCCESS) {
    GELOGE(FAILED, "[DataSlice][Status] special node %s GetOriOutputSlice failed", op->GetName().c_str());
    return FAILED;
  }
  AxisTypeInfo tmp_axis_type_info = DataSliceAdapter::GetTmpAxisTypeInfo(slice_info);
  bool valid_ori_info = DataSliceAdapter::CheckOriInfo(op);
  if (valid_ori_info) {
    std::vector<std::pair<Format, GeShape>> cache_input_info;
    std::vector<std::pair<Format, GeShape>> cache_output_info;
    DataSliceAdapter::SetOriOpInfo(op, cache_input_info, cache_output_info);
    op_proxy = OpDescUtils::CreateOperatorFromOpDesc(op);
    GELOGD("[DataSlice][Status] special node %s start infer axis slice", op->GetName().c_str());
    const graphStatus ret = static_cast<graphStatus>(node_slice_infer_ptr(op_proxy, tmp_axis_type_info,
        ori_output_slice, ori_input_slice));
    DataSliceAdapter::SetCurOpInfo(op, cache_input_info, cache_output_info);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(FAILED, "[DataSlice][Status]special node %s infer axis slice failed", op->GetName().c_str());
      return FAILED;
    }
  } else {
    GELOGE(FAILED, "[DataSlice][Check] node %s check ori_shape failed", op->GetName().c_str());
    return FAILED;
  }
  op_proxy.BreakConnect();
  DataSliceType cur_input_slice;
  if (DataSliceAdapter::GetCurInputSlice(op, slice_info, ori_input_slice, cur_input_slice) != SUCCESS) {
    GELOGE(FAILED, "[DataSlice][Status] special node %s GetCurInputSlice failed", op->GetName().c_str());
    return FAILED;
  }
  if (SetInputSlice(op, slice_info, cur_input_slice) != SUCCESS) {
    GELOGE(FAILED, "[DataSlice][Status]special node %s set axis slice failed", op->GetName().c_str());
      return FAILED;
  }
  return SUCCESS;
}

Status DataSliceHelper::InferDavinciCommonOpSlice(OpDescPtr &op, const AxisTypeInfo &slice_info)
{
  auto data_slice_infer_ptr = DataSliceFactory::GetInstance()->GetClassByAxisType(slice_info.GetAxisType());
  if (data_slice_infer_ptr == nullptr) {
    return FAILED;
  }
  Operator op_proxy;
  DataSliceType ori_input_slice;
  DataSliceType ori_output_slice;
  if (DataSliceAdapter::GetOriOutputSlice(op, slice_info, ori_output_slice) != SUCCESS) {
    GELOGE(FAILED, "[DataSlice][Status] special node %s GetOriOutputSlice failed", op->GetName().c_str());
    return FAILED;
  }
  AxisTypeInfo tmp_axis_type_info = DataSliceAdapter::GetTmpAxisTypeInfo(slice_info);
  bool valid_ori_info = DataSliceAdapter::CheckOriInfo(op);
  if (valid_ori_info) {
    std::vector<std::pair<Format, GeShape>> cache_input_info;
    std::vector<std::pair<Format, GeShape>> cache_output_info;
    DataSliceAdapter::SetOriOpInfo(op, cache_input_info, cache_output_info);
    op_proxy = OpDescUtils::CreateOperatorFromOpDesc(op);
    GELOGD("[DataSlice][Status] node %s start infer axis slice", op->GetName().c_str());
    auto ret = data_slice_infer_ptr->InferAxisSlice(op_proxy, tmp_axis_type_info, ori_output_slice, ori_input_slice);
    DataSliceAdapter::SetCurOpInfo(op, cache_input_info, cache_output_info);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[DataSlice][Check] node: %s InferAxisSlice failed", op->GetName().c_str());
      return FAILED;
    }
  } else {
    GELOGE(FAILED, "[DataSlice][Check] node %s check ori_shape failed", op->GetName().c_str());
    return FAILED;
  }
  op_proxy.BreakConnect();
  DataSliceType cur_input_slice;
  if (DataSliceAdapter::GetCurInputSlice(op, slice_info, ori_input_slice, cur_input_slice) != SUCCESS) {
    GELOGE(FAILED, "[DataSlice][Status] special node %s GetCurInputSlice failed", op->GetName().c_str());
    return FAILED;
  }
  if (SetInputSlice(op, slice_info, cur_input_slice) != SUCCESS) {
    GELOGE(FAILED, "[DataSlice][Status] node %s set axis slice failed", op->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

// infer input slice by output slice with format transformation
Status DataSliceHelper::InferDavinciAxisSlice(OpDescPtr &op, const AxisTypeInfo &slice_info)
{
  DataSliceAdapter::PrintOp(op);
  DataSliceAdapter::PrintAxis(op, {slice_info}, "current", true);

  // call register to get special op infer func
  auto node_slice_infer_ptr = OperatorFactoryImpl::GetInferAxisSliceFunc(op->GetType());
  if (node_slice_infer_ptr != nullptr) {
    return InferDavinciSpecialOpSlice(op, slice_info, node_slice_infer_ptr);
  }
  return InferDavinciCommonOpSlice(op, slice_info);
}

// get axis_type_info with currnet format
Status DataSliceHelper::GetDavinciSliceInfo(const NodePtr &node, std::vector<AxisTypeInfo> &axis_type_vec)
{
  // call register to get axis slice info
  auto axis_slice_info_ptr = OperatorFactoryImpl::GetInferAxisTypeInfoFunc(node->GetType());
  if (axis_slice_info_ptr == nullptr) {
    GELOGW("[DataSlice][Check] node: %s has no axis slice func.", node->GetName().c_str());
    return FAILED;
  }
  GELOGD("[DataSlice][Status] node %s get axis type info.", node->GetName().c_str());

  Operator op_proxy;
  OpDescPtr op = node->GetOpDesc();
  bool valid_ori_info = DataSliceAdapter::CheckOriInfo(op);
  if (valid_ori_info) {
    std::vector<std::pair<Format, GeShape>> cache_input_info;
    std::vector<std::pair<Format, GeShape>> cache_output_info;
    DataSliceAdapter::SetOriOpInfo(op, cache_input_info, cache_output_info);
    op_proxy = OpDescUtils::CreateOperatorFromNode(node);
    const graphStatus ret = static_cast<graphStatus>(axis_slice_info_ptr(op_proxy, axis_type_vec));
    DataSliceAdapter::SetCurOpInfo(op, cache_input_info, cache_output_info);
    if (ret != GRAPH_SUCCESS) {
      GEEVENT("[DataSlice][Status] node %s get axis slice failed", node->GetName().c_str());
      return FAILED;
    }
  } else {
    axis_type_vec.clear();
    GEEVENT("[DataSlice][Status] node %s check ori_shape failed, clear axis_type_vec", node->GetName().c_str());
    return SUCCESS;
  }
  op_proxy.BreakConnect();
  DataSliceAdapter::PrintOp(op);
  DataSliceAdapter::PrintAxis(op, axis_type_vec, "origin", false);
  DataSliceAdapter::TransAxisInfo(op, axis_type_vec);
  DataSliceAdapter::PrintAxis(op, axis_type_vec, "current", false);
  return SUCCESS;
}
}
