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

#include "graph/load/new_model_manager/model_utils.h"

#include <string>

#include "common/debug/log.h"
#include "common/op/ge_op_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "runtime/base.h"
#include "runtime/kernel.h"

#include "framework/common/debug/ge_log.h"
#include "graph/manager/graph_var_manager.h"

namespace ge {
///
/// @ingroup ge
/// @brief Get input size.
/// @return vector<uint32_t>
///
vector<int64_t> ModelUtils::GetInputSize(ConstOpDescPtr op_desc) {
  vector<int64_t> v_input_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_input_size);
  const size_t inputs_size = op_desc->GetInputsSize();
  const string op_type = op_desc->GetType();

  const vector<bool> v_is_input_const = op_desc->GetIsInputConst();
  for (size_t i = 0; i < inputs_size; ++i) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(i);
    if (tensor_desc == nullptr) {
      GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }

    int64_t tensor_size = 0;
    if ((i < v_is_input_const.size()) && v_is_input_const[i] && (op_type != NETOUTPUT)) {
      // TBE: add weights size to input
      GE_CHK_STATUS(TensorUtils::GetSize(*tensor_desc, tensor_size));
      if (tensor_size) {
        v_input_size.push_back(tensor_size);
      }
      continue;
    }

    GE_IF_BOOL_EXEC(
      TensorUtils::GetSize(*tensor_desc, tensor_size) != GRAPH_SUCCESS,
      GELOGI("Get size from TensorDesc failed, op : %s, input index : %zu", op_desc->GetName().c_str(), i);
      continue);

    v_input_size.push_back(tensor_size);
  }

  return v_input_size;
}

///
/// @ingroup ge
/// @brief Get output size.
/// @return vector<uint32_t>
///
vector<int64_t> ModelUtils::GetOutputSize(ConstOpDescPtr op_desc) {
  vector<int64_t> v_output_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_output_size);

  const size_t outputs_size = op_desc->GetOutputsSize();
  const vector<int64_t> v_output_offset = op_desc->GetOutputOffset();
  GE_IF_BOOL_EXEC(v_output_offset.size() != outputs_size,
                  GELOGW("Output param invalid: output_offset=%zu, outputs=%zu.", v_output_offset.size(), outputs_size);
                  return v_output_size;);

  for (size_t i = 0; i < outputs_size; ++i) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableOutputDesc(i);
    if (tensor_desc == nullptr) {
      GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }

    int64_t tensor_size = 0;
    GE_IF_BOOL_EXEC(
      TensorUtils::GetSize(*tensor_desc, tensor_size) != GRAPH_SUCCESS,
      GELOGI("Get size from TensorDesc failed, op : %s, output index : %zu", op_desc->GetName().c_str(), i);
      continue);

    v_output_size.push_back(tensor_size);
  }

  return v_output_size;
}

///
/// @ingroup ge
/// @brief Get workspace size.
/// @return vector<uint32_t>
///
vector<int64_t> ModelUtils::GetWorkspaceSize(ConstOpDescPtr op_desc) {
  vector<int64_t> v_workspace_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_workspace_size);

  const vector<int64_t> v_workspace_num = op_desc->GetWorkspace();
  const vector<int64_t> v_workspace_bytes = op_desc->GetWorkspaceBytes();
  if (v_workspace_num.size() != v_workspace_bytes.size()) {
    GELOGW("workspace_num[%zu]!= workspace_bytes[%zu]", v_workspace_num.size(), v_workspace_bytes.size());
    return v_workspace_size;
  }

  for (auto workspace_bytes : v_workspace_bytes) {
    v_workspace_size.push_back(workspace_bytes);
  }

  return v_workspace_size;
}

///
/// @ingroup ge
/// @brief Get weight size.
/// @return vector<uint32_t>
///
vector<int64_t> ModelUtils::GetWeightSize(ConstOpDescPtr op_desc) {
  vector<int64_t> v_weight_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_weight_size);

  // const op, get weight directly
  const string type_name = op_desc->GetType();
  if ((type_name == "Const") || (type_name == "Constant")) {
    ConstGeTensorPtr weight = nullptr;
    if (AttrUtils::GetTensor(*op_desc, ATTR_NAME_WEIGHTS, weight)) {
      v_weight_size.push_back(TensorUtils::GetWeightSize(weight));
    }

    return v_weight_size;
  }

  // other ops get weight from connected constop
  const size_t inputs_size = op_desc->GetInputsSize();
  const vector<bool> v_is_input_const = op_desc->GetIsInputConst();
  for (size_t i = 0; i < inputs_size; ++i) {
    if ((i < v_is_input_const.size()) && v_is_input_const[i]) {
      const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(i);
      if (tensor_desc == nullptr) {
        GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
        continue;
      }

      int64_t tensor_size = 0;
      (void)TensorUtils::GetSize(*tensor_desc, tensor_size);
      v_weight_size.push_back(tensor_size);
    }
  }

  return v_weight_size;
}

///
/// @ingroup ge
/// @brief Get weights.
/// @return vector<ConstGeTensorPtr>
///
vector<ConstGeTensorPtr> ModelUtils::GetWeights(ConstOpDescPtr op_desc) {
  vector<ConstGeTensorPtr> v_weights;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_weights);

  // const op, get weight directly
  const string op_type = op_desc->GetType();
  if ((op_type == "Const") || (op_type == "Constant")) {
    ConstGeTensorPtr weight = nullptr;
    if (AttrUtils::GetTensor(*op_desc, ATTR_NAME_WEIGHTS, weight)) {
      v_weights.push_back(weight);
    }

    return v_weights;
  }

  // other ops get weight from connected constop
  const size_t inputs_size = op_desc->GetInputsSize();
  const vector<bool> v_is_input_const = op_desc->GetIsInputConst();
  for (size_t i = 0; i < inputs_size; ++i) {
    if ((i < v_is_input_const.size()) && v_is_input_const[i]) {
      const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(i);
      if (tensor_desc == nullptr) {
        GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
        continue;
      }

      ConstGeTensorPtr weight = nullptr;
      if (AttrUtils::GetTensor(*tensor_desc, ATTR_NAME_WEIGHTS, weight)) {
        v_weights.push_back(weight);
      }
    }
  }

  return v_weights;
}

///
/// @ingroup ge
/// @brief Get AiCpuOp Input descriptor.
/// @return vector<::tagCcAICPUTensor>
///
vector<::tagCcAICPUTensor> ModelUtils::GetInputDescs(ConstOpDescPtr op_desc) {
  // AiCpuOp::GetInputDescs
  vector<::opTensor_t> v_input_descs;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_input_descs);

  const size_t inputs_size = op_desc->GetInputsSize();
  const vector<bool> v_is_input_const = op_desc->GetIsInputConst();

  for (size_t i = 0; i < inputs_size; ++i) {
    if ((i < v_is_input_const.size()) && v_is_input_const[i]) {  // skip Const input node
      continue;
    }

    const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(i);
    if (tensor_desc == nullptr) {
      GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }

    uint32_t dim_cnt = 0;
    GE_CHK_BOOL_EXEC_WARN(TensorUtils::GetRealDimCnt(*tensor_desc, dim_cnt) == GRAPH_SUCCESS, continue,
                          "Get dim_cnt failed");

    opTensor_t tmp;
    uint32_t tmp_fmt = tensor_desc->GetFormat();
    tmp.format = tagOpTensorFormat(tmp_fmt);
    tmp.dim_cnt = static_cast<int32_t>(dim_cnt);
    uint32_t tmp_type = tensor_desc->GetDataType();
    tmp.data_type = tagOpDataType(tmp_type);

    for (int32_t j = 0; j < 4; j++) {  // 4 dims
      tmp.dim[j] = (j < tmp.dim_cnt ? tensor_desc->GetShape().GetDim(j) : 1);
    }

    v_input_descs.push_back(tmp);
  }

  return v_input_descs;
}

///
/// @ingroup ge
/// @brief Get AiCpuOp Output descriptor.
/// @return vector<::tagCcAICPUTensor>
///
vector<::tagCcAICPUTensor> ModelUtils::GetOutputDescs(ConstOpDescPtr op_desc) {
  // AiCpuOp::GetOutputDescs
  vector<::opTensor_t> v_output_descs;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_output_descs);

  // init op output opTensor_t struct
  const size_t output_num = op_desc->GetOutputsSize();
  for (size_t i = 0; i < output_num; ++i) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableOutputDesc(i);
    if (tensor_desc == nullptr) {
      GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }

    uint32_t dim_cnt = 0;
    GE_CHK_BOOL_EXEC_WARN(TensorUtils::GetRealDimCnt(*tensor_desc, dim_cnt) == GRAPH_SUCCESS, continue,
                          "Get dim_cnt failed");

    opTensor_t tmp;
    uint32_t tmp_fmt = tensor_desc->GetFormat();
    tmp.format = tagOpTensorFormat(tmp_fmt);
    tmp.dim_cnt = static_cast<int32_t>(dim_cnt);
    uint32_t tmp_type = tensor_desc->GetDataType();
    tmp.data_type = tagOpDataType(tmp_type);

    for (int32_t j = 0; j < 4; j++) {  // 4 dims
      tmp.dim[j] = (j < tmp.dim_cnt ? tensor_desc->GetShape().GetDim(j) : 1);
    }

    v_output_descs.push_back(tmp);
  }

  return v_output_descs;
}

///
/// @ingroup ge
/// @brief Get input data address.
/// @return vector<void*>
///
vector<void *> ModelUtils::GetInputDataAddrs(const RuntimeParam &model_param, ConstOpDescPtr op_desc) {
  vector<void *> v_input_data_addr;  // init as:buf_base + op_def_->input(i));
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_input_data_addr);
  uint64_t session_id = model_param.session_id;

  const size_t inputs_size = op_desc->GetInputsSize();
  const vector<int64_t> v_input_offset = op_desc->GetInputOffset();

  const string op_type = op_desc->GetType();

  size_t non_const_index = 0;
  const vector<bool> v_is_input_const = op_desc->GetIsInputConst();
  vector<int64_t> v_memory_type;
  bool has_mem_type_attr = ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type);
  if (has_mem_type_attr && (v_memory_type.size() != inputs_size)) {
    GELOGE(PARAM_INVALID, "Fusion: check input size failed, op: %s, input v_memory_type size: %zu input numbers: %zu",
           op_desc->GetName().c_str(), v_memory_type.size(), inputs_size);
    return v_input_data_addr;
  }
  for (size_t i = 0; i < inputs_size; ++i) {
    if ((i < v_is_input_const.size()) && v_is_input_const[i] && (op_type != NETOUTPUT)) {
      // TBE: add weights address to input
      const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(i);
      if (tensor_desc == nullptr) {
        GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
        continue;
      }

      int64_t tensor_size = 0;
      GE_CHK_STATUS(TensorUtils::GetSize(*tensor_desc, tensor_size));
      if (tensor_size) {
        int64_t data_offset = 0;
        GE_CHK_STATUS(TensorUtils::GetDataOffset(*tensor_desc, data_offset));
        uint8_t *weight_addr = model_param.weight_base + data_offset;
        v_input_data_addr.push_back(weight_addr);
        GELOGI("[IMAS]GetInputDataAddrs graph_%u type[C] name[%s] input[%zu] memaddr[%p]", model_param.graph_id,
               op_desc->GetName().c_str(), i, weight_addr);
      }
      non_const_index++;
      continue;
    }

    GE_IF_BOOL_EXEC(non_const_index >= v_input_offset.size(),
                    GELOGW("offsets=%zu, inputs=%zu, index=%zu.", v_input_offset.size(), inputs_size, non_const_index);
                    break;);

    int64_t input_offset = v_input_offset[non_const_index];
    non_const_index++;
    GE_IF_BOOL_EXEC(model_param.var_size != 0 && ge::VarManager::Instance(session_id)->IsVarAddr(input_offset),
                    uint8_t *variable_addr = model_param.var_base + input_offset - model_param.logic_var_base;
                    v_input_data_addr.push_back(variable_addr);
                    GELOGI("[IMAS]GetInputDataAddrs graph_%u type[V] name[%s] input[%lu] memaddr[%p]",
                           model_param.graph_id, op_desc->GetName().c_str(), i, variable_addr);
                    continue);

    // feature maps
    uint8_t *mem_addr = nullptr;
    //  fusion
    if (has_mem_type_attr && v_memory_type[i] == RT_MEMORY_L1) {
      mem_addr = reinterpret_cast<uint8_t *>(reinterpret_cast<intptr_t>(input_offset));
      v_input_data_addr.push_back(mem_addr);
    } else {
      mem_addr = model_param.mem_base + input_offset;
      v_input_data_addr.push_back(mem_addr);
    }
    GELOGI("[IMAS]GetInputDataAddrs graph_%u type[F] name[%s] input[%zu] memaddr[%p]", model_param.graph_id,
           op_desc->GetName().c_str(), i, mem_addr);
  }

  return v_input_data_addr;
}

///
/// @ingroup ge
/// @brief Get output data address.
/// @return vector<void*>
///
vector<void *> ModelUtils::GetOutputDataAddrs(const RuntimeParam &model_param, ConstOpDescPtr op_desc) {
  vector<void *> v_output_data_addr;  // init as:buf_base + op_def_->output(i)
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_output_data_addr);
  uint64_t session_id = model_param.session_id;

  const size_t outputs_size = op_desc->GetOutputsSize();
  const vector<int64_t> v_output_offset = op_desc->GetOutputOffset();
  GE_IF_BOOL_EXEC(v_output_offset.size() != outputs_size,
                  GELOGW("Output param invalid: output_offset=%zu, outputs=%zu.", v_output_offset.size(), outputs_size);
                  return v_output_data_addr);
  vector<int64_t> v_memory_type;
  bool has_mem_type_attr = ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
  if (has_mem_type_attr && (v_memory_type.size() != outputs_size)) {
    GELOGE(PARAM_INVALID,
           "Fusion: check output size failed, op: %s, output v_memory_type size: %lu output numbers: %zu",
           op_desc->GetName().c_str(), v_memory_type.size(), outputs_size);
    return v_output_data_addr;
  }
  for (size_t i = 0; i < outputs_size; ++i) {
    GE_IF_BOOL_EXEC(model_param.var_size != 0 && ge::VarManager::Instance(session_id)->IsVarAddr(v_output_offset[i]),
                    uint8_t *variable_addr = model_param.var_base + v_output_offset[i] - model_param.logic_var_base;
                    v_output_data_addr.push_back(variable_addr);
                    GELOGI("[IMAS]GetOutputDataAddrs graph_%u type[V] name[%s] output[%zu] memaddr[%p]",
                           model_param.graph_id, op_desc->GetName().c_str(), i, variable_addr);
                    continue);
    // feature maps
    uint8_t *mem_addr = nullptr;
    //  fusion
    if (has_mem_type_attr && v_memory_type[i] == RT_MEMORY_L1) {
      mem_addr = reinterpret_cast<uint8_t *>(reinterpret_cast<intptr_t>(v_output_offset[i]));
      v_output_data_addr.push_back(mem_addr);
    } else {
      mem_addr = static_cast<uint8_t *>(model_param.mem_base + v_output_offset[i]);
      v_output_data_addr.push_back(mem_addr);
    }
    GELOGI("[IMAS]GetOutputDataAddrs graph_%u type[F] name[%s] output[%zu] memaddr[%p]", model_param.graph_id,
           op_desc->GetName().c_str(), i, mem_addr);
  }
  return v_output_data_addr;
}

///
/// @ingroup ge
/// @brief Get workspace data address.
/// @return vector<void*>
///
vector<void *> ModelUtils::GetWorkspaceDataAddrs(const RuntimeParam &model_param, ConstOpDescPtr op_desc) {
  vector<void *> v_workspace_data_addr;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_workspace_data_addr);

  const vector<int64_t> v_workspace_offset = op_desc->GetWorkspace();
  const vector<int64_t> v_workspace_bytes = op_desc->GetWorkspaceBytes();
  if (v_workspace_offset.size() != v_workspace_bytes.size()) {
    GELOGW("v_workspace_offset.size()[%zu] != v_workspace_bytes.size()[%zu]", v_workspace_offset.size(),
           v_workspace_bytes.size());
    return v_workspace_data_addr;
  }
  vector<int64_t> v_memory_type;
  bool has_mem_type_attr = ge::AttrUtils::GetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, v_memory_type);
  for (size_t i = 0; i < v_workspace_bytes.size(); ++i) {
    if (has_mem_type_attr && v_memory_type[i] == RT_MEMORY_L1) {
      v_workspace_data_addr.push_back(reinterpret_cast<uint8_t *>(reinterpret_cast<intptr_t>(v_workspace_offset[i])));
      GELOGI("Fusion: op: %s, GetWorkspaceDataAddrs mem_addr[workspace index %zu]:%p", op_desc->GetName().c_str(), i,
             reinterpret_cast<uint8_t *>(reinterpret_cast<intptr_t>(v_workspace_offset[i])));
    } else {
      int64_t workspace_offset = v_workspace_offset[i];
      int64_t workspace_bytes = v_workspace_bytes[i];
      uint8_t *mem_addr = workspace_bytes == 0 ? nullptr : model_param.mem_base + workspace_offset;
      v_workspace_data_addr.push_back(mem_addr);
      GELOGI("[IMAS]GetWorkspaceDataAddrs graph_%u type[F] name[%s] workspace[%zu] offset[%ld] bytes[%ld] memaddr[%p]",
             model_param.graph_id, op_desc->GetName().c_str(), i, workspace_offset, workspace_bytes, mem_addr);
    }
  }

  return v_workspace_data_addr;
}

///
/// @ingroup ge
/// @brief Get runtime memory address.
/// @return Status
///
Status ModelUtils::GetRtAddress(const RuntimeParam &param, uintptr_t logic_addr, uint8_t *&mem_addr) {
  uint8_t *runtime_base_addr = nullptr;
  if ((param.logic_mem_base <= logic_addr) && (logic_addr < param.logic_mem_base + param.mem_size)) {
    runtime_base_addr = param.mem_base - param.logic_mem_base;
    GELOGI("The logic addr:0x%lx is data address, base:0x%lx, size:%lu", logic_addr, param.logic_mem_base,
           param.mem_size);
  } else if ((param.logic_weight_base <= logic_addr) && (logic_addr < param.logic_weight_base + param.weight_size)) {
    runtime_base_addr = param.weight_base - param.logic_weight_base;
    GELOGI("The logic addr:0x%lx is weight address, base:0x%lx, size:%lu", logic_addr, param.logic_weight_base,
           param.weight_size);
  } else if ((param.logic_var_base <= logic_addr) && (logic_addr < param.logic_var_base + param.var_size)) {
    runtime_base_addr = param.var_base - param.logic_var_base;
    GELOGI("The logic addr:0x%lx is variable address, base:0x%lx, size:%lu", logic_addr, param.logic_var_base,
           param.var_size);
  } else if (logic_addr != 0) {
    mem_addr = nullptr;
    GELOGE(PARAM_INVALID, "The logic addr:0x%lx is abnormal", logic_addr);
    return PARAM_INVALID;
  }

  mem_addr = runtime_base_addr + logic_addr;
  return SUCCESS;
}
}  // namespace ge
