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
/// @ingroup domi_ome
/// @brief Check is Output Op.
/// @return bool
///
bool ModelUtils::IsOutput(ConstOpDescPtr op_desc) {
  GE_CHECK_NOTNULL_EXEC(op_desc, return false);
  size_t output_size = op_desc->GetOutputsSize();
  for (size_t i = 0; i < output_size; ++i) {
    bool output_tensor = false;
    GE_IF_BOOL_EXEC(TensorUtils::GetOutputTensor(op_desc->GetOutputDesc(i), output_tensor) != GRAPH_SUCCESS,
                    GELOGW("Get OutputTensor failed, name: %s, output index: %zu", op_desc->GetName().c_str(), i);
                    return false;);
    if (output_tensor) {
      return true;
    }
  }

  return false;
}

///
/// @ingroup domi_ome
/// @brief Check is the Input need trans code.
/// @return bool
///
bool ModelUtils::IsInputTensorNeedTrans(ConstOpDescPtr op_desc, size_t tensor_index) {
  GE_CHECK_NOTNULL_EXEC(op_desc, return false);
  const auto &input_desc = op_desc->GetInputDesc(tensor_index);
  const auto &output_desc = op_desc->GetOutputDesc(tensor_index);

  if ((output_desc.GetFormat() == FORMAT_NC1HWC0) && (output_desc.GetDataType() == DT_INT8)) {
    // AIPP input, add attribute in data op to tag aipp
    return false;
  }

  return (input_desc.GetFormat() != output_desc.GetFormat()) || (input_desc.GetDataType() != output_desc.GetDataType());
}

///
/// @ingroup domi_ome
/// @brief Get input size.
/// @return vector<uint32_t>
///
vector<uint32_t> ModelUtils::GetInputSize(ConstOpDescPtr op_desc) {
  vector<uint32_t> v_input_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_input_size);
  const size_t inputs_size = op_desc->GetInputsSize();
  const string op_type = op_desc->GetType();

  const vector<bool> v_is_input_const = op_desc->GetIsInputConst();
  for (size_t i = 0; i < inputs_size; ++i) {
    if ((i < v_is_input_const.size()) && v_is_input_const[i] && (op_type != NETOUTPUT)) {
      // TBE: add weights size to input
      GE_IF_BOOL_EXEC(true, GeTensorDesc tensor_desc = op_desc->GetInputDesc(i); uint32_t tensor_size = 0;
                      GE_CHK_STATUS(TensorUtils::GetSize(tensor_desc, tensor_size));
                      if (tensor_size) { v_input_size.push_back(tensor_size); });
      continue;
    }

    uint32_t tensor_size = 0;
    GE_IF_BOOL_EXEC(
        TensorUtils::GetSize(op_desc->GetInputDesc(i), tensor_size) != GRAPH_SUCCESS,
        GELOGI("Get size from TensorDesc failed, op : %s, input index : %zu", op_desc->GetName().c_str(), i);
        continue;);

    v_input_size.push_back(tensor_size);
  }

  return v_input_size;
}

///
/// @ingroup domi_ome
/// @brief Get output size.
/// @return vector<uint32_t>
///
vector<uint32_t> ModelUtils::GetOutputSize(ConstOpDescPtr op_desc) {
  vector<uint32_t> v_output_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_output_size);

  const size_t outputs_size = op_desc->GetOutputsSize();
  const vector<int64_t> v_output_offset = op_desc->GetOutputOffset();
  GE_IF_BOOL_EXEC(v_output_offset.size() != outputs_size,
                  GELOGW("Output param invalid: output_offset=%zu, outputs=%zu.", v_output_offset.size(), outputs_size);
                  return v_output_size;);

  for (size_t i = 0; i < outputs_size; ++i) {
    uint32_t tensor_size = 0;
    GE_IF_BOOL_EXEC(
        TensorUtils::GetSize(op_desc->GetOutputDesc(i), tensor_size) != GRAPH_SUCCESS,
        GELOGI("Get size from TensorDesc failed, op : %s, output index : %zu", op_desc->GetName().c_str(), i);
        continue;);

    v_output_size.push_back(tensor_size);
  }

  return v_output_size;
}

///
/// @ingroup domi_ome
/// @brief Get workspace size.
/// @return vector<uint32_t>
///
vector<uint32_t> ModelUtils::GetWorkspaceSize(ConstOpDescPtr op_desc) {
  vector<uint32_t> v_workspace_size;
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
/// @ingroup domi_ome
/// @brief Get weight size.
/// @return vector<uint32_t>
///
vector<uint32_t> ModelUtils::GetWeightSize(ConstOpDescPtr op_desc) {
  vector<uint32_t> v_weight_size;
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
      v_weight_size.push_back(TensorUtils::GetWeightSize(op_desc->GetInputDesc(i)));
    }
  }

  return v_weight_size;
}

///
/// @ingroup domi_ome
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
      ConstGeTensorPtr weight = nullptr;
      GeTensorDesc tensor_desc = op_desc->GetInputDesc(i);
      if (AttrUtils::GetTensor(tensor_desc, ATTR_NAME_WEIGHTS, weight)) {
        v_weights.push_back(weight);
      }
    }
  }

  return v_weights;
}

///
/// @ingroup domi_ome
/// @brief Save Output tensor info to vector.
/// @return Status
///
Status ModelUtils::GetOutputSize(ConstOpDescPtr op_desc, vector<uint32_t> &output_size_list,
                                 vector<uint32_t> &output_memory_size_list) {
  GE_CHECK_NOTNULL(op_desc);

  for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
    bool output_tensor = false;
    auto output_desc = op_desc->GetOutputDesc(i);
    GE_CHK_STATUS_RET(TensorUtils::GetOutputTensor(output_desc, output_tensor),
                      "get OutputTensor failed, op : %s, input index : %zu", op_desc->GetName().c_str(), i);

    if (output_tensor) {
      // get transferred parameters such as size
      uint32_t size = 0;
      uint32_t memory_size = 0;
      graphStatus graph_status0 = TensorUtils::GetTensorSizeInBytes(output_desc, size);
      graphStatus graph_status1 = TensorUtils::GetTensorMemorySizeInBytes(output_desc, memory_size);
      if ((graph_status0 != GRAPH_SUCCESS) || (graph_status1 != GRAPH_SUCCESS)) {
        return INTERNAL_ERROR;
      }
      output_size_list.push_back(size);
      output_memory_size_list.push_back(memory_size);
    }
  }

  return SUCCESS;
}

///
/// @ingroup domi_ome
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

    uint32_t dim_cnt = 0;
    const auto &descriptor = op_desc->GetInputDesc(i);
    GE_CHK_BOOL_EXEC_WARN(TensorUtils::GetRealDimCnt(descriptor, dim_cnt) == GRAPH_SUCCESS, continue,
                          "Get dim_cnt failed");

    opTensor_t tmp;
    uint32_t tmp_fmt = descriptor.GetFormat();
    tmp.format = tagOpTensorFormat(tmp_fmt);
    tmp.dim_cnt = static_cast<int32_t>(dim_cnt);
    uint32_t tmp_type = descriptor.GetDataType();
    tmp.data_type = tagOpDataType(tmp_type);

    for (int32_t j = 0; j < 4; j++) {  // 4 dims
      tmp.dim[j] = (j < tmp.dim_cnt ? descriptor.GetShape().GetDim(j) : 1);
    }

    v_input_descs.push_back(tmp);
  }

  return v_input_descs;
}

///
/// @ingroup domi_ome
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
    uint32_t dim_cnt = 0;
    const auto &descriptor = op_desc->GetOutputDesc(i);
    GE_CHK_BOOL_EXEC_WARN(TensorUtils::GetRealDimCnt(descriptor, dim_cnt) == GRAPH_SUCCESS, continue,
                          "Get dim_cnt failed");

    opTensor_t tmp;
    uint32_t tmp_fmt = descriptor.GetFormat();
    tmp.format = tagOpTensorFormat(tmp_fmt);
    tmp.dim_cnt = static_cast<int32_t>(dim_cnt);
    uint32_t tmp_type = descriptor.GetDataType();
    tmp.data_type = tagOpDataType(tmp_type);

    for (int32_t j = 0; j < 4; j++) {  // 4 dims
      tmp.dim[j] = static_cast<int32_t>(j < tmp.dim_cnt ? descriptor.GetShape().GetDim(j) : 1);
    }

    v_output_descs.push_back(tmp);
  }

  return v_output_descs;
}

///
/// @ingroup domi_ome
/// @brief Get input data address.
/// @return vector<void*>
///
vector<void *> ModelUtils::GetInputDataAddrs(const RuntimeParam &model_param, ConstOpDescPtr op_desc,
                                             bool need_convert) {
  vector<void *> v_input_data_addr;  // init as:buf_base + op_def_->input(i));
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_input_data_addr);
  uint64_t session_id = model_param.session_id;
  uint8_t *mem_base = model_param.mem_base;
  uint8_t *var_base = model_param.var_base;
  uint8_t *weight_base = model_param.weight_base;
  const uint64_t logic_mem_base = 0;
  uint64_t logic_weight_base = 0;
  uint64_t logic_var_base = model_param.logic_var_base;
  uint64_t mem_size = model_param.mem_size;
  uint64_t weight_size = model_param.weight_size;
  uint64_t var_size = model_param.var_size;

  if (need_convert) {
    Status status = ConvertVirtualAddressToPhysical(mem_base, mem_size, mem_base);
    if (status != SUCCESS) {
      GELOGE(RT_FAILED, "Convert virtual address to physical for mem_base failed.");
      return v_input_data_addr;
    }

    status = ConvertVirtualAddressToPhysical(weight_base, weight_size, weight_base);
    if (status != SUCCESS) {
      GELOGE(RT_FAILED, "Convert virtual address to physical for weight_base failed.");
      return v_input_data_addr;
    }

    status = ConvertVirtualAddressToPhysical(var_base, var_size, var_base);
    if (status != SUCCESS) {
      GELOGE(RT_FAILED, "Convert virtual address to physical for var_base failed.");
      return v_input_data_addr;
    }
  }

  const size_t inputs_size = op_desc->GetInputsSize();
  const vector<int64_t> v_input_offset = op_desc->GetInputOffset();

  const string op_type = op_desc->GetType();

  size_t non_const_index = 0;
  const vector<bool> v_is_input_const = op_desc->GetIsInputConst();
  for (size_t i = 0; i < inputs_size; ++i) {
    if ((i < v_is_input_const.size()) && v_is_input_const[i] && (op_type != NETOUTPUT)) {
      // TBE: add weights address to input
      GE_IF_BOOL_EXEC(true, GeTensorDesc tensor_desc = op_desc->GetInputDesc(i); uint32_t tensor_size = 0;
                        GE_CHK_STATUS(TensorUtils::GetSize(tensor_desc, tensor_size)); if (tensor_size) {
                          int64_t data_offset = 0;
                          GE_CHK_STATUS(TensorUtils::GetDataOffset(tensor_desc, data_offset));
                          uint8_t *weight_addr = static_cast<uint8_t *>(weight_base + data_offset - logic_weight_base);
                          v_input_data_addr.push_back(weight_addr);
                        });
      non_const_index++;
      continue;
    }

    GE_IF_BOOL_EXEC(non_const_index >= v_input_offset.size(),
                    GELOGW("offsets=%zu, inputs=%zu, index=%zu.", v_input_offset.size(), inputs_size, non_const_index);
                    break;);

    int64_t input_offset = v_input_offset[non_const_index];
    non_const_index++;
    GE_IF_BOOL_EXEC(var_size != 0 && ge::VarManager::Instance(session_id)->IsVarAddr(input_offset),
                      uint8_t *variable_addr = var_base + input_offset - logic_var_base;
                      v_input_data_addr.push_back(variable_addr);
                      continue;);

    bool input_tensor = false;
    GE_IF_BOOL_EXEC(TensorUtils::GetInputTensor(op_desc->GetOutputDesc(i), input_tensor) != GRAPH_SUCCESS,
                    GELOGW("get size from TensorDesc failed, op: %s, input index: %zu", op_desc->GetName().c_str(), i);
                    continue;);

    uint8_t *mem_addr = mem_base + input_offset - logic_mem_base;
    v_input_data_addr.push_back(mem_addr);
  }

  return v_input_data_addr;
}

///
/// @ingroup domi_ome
/// @brief Get output data address.
/// @return vector<void*>
///
vector<void *> ModelUtils::GetOutputDataAddrs(const RuntimeParam &model_param, ConstOpDescPtr op_desc,
                                              bool need_convert) {
  vector<void *> v_output_data_addr;  // init as:buf_base + op_def_->output(i)
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_output_data_addr);
  uint64_t session_id = model_param.session_id;
  uint8_t *mem_base = model_param.mem_base;
  uint8_t *var_base = model_param.var_base;
  const uint64_t logic_mem_base = 0;
  uint64_t logic_var_base = model_param.logic_var_base;
  uint64_t mem_size = model_param.mem_size;
  uint64_t var_size = model_param.var_size;

  if (need_convert) {
    Status status = ConvertVirtualAddressToPhysical(mem_base, mem_size, mem_base);
    if (status != SUCCESS) {
      GELOGE(RT_FAILED, "Convert virtual address to physical for mem_base failed.");
      return v_output_data_addr;
    }

    status = ConvertVirtualAddressToPhysical(var_base, var_size, var_base);
    if (status != SUCCESS) {
      GELOGE(RT_FAILED, "Convert virtual address to physical for var_base failed.");
      return v_output_data_addr;
    }
  }

  const size_t outputs_size = op_desc->GetOutputsSize();
  const vector<int64_t> v_output_offset = op_desc->GetOutputOffset();
  GE_IF_BOOL_EXEC(v_output_offset.size() != outputs_size,
                  GELOGW("Output param invalid: output_offset=%zu, outputs=%zu.", v_output_offset.size(), outputs_size);
                  return v_output_data_addr;);

  for (size_t i = 0; i < outputs_size; ++i) {
    GE_IF_BOOL_EXEC(var_size != 0 && ge::VarManager::Instance(session_id)->IsVarAddr(v_output_offset[i]),
                      uint8_t *variable_addr = static_cast<uint8_t *>(var_base + v_output_offset[i] - logic_var_base);
                      v_output_data_addr.push_back(variable_addr);
                      continue;);
    uint8_t *mem_addr = mem_base + v_output_offset[i] - logic_mem_base;
    v_output_data_addr.push_back(mem_addr);
  }

  return v_output_data_addr;
}

///
/// @ingroup domi_ome
/// @brief Get workspace data address.
/// @return vector<void*>
///
vector<void *> ModelUtils::GetWorkspaceDataAddrs(const RuntimeParam &model_param, ConstOpDescPtr op_desc) {
  vector<void *> v_workspace_data_addr;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_workspace_data_addr);
  uint8_t *mem_base = model_param.mem_base;
  uint64_t mem_size = model_param.mem_size;

  Status status = ConvertVirtualAddressToPhysical(mem_base, mem_size, mem_base);
  if (status != SUCCESS) {
    GELOGE(RT_FAILED, "Convert virtual address to physical for mem_base failed.");
    return v_workspace_data_addr;
  }

  const vector<int64_t> v_workspace_num = op_desc->GetWorkspace();
  const vector<int64_t> v_workspace_bytes = op_desc->GetWorkspaceBytes();
  if (v_workspace_num.size() != v_workspace_bytes.size()) {
    GELOGW("v_workspace_num.size()[%zu] != v_workspace_bytes.size()[%zu]", v_workspace_num.size(),
           v_workspace_bytes.size());
    return v_workspace_data_addr;
  }

  for (size_t i = 0; i < v_workspace_bytes.size(); ++i) {
    int64_t workspace_num = v_workspace_num[i];
    int64_t workspace_bytes = v_workspace_bytes[i];
    v_workspace_data_addr.push_back(workspace_bytes == 0 ? nullptr : mem_base + workspace_num);
  }

  return v_workspace_data_addr;
}

Status ModelUtils::ConvertVirtualAddressToPhysical(uint8_t *virtual_address, uint64_t size,
                                                   uint8_t *&physical_address) {
  // Indicates whether use physical address.
  const char *use_physical_address = std::getenv("GE_USE_PHYSICAL_ADDRESS");
  if (use_physical_address == nullptr || virtual_address == 0 || size == 0) {
    return SUCCESS;
  }

  rtError_t ret = rtKernelConfigTransArg(virtual_address, size, 0, reinterpret_cast<void **>(&physical_address));
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rtKernelConfigTransArg failed, ret: 0x%X", ret);
    return RT_FAILED;
  }

  return SUCCESS;
}
}  // namespace ge
