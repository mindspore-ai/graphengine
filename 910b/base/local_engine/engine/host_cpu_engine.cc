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
#include "local_engine/engine/host_cpu_engine.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "register/host_cpu_context.h"
#include "common/plugin/ge_util.h"
#include "common/plugin/plugin_manager.h"
#include "common/math/math_util.h"

namespace ge {
namespace {
const std::string kConstantFoldingName = "libconstant_folding_ops.so";

Status GetDataNumber(const GeTensorDesc &out_desc, uint64_t &data_num) {
  int64_t num_size = out_desc.GetShape().IsScalar() ? 1 : out_desc.GetShape().GetShapeSize();
  if (out_desc.GetShape().IsUnknownShape()) {
    std::vector<std::pair<int64_t, int64_t>> range;
    if (out_desc.GetShapeRange(range) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "GetShapeRange failed.");
      GELOGE(INTERNAL_ERROR, "[Get][ShapeRange] failed.");
      return INTERNAL_ERROR;
    }
    int64_t max_range_size = 1;
    for (const auto& item : range) {
      FMK_INT64_MULCHECK(max_range_size, item.second);
      max_range_size *= item.second;
    }
    num_size = max_range_size;
  }
  if (num_size < 0) {
    GELOGW("[Check][Param] Get negative size, num_size=%ld.", num_size);
    return INTERNAL_ERROR;
  }
  data_num = static_cast<uint64_t>(num_size);
  return SUCCESS;
}

Status CreateOutputByDataType(const ConstOpDescPtr &op_desc, const GeTensorDesc &output_desc, const size_t index,
                              const std::vector<GeTensorPtr> &outputs, std::map<std::string, Tensor> &named_outputs) {
  GeTensorPtr ge_tensor = nullptr;
  Status ret = SUCCESS;
  if (outputs.size() != op_desc->GetOutputsSize()) {
    uint64_t data_num = 0U;
    ret = GetDataNumber(output_desc, data_num);
    if (ret != SUCCESS) {
      GELOGW("[Get][Number] node:%s get size for output %zu failed", op_desc->GetName().c_str(), index);
      return INTERNAL_ERROR;
    }
    const int64_t size = ge::GetSizeInBytes(static_cast<int64_t>(data_num), output_desc.GetDataType());
    if (size < 0) {
      return INTERNAL_ERROR;
    }
    ge_tensor = MakeShared<GeTensor>(output_desc, static_cast<size_t>(size));
    GE_CHECK_NOTNULL(ge_tensor);
    GELOGD("node:%s allocate output %zu success, size=%ld", op_desc->GetName().c_str(), index, size);
    ge_tensor->MutableTensorDesc().SetDataType(output_desc.GetDataType());
    ge_tensor->MutableTensorDesc().SetShape(output_desc.GetShape());
  } else {
    ge_tensor = outputs[index];
    GE_CHECK_NOTNULL(ge_tensor);
    GELOGD("node:%s existed output %zu", op_desc->GetName().c_str(), index);
  }
  Tensor tensor = TensorAdapter::AsTensor(*ge_tensor);
  const std::string tensor_name = op_desc->GetOutputNameByIndex(static_cast<uint32_t>(index));
  GE_RETURN_WITH_LOG_IF_TRUE(tensor_name.empty(), "[Get][OutputName] failed. node = %s, index = %zu",
                             op_desc->GetName().c_str(), index);
  (void)named_outputs.emplace(tensor_name, tensor);
  return ret;
}
}

HostCpuEngine &HostCpuEngine::GetInstance() {
  static HostCpuEngine instance;
  return instance;
}

ge::Status HostCpuEngine::Initialize(const std::string &path_base) {
  const std::lock_guard<std::mutex> lock(mu_);
  if (initialized_) {
      GELOGI("HostCpuEngine is already initialized");
      return SUCCESS;
  }
  std::string lib_dir;
  GE_CHK_STATUS_RET_NOLOG(PluginManager::GetConstantFoldingOpsPath(path_base, lib_dir));

  std::vector<std::string> so_paths;
  if (ListSoFiles(lib_dir, so_paths) == SUCCESS) {
    (void) LoadLibs(so_paths);
  }

  initialized_ = true;
  return SUCCESS;
}

void HostCpuEngine::Finalize() const {
  GELOGI("start HostCpuEngine::Finalize");
}

Status HostCpuEngine::PrepareInputs(const ge::ConstOpDescPtr &op_desc,
                                    const std::vector<ConstGeTensorPtr> &inputs,
                                    std::map<std::string, const Tensor> &named_inputs) {
  const auto num_inputs = op_desc->GetInputsSize();
  if (num_inputs != inputs.size()) {
    REPORT_INNER_ERROR("E19999", "Mismatching input sizes. op_desc:%s(%s) has %zu input(s), but given %zu",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), num_inputs, inputs.size());
    GELOGE(PARAM_INVALID, "[Check][Param] Mismatching input sizes. op_desc:%s(%s) has %zu input(s), but given %zu",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), num_inputs, inputs.size());
    return PARAM_INVALID;
  }

  for (size_t i = 0U; i < num_inputs; ++i) {
    const auto ge_tensor = inputs[i];
    GE_CHECK_NOTNULL(ge_tensor);
    auto tensor = TensorAdapter::AsTensor(*ge_tensor);
    auto tensor_name = op_desc->GetInputNameByIndex(static_cast<uint32_t>(i));
    GE_RETURN_WITH_LOG_IF_TRUE(tensor_name.empty(), "[Get][InputName] failed. node = %s, index = %zu",
                               op_desc->GetName().c_str(), i);
    GELOGD("Successfully inserted input tensor. node = %s, index = %zu, input name = %s",
           op_desc->GetName().c_str(), i, tensor_name.c_str());
    (void)named_inputs.emplace(tensor_name, tensor);
  }

  return SUCCESS;
}

Status HostCpuEngine::PrepareOutputs(const ge::ConstOpDescPtr &op_desc,
                                     std::vector<GeTensorPtr> &outputs,
                                     std::map<std::string, Tensor> &named_outputs) {
  if ((!outputs.empty()) && (outputs.size() != op_desc->GetOutputsSize())) {
    GELOGW("size of outputs not match, size of outputs = %zu, exactly output_num=%zu.",
           outputs.size(), op_desc->GetOutputsSize());
    outputs.clear();
  }

  Status ret = SUCCESS;
  std::set<DataType> output_data_type_set = { DT_BOOL, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16,
                                              DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT4,
                                              DT_COMPLEX64, DT_COMPLEX128 };
  for (size_t i = 0U; i < op_desc->GetOutputsSize(); ++i) {
    const auto &out_desc = op_desc->GetOutputDesc(static_cast<uint32_t>(i));
    const std::set<DataType>::const_iterator &output_data_type_iter = output_data_type_set.find(out_desc.GetDataType());
    if (output_data_type_iter == output_data_type_set.cend()) {
      GELOGW("data type %s not support.", TypeUtils::DataTypeToSerialString(out_desc.GetDataType()).c_str());
      ret = NOT_CHANGED;
      break;
    }

    ret = CreateOutputByDataType(op_desc, out_desc, i, outputs, named_outputs);
    if (ret != SUCCESS) {
      break;
    }
  }

  return ret;
}

Status HostCpuEngine::RunInternal(const ge::OpDescPtr &op_desc, HostCpuOp &op_kernel,
                                  const std::map<std::string, const Tensor> &named_inputs,
                                  std::map<std::string, Tensor> &named_outputs) {
  GELOGD("Run operation on host cpu, op name: %s", op_desc->GetName().c_str());
  Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  const auto ret = op_kernel.Compute(op, named_inputs, named_outputs);
  if (ret != GRAPH_SUCCESS) {
    GELOGW("Failed to compute host cpu op. node = %s", op_desc->GetName().c_str());
    return FAILED;
  }
  op.BreakConnect();

  return SUCCESS;
}

Status HostCpuEngine::Run(const NodePtr &node, HostCpuOp &kernel, const std::vector<ConstGeTensorPtr> &inputs,
                          std::vector<GeTensorPtr> &outputs) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());

  GELOGD("Run node by host cpu engine. node name = %s", node->GetName().c_str());
  std::map<std::string, const Tensor> named_inputs;
  std::map<std::string, Tensor> named_outputs;
  const auto op_desc = node->GetOpDesc();
  GE_CHK_STATUS_RET_NOLOG(PrepareInputs(op_desc, inputs, named_inputs));
  GE_CHK_STATUS_RET_NOLOG(PrepareOutputs(op_desc, outputs, named_outputs));
  GE_CHK_STATUS_RET_NOLOG(RunInternal(op_desc, kernel, named_inputs, named_outputs));

  std::vector<GeTensorPtr> tmp_outputs;
  for (size_t i = 0U; i < op_desc->GetOutputsSize(); i++) {
    const auto tensor_name = op_desc->GetOutputNameByIndex(static_cast<uint32_t>(i));
    if (tensor_name.empty()) {
      REPORT_INNER_ERROR("E19999", "GetOutputNameByIndex failed, node = %s, index = %zu",
                         op_desc->GetName().c_str(), i);
      GELOGE(INTERNAL_ERROR, "[Get][OutputName] failed. node = %s, index = %zu", op_desc->GetName().c_str(), i);
      return INTERNAL_ERROR;
    }
    const std::map<std::string, Tensor>::const_iterator iter = named_outputs.find(tensor_name);
    if (iter == named_outputs.cend()) {
       REPORT_INNER_ERROR("E19999", "get output tensor failed, node = %s, index = %zu, tensor_name = %s",
                          op_desc->GetName().c_str(), i, tensor_name.c_str());
       GELOGE(INTERNAL_ERROR, "[Get][OutputTensor] failed. node = %s, index = %zu, tensor_name = %s",
              op_desc->GetName().c_str(), i, tensor_name.c_str());
      return INTERNAL_ERROR;
    }
    auto ge_tensor = MakeShared<GeTensor>(TensorAdapter::AsGeTensor(iter->second));
    GE_CHECK_NOTNULL(ge_tensor);
    tmp_outputs.emplace_back(ge_tensor);
  }
  GELOGD("Run node by host cpu engine successfully. name node = %s", node->GetName().c_str());
  outputs.swap(tmp_outputs);
  return SUCCESS;
}

static int32_t RegularFileFilterFunc(const mmDirent *const entry) {
  const bool flag = (static_cast<int32_t>(entry->d_type) == static_cast<int32_t>(DT_REG));
  return static_cast<int32_t>(flag);
}

Status HostCpuEngine::ListSoFiles(const std::string &base_dir, std::vector<std::string> &names) {
  std::string real_path = base_dir;
  GE_CHK_STATUS_RET_NOLOG(GetEngineRealPath(real_path));
  real_path.push_back('/');
  mmDirent **entries = nullptr;
  const auto ret = mmScandir(real_path.c_str(), &entries, &RegularFileFilterFunc, nullptr);
  if (ret < 0) {
    GELOGW("scan dir failed. path = %s, ret = %d", real_path.c_str(), ret);
    return INTERNAL_ERROR;
  }

  for (int32_t i = 0; i < ret; ++i) {
    const mmDirent *const dir_ent = *PtrAdd<mmDirent *>(entries, static_cast<size_t>(ret), static_cast<size_t>(i));
    const std::string name = std::string(dir_ent->d_name);
    if (IsSoFile(name)) {
      names.emplace_back(real_path + name);
    }
  }

  mmScandirFree(entries, ret);
  GELOGI("Found %d libs to load", ret);
  return SUCCESS;
}

bool HostCpuEngine::IsSoFile(const std::string &file_name) {
  static const std::string so_suffix(".so");
  const auto pos = file_name.rfind(so_suffix);
  if (pos == std::string::npos) {
    return false;
  }

  return pos == (file_name.size() - so_suffix.size());
}

Status HostCpuEngine::LoadLibs(std::vector<std::string> &lib_paths) {
  for (auto &so_path : lib_paths) {
    GE_CHK_STATUS_RET_NOLOG(GetEngineRealPath(so_path));
    GE_CHK_STATUS_RET_NOLOG(LoadLib(so_path));
  }

  return SUCCESS;
}

Status HostCpuEngine::LoadLib(const std::string &lib_path) {
  GELOGI("To invoke dlopen on lib: %s", lib_path.c_str());
  const auto open_flag = static_cast<uint32_t>(MMPA_RTLD_NOW) | static_cast<uint32_t>(MMPA_RTLD_GLOBAL);
  auto handle = mmDlopen(lib_path.c_str(), static_cast<int32_t>(open_flag));
  if (handle == nullptr) {
    const char_t *error = mmDlerror();
    error = (error == nullptr) ? "" : error;
    REPORT_CALL_ERROR("E19999", "mmDlopen failed, path = %s, error = %s", lib_path.c_str(), error);
    GELOGE(INTERNAL_ERROR, "[Invoke][DlOpen] failed. path = %s, error = %s", lib_path.c_str(), error);
    return INTERNAL_ERROR;
  }

  const auto initialize = reinterpret_cast<Status (*)(const HostCpuContext &)>(mmDlsym(handle, "Initialize"));
  if (initialize != nullptr) {
    GELOGI("Invoke function Initialize in lib: %s", lib_path.c_str());
    if (initialize(HostCpuContext()) != SUCCESS) {
      GELOGW("Failed to invoke function Initialize in lib: %s", lib_path.c_str());
    }
  }

  GELOGI("Lib: %s has been opened", lib_path.c_str());
  if (lib_path.find(kConstantFoldingName) != lib_path.npos) {
    constant_folding_handle_ = handle;
  }
  lib_handles_.emplace_back(handle);
  return SUCCESS;
}

Status HostCpuEngine::GetEngineRealPath(std::string &path) {
  const std::string real_path = RealPath(path.c_str());
  if (real_path.empty()) {
    GELOGW("File path %s is invalid.", path.c_str());
    return INTERNAL_ERROR;
  }

  path = real_path;
  return SUCCESS;
}
} // namespace ge
