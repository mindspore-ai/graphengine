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

#include "ge_local_engine/ops_kernel_store/ge_local_ops_kernel_info.h"
#include <memory>
#include "common/constant/constant.h"
#include "framework/common/debug/ge_log.h"
#include "common/ge_inner_error_codes.h"
#include "common/ge/ge_util.h"
#include "common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "op/op_factory.h"
#include "proto/task.pb.h"

namespace {
const char *const kConstantOpType = "Constant";
const char *const kConstantOpAttrName = "value";
const char *const kDataOpType = "Data";
}  // namespace
namespace ge {
namespace ge_local {
using domi::TaskDef;
using std::map;
using std::string;
using std::vector;

Status GeLocalOpsKernelInfoStore::Initialize(const map<string, string> &options) {
  GELOGI("GeLocalOpsKernelInfoStore init start.");

  OpInfo default_op_info = {.engine = kGeLocalEngineName,
                            .opKernelLib = kGeLocalOpKernelLibName,
                            .computeCost = 0,
                            .flagPartial = false,
                            .flagAsync = false,
                            .isAtomic = false};
  // Init op_info_map_
  auto all_ops = OpFactory::Instance().GetAllOps();
  for (auto &op : all_ops) {
    op_info_map_[op] = default_op_info;
  }

  GELOGI("GeLocalOpsKernelInfoStore inited success. op num=%zu", op_info_map_.size());

  return SUCCESS;
}

Status GeLocalOpsKernelInfoStore::Finalize() {
  op_info_map_.clear();
  return SUCCESS;
}

Status GeLocalOpsKernelInfoStore::CalcOpRunningParam(Node &ge_node) {
  OpDescPtr op_desc = ge_node.GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(FAILED, "CalcOpRunningParam failed, as op desc is null");
    return FAILED;
  }
  const string node_name = ge_node.GetName();
  const string node_type = ge_node.GetType();
  size_t output_size = op_desc->GetOutputsSize();
  GELOGD("Calc op[%s:%s] op running param, output size=%zu.", node_name.c_str(), node_type.c_str(), output_size);

  for (size_t i = 0; i < output_size; ++i) {
    GeTensorDesc output_tensor = op_desc->GetOutputDesc(static_cast<uint32_t>(i));
    Format format = output_tensor.GetFormat();
    DataType data_type = output_tensor.GetDataType();

    int64_t mem_size = 0;
    graphStatus graph_status = TensorUtils::GetSize(output_tensor, mem_size);
    // If mem size has been set, no need reset.
    if ((graph_status == GRAPH_SUCCESS) && (mem_size > 0) && (data_type != DT_STRING)) {
      GELOGD("Op[%s:%s] out[%zu] mem size has been set, no need calc again, format=%s, data_type=%s, mem_size=%ld.",
             node_name.c_str(), node_type.c_str(), i, TypeUtils::FormatToSerialString(format).c_str(),
             TypeUtils::DataTypeToSerialString(data_type).c_str(), mem_size);
      continue;
    }

    int64_t output_mem_size = 0;
    GeShape output_shape = output_tensor.GetShape();
    if ((node_type == kConstantOpType) && (data_type == DT_STRING)) {
      graph_status = CalcConstantStrMemSize(op_desc, output_mem_size);
    } else if (node_type == kDataOpType) {
      int64_t output_size = 0;
      graph_status = TensorUtils::GetTensorMemorySizeInBytes(output_tensor, output_size);
      output_mem_size = output_size;
    } else {
      graph_status = TensorUtils::CalcTensorMemSize(output_shape, format, data_type, output_mem_size);
    }

    if (graph_status != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Calc op[%s:%s] out[%zu] mem size failed, format=%s, data_type=%s, error=%u.", node_name.c_str(),
             node_type.c_str(), i, TypeUtils::FormatToSerialString(format).c_str(),
             TypeUtils::DataTypeToSerialString(data_type).c_str(), graph_status);
      return FAILED;
    }

    if (output_mem_size < 0) {
      GELOGE(FAILED,
             "Calc op[%s:%s] out[%zu] mem size is negative(not support),"
             " format=%s, data_type=%s, mem_size=%ld.",
             node_name.c_str(), node_type.c_str(), i, TypeUtils::FormatToSerialString(format).c_str(),
             TypeUtils::DataTypeToSerialString(data_type).c_str(), output_mem_size);
      return FAILED;
    }
    GELOGI(
      "Calc op[%s:%s] out[%zu] mem size is %ld,"
      " format=%s, data_type=%s.",
      node_name.c_str(), node_type.c_str(), i, output_mem_size, TypeUtils::FormatToSerialString(format).c_str(),
      TypeUtils::DataTypeToSerialString(data_type).c_str());

    TensorUtils::SetSize(output_tensor, output_mem_size);

    graph_status = op_desc->UpdateOutputDesc(static_cast<uint32_t>(i), output_tensor);
    if (graph_status != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Update op[%s:%s] out[%zu] desc failed, format=%s, data_type=%s, error=%u.", node_name.c_str(),
             node_type.c_str(), i, TypeUtils::FormatToSerialString(format).c_str(),
             TypeUtils::DataTypeToSerialString(data_type).c_str(), graph_status);
      return FAILED;
    }
  }
  GELOGD("Calc op[%s:%s] running param success.", node_name.c_str(), node_type.c_str());
  return SUCCESS;
}

Status GeLocalOpsKernelInfoStore::CalcConstantStrMemSize(const OpDescPtr &op_desc, int64_t &mem_size) {
  if (op_desc == nullptr) {
    GELOGE(FAILED, "CalcConstantStrMemSize failed, as op desc is null");
    return FAILED;
  }
  ConstGeTensorPtr value = MakeShared<const GeTensor>();
  if (value == nullptr) {
    GELOGE(FAILED, "make shared ConstGeTensor exception.");
    return FAILED;
  }
  // Constant op attr name is "value"
  if (!AttrUtils::GetTensor(op_desc, kConstantOpAttrName, value)) {
    GELOGE(FAILED, "Get Constant op attr value failed");
    return FAILED;
  }
  mem_size = static_cast<int64_t>(value->GetData().size());
  return GRAPH_SUCCESS;
}

void GeLocalOpsKernelInfoStore::GetAllOpsKernelInfo(map<string, OpInfo> &infos) const { infos = op_info_map_; }

Status GeLocalOpsKernelInfoStore::GenerateTask(const Node &node, RunContext &context, vector<TaskDef> &tasks) {
  string name = node.GetName();
  string type = node.GetType();
  GELOGD("Ge local generate task for node:%s(%s) begin, tasks.size()=%zu.", name.c_str(), type.c_str(), tasks.size());

  auto op = OpFactory::Instance().CreateOp(node, context);
  if (op == nullptr) {
    GELOGE(FAILED, "CreateOp for node:%s(%s) failed.", name.c_str(), type.c_str());
    return FAILED;
  }

  Status ret = op->Run();
  if (ret != SUCCESS) {
    GELOGE(ret, "Node:%s(%s) op run failed.", name.c_str(), type.c_str());
    return ret;
  }
  GELOGI("Ge local generate task for node:%s(%s) end, tasks.size()=%zu.", name.c_str(), type.c_str(), tasks.size());
  return ret;
}

bool GeLocalOpsKernelInfoStore::CheckSupported(const OpDescPtr &op_desc, std::string &) const {
  if (op_desc == nullptr) {
    return false;
  }
  return op_info_map_.count(op_desc->GetType()) > 0;
}

Status GeLocalOpsKernelInfoStore::CreateSession(const map<string, string> &session_options) {
  // Do nothing
  return SUCCESS;
}

Status GeLocalOpsKernelInfoStore::DestroySession(const map<string, string> &session_options) {
  // Do nothing
  return SUCCESS;
}
}  // namespace ge_local
}  // namespace ge
