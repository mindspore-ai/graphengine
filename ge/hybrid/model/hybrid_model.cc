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

#include "hybrid_model.h"
#include <vector>
#include "graph/debug/ge_attr_define.h"
#include "graph/load/new_model_manager/model_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "hybrid/model/hybrid_model_builder.h"
#include "hybrid/node_executor/node_executor.h"
#include "common/op/ge_op_utils.h"

namespace ge {
namespace hybrid {
namespace {
const int64_t kMemSizeUnknownShape = -1; // Unknown shape mem size
}

HybridModel::HybridModel(GeRootModelPtr ge_model) : ge_root_model_(std::move(ge_model)) {
}

HybridModel::~HybridModel() {
  GELOGD("[%s] HybridModel destroyed.", model_name_.c_str());
}

Status HybridModel::Init() {
  GELOGD("Start to init hybrid model.");
  GE_CHK_STATUS_RET(HybridModelBuilder(*this).Build(), "Failed to build hybrid model.");
  GELOGD("HybridModel initialized successfully.");
  return SUCCESS;
}

TensorValue* HybridModel::GetVariable(const string &name) const {
  auto it = variable_tensors_.find(name);
  if (it == variable_tensors_.end()) {
    GELOGD("Failed to get variable tensor. var name = [%s]", name.c_str());
    return nullptr;
  }

  GELOGD("Got variable tensor. var name = [%s], tensor = %s", name.c_str(), it->second->DebugString().c_str());
  return it->second.get();
}

NodePtr HybridModel::GetVariableNode(const string &name) const {
  auto it = device_variable_nodes_.find(name);
  if (it != device_variable_nodes_.end()) {
    return it->second;
  }
  auto host_find = host_variable_nodes_.find(name);
  if (host_find != host_variable_nodes_.end()) {
    return host_find->second;
  }
  GELOGD("Failed to get variable node by name = [%s]", name.c_str());
  return nullptr;
}

const std::vector<domi::TaskDef> *HybridModel::GetTaskDefs(const NodePtr &node) const {
  auto it = task_defs_.find(node);
  if (it == task_defs_.end()) {
    return nullptr;
  }

  return &it->second;
}

NodeItem *HybridModel::MutableNodeItem(const NodePtr &node) {
  auto it = node_items_.find(node);
  if (it == node_items_.end()) {
    return nullptr;
  }

  return it->second.get();
}

const NodeItem *HybridModel::GetNodeItem(const NodePtr &node) const {
  auto it = node_items_.find(node);
  if (it == node_items_.end()) {
    return nullptr;
  }

  return it->second.get();
}

GeModelPtr HybridModel::GetGeModel(const NodePtr &node) const {
  auto it = known_shape_sub_models_.find(node);
  if (it == known_shape_sub_models_.end()) {
    GELOGE(INTERNAL_ERROR, "[%s] Failed to get GeModel for subgraph node.", node->GetName().c_str());
    return nullptr;
  }

  return it->second;
}

const GraphItem* HybridModel::GetRootGraphItem() const {
  return root_graph_item_.get();
}

const GraphItem *HybridModel::GetSubgraphItem(const std::string &graph_name) const {
  GELOGD("To find subgraph item by name = %s", graph_name.c_str());
  auto it = subgraph_items_.find(graph_name);
  if (it == subgraph_items_.end()) {
    GELOGD("Subgraph item not found by node = %s", graph_name.c_str());
    return nullptr;
  }

  return it->second.get();
}

const GraphItem *HybridModel::GetSubgraphItem(const ComputeGraphPtr &subgraph) const {
  if (subgraph == nullptr) {
    GELOGE(PARAM_INVALID, "subgraph is nullptr");
    return nullptr;
  }

  auto subgraph_name = subgraph->GetName();
  return GetSubgraphItem(subgraph_name);
}

const string &HybridModel::GetModelName() const {
  return model_name_;
}

Status HybridModel::GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type) {
  // dynamic shape do not need dynamic batch
  batch_info = {};
  dynamic_type = -1;
  return SUCCESS;
}

void HybridModel::GetUserDesignateShapeOrder(std::vector<std::string> &user_input_shape_order) {
  // dynamic shape do not need dynamic batch
  user_input_shape_order = {};
}

void HybridModel::GetModelAttr(std::vector<std::string> &dynamic_output_shape_info) {
  dynamic_output_shape_info = {};
}

Status HybridModel::GetInputOutputDescInfo(vector<InputOutputDescInfo> &input_desc,
                                           vector<InputOutputDescInfo> &output_desc,
                                           std::vector<uint32_t> &input_formats,
                                           std::vector<uint32_t> &output_formats) {
  auto node_item_list = root_graph_item_->GetInputNodes();
  if (node_item_list.empty()) {
    GELOGE(FAILED, "node item list is empty!");
    return FAILED;
  }

  GE_CHECK_NOTNULL(node_item_list[0]->node);
  GE_CHECK_NOTNULL(node_item_list[0]->node->GetOpDesc());
  if (node_item_list[0]->node->GetOpDesc()->GetInputsSize() != 1) {
    GELOGE(FAILED, "input size of op is not 1!");
    return FAILED;
  }

  GE_CHK_STATUS_RET(GetInputDescInfo(input_desc, input_formats), "get input desc info failed");
  GE_CHK_STATUS_RET(GetOutputDescInfo(output_desc, output_formats), "get ouput desc info failed");

  return SUCCESS;
}

void HybridModel::SetInputDimsAndShapeRangesInfo(const vector<int64_t> &model_input_dims,
                                                 std::vector<std::pair<int64_t, int64_t>> &shape_ranges,
                                                 InputOutputDescInfo &input) {
  for (auto model_input_dim : model_input_dims) {
    input.shape_info.dims.push_back(model_input_dim);
  }
  input.shape_info.shape_ranges = shape_ranges;
  return;
}

void HybridModel::CreateInputDimsInfo(const OpDescPtr &op_desc, InputOutputDescInfo &input) {
  std::vector<std::pair<int64_t,int64_t>> shape_ranges;
  if (is_new_model_desc_ && op_desc->HasAttr(ATTR_NAME_INPUT_DIMS)) {
    // When static aipp is set, need to get the model input dims which processed by aipp
    vector<int64_t> model_input_dims;
    (void)AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_DIMS, model_input_dims);
    SetInputDimsAndShapeRangesInfo(model_input_dims, shape_ranges, input);
    return;
  }
  // judge if this data is linked dynamic aipp first, multiply batch has been considered
  if (op_desc->HasAttr("_dynamic_aipp_input_dims")) {
    vector<int64_t> dynamic_aipp_input_dims;
    (void)AttrUtils::GetListInt(op_desc, "_dynamic_aipp_input_dims", dynamic_aipp_input_dims);
    SetInputDimsAndShapeRangesInfo(dynamic_aipp_input_dims, shape_ranges, input);
    return;
  } else {
    vector<int64_t> input_dims = op_desc->GetInputDescPtr(0)->GetShape().GetDims();
    op_desc->GetInputDescPtr(0)->GetShapeRange(shape_ranges);
    SetInputDimsAndShapeRangesInfo(input_dims, shape_ranges, input);
    return;
  }
}

Status HybridModel::GetInputDescInfo(vector<InputOutputDescInfo> &input_desc, std::vector<uint32_t> &formats) {
  auto node_item_list = root_graph_item_->GetInputNodes();
  for (auto &node_item : node_item_list) {
    InputOutputDescInfo input;

    GE_CHECK_NOTNULL(node_item->node);
    auto op_desc = node_item->node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    GE_CHECK_NOTNULL(op_desc->GetInputDescPtr(0));

    Format format = op_desc->GetInputDescPtr(0)->GetFormat();
    input.data_type = op_desc->GetInputDescPtr(0)->GetDataType();
    input.name = op_desc->GetName();

    int64_t input_size = 0;
    GE_CHK_STATUS_RET(TensorUtils::GetSize(*op_desc->GetInputDescPtr(0), input_size), "get input size failed.");

    // support dynamic shape
    if (input_size < 0) {
      GELOGD("dynamic shape scene, input size is unknown. "
             "format=%d, data_type=%d, input_size=%ld",
             format, input.data_type, input_size);
      input_size = kMemSizeUnknownShape;   // -1
    }

    // not support dynamic shape input for now, so input_size here will be not less than zero.
    input.size = input_size;

    CreateInputDimsInfo(op_desc, input);

    formats.push_back(format);
    input_desc.push_back(input);
  }
  is_new_model_desc_ = false;
  return SUCCESS;
}

void HybridModel::CreateOutput(ConstGeTensorDescPtr &output_desc,
                               InputOutputDescInfo &output_desc_info, uint32_t &format_result) {
  GE_IF_BOOL_EXEC(output_desc == nullptr, GELOGE(FAILED, "output desc ptr is nullptr"); return );
  Format format = output_desc->GetFormat();
  GeShape shape = output_desc->GetShape();
  std::vector<std::pair<int64_t,int64_t>> shape_ranges;
  output_desc->GetShapeRange(shape_ranges);
  DataType data_type = output_desc->GetDataType();
  format_result = format;
  if (format == FORMAT_FRACTAL_Z) {  // FraczToHWCK
    int64_t k = shape.GetDim(0);                                           // 0: first dim
    int64_t c = shape.GetDim(1);                                           // 1: second dim
    int64_t h = shape.GetDim(2);                                           // 2: third dim
    int64_t w = shape.GetDim(3);                                           // 3: forth dim
    output_desc_info.shape_info.dims.push_back(h);
    output_desc_info.shape_info.dims.push_back(w);
    output_desc_info.shape_info.dims.push_back(c);
    output_desc_info.shape_info.dims.push_back(k);
    if (shape_ranges.size() == 4) {                   // 4 dims
      output_desc_info.shape_info.shape_ranges.push_back(shape_ranges[2]);  // h:2
      output_desc_info.shape_info.shape_ranges.push_back(shape_ranges[3]);  // w:3
      output_desc_info.shape_info.shape_ranges.push_back(shape_ranges[1]);  // c:1
      output_desc_info.shape_info.shape_ranges.push_back(shape_ranges[0]);  // k:0
    }
    format_result = FORMAT_HWCN;
  } else {
    for (size_t j = 0; j < shape.GetDimNum(); j++) {
      output_desc_info.shape_info.dims.push_back(shape.GetDim(j));
    }
    output_desc_info.shape_info.shape_ranges = shape_ranges;
  }
  int64_t tensor_size = 0;
  (void)TensorUtils::CalcTensorMemSize(shape, format, data_type, tensor_size);
  output_desc_info.size = static_cast<uint64_t>(tensor_size);
  output_desc_info.data_type = output_desc->GetDataType();
}

Status HybridModel::GetOutputDescInfo(vector<InputOutputDescInfo> &output_desc, std::vector<uint32_t> &formats) {
  std::vector<ConstGeTensorDescPtr> output_desc_list;
  // output_desc_list contains vaild input desc
  GE_CHK_STATUS_RET(root_graph_item_->GetOutputDescList(output_desc_list), "get output desc info failed");

  vector<std::string> out_node_names;
  (void)ge::AttrUtils::GetListStr(ge_root_model_->GetRootGraph(), ATTR_MODEL_OUT_NODES_NAME, out_node_names);

  GE_CHECK_NOTNULL(root_graph_item_->GetOutputNode());
  auto op_desc = root_graph_item_->GetOutputNode()->op_desc;
  GE_CHECK_NOTNULL(op_desc);

  auto out_size = static_cast<uint32_t>(op_desc->GetInputsSize());
  GE_CHK_BOOL_RET_STATUS(out_size == output_desc_list.size(),
      FAILED, "output size[%u] not match output_desc_list size[%zu]", out_size, output_desc_list.size());

  for (uint32_t index = 0; index < out_size; ++index) {
    string output_name;
    std::vector<std::string> src_name = op_desc->GetSrcName();
    std::vector<int64_t> src_index = op_desc->GetSrcIndex();
    if (out_size == out_node_names.size()) {
      bool contains_colon = out_node_names[index].find(":") != std::string::npos;
      output_name = contains_colon ? out_node_names[index] : out_node_names[index] +
          ":" + std::to_string(src_index[index]);
    } else {
      output_name = std::string("output_") + std::to_string(index) + "_" + src_name[index] +
          "_" + std::to_string(src_index[index]);
    }

    InputOutputDescInfo output_desc_info;
    output_desc_info.name = output_name;

    uint32_t format_result;
    CreateOutput(output_desc_list[index], output_desc_info, format_result);
    output_desc.push_back(output_desc_info);
    formats.push_back(format_result);
  }
  return SUCCESS;
}

TensorValue *HybridModel::GetConstant(const NodePtr &node) const {
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "Param is null");
    return nullptr;
  }

  auto it = constant_tensors_.find(node);
  if (it == constant_tensors_.end()) {
    GELOGD("constant not found, node name = [%s]", node->GetName().c_str());
    return nullptr;
  }

  GELOGD("Got constant tensor, node name = [%s], tensor = %s",
         node->GetName().c_str(),
         it->second->DebugString().c_str());
  return it->second.get();
}

TensorValue * HybridModel::GetTensor(const NodePtr &node) const {
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "Param is null");
    return nullptr;
  }

  if (node->GetType() == CONSTANT) {
    return GetConstant(node);
  }

  return GetVariable(node->GetName());
}
}  // namespace hybrid
}  // namespace ge
