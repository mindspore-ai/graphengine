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

#include "graph/build/model_builder.h"
#include <iostream>
#include <set>
#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/buffer.h"
#include "graph/build/label_allocator.h"
#include "graph/build/stream_allocator.h"
#include "graph/common/omg_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_error_codes.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/optimize/common/params.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/ge_context.h"
#include "init/gelib.h"
#include "memory/memory_assigner.h"
#include "omg/version.h"
#include "register/op_registry.h"

using domi::AIPP_CONV_FLAG;
using domi::AIPP_DATA_FLAG;
using domi::AIPP_DATA_TYPE;
using domi::AippOpParams;
using domi::ATOMICADDRCLEAN;
using domi::ATTR_NAME_AUTOMIC_ADD_MEM_SIZE;
using domi::ATTR_NAME_AUTOMIC_ADD_START;
using domi::CAST;
using domi::CHW_DIM_H;
using domi::CHW_DIM_W;
using domi::CONCAT;
using domi::CONSTANT;
using domi::CONSTANTOP;
using domi::CONVOLUTION;
using domi::DATA;
using domi::DATA_TYPE;
using domi::DEFAULT_FORMAT;
using domi::DIM_DEFAULT_SIZE;
using domi::DOMI_TENSOR_NC1HWC0;
using domi::HWC_DIM_H;
using domi::HWC_DIM_W;
using domi::LOOPCOND;
using domi::MODEL_ATTR_TASKS;
using domi::ModelTaskDef;
using domi::NCHW_DIM_H;
using domi::NCHW_DIM_N;
using domi::NCHW_DIM_W;
using domi::NETOUTPUT;
using domi::NHWC_DIM_H;
using domi::NHWC_DIM_W;
using domi::PlatformVersionManager;
using domi::STREAMMERGE;
using domi::VARIABLE;
using domi::XRGB_CHN_NUM;
using ge::FAILED;
using ge::PARAM_INVALID;
using ge::SUCCESS;
using std::map;
using std::set;
using std::string;
using std::vector;

namespace {
const uint32_t kWeightsStartOffset = 512;
const int32_t kWrongIndex = -2;

const float kImgRatioYUV420SP_U8 = 1.5;
const int kImgRatioRGB888_U8 = 3;
const int kImgRatioNC1HWC0DI_FP16 = 12;
const int kInvalidIndexNum = -1;

const uint32_t kInputDimensions2D = 2;
const uint32_t kInputDimensions3D = 3;

const char *const kVectorCore = "VectorCore";
const char *const kCoreType = "ge.engineType";
const std::string kEnableL1Fusion = "ge.l1Fusion";

const set<string> adjust_layer_type_ = {CONVOLUTION};

bool IsGeLocalOp(const ge::ConstOpDescPtr &op_desc) {
  auto type = op_desc->GetType();
  if (type == CONSTANTOP) {
    // constant op just has one output
    ge::GeTensorDesc output_desc = op_desc->GetOutputDesc(0);
    return !(output_desc.GetDataType() == ge::DT_STRING);
  }
  const set<string> ge_local_set = {domi::STREAMMERGE, domi::MEMCPYASYNC, domi::STREAMACTIVE,  domi::STREAMSWITCH,
                                    domi::VARIABLE,    domi::NOOP,        domi::CONSTANT,      domi::ENTER,
                                    domi::REFENTER,    domi::LOOPCOND,    domi::NEXTITERATION, domi::REFNEXTITERATION,
                                    domi::EXIT,        domi::REFEXIT};
  return (ge_local_set.find(type) != ge_local_set.end());
}
}  // namespace

namespace ge {
ModelBuilder::ModelBuilder(ge::ComputeGraphPtr compute_graph, const vector<SubGraphInfoPtr> &subgraphs,
                           const map<string, int> &stream_max_parallel_num, bool hcom_parallel, int mode)
    : mem_offset_(0),
      weight_offset_(kWeightsStartOffset),
      compute_graph_(std::move(compute_graph)),
      subgraphs_(subgraphs),
      stream_num_(0),
      event_num_(0),
      label_num_(0),
      stream_max_parallel_num_(stream_max_parallel_num),
      hcom_parallel_(hcom_parallel),
      build_mode_(mode),
      max_mem_offset_(0),
      platform_type_(0),
      is_loop_graph_(false),
      is_l1_fusion_enable_(false) {}

ModelBuilder::~ModelBuilder() {}

Status ModelBuilder::CalcOutputSize(const ge::NodePtr &n) {
  GE_CHECK_NOTNULL(n);
  auto node_op_desc = n->GetOpDesc();
  GE_CHECK_NOTNULL(node_op_desc);
  uint32_t index = 0;
  for (const auto &output_desc_ptr : node_op_desc->GetAllOutputsDescPtr()) {
    GeTensorDesc &desc_temp = *output_desc_ptr;

    uint32_t dim_num = static_cast<uint32_t>(desc_temp.GetShape().GetDimNum());
    GE_IF_BOOL_EXEC(dim_num > DIM_DEFAULT_SIZE, TensorUtils::SetRealDimCnt(desc_temp, dim_num));
    // calculate tensor size
    int64_t size_temp = 0;
    graphStatus graph_status = TensorUtils::GetTensorMemorySizeInBytes(desc_temp, size_temp);
    if (graph_status != GRAPH_SUCCESS) {
      GELOGE(graph_status, "GetTensorMemorySizeInBytes failed!");
      return FAILED;
    }
    TensorUtils::SetSize(desc_temp, size_temp);
    if (node_op_desc->UpdateOutputDesc(index, desc_temp) != SUCCESS) {
      GELOGE(FAILED, "UpdateOutputDesc failed.");
      return FAILED;
    }

    GELOGD("update output desc, dim_size: %u, mem_size: %ld, format: %s, type: %s, node name:%s", dim_num, size_temp,
           TypeUtils::FormatToSerialString(desc_temp.GetFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(desc_temp.GetDataType()).c_str(), node_op_desc->GetName().c_str());
    index++;
  }

  return SUCCESS;
}

void ModelBuilder::SetInputIsConst(const ge::NodePtr &n) {
  auto node_op_desc = n->GetOpDesc();
  if (node_op_desc == nullptr) {
    GELOGW("node_op_desc is nullptr!");
    return;
  }
  auto is_input_const = node_op_desc->GetIsInputConst();

  // must set all true input_const to false
  for (size_t i = 0; i < is_input_const.size(); i++) {
    is_input_const[i] = false;
  }
  auto in_data_anchors = n->GetAllInDataAnchors();
  for (size_t index = 0; index < in_data_anchors.size(); index++) {
    auto in_data_anchor = in_data_anchors.at(index);
    const auto &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    const auto &src_node = peer_out_anchor->GetOwnerNode();
    if (src_node->GetType() == CONSTANT) {
      GELOGI("SetIsInputConst const");
      for (size_t i = is_input_const.size(); i <= index; ++i) {
        is_input_const.push_back(false);
      }
      is_input_const[index] = true;

      vector<GeTensorPtr> weights = OpDescUtils::MutableWeights(src_node);
      if (weights.empty()) {
        GELOGW("SetInputIsConst weights is empty");
        return;
      }
      GeTensorPtr weight = weights[0];
      GE_IF_BOOL_EXEC(weight == nullptr, continue);
      GeTensorDesc &tensor_desc = weight->MutableTensorDesc();
      int64_t data_offset = 0;
      if (TensorUtils::GetDataOffset(tensor_desc, data_offset) != GRAPH_SUCCESS) {
        GELOGW("Get Offset from weight failed");
        return;
      }
      auto input_tensor = node_op_desc->MutableInputDesc(static_cast<uint32_t>(index));
      if (input_tensor == nullptr) {
        GELOGW("Get input_tensor failed");
        return;
      }
      TensorUtils::SetDataOffset(*input_tensor, data_offset);
    } else if (src_node->GetType() == CONSTANTOP) {
      if ((index < is_input_const.size()) && is_input_const[index]) {
        is_input_const[index] = false;
      }
    }
  }

  std::string input_const_info = domi::ToString(is_input_const);
  GELOGD("update opdesc:%s InputConst:%s", node_op_desc->GetName().c_str(), input_const_info.c_str());
  node_op_desc->SetIsInputConst(is_input_const);
}

Status ModelBuilder::AdjustConstWeightSize(const ge::NodePtr &node, size_t &mem_offset) {
  GE_CHECK_NOTNULL(node);
  if (node->GetType() == CONSTANT) {
    vector<GeTensorPtr> weights = OpDescUtils::MutableWeights(node);
    if (weights.empty()) {
      GELOGE(FAILED, "weights size of node %s is empty", node->GetName().c_str());
      return FAILED;
    }
    GeTensorPtr weight = weights[0];
    if (weight == nullptr) {
      GELOGE(FAILED, "weights[0] is null.");
      return FAILED;
    }
    GeTensorDesc &tensor_desc = weight->MutableTensorDesc();
    size_t output_size = weight->GetData().size();
    TensorUtils::SetDataOffset(tensor_desc, mem_offset);
    mem_offset += output_size;
  }
  return SUCCESS;
}

Status ModelBuilder::SetInputOutputDesc() {
  Status ret;
  GELOGI("Start to SetInputOutputDesc.");

  for (const ge::NodePtr &n : compute_graph_->GetDirectNode()) {
    auto node_op_desc = n->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);

    if (!is_loop_graph_ && node_op_desc->GetType() == LOOPCOND) {
      is_loop_graph_ = true;
    }

    if (node_op_desc->GetType() == DATA_TYPE || node_op_desc->GetType() == AIPP_DATA_TYPE) {
      GELOGD("Data node: %s.", n->GetName().c_str());
      continue;
    }

    GE_IF_BOOL_EXEC(n->GetInAllNodes().empty() && n->GetOutAllNodes().empty(), continue;);

    SetInputIsConst(n);
    if (IsGeLocalOp(n->GetOpDesc())) {
      GE_CHK_STATUS_RET(CalcOutputSize(n), "Calculate output size failed");
    }
    ret = AdjustConstWeightSize(n, weight_offset_);
    GE_CHK_STATUS_RET(ret, "AdjustConstWeightSize failed");

    GE_IF_BOOL_EXEC(((weight_offset_ > 0) && (weight_offset_ % MEM_ALIGN_SIZE != 0)),
                    weight_offset_ = (weight_offset_ + MEM_ALIGN_SIZE - 1) / MEM_ALIGN_SIZE * MEM_ALIGN_SIZE);
  }
  GE_CHK_STATUS_RET(compute_graph_->TopologicalSorting(), "TopologicalSorting failed");
  return SUCCESS;
}

void ModelBuilder::AddNodeInputProperty() {
  for (const ge::NodePtr &node : compute_graph_->GetDirectNode()) {
    auto node_op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, GELOGW("node_op_desc is nullptr!"); return );
    vector<string> src_name_list;
    vector<int64_t> src_index_list;
    for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
      auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
      GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, GELOGW("peer_out_anchor is nullptr!"); continue);
      GE_IF_BOOL_EXEC(node_op_desc->HasAttr(MERGE_PRENODE_FLAG), continue);

      ge::NodePtr src_node = peer_out_anchor->GetOwnerNode();
      src_name_list.emplace_back(src_node->GetName());
      src_index_list.emplace_back(peer_out_anchor->GetIdx());
    }
    auto in_control_anchor = node->GetInControlAnchor();
    if (in_control_anchor != nullptr) {
      string src_name_temp;
      for (const auto &out_control_anchor : in_control_anchor->GetPeerOutControlAnchors()) {
        ge::NodePtr src_node = out_control_anchor->GetOwnerNode();
        src_name_temp = src_name_temp.empty() ? src_node->GetName() : src_name_temp + ":" + src_node->GetName();
      }
      GE_IF_BOOL_EXEC(!src_name_temp.empty(), src_name_list.emplace_back(src_name_temp);)
    }
    node_op_desc->SetSrcName(src_name_list);
    node_op_desc->SetSrcIndex(src_index_list);
  }

  for (const ge::NodePtr &node : compute_graph_->GetDirectNode()) {
    auto node_op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, GELOGW("node_op_desc is nullptr!"); return );
    GE_IF_BOOL_EXEC(node_op_desc->GetType() == NETOUTPUT, continue);
    auto out_control_anchor = node->GetOutControlAnchor();
    GE_IF_BOOL_EXEC(out_control_anchor == nullptr, GELOGW("out_control_anchor is nullptr"); return );
    vector<string> dst_name_list;
    vector<int64_t> dst_index_list;
    string dst_name_temp;
    for (const auto &in_control_anchor : out_control_anchor->GetPeerInControlAnchors()) {
      ge::NodePtr dst_node = in_control_anchor->GetOwnerNode();
      dst_name_temp = dst_name_temp.empty() ? dst_node->GetName() : dst_name_temp + ":" + dst_node->GetName();
    }
    GE_IF_BOOL_EXEC(!dst_name_temp.empty(), dst_name_list.emplace_back(dst_name_temp));

    GE_IF_BOOL_EXEC(!out_control_anchor->GetPeerInControlAnchors().empty(),
                    dst_index_list.emplace_back(kInvalidIndexNum));

    for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      GE_IF_BOOL_EXEC(node_op_desc->HasAttr(MERGE_PRENODE_FLAG), break);
      dst_name_temp = "";
      int64_t dst_index = kWrongIndex;  // assign an impossible value to dst_index.
      for (const auto &in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        GE_IF_BOOL_EXEC(in_data_anchor == nullptr, GELOGW("in_data_anchor is nullptr"); return );
        ge::NodePtr dst_node = in_data_anchor->GetOwnerNode();
        dst_name_temp = dst_name_temp.empty() ? dst_node->GetName() : dst_name_temp + ":" + dst_node->GetName();
        dst_index = in_data_anchor->GetIdx();
      }
      GE_IF_BOOL_EXEC(dst_index != kWrongIndex, dst_index_list.emplace_back(dst_index));  // not found
      GE_IF_BOOL_EXEC(!dst_name_temp.empty(), dst_name_list.emplace_back(dst_name_temp));
    }
    node_op_desc->SetDstName(dst_name_list);
    node_op_desc->SetDstIndex(dst_index_list);
  }
}

Status ModelBuilder::AdjustInputTensorFlag() {
  GELOGI("Start to AdjustInputTensorFlag.");
  for (const ge::NodePtr &n : compute_graph_->GetDirectNode()) {
    if ((n->GetType() == DATA_TYPE) || (n->GetType() == AIPP_DATA_TYPE)) {
      GELOGD("Data node: %s.", n->GetName().c_str());
      for (const auto &anchor : n->GetAllOutDataAnchors()) {
        for (const auto &in_anchors : anchor->GetPeerInDataAnchors()) {
          GE_IF_BOOL_EXEC(in_anchors == nullptr, continue);
          auto owner_node = in_anchors->GetOwnerNode();
          auto owner_node_op_desc = owner_node->GetOpDesc();
          GE_IF_BOOL_EXEC(owner_node_op_desc == nullptr, continue);
          auto input_desc = owner_node_op_desc->GetInputDesc(in_anchors->GetIdx());
          ge::TensorUtils::SetInputTensor(input_desc, true);
          if (owner_node_op_desc->UpdateInputDesc(in_anchors->GetIdx(), input_desc) != SUCCESS) {
            GELOGE(FAILED, "UpdateOutputDesc failed.");
            return FAILED;
          }
        }
      }
    }
  }
  return SUCCESS;
}
void ModelBuilder::InitL1FusionOption() {
  string is_l1_fusion_enable = "false";
  graphStatus ret = ge::GetContext().GetOption(kEnableL1Fusion, is_l1_fusion_enable);
  if (ret == GRAPH_SUCCESS) {
    is_l1_fusion_enable_ = is_l1_fusion_enable == "true";
    GELOGD("The value of %s is %s.", kEnableL1Fusion.c_str(), is_l1_fusion_enable.c_str());
  } else {
    GELOGW("The value of %s is empty.", kEnableL1Fusion.c_str());
  }
}

Status ModelBuilder::BuildModelDef(ge::Model &model) {
  ClearOriginalFormat();

  max_mem_offset_ = mem_offset_;
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_MEMORY_SIZE, max_mem_offset_),
                   GELOGE(FAILED, "SetInt of ATTR_MODEL_MEMORY_SIZE failed.");
                   return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_STREAM_NUM, stream_num_),
                   GELOGE(FAILED, "SetInt of ATTR_MODEL_STREAM_NUM failed.");
                   return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_WEIGHT_SIZE, weight_offset_),
                   GELOGE(FAILED, "SetInt of ATTR_MODEL_WEIGHT_SIZE failed.");
                   return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_EVENT_NUM, event_num_),
                   GELOGE(FAILED, "SetInt of ATTR_MODEL_EVENT_NUM failed.");
                   return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_LABEL_NUM, label_num_),
                   GELOGE(FAILED, "SetInt of ATTR_MODEL_LABEL_NUM failed.");
                   return FAILED);
  string ge_core_type;
  Status ret = ge::GetContext().GetOption(kCoreType, ge_core_type);
  if (ret != SUCCESS) {
    GELOGW("get the option CORE_TYPE fail, set it to default value VECTOR_ENGINE");
  }
  int64_t core_type = (ge_core_type == kVectorCore) ? 1 : 0;
  GELOGI("core_type: %ld", core_type);
  if (!ge::AttrUtils::SetInt(&model, ATTR_MODEL_CORE_TYPE, core_type)) {
    GELOGE(FAILED, "SetInt of ATTR_CORE_TYPE failed.");
  }
  InitL1FusionOption();
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetBool(&model, ATTR_NAME_SWITCH_FOR_L1_FUSION, is_l1_fusion_enable_),
                   GELOGE(FAILED, "SetBool of ATTR_NAME_SWITCH_FOR_L1_FUSION failed.");
                   return FAILED);
  model.SetName(compute_graph_->GetName());
  model.SetGraph(ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph_));

  GELOGI("weight_offset_: %zu", weight_offset_);

  GELOGI("Set event num: %ld.", event_num_);

  if (Params::Instance() == nullptr) {
    return FAILED;
  }

  platform_type_ = Params::Instance()->GetTarget_8bit();
  return SUCCESS;
}

void ModelBuilder::ClearOriginalFormat() {
  for (const ge::NodePtr &n : compute_graph_->GetDirectNode()) {
    auto node_op_desc = n->GetOpDesc();
    if (node_op_desc != nullptr) {
      if (node_op_desc->HasAttr(ATTR_NAME_FORMAT)) {
        if (node_op_desc->DelAttr(ATTR_NAME_FORMAT) != SUCCESS) {
          GELOGW("DelAttr ATTR_NAME_FORMAT failed.");
        }
      }

      GE_IF_BOOL_EXEC(
        node_op_desc->HasAttr(ATTR_NAME_INFERRED_FORMAT),
        if (node_op_desc->DelAttr(ATTR_NAME_INFERRED_FORMAT) != SUCCESS) {
          GELOGW("DelAttr ATTR_NAME_INFERRED_FORMAT failed.");
        });

      GE_IF_BOOL_EXEC(
        node_op_desc->HasAttr(ATTR_NAME_PRED_PERMUTE_DELETED),
        if (node_op_desc->DelAttr(ATTR_NAME_PRED_PERMUTE_DELETED) != SUCCESS) {
          GELOGW("DelAttr ATTR_NAME_PRED_PERMUTE_DELETED failed.");
        });

      GE_IF_BOOL_EXEC(
        node_op_desc->HasAttr(ATTR_NAME_IGNORE_PRED_FORMAT),
        if (node_op_desc->DelAttr(ATTR_NAME_IGNORE_PRED_FORMAT) != SUCCESS) {
          GELOGW("DelAttr ATTR_NAME_IGNORE_PRED_FORMAT failed.");
        });
    }
  }
}

Status ModelBuilder::MergeWeights() {
  if (weight_offset_ == 0) {
    return SUCCESS;
  }

  ge::Buffer buffer(weight_offset_);
  weight_buffer_ = buffer;
  auto base_addr = weight_buffer_.GetData();

  for (const ge::NodePtr &node : compute_graph_->GetAllNodes()) {
    auto op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(op_desc == nullptr, continue);
    if (node->GetType() != CONSTANT) {
      continue;
    }

    // Get const op weight pointer
    ge::GeTensorPtr weight = nullptr;
    // If MutableTensor failed, weight is nullptr.
    (void)ge::AttrUtils::MutableTensor(op_desc, ATTR_NAME_WEIGHTS, weight);
    if (weight == nullptr) {
      GELOGE(FAILED, "Can't get const op weight, name: %s", node->GetName().c_str());
      return FAILED;
    }

    // Get const op weight offset
    int64_t offset = 0;
    if (ge::TensorUtils::GetDataOffset(weight->GetTensorDesc(), offset) != SUCCESS) {
      GELOGW("Can't get const op offset, name: %s", node->GetName().c_str());
      continue;  // continue to merge if can not get offset
    }

    // Get const op weight data
    auto weight_data = weight->MutableData();

    // copy const op weight data to buffer
    GELOGI("Move weight data to buffer, name: %s offset: %ld", node->GetName().c_str(), offset);
    ge::TensorUtils::SetWeightSize(weight->MutableTensorDesc(), static_cast<uint32_t>(weight_data.size()));
    if ((offset == 0) || (weight_data.size() == 0)) {
      GELOGI("Size or offset is 0. size: %lu  offset: %ld", weight_data.size(), offset);
      continue;
    }
    if (weight_data.data() != nullptr) {
      GE_IF_BOOL_EXEC(base_addr == nullptr, GELOGE(FAILED, "Base addr is nullptr."); return FAILED);
      GE_CHK_BOOL_EXEC(
        memcpy_s(base_addr + offset, weight_offset_ - offset, weight_data.data(), weight_data.size()) == EOK,
        return FAILED, "call memcpy_s failed.");
    }

    weight_data.clear();
  }

  return SUCCESS;
}

Status ModelBuilder::SaveDataToModel(ge::Model &model, ge::GeModel &ge_model) {
  // Add weight
  ge_model.SetWeight(weight_buffer_);

  // Add TBE Kernels
  for (const ge::NodePtr &n : compute_graph_->GetDirectNode()) {
    auto node_op_desc = n->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);
    TBEKernelPtr tbe_kernel = node_op_desc->TryGetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, TBEKernelPtr());
    GE_IF_BOOL_EXEC(tbe_kernel == nullptr, continue);
    tbe_kernel_store_.AddTBEKernel(tbe_kernel);
    GELOGD("Add tbe kernel bin %s", tbe_kernel->GetName().c_str());
  }
  if (!tbe_kernel_store_.Build()) {
    GELOGE(FAILED, "TBE Kernels store build failed!");
    return FAILED;
  }
  ge_model.SetTBEKernelStore(tbe_kernel_store_);

  // Add task
  GeAttrValue::BYTES task_def_bytes;
  if (!AttrUtils::GetZeroCopyBytes(model, MODEL_ATTR_TASKS, task_def_bytes)) {
    GELOGE(INTERNAL_ERROR, "Get zero copy bytes fail.");
    return INTERNAL_ERROR;
  }
  int byte_size = static_cast<int>(task_def_bytes.GetSize());
  std::shared_ptr<ModelTaskDef> task = ge::MakeShared<ModelTaskDef>();
  GE_CHECK_NOTNULL(task);
  GE_CHK_BOOL_EXEC(ReadProtoFromArray(task_def_bytes.GetData(), byte_size, task.get()), return INTERNAL_ERROR,
                   "ReadProtoFromArray failed.");
  ge_model.SetModelTaskDef(task);

  // Add graph
  ge_model.SetName(model.GetName());
  ge_model.SetGraph(model.GetGraph());
  ge_model.SetVersion(model.GetVersion());
  ge_model.SetPlatformVersion(model.GetPlatformVersion());
  ge_model.SetPlatformType(platform_type_);
  ge_model.SetAttr(model.MutableAttrMap());
  return SUCCESS;
}

void ModelBuilder::SetModelVersion(ge::Model &model) {
  // set framework_version TO model
  string framework_version;
  uint32_t counter = 0;
  Status frame_rt = PlatformVersionManager::GetPlatformVersion(framework_version);
  GE_IF_BOOL_EXEC((frame_rt == SUCCESS),
                  string model_framework_version = framework_version + "." + std::to_string(counter);
                  model.SetPlatformVersion(model_framework_version););

  // set IR Version TO model
  model.SetVersion(static_cast<uint32_t>(OM_PROTO_VERSION));
}

Status ModelBuilder::PreBuildModel() {
  if ((compute_graph_ == nullptr) || !(compute_graph_->IsValid())) {
    GELOGE(FAILED, "Graph_ is not valid.");
    return FAILED;
  }
  GELOGI("BuildModel begin.");

  GE_CHK_STATUS_RET(SetInputOutputDesc(), "SetInputOutputDesc Failed!");

  AddNodeInputProperty();

  return SUCCESS;
}

Status ModelBuilder::BuildModelForGetTask(ge::Model &model) {
  GE_CHK_STATUS_RET(AdjustInputTensorFlag(), "AdjustInputTensorFlag failed!");

  // Assign functional op labels.
  GE_TIMESTAMP_START(AssignFunctionalLabels);
  LabelAllocator label_allocator(compute_graph_);
  GE_CHK_STATUS_RET(label_allocator.AssignFunctionalLabels(label_num_), "Assign label failed.");
  GE_TIMESTAMP_END(AssignFunctionalLabels, "ModelBuilder::AssignFunctionalLabels");

  // Assign logical streams.
  StreamAllocator stream_allocator(compute_graph_, subgraphs_);
  GE_TIMESTAMP_START(AssignLogicalStreams);
  GE_CHK_STATUS_RET(stream_allocator.AssignLogicalStreams(stream_max_parallel_num_, hcom_parallel_),
                    "Assign logical streams failed.");
  GE_TIMESTAMP_END(AssignLogicalStreams, "GraphBuilder::AssignLogicalStreams");

  GE_TIMESTAMP_START(AssignMemory);
  MemoryAssigner mem_assigner(compute_graph_);
  GE_CHK_STATUS_RET(mem_assigner.AssignMemory(is_loop_graph_, mem_offset_), "Assign Memory Failed!");
  GE_TIMESTAMP_END(AssignMemory, "GraphBuilder::AssignMemory");

  // Compile single op in graph build stage
  GE_TIMESTAMP_START(CompileSingleOp);
  GE_CHK_STATUS_RET(CompileSingleOp(), "ATC builder CompileSingleOp() return fail.");
  GE_TIMESTAMP_END(CompileSingleOp, "GraphBuilder::CompileSingleOp");

  // Refresh real streams and insert event nodes.
  GE_TIMESTAMP_START(RefreshRealStream);
  GE_CHK_STATUS_RET(stream_allocator.RefreshRealStream(stream_num_, event_num_), "RefreshRealStream failed.");
  GE_TIMESTAMP_END(RefreshRealStream, "GraphBuilder::RefreshRealStream");

  GE_TIMESTAMP_START(MergeWeights);
  GE_CHK_STATUS_RET(MergeWeights(), "MergeWeights Failed!");
  GE_TIMESTAMP_END(MergeWeights, "GraphBuilder::MergeWeights");

  GE_TIMESTAMP_START(BuildModelDef);
  GE_CHK_STATUS_RET(BuildModelDef(model), "BuildModelDef failed!");
  GE_TIMESTAMP_END(BuildModelDef, "GraphBuilder::BuildModelDef");

  SetModelVersion(model);

  return SUCCESS;
}

ge::Buffer ModelBuilder::GetWeightBuffer() const { return weight_buffer_; }
Status ModelBuilder::CompileSingleOp() {
  GELOGD("Begin to compile single op.");
  // Create ge instance
  std::shared_ptr<GELib> instance = ge::GELib::GetInstance();
  if ((instance == nullptr) || !instance->InitFlag()) {
    GELOGE(ge::GE_CLI_GE_NOT_INITIALIZED, "CompileSingleOp failed.");
    return ge::GE_CLI_GE_NOT_INITIALIZED;
  }

  GE_TIMESTAMP_CALLNUM_START(BatchCompileOp);
  std::unordered_map<string, vector<ge::NodePtr>> node_vector_map;
  for (auto &node : compute_graph_->GetAllNodes()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }

    //  Graph build stage only supports the individual compilation of atomic clean operator
    if (op_desc->GetType() == ATOMICADDRCLEAN) {
      GELOGD("Begin to compile single op, op name is %s.", op_desc->GetName().c_str());
      string kernel_lib_name = op_desc->GetOpKernelLibName();
      if (kernel_lib_name.empty()) {
        // Reset op kernel lib
        (void)instance->DNNEngineManagerObj().GetDNNEngineName(op_desc);
        kernel_lib_name = op_desc->GetOpKernelLibName();
        if (kernel_lib_name.empty()) {
          GELOGE(ge::INTERNAL_ERROR, "Get node:%s(%s) kernel lib failed.", node->GetName().c_str(),
                 node->GetType().c_str());
          return ge::INTERNAL_ERROR;
        }
      }

      OpsKernelInfoStorePtr kernel_info = instance->OpsKernelManagerObj().GetOpsKernelInfoStore(kernel_lib_name);
      if (kernel_info != nullptr) {
        node_vector_map[kernel_lib_name].emplace_back(node);
      } else {
        GELOGE(ge::GE_GRAPH_PARAM_NULLPTR, "Get op %s ops kernel info store failed", node->GetName().c_str());
        return ge::GE_GRAPH_PARAM_NULLPTR;
      }
    }
  }
  for (auto &it : node_vector_map) {
    auto &kernel_lib_name = it.first;
    auto &node_vector = it.second;
    OpsKernelInfoStorePtr kernel_info = instance->OpsKernelManagerObj().GetOpsKernelInfoStore(kernel_lib_name);
    GE_CHECK_NOTNULL(kernel_info);
    GE_TIMESTAMP_RESTART(BatchCompileOp);
    auto ret = kernel_info->CompileOp(node_vector);
    GEEVENT("[GEPERFTRACE] The node size of compile op of %s is %zu", kernel_lib_name.c_str(), node_vector.size());
    GE_TIMESTAMP_ADD(BatchCompileOp);
    if (ret != ge::SUCCESS) {
      GELOGE(ret, "Compile op failed, kernel lib name is %s", kernel_lib_name.c_str());
      return ret;
    }
  }
  GE_TIMESTAMP_CALLNUM_END(BatchCompileOp, "GraphBuild::CompileOp");
  return ge::SUCCESS;
}
}  // namespace ge
