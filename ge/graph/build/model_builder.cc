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
#include "graph/build/model_builder.h"
#include <securectype.h>
#include <iostream>
#include <set>
#include <unordered_map>
#include "common/ge/ge_util.h"
#include "common/dump/dump_manager.h"
#include "framework/common/debug/ge_log.h"
#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/buffer.h"
#include "graph/build/stream_allocator.h"
#include "graph/common/omg_util.h"
#include "graph/common/ge_call_wrapper.h"
#include "graph/common/local_context.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_context.h"
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
#include "init/gelib.h"
#include "memory/memory_assigner.h"
#include "omg/version.h"
#include "register/op_registry.h"
#include "graph/passes/set_input_output_offset_pass.h"

using std::map;
using std::set;
using std::string;
using std::vector;

namespace {
const uint32_t kWeightsStartOffset = 512;
const int32_t kWrongIndex = -2;
const int kInvalidIndexNum = -1;

const char *const kVectorCore = "VectorCore";
const char *const kCoreType = "ge.engineType";
const std::string kEnableL1Fusion = "ge.l1Fusion";

const set<string> adjust_layer_type_ = {ge::CONVOLUTION};

bool IsGeLocalOp(const ge::ConstOpDescPtr &op_desc) {
  auto type = op_desc->GetType();
  if (type == ge::CONSTANTOP) {
    // constant op just has one output
    ge::GeTensorDesc output_desc = op_desc->GetOutputDesc(0);
    return !(output_desc.GetDataType() == ge::DT_STRING);
  }
  const set<string> ge_local_set = {ge::STREAMMERGE, ge::MEMCPYASYNC, ge::STREAMACTIVE,  ge::STREAMSWITCH,
                                    ge::VARIABLE,    ge::NOOP,        ge::CONSTANT,      ge::ENTER,
                                    ge::REFENTER,    ge::LOOPCOND,    ge::NEXTITERATION, ge::REFNEXTITERATION,
                                    ge::EXIT,        ge::REFEXIT,     ge::MERGE,         ge::MEMCPYADDRASYNC};
  return (ge_local_set.find(type) != ge_local_set.end());
}
}  // namespace

namespace ge {
ModelBuilder::ModelBuilder(uint64_t session_id, ge::ComputeGraphPtr compute_graph,
                           const Graph2SubGraphInfoList &subgraphs, const map<string, int> &stream_max_parallel_num,
                           bool hcom_parallel, int mode)
    : session_id_(session_id),
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
      p2p_mem_offset_(0),
      zero_copy_mem_size_(0),
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
      REPORT_CALL_ERROR("E19999", "Get tensor size in bytes failed for op:%s(%s) index:%u",
                        node_op_desc->GetName().c_str(), node_op_desc->GetType().c_str(), index);
      GELOGE(graph_status, "GetTensorMemorySizeInBytes failed!");
      return FAILED;
    }
    TensorUtils::SetSize(desc_temp, size_temp);
    if (node_op_desc->UpdateOutputDesc(index, desc_temp) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Update Output desc size failed for op:%s(%s) index:%u",
                        node_op_desc->GetName().c_str(), node_op_desc->GetType().c_str(), index);
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

bool ModelBuilder::SetInputConst(const OpDescPtr &op_desc, const NodePtr &src_node, size_t index,
                                 vector<bool> &is_input_const) {
  GELOGI("SetIsInputConst const: %s, source node: %s", op_desc->GetName().c_str(), src_node->GetName().c_str());
  for (size_t i = is_input_const.size(); i <= index; ++i) {
    is_input_const.push_back(false);
  }
  is_input_const[index] = true;

  vector<GeTensorPtr> weights = OpDescUtils::MutableWeights(src_node);
  if (weights.empty()) {
    GELOGW("SetInputIsConst weights is empty, node: %s", src_node->GetName().c_str());
    return false;
  }
  GeTensorPtr weight = weights[0];
  GE_IF_BOOL_EXEC(weight == nullptr, return true);
  GeTensorDesc &tensor_desc = weight->MutableTensorDesc();
  int64_t data_offset = 0;
  if (TensorUtils::GetDataOffset(tensor_desc, data_offset) != GRAPH_SUCCESS) {
    GELOGW("Get Offset from weight failed");
    return false;
  }
  auto input_tensor = op_desc->MutableInputDesc(static_cast<uint32_t>(index));
  if (input_tensor == nullptr) {
    GELOGW("Get input_tensor failed");
    return false;
  }
  TensorUtils::SetDataOffset(*input_tensor, data_offset);
  return true;
}

void ModelBuilder::SetInputIsConst(const ge::NodePtr &n) {
  auto node_op_desc = n->GetOpDesc();
  GE_CHECK_NOTNULL_JUST_RETURN(node_op_desc);

  auto is_input_const = node_op_desc->GetIsInputConst();

  // must set all true input_const to false
  for (size_t i = 0; i < is_input_const.size(); i++) {
    is_input_const[i] = false;
  }

  std::string const_type;
  auto in_data_anchors = n->GetAllInDataAnchors();
  for (size_t index = 0; index < in_data_anchors.size(); index++) {
    auto in_data_anchor = in_data_anchors.at(index);
    const auto &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    const auto &src_node = peer_out_anchor->GetOwnerNode();
    if (!NodeUtils::GetConstOpType(src_node, const_type)) {
      continue;
    }

    if (const_type == CONSTANT) {
      if (!SetInputConst(node_op_desc, src_node, index, is_input_const)) {
        return;
      }
    } else {
      if ((index < is_input_const.size()) && is_input_const[index]) {
        is_input_const[index] = false;
      }
    }
  }

  GELOGD("update opdesc:%s InputConst:%s", node_op_desc->GetName().c_str(), ToString(is_input_const).c_str());
  node_op_desc->SetIsInputConst(is_input_const);
}

Status ModelBuilder::AdjustConstWeightSize(const ge::NodePtr &node, size_t &mem_offset) {
  GE_CHECK_NOTNULL(node);
  if (node->GetType() == CONSTANT) {
    vector<GeTensorPtr> weights = OpDescUtils::MutableWeights(node);
    if (weights.empty()) {
      REPORT_INNER_ERROR("E19999", "Check weights size of node %s(%s) is empty",
                         node->GetName().c_str(), node->GetType().c_str());
      GELOGE(FAILED, "weights size of node %s is empty", node->GetName().c_str());
      return FAILED;
    }
    GeTensorPtr weight = weights[0];
    if (weight == nullptr) {
      REPORT_INNER_ERROR("E19999", "Check weight of node %s(%s) is nullptr",
                         node->GetName().c_str(), node->GetType().c_str());
      GELOGE(FAILED, "weights[0] is null.");
      return FAILED;
    }
    GeTensorDesc &tensor_desc = weight->MutableTensorDesc();
    size_t output_size = weight->GetData().size();
    TensorUtils::SetDataOffset(tensor_desc, mem_offset);
    GELOGD("Node: %s, weight size: %zu.", node->GetName().c_str(), output_size);
    mem_offset += output_size;
  }
  return SUCCESS;
}

Status ModelBuilder::SetInputOutputDesc() {
  Status ret;

  for (const ge::NodePtr &n : compute_graph_->GetNodes(compute_graph_->GetGraphUnknownFlag())) {
    auto node_op_desc = n->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);

    if (!is_loop_graph_ && node_op_desc->GetType() == LOOPCOND) {
      is_loop_graph_ = true;
    }
    // if user set input node format ND, the expected node for data and netoutput format is ND in
    // final graph.
    if ((compute_graph_->GetParentGraph() == nullptr) && (GetLocalOmgContext().format == domi::DOMI_TENSOR_ND) && (!node_op_desc->HasAttr("_is_single_op")) &&
        ((node_op_desc->GetType() == DATA_TYPE) || (node_op_desc->GetType() == NETOUTPUT))) {
      auto inputDescsPtr = node_op_desc->GetAllInputsDescPtr();
      auto outputDescsPtr = node_op_desc->GetAllOutputsDescPtr();
      ge::Format format = ge::FORMAT_ND;
      for (auto &inputDescPtr : inputDescsPtr) {
        GE_CHECK_NOTNULL(inputDescPtr);
        inputDescPtr->SetFormat(format);
        inputDescPtr->SetOriginFormat(format);
      }
      for (auto &outputDescPtr : outputDescsPtr) {
        GE_CHECK_NOTNULL(outputDescPtr);
        outputDescPtr->SetFormat(format);
        outputDescPtr->SetOriginFormat(format);
      }
    }

    if (node_op_desc->GetType() == DATA_TYPE || node_op_desc->GetType() == AIPP_DATA_TYPE) {
      GELOGD("Data node: %s.", n->GetName().c_str());
      continue;
    }

    GE_IF_BOOL_EXEC(n->GetInAllNodes().empty() && n->GetOutAllNodes().empty(), continue;);

    SetInputIsConst(n);
    bool is_unknow = false;
    (void)NodeUtils::GetNodeUnknownShapeStatus(*n, is_unknow);
    if ((IsGeLocalOp(n->GetOpDesc())) && (!is_unknow)) {
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
  for (const ge::NodePtr &node : compute_graph_->GetNodes(compute_graph_->GetGraphUnknownFlag())) {
    auto node_op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, GELOGW("node_op_desc is nullptr!"); return);
    vector<string> src_name_list;
    vector<int64_t> src_index_list;
    for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
      auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
      GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
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

  for (const ge::NodePtr &node : compute_graph_->GetNodes(compute_graph_->GetGraphUnknownFlag())) {
    auto node_op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, GELOGW("node_op_desc is nullptr!"); return);
    GE_IF_BOOL_EXEC(node_op_desc->GetType() == NETOUTPUT, continue);
    auto out_control_anchor = node->GetOutControlAnchor();
    GE_IF_BOOL_EXEC(out_control_anchor == nullptr, GELOGW("out_control_anchor is nullptr"); return);
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
        GE_IF_BOOL_EXEC(in_data_anchor == nullptr, GELOGW("in_data_anchor is nullptr"); return);
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
  for (const ge::NodePtr &n : compute_graph_->GetNodes(compute_graph_->GetGraphUnknownFlag())) {
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
            REPORT_CALL_ERROR("E19999", "Update Input desc size failed for op:%s(%s) index:%u",
                              owner_node_op_desc->GetName().c_str(), owner_node_op_desc->GetType().c_str(),
                              in_anchors->GetIdx());
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
  string buffer_optimize = "off_optimize";
  graphStatus ret = ge::GetContext().GetOption(BUFFER_OPTIMIZE, buffer_optimize);
  if (ret == GRAPH_SUCCESS) {
    bool off_superkernel = false;
    (void)AttrUtils::GetBool(compute_graph_, ATTR_NAME_OFF_SUPERKERNEL_ATTR, off_superkernel);
    is_l1_fusion_enable_ = ((buffer_optimize == "l1_optimize") && (!off_superkernel));
    GELOGI("Compute graph %s the value of %s is %s, superkernel flag %d.", compute_graph_->GetName().c_str(),
           BUFFER_OPTIMIZE.c_str(), buffer_optimize.c_str(), is_l1_fusion_enable_);
  } else {
    GELOGW("The value of %s is empty.", kEnableL1Fusion.c_str());
  }
}

Status ModelBuilder::BuildModelDef(ge::Model &model) {
  ClearOriginalFormat();

  max_mem_offset_ = mem_type_to_mem_offset_[RT_MEMORY_HBM];
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_MEMORY_SIZE, max_mem_offset_),
                   REPORT_INNER_ERROR("E19999", "Set Attr:%s in model failed",
                                      ATTR_MODEL_MEMORY_SIZE.c_str());
                   GELOGE(FAILED, "SetInt of ATTR_MODEL_MEMORY_SIZE failed.");
                   return FAILED);
  if (mem_type_to_mem_offset_.find(RT_MEMORY_P2P_DDR) != mem_type_to_mem_offset_.end()) {
    p2p_mem_offset_ = mem_type_to_mem_offset_[RT_MEMORY_P2P_DDR];
  }
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_P2P_MEMORY_SIZE, p2p_mem_offset_),
                   REPORT_INNER_ERROR("E19999", "Set Attr:%s in model failed",
                                      ATTR_MODEL_P2P_MEMORY_SIZE.c_str());
                   GELOGE(FAILED, "SetInt of ATTR_MODEL_P2P_MEMORY_SIZE failed.");
                       return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_WEIGHT_SIZE, weight_offset_),
                   REPORT_INNER_ERROR("E19999", "Set Attr:%s in model failed",
                                      ATTR_MODEL_WEIGHT_SIZE.c_str());
                   GELOGE(FAILED, "SetInt of ATTR_MODEL_WEIGHT_SIZE failed.");
                   return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_STREAM_NUM, stream_num_),
                   REPORT_INNER_ERROR("E19999", "Set Attr:%s in model failed",
                                      ATTR_MODEL_STREAM_NUM.c_str());
                   GELOGE(FAILED, "SetInt of ATTR_MODEL_STREAM_NUM failed.");
                   return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_EVENT_NUM, event_num_),
                   REPORT_INNER_ERROR("E19999", "Set Attr:%s in model failed",
                                      ATTR_MODEL_EVENT_NUM.c_str());
                   GELOGE(FAILED, "SetInt of ATTR_MODEL_EVENT_NUM failed.");
                   return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListInt(&model, ATTR_MODEL_HUGE_STREAM_LIST, huge_streams_),
                   REPORT_INNER_ERROR("E19999", "Set Attr:%s in model failed",
                                      ATTR_MODEL_HUGE_STREAM_LIST.c_str());
                   GELOGE(FAILED, "SetInt of ATTR_MODEL_HUGE_STREAM_LIST failed.");
                   return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_LABEL_NUM, label_num_),
                   REPORT_INNER_ERROR("E19999", "Set Attr:%s in model failed",
                                      ATTR_MODEL_LABEL_NUM.c_str());
                   GELOGE(FAILED, "SetInt of ATTR_MODEL_LABEL_NUM failed.");
                   return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(&model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, zero_copy_mem_size_),
                   REPORT_INNER_ERROR("E19999", "Set Attr:%s in model failed",
                                      ATTR_MODEL_ZERO_COPY_MEMORY_SIZE.c_str());
                   GELOGE(FAILED, "SetInt of ATTR_MODEL_ZERO_COPY_MEMORY_SIZE failed.");
                   return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListStr(&model, ATTR_MODEL_OUT_NODES_NAME, GetLocalOmgContext().net_out_nodes),
                   REPORT_INNER_ERROR("E19999", "Set Attr:%s in model failed",
                                      ATTR_MODEL_OUT_NODES_NAME.c_str());
                   GELOGE(FAILED, "SetListStr of ATTR_MODEL_OUT_NODES_NAME failed.");
                   return FAILED);
  GELOGI("For model, max_mem_offset_: %zu, p2p_mem_size: %zu, zero_copy_mem_size_: %zu", max_mem_offset_,
         p2p_mem_offset_, zero_copy_mem_size_);
  string fp_ceiling_mode;
  if (ge::GetContext().GetOption("ge.fpCeilingMode", fp_ceiling_mode) == SUCCESS) {
    if (!ge::AttrUtils::SetStr(&model, ATTR_FP_CEILING_MODE, fp_ceiling_mode)) {
      REPORT_INNER_ERROR("E19999", "Set Attr:%s in model failed",
                         ATTR_FP_CEILING_MODE.c_str());
      GELOGE(FAILED, "Failed to set attr ATTR_FP_CEILING_MODE");
      return FAILED;
    }
    GELOGI("Set attr ATTR_FP_CEILING_MODE to model, value is %s.", fp_ceiling_mode.c_str());
  }

  string ge_core_type;
  Status ret = ge::GetContext().GetOption(kCoreType, ge_core_type);
  if (ret != SUCCESS) {
    GELOGW("get the option CORE_TYPE fail, set it to default value VECTOR_ENGINE");
  }
  int64_t core_type = (ge_core_type == kVectorCore) ? 1 : 0;
  GELOGI("core_type: %ld", core_type);
  if (!ge::AttrUtils::SetInt(&model, ATTR_MODEL_CORE_TYPE, core_type)) {
    REPORT_INNER_ERROR("E19999", "Set Attr:%s in model failed",
                       ATTR_MODEL_CORE_TYPE.c_str());
    GELOGE(FAILED, "SetInt of ATTR_CORE_TYPE failed.");
  }
  InitL1FusionOption();
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetBool(&model, ATTR_NAME_SWITCH_FOR_L1_FUSION, is_l1_fusion_enable_),
                   REPORT_INNER_ERROR("E19999", "Set Attr:%s in model failed",
                                      ATTR_NAME_SWITCH_FOR_L1_FUSION.c_str());
                   GELOGE(FAILED, "SetBool of ATTR_NAME_SWITCH_FOR_L1_FUSION failed.");
                   return FAILED);
  const DumpProperties &dump_properties = DumpManager::GetInstance().GetDumpProperties(session_id_);
  bool is_op_debug = dump_properties.IsOpDebugOpen();
  if (is_op_debug) {
    if (!ge::AttrUtils::SetBool(&model, ATTR_OP_DEBUG_FLAG, is_op_debug)) {
      REPORT_INNER_ERROR("E19999", "Set Attr:%s in model failed",
                         ATTR_OP_DEBUG_FLAG.c_str());
      GELOGE(FAILED, "SetBool of ATTR_OP_DEBUG_FLAG failed.");
      return FAILED;
    }
    uint32_t op_debug_mode = dump_properties.GetOpDebugMode();
    GELOGI("Get op debug mode:%d", op_debug_mode);
    if (!ge::AttrUtils::SetInt(&model, ATTR_OP_DEBUG_MODE, op_debug_mode)) {
      REPORT_INNER_ERROR("E19999", "Set Attr:%s in model failed",
                         ATTR_OP_DEBUG_MODE.c_str());
      GELOGE(FAILED, "SetBool of ATTR_OP_DEBUG_MODE failed.");
      return FAILED;
    }
  }
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
  for (const ge::NodePtr &n : compute_graph_->GetNodes(compute_graph_->GetGraphUnknownFlag())) {
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

  for (const ge::NodePtr &node : compute_graph_->GetNodes(compute_graph_->GetGraphUnknownFlag())) {
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
      REPORT_INNER_ERROR("E19999", "Can't get const weight in op:%s(%s)",
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
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
    GELOGI("Move to buffer, name: %s offset: %ld size: %zu", node->GetName().c_str(), offset, weight_data.size());
    ge::TensorUtils::SetWeightSize(weight->MutableTensorDesc(), static_cast<uint32_t>(weight_data.size()));
    if ((offset == 0) || (weight_data.size() == 0)) {
      GELOGI("Size or offset is 0. size: %lu  offset: %ld", weight_data.size(), offset);
      continue;
    }
    if (weight_data.data() != nullptr) {
      GE_IF_BOOL_EXEC(base_addr == nullptr,
                      REPORT_INNER_ERROR("E19999", "Check weight in op:%s(%s) is nullptr",
                                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
                      GELOGE(FAILED, "Base addr is nullptr.");
                      return FAILED);
      if (weight_offset_ - offset < weight_data.size()) {
        REPORT_INNER_ERROR("E19999", "left weight size not enough for op:%s(%s) left_size:%zu, weight_size:%zu",
                           op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                           weight_offset_ - offset, weight_data.size());
        GELOGE(FAILED, "left weight size not enough. left_size:%lu, weight_size:%lu",
               weight_offset_ - offset, weight_data.size());
        return FAILED;
      }
      uintptr_t dst_ptr = reinterpret_cast<uintptr_t>(base_addr) + offset;
      uintptr_t src_ptr = reinterpret_cast<uintptr_t>(weight_data.data());
      size_t left_size = weight_data.size();
      while (left_size > SECUREC_MEM_MAX_LEN) {
        auto err = memcpy_s(reinterpret_cast<void *>(dst_ptr), SECUREC_MEM_MAX_LEN, reinterpret_cast<void *>(src_ptr),
                            SECUREC_MEM_MAX_LEN);
        if (err != EOK) {
          REPORT_CALL_ERROR("E19999", "mem copy failed. errret:%u, "
                            "dst_ptr:%lx, dst_size:%lu, src_ptr:%lx, src_size:%lu,",
                            err, dst_ptr, SECUREC_MEM_MAX_LEN, src_ptr, SECUREC_MEM_MAX_LEN);
          GELOGE(FAILED, "mem copy failed. errret:%u, "
                 "dst_ptr:%lx, dst_size:%lu, src_ptr:%lx, src_size:%lu",
                 err, dst_ptr, SECUREC_MEM_MAX_LEN, src_ptr, SECUREC_MEM_MAX_LEN);
          return FAILED;
        }
        left_size -= SECUREC_MEM_MAX_LEN;
        dst_ptr = dst_ptr + SECUREC_MEM_MAX_LEN;
        src_ptr = src_ptr + SECUREC_MEM_MAX_LEN;
      }
      auto err = memcpy_s(reinterpret_cast<void *>(dst_ptr), left_size, reinterpret_cast<void *>(src_ptr), left_size);
      if (err != EOK) {
        REPORT_CALL_ERROR("E19999", "mem copy failed. errret:%u, "
                          "dst_ptr:%lx, dst_size:%lu, src_ptr:%lx, src_size:%lu,",
                          err, dst_ptr, SECUREC_MEM_MAX_LEN, src_ptr, SECUREC_MEM_MAX_LEN);
        GELOGE(FAILED, "mem copy failed. errret:%u, "
               "dst_ptr:%lx, dst_size:%lu, src_ptr:%lx, src_size:%lu",
               err, dst_ptr, SECUREC_MEM_MAX_LEN, src_ptr, SECUREC_MEM_MAX_LEN);
        return FAILED;
      }
    }
    weight->ClearData();
  }

  return SUCCESS;
}

Status ModelBuilder::SaveAtomicTBEKernel(const OpDescPtr &op_desc) {
  ge::NodePtr atomic_clean_node = nullptr;
  atomic_clean_node = op_desc->TryGetExtAttr("atomic_clean_node_ptr", atomic_clean_node);
  if (atomic_clean_node == nullptr) {
    return SUCCESS;
  }

  ge::OpDescPtr atomic_op_desc = atomic_clean_node->GetOpDesc();
  GE_CHECK_NOTNULL(atomic_op_desc);
  TBEKernelPtr tbe_kernel = atomic_op_desc->TryGetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, TBEKernelPtr());
  if (tbe_kernel == nullptr) {
    std::string kernel_name;
    GeAttrValue::BYTES kernel_buffer;
    (void) AttrUtils::GetStr(atomic_op_desc, ATTR_NAME_TBE_KERNEL_NAME, kernel_name);
    (void) AttrUtils::GetBytes(atomic_op_desc, ATTR_NAME_TBE_KERNEL_BUFFER, kernel_buffer);
    if (!kernel_name.empty() && (kernel_buffer.GetSize() > 0)) {
      GE_CHECK_NOTNULL(kernel_buffer.GetData());
      std::vector<char> data(kernel_buffer.GetData(), kernel_buffer.GetData() + kernel_buffer.GetSize());
      tbe_kernel = MakeShared<OpKernelBin>(kernel_name, std::move(data));
      GE_CHECK_NOTNULL(tbe_kernel);
    }
  }
  if (tbe_kernel == nullptr) {
    GELOGD("Atomic_clean_node doesn't have tbe_kernel.");
    return SUCCESS;
  }
  tbe_kernel_store_.AddTBEKernel(tbe_kernel);
  GELOGD("Atomic_clean_node tbe_kernel_name %s!", tbe_kernel->GetName().c_str());
  (void) AttrUtils::SetStr(op_desc, ATOMIC_ATTR_TBE_KERNEL_NAME, tbe_kernel->GetName());

  std::string kernel_name;
  (void) AttrUtils::GetStr(atomic_op_desc, atomic_op_desc->GetName() + "_kernelname", kernel_name);
  (void) AttrUtils::SetStr(op_desc, op_desc->GetName() + "_atomic_kernelname", kernel_name);

  std::string meta_data;
  (void) AttrUtils::GetStr(atomic_op_desc, TVM_ATTR_NAME_METADATA, meta_data);
  (void) AttrUtils::SetStr(op_desc, ATOMIC_ATTR_TVM_METADATA, meta_data);

  std::string json_string;
  (void) AttrUtils::GetStr(atomic_op_desc, TVM_ATTR_NAME_MAGIC, json_string);
  (void) AttrUtils::SetStr(op_desc, ATOMIC_ATTR_TVM_MAGIC, json_string);
  return SUCCESS;
}

Status ModelBuilder::SaveDataToModel(ge::Model &model, ge::GeModel &ge_model) {
  // Add weight
  ge_model.SetWeight(weight_buffer_);

  // Add TBE Kernels and custom aicpu op bin
  std::set<std::string> tbe_name_set;
  std::set<std::string> aicpu_name_set;
  std::set<std::string> aicpu_op_types;
  std::set<std::string> aicpu_tf_op_types;
  for (const ge::NodePtr &n : compute_graph_->GetNodes(compute_graph_->GetGraphUnknownFlag())) {
    auto node_op_desc = n->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);
    // check aicpu op type
    CollectCheckAicpuAttr(node_op_desc, aicpu_op_types, aicpu_tf_op_types);
    TBEKernelPtr tbe_kernel = node_op_desc->TryGetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, TBEKernelPtr());
    if (tbe_kernel == nullptr) {
      std::string kernel_name;
      GeAttrValue::BYTES kernel_buffer;
      (void) AttrUtils::GetStr(node_op_desc, ATTR_NAME_TBE_KERNEL_NAME, kernel_name);
      (void) AttrUtils::GetBytes(node_op_desc, ATTR_NAME_TBE_KERNEL_BUFFER, kernel_buffer);
      if (!kernel_name.empty() && (kernel_buffer.GetSize() > 0)) {
        GE_CHECK_NOTNULL(kernel_buffer.GetData());
        std::vector<char> data(kernel_buffer.GetData(), kernel_buffer.GetData() + kernel_buffer.GetSize());
        tbe_kernel = std::make_shared<OpKernelBin>(kernel_name, std::move(data));
      }
    }
    GE_IF_BOOL_EXEC(tbe_kernel == nullptr, continue);
    if (tbe_name_set.count(tbe_kernel->GetName()) > 0) {
      REPORT_INNER_ERROR("E19999", "tbe_kernel name %s can't be the same, judge for op:%s(%s),",
                         tbe_kernel->GetName().c_str(), n->GetName().c_str(), n->GetType().c_str());
      GELOGE(FAILED, "tbe_kernel name %s can't be the same", tbe_kernel->GetName().c_str());
      return FAILED;
    }
    tbe_name_set.insert(tbe_kernel->GetName());
    tbe_kernel_store_.AddTBEKernel(tbe_kernel);

    GE_CHK_STATUS_RET(SaveAtomicTBEKernel(node_op_desc), "[Save][TBEKernel] save atomic tbekernel failed!");
  }

  SetModelCheckAicpuAttr(model, aicpu_op_types, aicpu_tf_op_types);

  for (const ge::NodePtr &n : compute_graph_->GetNodes(compute_graph_->GetGraphUnknownFlag())) {
    auto node_op_desc = n->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);
    CustAICPUKernelPtr cust_aicpu_kernel =
        node_op_desc->TryGetExtAttr(ge::OP_EXTATTR_CUSTAICPU_KERNEL, CustAICPUKernelPtr());
    GE_IF_BOOL_EXEC(cust_aicpu_kernel == nullptr, continue);
    if (aicpu_name_set.count(cust_aicpu_kernel->GetName()) > 0) {
      REPORT_INNER_ERROR("E19999", "aicpu_kernel name %s can't be the same, judge for op:%s(%s),",
                         cust_aicpu_kernel->GetName().c_str(), n->GetName().c_str(), n->GetType().c_str());
      GELOGE(FAILED, "aicpu_kernel name %s can't be the same", cust_aicpu_kernel->GetName().c_str());
      return FAILED;
    }
    aicpu_name_set.insert(cust_aicpu_kernel->GetName());
    cust_aicpu_kernel_store_.AddCustAICPUKernel(cust_aicpu_kernel);
    GELOGI("Add cust aicpu kernel bin %s", cust_aicpu_kernel->GetName().c_str());
  }

  if (!tbe_kernel_store_.Build()) {
    GELOGE(FAILED, "TBE Kernels store build failed!");
    return FAILED;
  }
  if (!cust_aicpu_kernel_store_.Build()) {
    GELOGE(FAILED, "custom AICPU kernels store build failed!");
    return FAILED;
  }
  ge_model.SetTBEKernelStore(tbe_kernel_store_);
  ge_model.SetCustAICPUKernelStore(cust_aicpu_kernel_store_);

  // Add task
  GeAttrValue::BYTES task_def_bytes;
  if (!AttrUtils::GetZeroCopyBytes(model, MODEL_ATTR_TASKS, task_def_bytes)) {
    REPORT_CALL_ERROR("E19999", "Get attr:%s in model failed", MODEL_ATTR_TASKS.c_str());
    GELOGE(INTERNAL_ERROR, "Get zero copy bytes fail.");
    return INTERNAL_ERROR;
  }
  int byte_size = static_cast<int>(task_def_bytes.GetSize());
  std::shared_ptr<domi::ModelTaskDef> task = ge::MakeShared<domi::ModelTaskDef>();
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
    REPORT_INNER_ERROR("E19999", "Check compute_graph no valid");
    GELOGE(FAILED, "Graph_ is not valid.");
    return FAILED;
  }

  GE_CHK_STATUS_RET(SetInputOutputDesc(), "SetInputOutputDesc Failed!");

  AddNodeInputProperty();

  return SUCCESS;
}

Status ModelBuilder::BuildModelForGetTask(ge::Model &model) {
  GE_CHK_STATUS_RET(AdjustInputTensorFlag(), "AdjustInputTensorFlag failed!");

  ErrorManager::GetInstance().SetStage(ErrorMessage::kModelCompile, ErrorMessage::kStreamAlloc);
  // Assign logical streams.
  StreamAllocator stream_allocator(compute_graph_, subgraphs_);
  GE_TIMESTAMP_START(AssignLogicalStreams);
  GE_CHK_STATUS_RET(stream_allocator.AssignLogicalStreams(stream_max_parallel_num_, hcom_parallel_),
                    "Assign logical streams failed.");
  GE_TIMESTAMP_END(AssignLogicalStreams, "GraphBuilder::AssignLogicalStreams");

  ErrorManager::GetInstance().SetStage(ErrorMessage::kModelCompile, ErrorMessage::kMemoryAlloc);
  // Assign functional op labels.
  auto root_graph = GraphUtils::FindRootGraph(compute_graph_);
  (void)AttrUtils::GetInt(*root_graph, ATTR_MODEL_LABEL_NUM, label_num_);

  GE_TIMESTAMP_START(AssignMemory);
  MemoryAssigner mem_assigner(compute_graph_);
  GE_CHK_STATUS_RET(mem_assigner.AssignMemory(is_loop_graph_, mem_type_to_mem_offset_, zero_copy_mem_size_),
                    "Assign Memory Failed!");
  GE_TIMESTAMP_END(AssignMemory, "GraphBuilder::AssignMemory");

  ErrorManager::GetInstance().SetStage(ErrorMessage::kModelCompile, ErrorMessage::kOther);
  GE_TIMESTAMP_START(SetInputOutputOffset);
  SetInputOutputOffsetPass input_output_offset;
  GE_CHK_STATUS_RET(input_output_offset.Run(compute_graph_), "Set input output offset failed.");
  GE_TIMESTAMP_END(SetInputOutputOffset, "SetInputOutputOffsetPass::Run");

  // Compile single op in graph build stage
  GE_TIMESTAMP_START(CompileSingleOp);
  GE_CHK_STATUS_RET(CompileSingleOp(), "ATC builder CompileSingleOp() return fail.");
  GE_TIMESTAMP_EVENT_END(CompileSingleOp, "GraphBuilder::CompileSingleOp");

  ErrorManager::GetInstance().SetStage(ErrorMessage::kModelCompile, ErrorMessage::kStreamAlloc);
  // Refresh real streams and insert event nodes.
  GE_TIMESTAMP_START(RefreshRealStream);
  GE_CHK_STATUS_RET(stream_allocator.RefreshRealStream(stream_num_, event_num_), "RefreshRealStream failed.");
  huge_streams_ = stream_allocator.GetHugeStreams();
  GE_TIMESTAMP_END(RefreshRealStream, "GraphBuilder::RefreshRealStream");

  ErrorManager::GetInstance().SetStage(ErrorMessage::kModelCompile, ErrorMessage::kOther);
  GE_TIMESTAMP_START(MergeWeights);
  GE_CHK_STATUS_RET(MergeWeights(), "MergeWeights Failed!");
  GE_TIMESTAMP_END(MergeWeights, "GraphBuilder::MergeWeights");

  GE_TIMESTAMP_START(BuildModelDef);
  GE_CHK_STATUS_RET(BuildModelDef(model), "BuildModelDef failed!");
  GE_TIMESTAMP_END(BuildModelDef, "GraphBuilder::BuildModelDef");

  SetModelVersion(model);

  return SUCCESS;
}

Status ModelBuilder::BuildModelForGetDynShapeTask(ge::Model &model_def) {
  GE_TIMESTAMP_START(BuildModelDef);
  GE_CHK_STATUS_RET(BuildModelDef(model_def), "BuildModelDef failed!");
  GE_TIMESTAMP_END(BuildModelDef, "GraphBuilder::BuildModelDef");
  SetModelVersion(model_def);
  return SUCCESS;
}

ge::Buffer ModelBuilder::GetWeightBuffer() const { return weight_buffer_; }
Status ModelBuilder::CompileSingleOp() {
  GELOGD("Begin to compile single op.");
  // Create ge instance
  std::shared_ptr<GELib> instance = ge::GELib::GetInstance();
  if ((instance == nullptr) || !instance->InitFlag()) {
    REPORT_INNER_ERROR("E19999", "Check GELib instance not init before");
    GELOGE(ge::GE_CLI_GE_NOT_INITIALIZED, "CompileSingleOp failed.");
    return ge::GE_CLI_GE_NOT_INITIALIZED;
  }

  GE_TIMESTAMP_CALLNUM_START(BatchCompileOp);
  std::unordered_map<string, vector<ge::NodePtr>> node_vector_map;
  for (auto &node : compute_graph_->GetNodes(compute_graph_->GetGraphUnknownFlag())) {
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
        (void)instance->DNNEngineManagerObj().GetDNNEngineName(node);
        kernel_lib_name = op_desc->GetOpKernelLibName();
        if (kernel_lib_name.empty()) {
          REPORT_INNER_ERROR("E19999", "Check kernel lib name empty of op:%s(%s)",
                             node->GetName().c_str(), node->GetType().c_str());
          GELOGE(ge::INTERNAL_ERROR, "Get node:%s(%s) kernel lib failed.", node->GetName().c_str(),
                 node->GetType().c_str());
          return ge::INTERNAL_ERROR;
        }
      }

      OpsKernelInfoStorePtr kernel_info = instance->OpsKernelManagerObj().GetOpsKernelInfoStore(kernel_lib_name);
      if (kernel_info != nullptr) {
        node_vector_map[kernel_lib_name].emplace_back(node);
      } else {
        REPORT_INNER_ERROR("E19999", "Get ops kernel info store failed for op:%s(%s), op_kernel_name:%s,",
                           node->GetName().c_str(), node->GetType().c_str(), kernel_lib_name.c_str());
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
    GELOGI("[GEPERFTRACE] The node size of compile op of %s is %zu", kernel_lib_name.c_str(), node_vector.size());
    GE_TIMESTAMP_ADD(BatchCompileOp);
    if (ret != ge::SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Batch compile op failed, kernel lib name, node size:%zu,",
                        node_vector.size());
      GELOGE(ret, "Compile op failed, kernel lib name is %s", kernel_lib_name.c_str());
      return ret;
    }
  }
  GE_TIMESTAMP_CALLNUM_END(BatchCompileOp, "GraphBuild::CompileOp");
  return ge::SUCCESS;
}

void ModelBuilder::CollectCheckAicpuAttr(const OpDescPtr &op_desc, std::set<std::string> &aicpu_op_types,
                                         std::set<std::string> &aicpu_tf_op_types) {
  std::string aicpu_optype;
  bool has_attr_check_cpu = ge::AttrUtils::GetStr(op_desc, "needCheckCpu", aicpu_optype);
  std::vector<std::string> tf_optypes;
  bool has_attr_check_tf = ge::AttrUtils::GetListStr(op_desc, "needCheckTf", tf_optypes);
  if (has_attr_check_cpu && !aicpu_optype.empty()) {
    aicpu_op_types.insert(aicpu_optype);
  }

  if (has_attr_check_tf && !tf_optypes.empty()) {
    aicpu_tf_op_types.insert(tf_optypes.begin(), tf_optypes.end());
  }

  return;
}

void ModelBuilder::SetModelCheckAicpuAttr(ge::Model &model, std::set<std::string> &aicpu_op_types,
                                          std::set<std::string> &aicpu_tf_op_types) {
  std::vector<std::string> aicpu_optype_list;
  std::vector<std::string> aicpu_tf_optype_list;
  if (ge::AttrUtils::GetListStr(&model, "needCheckCpu", aicpu_optype_list)) {
    GELOGI("Already have aicpu optype size: %zu", aicpu_optype_list.size());
    aicpu_op_types.insert(aicpu_optype_list.begin(), aicpu_optype_list.end());
  }

  if (ge::AttrUtils::GetListStr(&model, "needCheckTf", aicpu_tf_optype_list)) {
    GELOGI("Already have aicpu tf optype size: %zu", aicpu_tf_optype_list.size());
    aicpu_tf_op_types.insert(aicpu_tf_optype_list.begin(), aicpu_tf_optype_list.end());
  }

  // reset list with set
  aicpu_optype_list.assign(aicpu_op_types.begin(), aicpu_op_types.end());
  aicpu_tf_optype_list.assign(aicpu_tf_op_types.begin(), aicpu_tf_op_types.end());
  GELOGI(
    "Check Aicpu op types ComputeGraph: %s aicpu_op_types: %zu, aicpu_optype_list: %zu, aicpu_tf_op_types: %zu, "
    "aicpu_tf_optype_list:%zu.",
    compute_graph_->GetName().c_str(), aicpu_op_types.size(), aicpu_optype_list.size(), aicpu_tf_op_types.size(),
    aicpu_tf_optype_list.size());
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListStr(&model, "needCheckCpu", aicpu_optype_list), return,
                   "Set attr needCheckCpu fail.");

  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListStr(&model, "needCheckTf", aicpu_tf_optype_list), return,
                   "Set attr needCheckTf fail.");
  return;
}
}  // namespace ge
