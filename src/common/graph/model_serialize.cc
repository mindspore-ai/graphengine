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

#include "graph/model_serialize.h"
#include <google/protobuf/text_format.h>

#include <queue>
#include <iostream>

#include "debug/ge_attr_define.h"
#include "debug/ge_log.h"
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/detail/model_serialize_imp.h"
#include "proto/ge_ir.pb.h"
#include "utils/graph_utils.h"
#include "debug/ge_op_types.h"

using std::map;
using std::string;

namespace ge {
bool ModelSerializeImp::ParseNodeIndex(const string &node_index, string &node_name, int32_t &index) {
  auto sep = node_index.rfind(":");
  if (sep == string::npos) {
    GELOGW("separator is not found in node_index.");
    return false;
  }
  node_name = node_index.substr(0, sep);
  auto index_str = node_index.substr(sep + 1);
  index = static_cast<int32_t>(std::strtol(index_str.c_str(), nullptr, 10));
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ModelSerializeImp::SerializeTensor(const ConstGeTensorPtr &tensor,
                                                                                       proto::TensorDef *tensor_proto) {
  GE_CHK_BOOL_EXEC(tensor != nullptr, return false, "tensor is null.");
  GE_CHK_BOOL_EXEC(tensor_proto != nullptr, return false, "tensor_proto is null.");

  if (tensor->tensor_def_.GetProtoMsg() != nullptr) {
    *tensor_proto = *tensor->tensor_def_.GetProtoMsg();
    return true;
  }
  return false;
}

bool ModelSerializeImp::SerializeEdge(const NodePtr &node, proto::OpDef *op_def_proto) {
  GE_CHK_BOOL_EXEC(node != nullptr, return false, "node is null.");
  GE_CHK_BOOL_EXEC(op_def_proto != nullptr, return false, "op_def_proto is null.");

  op_def_proto->clear_input();
  // Inputs
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    if (in_data_anchor != nullptr) {
      auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
      if (peer_out_anchor != nullptr && peer_out_anchor->GetOwnerNode()) {
        op_def_proto->add_input(peer_out_anchor->GetOwnerNode()->GetName() + ":" +
                                std::to_string(peer_out_anchor->GetIdx()));
      } else {
        op_def_proto->add_input("");
      }
    }
  }
  // Control edge
  auto control_anchor = node->GetInControlAnchor();
  if (control_anchor != nullptr) {
    auto peer_out_anchors = control_anchor->GetPeerOutControlAnchors();
    for (const auto &peer_out_anchor : peer_out_anchors) {
      if (peer_out_anchor != nullptr && peer_out_anchor->GetOwnerNode()) {
        op_def_proto->add_input(peer_out_anchor->GetOwnerNode()->GetName() + ":-1");
      }
    }
  }
  return true;
}

bool ModelSerializeImp::SerializeOpDesc(const ConstOpDescPtr &op_desc, proto::OpDef *op_def_proto, bool is_dump) {
  GE_CHK_BOOL_EXEC(op_desc != nullptr, return false, "op_desc is null.");
  GE_CHK_BOOL_EXEC(op_def_proto != nullptr, return false, "op_def_proto is null.");
  if (op_desc->op_def_.GetProtoMsg() != nullptr) {
    *op_def_proto = *op_desc->op_def_.GetProtoMsg();
    // Delete unnecessary attr
    if (is_dump) {
      auto attr = op_def_proto->mutable_attr();
      attr->erase(ATTR_NAME_FRAMEWORK_NODE_DEF);
      attr->erase(ATTR_NAME_FRAMEWORK_OP_DEF);
      attr->erase(ATTR_NAME_FRAMEWORK_FUNC_DEF);
      GE_IF_BOOL_EXEC((op_def_proto->type() == CONSTANT || op_def_proto->type() == CONSTANTOP),
                      attr->erase(ATTR_NAME_WEIGHTS));
    }
    op_def_proto->clear_input_desc();
    op_def_proto->clear_output_desc();
    // Input descs
    if (op_desc->GetAllInputsSize() > 0) {
      auto size = static_cast<uint32_t>(op_desc->GetAllInputsSize());
      for (uint32_t i = 0; i < size; i++) {
        auto tensor_desc = op_desc->GetInputDescPtrDfault(i);
        if (tensor_desc != nullptr && tensor_desc->tensor_descriptor_.GetProtoMsg() != nullptr) {
          *op_def_proto->add_input_desc() = *(tensor_desc->tensor_descriptor_.GetProtoMsg());
        }
      }
    }
    // Output descs
    if (op_desc->GetOutputsSize() > 0) {
      auto size = static_cast<uint32_t>(op_desc->GetOutputsSize());
      for (uint32_t i = 0; i < size; i++) {
        auto tensor_desc = op_desc->GetOutputDescPtr(i);
        if (tensor_desc != nullptr && tensor_desc->tensor_descriptor_.GetProtoMsg() != nullptr) {
          *op_def_proto->add_output_desc() = *(tensor_desc->tensor_descriptor_.GetProtoMsg());
        }
      }
    }

    op_def_proto->set_id(op_desc->GetId());
    for (const std::string &name : op_desc->GetSubgraphInstanceNames()) {
      op_def_proto->add_subgraph_name(name);
    }
    OpDescToAttrDef(op_desc, op_def_proto);
  }
  return true;
}

void ModelSerializeImp::OpDescToAttrDef(const ConstOpDescPtr &op_desc, proto::OpDef *op_def_proto) {
  proto::AttrDef key_in;
  proto::AttrDef value_in;
  auto op_desc_attr = op_def_proto->mutable_attr();
  if (!op_desc->input_name_idx_.empty()) {
    for (auto &item : op_desc->input_name_idx_) {
      key_in.mutable_list()->add_s(item.first);
      value_in.mutable_list()->add_i(item.second);
    }
    op_desc_attr->insert({"_input_name_key", key_in});
    op_desc_attr->insert({"_input_name_value", value_in});
  }
  proto::AttrDef key_out;
  proto::AttrDef value_out;
  if (!op_desc->output_name_idx_.empty()) {
    for (auto &item : op_desc->output_name_idx_) {
      key_out.mutable_list()->add_s(item.first);
      value_out.mutable_list()->add_i(item.second);
    }
    op_desc_attr->insert({"_output_name_key", key_out});
    op_desc_attr->insert({"_output_name_value", value_out});
  }
  proto::AttrDef opt_input;
  if (!op_desc->optional_input_names_.empty()) {
    for (auto &item : op_desc->optional_input_names_) {
      opt_input.mutable_list()->add_s(item);
    }
    op_desc_attr->insert({"_opt_input", opt_input});
  }
}

bool ModelSerializeImp::SerializeNode(const NodePtr &node, proto::OpDef *op_def_proto, bool is_dump) {
  if (node == nullptr || op_def_proto == nullptr) {
    GELOGE(GRAPH_FAILED, "Input Para Node Invalid");
    return false;
  }
  if (!SerializeOpDesc(node->GetOpDesc(), op_def_proto, is_dump)) {
    GELOGE(GRAPH_FAILED, "Serialize OpDesc failed");
    return false;
  }
  if (SerializeEdge(node, op_def_proto)) {
    return true;
  } else {
    return false;
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ModelSerializeImp::SerializeGraph(const ConstComputeGraphPtr &graph,
                                                                                      proto::GraphDef *graph_proto,
                                                                                      bool is_dump) {
  if (graph == nullptr || graph_proto == nullptr) {
    GELOGE(GRAPH_FAILED, "Input para Invalid");
    return false;
  }
  graph_proto->set_name(graph->GetName());
  // Inputs
  for (const auto &input : graph->GetInputNodes()) {
    if (input != nullptr) {
      graph_proto->add_input(input->GetName() + ":0");
    }
  }
  // Outputs
  for (const auto &output : graph->GetGraphOutNodesInfo()) {
    if (output.first != nullptr) {
      graph_proto->add_output(output.first->GetName() + ":" + std::to_string(output.second));
      GELOGI("Add output to graph proto, node name:%s, index:%ld", output.first->GetName().c_str(), output.second);
    }
  }
  if (graph->attrs_.GetProtoMsg() != nullptr) {
    *graph_proto->mutable_attr() = *graph->attrs_.GetProtoMsg();
  }
  for (const auto &node : graph->GetDirectNode()) {
    if (!SerializeNode(node, graph_proto->add_op(), is_dump)) {
      if (node->GetOpDesc() != nullptr) {
        GELOGE(GRAPH_FAILED, "Serialize Node %s failed", node->GetName().c_str());
      }
      return false;
    }
  }
  return true;
}

bool ModelSerializeImp::SerializeModel(const Model &model, proto::ModelDef *model_proto, bool is_dump) {
  if (model_proto == nullptr) {
    GELOGE(GRAPH_FAILED, "model_proto para Invalid");
    return false;
  }
  model_proto->set_name(model.GetName());
  model_proto->set_custom_version(model.GetPlatformVersion());
  model_proto->set_version(model.GetVersion());
  if (model.attrs_.GetProtoMsg()) {
    *model_proto->mutable_attr() = *model.attrs_.GetProtoMsg();
  }
  auto &graph = model.graph_;
  auto compute_graph = GraphUtils::GetComputeGraph(graph);
  if (compute_graph == nullptr) {
    GELOGE(GRAPH_FAILED, "GetComputeGraph return nullptr");
    return false;
  }
  if (!SerializeGraph(compute_graph, model_proto->add_graph(), is_dump)) {
    GELOGE(GRAPH_FAILED, "SerializeGraph fail");
    return false;
  }

  for (auto subgraph : compute_graph->GetAllSubgraphs()) {
    if (!SerializeGraph(subgraph, model_proto->add_graph(), is_dump)) {
      GELOGE(GRAPH_FAILED, "Serialize subgraph failed");
      return false;
    }
  }

  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ModelSerializeImp::UnserializeTensor(
  GeTensorPtr &tensor, proto::TensorDef &tensor_proto) {
  tensor = std::shared_ptr<GeTensor>(new (std::nothrow) GeTensor(protobuf_owner_, &tensor_proto));
  if (tensor == nullptr) {
    GELOGE(GRAPH_FAILED, "tensor is nullptr");
    return false;
  } else {
    return true;
  }
}

void ModelSerializeImp::AttrDefToOpDesc(OpDescPtr &op_desc, std::vector<string> &key_in, std::vector<string> &key_out,
                                        std::vector<uint32_t> &value_in, std::vector<uint32_t> &value_out,
                                        std::vector<string> &opt_input) {
  if (!key_in.empty()) {
    if (key_in.size() != value_in.size()) {
      GELOGW("Key and value vector size is different. key_size: %zu, value_size: %zu.", key_out.size(),
             value_in.size());
    } else {
      for (uint32_t i = 0; i < key_in.size(); ++i) {
        op_desc->input_name_idx_.insert(std::pair<string, uint32_t>(key_in.at(i), value_in.at(i)));
      }
    }
  }
  if (!key_out.empty()) {
    if (key_out.size() != value_out.size()) {
      GELOGW("Key and value vector size is different. key_size: %zu, value_size: %zu.", key_out.size(),
             value_out.size());
    } else {
      for (uint32_t i = 0; i < key_out.size(); ++i) {
        op_desc->output_name_idx_.insert(std::pair<string, uint32_t>(key_out.at(i), value_out.at(i)));
      }
    }
  }
  if (!opt_input.empty()) {
    for (const auto &i : opt_input) {
      op_desc->optional_input_names_.insert(i);
    }
  }
}

bool ModelSerializeImp::UnserializeOpDesc(OpDescPtr &op_desc, proto::OpDef &op_def_proto) {
  std::vector<string> opt_input;
  std::vector<string> key_in;
  std::vector<uint32_t> value_in;
  if (op_def_proto.attr().count("_opt_input") > 0) {
    auto &name_list = op_def_proto.attr().at("_opt_input").list();
    for (const auto &item_s : name_list.s()) {
      opt_input.push_back(item_s);
    }
    auto op_desc_attr = op_def_proto.mutable_attr();
    op_desc_attr->erase("_opt_input");
  }
  if (op_def_proto.attr().count("_input_name_key") > 0) {
    auto &output_name_key_list = op_def_proto.attr().at("_input_name_key").list();
    for (const auto &item_s : output_name_key_list.s()) {
      key_in.push_back(item_s);
    }
    auto op_desc_attr = op_def_proto.mutable_attr();
    op_desc_attr->erase("_input_name_key");
  }
  if (op_def_proto.attr().count("_input_name_value") > 0) {
    auto &input_name_value_list = op_def_proto.attr().at("_input_name_value").list();
    for (const auto &item_i : input_name_value_list.i()) {
      value_in.push_back(static_cast<uint32_t>(item_i));
    }
    auto op_desc_attr = op_def_proto.mutable_attr();
    op_desc_attr->erase("_input_name_value");
  }
  std::vector<string> key_out;
  std::vector<uint32_t> value_out;
  if (op_def_proto.attr().count("_output_name_key") > 0) {
    auto &output_name_key_list = op_def_proto.attr().at("_output_name_key").list();
    for (const auto &item_s : output_name_key_list.s()) {
      key_out.push_back(item_s);
    }
    auto op_desc_attr = op_def_proto.mutable_attr();
    op_desc_attr->erase("_output_name_key");
  }
  if (op_def_proto.attr().count("_output_name_value") > 0) {
    auto &output_name_value_list = op_def_proto.attr().at("_output_name_value").list();
    for (const auto &item_i : output_name_value_list.i()) {
      value_out.push_back(static_cast<uint32_t>(item_i));
    }
    auto op_desc_attr = op_def_proto.mutable_attr();
    op_desc_attr->erase("_output_name_value");
  }

  op_desc = std::shared_ptr<OpDesc>(new (std::nothrow) OpDesc(protobuf_owner_, &op_def_proto));
  GE_CHK_BOOL_EXEC(op_desc != nullptr, return false, "op_desc is nullptr.");

  // Input tensor
  for (auto &input_desc : *op_def_proto.mutable_input_desc()) {
    std::shared_ptr<GeTensorDesc> temp_value =
      std::shared_ptr<GeTensorDesc>(new (std::nothrow) GeTensorDesc(protobuf_owner_, &input_desc));
    GE_CHK_BOOL_RET_STATUS(temp_value != nullptr, false, "temp_value is nullptr");
    op_desc->inputs_desc_.push_back(temp_value);
  }
  // Output tensor
  for (auto &output_desc : *op_def_proto.mutable_output_desc()) {
    std::shared_ptr<GeTensorDesc> temp_value =
      std::shared_ptr<GeTensorDesc>(new (std::nothrow) GeTensorDesc(protobuf_owner_, &output_desc));
    GE_CHK_BOOL_RET_STATUS(temp_value != nullptr, false, "temp_value is nullptr");
    op_desc->outputs_desc_.push_back(temp_value);
  }

  op_desc->SetId(op_def_proto.id());
  uint32_t graph_index = 0;
  for (const std::string &name : op_def_proto.subgraph_name()) {
    op_desc->AddSubgraphName(name);
    op_desc->SetSubgraphInstanceName(graph_index++, name);
  }

  // insert name index by key and value
  AttrDefToOpDesc(op_desc, key_in, key_out, value_in, value_out, opt_input);

  return true;
}

bool ModelSerializeImp::UnserializeNode(ComputeGraphPtr &graph, proto::OpDef &op_def_proto) {
  GE_RT_FALSE_CHECK_NOTNULL(graph);
  OpDescPtr op_desc = nullptr;
  if (!UnserializeOpDesc(op_desc, op_def_proto)) {
    GELOGW("UnserializeOpDesc error.");
  }

  NodePtr node = graph->AddNode(op_desc, op_desc->GetId());
  GE_CHK_BOOL_EXEC(node != nullptr, return false, "node is nullptr.");

  // Inputs
  int dst_index = 0;
  for (const auto &input : op_def_proto.input()) {
    string node_name;
    int32_t index = 0;
    if (ParseNodeIndex(input, node_name, index)) {
      node_input_node_names_.push_back(NodeNameNodeReq{node_name, index, node, dst_index, op_def_proto.name()});
    }
    if (index >= 0) {
      dst_index++;
    }
  }
  node_map_[op_def_proto.name()] = node;
  return true;
}

bool ModelSerializeImp::HandleNodeNameRef() {
  // Edges
  for (auto &item : node_input_node_names_) {
    auto src_node_it = node_map_.find(item.src_node_name);
    if (src_node_it == node_map_.end()) {
      GELOGE(GRAPH_FAILED, "cannot find node %s", item.src_node_name.c_str());
      return false;
    }
    GE_IF_BOOL_EXEC(src_node_it->second == nullptr || item.dst_node == nullptr, continue);
    if (item.src_out_index >= 0) {
      auto src_anchor = src_node_it->second->GetOutDataAnchor(item.src_out_index);
      auto dst_anchor = item.dst_node->GetInDataAnchor(item.dst_in_index);
      if (src_anchor == nullptr || dst_anchor == nullptr) {
        GELOGE(GRAPH_FAILED, "get anchor failed %s:%d, %s:%d ", item.src_node_name.c_str(), item.src_out_index,
               item.dst_node_name.c_str(), item.dst_in_index);
        return false;
      }
      GE_CHK_BOOL_ONLY_LOG((src_anchor->LinkTo(dst_anchor) == GRAPH_SUCCESS), " linkTo failed.");  // lint !e737
    } else {
      // Control edge
      auto src_anchor = src_node_it->second->GetOutControlAnchor();
      auto dst_anchor = item.dst_node->GetInControlAnchor();
      if (src_anchor != nullptr && dst_anchor != nullptr) {
        GE_CHK_BOOL_ONLY_LOG((src_anchor->LinkTo(dst_anchor) == GRAPH_SUCCESS), " linkTo failed.");  // lint !e737
      }
    }
  }
  // Graph input
  for (auto &item : graph_input_node_names_) {
    auto node_it = node_map_.find(item.node_name);
    if (node_it == node_map_.end()) {
      GELOGE(GRAPH_FAILED, "cannot find node %s", item.node_name.c_str());
      return false;
    }
    GE_IF_BOOL_EXEC(item.graph == nullptr, continue);
    auto ret = item.graph->AddInputNode(node_it->second);
    if (ret == nullptr) {
      return false;
    }
  }
  // Graph output
  for (auto &item : graph_output_node_names_) {
    auto node_it = node_map_.find(item.node_name);
    if (node_it == node_map_.end()) {
      GELOGE(GRAPH_FAILED, "cannot find node %s", item.node_name.c_str());
      return false;
    }

    GE_IF_BOOL_EXEC(item.graph == nullptr, continue);
    auto ret = item.graph->AddOutputNodeByIndex(node_it->second, item.index);
    GELOGI("node name:%s, item.index:%ld", node_it->second->GetName().c_str(), item.index);
    if (ret == nullptr) {
      GELOGE(GRAPH_FAILED, "AddOutputNode failed.");
      return false;
    }
  }
  node_input_node_names_.clear();
  graph_input_node_names_.clear();
  graph_output_node_names_.clear();
  node_map_.clear();
  return true;
}

bool ModelSerializeImp::RebuildOwnership(ComputeGraphPtr &compute_graph, map<string, ComputeGraphPtr> &subgraphs) {
  std::queue<ComputeGraphPtr> all_graphs;
  all_graphs.emplace(compute_graph);
  while (!all_graphs.empty()) {
    ComputeGraphPtr graph = all_graphs.front();
    all_graphs.pop();

    for (const NodePtr &node : graph->GetDirectNode()) {
      const OpDescPtr op_desc = node->GetOpDesc();
      for (const std::string &name : op_desc->GetSubgraphInstanceNames()) {
        auto it = subgraphs.find(name);
        if (it == subgraphs.end()) {
          GELOGE(GRAPH_FAILED, "Node:%s, Subgraph:%s not found, num:%zu.", op_desc->GetName().c_str(), name.c_str(),
                 subgraphs.size());
          return false;
        }

        ComputeGraphPtr &subgraph = it->second;
        subgraph->SetParentGraph(graph);
        subgraph->SetParentNode(node);
        compute_graph->AddSubgraph(subgraph->GetName(), subgraph);
        all_graphs.emplace(subgraph);
      }
    }
  }

  return true;
}

bool ModelSerializeImp::UnserializeModel(Model &model, proto::ModelDef &model_proto) {
  model.name_ = model_proto.name();
  model.version_ = model_proto.version();
  model.platform_version_ = model_proto.custom_version();
  model.attrs_ = ProtoAttrMapHelper(protobuf_owner_, model_proto.mutable_attr());

  auto &graphs_proto = *model_proto.mutable_graph();
  if (!graphs_proto.empty()) {
    auto &graph_proto = graphs_proto[0];
    ComputeGraphPtr compute_graph_ptr;
    if (UnserializeGraphWithoutEdge(compute_graph_ptr, graph_proto)) {
      model.graph_ = GraphUtils::CreateGraphFromComputeGraph(compute_graph_ptr);
    }

    // 0 is main graph, following is subgraph.
    map<string, ComputeGraphPtr> subgraphs;
    for (int idx = 1; idx < graphs_proto.size(); ++idx) {
      ComputeGraphPtr subgraph;
      ModelSerializeImp impl;
      if (!impl.UnserializeGraphWithoutEdge(subgraph, graphs_proto[idx])) {
        GELOGE(GRAPH_FAILED, "UnserializeGraphWithoutEdge failed");
        return false;
      }

      if (!impl.HandleNodeNameRef()) {
        GELOGE(GRAPH_FAILED, "HandleNodeNameRef failed");
        return false;
      }

      subgraphs[subgraph->GetName()] = subgraph;
    }

    if (!RebuildOwnership(compute_graph_ptr, subgraphs)) {
      GELOGE(GRAPH_FAILED, "Rebuild graph ownership failed");
      return false;
    }
  }

  if (!HandleNodeNameRef()) {
    GELOGE(GRAPH_FAILED, "HandleNodeNameRef failed");
    return false;
  }
  return true;
}

bool ModelSerializeImp::UnserializeGraphWithoutEdge(ComputeGraphPtr &graph, proto::GraphDef &graph_proto) {
  graph = ComGraphMakeShared<ComputeGraph>(graph_proto.name());
  if (graph == nullptr) {
    GELOGE(GRAPH_FAILED, "ComputeGraph make shared failed");
    return false;
  }

  // Inputs
  for (auto input : graph_proto.input()) {
    string node_name;
    int32_t index;
    if (ParseNodeIndex(input, node_name, index)) {
      graph_input_node_names_.push_back(NodeNameGraphReq{node_name, index, graph});
    }
  }
  // Outputs
  for (auto output : graph_proto.output()) {
    string node_name;
    int32_t index;
    if (ParseNodeIndex(output, node_name, index)) {
      graph_output_node_names_.push_back(NodeNameGraphReq{node_name, index, graph});
    }
  }
  graph->attrs_ = ProtoAttrMapHelper(protobuf_owner_, graph_proto.mutable_attr());
  for (auto &op_def_proto : *graph_proto.mutable_op()) {
    if (!UnserializeNode(graph, op_def_proto)) {
      GELOGE(GRAPH_FAILED, "UnserializeNode fail");
      return false;
    }
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ModelSerializeImp::UnserializeGraph(ComputeGraphPtr &graph,
                                                                                        proto::GraphDef &graph_proto) {
  if (!UnserializeGraphWithoutEdge(graph, graph_proto)) {
    GELOGW("UnserializeGraphWithoutEdge fail");
  }
  if (!HandleNodeNameRef()) {
    GELOGE(GRAPH_FAILED, "Link Anchor or set graph input or output fail");
    return false;
  }
  return true;
}

bool ReadProtoFromBinaryFile(const uint8_t *data, size_t len, google::protobuf::Message *proto) {
  GE_CHK_BOOL_EXEC(data != nullptr, return false, "data is null.");
  GE_CHK_BOOL_EXEC(proto != nullptr, return false, "proto is null.");

  google::protobuf::io::CodedInputStream coded_stream(data, len);
  // 2048M -1
  coded_stream.SetTotalBytesLimit(INT32_MAX, -1);
  if (!proto->ParseFromCodedStream(&coded_stream)) {
    GELOGE(GRAPH_FAILED, "ReadProtoFromBinaryFile failed len %zu", len);
    return false;
  }
  return true;
}

Buffer ModelSerialize::SerializeModel(const Model &model, bool is_dump) {
  proto::ModelDef model_def;
  ModelSerializeImp imp;
  if (!imp.SerializeModel(model, &model_def, is_dump)) {
    return Buffer();
  }
#if !defined(__ANDROID__) && !defined(ANDROID)
  Buffer buffer(model_def.ByteSizeLong());
#else
  Buffer buffer(model_def.ByteSize());
#endif
  GE_CHK_BOOL_ONLY_LOG(buffer.GetSize() != 0, "get size failed");
  GE_CHK_BOOL_ONLY_LOG((buffer.GetData() != nullptr), "get size failed");
  auto ret = model_def.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));
  if (ret != true) {
    GELOGW("serialize to array fail.");
  }
  return buffer;
}

size_t ModelSerialize::GetSerializeModelSize(const Model &model) {
  proto::ModelDef model_def;
  ModelSerializeImp imp;
  if (!imp.SerializeModel(model, &model_def)) {
    return 0;
  }
#if !defined(__ANDROID__) && !defined(ANDROID)
  return model_def.ByteSizeLong();
#else
  return model_def.ByteSize();
#endif
}

Model ModelSerialize::UnserializeModel(const uint8_t *data, size_t len) {
  if (data == nullptr) {
    GELOGE(GRAPH_FAILED, "data is nullptr");
    return Model();
  }

  std::shared_ptr<proto::ModelDef> model_proto_ptr;
  model_proto_ptr = ComGraphMakeShared<proto::ModelDef>();
  if (model_proto_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "proto::ModelDef make shared failed");
    return Model();
  }

  auto &model_proto = *model_proto_ptr;
  if (!ReadProtoFromBinaryFile(data, len, &model_proto)) {
    GELOGE(GRAPH_FAILED, "ParseFromArray fail");
    return Model();
  }

  Model model;
  ModelSerializeImp imp;
  imp.SetProtobufOwner(model_proto_ptr);
  if (!imp.UnserializeModel(model, model_proto)) {
    GELOGE(GRAPH_FAILED, "Unserialize Model fail");
    return Model();
  }
  return model;
}

Model ModelSerialize::UnserializeModel(ge::proto::ModelDef &model_def) {
  std::shared_ptr<proto::ModelDef> model_def_ptr = ComGraphMakeShared<proto::ModelDef>(model_def);
  GE_CHK_BOOL_EXEC(model_def_ptr != nullptr, return Model(), "mode_def make shared failed");

  ModelSerializeImp imp;
  imp.SetProtobufOwner(model_def_ptr);
  Model model;
  if (!imp.UnserializeModel(model, *model_def_ptr)) {
    GELOGE(GRAPH_FAILED, "Unserialize Model fail");
    return Model();
  }
  return model;
}

Buffer ModelSerialize::SerializeGraph(const ComputeGraphPtr &graph) {
  proto::GraphDef graph_def;
  ModelSerializeImp imp;
  if (!imp.SerializeGraph(graph, &graph_def)) {
    return Buffer();
  }
#if !defined(__ANDROID__) && !defined(ANDROID)
  Buffer buffer(graph_def.ByteSizeLong());
#else
  Buffer buffer(graph_def.ByteSize());
#endif
  GE_CHK_BOOL_ONLY_LOG((buffer.GetSize() != 0), "get size failed");
  GE_CHK_BOOL_ONLY_LOG((buffer.GetData() != nullptr), "get size failed");
  auto ret = graph_def.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));
  if (ret != true) {
    GE_LOGE("serialize to array fail.");
  }

  return buffer;
}

ComputeGraphPtr ModelSerialize::UnserializeGraph(const uint8_t *data, size_t len) {
  if (data == nullptr) {
    GELOGE(GRAPH_FAILED, "data is nullptr");
    return nullptr;
  }

  std::shared_ptr<proto::GraphDef> graph_proto_ptr;
  graph_proto_ptr = ComGraphMakeShared<proto::GraphDef>();
  if (graph_proto_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "proto::GraphDef make shared failed");
    return nullptr;
  }
  proto::GraphDef &graph_proto = *graph_proto_ptr;
  if (!ReadProtoFromBinaryFile(data, len, &graph_proto)) {
    GELOGE(GRAPH_FAILED, "ParseFromArray fail");
    return nullptr;
  }

  ComputeGraphPtr graph;
  ModelSerializeImp imp;
  imp.SetProtobufOwner(graph_proto_ptr);
  if (!imp.UnserializeGraph(graph, graph_proto)) {
    return nullptr;
  }
  return graph;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Buffer ModelSerialize::SerializeOpDesc(const ConstOpDescPtr &op_desc) {
  proto::OpDef op_def;
  ModelSerializeImp imp;
  if (!imp.SerializeOpDesc(op_desc, &op_def)) {
    return Buffer();
  }
#if !defined(__ANDROID__) && !defined(ANDROID)
  Buffer buffer(op_def.ByteSizeLong());
#else
  Buffer buffer(op_def.ByteSize());
#endif
  GE_CHK_BOOL_ONLY_LOG((buffer.GetSize() != 0), "get size failed");
  GE_CHK_BOOL_ONLY_LOG((buffer.GetData() != nullptr), "get size failed");
  auto ret = op_def.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));
  if (ret != true) {
    GE_LOGE("serialize to array fail.");
  }

  return buffer;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr ModelSerialize::UnserializeOpDesc(const uint8_t *data,
                                                                                           size_t len) {
  if (data == nullptr) {
    GELOGE(GRAPH_FAILED, "data is nullptr");
    return nullptr;
  }

  std::shared_ptr<proto::OpDef> op_def_ptr;
  op_def_ptr = ComGraphMakeShared<proto::OpDef>();
  if (op_def_ptr == nullptr) {
    GELOGE(GRAPH_FAILED, "proto::OpDef make shared failed");
    return nullptr;
  }
  proto::OpDef &op_def = *op_def_ptr;
  if (!ReadProtoFromBinaryFile(data, len, &op_def)) {
    GELOGE(GRAPH_FAILED, "ParseFromArray fail");
    return nullptr;
  }

  OpDescPtr op_desc;
  ModelSerializeImp imp;
  imp.SetProtobufOwner(op_def_ptr);
  if (!imp.UnserializeOpDesc(op_desc, op_def)) {
    GELOGW("UnserializeOpDesc error.");
  }
  return op_desc;
}
}  // namespace ge
