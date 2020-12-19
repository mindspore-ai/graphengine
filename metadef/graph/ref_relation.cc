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

#include "graph/ref_relation.h"

#include <unordered_set>
#include <unordered_map>

#include "utils/mem_utils.h"
#include "debug/ge_log.h"
#include "debug/ge_op_types.h"
#include "debug/ge_util.h"
#include "debug/ge_attr_define.h"
#include "graph/ge_error_codes.h"
#include "graph/utils/graph_utils.h"
#include "framework/common/debug/ge_log.h"

using namespace std;
using namespace ge;
namespace ge {
namespace {
  const char *kRefIndex = "_parent_node_index";
  const string kWhile = "While";
  const string kIf = "If";
  const string kCase = "Case";

  const uint16_t kMaxElementNum = 100;

  std::unordered_set<string> function_op = {
    kWhile,
    kIf,
    kCase
  };
}

/* Impl */
class RefRelations::Impl {
public:
  graphStatus LookUpRefRelations(const RefCell &key, unordered_set<RefCell, RefCellHash> &result) {
    unsigned long number = static_cast<unsigned long>(reinterpret_cast<uintptr_t>(key.node.get()));
    std::string lookup_key = key.node_name + std::to_string(key.in_out) + std::to_string(key.in_out_idx)
                           + std::to_string(number);
    auto iter = look_up_table_.find(lookup_key);
    if (iter != look_up_table_.end()) {
      for (auto &c : iter->second) {
        result.insert(c);
      }
      return GRAPH_SUCCESS;
    }
    GELOGW("can not find any relations! key value of dest relation is %s", lookup_key.c_str());
    return GRAPH_SUCCESS;
  };
  graphStatus BuildRefRelations(ge::ComputeGraph &root_graph);
  graphStatus Clear() {
    GELOGD("Start clear boundary reflections between main graph and sub graph!");
    look_up_table_.clear();
    values_.clear();
    return GRAPH_SUCCESS;
  };
private:
  graphStatus BuildLookUpTables();
  graphStatus BuildRefRelationsForBranch(
                  const NodePtr &root_node,
                  const vector<vector<NodePtr>> &classed_data_nodes,
                  const vector<vector<std::pair<NodePtr, size_t>>> &classed_netoutput_nodes,
                  vector<vector<RefCell>> &node_refs);
  graphStatus BuildRefRelationsForWhile(
                  const NodePtr &root_node,
                  const vector<vector<NodePtr>> &classed_data_nodes,
                  const vector<vector<std::pair<NodePtr, size_t>>> &classed_netoutput_nodes,
                  vector<vector<RefCell>> &node_refs);
  graphStatus BuildRelationsWithFuncNodeType(
                  const NodePtr &root_node,
                  const vector<vector<NodePtr>> &classed_data_nodes,
                  const vector<vector<std::pair<NodePtr, size_t>>> &classed_netoutput_nodes,
                  vector<vector<RefCell>> &node_refs);
  void GetDataAndNetoutputOfSubGraph(
                  const ge::ComputeGraph &root_graph,
                  vector<NodePtr> &data_nodes,
                  vector<NodePtr> &netoutput_nodes,
                  const std::vector<std::string> &sub_graph_names,
                  const std::string &node_type);

  graphStatus GetRootGraph(ge::ComputeGraph &graph, ge::ComputeGraph &root_graph);
  graphStatus ProcessSubgraphDataNodes(
                 vector<NodePtr> &data_nodes,
                 vector<vector<NodePtr>> &classed_data_nodes);
  graphStatus ProcessSubgraphNetoutput(
                  const vector<NodePtr> &netoutput_nodes,
                  vector<vector<std::pair<NodePtr, size_t>>> &classed_netoutput_nodes);

  std::unordered_map<string, vector<RefCell>> look_up_table_;
  std::vector<vector<vector<RefCell>>> values_;
};

// Node Level
graphStatus RefRelations::Impl::BuildRefRelationsForBranch(
                            const NodePtr &root_node,
                            const vector<vector<NodePtr>> &classed_data_nodes,
                            const vector<vector<std::pair<NodePtr, size_t>>> &classed_netoutput_nodes,
                            vector<vector<RefCell>> &node_refs) {
  GELOGD("Enter BuildRefRelationsForBranch!");

  size_t ref_i = 0;
  for (const auto &ref_i_data_nodes : classed_data_nodes) {
    vector<RefCell> in_ref_i_all_refs;
    RefCell cell_root;
    cell_root.node_name = root_node->GetName();
    cell_root.node = root_node;
    cell_root.in_out = NODE_IN;
    cell_root.in_out_idx = ref_i;
    in_ref_i_all_refs.emplace_back(cell_root);
    for (const auto &data : ref_i_data_nodes) {
      RefCell cell_in;
      RefCell cell_out;
      cell_in.node_name = data->GetName();
      cell_in.node = data;
      cell_in.in_out = NODE_IN;
      cell_in.in_out_idx = 0;
      cell_out.node_name = data->GetName();
      cell_out.node = data;
      cell_out.in_out = NODE_OUT;
      cell_out.in_out_idx = 0;
      in_ref_i_all_refs.emplace_back(cell_in);
      in_ref_i_all_refs.emplace_back(cell_out);
    }
    node_refs.emplace_back(in_ref_i_all_refs);
    ref_i++;
  }

  size_t ref_o = 0;
  for (const auto &ref_o_net_nodes : classed_netoutput_nodes) {
    vector<RefCell> out_ref_i_all_refs;
    RefCell cell_root;
    cell_root.node_name = root_node->GetName();
    cell_root.node = root_node;
    cell_root.in_out = NODE_OUT;
    cell_root.in_out_idx = ref_o;
    out_ref_i_all_refs.emplace_back(cell_root);
    for (const auto &ele : ref_o_net_nodes) {
      RefCell cell_netoutput_in;
      cell_netoutput_in.node_name = (ele.first)->GetName();
      cell_netoutput_in.node = ele.first;
      cell_netoutput_in.in_out = NODE_IN;
      cell_netoutput_in.in_out_idx = ele.second;
      out_ref_i_all_refs.emplace_back(cell_netoutput_in);
    }
    node_refs.emplace_back(out_ref_i_all_refs);
    ref_o++;
  }
  return GRAPH_SUCCESS;
}

graphStatus RefRelations::Impl::BuildLookUpTables() {
  GELOGD("start to build look up table!");
  for (size_t i = 0; i < values_.size(); i++) {
    vector<vector<RefCell>> &val = values_[i];
    for (const auto &ele : val) {
      for (const auto &ref_cell : ele) {
        string key = ref_cell.node_name + std::to_string(ref_cell.in_out) +
          std::to_string(ref_cell.in_out_idx) +
          std::to_string(static_cast<unsigned long>(reinterpret_cast<uintptr_t>(ref_cell.node.get())));
        look_up_table_[key] = ele;
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus RefRelations::Impl::BuildRefRelationsForWhile(
                const NodePtr &root_node,
                const vector<vector<NodePtr>> &classed_data_nodes,
                const vector<vector<std::pair<NodePtr, size_t>>> &classed_netoutput_nodes,
                vector<vector<RefCell>> &node_refs) {
  GELOGD("Enter BuildRefRelations for while op!");
  // data_nodes has been sorted
  // for while, input num must be same as output num
  auto input_num = root_node->GetAllInDataAnchorsSize();
  NodePtr netoutput = nullptr;

  size_t ref_i = 0;
  while (ref_i < input_num) {
    auto &ref_i_data_nodes = classed_data_nodes[ref_i];
    auto &ref_i_net_nodes = classed_netoutput_nodes[ref_i];

    vector<RefCell> ref_i_all_refs;
    RefCell cell_root_i;
    RefCell cell_root_o;
    cell_root_i.node_name = root_node->GetName();
    cell_root_i.node = root_node;
    cell_root_i.in_out = NODE_IN;
    cell_root_i.in_out_idx = ref_i;
    ref_i_all_refs.emplace_back(cell_root_i);
    cell_root_o.node_name = root_node->GetName();
    cell_root_o.node = root_node;
    cell_root_o.in_out = NODE_OUT;
    cell_root_o.in_out_idx = ref_i;
    ref_i_all_refs.emplace_back(cell_root_o);
    for (const auto &data : ref_i_data_nodes) {
      RefCell cell_in;
      RefCell cell_out;
      cell_in.node_name = data->GetName();
      cell_in.node = data;
      cell_in.in_out = NODE_IN;
      cell_in.in_out_idx = 0;
      cell_out.node_name = data->GetName();
      cell_out.node = data;
      cell_out.in_out = NODE_OUT;
      cell_out.in_out_idx = 0;
      ref_i_all_refs.emplace_back(cell_in);
      ref_i_all_refs.emplace_back(cell_out);
    }

    for (const auto &ele : ref_i_net_nodes) {
      RefCell cell_netoutput_in;
      RefCell cell_netoutput_out;
      cell_netoutput_in.node_name = (ele.first)->GetName();
      cell_netoutput_in.node = ele.first;
      cell_netoutput_in.in_out = NODE_IN;
      cell_netoutput_in.in_out_idx = ele.second;
      ref_i_all_refs.emplace_back(cell_netoutput_in);
      netoutput = ele.first;
    }
    node_refs.emplace_back(ref_i_all_refs);
    ref_i++;
  }
  /* There exist scene like the follows, it means data0 data1 netoutput 0'th
   * and 1'th tensor should be the same addr.
   * Data0  Data1
   *      \/
   *      /\
   *   netoutput
   */
  if (netoutput == nullptr) {
    return GRAPH_SUCCESS;
  }
  for (const auto &in_anchor : netoutput->GetAllInDataAnchors()) {
    auto peer_out_data_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_data_anchor == nullptr) {
      continue;
    }
    auto peer_out_data_node = peer_out_data_anchor->GetOwnerNode();
    if (peer_out_data_node == nullptr || peer_out_data_node->GetOpDesc() == nullptr) {
      GELOGW("Node[%s]\'s peer_out_data_node or peer_out_data_node desc is null", (netoutput->GetName()).c_str());
      continue;
    }
    if (peer_out_data_node->GetType() != DATA) {
      continue;
    }
    auto in_data_anchor_idx = in_anchor->GetIdx();
    auto net_in_desc =
      netoutput->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(in_data_anchor_idx));
    int ref_d = 0;
    int ref_n = 0;
    (void)AttrUtils::GetInt(peer_out_data_node->GetOpDesc(), kRefIndex, ref_d);
    (void)AttrUtils::GetInt(net_in_desc, kRefIndex, ref_n);

    node_refs[ref_d].insert(node_refs[ref_d].end(), node_refs[ref_n].begin(), node_refs[ref_n].end());
    node_refs[ref_n].insert(node_refs[ref_n].end(), node_refs[ref_d].begin(), node_refs[ref_d].end());
  }


  return GRAPH_SUCCESS;
}
// build ref relations according to diff func op type
graphStatus RefRelations::Impl::BuildRelationsWithFuncNodeType(
                const NodePtr &root_node,
                const vector<vector<NodePtr>> &classed_data_nodes,
                const vector<vector<std::pair<NodePtr, size_t>>> &classed_netoutput_nodes,
                vector<vector<RefCell>> &node_refs) {
  // data_nodes has been sorted
  auto node_type = root_node->GetType();

  auto status = GRAPH_SUCCESS;
  if (node_type != kWhile) {
    status = BuildRefRelationsForBranch(root_node, classed_data_nodes, classed_netoutput_nodes, node_refs);
  } else {
    status = BuildRefRelationsForWhile(root_node, classed_data_nodes, classed_netoutput_nodes, node_refs);
  }
  return status;
}

void RefRelations::Impl::GetDataAndNetoutputOfSubGraph(
                const ge::ComputeGraph &root_graph,
                vector<NodePtr> &data_nodes,
                vector<NodePtr> &netoutput_nodes,
                const std::vector<std::string> &sub_graph_names,
                const std::string &node_type) {
  int sub_graph_idx = 0;
  for (const auto &name : sub_graph_names) {
    auto sub_graph = root_graph.GetSubgraph(name);
    if (sub_graph == nullptr) {
      GELOGW("Can not find the sub graph %s for root graph %s.", name.c_str(), root_graph.GetName().c_str());
      continue;
    }
    for (const auto &sub_graph_node : sub_graph->GetDirectNode()) {
      auto sub_graph_node_type = sub_graph_node->GetType();

      if (sub_graph_node_type == DATA) {
        data_nodes.emplace_back(sub_graph_node);
      } else if (sub_graph_node_type == NETOUTPUT) {
        // if while, the first subgraph must be cond subgraph.
        // There is no meaning for refs ,so continue
        if (node_type == kWhile && sub_graph_idx == 0) {
          continue;
        }
        netoutput_nodes.emplace_back(sub_graph_node);
      }
      continue;
    }
    sub_graph_idx++;
  }
}

graphStatus RefRelations::Impl::GetRootGraph(ge::ComputeGraph &graph, ge::ComputeGraph &root_graph) {
  auto parent_graph_ptr = graph.GetParentGraph();
  if (parent_graph_ptr == nullptr) {
    root_graph = graph;
    return GRAPH_SUCCESS;
  }
  auto root_graph_ptr = GraphUtils::FindRootGraph(parent_graph_ptr);
  if (root_graph_ptr == nullptr) {
    GE_LOGE("Get null root graph");
    return GRAPH_PARAM_INVALID;
  }
  root_graph = *root_graph_ptr;
  return GRAPH_SUCCESS;
}

graphStatus RefRelations::Impl::ProcessSubgraphDataNodes(
                       vector<NodePtr> &data_nodes,
                       vector<vector<NodePtr>> &classed_data_nodes) {
  GELOGD("start to process subgraph data nodes!");
  int max_ref_idx = 0;
  for (const auto &e : data_nodes) {
    int i;
    bool is_exist = true;
    is_exist = AttrUtils::GetInt(e->GetOpDesc(), kRefIndex, i);
    if (!is_exist) {
      GELOGE(GRAPH_FAILED, "Invalid SubGraph NetOutput node[%s].no attr %s",
             e->GetName().c_str(), kRefIndex);
      return GRAPH_FAILED;
    }
    max_ref_idx = (i > max_ref_idx) ? i : max_ref_idx;
  }

  while (!data_nodes.empty()) {
    auto data = data_nodes.back();
    data_nodes.pop_back();
    int ref_idx = 0;
    (void)AttrUtils::GetInt(data->GetOpDesc(), kRefIndex, ref_idx);
    if (ref_idx >= static_cast<int>(classed_data_nodes.size())) {
      return GRAPH_FAILED;
    }
    classed_data_nodes[ref_idx].emplace_back(data);
  }
  return GRAPH_SUCCESS;
}

graphStatus RefRelations::Impl::ProcessSubgraphNetoutput(
                  const vector<NodePtr> &netoutput_nodes,
                  vector<vector<std::pair<NodePtr, size_t>>> &classed_netoutput_nodes) {
  GELOGD("[RefRelations]Start to process subgraph netoutput!");
  for (const auto &sub_netoutput_node : netoutput_nodes) {
    auto op_desc = sub_netoutput_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    for (const auto &in_data_anchor : sub_netoutput_node->GetAllInDataAnchors()) {
      auto in_desc = op_desc->MutableInputDesc(in_data_anchor->GetIdx());
      if (in_desc == nullptr) {
        GELOGE(GRAPH_FAILED, "Invalid NetOutput node [%s] idx [%lu], no tensor on it",
               sub_netoutput_node->GetName().c_str(), in_data_anchor->GetIdx());
        return GRAPH_FAILED;
      }
      int ref_o;
      if (AttrUtils::GetInt(in_desc, kRefIndex, ref_o)) {
        if (ref_o >= static_cast<int>(classed_netoutput_nodes.size())) {
          return GRAPH_FAILED;
        }
        classed_netoutput_nodes[ref_o].emplace_back(std::pair<NodePtr, size_t>(
          {sub_netoutput_node, static_cast<size_t>(in_data_anchor->GetIdx())}
        ));
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus RefRelations::Impl::BuildRefRelations(ge::ComputeGraph &graph) {
  GELOGD("Start to build ref relations!");
  /* First Step: Get root graph */
  ge::ComputeGraph &root_graph = graph;
  auto status = GetRootGraph(graph, root_graph);
  if (status != GRAPH_SUCCESS) {
    return status;
  }

  for (const auto &node : graph.GetAllNodes()) {
    auto node_type = node->GetType();
    std::vector<NodePtr> ref_nodes;
    auto op_desc = node->GetOpDesc();
    auto sub_graph_names = op_desc->GetSubgraphInstanceNames();
    if (sub_graph_names.empty()) {
      continue;
    }
    vector<NodePtr> data_nodes;
    vector<NodePtr> netoutput_nodes;
    // Get data and netoutput of sub_graph
    GetDataAndNetoutputOfSubGraph(root_graph, data_nodes, netoutput_nodes, sub_graph_names, node_type);
    size_t max_elem_num = (data_nodes.size() > kMaxElementNum) ? data_nodes.size() : kMaxElementNum;
    vector<vector<NodePtr>> classed_data_nodes(max_elem_num);   // according to ref_idx
    vector<vector<std::pair<NodePtr, size_t>>> classed_netoutput_nodes(max_elem_num);   // according to ref_idx
    status = ProcessSubgraphDataNodes(data_nodes, classed_data_nodes);
    if (status != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "classfy data nodes failed!");
      return status;
    }

    // for netoutput
    // check netoutput
    // here main graph output number must be the same as every sub_graph netoutput node
    // key: netoutput node_ptr ,<ref_idx, net_in_idx>
    status = ProcessSubgraphNetoutput(netoutput_nodes, classed_netoutput_nodes);
    if (status != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "process netoutput failed!");
      return status;
    }

    vector<vector<RefCell>> node_refs;
    status = BuildRelationsWithFuncNodeType(node, classed_data_nodes, classed_netoutput_nodes, node_refs);
    if (status != GRAPH_SUCCESS) {
      GELOGE(status, "BuildRelationsWithFuncNodeType Failed! Node is [%s]!", node->GetName().c_str());
      return status;
    }
    if (!node_refs.empty()) {
      values_.push_back(node_refs);
    }
  }
  /* Seconde Step: generate map */
  status = BuildLookUpTables();
  if (status != GRAPH_SUCCESS) {
    GELOGE(status, "Build look up tables failed!");
    return status;
  }
  return GRAPH_SUCCESS;
}

/* Ref Relations Interface */
RefRelations::RefRelations() {
  impl_ = MakeShared<Impl>();
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "MakeShared failed!");
    return;
  }
}

graphStatus RefRelations::LookUpRefRelations(const RefCell &key, unordered_set<RefCell, RefCellHash> &result) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->LookUpRefRelations(key, result);
}

graphStatus RefRelations::BuildRefRelations(ge::ComputeGraph &root_graph) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->BuildRefRelations(root_graph);
}

graphStatus RefRelations::Clear() {
  GE_CHECK_NOTNULL(impl_);
  return impl_->Clear();
}
}