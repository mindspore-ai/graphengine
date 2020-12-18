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

#include "register/graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "graph/debug/ge_log.h"
#include "graph/utils/graph_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"
#include "register/graph_optimizer/fusion_common/graph_pass_util.h"
#include "register/graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.h"

namespace fe {
static const string STREAM_LABEL = "_stream_label";
PatternFusionBasePass::PatternFusionBasePass() {
  pattern_fusion_base_pass_impl_ptr_ = std::make_shared<PatternFusionBasePassImpl>();
}

PatternFusionBasePass::~PatternFusionBasePass() {}

Status PatternFusionBasePass::Run(ge::ComputeGraph &graph, OpsKernelInfoStorePtr ops_kernel_info_store_ptr) {
  // save the opskernelstoreptr which will be uesd while checking op support
  pattern_fusion_base_pass_impl_ptr_->SetOpsKernelInfoStore(ops_kernel_info_store_ptr);

  return Run(graph);
}
/**
 * @ingroup fe
 * @brief execute pass
 */
Status PatternFusionBasePass::Run(ge::ComputeGraph &graph) {
  Mappings mappings;
  bool is_patterns_ok = true;
  // build Pattern
  vector<FusionPattern *> patterns;
  pattern_fusion_base_pass_impl_ptr_->GetPatterns(patterns);
  if (patterns.empty()) {
    patterns = DefinePatterns();
    for (FusionPattern *pattern : patterns) {
      if (pattern != nullptr) {
        bool ok = pattern->Build();
        if (!ok) {
          GELOGW("this pattern: %s build not success.", pattern->GetName().c_str());
        }
        pattern->Dump();
        is_patterns_ok = is_patterns_ok && ok;
      }
    }

    pattern_fusion_base_pass_impl_ptr_->SetPatterns(patterns);
  }

  if (!is_patterns_ok) {
    return FAILED;
  }
  NodeMapInfoPtr node_map_info = nullptr;
  if (GraphPassUtil::GetOpTypeMapToGraph(node_map_info, graph) == SUCCESS) {
    node_map_info->run_count++;
  }
  // do matching and fusion for each pattern
  bool final_changed = false;
  for (const FusionPattern *pattern : patterns) {
    if (pattern != nullptr) {
      bool changed = false;
      Status ret = RunOnePattern(graph, *pattern, changed);
      if (ret != SUCCESS) {
        GELOGW("run pattern %s not success, graph is not changed by it.", pattern->GetName().c_str());
        return ret;
      }

      final_changed = final_changed || changed;
    }
  }
  return final_changed ? SUCCESS : NOT_CHANGED;
}

static bool CheckStreamLabel(vector<ge::NodePtr> &fused_nodes) {
  string stream_label = "";
  for (auto &n : fused_nodes) {
    string stream_label_tmp = "";
    if (!ge::AttrUtils::GetStr(n->GetOpDesc(), STREAM_LABEL, stream_label_tmp)) {
      stream_label_tmp = "null";
    }
    if (stream_label == "") {
      stream_label = stream_label_tmp;
    } else if (stream_label != "" && stream_label != stream_label_tmp) {
      return false;
    }
  }
  return true;
}

static bool SetStreamLabelToFusedNodes(vector<ge::NodePtr> &fused_nodes, ge::NodePtr first_node) {
  string stream_label = "";
  if (ge::AttrUtils::GetStr(first_node->GetOpDesc(), STREAM_LABEL, stream_label)) {
    for (ge::NodePtr &node : fused_nodes) {
      if (!ge::AttrUtils::SetStr(node->GetOpDesc(), STREAM_LABEL, stream_label)) {
        GELOGW("newNode set _stream_label error, fusion failed.");
        return false;
      }
    }
  }
  return true;
}
/**
 * @ingroup fe
 * @brief do matching and fusion in graph based on the pattern
 */
Status PatternFusionBasePass::RunOnePattern(ge::ComputeGraph &graph, const FusionPattern &pattern, bool &changed) {
  changed = false;
  Mappings mappings;
  int32_t effect_times = 0;
  uint32_t graph_id = graph.GetGraphID();
  FusionInfo fusion_info(graph.GetSessionID(), to_string(graph_id), GetName(), static_cast<int32_t>(mappings.size()),
                         effect_times);
  origin_op_anchors_map_.clear();
  // match all patterns in graph, and save them to mappings
  if (!MatchAll(graph, pattern, mappings)) {
    GELOGD("GraphFusionPass[%s]: pattern=%s, matched_times=%zu, effected_times=%d.", GetName().c_str(),
           pattern.GetName().c_str(), mappings.size(), effect_times);
    return SUCCESS;
  }

  GELOGD("This graph has been matched with pattern[%s]. The mappings are as follows.", pattern.GetName().c_str());

  // print the results of matching
  pattern_fusion_base_pass_impl_ptr_->DumpMappings(pattern, mappings);
  NodeMapInfoPtr node_map_info = nullptr;
  // get nodes by type from node
  (void)GraphPassUtil::GetOpTypeMapToGraph(node_map_info, graph);
  // do fusion for each mapping
  for (Mapping &mapping : mappings) {
    vector<ge::NodePtr> fus_nodes;
    ge::NodePtr first_node = nullptr;
    for (auto &item : mapping) {
      if (!item.second.empty()) {
        first_node = item.second[0];
        break;
      }
    }

    Status status = Fusion(graph, mapping, fus_nodes);
    if (!SetStreamLabelToFusedNodes(fus_nodes, first_node)) {
      return FAILED;
    }

    if (status != SUCCESS && status != NOT_CHANGED) {
      GELOGE(status, "Fail to fuse the graph with pattern[%s].", pattern.GetName().c_str());
      return status;
    }

    if (status == SUCCESS) {
      effect_times++;
      std::vector<ge::NodePtr> original_nodes;
      for (auto &item : mapping) {
        if (!item.second.empty()) {
          for (auto &node : item.second) {
            original_nodes.push_back(node);
          }
        }
      }
      SetDataDumpAttr(original_nodes, fus_nodes);
      if (!fus_nodes.empty()) {
        // add fusednode to node map info
        for (ge::NodePtr &node : fus_nodes) {
          (void)GraphPassUtil::AddNodeFromOpTypeMap(node_map_info, node);
        }
      }
    }
    changed = (changed || status == SUCCESS);
  }

  // get match times and effect times
  FusionStatisticRecorder &fusion_statistic_inst = FusionStatisticRecorder::Instance();
  fusion_info.SetMatchTimes(static_cast<int32_t>(mappings.size()));
  fusion_info.SetEffectTimes(effect_times);
  fusion_statistic_inst.UpdateGraphFusionMatchTimes(fusion_info);
  fusion_statistic_inst.UpdateGraphFusionEffectTimes(fusion_info);
  GELOGD("GraphId[%d], GraphFusionPass[%s]: pattern=%s, matched_times=%zu, effected_times=%d.", graph_id,
         GetName().c_str(), pattern.GetName().c_str(), mappings.size(), effect_times);
  return SUCCESS;
}

Status PatternFusionBasePass::SetDataDumpAttr(vector<ge::NodePtr> &original_nodes, vector<ge::NodePtr> &fus_nodes) {
  for (auto &oriNode : original_nodes) {
    auto itr = origin_op_anchors_map_.find(oriNode);
    if (itr != origin_op_anchors_map_.end()) {
      for (const auto &anchor_iter : itr->second) {
        auto next_node_in_achor = anchor_iter.first;
        auto fusion_node_out_data_anchor = next_node_in_achor->GetPeerOutAnchor();
        if (fusion_node_out_data_anchor == nullptr) {
          GELOGW("fusionNodeOutDataAnchor is null");
          return FAILED;
        }

        auto fusion_node = fusion_node_out_data_anchor->GetOwnerNode();
        if (fusion_node == nullptr) {
          GELOGW("fusionNode is null");
          return FAILED;
        }

        if (pattern_fusion_base_pass_impl_ptr_->IsNodesExist(fusion_node, fus_nodes)) {
          auto origin_node_out_anchor = anchor_iter.second;
          if (origin_node_out_anchor == nullptr) {
            GELOGW("originNodeOutAnchor is null");
            return FAILED;
          }

          auto origin_node = origin_node_out_anchor->GetOwnerNode();
          if (origin_node == nullptr) {
            GELOGW("originNode is null");
            return FAILED;
          }

          uint32_t origin_index = origin_node_out_anchor->GetIdx();
          uint32_t fusion_index = fusion_node_out_data_anchor->GetIdx();
          (void)GraphPassUtil::SetOutputDescAttr(origin_index, fusion_index, origin_node, fusion_node);
        }
      }
    }
  }

  for (auto &node : fus_nodes) {
    GraphPassUtil::RecordOriginalNames(original_nodes, node);
  }
  if (fus_nodes.size() > 1) {
    bool is_multi_op = true;
    for (ge::NodePtr &node : fus_nodes) {
      ge::AttrUtils::SetBool(node->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_IS_MULTIOP, is_multi_op);
    }
  }

  return SUCCESS;
}

bool PatternFusionBasePass::CheckOpSupported(const ge::OpDescPtr &op_desc_ptr) {
  return pattern_fusion_base_pass_impl_ptr_->CheckOpSupported(op_desc_ptr);
}

/**
 * @ingroup fe
 * @brief match all nodes in graph according to pattern
 */
// match nodes in graph according to pattern, the algorithm is shown as
// following:
// 1. get output node from pattern
// 2. Search for candidate nodes in Graph (network Graph generated after
//    parsing) according to Op Type and
// (optional), and add the candidate node to the list of candidates
// 3. For each Node in the candidate list, check whether the type and the number
//    of precursors are consistent with the description of corresponding Op
//    in pattern. If they are consistent, add the precursor Node to the
//    candidate list, and add "PatternOp-GraphNode" to the mapping; otherwise,
//    return an empty mapping
// 4. repeat step 3 until all the Ops in pattern are matched
// 5. if all the Ops in pattern are matched successfully, return the mapping of
//    PatternOp and GraphNode
bool PatternFusionBasePass::MatchAll(ge::ComputeGraph &graph, const FusionPattern &pattern, Mappings &mappings) {
  vector<ge::NodePtr> matched_output_nodes;

  // find all the output nodes of pattern in the graph based on Op type
  std::shared_ptr<FusionPattern::OpDesc> output_op_desc = pattern.GetOutput();
  if (output_op_desc == nullptr) {
    return false;
  }

  if (!pattern_fusion_base_pass_impl_ptr_->GetMatchOutputNodes(graph, pattern, matched_output_nodes)) {
    return false;
  }

  // begin matching from every output node
  for (ge::NodePtr &output_node : matched_output_nodes) {
    Mapping mapping;
    if (pattern_fusion_base_pass_impl_ptr_->MatchFromOutput(output_node, output_op_desc, mapping)) {
      // node attr _stream_label must be equal
      auto fusion_nodes = GetNodesFromMapping(mapping);
      if (!CheckStreamLabel(fusion_nodes)) {
        return false;
      }
      mappings.push_back(mapping);

      // Record output nodes anchor vs succeed node anchor map
      RecordOutputAnchorMap(output_node);
    }
  }
  // if matching is successful, return true; otherwise false
  return !mappings.empty();
}

/*
 * @brief: get all fusion nodes matched
 * @param [in] mapping: fusion node group
 * @return std::vector<ge::NodePtr>: all fusion nodes list
 */
vector<ge::NodePtr> PatternFusionBasePass::GetNodesFromMapping(const Mapping &mapping) {
  std::vector<ge::NodePtr> nodes;
  for (auto &item : mapping) {
    for (const auto &node : item.second) {
      nodes.push_back(node);
    }
  }
  return nodes;
}

/**
 * @ingroup fe
 * @brief get an op from mapping according to ID
 */
ge::NodePtr PatternFusionBasePass::GetNodeFromMapping(const string &id, const Mapping &mapping) {
  for (auto &item : mapping) {
    std::shared_ptr<OpDesc> op_desc = item.first;
    if (op_desc != nullptr && op_desc->id == id) {
      if (item.second.empty()) {
        return nullptr;
      } else {
        return item.second[0];
      }
    }
  }
  return nullptr;
}

void PatternFusionBasePass::RecordOutputAnchorMap(ge::NodePtr output_node) {
  for (const auto &output_anchor : output_node->GetAllOutDataAnchors()) {
    if (output_anchor == nullptr) {
      continue;
    }

    for (const auto &peer_in_anchor : output_anchor->GetPeerInDataAnchors()) {
      if (peer_in_anchor == nullptr) {
        continue;
      }

      // Record anchor map
      auto itr = origin_op_anchors_map_.find(output_node);
      if (itr == origin_op_anchors_map_.end()) {
        std::map<ge::InDataAnchorPtr, ge::OutDataAnchorPtr> anchorMap;
        anchorMap[peer_in_anchor] = output_anchor;
        origin_op_anchors_map_.emplace(make_pair(output_node, anchorMap));
      } else {
        itr->second.emplace(make_pair(peer_in_anchor, output_anchor));
      }
    }
  }
}

void PatternFusionBasePass::ClearOutputAnchorMap() { origin_op_anchors_map_.clear(); }
}  // namespace fe
