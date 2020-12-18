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

#ifndef INC_COMMON_UTILS_AI_CORE_COMMON_GRAPH_COMMON_H_
#define INC_COMMON_UTILS_AI_CORE_COMMON_GRAPH_COMMON_H_

#include "graph/compute_graph.h"
#include "common/aicore_util_types.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace fe {

using k_scope_node_map_t = std::map<int64_t, std::vector<ge::NodePtr>>;
using k_scope_node_pair_t = std::pair<int64_t, std::vector<ge::NodePtr>>;

class GraphCommImpl;
using GraphCommImplPtr = std::unique_ptr<GraphCommImpl>;

class GraphComm {
public:
  GraphComm(const string &engine_name);
  virtual ~GraphComm();
  GraphComm(const GraphComm &in) = delete;
  GraphComm &operator=(const GraphComm &in) = delete;

  Status GetscopeNodeMap(ge::ComputeGraph &graph, k_scope_node_map_t &fusion_map);

  Status CopyFusionOpNodes(vector<FusionDataFlow> &fus_input_edge_list,
                           vector<FusionDataFlow> &fus_output_edge_list,
                           vector<ge::NodePtr> &fus_nodelist,
                           ge::OpDescPtr fusion_op_desc,
                           ge::ComputeGraphPtr fusion_graph);

  Status CopyFusionOpEdges(ge::OpDescPtr fusion_op_desc,
                           ge::ComputeGraph &orig_graph,
                           ge::ComputeGraphPtr fusion_graph);

  Status GetNodeDataFlowMap(
      const ge::NodePtr &fus_node,
      std::map<ge::NodePtr, std::map<ge::AnchorPtr, ge::AnchorPtr>>
          &fusion_op_anchors_map,
      ge::kFusionDataFlowVec_t &fus_dataflow_list, const int &map_type);

  Status GetFusionNodeEdgeList(std::vector<ge::NodePtr> &fus_nodelist,
                               std::vector<FusionDataFlow> &fus_input_edge_list,
                               std::vector<FusionDataFlow> &fus_output_edge_list);
  void ClearFusionSrc();

  void ClearFusionDst();

  void
  AddFusionOutputSrc(const uint32_t &src_op_id, const ge::AnchorPtr &src_anchor,
                     const int32_t &fusion_src_index,
                     std::pair<string, ge::AnchorPtr> &node_dataindex_pair);

  void AddFusionInputSrc(const uint32_t &src_op_id,
                         const ge::AnchorPtr &src_anchor,
                         const int32_t &fusion_dst_index,
                         std::pair<string, ge::AnchorPtr> &node_dataindex_pair);

  void SaveFusionDst(const uint32_t &dst_op_id, ge::AnchorPtr dst_anchor);

  bool IsFusionDstExist(const uint32_t &dst_op_id,
                        const ge::AnchorPtr &dst_anchor);

  bool GetFusionSrc(const uint32_t &src_op_id, const ge::AnchorPtr &src_anchor,
                    int32_t &fusion_src_index, int32_t &fusion_dst_index);

  Status
  GetFusionNodeCtrlEdgeList(vector<ge::NodePtr> &fus_nodelist,
                            vector<FusionDataFlow> &fus_input_ctrl_edge_list,
                            vector<FusionDataFlow> &fus_output_ctrl_edge_list);

  Status MergeFusionNodeEdgeList(ge::NodePtr &fus_node,
                                 vector<ge::NodePtr> &fus_nodelist,
                                 vector<FusionDataFlow> &fus_input_edge_list,
                                 vector<FusionDataFlow> &fus_output_edge_list);

  Status MergeFusionNodeCtrlEdgeList(ge::NodePtr &fus_node,
                                     vector<ge::NodePtr> &fus_nodelist,
                                     vector<FusionDataFlow> &fus_input_edge_list,
                                     vector<FusionDataFlow> &fus_output_edge_list);

  string GetEngineName();

private:
  Status
  MergeFusionNodeInputEdgeList(ge::NodePtr fus_node,
                               std::vector<ge::NodePtr> &fus_nodelist,
                               std::vector<FusionDataFlow> &fus_input_edge_list);
  Status
  MergeFusionNodeOutputEdgeList(ge::NodePtr fus_node,
                                std::vector<ge::NodePtr> &fus_nodelist,
                                std::vector<FusionDataFlow> &fus_output_edge_list);

  string engine_name_;

  std::vector<FusionOpSrc> exist_fusion_src_list_;
  std::vector<FusionOpDst> exist_fusion_dst_list_;

  // std::vector<std::multimap<std::string, uint32_t>>
  ge::kFusionDataFlowVec_t fusion_input_dataflow_list_;

  // std::vector<std::multimap<std::string, ge::AnchorPtr>>
  ge::kFusionDataFlowVec_t fusion_output_dataflow_list_;

  GraphCommImplPtr graph_comm_impl_ptr_;
};
} // namespace fe
#endif
