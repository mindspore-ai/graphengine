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

#ifndef GE_GRAPH_PASSES_VARIABLE_OP_PASS_H_
#define GE_GRAPH_PASSES_VARIABLE_OP_PASS_H_
#include <map>
#include <set>
#include <stack>
#include "graph/common/transop_util.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/graph.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/util/variable_accelerate_ctrl.h"
#include "inc/graph_pass.h"

namespace ge {
namespace variable_op {
struct NodeDesc {
  ge::GeTensorDesc input;
  ge::GeTensorDesc output;
  bool is_update = false;
};
}  // namespace variable_op
class VariableOpPass : public GraphPass {
 public:
  explicit VariableOpPass(VarAccelerateCtrl *ctrl) : var_accelerate_ctrl_(ctrl) {}

  ~VariableOpPass() override = default;

  Status Run(ge::ComputeGraphPtr graph) override;

 private:
  Status UpdateTransRoad(VarTransRoad &fusion_road, vector<string> &trans_road_order,
                         map<string,pair<string, bool>> &trans_type_to_changed_desc,
                         map<string,vector<NodePtr>> &trans_type_to_trans_ops);

  Status DealFusion(const ge::NodePtr &var_node, VarTransRoad &fusion_road,
                    map<string, pair<string, bool>> trans_type_to_changed_desc,
                    map<string, vector<NodePtr>> trans_type_to_trans_ops,
                    vector<pair<NodePtr, NodePtr>> &delete_trans_nodes);

  Status RenewTransOpDesc(ge::NodePtr &node, bool is_reverse);

  Status RenewTransRoadDesc(const NodePtr &var, VarTransRoad &fusion_road);

  Status CheckIfCouldBeOptimized(const NodePtr &var, vector<string> &trans_road_order,
                                 map<string, pair<string, bool>> &trans_type_to_changed_desc,
                                 map<string, vector<NodePtr>> &trans_type_to_trans_ops, bool &flag);

  Status FusionIfNeed(const NodePtr &var, VarTransRoad &fusion_road);

  Status GetSameTransOP(const NodePtr &var, vector<string> &trans_road_order,
                        map<string, pair<string, bool>> &trans_type_to_changed_desc,
                        map<string, vector<NodePtr>> &trans_type_to_trans_ops, bool &is_match);

  Status GetFisrtPathTransInfo(const NodePtr &var, vector<string> &trans_road_order,
                               map<string, pair<string, bool>> &trans_type_to_changed_desc,
                               map<string, vector<NodePtr>> &trans_type_to_trans_ops);

  void VariableDFS(const NodePtr &node, map<string, pair<string, bool>> &trans_type_to_changed_desc,
                   map<string, vector<NodePtr>> &trans_type_to_trans_ops, bool &is_match);

  Status UpdateTransInfo(vector<NodePtr> &cur_path,  bool& is_match,
                         map<string, pair<string, bool>> &trans_type_to_changed_desc,
                         map<string, vector<NodePtr>> &trans_type_to_trans_ops);

  Status GetAndCheckTransOpOfVarRef(const ge::NodePtr &var_node, bool &pass_check,
                                    map<string, pair<string, bool>> &trans_type_to_changed_desc,
                                    vector<pair<NodePtr, NodePtr>> &delete_var_ref_trans_nodes);

  Status CheckTransOpOfVarAndVarRefSymmetry(NodePtr &var_ref_trans_op, const string &desc_diff, bool &is_symmetry);

  Status UpdateVarAndRefOutputFormatInfo(const GeTensorDesc &final_output, const ge::NodePtr &node);

  Status GenerateVariableVariableRefMap(const ComputeGraphPtr &compute_graph);

  void CopyVariableFormatDataTypeAndShape(const GeTensorDesc &src_tensor_desc, GeTensorDesc &dst_tensor_desc);

  Status UpdateIOFormatInfo(const GeTensorDesc &final_output, std::set<NodePtr> &nodes);

  Status RenewVarDesc(ge::ComputeGraphPtr &graph);

  Status RenewVarDesc(uint64_t session_id, const NodePtr &node, const VarTransRoad &fusion_road);

  map<NodePtr, std::set<NodePtr>> var_and_var_ref_map_;

  VarAccelerateCtrl *var_accelerate_ctrl_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_VARIABLE_OP_PASS_H_
