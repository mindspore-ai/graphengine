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

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_PATTERN_FUSION_BASE_PASS_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_PATTERN_FUSION_BASE_PASS_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "common/opskernel/ops_kernel_info_store.h"
#include "register/graph_optimizer/graph_fusion/fusion_pattern.h"
#include "register/graph_optimizer/graph_fusion/graph_pass.h"

using std::initializer_list;
using std::map;
using std::string;
using std::vector;

using namespace std;

namespace fe {
using OpsKernelInfoStorePtr = std::shared_ptr<ge::OpsKernelInfoStore>;
class PatternFusionBasePassImpl;
using PatternFusionBasePassImplPtr = std::shared_ptr<PatternFusionBasePassImpl>;

/** Pass based on pattern
 * @ingroup FUSION_PASS_GROUP
 * @note New virtual methods should be append at the end of this class
 */
class PatternFusionBasePass : public GraphPass {
 public:
  using OpDesc = FusionPattern::OpDesc;
  using Mapping = map<const std::shared_ptr<OpDesc>, vector<ge::NodePtr>>;
  using Mappings = vector<Mapping>;

  PatternFusionBasePass();
  virtual ~PatternFusionBasePass();

  /** execute pass
   *
   * @param [in] graph, the graph waiting for pass level optimization
   * @return SUCCESS, successfully optimized the graph by the pass
   * @return NOT_CHANGED, the graph did not change
   * @return FAILED, fail to modify graph
   */
  Status Run(ge::ComputeGraph &graph) override;

  /** execute pass
   *
   * @param [in] graph, the graph waiting for pass level optimization
   * @param [ops_kernel_info_store_ptr, OP info kernel instance
   * @return SUCCESS, successfully optimized the graph by the pass
   * @return NOT_CHANGED, the graph did not change
   * @return FAILED, fail to modify graph
   */
  virtual Status Run(ge::ComputeGraph &graph, OpsKernelInfoStorePtr ops_kernel_info_store_ptr);

 protected:
  virtual vector<FusionPattern *> DefinePatterns() = 0;
  virtual Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &new_nodes) = 0;

  std::vector<ge::NodePtr> GetNodesFromMapping(const Mapping &mapping);
  ge::NodePtr GetNodeFromMapping(const string &id, const Mapping &mapping);

  void RecordOutputAnchorMap(ge::NodePtr output_node);
  void ClearOutputAnchorMap();

  Status SetDataDumpAttr(vector<ge::NodePtr> &original_nodes, vector<ge::NodePtr> &fus_nodes);

  bool CheckOpSupported(const ge::OpDescPtr &op_desc_ptr);

 private:
  /** match all nodes in graph according to pattern
   *
   * @param pattern fusion pattern defined
   * @param mappings match result
   * @return SUCCESS, successfully add edge
   * @return FAILED, fail
   */
  bool MatchAll(ge::ComputeGraph &graph, const FusionPattern &pattern, Mappings &mappings);

  Status RunOnePattern(ge::ComputeGraph &graph, const FusionPattern &pattern, bool &changed);  // lint !e148

  /** Internal implement class ptr */
  std::shared_ptr<PatternFusionBasePassImpl> pattern_fusion_base_pass_impl_ptr_;

  std::unordered_map<ge::NodePtr, std::map<ge::InDataAnchorPtr, ge::OutDataAnchorPtr>> origin_op_anchors_map_;
};
}  // namespace fe

#endif  // INC_REGISTER_GRAPH_OPTIMIZER_PATTERN_FUSION_BASE_PASS_H_
