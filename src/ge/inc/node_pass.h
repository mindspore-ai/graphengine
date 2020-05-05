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

#ifndef GE_INC_NODE_PASS_H_
#define GE_INC_NODE_PASS_H_

#include <vector>
#include "common/op/ge_op_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/range_vistor.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "inc/pass.h"
namespace ge {
///
/// @ingroup domi_omg
/// @brief node pass
/// @author
///
class NodePass : public Pass<ge::Node> {
 public:
  ///
  /// run node pass
  /// @param [in] node node to be optimized
  /// @return SUCCESS optimized successfully
  /// @return TO_BE_DELETED optimized successfully and the node need to be deleted
  /// @return NOT_CHANGED not optimized
  /// @return others optimize failed
  /// @author
  ///
  virtual Status Run(ge::NodePtr node) = 0;

  /// Optimize to weight, Set the "is input const" flag of the output node to true
  /// @param [in] node node to be optimized
  /// @return SUCCESS optimized successfully
  /// @return others optimize failed
  ///
  Status SetOutNodeWeightDef(ge::NodePtr node, std::vector<ge::GeTensorPtr> &v_weight);

  /// Update node connection relationship
  /// @param [in] node The node to be optimized
  /// @return SUCCESS Optimized successfully
  /// @return FAILED Optimization failure
  ///
  Status UpdateNodeInfo(ge::NodePtr node);
};
}  // namespace ge
#endif  // GE_INC_NODE_PASS_H_
