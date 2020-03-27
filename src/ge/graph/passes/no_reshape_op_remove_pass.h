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

#ifndef GE_GRAPH_PASSES_NO_RESHAPE_OP_REMOVE_PASS_H_
#define GE_GRAPH_PASSES_NO_RESHAPE_OP_REMOVE_PASS_H_

#include <list>
#include <string>
#include <vector>

#include "graph/passes/base_pass.h"

namespace ge {
class NoReshapeOpRemovePass : public BaseNodePass {
 public:
  ///
  /// Entry of the NoReshapeOpRemovePass optimizer
  /// @param [in] node: Input Node
  /// @return SUCCESS: Dont find need to delete node
  /// @return NOT_CHANGED: find need to delete node
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status Run(ge::NodePtr &node) override;

 private:
  ///
  /// check node input and  output dims and format if can be delete
  /// @param [in] opDescPtr: To be checked opDesc
  /// @return SUCCESS: Check Node Success
  /// @return OTHERS: Check Node Failed
  /// @author
  ///
  Status CheckNodeShapeAndForamt(ge::NodePtr &node);

  ///
  /// check linked reshape op  if can be delete
  /// @param [in] node: To be compare Node with opDescPtr
  /// @return vector<ge::NodePtr>: To be delete nodes
  /// @author
  ///
  vector<ge::NodePtr> CheckLinkedReshape(ge::NodePtr &node);

  ///
  /// check node input and  output dims and format if can be delete
  /// @param [in] type: Check type
  /// @param [in/out] path: outnode list
  /// @return TRUE: To be delete
  /// @return FALSE: To be Not delete
  /// @author
  ///
  bool CheckOutDataNodesType(const string &type, std::list<ge::NodePtr> &path);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_NO_RESHAPE_OP_REMOVE_PASS_H_
