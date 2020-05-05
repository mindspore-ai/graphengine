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

#ifndef GE_GRAPH_PASSES_WHILE_OP_LABEL_PASS_H_
#define GE_GRAPH_PASSES_WHILE_OP_LABEL_PASS_H_

#include "graph/node.h"
#include "graph/label/label_maker.h"
/*******************************************************************************
                                                                +-----------+
                                                                |   Node    |
                                                                +-----------+
                                                                |   Node    |
                                                                +-----------+
                                                                |   While   |
                                                                +-----------+
              +-----------+
              |   Node    |                                     +-----------+
              +-----------+                                     | LabelSet  |\
              |   Node    |                                     +-----------+ \
              +-----------+                                     |     c     |  \
              |   While   |                                     +-----------+   A
              +-----------+                                     |     o     |   |
              |   Node    |                                     +-----------+   |
              +-----------+                                     |     n     |   |
              |   Node    |                                     +-----------+   |
              +-----------+                                     |     d     |   |
              |   Node    |                                     +-----------+   |
              +-----------+                                    /|SwitchByIdx|   |
                                                              / +-----------+   |
                                               ====>         /                  |
                                                            | \ +-----------+   |
                                                            |  \|LabelSet(1)|   |
                                                            |   +-----------+   |
    +-----------+      +-----------+                        |   |     b     |   |
    |     c     |      |     b     |                        |   +-----------+   |
    +-----------+      +-----------+                        |   |     o     |   |
    |     o     |      |     o     |                        |   +-----------+   |
    +-----------+      +-----------+                        |   |     d     |   |
    |     n     |      |     d     |                        |   +-----------+   |
    +-----------+      +-----------+                        |   |     y     |  /
    |     d     |      |     y     |                        V   +-----------+ /
    +-----------+      +-----------+                         \  | LabelGoto |/
                                                              \ +-----------+
                                                               \|LabelSet(0)|
                                                                +-----------+

                                                                +-----------+
                                                                |   Node    |
                                                                +-----------+
                                                                |   Node    |
                                                                +-----------+
                                                                |   Node    |
                                                                +-----------+
*******************************************************************************/

namespace ge {
class WhileOpLabelMaker : public LabelMaker {
 public:
  WhileOpLabelMaker(const ComputeGraphPtr &graph, const NodePtr &owner) : LabelMaker(graph, owner) {}

  ~WhileOpLabelMaker() override {}

  virtual Status Run(uint32_t &label_index);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_WHILE_OP_LABEL_PASS_H_
