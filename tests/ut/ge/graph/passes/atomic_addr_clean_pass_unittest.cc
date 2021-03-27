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

#include <gtest/gtest.h>
#include "graph/passes/atomic_addr_clean_pass.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "inc/pass_manager.h"
using namespace testing;

namespace ge {
class UtestGraphPassesAtomicAddrCleanPass : public Test {
public:
  UtestGraphPassesAtomicAddrCleanPass() {
    graph_ = std::make_shared<ComputeGraph>("test");
  }

  NodePtr NewNode(const string &name, const string &type, int input_cnt, int output_cnt) {
    OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
    for (int i = 0; i < input_cnt; ++i) {
      op_desc->AddInputDesc(GeTensorDesc());
    }
    for (int i = 0; i < output_cnt; ++i) {
      op_desc->AddOutputDesc(GeTensorDesc());
    }
    NodePtr node = graph_->AddNode(op_desc);
    return node;
  }

  ComputeGraphPtr graph_;
};

// node1 -> node2 -> node3
TEST_F(UtestGraphPassesAtomicAddrCleanPass, pass_run_success) {
  auto node1 = NewNode("node1", DATA, 0, 1);
  auto node2 = NewNode("node2", RELU, 1, 1);
  auto node3 = NewNode("node3", NETOUTPUT, 1, 0);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
  AtomicAddrCleanPass atomi_addr_clean_pass;
  Status ret = atomi_addr_clean_pass.Run(graph_);
  EXPECT_EQ(ret, SUCCESS);
}
}  // namespace ge
