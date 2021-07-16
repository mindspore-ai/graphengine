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
#include <memory>

#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "omg/omg_inner_types.h"
#include "../passes/graph_builder_utils.h"

#define protected public
#define private public
#include "graph/build/memory/binary_block_mem_assigner.h"
#include "graph/build/memory/graph_mem_assigner.h"
#include "graph/build/memory/hybrid_mem_assigner.h"
#include "graph/build/memory/max_block_mem_assigner.h"
#include "graph/manager/graph_var_manager.h"
#undef protected
#undef private

using namespace std;
using namespace testing;
using namespace ge;
using domi::GetContext;

class UtestTaskGeneratorTest : public testing::Test {
 public:
    ge::ComputeGraphPtr BuildGraphWithVar(int64_t session_id) {
      // init
      MemManager::Instance().Initialize(std::vector<rtMemType_t>({RT_MEMORY_HBM}));
      VarManager::Instance(session_id)->Init(0, 0, 0, 0);
      ge::ut::GraphBuilder builder("graph");
      auto var_input = builder.AddNode("var", "Variable", 1, 1);
      auto const_input = builder.AddNode("const", "Const", 1, 1);
      auto assign = builder.AddNode("assgin", "Assign", 2, 1);
      // add link
      builder.AddDataEdge(var_input, 0, assign, 0);
      builder.AddDataEdge(const_input, 0, assign, 1);
      // set offset
      var_input->GetOpDesc()->SetOutputOffset({10000});
      const_input->GetOpDesc()->SetOutputOffset({1000});
      assign->GetOpDesc()->SetInputOffset({10100, 1000});
      assign->GetOpDesc()->SetOutputOffset({10100});
      // set inner offset
      int64_t inner_offset = 100;
      ge::AttrUtils::SetInt(assign->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_INNER_OFFSET, inner_offset);
      ge::AttrUtils::SetInt(assign->GetOpDesc()->MutableOutputDesc(0), ATTR_NAME_INNER_OFFSET, inner_offset);
      // add var addr
      VarManager::Instance(session_id)->var_resource_->var_offset_map_.emplace(10000, RT_MEMORY_HBM);

      return builder.GetGraph();
    }

protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(UtestMemoryAssignerTest, graph_memory_assign_continuous_input) {
  ge::ComputeGraphPtr compute_graph = make_shared<ge::ComputeGraph>("");
  GraphMemoryAssigner graph_mem_assigner(compute_graph);
  map<uint64_t, size_t> mem_type_to_offset = {};
  Status ret = ReAssignMemory(false, mem_type_to_offset);
  EXPECT_EQ(ret, ge::FAILED);
}

