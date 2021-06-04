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
#include "graph/build/task_generator.h"
#include "graph/manager/graph_mem_manager.h"
#include "graph/manager/graph_var_manager.h"
#undef protected
#undef private

using namespace std;
using namespace testing;
using namespace ge;
namespace {
const char *const kIsInputVar = "INPUT_IS_VAR";
const char *const kIsOutputVar = "OUTPUT_IS_VAR";
}
class UtestTaskGeneratorTest : public testing::Test {
 public:
  ge::ComputeGraphPtr BuildGraphFpProfiling() {
    ge::ut::GraphBuilder builder("graph");
    auto data = builder.AddNode("data", "phony", 1, 1);
    auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
    auto netoutput = builder.AddNode("netoutput", "NetOutput", 2, 0);
    auto op_desc = data->GetOpDesc();
    (void)AttrUtils::SetStr(op_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "IteratorV2");
    op_desc->SetOpKernelLibName("GE");
    builder.AddDataEdge(data, 0, addn1, 0);
    builder.AddDataEdge(addn1, 0, netoutput, 0);
    return builder.GetGraph();
  }
  ge::ComputeGraphPtr BuildGraphBpProfiling() {
    ge::ut::GraphBuilder builder("graph");
    auto data = builder.AddNode("data", "phony", 1, 1);
    auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
    auto netoutput = builder.AddNode("netoutput", "NetOutput", 2, 0);
    auto op_desc = data->GetOpDesc();
    (void)AttrUtils::SetStr(op_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "IteratorV2");
    op_desc->SetOpKernelLibName("GE");
    builder.AddDataEdge(data, 0, addn1, 0);
    builder.AddControlEdge(addn1, netoutput);
    return builder.GetGraph();
  }
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

TEST_F(UtestTaskGeneratorTest, AutoFindFpOpIndex) {
  auto graph = BuildGraphFpProfiling();
  TaskGenerator task_generator(nullptr, 0);
  ProfilingPoint profiling_point;
  profiling_point.fp_index = -1;
  EXPECT_EQ(task_generator.AutoFindFpOpIndex(graph, profiling_point), SUCCESS);
  // addn1 is fp
  EXPECT_EQ(profiling_point.fp_index, 2);
}

TEST_F(UtestTaskGeneratorTest, FindLastBpFromBpNode) {
  auto graph = BuildGraphBpProfiling();
  TaskGenerator task_generator(nullptr, 0);
  auto net_output = graph->FindNode("netoutput");
  // netoutput has no data input, return default value 0
  EXPECT_EQ(task_generator.FindLastBpFromBpNode(graph, net_output), 0);
}

TEST_F(UtestTaskGeneratorTest, UpdateOpIsVarAttr) {
  int64_t session_id = 0;
  ge::ComputeGraphPtr graph = BuildGraphWithVar(session_id);
  graph->SetSessionID(session_id);
  TaskGenerator task_generator(nullptr, 0);
  auto assign = graph->FindNode("assgin");
  task_generator.UpdateOpIsVarAttr(assign->GetOpDesc(), session_id);
  // input
  vector<bool> input_var;
  AttrUtils::GetListBool(assign->GetOpDesc(), kIsInputVar, input_var);
  EXPECT_EQ(input_var.size(), 2);
  EXPECT_EQ(input_var[0], true);
  EXPECT_EQ(input_var[1], false);
  // output
  vector<bool> output_var;
  AttrUtils::GetListBool(assign->GetOpDesc(), kIsOutputVar, output_var);
  EXPECT_EQ(output_var.size(), 1);
  EXPECT_EQ(output_var[0], true);

  MemManager::Instance().Finalize();
}
