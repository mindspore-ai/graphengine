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

class UtestMemoryAssignerTest : public testing::Test {
 public:
  ge::OpDescPtr CreateOpWithWsSize(const string &name, int64_t wsByte, const string &type = "some") {
    ge::OpDescPtr op_def = make_shared<ge::OpDesc>(name, type);
    auto desc_temp_ptr = make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;

    TensorUtils::SetSize(desc_temp, 1024);
    op_def->AddInputDesc(desc_temp);
    op_def->AddOutputDesc(desc_temp);

    std::vector<int64_t> workspace_bytes;
    workspace_bytes.push_back(wsByte);
    op_def->SetWorkspaceBytes(workspace_bytes);
    return op_def;
  }
  ge::OpDescPtr CreateRefOpWithWsSize(const string &name, int64_t wsByte, const string &type = "some") {
    ge::OpDescPtr op_def = make_shared<ge::OpDesc>(name, type);
    auto desc_temp_ptr = make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;

    TensorUtils::SetSize(desc_temp, 1024);
    op_def->AddInputDesc(desc_temp);

    auto desc_output_ptr = make_shared<ge::GeTensorDesc>();
    auto desc_output = *desc_output_ptr;
    TensorUtils::SetSize(desc_output, 6500);
    ge::TensorUtils::SetReuseInput(desc_output, true);
    ge::TensorUtils::SetReuseInputIndex(desc_output, 0);
    op_def->AddOutputDesc(desc_output);

    std::vector<int64_t> workspace_bytes;
    workspace_bytes.push_back(wsByte);
    op_def->SetWorkspaceBytes(workspace_bytes);
    return op_def;
  }
  void MakeGraph(ge::ComputeGraphPtr &graph, const string &type = "some") {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 6000, type);
    op_def_a->SetStreamId(0);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 120000);
    op_def_b->SetStreamId(0);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 16000);
    op_def_c->SetStreamId(1);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 24000);
    op_def_d->SetStreamId(2);
    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 24000);
    op_def_e->SetStreamId(3);
    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 30000);
    op_def_f->SetStreamId(2);
    ge::OpDescPtr op_def_g = CreateOpWithWsSize("G", 32000);
    op_def_g->SetStreamId(3);
    ge::OpDescPtr op_def_h = CreateOpWithWsSize("H", 48000);
    op_def_h->SetStreamId(2);
    ge::OpDescPtr op_def_i = CreateOpWithWsSize("I", 60000);
    op_def_i->SetStreamId(2);
    ge::OpDescPtr op_def_j = CreateOpWithWsSize("J", 256000, NETOUTPUT);
    op_def_j->SetStreamId(3);

    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);
    ge::NodePtr node_g = graph->AddNode(op_def_g);
    ge::NodePtr node_h = graph->AddNode(op_def_h);
    ge::NodePtr node_i = graph->AddNode(op_def_i);
    ge::NodePtr node_j = graph->AddNode(op_def_j);

    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_g->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_g->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_f->GetOutDataAnchor(0), node_h->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_g->GetOutDataAnchor(0), node_j->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_h->GetOutDataAnchor(0), node_i->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_i->GetOutDataAnchor(0), node_j->GetInDataAnchor(1));

    GetContext().out_nodes_map["H"] = {0};
    GetContext().out_nodes_map["I"] = {0};
    GetContext().out_nodes_map["J"] = {0};
    graph->TopologicalSorting();
  }

  void MakeReuseGraph(ge::ComputeGraphPtr graph) {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 6000);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 120000);
    ge::OpDescPtr op_def_c = CreateRefOpWithWsSize("C", 120000);
    ge::OpDescPtr op_def_d = make_shared<ge::OpDesc>("D", "CONSTANT");

    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);

    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    GetContext().out_nodes_map["B"] = {0};
    GetContext().out_nodes_map["C"] = {0};
    graph->TopologicalSorting();
  }

  ComputeGraphPtr MakeCascadeContinuousMemoryGraph() {
    ge::ut::GraphBuilder builder("graph");
    auto data = builder.AddNode("data", "Data", 1, 1);
    auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
    auto addn2 = builder.AddNode("addn2", "AddN", 1, 1);
    auto addn3 = builder.AddNode("addn3", "AddN", 1, 1);
    auto concat1 = builder.AddNode("concat1", "Concat", 2, 1);
    auto concat2 = builder.AddNode("concat2", "Concat", 2, 1);
    auto netoutput = builder.AddNode("netoutput", "NetOutput", 2, 0);

    ge::AttrUtils::SetBool(concat1->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
    ge::AttrUtils::SetBool(concat1->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT_ALLOC, true);
    ge::AttrUtils::SetBool(concat1->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);

    ge::AttrUtils::SetBool(concat2->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
    ge::AttrUtils::SetBool(concat2->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT_ALLOC, true);
    ge::AttrUtils::SetBool(concat2->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);

    addn1->GetOpDesc()->SetOutputOffset({100});
    addn2->GetOpDesc()->SetOutputOffset({200});
    concat1->GetOpDesc()->SetOutputOffset({100});
    addn3->GetOpDesc()->SetOutputOffset({700});
    concat2->GetOpDesc()->SetOutputOffset({500});

    ge::AttrUtils::SetListInt(addn1->GetOpDesc(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, {100});
    ge::AttrUtils::SetListInt(addn2->GetOpDesc(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, {100});
    ge::AttrUtils::SetListInt(addn3->GetOpDesc(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, {100});
    ge::AttrUtils::SetListInt(concat1->GetOpDesc(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, {200});
    ge::AttrUtils::SetListInt(concat2->GetOpDesc(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, {300});


    builder.AddDataEdge(data, 0, addn1, 0);
    builder.AddDataEdge(data, 0, addn2, 0);
    builder.AddDataEdge(addn1, 0, concat1, 0);
    builder.AddDataEdge(addn2, 0, concat1, 1);
    builder.AddDataEdge(concat1, 0, concat2, 0);
    builder.AddDataEdge(addn3, 0, concat2, 1);

    return builder.GetGraph();
  }

  ComputeGraphPtr MakeRefNodeGraph() {
    ge::ut::GraphBuilder builder("graph");
    auto var_input = builder.AddNode("var", "Variable", 1, 1);
    auto const_input = builder.AddNode("const", "Const", 1, 1);
    auto assign = builder.AddNode("assgin", "Assign", 2, 1);
     // add link
    builder.AddDataEdge(var_input, 0, assign, 0);
    builder.AddDataEdge(const_input, 0, assign, 1);
    // set offset
    assign->GetOpDesc()->SetInputOffset({100, 0});
    assign->GetOpDesc()->SetOutputOffset({10000});
    var_input->GetOpDesc()->SetOutputOffset({10000});
    const_input->GetOpDesc()->SetOutputOffset({1000});
    // set mem type
    ge::AttrUtils::SetListInt(assign->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, {RT_MEMORY_HBM, RT_MEMORY_L1});
    // set ref
    auto output_tensordesc = assign->GetOpDesc()->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc, true);
    uint32_t reuse_input_index = 0;
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc, reuse_input_index);

    return builder.GetGraph();
  }

 protected:
  void SetUp() {}

  void TearDown() { GetContext().out_nodes_map.clear(); }
};

/*
TEST_F(UtestMemoryAssignerTest, MemoryBlock_Resize_RealSizeList_is_empty) {
  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");
  ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 6000);
  ge::NodePtr node_a = graph->AddNode(op_def_a);
  MemoryBlock* memory_block = new MemoryBlock(0);
  memory_block->Init(1, kOutput, node_a, 0, 1);
  memory_block->real_size_list_.clear();
  memory_block->Resize();

  EXPECT_EQ(memory_block->Size(), 0);

  delete memory_block;
}
*/

namespace ge {

class MockBlockMemAssigner : public BlockMemAssigner {
 public:
  explicit MockBlockMemAssigner(ge::ComputeGraphPtr compute_graph, const std::map<std::string, std::string> &anchor_to_symbol, const std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors) : BlockMemAssigner(compute_graph, anchor_to_symbol, symbol_to_anchors) {};

  virtual ~MockBlockMemAssigner(){};

  Status GetMemoryRanges(std::vector<int64_t> &ranges) override { return FAILED; }
};
}  // namespace ge

// when check GetMemoryRanges return fail, Assign return fail
TEST_F(UtestMemoryAssignerTest, Mock_block_mem_assigner_failed) {
  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");
  MakeGraph(graph);
  std::map<std::string, std::string> anchor_to_symbol;
  std::map<std::string, std::list<NodeIndexIO>> symbol_to_anchors;
  EXPECT_EQ(GraphUtils::GetRefMapping(graph, symbol_to_anchors, anchor_to_symbol), GRAPH_SUCCESS);

  MockBlockMemAssigner mock_assigner(graph, anchor_to_symbol, symbol_to_anchors);
  EXPECT_EQ(mock_assigner.Assign(), FAILED);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_assign_continuous_input) {
  ge::ComputeGraphPtr graph = MakeCascadeContinuousMemoryGraph();
  auto addn1 = graph->FindNode("addn1");
  auto addn2 = graph->FindNode("addn2");
  EXPECT_EQ(addn1->GetOpDesc()->GetOutputOffset()[0], 100);
  EXPECT_EQ(addn2->GetOpDesc()->GetOutputOffset()[0], 200);
  GraphMemoryAssigner memoryAssigner(graph);
  MemoryOffset memory_offset(RT_MEMORY_HBM, 0);
  memoryAssigner.memory_offset_.emplace(RT_MEMORY_HBM, memory_offset);
  EXPECT_EQ(memoryAssigner.ReAssignContinuousMemory(false), GRAPH_SUCCESS);
  EXPECT_EQ(addn1->GetOpDesc()->GetOutputOffset()[0], 500);
  EXPECT_EQ(addn2->GetOpDesc()->GetOutputOffset()[0], 600);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_set_last_used_attr) {
  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");
  MakeGraph(graph);
  auto node_f = graph->FindNode("F");
  MemoryAssigner memory_assigner(graph);
  map<int64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(false, mem_offset, zero_memory_size), GRAPH_SUCCESS);

  bool flag = 0;
  (void) ge::AttrUtils::GetBool(node_f->GetOpDesc()->GetInputDesc(0), ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, flag);
  EXPECT_EQ(flag, true);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_assign_ref_var) {
  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");
  MakeGraph(graph, VARIABLE);
  auto node_a = graph->FindNode("A");
  auto node_b = graph->FindNode("B");
  std::string value = "A";
  (void) ge::AttrUtils::SetStr(node_b->GetOpDesc()->MutableOutputDesc(0), REF_VAR_SRC_VAR_NAME, value);
  MemoryAssigner memory_assigner(graph);
  map<int64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  EXPECT_EQ(memory_assigner.AssignMemory(false, mem_offset, zero_memory_size), GRAPH_SUCCESS);

  EXPECT_EQ(node_b->GetOpDesc()->GetOutputOffset()[0], node_a->GetOpDesc()->GetOutputOffset()[0]);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_assign_ref_var_not_found) {
  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");
  MakeGraph(graph, VARIABLE);

  ge::ComputeGraphPtr sub_graph = make_shared<ge::ComputeGraph>("");
  MakeReuseGraph(sub_graph);
  graph->AddSubGraph(sub_graph);

  auto node_a = graph->FindNode("A");
  auto node_b = graph->FindNode("B");
  std::string value = "M";
  (void) ge::AttrUtils::SetStr(node_b->GetOpDesc()->MutableOutputDesc(0), REF_VAR_SRC_VAR_NAME, value);
  MemoryAssigner memory_assigner(graph);
  map<int64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  EXPECT_NE(memory_assigner.AssignMemory(false, mem_offset, zero_memory_size), GRAPH_SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_assign_set_input_offset) {
  ge::ComputeGraphPtr graph = MakeRefNodeGraph();
  auto assgin = graph->FindNode("assgin");
  EXPECT_EQ(assgin->GetOpDesc()->GetOutputOffset()[0], 10000);
  EXPECT_EQ(assgin->GetOpDesc()->GetInputOffset()[0], 100);
  EXPECT_EQ(assgin->GetOpDesc()->GetInputOffset()[1], 0);
  GraphMemoryAssigner memoryAssigner(graph);
  MemoryOffset memory_offset(RT_MEMORY_HBM, 0);
  memoryAssigner.memory_offset_.emplace(RT_MEMORY_HBM, memory_offset);
  EXPECT_EQ(memoryAssigner.SetInputOffset(), GRAPH_SUCCESS);
  EXPECT_EQ(assgin->GetOpDesc()->GetOutputOffset()[0], 10100);
  EXPECT_EQ(assgin->GetOpDesc()->GetInputOffset()[0], 10100);
  EXPECT_EQ(assgin->GetOpDesc()->GetInputOffset()[1], 0);
  EXPECT_EQ(memoryAssigner.CheckOffset(), GRAPH_SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_assign_update_ref_op_offset_reverse) {
    ge::ut::GraphBuilder builder("graph");
    auto data_input = builder.AddNode("data", "Data", 1, 1);
    auto const_input = builder.AddNode("const", "Const", 1, 1);
    auto add = builder.AddNode("add", "Add", 2, 1);
     // add link
    builder.AddDataEdge(data_input, 0, add, 0);
    builder.AddDataEdge(const_input, 0, add, 1);
    // set ref
    uint32_t reuse_input_index = 0;
    auto output_tensordesc = data_input->GetOpDesc()->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc, true);
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc, reuse_input_index);
    auto output_tensordesc1 = add->GetOpDesc()->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc1, true);
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc1, reuse_input_index);
    ge::ComputeGraphPtr graph = builder.GetGraph();

    GraphMemoryAssigner memoryAssigner(graph);
    EXPECT_EQ(memoryAssigner.UpdateRefOpOffsetReverse(add), SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_assign_atomic_output_and_workspace) {
  ge::ut::GraphBuilder builder("graph");
  auto data_input = builder.AddNode("data", "Data", 1, 1);
  auto const_input = builder.AddNode("const", "Const", 1, 1);
  auto add = builder.AddNode("add", "Add", 2, 1);
  // add link
  builder.AddDataEdge(data_input, 0, add, 0);
  builder.AddDataEdge(const_input, 0, add, 1);
  ge::ComputeGraphPtr graph = builder.GetGraph();

  auto node = graph->FindNode("add");
  EXPECT_NE(node, nullptr);
  auto output_tensor_desc = node->GetOpDesc()->MutableOutputDesc(0);
  ge::TensorUtils::SetSize(*output_tensor_desc, 100);
  vector<int64_t> output_list = {0};
  node->GetOpDesc()->SetOutputOffset(output_list);
  vector<int64_t> workspace_list = {0};
  node->GetOpDesc()->SetWorkspace(workspace_list);
  vector<int64_t> atomic_output_index = {0};
  bool set_attr = ge::AttrUtils::SetListInt(node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
  EXPECT_EQ(set_attr, true);

  map<string, map<int64_t, int64_t>> workspace_info;
  workspace_info["add"][0] = 100;
  set_attr = node->GetOpDesc()->SetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspace_info);
  EXPECT_EQ(set_attr, true);

  {
    bool is_fusion_node = false;
    set_attr = ge::AttrUtils::SetBool(node->GetOpDesc(), ATOMIC_ATTR_IS_FUSION_NODE, is_fusion_node);
    EXPECT_EQ(set_attr, true);

    GraphMemoryAssigner graph_memory_assigner(graph);
    graph_memory_assigner.memory_offset_.insert({RT_MEMORY_HBM, MemoryOffset(RT_MEMORY_HBM, 0)});
    vector<int64_t> mem_offset_end;
    Status ret = graph_memory_assigner.AssignAtomicOutputAndWorkspaceMemory(node, mem_offset_end);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(mem_offset_end.size(), 2);
    MemoryOffset mem_offset = graph_memory_assigner.memory_offset_.at(RT_MEMORY_HBM);
    EXPECT_EQ(mem_offset.mem_offset_, 1024);
  }

  {
    bool is_fusion_node = true;
    set_attr = ge::AttrUtils::SetBool(node->GetOpDesc(), ATOMIC_ATTR_IS_FUSION_NODE, is_fusion_node);
    EXPECT_EQ(set_attr, true);

    GraphMemoryAssigner graph_memory_assigner(graph);
    graph_memory_assigner.memory_offset_.insert({RT_MEMORY_HBM, MemoryOffset(RT_MEMORY_HBM, 0)});
    vector<int64_t> mem_offset_end;
    Status ret = graph_memory_assigner.AssignAtomicOutputAndWorkspaceMemory(node, mem_offset_end);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(mem_offset_end.size(), 2);
    MemoryOffset mem_offset = graph_memory_assigner.memory_offset_.at(RT_MEMORY_HBM);
    EXPECT_EQ(mem_offset.mem_offset_, 1024);
  }
}
