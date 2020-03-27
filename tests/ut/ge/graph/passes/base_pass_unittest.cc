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

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#define protected public
#include "graph/passes/base_pass.h"
#undef protected

#include "external/graph/ge_error_codes.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder_utils.h"

template class std::unordered_set<ge::NodePtr>;

using namespace domi;

namespace ge {
class TestPass : public BaseNodePass {
 public:
  TestPass() = default;
  TestPass(bool dead_loop) : dead_loop_(dead_loop), run_times_(0) {}

  Status Run(NodePtr &node) override {
    ++run_times_;
    iter_nodes_.push_back(node);
    auto iter = names_to_add_del_.find(node->GetName());
    if (iter != names_to_add_del_.end()) {
      for (const auto &node_name : iter->second) {
        auto del_node = node->GetOwnerComputeGraph()->FindNode(node_name);
        GraphUtils::IsolateNode(del_node, {0});
        AddNodeDeleted(del_node.get());
      }
    }
    iter = names_to_add_repass_.find(node->GetName());
    if (iter != names_to_add_repass_.end()) {
      auto all_nodes = node->GetOwnerComputeGraph()->GetAllNodes();
      for (const auto &node_name : iter->second) {
        for (auto &node_re_pass : all_nodes) {
          if (node_re_pass->GetName() == node_name) {
            AddRePassNode(node_re_pass);
            break;
          }
        }
      }
      if (!dead_loop_) {
        names_to_add_repass_.erase(iter);
      }
    }
    return SUCCESS;
  }
  void clear() { iter_nodes_.clear(); }
  std::vector<NodePtr> GetIterNodes() { return iter_nodes_; }

  void AddRePassNodeName(const std::string &iter_node, const std::string &re_pass_node) {
    names_to_add_repass_[iter_node].insert(re_pass_node);
  }
  void AddDelNodeName(const std::string &iter_node, const std::string &del_node) {
    names_to_add_del_[iter_node].insert(del_node);
  }
  unsigned int GetRunTimes() { return run_times_; }

 private:
  std::vector<NodePtr> iter_nodes_;
  std::map<std::string, std::unordered_set<std::string>> names_to_add_del_;
  std::map<std::string, std::unordered_set<std::string>> names_to_add_repass_;
  bool dead_loop_;
  unsigned int run_times_;
};

class TestDelPass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override { return SUCCESS; }
};

class UTEST_graph_passes_base_pass : public testing::Test {
 protected:
  UTEST_graph_passes_base_pass() {
    auto p1 = new TestPass;
    names_to_pass_.push_back(std::make_pair("test1", p1));
  }
  void SetUp() override {
    for (auto &name_to_pass : names_to_pass_) {
      dynamic_cast<TestPass *>(name_to_pass.second)->clear();
    }
  }
  ~UTEST_graph_passes_base_pass() override {
    for (auto &name_to_pass : names_to_pass_) {
      delete name_to_pass.second;
    }
  }
  NamesToPass names_to_pass_;
};

using namespace domi;

///      reshape1
///        |
///       add1
///     /     \
///    |      |
///  data1  const1
ComputeGraphPtr BuildGraph1() {
  auto builder = ut::GraphBuilder("g1");
  auto data = builder.AddNode("data1", DATA, 0, 1);
  auto a1 = builder.AddNode("add1", ADD, 2, 1);
  auto c1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto r1 = builder.AddNode("reshape1", RESHAPE, 1, 1);

  builder.AddDataEdge(data, 0, a1, 0);
  builder.AddDataEdge(c1, 0, a1, 1);
  builder.AddDataEdge(a1, 0, r1, 0);

  return builder.GetGraph();
}

///               sum1
///             /     \
///            /       \
///          /          \
///      reshape1      addn1
///        |      c      |
///       add1  <---  shape1
///     /     \         |
///    |      |         |
///  data1  const1    const2
ComputeGraphPtr BuildGraph2() {
  auto builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto const2 = builder.AddNode("const2", CONSTANT, 0, 1);
  auto add1 = builder.AddNode("add1", ADD, 2, 1);
  auto shape1 = builder.AddNode("shape1", SHAPE, 1, 1);
  auto reshape1 = builder.AddNode("reshape1", RESHAPE, 1, 1);
  auto addn1 = builder.AddNode("addn1", ADDN, 1, 1);
  auto sum1 = builder.AddNode("sum1", SUM, 2, 1);

  builder.AddDataEdge(data1, 0, add1, 0);
  builder.AddDataEdge(const1, 0, add1, 1);
  builder.AddDataEdge(const2, 0, shape1, 0);
  builder.AddControlEdge(shape1, add1);
  builder.AddDataEdge(add1, 0, reshape1, 0);
  builder.AddDataEdge(shape1, 0, addn1, 0);
  builder.AddDataEdge(reshape1, 0, sum1, 0);
  builder.AddDataEdge(addn1, 0, sum1, 1);

  return builder.GetGraph();
}

///   rnextiteration
///    |  |
///   merge
///    |
///  data1
ComputeGraphPtr BuildGraph3() {
  auto builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto merge1 = builder.AddNode("merge1", MERGE, 2, 1);
  auto next1 = builder.AddNode("next1", NEXTITERATION, 1, 1);

  builder.AddDataEdge(data1, 0, merge1, 0);
  builder.AddDataEdge(merge1, 0, next1, 0);
  builder.AddDataEdge(next1, 0, merge1, 1);
  builder.AddControlEdge(merge1, next1);
  builder.AddControlEdge(next1, merge1);

  return builder.GetGraph();
}

void CheckIterOrder(TestPass *pass, std::vector<std::unordered_set<std::string>> &nodes_layers) {
  std::unordered_set<std::string> layer_nodes;
  size_t layer_index = 0;
  for (const auto &node : pass->GetIterNodes()) {
    layer_nodes.insert(node->GetName());
    EXPECT_LT(layer_index, nodes_layers.size());
    if (layer_nodes == nodes_layers[layer_index]) {
      layer_index++;
      layer_nodes.clear();
    }
  }
  EXPECT_EQ(layer_index, nodes_layers.size());
}

///      Op1
///       |
///     Merge
///      / \
///    Op2 Op3
TEST_F(UTEST_graph_passes_base_pass, DelIsolateFail) {
  auto builder = ut::GraphBuilder("g1");
  auto merge_node = builder.AddNode("Merge", MERGE, 1, 1);
  auto node1 = builder.AddNode("Op1", RELU, 1, 1);
  auto node2 = builder.AddNode("Op2", CONVOLUTION, 1, 1);
  auto node3 = builder.AddNode("Op3", CONVOLUTION, 1, 1);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), merge_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), node3->GetInDataAnchor(0));

  EXPECT_EQ(node1->GetOutDataNodes().size(), 1);

  TestDelPass del_pass;
  auto ret = del_pass.IsolateAndDeleteNode(merge_node, {0, -1});
  EXPECT_EQ(ret, FAILED);

  OpDescPtr op_desc = std::make_shared<OpDesc>("merge", MERGE);
  NodePtr node = shared_ptr<Node>(new (std::nothrow) Node(op_desc, nullptr));
  ret = del_pass.IsolateAndDeleteNode(node, {0, -1});
  EXPECT_EQ(ret, FAILED);
}

///      Op1
///       |
///     Merge
///      / \
///    Op2 Op3
TEST_F(UTEST_graph_passes_base_pass, DelIsolateSuccess) {
  auto builder = ut::GraphBuilder("g1");
  auto merge_node = builder.AddNode("Merge", MERGE, 1, 2);
  auto node1 = builder.AddNode("Op1", RELU, 1, 1);
  auto node2 = builder.AddNode("Op2", CONVOLUTION, 1, 1);
  auto node3 = builder.AddNode("Op3", CONVOLUTION, 1, 1);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), merge_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), node3->GetInDataAnchor(0));

  EXPECT_EQ(node1->GetOutDataNodes().size(), 1);

  TestDelPass del_pass;
  auto ret = del_pass.IsolateAndDeleteNode(merge_node, {0, -1});
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UTEST_graph_passes_base_pass, DataGraph) {
  auto graph = BuildGraph1();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass_), SUCCESS);
  auto *pass = dynamic_cast<TestPass *>(names_to_pass_[0].second);

  EXPECT_EQ(pass->GetIterNodes().size(), 4);
  std::vector<std::unordered_set<std::string>> layers;
  layers.push_back({"data1", "const1"});
  layers.push_back({"add1"});
  layers.push_back({"reshape1"});
  CheckIterOrder(pass, layers);
}

TEST_F(UTEST_graph_passes_base_pass, GraphWithControlLink) {
  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass_), SUCCESS);
  auto *pass = dynamic_cast<TestPass *>(names_to_pass_[0].second);

  EXPECT_EQ(pass->GetIterNodes().size(), 8);
  EXPECT_EQ(pass->GetIterNodes().at(3)->GetName(), "shape1");

  std::vector<std::unordered_set<std::string>> layers;
  layers.push_back({"data1", "const1", "const2"});
  layers.push_back({"shape1"});
  layers.push_back({"add1", "addn1", "reshape1"});
  layers.push_back({"sum1"});
  CheckIterOrder(pass, layers);
}

TEST_F(UTEST_graph_passes_base_pass, RePassAfter) {
  NamesToPass names_to_pass;
  auto test_pass = TestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddRePassNodeName("add1", "sum1");
  test_pass.AddRePassNodeName("shape1", "sum1");
  test_pass.AddRePassNodeName("shape1", "add1");
  test_pass.AddRePassNodeName("data1", "add1");

  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetIterNodes().size(), 8);
}

TEST_F(UTEST_graph_passes_base_pass, RePassBefore) {
  NamesToPass names_to_pass;
  auto test_pass = TestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddRePassNodeName("add1", "data1");

  auto graph = BuildGraph1();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetIterNodes().size(), 5);
  EXPECT_EQ(test_pass.GetIterNodes().at(2)->GetName(), "add1");
  EXPECT_EQ(test_pass.GetIterNodes().at(3)->GetName(), "reshape1");
  EXPECT_EQ(test_pass.GetIterNodes().at(4)->GetName(), "data1");
}

TEST_F(UTEST_graph_passes_base_pass, RePassBeforeMultiTimes) {
  NamesToPass names_to_pass;
  auto test_pass = TestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddRePassNodeName("add1", "data1");
  test_pass.AddRePassNodeName("add1", "const1");
  test_pass.AddRePassNodeName("reshape1", "data1");

  auto graph = BuildGraph1();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetIterNodes().size(), 6);
  EXPECT_EQ(test_pass.GetIterNodes().at(2)->GetName(), "add1");
  EXPECT_EQ(test_pass.GetIterNodes().at(3)->GetName(), "reshape1");
}

TEST_F(UTEST_graph_passes_base_pass, DelAfter) {
  NamesToPass names_to_pass;
  auto test_pass = TestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddDelNodeName("add1", "sum1");

  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetIterNodes().size(), 7);
}

TEST_F(UTEST_graph_passes_base_pass, DelAfterMultiple) {
  NamesToPass names_to_pass;
  auto test_pass = TestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddDelNodeName("add1", "sum1");
  test_pass.AddDelNodeName("add1", "reshape1");

  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetIterNodes().size(), 6);
}

TEST_F(UTEST_graph_passes_base_pass, DelAfterBreakLink) {
  NamesToPass names_to_pass;
  auto test_pass = TestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddDelNodeName("shape1", "add1");
  test_pass.AddDelNodeName("shape1", "addn1");
  test_pass.AddRePassNodeName("shape1", "shape1");
  test_pass.AddRePassNodeName("shape1", "reshape1");
  test_pass.AddRePassNodeName("shape1", "sum1");

  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetIterNodes().size(), 7);
}

TEST_F(UTEST_graph_passes_base_pass, DelSelfAndAfter) {
  NamesToPass names_to_pass;
  auto test_pass = TestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddDelNodeName("shape1", "add1");
  test_pass.AddDelNodeName("shape1", "addn1");

  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetIterNodes().size(), 4);
}

TEST_F(UTEST_graph_passes_base_pass, DelBefore) {
  NamesToPass names_to_pass;
  auto test_pass = TestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddDelNodeName("reshape1", "add1");
  test_pass.AddDelNodeName("sum1", "addn1");

  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetIterNodes().size(), 8);
}

TEST_F(UTEST_graph_passes_base_pass, RePassAndDel) {
  NamesToPass names_to_pass;
  auto test_pass = TestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddRePassNodeName("add1", "sum1");
  test_pass.AddDelNodeName("reshape1", "sum1");

  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetIterNodes().size(), 7);
}

TEST_F(UTEST_graph_passes_base_pass, DeadLoop) {
  NamesToPass names_to_pass;
  auto test_pass = TestPass(true);
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddRePassNodeName("add1", "sum1");
  test_pass.AddRePassNodeName("sum1", "add1");

  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetRunTimes(), 1007);
}

TEST_F(UTEST_graph_passes_base_pass, WhileLoop) {
  NamesToPass names_to_pass;
  auto test_pass = TestPass(true);
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  auto graph = BuildGraph3();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
}
}  // namespace ge
