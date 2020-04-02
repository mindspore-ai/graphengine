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

#include <string>
#include <vector>
#include <gtest/gtest.h>

#define protected public
#define private public
#include "graph/manager/graph_manager_utils.h"
#undef protected
#undef private

#include "graph/build/logical_stream_allocator.h"

#include "common/types.h"
#include "common/util.h"

#include "graph/compute_graph.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"

using namespace std;

namespace ge {
class UtestLogicalStreamAllocator : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  static SubGraphInfoPtr BuildSubGraph(ComputeGraphPtr compute_graph, const string &engine_name,
                                       const string &stream_label = "") {
    SubGraphInfoPtr subgraph = make_shared<SubGraphInfo>();
    subgraph->SetSubGraph(compute_graph);
    subgraph->SetEngineName(engine_name);
    subgraph->SetStreamLabel(stream_label);
    return subgraph;
  }

  NodePtr AddPlaceHolder(ComputeGraphPtr compute_graph, const string &name) {
    OpDescPtr op_desc = std::make_shared<OpDesc>(name, "PlaceHolder");
    op_desc->AddInputDesc(GeTensorDesc());
    op_desc->AddOutputDesc(GeTensorDesc());
    NodePtr node = compute_graph->AddNode(op_desc);
    node->SetOwnerComputeGraph(compute_graph);
    return node;
  }

  NodePtr AddEnd(ComputeGraphPtr compute_graph, const string &name) {
    OpDescPtr op_desc = std::make_shared<OpDesc>(name, "End");
    op_desc->AddInputDesc(GeTensorDesc());
    op_desc->AddOutputDesc(GeTensorDesc());
    NodePtr node = compute_graph->AddNode(op_desc);
    node->SetOwnerComputeGraph(compute_graph);
    return node;
  }

  void AddPlaceHolderAndEnd(SubGraphInfoPtr subgraph, int in_num, int out_num) {
    ComputeGraphPtr compute_graph = subgraph->GetSubGraph();

    std::unordered_map<ge::NodePtr, ge::NodePtr> pld_2_end_map;
    if (in_num == 1) {
      NodePtr node = AddPlaceHolder(compute_graph, "placeholder");
      pld_2_end_map.emplace(node, nullptr);
    } else {
      for (int i = 0; i < in_num; i++) {
        NodePtr node = AddPlaceHolder(compute_graph, "placeholder" + to_string(i + 1));
        pld_2_end_map.emplace(node, nullptr);
      }
    }
    subgraph->SetPld2EndMap(pld_2_end_map);

    std::unordered_map<ge::NodePtr, ge::NodePtr> end_2_pld_map;
    if (out_num == 1) {
      NodePtr node = AddEnd(compute_graph, "end");
      end_2_pld_map.emplace(node, nullptr);
    } else {
      for (int i = 0; i < out_num; i++) {
        NodePtr node = AddEnd(compute_graph, "end" + to_string(i + 1));
        end_2_pld_map.emplace(node, nullptr);
      }
    }

    subgraph->SetEnd2PldMap(end_2_pld_map);
  }

  SubGraphInfoPtr CreateDataSubgraph(const string &name = "data") {
    ComputeGraphPtr compute_graph = make_shared<ComputeGraph>(name);
    OpDescPtr op_desc = std::make_shared<OpDesc>("data", "Data");
    op_desc->AddOutputDesc(GeTensorDesc());
    compute_graph->AddNode(op_desc);

    SubGraphInfoPtr subgraph = BuildSubGraph(compute_graph, "ge_local", "");
    AddPlaceHolderAndEnd(subgraph, 0, 1);
    return subgraph;
  }

  SubGraphInfoPtr CreateConstSubgraph(const string &name = "const") {
    ComputeGraphPtr compute_graph = make_shared<ComputeGraph>(name);
    OpDescPtr op_desc = std::make_shared<OpDesc>("constant", "Constant");
    op_desc->AddOutputDesc(GeTensorDesc());
    compute_graph->AddNode(op_desc);

    SubGraphInfoPtr subgraph = BuildSubGraph(compute_graph, "ge_local", "");
    AddPlaceHolderAndEnd(subgraph, 0, 1);
    return subgraph;
  }

  SubGraphInfoPtr CreateSubgraphWithNodeName(const string &graph_name, const string &node_name, const string &engine,
                                             const string &stream_label = "", int in_num = 1, int out_num = 1) {
    ComputeGraphPtr compute_graph = make_shared<ComputeGraph>(graph_name);
    OpDescPtr op_desc = std::make_shared<OpDesc>(node_name, "Relu");
    op_desc->AddInputDesc(GeTensorDesc());
    op_desc->AddOutputDesc(GeTensorDesc());
    compute_graph->AddNode(op_desc);

    SubGraphInfoPtr subgraph = BuildSubGraph(compute_graph, engine, stream_label);
    AddPlaceHolderAndEnd(subgraph, in_num, out_num);

    return subgraph;
  }

  SubGraphInfoPtr CreateSubgraphWithName(const string &name, const string &engine, const string &stream_label = "",
                                         int in_num = 1, int out_num = 1) {
    ComputeGraphPtr compute_graph = make_shared<ComputeGraph>(name);
    OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
    op_desc->AddInputDesc(GeTensorDesc());
    op_desc->AddOutputDesc(GeTensorDesc());
    compute_graph->AddNode(op_desc);

    SubGraphInfoPtr subgraph = BuildSubGraph(compute_graph, engine, stream_label);
    AddPlaceHolderAndEnd(subgraph, in_num, out_num);

    return subgraph;
  }

  SubGraphInfoPtr CreateSubgraph(const string &engine, const string &stream_label = "", int in_num = 1,
                                 int out_num = 1) {
    return CreateSubgraphWithName("graph", engine, stream_label, in_num, out_num);
  }

  void LinkSubGraph(SubGraphInfoPtr subgraph1, const string &end_name, SubGraphInfoPtr subgraph2,
                    const string &placeholder_name) {
    NodePtr end_node = subgraph1->GetSubGraph()->FindNode(end_name);
    assert(end_node != nullptr);

    NodePtr placeholder_node = subgraph2->GetSubGraph()->FindNode(placeholder_name);
    assert(placeholder_node != nullptr);

    NodePtr const_node = subgraph1->GetSubGraph()->FindNode("constant");
    if (const_node != nullptr) {
      AttrUtils::SetStr(placeholder_node->GetOpDesc(), "parentOpType", "Constant");
    } else {
      AttrUtils::SetStr(placeholder_node->GetOpDesc(), "parentOpType", "xxx");
    }

    subgraph1->end_to_pld_[end_node] = placeholder_node;
    subgraph2->pld_to_end_[placeholder_node] = end_node;
  }

  int64_t GetStream(SubGraphInfoPtr subgraph) {
    int64_t stream_id = kInvalidStream;
    ComputeGraphPtr compute_graph = subgraph->GetSubGraph();
    for (NodePtr node : compute_graph->GetDirectNode()) {
      if (stream_id == kInvalidStream) {
        stream_id = node->GetOpDesc()->GetStreamId();
      } else {
        assert(stream_id == node->GetOpDesc()->GetStreamId());
      }
    }

    return stream_id;
  }

  bool ExpectStreamEq(SubGraphInfoPtr subgraph, int64_t expect) { return GetStream(subgraph) == expect; }

  bool ExpectStreamNe(SubGraphInfoPtr subgraph, int64_t expect) { return GetStream(subgraph) != expect; }
  Status AssignLogicalStreams(vector<SubGraphInfoPtr> subgraphs, vector<EngineConfPtr> &confs,
                              std::map<std::string, int> &max_parallel_num, ComputeGraphPtr &whole_graph) {
    SchedulerConf scheduler_conf;
    if (confs.empty()) {
      for (const auto &subgraph : subgraphs) {
        EngineConfPtr conf = make_shared<EngineConf>();
        conf->id = subgraph->GetEngineName();
        if (conf->id == "ge_local") {
          conf->skip_assign_stream = true;
          conf->attach = true;
        }
        scheduler_conf.cal_engines[conf->id] = conf;
      }
    } else {
      for (auto &conf : confs) {
        scheduler_conf.cal_engines[conf->id] = conf;
      }
    }

    for (const auto &item : scheduler_conf.cal_engines) {
      EngineConfPtr conf = item.second;
      conf->scheduler_id = "scheduler";
    }

    map<string, SchedulerConf> scheduler_confs;
    scheduler_confs["scheduler"] = scheduler_conf;
    LogicalStreamAllocator allocator(scheduler_confs, max_parallel_num);
    int64_t stream_num = 0;
    return allocator.Assign(whole_graph, subgraphs, stream_num);
  }

  Status AssignLogicalStreams(vector<SubGraphInfoPtr> subgraphs, std::map<std::string, int> &max_parallel_num,
                              vector<EngineConfPtr> &confs) {
    ComputeGraphPtr whole_graph = make_shared<ComputeGraph>("whole_graph");
    return AssignLogicalStreams(subgraphs, confs, max_parallel_num, whole_graph);
  }

  Status AssignLogicalStreams(vector<SubGraphInfoPtr> subgraphs,
                              vector<EngineConfPtr> confs = vector<EngineConfPtr>()) {
    std::map<std::string, int> max_parallel_num;
    return AssignLogicalStreams(subgraphs, max_parallel_num, confs);
  }

  Status AssignLogicalStreams(vector<SubGraphInfoPtr> subgraphs, std::map<std::string, int> &max_parallel_num) {
    vector<EngineConfPtr> confs;
    return AssignLogicalStreams(subgraphs, max_parallel_num, confs);
  }

  /// typical case
  ///         Subgraph3_1  Subgraph3_2
  ///      (GenMask1,cpu) (GenMask2,cpu)
  ///                 |     |
  /// Subgraph1  ->  Subgraph2           ->               Subgraph8  ->  Subgraph10
  /// (GetNext,cpu) (DoMask,core)                     (AllReduce1,hccl) (Apply1,core)
  ///                  |                                     |
  ///             Subgraph4 -> Subgraph5 -> Subgraph6 ->  Subgraph7  ->  Subgraph9
  ///              (cpu)        (core)        (core)  (AllReduce2,hccl) (Apply2,core)
  void TestAll(int parallel_num) {
    auto const1 = CreateConstSubgraph();
    auto const2 = CreateConstSubgraph();
    auto get_next = CreateSubgraphWithName("get_next", "aicpu", "get_next", 0, 1);
    auto genmask1 = CreateSubgraphWithName("genmask1", "aicpu", "", 1, 1);
    auto genmask2 = CreateSubgraphWithName("genmask2", "aicpu", "", 1, 1);
    auto domask = CreateSubgraphWithName("domask", "aicore", "", 3, 2);
    auto subgraph4 = CreateSubgraphWithName("subgraph4", "aicpu", "", 1, 1);
    auto subgraph5 = CreateSubgraphWithName("subgraph5", "aicore", "", 1, 1);
    auto subgraph6 = CreateSubgraphWithName("subgraph6", "aicore", "", 1, 1);
    auto allreduce1 = CreateSubgraphWithName("allreduce1", "hccl", "", 1, 2);
    auto allreduce2 = CreateSubgraphWithName("allreduce2", "hccl", "", 2, 1);
    auto apply1 = CreateSubgraphWithName("apply1", "aicore", "", 1, 1);
    auto apply2 = CreateSubgraphWithName("apply2", "aicore", "", 1, 1);

    LinkSubGraph(const1, "end", genmask1, "placeholder");
    LinkSubGraph(const2, "end", genmask2, "placeholder");
    LinkSubGraph(get_next, "end", domask, "placeholder1");
    LinkSubGraph(genmask1, "end", domask, "placeholder2");
    LinkSubGraph(genmask2, "end", domask, "placeholder3");
    LinkSubGraph(domask, "end1", subgraph4, "placeholder");
    LinkSubGraph(domask, "end2", allreduce1, "placeholder");
    LinkSubGraph(subgraph4, "end", subgraph5, "placeholder");
    LinkSubGraph(subgraph5, "end", subgraph6, "placeholder");
    LinkSubGraph(subgraph6, "end", allreduce2, "placeholder1");
    LinkSubGraph(allreduce1, "end1", allreduce2, "placeholder2");
    LinkSubGraph(allreduce1, "end2", apply1, "placeholder");
    LinkSubGraph(allreduce2, "end", apply2, "placeholder");

    EngineConfPtr conf1 = make_shared<EngineConf>();
    conf1->id = "ge_local";
    conf1->skip_assign_stream = true;
    conf1->attach = true;
    EngineConfPtr conf2 = make_shared<EngineConf>();
    conf2->id = "aicore";
    EngineConfPtr conf3 = make_shared<EngineConf>();
    conf3->id = "aicpu";
    conf3->attach = true;
    EngineConfPtr conf4 = make_shared<EngineConf>();
    conf4->id = "hccl";
    conf4->independent = true;
    vector<EngineConfPtr> confs = {conf1, conf2, conf3, conf4};

    std::map<std::string, int> max_parallel_num;
    max_parallel_num["aicore"] = parallel_num;
    max_parallel_num["aicpu"] = parallel_num;

    Status status = AssignLogicalStreams({const1, const2, get_next, genmask1, genmask2, domask, subgraph4, subgraph5,
                                          subgraph6, allreduce1, allreduce2, apply1, apply2},
                                         max_parallel_num, confs);
    EXPECT_EQ(status, ge::SUCCESS);

    EXPECT_EQ(GetStream(get_next), 0);
    EXPECT_EQ(GetStream(allreduce1), 1);
    EXPECT_EQ(GetStream(allreduce2), 1);

    EXPECT_EQ(GetStream(subgraph4), GetStream(subgraph5));
    EXPECT_EQ(GetStream(subgraph5), GetStream(subgraph6));

    EXPECT_NE(GetStream(get_next), GetStream(subgraph4));
    EXPECT_NE(GetStream(genmask2), GetStream(subgraph4));

    if (parallel_num == 1) {
      EXPECT_EQ(GetStream(apply1), GetStream(apply2));
    } else {
      EXPECT_NE(GetStream(apply1), GetStream(apply2));
    }
  }

  /// Set one graph:
  ///  stream id:       1     1                1
  ///                   B --> C(AllReduce) --- D
  ///                  /
  ///  stream id:  0  A
  ///                  \
  ///                   E --> F(AllReduce) --- G
  ///  stream id:       2     2                2
  ///
  void make_graph_with_allreduce(ge::ComputeGraphPtr graph) {
    ge::OpDescPtr op_a = make_shared<ge::OpDesc>("A", DATA);
    auto desc_temp_ptr = make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    op_a->AddInputDesc(desc_temp);
    op_a->AddOutputDesc(desc_temp);

    ge::OpDescPtr op_b = make_shared<ge::OpDesc>("B", "testa");
    op_b->AddInputDesc(desc_temp);
    op_b->AddOutputDesc(desc_temp);

    ge::OpDescPtr op_c = make_shared<ge::OpDesc>("C", "HcomAllReduce");
    op_c->AddInputDesc(desc_temp);
    op_c->AddOutputDesc(desc_temp);

    ge::OpDescPtr op_d = make_shared<ge::OpDesc>("D", "testa");
    op_d->AddInputDesc(desc_temp);
    op_d->AddOutputDesc(desc_temp);

    ge::OpDescPtr op_e = make_shared<ge::OpDesc>("E", "testa");
    op_e->AddInputDesc(desc_temp);
    op_e->AddOutputDesc(desc_temp);

    ge::OpDescPtr op_f = make_shared<ge::OpDesc>("F", "HcomAllReduce");
    op_f->AddInputDesc(desc_temp);
    op_f->AddOutputDesc(desc_temp);

    ge::OpDescPtr op_g = make_shared<ge::OpDesc>("G", "testa");
    op_g->AddInputDesc(desc_temp);
    op_g->AddOutputDesc(desc_temp);

    // add node
    ge::NodePtr node_a = graph->AddNode(op_a);
    ge::NodePtr node_b = graph->AddNode(op_b);
    ge::NodePtr node_c = graph->AddNode(op_c);
    ge::NodePtr node_d = graph->AddNode(op_d);
    ge::NodePtr node_e = graph->AddNode(op_e);
    ge::NodePtr node_f = graph->AddNode(op_f);
    ge::NodePtr node_g = graph->AddNode(op_g);

    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_f->GetOutDataAnchor(0), node_g->GetInDataAnchor(0));

    // add stream id
    node_a->GetOpDesc()->SetStreamId(0);
    node_b->GetOpDesc()->SetStreamId(1);
    node_c->GetOpDesc()->SetStreamId(1);
    node_d->GetOpDesc()->SetStreamId(1);
    node_e->GetOpDesc()->SetStreamId(2);
    node_f->GetOpDesc()->SetStreamId(2);
    node_g->GetOpDesc()->SetStreamId(2);
  }
};

// case of single subgraph (without streamlabel)
TEST_F(UtestLogicalStreamAllocator, test_single_subgraph) {
  SubGraphInfoPtr subgraph = CreateSubgraph("engine1", "");
  Status status = AssignLogicalStreams({subgraph});
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_EQ(GetStream(subgraph), 0);
}

// case of single subgraph (with streamlabel)
TEST_F(UtestLogicalStreamAllocator, test_single_subgraph_with_label) {
  SubGraphInfoPtr subgraph = CreateSubgraph("engine1", "label1");
  Status status = AssignLogicalStreams({subgraph});
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_EQ(GetStream(subgraph), 0);
}

// if the subgraphs are with same engine, then reues stream
TEST_F(UtestLogicalStreamAllocator, test_same_engine) {
  SubGraphInfoPtr data = CreateDataSubgraph();
  SubGraphInfoPtr subgraph1 = CreateSubgraphWithName("graph1", "engine1", "");
  SubGraphInfoPtr subgraph2 = CreateSubgraphWithName("graph2", "engine1", "");
  SubGraphInfoPtr subgraph3 = CreateSubgraphWithName("graph3", "engine1", "");
  SubGraphInfoPtr subgraph4 = CreateSubgraphWithName("graph4", "engine1", "");
  LinkSubGraph(data, "end", subgraph1, "placeholder");
  LinkSubGraph(subgraph1, "end", subgraph2, "placeholder");
  LinkSubGraph(subgraph2, "end", subgraph3, "placeholder");
  LinkSubGraph(subgraph3, "end", subgraph4, "placeholder");

  std::map<std::string, int> max_parallel_num;
  max_parallel_num["engine1"] = 100;

  Status status = AssignLogicalStreams({subgraph1, subgraph2, subgraph3, subgraph4}, max_parallel_num);
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_EQ(GetStream(subgraph1), 0);
  EXPECT_EQ(GetStream(subgraph2), 0);
  EXPECT_EQ(GetStream(subgraph3), 0);
  EXPECT_EQ(GetStream(subgraph4), 0);
}

//  if the subgraphs are with different engine and different control uint, then unreues stream
TEST_F(UtestLogicalStreamAllocator, test_diff_engine) {
  SubGraphInfoPtr data = CreateDataSubgraph();
  SubGraphInfoPtr subgraph1 = CreateSubgraph("engine1", "");
  SubGraphInfoPtr subgraph2 = CreateSubgraph("engine2", "");
  SubGraphInfoPtr subgraph3 = CreateSubgraph("engine3", "");
  LinkSubGraph(data, "end", subgraph1, "placeholder");
  LinkSubGraph(subgraph1, "end", subgraph2, "placeholder");
  LinkSubGraph(subgraph2, "end", subgraph3, "placeholder");

  Status status = AssignLogicalStreams({subgraph1, subgraph2, subgraph3});
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_EQ(GetStream(subgraph1), 0);
  EXPECT_EQ(GetStream(subgraph2), 1);
  EXPECT_EQ(GetStream(subgraph3), 2);
}

//  if the subgraphs are with different engine and same control uint, then reues stream
TEST_F(UtestLogicalStreamAllocator, test_engine_attach) {
  SubGraphInfoPtr data = CreateDataSubgraph();
  SubGraphInfoPtr subgraph1 = CreateSubgraphWithName("graph1", "engine1", "");
  SubGraphInfoPtr subgraph2 = CreateSubgraphWithName("graph2", "engine2", "");
  SubGraphInfoPtr subgraph3 = CreateSubgraphWithName("graph3", "engine3", "");
  SubGraphInfoPtr subgraph4 = CreateSubgraphWithName("graph4", "engine4", "");
  LinkSubGraph(data, "end", subgraph1, "placeholder");
  LinkSubGraph(subgraph1, "end", subgraph2, "placeholder");
  LinkSubGraph(subgraph2, "end", subgraph3, "placeholder");
  LinkSubGraph(subgraph3, "end", subgraph4, "placeholder");

  EngineConfPtr conf1 = make_shared<EngineConf>();
  conf1->id = subgraph1->GetEngineName();
  EngineConfPtr conf2 = make_shared<EngineConf>();
  conf2->id = subgraph2->GetEngineName();
  conf2->attach = true;
  EngineConfPtr conf3 = make_shared<EngineConf>();
  conf3->id = subgraph3->GetEngineName();
  conf3->attach = true;
  EngineConfPtr conf4 = make_shared<EngineConf>();
  conf4->id = subgraph4->GetEngineName();

  Status status = AssignLogicalStreams({subgraph1, subgraph2, subgraph3, subgraph4}, {conf1, conf2, conf3, conf4});
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_EQ(GetStream(subgraph1), 0);
  EXPECT_EQ(GetStream(subgraph2), 0);
  EXPECT_EQ(GetStream(subgraph3), 0);
  EXPECT_EQ(GetStream(subgraph4), 1);
}

// if the param of engine skip_assign_stream is true, unset stream, stream id is 0
TEST_F(UtestLogicalStreamAllocator, test_skip_assign_stream) {
  SubGraphInfoPtr data = CreateDataSubgraph();
  SubGraphInfoPtr subgraph1 = CreateSubgraph("engine1", "");
  SubGraphInfoPtr subgraph2 = CreateSubgraph("engine2", "");
  SubGraphInfoPtr subgraph3 = CreateSubgraph("engine3", "");
  SubGraphInfoPtr subgraph4 = CreateSubgraph("engine4", "");
  LinkSubGraph(data, "end", subgraph1, "placeholder");
  LinkSubGraph(subgraph1, "end", subgraph2, "placeholder");
  LinkSubGraph(subgraph2, "end", subgraph3, "placeholder");
  LinkSubGraph(subgraph3, "end", subgraph4, "placeholder");

  EngineConfPtr conf1 = make_shared<EngineConf>();
  conf1->id = subgraph1->GetEngineName();
  EngineConfPtr conf2 = make_shared<EngineConf>();
  conf2->id = subgraph2->GetEngineName();
  EngineConfPtr conf3 = make_shared<EngineConf>();
  conf3->id = subgraph3->GetEngineName();
  conf3->skip_assign_stream = true;
  conf3->attach = true;
  EngineConfPtr conf4 = make_shared<EngineConf>();
  conf4->id = subgraph4->GetEngineName();

  Status status = AssignLogicalStreams({subgraph1, subgraph2, subgraph3, subgraph4}, {conf1, conf2, conf3, conf4});
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_EQ(GetStream(subgraph1), 0);
  EXPECT_EQ(GetStream(subgraph2), 1);
  EXPECT_EQ(GetStream(subgraph4), 2);
}

// if stream id of same label is different, then different label with different stream id
TEST_F(UtestLogicalStreamAllocator, test_stream_label) {
  SubGraphInfoPtr data = CreateDataSubgraph();
  SubGraphInfoPtr subgraph1 = CreateSubgraph("engine1", "label1");
  SubGraphInfoPtr subgraph2 = CreateSubgraph("engine2", "label2");
  SubGraphInfoPtr subgraph3 = CreateSubgraph("engine3", "label1");
  SubGraphInfoPtr subgraph4 = CreateSubgraph("engine4", "label2");
  SubGraphInfoPtr subgraph5 = CreateSubgraph("engine5", "label2");
  LinkSubGraph(data, "end", subgraph1, "placeholder");
  LinkSubGraph(subgraph1, "end", subgraph2, "placeholder");
  LinkSubGraph(subgraph2, "end", subgraph3, "placeholder");
  LinkSubGraph(subgraph3, "end", subgraph4, "placeholder");
  LinkSubGraph(subgraph4, "end", subgraph5, "placeholder");

  Status status = AssignLogicalStreams({subgraph1, subgraph2, subgraph3, subgraph4, subgraph5});
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_EQ(GetStream(subgraph1), 0);
  EXPECT_EQ(GetStream(subgraph2), 1);
  EXPECT_EQ(GetStream(subgraph3), 0);
  EXPECT_EQ(GetStream(subgraph4), 1);
  EXPECT_EQ(GetStream(subgraph5), 1);
}

TEST_F(UtestLogicalStreamAllocator, test_label_not_reusable) {
  SubGraphInfoPtr data = CreateDataSubgraph();
  SubGraphInfoPtr subgraph1 = CreateSubgraphWithName("graph1", "engine1", "label1");
  SubGraphInfoPtr subgraph2 = CreateSubgraphWithName("graph2", "engine1", "label1");
  SubGraphInfoPtr subgraph3 = CreateSubgraphWithName("graph3", "engine1", "");
  SubGraphInfoPtr subgraph4 = CreateSubgraphWithName("graph4", "engine1", "");
  LinkSubGraph(data, "end", subgraph1, "placeholder");
  LinkSubGraph(subgraph1, "end", subgraph2, "placeholder");
  LinkSubGraph(subgraph2, "end", subgraph3, "placeholder");
  LinkSubGraph(subgraph3, "end", subgraph4, "placeholder");

  Status status = AssignLogicalStreams({subgraph1, subgraph2, subgraph3, subgraph4});
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_EQ(GetStream(subgraph1), 0);
  EXPECT_EQ(GetStream(subgraph2), 0);
  EXPECT_EQ(GetStream(subgraph3), 1);
  EXPECT_EQ(GetStream(subgraph4), 1);
}
///    data
///   |   |
/// sub1 sub2
///   \   /
///    sub3
TEST_F(UtestLogicalStreamAllocator, test_label_not_reusable2) {
  SubGraphInfoPtr data = CreateDataSubgraph();
  SubGraphInfoPtr subgraph1 = CreateSubgraphWithName("graph1", "engine1", "label1");
  SubGraphInfoPtr subgraph2 = CreateSubgraphWithName("graph2", "engine1", "label2");
  SubGraphInfoPtr subgraph3 = CreateSubgraphWithName("graph3", "engine2", "", 2, 1);
  LinkSubGraph(data, "end", subgraph1, "placeholder");
  LinkSubGraph(data, "end", subgraph2, "placeholder");
  LinkSubGraph(subgraph1, "end", subgraph3, "placeholder1");
  LinkSubGraph(subgraph2, "end", subgraph3, "placeholder2");

  EngineConfPtr conf1 = make_shared<EngineConf>();
  conf1->id = subgraph1->GetEngineName();
  EngineConfPtr conf2 = make_shared<EngineConf>();
  conf2->id = subgraph3->GetEngineName();
  conf2->attach = true;
  Status status = AssignLogicalStreams({subgraph1, subgraph2, subgraph3}, {conf1, conf2});
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_EQ(GetStream(subgraph1), 0);
  EXPECT_EQ(GetStream(subgraph2), 1);
  EXPECT_EQ(GetStream(subgraph3), 2);
}

/// case of multi-output, then unuse stream
///      sub1
///    /  |   \
/// sub2 sub3 sub4
TEST_F(UtestLogicalStreamAllocator, test_multiOut_new_stream) {
  SubGraphInfoPtr data = CreateDataSubgraph();
  SubGraphInfoPtr subgraph1 = CreateSubgraph("engine1", "", 1, 3);
  SubGraphInfoPtr subgraph2 = CreateSubgraph("engine1");
  SubGraphInfoPtr subgraph3 = CreateSubgraph("engine1");
  SubGraphInfoPtr subgraph4 = CreateSubgraph("engine1");
  LinkSubGraph(data, "end", subgraph1, "placeholder");
  LinkSubGraph(subgraph1, "end1", subgraph2, "placeholder");
  LinkSubGraph(subgraph1, "end2", subgraph3, "placeholder");
  LinkSubGraph(subgraph1, "end3", subgraph4, "placeholder");

  std::map<std::string, int> max_parallel_num;
  max_parallel_num["engine1"] = 100;
  Status status = AssignLogicalStreams({subgraph1, subgraph2, subgraph3, subgraph4}, max_parallel_num);
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_EQ(GetStream(subgraph1), 0);
  EXPECT_EQ(GetStream(subgraph2), 1);
  EXPECT_EQ(GetStream(subgraph3), 2);
  EXPECT_EQ(GetStream(subgraph4), 3);
}

/// if paralle id 1, then use stream
///        sub1
///    /   |   |   \
/// sub2 sub3 sub4 sub5
TEST_F(UtestLogicalStreamAllocator, test_parallel_one) {
  SubGraphInfoPtr data = CreateDataSubgraph();
  SubGraphInfoPtr subgraph1 = CreateSubgraph("engine1", "", 1, 4);
  SubGraphInfoPtr subgraph2 = CreateSubgraph("engine1");
  SubGraphInfoPtr subgraph3 = CreateSubgraph("engine2");
  SubGraphInfoPtr subgraph4 = CreateSubgraph("engine1");
  SubGraphInfoPtr subgraph5 = CreateSubgraph("engine2");
  LinkSubGraph(data, "end", subgraph1, "placeholder");
  LinkSubGraph(subgraph1, "end1", subgraph2, "placeholder");
  LinkSubGraph(subgraph1, "end2", subgraph3, "placeholder");
  LinkSubGraph(subgraph1, "end3", subgraph4, "placeholder");
  LinkSubGraph(subgraph1, "end4", subgraph5, "placeholder");

  std::map<std::string, int> max_parallel_num;
  max_parallel_num["engine1"] = 1;
  max_parallel_num["engine2"] = 1;
  Status status = AssignLogicalStreams({subgraph1, subgraph2, subgraph3, subgraph4, subgraph5}, max_parallel_num);
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_EQ(GetStream(subgraph1), 0);
  EXPECT_EQ(GetStream(subgraph2), 0);
  EXPECT_EQ(GetStream(subgraph3), 1);
  EXPECT_EQ(GetStream(subgraph4), 0);
  EXPECT_EQ(GetStream(subgraph5), 1);
}

/// if the param of engine independent is true, then set independent stream
///        sub1
///    /   |   |   \
/// sub2 sub3 sub4 sub5
TEST_F(UtestLogicalStreamAllocator, test_independent) {
  SubGraphInfoPtr data = CreateDataSubgraph();
  SubGraphInfoPtr subgraph1 = CreateSubgraph("engine1", "", 1, 4);
  SubGraphInfoPtr subgraph2 = CreateSubgraph("engine1");
  SubGraphInfoPtr subgraph3 = CreateSubgraph("engine2");
  SubGraphInfoPtr subgraph4 = CreateSubgraph("engine1");
  SubGraphInfoPtr subgraph5 = CreateSubgraph("engine2");
  LinkSubGraph(data, "end", subgraph1, "placeholder");
  LinkSubGraph(subgraph1, "end1", subgraph2, "placeholder");
  LinkSubGraph(subgraph1, "end2", subgraph3, "placeholder");
  LinkSubGraph(subgraph1, "end3", subgraph4, "placeholder");
  LinkSubGraph(subgraph1, "end4", subgraph5, "placeholder");

  std::map<std::string, int> max_parallel_num;
  max_parallel_num["engine1"] = 100;
  max_parallel_num["engine2"] = 100;

  EngineConfPtr conf1 = make_shared<EngineConf>();
  conf1->id = "engine1";
  conf1->independent = true;
  EngineConfPtr conf2 = make_shared<EngineConf>();
  conf2->id = "engine2";
  conf2->independent = true;
  vector<EngineConfPtr> confs = {conf1, conf2};

  Status status =
      AssignLogicalStreams({subgraph1, subgraph2, subgraph3, subgraph4, subgraph5}, max_parallel_num, confs);
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_EQ(GetStream(subgraph1), 0);
  EXPECT_EQ(GetStream(subgraph2), 0);
  EXPECT_EQ(GetStream(subgraph3), 1);
  EXPECT_EQ(GetStream(subgraph4), 0);
  EXPECT_EQ(GetStream(subgraph5), 1);
}

/// set stream based on stream label, and then based on independent
///        sub1
///    /   |   |   \
/// sub2 sub3 sub4 sub5
TEST_F(UtestLogicalStreamAllocator, test_independent_switch_label) {
  SubGraphInfoPtr data = CreateDataSubgraph();
  SubGraphInfoPtr subgraph1 = CreateSubgraphWithName("sub1", "engine0", "", 1, 4);
  SubGraphInfoPtr subgraph2 = CreateSubgraphWithName("sub2", "engine1", "label1");
  SubGraphInfoPtr subgraph3 = CreateSubgraphWithName("sub3", "engine2", "label1");
  SubGraphInfoPtr subgraph4 = CreateSubgraphWithName("sub4", "engine1", "label2");
  SubGraphInfoPtr subgraph5 = CreateSubgraphWithName("sub5", "engine2", "label2");
  LinkSubGraph(data, "end", subgraph1, "placeholder");
  LinkSubGraph(subgraph1, "end1", subgraph2, "placeholder");
  LinkSubGraph(subgraph1, "end2", subgraph3, "placeholder");
  LinkSubGraph(subgraph1, "end3", subgraph4, "placeholder");
  LinkSubGraph(subgraph1, "end4", subgraph5, "placeholder");

  std::map<std::string, int> max_parallel_num;
  max_parallel_num["engine0"] = 1;
  max_parallel_num["engine1"] = 100;
  max_parallel_num["engine2"] = 100;

  EngineConfPtr conf1 = make_shared<EngineConf>();
  conf1->id = "engine0";
  conf1->independent = false;
  EngineConfPtr conf2 = make_shared<EngineConf>();
  conf2->id = "engine1";
  conf2->independent = false;
  EngineConfPtr conf3 = make_shared<EngineConf>();
  conf3->id = "engine2";
  conf3->independent = true;
  vector<EngineConfPtr> confs = {conf1, conf2, conf3};

  Status status =
      AssignLogicalStreams({subgraph1, subgraph2, subgraph3, subgraph4, subgraph5}, max_parallel_num, confs);
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_EQ(GetStream(subgraph1), 4);
  EXPECT_EQ(GetStream(subgraph2), 0);
  EXPECT_EQ(GetStream(subgraph3), 2);
  EXPECT_EQ(GetStream(subgraph4), 1);
  EXPECT_EQ(GetStream(subgraph5), 3);
}

/// subgraph without input of locate stream
/// data genmask1
///   |    /
/// domask1 genmask2
///   |    /
/// domask2 genmask3
///   |    /
/// domask3
TEST_F(UtestLogicalStreamAllocator, test_no_input) {
  auto data = CreateDataSubgraph();
  auto genmask1 = CreateSubgraphWithName("genmask1", "engine1", "", 0, 1);
  auto domask1 = CreateSubgraphWithName("domask1", "engine1", "", 2, 1);
  auto genmask2 = CreateSubgraphWithName("genmask2", "engine1", "", 0, 1);
  auto domask2 = CreateSubgraphWithName("domask2", "engine1", "", 2, 1);
  auto genmask3 = CreateSubgraphWithName("genmask3", "engine1", "", 0, 1);
  auto domask3 = CreateSubgraphWithName("domask3", "engine1", "", 2, 1);

  LinkSubGraph(data, "end", domask1, "placeholder1");
  LinkSubGraph(genmask1, "end", domask1, "placeholder2");
  LinkSubGraph(domask1, "end", domask2, "placeholder1");
  LinkSubGraph(genmask2, "end", domask2, "placeholder2");
  LinkSubGraph(domask2, "end", domask3, "placeholder1");
  LinkSubGraph(genmask3, "end", domask3, "placeholder2");

  Status status = AssignLogicalStreams({data, genmask1, domask1, genmask2, domask2, genmask3, domask3});
  EXPECT_EQ(status, ge::SUCCESS);

  EXPECT_EQ(GetStream(genmask1), 0);
  EXPECT_EQ(GetStream(genmask2), 0);
  EXPECT_EQ(GetStream(genmask3), 0);
  EXPECT_EQ(GetStream(domask1), 0);
  EXPECT_EQ(GetStream(domask2), 0);
  EXPECT_EQ(GetStream(domask3), 0);
}

/// subgraph with const input locate stream
/// data genmask1 - const1
///   |    /
/// domask1 genmask2 - const2
///   |    /
/// domask2 genmask3 - const3
///   |    /
/// domask3
TEST_F(UtestLogicalStreamAllocator, test_const_input) {
  auto data = CreateDataSubgraph();
  auto const1 = CreateConstSubgraph();
  auto const2 = CreateConstSubgraph();
  auto const3 = CreateConstSubgraph();
  auto genmask1 = CreateSubgraphWithName("genmask1", "engine1", "", 1, 1);
  auto domask1 = CreateSubgraphWithName("domask1", "engine1", "", 2, 1);
  auto genmask2 = CreateSubgraphWithName("genmask2", "engine1", "", 1, 1);
  auto domask2 = CreateSubgraphWithName("domask2", "engine1", "", 2, 1);
  auto genmask3 = CreateSubgraphWithName("genmask3", "engine1", "", 1, 1);
  auto domask3 = CreateSubgraphWithName("domask3", "engine1", "", 2, 1);

  LinkSubGraph(const1, "end", genmask1, "placeholder");
  LinkSubGraph(const2, "end", genmask2, "placeholder");
  LinkSubGraph(const3, "end", genmask3, "placeholder");
  LinkSubGraph(data, "end", domask1, "placeholder1");
  LinkSubGraph(genmask1, "end", domask1, "placeholder2");
  LinkSubGraph(domask1, "end", domask2, "placeholder1");
  LinkSubGraph(genmask2, "end", domask2, "placeholder2");
  LinkSubGraph(domask2, "end", domask3, "placeholder1");
  LinkSubGraph(genmask3, "end", domask3, "placeholder2");

  Status status =
      AssignLogicalStreams({data, const1, const2, const3, genmask1, domask1, genmask2, domask2, genmask3, domask3});
  EXPECT_EQ(status, ge::SUCCESS);

  EXPECT_EQ(GetStream(genmask1), 0);
  EXPECT_EQ(GetStream(genmask2), 0);
  EXPECT_EQ(GetStream(genmask3), 0);
  EXPECT_EQ(GetStream(domask1), 0);
  EXPECT_EQ(GetStream(domask2), 0);
  EXPECT_EQ(GetStream(domask3), 0);
}

TEST_F(UtestLogicalStreamAllocator, TestAllParallelNum1) { TestAll(1); }

TEST_F(UtestLogicalStreamAllocator, TestAllParallelNum2) { TestAll(2); }

TEST_F(UtestLogicalStreamAllocator, TestReusableSubgraphNotAssignedStream) {
  SubGraphInfoPtr data = CreateDataSubgraph();
  SubGraphInfoPtr subgraph1 = CreateSubgraphWithName("graph1", "engine1", "");
  SubGraphInfoPtr subgraph2 = CreateSubgraphWithName("graph2", "engine1", "");
  LinkSubGraph(data, "end", subgraph1, "placeholder");
  LinkSubGraph(subgraph1, "end", subgraph2, "placeholder");

  Status status = AssignLogicalStreams({data, subgraph2, subgraph1});
  EXPECT_EQ(status, ge::SUCCESS);
}
/// Optimize for case like:
///  NodeA(stream1) -> Const(stream2) -> NodeB(stream1)
/// To case:
///  NodeA(stream1) -> Const(stream1) -> NodeB(stream1)
/// Which could reduce event number (Const could be other type which belong to skipped engine subgraph)

/// data
///   |
/// subgraph1(label)
///   |
/// const2
///   |
/// subgrah3(label)
TEST_F(UtestLogicalStreamAllocator, test_reassign_stream) {
  SubGraphInfoPtr data = CreateDataSubgraph();
  SubGraphInfoPtr subgraph1 = CreateSubgraphWithNodeName("subgraph1", "relu1", "engine1", "label");
  SubGraphInfoPtr const2 = CreateSubgraphWithNodeName("const2", "const2", "ge_local");
  SubGraphInfoPtr subgraph3 = CreateSubgraphWithNodeName("subgrah3", "relu3", "engine1", "label");

  LinkSubGraph(data, "end", subgraph1, "placeholder");
  LinkSubGraph(subgraph1, "end", const2, "placeholder");
  LinkSubGraph(const2, "end", subgraph3, "placeholder");

  EngineConfPtr conf1 = make_shared<EngineConf>();
  conf1->id = subgraph1->GetEngineName();
  EngineConfPtr conf2 = make_shared<EngineConf>();
  conf2->id = const2->GetEngineName();
  conf2->skip_assign_stream = true;
  EngineConfPtr conf3 = make_shared<EngineConf>();
  conf3->id = subgraph3->GetEngineName();

  auto node1 = subgraph1->GetSubGraph()->FindNode("relu1");
  auto node2 = const2->GetSubGraph()->FindNode("const2");
  auto node3 = subgraph3->GetSubGraph()->FindNode("relu3");
  ComputeGraphPtr whole_graph = make_shared<ComputeGraph>("whole_graph");
  auto node1_1 = whole_graph->AddNode(node1->GetOpDesc());
  auto node1_2 = whole_graph->AddNode(node2->GetOpDesc());
  auto node1_3 = whole_graph->AddNode(node3->GetOpDesc());
  GraphUtils::AddEdge(node1_1->GetOutControlAnchor(), node1_2->GetInControlAnchor());
  GraphUtils::AddEdge(node1_2->GetOutDataAnchor(0), node1_3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node1->GetOutControlAnchor(), node2->GetInControlAnchor());

  std::map<std::string, int> max_parallel_num;
  vector<SubGraphInfoPtr> subgraphs = {subgraph1, const2, subgraph3};
  vector<EngineConfPtr> confs = {conf1, conf2, conf3};
  Status status = AssignLogicalStreams(subgraphs, confs, max_parallel_num, whole_graph);

  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_EQ(GetStream(subgraph1), 0);
  EXPECT_EQ(node2->GetOpDesc()->GetStreamId(), 0);
  EXPECT_EQ(GetStream(subgraph3), 0);
}

TEST_F(UtestLogicalStreamAllocator, test_all_reduce_parallel_pass) {
  graphStatus ret = GRAPH_SUCCESS;

  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");
  graph->SetName("TestAllReduceParallelPass");
  make_graph_with_allreduce(graph);

  std::map<std::string, int> max_parallel_num;
  LogicalStreamPass::Context context;
  context.next_stream = 5;
  context.hcom_parallel = true;
  vector<LogicalStreamPass::SubgraphPtr> subgraphs;
  LogicalStreamPassPtr allreduce_pass = std::make_shared<AllReduceParallelPass>();
  ret = allreduce_pass->Run(graph, subgraphs, context);

  EXPECT_EQ(ret, NOT_CHANGED);
}

}  // namespace ge
