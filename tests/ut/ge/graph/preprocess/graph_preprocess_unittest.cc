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

#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "common/util.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

#define private public
#define protected public
#include "graph/preprocess/graph_preprocess.h"
#include "ge/ge_api.h"
#undef private
#undef protected

using namespace std;
namespace ge {
class UtestGraphPreproces : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

ComputeGraphPtr BuildGraph1(){
  auto builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1",DATA,1,1);
  auto data_opdesc = data1->GetOpDesc();
  AttrUtils::SetInt(data_opdesc, ATTR_NAME_INDEX, 0);
  data1->UpdateOpDesc(data_opdesc);
  return builder.GetGraph();
}

ComputeGraphPtr BuildGraph2() {
  auto builder = ut::GraphBuilder("g2");
  auto data1 = builder.AddNode("data1", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, std::vector<int64_t>({22, -1}));
  ge::AttrUtils::SetStr(data1->GetOpDesc(), ATTR_ATC_USER_DEFINE_DATATYPE, "DT_INT8");
  auto data_opdesc = data1->GetOpDesc();
  AttrUtils::SetInt(data_opdesc, ATTR_NAME_INDEX, 0);

  data1->UpdateOpDesc(data_opdesc);
  return builder.GetGraph();
}

ComputeGraphPtr BuildGraph3() {
  auto builder = ut::GraphBuilder("g3");
  auto data1 = builder.AddNode("data1", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT);
  ge::AttrUtils::SetStr(data1->GetOpDesc(), ATTR_ATC_USER_DEFINE_DATATYPE, "DT_INT8");
  auto data_opdesc = data1->GetOpDesc();
  AttrUtils::SetInt(data_opdesc, ATTR_NAME_INDEX, 0);

  data1->UpdateOpDesc(data_opdesc);
  return builder.GetGraph();
}

/*
 *   MapIndex   Data1          subgraph1        subgraph2
 *         \    /
 *          Case      ===>       Data2            Data3
 *           |
 *       Netoutput
 */
ComputeGraphPtr BuildGraph4() {
  auto builder = ut::GraphBuilder("mbatch_Case");

  auto data1 = builder.AddNode("data1", DATA, 1, 1);
  auto data_desc = data1->GetOpDesc();
  AttrUtils::SetStr(data_desc, ATTR_ATC_USER_DEFINE_DATATYPE, "DT_FLOAT16");
  AttrUtils::SetStr(data_desc, "mbatch-switch-name", "case1");
  AttrUtils::SetInt(data_desc, ATTR_NAME_INDEX, 0);

  auto mapindex1 = builder.AddNode("mapindex1", "MapIndex", 0, 1);
  auto case1 = builder.AddNode("case1", CASE, 2, 1);
  auto netoutput1 = builder.AddNode("netoutput1", NETOUTPUT, 1, 0);

  builder.AddDataEdge(mapindex1, 0, case1, 0);
  builder.AddDataEdge(data1, 0, case1, 1);
  builder.AddDataEdge(case1, 0, netoutput1, 0);

  return builder.GetGraph();
}

ComputeGraphPtr BuildGraph4_Subgraph(string graph_name) {
  auto builder = ut::GraphBuilder(graph_name);
  auto data1 = builder.AddNode(graph_name + "_data1", DATA, 1, 1);
  auto data_desc = data1->GetOpDesc();
  AttrUtils::SetInt(data_desc, ATTR_NAME_PARENT_NODE_INDEX, 1);
  return builder.GetGraph();
}

TEST_F(UtestGraphPreproces, test_dynamic_input_shape_parse) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph1();
  // prepare user_input & graph option
  ge::GeTensorDesc tensor1;
  tensor1.SetFormat(ge::FORMAT_NCHW);
  tensor1.SetShape(ge::GeShape({3, 12, 5, 5}));
  tensor1.SetDataType(ge::DT_FLOAT);
  GeTensor input1(tensor1);
  std::vector<GeTensor> user_input = {input1};
  std::map<string,string> graph_option = {{"ge.exec.dynamicGraphExecuteMode","dynamic_execute"},
                                          {"ge.exec.dataInputsShapeRange","[3,1~20,2~10,5]"}};
  auto ret = graph_prepare.UpdateInput(user_input, graph_option);
  EXPECT_EQ(ret, ge::SUCCESS);
  // check data node output shape_range and shape
  auto data_node = graph_prepare.compute_graph_->FindNode("data1");
  auto data_output_desc = data_node->GetOpDesc()->GetOutputDescPtr(0);
  vector<int64_t> expect_shape = {3,-1,-1,5};
  auto result_shape = data_output_desc->GetShape();
  EXPECT_EQ(result_shape.GetDimNum(), expect_shape.size());
  for(size_t i =0; i< expect_shape.size(); ++i){
      EXPECT_EQ(result_shape.GetDim(i), expect_shape.at(i));
  }
}

TEST_F(UtestGraphPreproces, test_check_user_input) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph1();

  vector<int64_t> dim = {2, -3};
  GeTensor tensor;
  tensor.SetTensorDesc(GeTensorDesc(GeShape(dim)));
  std::vector<GeTensor> user_input;
  user_input.emplace_back(tensor);

  Status ret = graph_prepare.CheckUserInput(user_input);
  EXPECT_EQ(ret, GE_GRAPH_INIT_FAILED);
}

TEST_F(UtestGraphPreproces, test_update_input_output1) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph3();

  Status ret = graph_prepare.UpdateInputOutputByOptions();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPreproces, test_update_dtype_mbatch_case) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph4();
  auto parent_graph = graph_prepare.compute_graph_;
  auto subgraph1 = BuildGraph4_Subgraph("subgraph1");
  auto subgraph2 = BuildGraph4_Subgraph("subgraph2");

  auto data1 = parent_graph->FindNode("data1");
  auto data_desc = data1->GetOpDesc();

  auto case_node = parent_graph->FindNode("case1");
  EXPECT_NE(case_node, nullptr);
  case_node->GetOpDesc()->AddSubgraphName("subgraph1");
  case_node->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph1");
  subgraph1->SetParentNode(case_node);
  subgraph1->SetParentGraph(parent_graph);
  EXPECT_EQ(parent_graph->AddSubgraph("subgraph1", subgraph1), GRAPH_SUCCESS);

  case_node->GetOpDesc()->AddSubgraphName("subgraph2");
  case_node->GetOpDesc()->SetSubgraphInstanceName(1, "subgraph2");
  subgraph2->SetParentNode(case_node);
  subgraph2->SetParentGraph(parent_graph);
  EXPECT_EQ(parent_graph->AddSubgraph("subgraph2", subgraph2), GRAPH_SUCCESS);

  Status ret = graph_prepare.UpdateInputOutputByOptions();
  EXPECT_EQ(ret, SUCCESS);

  auto case_desc = case_node->GetOpDesc();
  auto case_input = case_desc->MutableInputDesc(1);
  EXPECT_EQ(case_input->GetDataType(), 1);

  auto sub1_data1 = subgraph1->FindNode("subgraph1_data1");
  EXPECT_NE(sub1_data1, nullptr);
  auto data1_desc = sub1_data1->GetOpDesc();
  auto data1_input = data1_desc->MutableInputDesc(0);
  EXPECT_EQ(data1_input->GetDataType(), 1);
  auto data1_output = data1_desc->MutableOutputDesc(0);
  EXPECT_EQ(data1_output->GetDataType(), 1);
}
}