/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <map>
#include "external/ge/ge_api.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/types.h"
#include "framework.h"
#include "framework/utils/builder/graph_builder_utils.h"
#include "graph/operator_reg.h"
#include "graph/operator.h"
#define protected public
#define private public
#include "graph/utils/op_desc_utils.h"
#undef protected
#undef private

using namespace std;
using namespace ge;
namespace {
/** data a = 2;
*  for(int i =0; i<5; ++i){
*    a=a * 2;
* }
*  return a;
*                     ----------------------------------------------|
*                    |  const(5)             exit  const(1)         |
*                    |     \                  /       \             |
*   data(i)--Enter--merge--less--loopcond--switch-----add-----nextiteration
*                       \________________\___/
*                                   ------\------------------------|
*                                  |       \        const(2)       |
*                                  |        \         \            |
*                 data(a)--Enter--merge--switch------mul-----nextiteration
*                                            \
*                                           exit
*                                             \
*                                           netoutput
*
**/
Graph BuildV1ControlFlowGraph(){
  // build graph
  st::ComputeGraphBuilder graphBuilder("g1");
  auto data_i = graphBuilder.AddNode("data_i",DATA,1,1);
  auto enter_i = graphBuilder.AddNode("enter_i",ENTER,1,1);
  ge::AttrUtils::SetStr(enter_i->GetOpDesc(), ENTER_ATTR_FRAME_NAME, "1");
  auto merge_i = graphBuilder.AddNode("merge_i",MERGE,2,1);
  auto const_5 = graphBuilder.AddNode("const_5",CONSTANT,0,1);
  auto less = graphBuilder.AddNode("less",LESS,2,1);
  auto loopcond = graphBuilder.AddNode("loopcond",LOOPCOND,1,1,FORMAT_NCHW,DT_BOOL);
  auto switch_i = graphBuilder.AddNode("switch_i",SWITCH,2,2);
  auto exit_i =  graphBuilder.AddNode("switch_i",EXIT,1,1);
  auto const_1 =  graphBuilder.AddNode("const_1",CONSTANT,0,1);
  auto add =  graphBuilder.AddNode("add",ADD,2,1);
  auto next_iteration_i =  graphBuilder.AddNode("next_iteration_i",NEXTITERATION,1,1);

  auto data_a = graphBuilder.AddNode("data_a",DATA,1,1);
  auto enter_a = graphBuilder.AddNode("enter_a",ENTER,1,1);
  ge::AttrUtils::SetStr(enter_a->GetOpDesc(), ENTER_ATTR_FRAME_NAME, "1");
  auto merge_a = graphBuilder.AddNode("merge_a",MERGE,2,1);
  auto switch_a = graphBuilder.AddNode("switch_a",SWITCH,2,2);
  auto exit_a =  graphBuilder.AddNode("exit_a",EXIT,1,1);
  auto mul =  graphBuilder.AddNode("mul",MUL,2,1);
  auto const_2 =  graphBuilder.AddNode("const_2",CONSTANT,0,1);
  auto next_iteration_a =  graphBuilder.AddNode("next_iteration_a",NEXTITERATION,1,1);
  auto netoutput =  graphBuilder.AddNode("netoutput",NETOUTPUT,2,2);
  // i = i+1
  graphBuilder.AddDataEdge(data_i, 0, enter_i,0);
  graphBuilder.AddDataEdge(enter_i, 0, merge_i,0);
  graphBuilder.AddDataEdge(next_iteration_i, 0, merge_i,1);
  graphBuilder.AddDataEdge(merge_i, 0, less,0);
  graphBuilder.AddDataEdge(const_5, 0, less,1);
  graphBuilder.AddDataEdge(less, 0, loopcond,0);
  graphBuilder.AddDataEdge(loopcond, 0, switch_i,1);
  graphBuilder.AddDataEdge(merge_i, 0, switch_i,0);
  graphBuilder.AddDataEdge(switch_i, 0, exit_i,0);
  graphBuilder.AddDataEdge(switch_i, 1, add,0);
  graphBuilder.AddDataEdge(const_1, 0, add,1);
  graphBuilder.AddDataEdge(add, 0, next_iteration_i,0);
  graphBuilder.AddDataEdge(exit_i, 0, netoutput,1);
  // a=a*2
  graphBuilder.AddDataEdge(data_a, 0, enter_a,0);
  graphBuilder.AddDataEdge(enter_a, 0, merge_a,0);
  graphBuilder.AddDataEdge(next_iteration_a, 0, merge_a,1);
  graphBuilder.AddDataEdge(loopcond, 0, switch_a,1);
  graphBuilder.AddDataEdge(merge_a, 0, switch_a,0);
  graphBuilder.AddDataEdge(switch_a, 0, exit_a,0);
  graphBuilder.AddDataEdge(switch_a, 1, mul,0);
  graphBuilder.AddDataEdge(const_2, 0, mul,1);
  graphBuilder.AddDataEdge(mul, 0, next_iteration_a,0);
  graphBuilder.AddDataEdge(exit_a, 0, netoutput,0);
  // set const weight
  int64_t dims_size = 1;
  vector<int64_t> data_vec = {5};
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<int32_t> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_INT32);
  GeTensorPtr data_tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                  data_value_vec.size() * sizeof(int32_t));
  OpDescUtils::SetWeights(const_5->GetOpDesc(), data_tensor);
  OpDescUtils::SetWeights(const_2->GetOpDesc(), data_tensor);
  OpDescUtils::SetWeights(const_1->GetOpDesc(), data_tensor);

  return graphBuilder.GetGraph();
}
}
class FrameworkTest : public testing::Test {
 protected:
  void SetUp() {
    // ge initialize
    map<AscendString, AscendString> options;
    auto ret = ge::GEInitialize(options);
    EXPECT_EQ(ret, SUCCESS);
  }
  void TearDown() {
  }
};

///     data   data
///       \    /
///        add
TEST_F(FrameworkTest, test_framework_add) {
  // build graph
  st::ComputeGraphBuilder graphBuilder("g1");
  auto data1 = graphBuilder.AddNode("data1",DATA,1,1);
  auto data2 = graphBuilder.AddNode("data2",DATA,1,1);
  auto add = graphBuilder.AddNode("add",ADD,2,1);
  graphBuilder.AddDataEdge(data1, 0, add,0);
  graphBuilder.AddDataEdge(data2, 0, add,1);
  Graph graph = graphBuilder.GetGraph();

  // new session & add graph
  map<AscendString, AscendString> options;
  Session session(options);
  auto ret = session.AddGraph(1, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

/** data a = 2;
 *  for(int i =0; i<5; ++i){
 *    a=a * 2;
 * }
 *  return a;
 *                     ----------------------------------------------|
 *                    |  const(5)             exit  const(1)         |
 *                    |     \                  /       \             |
 *   data(i)--Enter--merge--less--loopcond--switch-----add-----nextiteration
 *                       \________________\___/
 *                                   ------\------------------------|
 *                                  |       \        const(2)       |
 *                                  |        \         \            |
 *                 data(a)--Enter--merge--switch------mul-----nextiteration
 *                                            \
 *                                           exit
 *                                             \
 *                                           netoutput
 *
**/
TEST_F(FrameworkTest, test_framework_v1_control_flow) {
  // build graph
  Graph graph = BuildV1ControlFlowGraph();
  // new session & add graph
  map<AscendString, AscendString> options;
  Session session(options);
  auto ret = session.AddGraph(2, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(2, inputs);
  EXPECT_EQ(ret, SUCCESS);
  // check result
}
