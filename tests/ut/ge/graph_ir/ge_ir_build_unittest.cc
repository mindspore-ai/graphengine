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
#include "ir_build/atc_ir_common.h"
#include "graph/testcase/ge_graph/graph_builder_utils.h"

#define protected public
#define private public

#undef private
#undef protected

const string DATA = "Data";
const string AddNYes = "AddNYes";
const string NETOUTPUT = "NetOutput";

using namespace ge;
class UtestIrCommon : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

static ge::OpDescPtr CreateOpDesc(const std::string &name, const std::string &type) {
  OpDescPtr op_desc = std::make_shared<ge::OpDesc>(name, type);
  ge::GeTensorDesc ge_tensor_desc;
  op_desc->AddInputDesc("input", ge_tensor_desc);
  op_desc->AddOutputDesc("output", ge_tensor_desc);

  return op_desc;
}

static ComputeGraphPtr BuildComputeGraph() {
  auto builder = ut::GraphBuilder("test");
  auto data1 = builder.AddNode("input1", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 2, 3});
  auto data2 = builder.AddNode("input2", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {4, 10});
  auto addn1 = builder.AddNode("addn1", AddNYes, 2, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data1, 0, addn1, 0);
  builder.AddDataEdge(data2, 0, addn1, 1);
  builder.AddDataEdge(addn1, 0,netoutput, 0);

  return builder.GetGraph();
}

TEST(UtestIrCommon, update_data_op_shape) {
  ge::OpDescPtr op_desc = CreateOpDesc("Data", "Data");
  map<string, vector<int64_t>> shape_map;
  shape_map["Data"] = {{1,2}};

  Status ret = UpdateDataOpShape(op_desc, shape_map);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST(UtestIrCommon, update_dynamic_shape_range_success) {
  ComputeGraphPtr graph = BuildComputeGraph();
  std::string input_shape_range = "input1:[1, 2~3, -1];input2:[3~5, 10]";

  Status ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST(UtestIrCommon, update_dynamic_shape_range_failed) {
  ComputeGraphPtr graph = BuildComputeGraph();
  // 1
  std::string input_shape_range = "input1;[1, 2~3, -1]";
  Status ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::PARAM_INVALID);

  // 2
  input_shape_range = "input1:[1, 2~3, -1)";
  ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::PARAM_INVALID);

  //3
  input_shape_range = "input1:[1, 3~2, -1];input2:[3~5, 10]";
  ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::FAILED);

  //4
  input_shape_range = "input1:[1, 2~-3, -1]";
  ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}
