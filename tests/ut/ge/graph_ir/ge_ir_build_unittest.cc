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
#include "ir_build/option_utils.h"
#include "graph/testcase/ge_graph/graph_builder_utils.h"
#include "graph/debug/ge_attr_define.h"

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

TEST(UtestIrCommon, update_data_op_shape_range) {
  ge::OpDescPtr op_desc = CreateOpDesc("Data", "Data");
  std::vector<std::vector<std::pair<int64_t, int64_t>>> index_shape_range_map;

  std::pair<int64_t, int64_t> range_pair(1, 2);
  vector<pair<int64_t, int64_t>> range_pair_tmp = { range_pair };

  index_shape_range_map.push_back(range_pair_tmp);

  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  Status ret = UpdateDataOpShapeRange(op_desc, index_shape_range_map);
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

  //5
  input_shape_range = "input:[1, 2~3, -1]";
  ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::PARAM_INVALID);

  //6
  input_shape_range = "addn1:[1, 2~3, -1]";
  ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST(UtestIrCommon, check_dynamic_image_size_fail) {
  map<string, vector<int64_t>> shape_map;
  shape_map["input1"] = {8, 3, -1, -1};
  string input_format = "NCHW";
  string dynamic_image_size = "@64,64;128,128;";

  bool ret = CheckDynamicImagesizeInputShapeValid(shape_map, input_format, dynamic_image_size);
  EXPECT_EQ(ret, false);
}

TEST(UtestIrCommon, check_input_format_failed) {
  std::string format = "invalid";
  Status ret = CheckInputFormat(format);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST(UtestIrCommon, check_dynamic_batch_size_input_shape_succ) {
  map<string, vector<int64_t>> shape_map;
  shape_map.insert(std::pair<string, vector<int64_t>>("data", {-1, 2, 3}));
  std::string dynamic_batch_size = "11";

  bool ret = CheckDynamicBatchSizeInputShapeValid(shape_map, dynamic_batch_size);
  EXPECT_EQ(ret, true);
}

TEST(UtestIrCommon, check_dynamic_images_size_input_shape_succ) {
  map<string, vector<int64_t>> shape_map;
  shape_map.insert(std::pair<string, vector<int64_t>>("data", {4, -1, -1, 5}));
  std::string input_format = "NCHW";
  std::string dynamic_image_size = "4,5";

  Status ret = CheckDynamicImagesizeInputShapeValid(shape_map, input_format, dynamic_image_size);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST(UtestIrCommon, check_dynamic_input_param_succ) {
  string dynamic_batch_size = "1";
  string dynamic_image_size;
  string dynamic_dims;
  string input_shape = "data:1,3,244,244";
  string input_shape_range;
  string input_format = "NCHW";
  bool is_dynamic_input = false;

  Status ret = CheckDynamicInputParamValid(dynamic_batch_size, dynamic_image_size, dynamic_dims,
                                           input_shape, input_shape_range, input_format,is_dynamic_input);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST(UtestIrCommon, check_compress_weight) {
  std::string enable_compress_weight = "true";
  std::string compress_weight_conf="./";
  Status ret = CheckCompressWeightParamValid(enable_compress_weight, compress_weight_conf);
  EXPECT_EQ(ret, PARAM_INVALID);

  enable_compress_weight = "yes";
  compress_weight_conf = "./";
  ret = CheckCompressWeightParamValid(enable_compress_weight, compress_weight_conf);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST(UtestIrCommon, check_param_failed) {
  std::string param_invalid = "invalid";

  Status ret = CheckOutputTypeParamValid(param_invalid);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = CheckBufferOptimizeParamValid(param_invalid);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = CheckKeepTypeParamValid(param_invalid);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = CheckInsertOpConfParamValid(param_invalid);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = CheckDisableReuseMemoryParamValid(param_invalid);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = CheckEnableSingleStreamParamValid(param_invalid);
  EXPECT_EQ(ret, PARAM_INVALID);

  std::string optypelist_for_implmode;
  std::string op_select_implmode = "1";
  ret = CheckImplmodeParamValid(optypelist_for_implmode, op_select_implmode);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = CheckLogParamValidAndSetLogLevel(param_invalid);
}
