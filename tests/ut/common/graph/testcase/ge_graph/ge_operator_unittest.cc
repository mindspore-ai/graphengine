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
#include <iostream>

#define private public
#define protected public
#include "graph/operator.h"

#include "graph/def_types.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/graph.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#undef private
#undef protected

using namespace std;
using namespace ge;

class UtestGeOperator : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
  string vec2str(vector<uint8_t> &vec) {
    string str((char *)vec.data(), vec.size());
    return str;
  }
};

TEST_F(UtestGeOperator, quant) {
  Operator op("quant");

  UsrQuantizeFactorParams q1;
  q1.quantize_algo = USR_HALF_OFFSET_ALGO;
  q1.scale_type = USR_SCALAR_SCALE;

  q1.quantize_param.scale_mode = USR_SQRT_MODE;
  string s1 = "value1";
  q1.quantize_param.set_scale_value(s1.data(), s1.size());
  q1.quantize_param.scale_offset = 5;
  string s2 = "value2";
  q1.quantize_param.set_offset_data_value(s2.data(), s2.size());
  q1.quantize_param.offset_data_offset = 6;
  string s3 = "value3";
  q1.quantize_param.set_offset_weight_value(s3.data(), s3.size());
  q1.quantize_param.offset_weight_offset = 7;
  string s4 = "value4";
  q1.quantize_param.set_offset_pad_value(s4.data(), s4.size());
  q1.quantize_param.offset_pad_offset = 8;

  q1.dequantize_param.scale_mode = USR_SQRT_MODE;
  q1.dequantize_param.set_scale_value(s1.data(), s1.size());
  q1.dequantize_param.scale_offset = 15;
  q1.dequantize_param.set_offset_data_value(s2.data(), s2.size());
  q1.dequantize_param.offset_data_offset = 16;
  q1.dequantize_param.set_offset_weight_value(s3.data(), s3.size());
  q1.dequantize_param.offset_weight_offset = 17;
  q1.dequantize_param.set_offset_pad_value(s4.data(), s4.size());
  q1.dequantize_param.offset_pad_offset = 18;

  q1.requantize_param.scale_mode = USR_SQRT_MODE;
  q1.requantize_param.set_scale_value(s1.data(), s1.size());
  q1.requantize_param.scale_offset = 25;
  q1.requantize_param.set_offset_data_value(s2.data(), s2.size());
  q1.requantize_param.offset_data_offset = 26;
  q1.requantize_param.set_offset_weight_value(s3.data(), s3.size());
  q1.requantize_param.offset_weight_offset = 27;
  q1.requantize_param.set_offset_pad_value(s4.data(), s4.size());
  q1.requantize_param.offset_pad_offset = 28;

  q1.quantizecalc_param.set_offsetw(s1.data(), s1.size());
  q1.quantizecalc_param.set_offsetd(s2.data(), s2.size());
  q1.quantizecalc_param.set_scalereq(s3.data(), s3.size());
  q1.quantizecalc_param.set_offsetdnext(s4.data(), s4.size());
  q1.quantizecalc_param.offsetw_offset = 34;
  q1.quantizecalc_param.offsetd_offset = 35;
  q1.quantizecalc_param.scaledreq_offset = 36;
  q1.quantizecalc_param.offsetdnext_offset = 37;

  op.SetAttr("quantize_factor", q1);
  UsrQuantizeFactorParams q2;
  op.GetAttr("quantize_factor", q2);

  EXPECT_EQ(q2.quantize_algo, USR_HALF_OFFSET_ALGO);
  EXPECT_EQ(q2.scale_type, USR_SCALAR_SCALE);

  EXPECT_EQ(q2.quantize_param.scale_mode, USR_SQRT_MODE);
  EXPECT_EQ(vec2str(q2.quantize_param.scale_value), s1);

  EXPECT_EQ(q2.quantize_param.scale_offset, 5);
  EXPECT_EQ(vec2str(q2.quantize_param.offset_data_value), s2);
  EXPECT_EQ(q2.quantize_param.offset_data_offset, 6);
  EXPECT_EQ(vec2str(q2.quantize_param.offset_weight_value), s3);
  EXPECT_EQ(q2.quantize_param.offset_weight_offset, 7);
  EXPECT_EQ(vec2str(q2.quantize_param.offset_pad_value), s4);
  EXPECT_EQ(q2.quantize_param.offset_pad_offset, 8);

  EXPECT_EQ(q2.dequantize_param.scale_mode, USR_SQRT_MODE);
  EXPECT_EQ(vec2str(q2.dequantize_param.scale_value), s1);
  EXPECT_EQ(q2.dequantize_param.scale_offset, 15);
  EXPECT_EQ(vec2str(q2.dequantize_param.offset_data_value), s2);
  EXPECT_EQ(q2.dequantize_param.offset_data_offset, 16);
  EXPECT_EQ(vec2str(q2.dequantize_param.offset_weight_value), s3);
  EXPECT_EQ(q2.dequantize_param.offset_weight_offset, 17);
  EXPECT_EQ(vec2str(q2.dequantize_param.offset_pad_value), s4);
  EXPECT_EQ(q2.dequantize_param.offset_pad_offset, 18);

  EXPECT_EQ(q2.requantize_param.scale_mode, USR_SQRT_MODE);
  EXPECT_EQ(vec2str(q2.requantize_param.scale_value), s1);
  EXPECT_EQ(q2.requantize_param.scale_offset, 25);
  EXPECT_EQ(vec2str(q2.requantize_param.offset_data_value), s2);
  EXPECT_EQ(q2.requantize_param.offset_data_offset, 26);
  EXPECT_EQ(vec2str(q2.requantize_param.offset_weight_value), s3);
  EXPECT_EQ(q2.requantize_param.offset_weight_offset, 27);
  EXPECT_EQ(vec2str(q2.requantize_param.offset_pad_value), s4);
  EXPECT_EQ(q2.requantize_param.offset_pad_offset, 28);

  EXPECT_EQ(vec2str(q2.quantizecalc_param.offsetw), s1);
  EXPECT_EQ(vec2str(q2.quantizecalc_param.offsetd), s2);
  EXPECT_EQ(vec2str(q2.quantizecalc_param.scalereq), s3);
  EXPECT_EQ(vec2str(q2.quantizecalc_param.offsetdnext), s4);
  EXPECT_EQ(q2.quantizecalc_param.offsetw_offset, 34);
  EXPECT_EQ(q2.quantizecalc_param.offsetd_offset, 35);
  EXPECT_EQ(q2.quantizecalc_param.scaledreq_offset, 36);
  EXPECT_EQ(q2.quantizecalc_param.offsetdnext_offset, 37);

  EXPECT_EQ(QuantizeFactorHasData(q2.quantize_param), true);
  EXPECT_EQ(QuantizeFactorHasData(q2.dequantize_param), true);
  EXPECT_EQ(QuantizeFactorHasData(q2.requantize_param), true);
  EXPECT_EQ(QuantizeFactorHasData(q2.quantizecalc_param), true);
}

TEST_F(UtestGeOperator, try_get_input_desc) {
  Operator data("data0");

  TensorDesc td;
  graphStatus ret = data.TryGetInputDesc("const", td);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGeOperator, get_dynamic_input_num) {
  Operator const_node("constNode");

  (void)const_node.DynamicInputRegister("data", 2, 1);
  int num = const_node.GetDynamicInputNum("data");
  EXPECT_EQ(num, 2);
}

TEST_F(UtestGeOperator, infer_format_func_register) {
  Operator add("add");
  std::function<graphStatus(Operator &)> func = nullptr;
  add.InferFormatFuncRegister(func);
}

graphStatus TestFunc(Operator &op) { return 0; }
TEST_F(UtestGeOperator, get_infer_format_func_register) {
  (void)OperatorFactoryImpl::GetInferFormatFunc("add");
  std::function<graphStatus(Operator &)> func = TestFunc;
  OperatorFactoryImpl::RegisterInferFormatFunc("add", TestFunc);
  (void)OperatorFactoryImpl::GetInferFormatFunc("add");
}

TEST_F(UtestGeOperator, get_attr_names_and_types) {
  Operator attr("attr");
  (void)attr.GetAllAttrNamesAndTypes();
}