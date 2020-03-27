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

#include <assert.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <sstream>

#include "common.h"
#include "ge_api.h"
#include "graph.h"
#include "ops/all_ops.h"
#include "types.h"
#include "utils/tensor_utils.h"

using namespace std;
using namespace ge;
using namespace op;

typedef bool (*Func)(Graph &graph);

#define PADDING_MODE 6
#define GRAD_PADDING_MODE 3
vector<int64_t> pad_1{1, 1, 1, 1};
vector<int64_t> pad_0{0, 0, 0, 0};
vector<int64_t> stride_1{1, 1};
vector<int64_t> stride_2{2, 2};

// (int out_channels, int h, int w, vector<uint_64> stride{1,1}, vector<uint_64> pad{1,1,1,1}, op::Data() input)
#define GENERATE_CONV_VAR(LAYER, BLK, OPNUM, in_channels, out_channels, h, w, stride, pad, input)                     \
  auto &LAYER##_##BLK##_##OPNUM##_input = input;                                                                      \
                                                                                                                      \
  TensorDesc LAYER##_##BLK##_##OPNUM##_desc(ge::Shape({out_channels, in_channels, h, w}), FORMAT_NCHW, DT_FLOAT);     \
  auto LAYER##_##BLK##_##OPNUM##_weight = op::Variable(string(#LAYER) + string(#BLK) + string(#OPNUM) + "_weight");   \
  LAYER##_##BLK##_##OPNUM##_weight.update_output_desc_y(LAYER##_##BLK##_##OPNUM##_desc);                              \
                                                                                                                      \
  auto LAYER##_##BLK##_##OPNUM##_mom_weight =                                                                         \
      op::Variable(string(#LAYER) + string(#BLK) + string(#OPNUM) + "_mom_weight");                                   \
  LAYER##_##BLK##_##OPNUM##_mom_weight.update_output_desc_y(LAYER##_##BLK##_##OPNUM##_desc);                          \
  LAYER##_##BLK##_##OPNUM##_mom_weight.update_input_desc_x(LAYER##_##BLK##_##OPNUM##_desc);                           \
                                                                                                                      \
  cout << string(#LAYER) + string(#BLK) + string(#OPNUM) << "'s weight shape is:" << in_channels << out_channels << h \
       << w << endl;                                                                                                  \
  cout << string(#LAYER) + string(#BLK) + string(#OPNUM)                                                              \
       << "'s input_x op's shape is:" << input.GetOutputDesc("y").GetShape().GetDim(2) << endl;                       \
  auto LAYER##_##BLK##_##OPNUM##_tmp_dims = input.GetOutputDesc("y").GetShape().GetDims();                            \
  for (auto LAYER##_##BLK##_##OPNUM##_tmp_it = LAYER##_##BLK##_##OPNUM##_tmp_dims.begin();                            \
       LAYER##_##BLK##_##OPNUM##_tmp_it != LAYER##_##BLK##_##OPNUM##_tmp_dims.end();                                  \
       LAYER##_##BLK##_##OPNUM##_tmp_it++) {                                                                          \
    cout << *LAYER##_##BLK##_##OPNUM##_tmp_it;                                                                        \
  }                                                                                                                   \
  cout << endl;                                                                                                       \
                                                                                                                      \
  auto LAYER##_##BLK##_##OPNUM = op::Conv2D(string(#LAYER) + string(#BLK) + string(#OPNUM))                           \
                                     .set_input_x(input, "y")                                                         \
                                     .set_input_filter(LAYER##_##BLK##_##OPNUM##_weight)                              \
                                     .set_attr_strides({1, 1, stride[0], stride[1]})                                  \
                                     .set_attr_pads(pad);                                                             \
  update_op_format(LAYER##_##BLK##_##OPNUM);

#define GENERATE_CONSTANT(LAYER, BLK, OPNUM, CONSTNAME)                                                           \
  Tensor LAYER##_##BLK##_##OPNUM##_##CONSTNAME##_tensor;                                                          \
  float *LAYER##_##BLK##_##OPNUM##_##CONSTNAME##_data = new float[LAYER##_##BLK##_##OPNUM##_size];                \
  for (int i = 0; i < (int)LAYER##_##BLK##_##OPNUM##_size; i++) {                                                 \
    *(LAYER##_##BLK##_##OPNUM##_##CONSTNAME##_data + i) = 0.01;                                                   \
  }                                                                                                               \
  LAYER##_##BLK##_##OPNUM##_##CONSTNAME##_tensor.SetData((uint8_t *)LAYER##_##BLK##_##OPNUM##_##CONSTNAME##_data, \
                                                         LAYER##_##BLK##_##OPNUM##_size * sizeof(float));         \
  LAYER##_##BLK##_##OPNUM##_##CONSTNAME##_tensor.SetTensorDesc(LAYER##_##BLK##_##OPNUM##_desc);                   \
                                                                                                                  \
  auto LAYER##_##BLK##_##OPNUM##_##CONSTNAME##_constant =                                                         \
      op::Constant().set_attr_value(LAYER##_##BLK##_##OPNUM##_##CONSTNAME##_tensor);                              \
  LAYER##_##BLK##_##OPNUM##_##CONSTNAME##_constant.update_output_desc_y(LAYER##_##BLK##_##OPNUM##_desc);          \
  delete[] LAYER##_##BLK##_##OPNUM##_##CONSTNAME##_data;

#define GENERATE_CONV_VAR_VAR(LAYER, BLK, OPNUM, in_channels, out_channels, h, w, stride, pad, input)               \
  TensorDesc LAYER##_##BLK##_##OPNUM##_desc(ge::Shape({out_channels, in_channels, h, w}), FORMAT_NCHW, DT_FLOAT);   \
  uint32_t LAYER##_##BLK##_##OPNUM##_size = LAYER##_##BLK##_##OPNUM##_desc.GetShape().GetShapeSize();               \
  auto LAYER##_##BLK##_##OPNUM##_weight = op::Variable(string(#LAYER) + string(#BLK) + string(#OPNUM) + "_weight"); \
  LAYER##_##BLK##_##OPNUM##_weight.update_output_desc_y(LAYER##_##BLK##_##OPNUM##_desc);                            \
                                                                                                                    \
  auto LAYER##_##BLK##_##OPNUM##_mom_weight =                                                                       \
      op::Variable(string(#LAYER) + string(#BLK) + string(#OPNUM) + "_mom_weight");                                 \
  LAYER##_##BLK##_##OPNUM##_mom_weight.update_output_desc_y(LAYER##_##BLK##_##OPNUM##_desc);                        \
                                                                                                                    \
  GENERATE_CONSTANT(LAYER, BLK, OPNUM, weight);                                                                     \
  auto LAYER##_##BLK##_##OPNUM##_weight_assign = op::Assign()                                                       \
                                                     .set_input_ref(LAYER##_##BLK##_##OPNUM##_weight)               \
                                                     .set_input_value(LAYER##_##BLK##_##OPNUM##_weight_constant);   \
                                                                                                                    \
  GENERATE_CONSTANT(LAYER, BLK, OPNUM, mom_weight);                                                                 \
  auto LAYER##_##BLK##_##OPNUM##_mom_weight_assign =                                                                \
      op::Assign()                                                                                                  \
          .set_input_ref(LAYER##_##BLK##_##OPNUM##_mom_weight)                                                      \
          .set_input_value(LAYER##_##BLK##_##OPNUM##_mom_weight_constant);                                          \
                                                                                                                    \
  input.push_back(LAYER##_##BLK##_##OPNUM##_weight);                                                                \
  input.push_back(LAYER##_##BLK##_##OPNUM##_mom_weight);

// (int out_channels, Operator& input)
#define GENERATE_BN_VAR(LAYER, BLK, OPNUM, out_channels, input)                                                   \
  auto &LAYER##_##BLK##_##OPNUM##_input = input;                                                                  \
                                                                                                                  \
  TensorDesc LAYER##_##BLK##_##OPNUM##_desc(ge::Shape({1, out_channels, 1, 1}), FORMAT_NCHW, DT_FLOAT);           \
  auto LAYER##_##BLK##_##OPNUM##_scale = op::Variable(string(#LAYER) + string(#BLK) + string(#OPNUM) + "_scale"); \
  LAYER##_##BLK##_##OPNUM##_scale.update_output_desc_y(LAYER##_##BLK##_##OPNUM##_desc);                           \
                                                                                                                  \
  auto LAYER##_##BLK##_##OPNUM##_mom_scale =                                                                      \
      op::Variable(string(#LAYER) + string(#BLK) + string(#OPNUM) + "_mom_scale");                                \
  LAYER##_##BLK##_##OPNUM##_mom_scale.update_output_desc_y(LAYER##_##BLK##_##OPNUM##_desc);                       \
                                                                                                                  \
  auto LAYER##_##BLK##_##OPNUM##_b = op::Variable(string(#LAYER) + string(#BLK) + string(#OPNUM) + "_b");         \
  LAYER##_##BLK##_##OPNUM##_b.update_output_desc_y(LAYER##_##BLK##_##OPNUM##_desc);                               \
                                                                                                                  \
  auto LAYER##_##BLK##_##OPNUM##_mom_b = op::Variable(string(#LAYER) + string(#BLK) + string(#OPNUM) + "_mom_b"); \
  LAYER##_##BLK##_##OPNUM##_mom_b.update_output_desc_y(LAYER##_##BLK##_##OPNUM##_desc);                           \
                                                                                                                  \
  auto LAYER##_##BLK##_##OPNUM##_mean = op::Variable(string(#LAYER) + string(#BLK) + string(#OPNUM) + "_mean");   \
  LAYER##_##BLK##_##OPNUM##_mean.update_output_desc_y(LAYER##_##BLK##_##OPNUM##_desc);                            \
  auto LAYER##_##BLK##_##OPNUM##_variance =                                                                       \
      op::Variable(string(#LAYER) + string(#BLK) + string(#OPNUM) + "_variance");                                 \
  LAYER##_##BLK##_##OPNUM##_variance.update_output_desc_y(LAYER##_##BLK##_##OPNUM##_desc);                        \
                                                                                                                  \
  auto LAYER##_##BLK##_##OPNUM = op::FusedBatchNorm(string(#LAYER) + string(#BLK) + string(#OPNUM))               \
                                     .set_input_x(input, "y")                                                     \
                                     .set_input_scale(LAYER##_##BLK##_##OPNUM##_scale)                            \
                                     .set_input_b(LAYER##_##BLK##_##OPNUM##_b)                                    \
                                     .set_input_mean(LAYER##_##BLK##_##OPNUM##_mean)                              \
                                     .set_input_variance(LAYER##_##BLK##_##OPNUM##_variance)                      \
                                     .set_attr_mode(1)                                                            \
                                     .set_attr_epsilon(1e-5)                                                      \
                                     .set_attr_is_training(true);

#define GENERATE_BN_VAR_VAR(LAYER, BLK, OPNUM, out_channels, input)                                                   \
  TensorDesc LAYER##_##BLK##_##OPNUM##_desc(ge::Shape({1, out_channels, 1, 1}), FORMAT_NCHW, DT_FLOAT);               \
  uint32_t LAYER##_##BLK##_##OPNUM##_size = LAYER##_##BLK##_##OPNUM##_desc.GetShape().GetShapeSize();                 \
  auto LAYER##_##BLK##_##OPNUM##_scale = op::Variable(string(#LAYER) + string(#BLK) + string(#OPNUM) + "_scale");     \
  LAYER##_##BLK##_##OPNUM##_scale.update_output_desc_y(LAYER##_##BLK##_##OPNUM##_desc);                               \
                                                                                                                      \
  auto LAYER##_##BLK##_##OPNUM##_mom_scale =                                                                          \
      op::Variable(string(#LAYER) + string(#BLK) + string(#OPNUM) + "_mom_scale");                                    \
  LAYER##_##BLK##_##OPNUM##_mom_scale.update_output_desc_y(LAYER##_##BLK##_##OPNUM##_desc);                           \
                                                                                                                      \
  auto LAYER##_##BLK##_##OPNUM##_b = op::Variable(string(#LAYER) + string(#BLK) + string(#OPNUM) + "_b");             \
  LAYER##_##BLK##_##OPNUM##_b.update_output_desc_y(LAYER##_##BLK##_##OPNUM##_desc);                                   \
                                                                                                                      \
  auto LAYER##_##BLK##_##OPNUM##_mom_b = op::Variable(string(#LAYER) + string(#BLK) + string(#OPNUM) + "_mom_b");     \
  LAYER##_##BLK##_##OPNUM##_mom_b.update_output_desc_y(LAYER##_##BLK##_##OPNUM##_desc);                               \
                                                                                                                      \
  auto LAYER##_##BLK##_##OPNUM##_mean = op::Variable(string(#LAYER) + string(#BLK) + string(#OPNUM) + "_mean");       \
  LAYER##_##BLK##_##OPNUM##_mean.update_output_desc_y(LAYER##_##BLK##_##OPNUM##_desc);                                \
  auto LAYER##_##BLK##_##OPNUM##_variance =                                                                           \
      op::Variable(string(#LAYER) + string(#BLK) + string(#OPNUM) + "_variance");                                     \
  LAYER##_##BLK##_##OPNUM##_variance.update_output_desc_y(LAYER##_##BLK##_##OPNUM##_desc);                            \
                                                                                                                      \
  GENERATE_CONSTANT(LAYER, BLK, OPNUM, scale);                                                                        \
                                                                                                                      \
  auto LAYER##_##BLK##_##OPNUM##_scale_assign = op::Assign()                                                          \
                                                    .set_input_ref(LAYER##_##BLK##_##OPNUM##_scale)                   \
                                                    .set_input_value(LAYER##_##BLK##_##OPNUM##_scale_constant);       \
  GENERATE_CONSTANT(LAYER, BLK, OPNUM, mom_scale);                                                                    \
                                                                                                                      \
  auto LAYER##_##BLK##_##OPNUM##_mom_scale_assign =                                                                   \
      op::Assign()                                                                                                    \
          .set_input_ref(LAYER##_##BLK##_##OPNUM##_mom_scale)                                                         \
          .set_input_value(LAYER##_##BLK##_##OPNUM##_mom_scale_constant);                                             \
                                                                                                                      \
  GENERATE_CONSTANT(LAYER, BLK, OPNUM, b);                                                                            \
                                                                                                                      \
  auto LAYER##_##BLK##_##OPNUM##_b_assign =                                                                           \
      op::Assign().set_input_ref(LAYER##_##BLK##_##OPNUM##_b).set_input_value(LAYER##_##BLK##_##OPNUM##_b_constant);  \
                                                                                                                      \
  GENERATE_CONSTANT(LAYER, BLK, OPNUM, mom_b);                                                                        \
                                                                                                                      \
  auto LAYER##_##BLK##_##OPNUM##_mom_b_assign = op::Assign()                                                          \
                                                    .set_input_ref(LAYER##_##BLK##_##OPNUM##_mom_b)                   \
                                                    .set_input_value(LAYER##_##BLK##_##OPNUM##_mom_b_constant);       \
  GENERATE_CONSTANT(LAYER, BLK, OPNUM, mean);                                                                         \
                                                                                                                      \
  auto LAYER##_##BLK##_##OPNUM##_mean_assign = op::Assign()                                                           \
                                                   .set_input_ref(LAYER##_##BLK##_##OPNUM##_mean)                     \
                                                   .set_input_value(LAYER##_##BLK##_##OPNUM##_mean_constant);         \
                                                                                                                      \
  GENERATE_CONSTANT(LAYER, BLK, OPNUM, variance);                                                                     \
                                                                                                                      \
  auto LAYER##_##BLK##_##OPNUM##_variance_assign = op::Assign()                                                       \
                                                       .set_input_ref(LAYER##_##BLK##_##OPNUM##_variance)             \
                                                       .set_input_value(LAYER##_##BLK##_##OPNUM##_variance_constant); \
                                                                                                                      \
  input.push_back(LAYER##_##BLK##_##OPNUM##_scale);                                                                   \
  input.push_back(LAYER##_##BLK##_##OPNUM##_mom_scale);                                                               \
  input.push_back(LAYER##_##BLK##_##OPNUM##_b);                                                                       \
  input.push_back(LAYER##_##BLK##_##OPNUM##_mom_b);                                                                   \
  input.push_back(LAYER##_##BLK##_##OPNUM##_mean);                                                                    \
  input.push_back(LAYER##_##BLK##_##OPNUM##_variance);

// (int out_channels, Operator& input)
#define GENERATE_RELU_VAR(LAYER, BLK, OPNUM, input) \
  auto &LAYER##_##BLK##_##OPNUM##_input = input;    \
  auto LAYER##_##BLK##_##OPNUM = op::Relu(string(#LAYER) + string(#BLK) + string(#OPNUM)).set_input_x(input, "y");

// (int out_channels, Operator& input)
#define GENERATE_MAXPOOL_VAR(LAYER, BLK, OPNUM, input)                                                 \
  auto &LAYER##_##BLK##_##OPNUM##_input = input;                                                       \
                                                                                                       \
  auto LAYER##_##BLK##_##OPNUM = op::MaxPoolWithArgmax(string(#LAYER) + string(#BLK) + string(#OPNUM)) \
                                     .set_input_x(input, "y")                                          \
                                     .set_attr_ksize({1, 3, 3, 1})                                     \
                                     .set_attr_padding("SAME")                                         \
                                     .set_attr_strides({1, 2, 2, 1});

// (int out_channels, Operator& input)
#define GENERATE_ADD_VAR(LAYER, BLK, OPNUM, input_x1, input_x2) \
  auto LAYER##_##BLK##_##OPNUM =                                \
      op::Add(string(#LAYER) + string(#BLK) + string(#OPNUM)).set_input_x1(input_x1, "y").set_input_x2(input_x2, "y");

// (int in_channels, int out_channels,vector<int64_t> stride{1,1}, Operator& input)
#define MAKE_RESIDUAL_BLOCK(LAYER, BLK, in_channels, out_channels, stride, input)                                 \
  auto &LAYER##_##BLK##_input = input;                                                                            \
  auto &LAYER##_##BLK##_stride = stride;                                                                          \
  int LAYER##_##BLK##_out_chls = out_channels / 4;                                                                \
                                                                                                                  \
  GENERATE_CONV_VAR(LAYER, BLK, conv1, in_channels, LAYER##_##BLK##_out_chls, 1, 1, stride, pad_0, input);        \
  GENERATE_BN_VAR(LAYER, BLK, bn1, LAYER##_##BLK##_out_chls, LAYER##_##BLK##_conv1);                              \
  GENERATE_RELU_VAR(LAYER, BLK, relu1, LAYER##_##BLK##_bn1);                                                      \
                                                                                                                  \
  GENERATE_CONV_VAR(LAYER, BLK, conv2, LAYER##_##BLK##_out_chls, LAYER##_##BLK##_out_chls, 3, 3, stride_1, pad_1, \
                    LAYER##_##BLK##_relu1);                                                                       \
  GENERATE_BN_VAR(LAYER, BLK, bn2, LAYER##_##BLK##_out_chls, LAYER##_##BLK##_conv2);                              \
  GENERATE_RELU_VAR(LAYER, BLK, relu2, LAYER##_##BLK##_bn2);                                                      \
                                                                                                                  \
  GENERATE_CONV_VAR(LAYER, BLK, conv3, LAYER##_##BLK##_out_chls, out_channels, 1, 1, stride_1, pad_0,             \
                    LAYER##_##BLK##_relu2);                                                                       \
  GENERATE_BN_VAR(LAYER, BLK, bn3, out_channels, LAYER##_##BLK##_conv3);                                          \
                                                                                                                  \
  GENERATE_CONV_VAR(LAYER, BLK, conv4, in_channels, out_channels, 1, 1, stride, pad_0, input);                    \
  GENERATE_BN_VAR(LAYER, BLK, bn4, out_channels, LAYER##_##BLK##_conv4);                                          \
                                                                                                                  \
  GENERATE_ADD_VAR(LAYER, BLK, add5, LAYER##_##BLK##_bn3, LAYER##_##BLK##_bn4);                                   \
  GENERATE_RELU_VAR(LAYER, BLK, relu5, LAYER##_##BLK##_add5);                                                     \
                                                                                                                  \
  auto &LAYER##_##BLK##_output = LAYER##_##BLK##_relu5;                                                           \
  auto &LAYER##_##BLK##_output_label = "y";

#define MAKE_RESIDUAL_BLOCK_VAR(LAYER, BLK, in_channels, out_channels, stride, input)                                 \
  int LAYER##_##BLK##_out_chls = out_channels / 4;                                                                    \
  GENERATE_CONV_VAR_VAR(LAYER, BLK, conv1, in_channels, LAYER##_##BLK##_out_chls, 1, 1, stride, pad_0, input);        \
  GENERATE_BN_VAR_VAR(LAYER, BLK, bn1, LAYER##_##BLK##_out_chls, input);                                              \
                                                                                                                      \
  GENERATE_CONV_VAR_VAR(LAYER, BLK, conv2, LAYER##_##BLK##_out_chls, LAYER##_##BLK##_out_chls, 3, 3, stride_1, pad_1, \
                        input);                                                                                       \
  GENERATE_BN_VAR_VAR(LAYER, BLK, bn2, LAYER##_##BLK##_out_chls, input);                                              \
                                                                                                                      \
  GENERATE_CONV_VAR_VAR(LAYER, BLK, conv3, LAYER##_##BLK##_out_chls, out_channels, 1, 1, stride_1, pad_0, input);     \
  GENERATE_BN_VAR_VAR(LAYER, BLK, bn3, out_channels, input);                                                          \
                                                                                                                      \
  GENERATE_CONV_VAR_VAR(LAYER, BLK, conv4, in_channels, out_channels, 1, 1, stride, pad_0, input);                    \
  GENERATE_BN_VAR_VAR(LAYER, BLK, bn4, out_channels, input);

// (int in_channels, int out_channels,vector<int64_t> stride{1,1}, Operator& input)
#define MAKE_NORMAL_BLOCK(LAYER, BLK, in_channels, out_channels, stride, input)                                   \
  auto &LAYER##_##BLK##_input = input;                                                                            \
  auto &LAYER##_##BLK##_stride = stride;                                                                          \
  int LAYER##_##BLK##_out_chls = out_channels / 4;                                                                \
                                                                                                                  \
  GENERATE_CONV_VAR(LAYER, BLK, conv1, in_channels, LAYER##_##BLK##_out_chls, 1, 1, stride, pad_0, input);        \
  GENERATE_BN_VAR(LAYER, BLK, bn1, LAYER##_##BLK##_out_chls, LAYER##_##BLK##_conv1);                              \
  GENERATE_RELU_VAR(LAYER, BLK, relu1, LAYER##_##BLK##_bn1);                                                      \
                                                                                                                  \
  GENERATE_CONV_VAR(LAYER, BLK, conv2, LAYER##_##BLK##_out_chls, LAYER##_##BLK##_out_chls, 3, 3, stride_1, pad_1, \
                    LAYER##_##BLK##_relu1);                                                                       \
  GENERATE_BN_VAR(LAYER, BLK, bn2, LAYER##_##BLK##_out_chls, LAYER##_##BLK##_conv2);                              \
  GENERATE_RELU_VAR(LAYER, BLK, relu2, LAYER##_##BLK##_bn2);                                                      \
                                                                                                                  \
  GENERATE_CONV_VAR(LAYER, BLK, conv3, LAYER##_##BLK##_out_chls, out_channels, 1, 1, stride_1, pad_0,             \
                    LAYER##_##BLK##_relu2);                                                                       \
  GENERATE_BN_VAR(LAYER, BLK, bn3, out_channels, LAYER##_##BLK##_conv3);                                          \
                                                                                                                  \
  GENERATE_ADD_VAR(LAYER, BLK, add5, LAYER##_##BLK##_bn3, input);                                                 \
  GENERATE_RELU_VAR(LAYER, BLK, relu5, LAYER##_##BLK##_add5);                                                     \
                                                                                                                  \
  auto &LAYER##_##BLK##_output = LAYER##_##BLK##_relu5;                                                           \
  auto &LAYER##_##BLK##_output_label = "y";

#define MAKE_NORMAL_BLOCK_VAR(LAYER, BLK, in_channels, out_channels, stride, input)                                   \
  int LAYER##_##BLK##_out_chls = out_channels / 4;                                                                    \
  GENERATE_CONV_VAR_VAR(LAYER, BLK, conv1, in_channels, LAYER##_##BLK##_out_chls, 1, 1, stride, pad_0, input);        \
  GENERATE_BN_VAR_VAR(LAYER, BLK, bn1, LAYER##_##BLK##_out_chls, input);                                              \
                                                                                                                      \
  GENERATE_CONV_VAR_VAR(LAYER, BLK, conv2, LAYER##_##BLK##_out_chls, LAYER##_##BLK##_out_chls, 3, 3, stride_1, pad_1, \
                        input);                                                                                       \
  GENERATE_BN_VAR_VAR(LAYER, BLK, bn2, LAYER##_##BLK##_out_chls, input);                                              \
                                                                                                                      \
  GENERATE_CONV_VAR_VAR(LAYER, BLK, conv3, LAYER##_##BLK##_out_chls, out_channels, 1, 1, stride_1, pad_0, input);     \
  GENERATE_BN_VAR_VAR(LAYER, BLK, bn3, out_channels, input);

// (int in_channels, int out_channels,vector<int64_t> stride{1,1}, Operator& input)
#define MAKE_RESIDUAL_LAYER(LAYER, in_channels, out_channels, stride, input)  \
  MAKE_RESIDUAL_BLOCK(LAYER, blk1, in_channels, out_channels, stride, input); \
                                                                              \
  auto &LAYER##_output = LAYER##_blk1_output;                                 \
  auto &LAYER##_output_label = LAYER##_blk1_output_label;

#define MAKE_RESIDUAL_LAYER_VAR(LAYER, in_channels, out_channels, stride, input) \
  MAKE_RESIDUAL_BLOCK_VAR(LAYER, blk1, in_channels, out_channels, stride, input);

// (int in_channels, int out_channels,vector<int64_t> stride{1,1}, Operator& input)
#define MAKE_NORMAL_LAYER(LAYER, in_channels, out_channels, stride, input)  \
  MAKE_NORMAL_BLOCK(LAYER, blk1, in_channels, out_channels, stride, input); \
                                                                            \
  auto &LAYER##_output = LAYER##_blk1_output;                               \
  auto &LAYER##_output_label = LAYER##_blk1_output_label;

#define MAKE_NORMAL_LAYER_VAR(LAYER, in_channels, out_channels, stride, input) \
  MAKE_NORMAL_BLOCK_VAR(LAYER, blk1, in_channels, out_channels, stride, input);

#define MAKE_RESNET50(input)                                         \
  MAKE_RESIDUAL_LAYER(layer1, 64, 256, stride_1, input)              \
  MAKE_NORMAL_LAYER(layer2, 256, 256, stride_1, layer1_output)       \
  MAKE_NORMAL_LAYER(layer3, 256, 256, stride_1, layer2_output)       \
  MAKE_RESIDUAL_LAYER(layer4, 256, 512, stride_2, layer3_output)     \
  MAKE_NORMAL_LAYER(layer5, 512, 512, stride_1, layer4_output)       \
  MAKE_NORMAL_LAYER(layer6, 512, 512, stride_1, layer5_output)       \
  MAKE_NORMAL_LAYER(layer7, 512, 512, stride_1, layer6_output)       \
  MAKE_RESIDUAL_LAYER(layer8, 512, 1024, stride_2, layer7_output)    \
  MAKE_NORMAL_LAYER(layer9, 1024, 1024, stride_1, layer8_output)     \
  MAKE_NORMAL_LAYER(layer10, 1024, 1024, stride_1, layer9_output)    \
  MAKE_NORMAL_LAYER(layer11, 1024, 1024, stride_1, layer10_output)   \
  MAKE_NORMAL_LAYER(layer12, 1024, 1024, stride_1, layer11_output)   \
  MAKE_NORMAL_LAYER(layer13, 1024, 1024, stride_1, layer12_output)   \
  MAKE_RESIDUAL_LAYER(layer14, 1024, 2048, stride_2, layer13_output) \
  MAKE_NORMAL_LAYER(layer15, 2048, 2048, stride_1, layer14_output)   \
  MAKE_NORMAL_LAYER(layer16, 2048, 2048, stride_1, layer15_output)   \
                                                                     \
  auto &resnet50_output = layer16_output;                            \
  auto &resnet50_output_label = layer16_output_label;

#define MAKE_RESNET50_VAR(inputs)                                \
  MAKE_RESIDUAL_LAYER_VAR(layer1, 64, 256, stride_1, inputs)     \
  MAKE_NORMAL_LAYER_VAR(layer2, 256, 256, stride_1, inputs)      \
  MAKE_NORMAL_LAYER_VAR(layer3, 256, 256, stride_1, inputs)      \
  MAKE_RESIDUAL_LAYER_VAR(layer4, 256, 512, stride_2, inputs)    \
  MAKE_NORMAL_LAYER_VAR(layer5, 512, 512, stride_1, inputs)      \
  MAKE_NORMAL_LAYER_VAR(layer6, 512, 512, stride_1, inputs)      \
  MAKE_NORMAL_LAYER_VAR(layer7, 512, 512, stride_1, inputs)      \
  MAKE_RESIDUAL_LAYER_VAR(layer8, 512, 1024, stride_2, inputs)   \
  MAKE_NORMAL_LAYER_VAR(layer9, 1024, 1024, stride_1, inputs)    \
  MAKE_NORMAL_LAYER_VAR(layer10, 1024, 1024, stride_1, inputs)   \
  MAKE_NORMAL_LAYER_VAR(layer11, 1024, 1024, stride_1, inputs)   \
  MAKE_NORMAL_LAYER_VAR(layer12, 1024, 1024, stride_1, inputs)   \
  MAKE_NORMAL_LAYER_VAR(layer13, 1024, 1024, stride_1, inputs)   \
  MAKE_RESIDUAL_LAYER_VAR(layer14, 1024, 2048, stride_2, inputs) \
  MAKE_NORMAL_LAYER_VAR(layer15, 2048, 2048, stride_1, inputs)   \
  MAKE_NORMAL_LAYER_VAR(layer16, 2048, 2048, stride_1, inputs)   \
//---------------------------------------------------------------------------------------------

// (Operator& input)
#define GENERATE_BIASADD_GRAD(LAYER, BLK, OPNUM, input)                                \
  auto LAYER##_##BLK##_##OPNUM##_grad =                                                \
      op::BiasAddGrad(string(#LAYER) + string(#BLK) + string(#OPNUM) + string("grad")) \
          .set_input_x(input, input.name_out_dx());

// (Operator& input)
#define GENERATE_MATMUL_GRAD(LAYER, BLK, OPNUM, input) \
  auto LAYER##_##BLK##_##OPNUM##_grad =                \
      op::MatMul(string(#LAYER) + string(#BLK) + string(#OPNUM) + string("grad")).set_input_x1(input);

// (Operator& input)
#define GENERATE_RESHAPE_GRAD(LAYER, BLK, OPNUM, input) \
  auto LAYER##_##BLK##_##OPNUM##_grad =                 \
      op::Reshape(string(#LAYER) + string(#BLK) + string(#OPNUM) + string("grad")).set_input_tensor(input);

// (Operator& input_grad, Operator& input_maxpool)
#define GENERATE_MAXPOOL_GRAD(LAYER, BLK, OPNUM, input_grad, input_maxpool)                      \
  auto LAYER##_##BLK##_##OPNUM##_grad =                                                          \
      op::MaxPoolGradWithArgmax(string(#LAYER) + string(#BLK) + string(#OPNUM) + string("grad")) \
          .set_input_x(LAYER##_##BLK##_##OPNUM##_input, "y")                                     \
          .set_input_grad(input_grad)                                                            \
          .set_input_argmax(input_maxpool, input_maxpool.name_out_argmax())                      \
          .set_attr_ksize({1, 1, 3, 3})                                                          \
          .set_attr_strides({1, 1, 2, 2})                                                        \
          .set_attr_padding("SAME");

// (Operator& input_dy)
#define GENERATE_RELU_GRAD(LAYER, BLK, OPNUM, input_dy, dy_label)                                                     \
  auto LAYER##_##BLK##_##OPNUM##_grad = op::ReluGrad(string(#LAYER) + string(#BLK) + string(#OPNUM) + string("grad")) \
                                            .set_input_gradients(input_dy, dy_label)                                  \
                                            .set_input_features(LAYER##_##BLK##_##OPNUM, "y");

// (Operator& input_dy)
#define GENERATE_BN_GRAD(LAYER, BLK, OPNUM, input_dy)                                                         \
  auto LAYER##_##BLK##_##OPNUM##_grad =                                                                       \
      op::FusedBatchNormGrad(string(#LAYER) + string(#BLK) + string(#OPNUM) + string("grad"))                 \
          .set_input_dy(input_dy, "backprops")                                                                \
          .set_input_x(LAYER##_##BLK##_##OPNUM##_input, "y")                                                  \
          .set_input_scale(LAYER##_##BLK##_##OPNUM##_scale)                                                   \
          .set_input_save_mean(LAYER##_##BLK##_##OPNUM, "save_mean")                                          \
          .set_input_save_inv_variance(LAYER##_##BLK##_##OPNUM, "save_inv_variance")                          \
          .set_attr_epsilon(0.0001);                                                                          \
                                                                                                              \
  auto LAYER##_##BLK##_##OPNUM##_momentum_scale =                                                             \
      op::ApplyMomentum()                                                                                     \
          .set_input_accum(LAYER##_##BLK##_##OPNUM##_mom_scale)                                               \
          .set_input_grad(LAYER##_##BLK##_##OPNUM##_grad, LAYER##_##BLK##_##OPNUM##_grad.name_out_bn_scale()) \
          .set_input_lr(label1)                                                                               \
          .set_input_momentum(label1)                                                                         \
          .set_input_var(LAYER##_##BLK##_##OPNUM##_scale);                                                    \
                                                                                                              \
  auto LAYER##_##BLK##_##OPNUM##_momentum_b =                                                                 \
      op::ApplyMomentum()                                                                                     \
          .set_input_accum(LAYER##_##BLK##_##OPNUM##_mom_b)                                                   \
          .set_input_grad(LAYER##_##BLK##_##OPNUM##_grad, LAYER##_##BLK##_##OPNUM##_grad.name_out_bn_bias())  \
          .set_input_lr(label1)                                                                               \
          .set_input_momentum(label1)                                                                         \
          .set_input_var(LAYER##_##BLK##_##OPNUM##_b);

// (Operator& input)
#define GENERATE_CONV_PROP_FILTER(LAYER, BLK, OPNUM, input_bngrad, stride)                                    \
  auto LAYER##_##BLK##_##OPNUM##_propfilter =                                                                 \
      op::Conv2DBackpropFilterD(string(#LAYER) + string(#BLK) + string(#OPNUM) + string("_propfilter"))       \
          .set_input_x(LAYER##_##BLK##_##OPNUM##_input, "y")                                                  \
          .set_attr_filter_sizes(LAYER##_##BLK##_##OPNUM##_desc.GetShape().GetDims())                         \
          .set_input_out_backprop(input_bngrad, input_bngrad.name_out_dx())                                   \
          .set_attr_strides(stride)                                                                           \
          .set_attr_pads({1, 1, 1, 1});                                                                       \
                                                                                                              \
  update_op_format(LAYER##_##BLK##_##OPNUM##_propfilter);                                                     \
  auto LAYER##_##BLK##_##OPNUM##_momentum_weight = op::ApplyMomentum()                                        \
                                                       .set_input_accum(LAYER##_##BLK##_##OPNUM##_mom_weight) \
                                                       .set_input_grad(LAYER##_##BLK##_##OPNUM##_propfilter)  \
                                                       .set_input_lr(label1)                                  \
                                                       .set_input_momentum(label1)                            \
                                                       .set_input_var(LAYER##_##BLK##_##OPNUM##_weight);

///.set_attr_input_sizes({input_bngrad.name_out_dx().GetOutputDesc().GetShape().GetDim(0),LAYER##_##BLK##_##OPNUM##_weight.GetOutputDesc().GetShape().GetDim(1),
///input_bngrad.name_out_dx().GetOutputDesc().GetShape().GetDim(2)*stride[2],
///input_bngrad.name_out_dx().GetOutputDesc().GetShape().GetDim(3)*stride[3]})
#define GENERATE_CONV_PROP_INPUT(LAYER, BLK, OPNUM, input_bngrad, stride)                                           \
  auto LAYER##_##BLK##_##OPNUM##_propinput =                                                                        \
      op::Conv2DBackpropInputD(string(#LAYER) + string(#BLK) + string(#OPNUM) + string("_propinput"))               \
          .set_attr_input_sizes(LAYER##_##BLK##_##OPNUM##_input.GetOutputDesc("y").GetShape().GetDims())            \
          .set_input_filters(LAYER##_##BLK##_##OPNUM##_weight)                                                      \
          .set_input_out_backprop(input_bngrad, input_bngrad.name_out_dx())                                         \
          .set_attr_strides(stride)                                                                                 \
          .set_attr_pads({1, 1, 1, 1});                                                                             \
  cout << string(#LAYER) + string(#BLK) + string(#OPNUM) + "_propinput"                                             \
       << "'s input_x op's shape is:" << input_bngrad.GetOutputDesc("dx").GetShape().GetDim(3) * stride[3] << endl; \
  cout << string(#LAYER) + string(#BLK) + string(#OPNUM) + "_propinput"                                             \
       << "'s input_x op's shape is:" << input_bngrad.GetOutputDesc("dx").GetShape().GetDim(2) * stride[2] << endl; \
                                                                                                                    \
  update_op_format(LAYER##_##BLK##_##OPNUM##_propinput);                                                            \
  auto &LAYER##_##BLK##_##OPNUM##_propinput_label = "y"

// (int out_channels, Operator& input)
#define GENERATE_ADD_GRAD(LAYER, BLK, OPNUM, input_x1, input_x1_label, input_x2, input_x2_label)                 \
  auto LAYER##_##BLK##_##OPNUM##_grad = op::Add(string(#LAYER) + string(#BLK) + string(#OPNUM) + string("grad")) \
                                            .set_input_x1(input_x1, input_x1_label)                              \
                                            .set_input_x2(input_x2, input_x2_label);

// (Operator& input)
#define MAKE_RESIDUAL_BLOCK_GRAD(LAYER, BLK, input_dy, dy_label)                                              \
  GENERATE_RELU_GRAD(LAYER, BLK, relu5, input_dy, dy_label);                                                  \
                                                                                                              \
  GENERATE_BN_GRAD(LAYER, BLK, bn4, LAYER##_##BLK##_relu5_grad);                                              \
  GENERATE_CONV_PROP_FILTER(LAYER, BLK, conv4, LAYER##_##BLK##_bn4_grad, LAYER##_##BLK##_stride);             \
  GENERATE_CONV_PROP_INPUT(LAYER, BLK, conv4, LAYER##_##BLK##_bn4_grad, LAYER##_##BLK##_stride);              \
                                                                                                              \
  GENERATE_BN_GRAD(LAYER, BLK, bn3, LAYER##_##BLK##_relu5_grad);                                              \
  GENERATE_CONV_PROP_FILTER(LAYER, BLK, conv3, LAYER##_##BLK##_bn3_grad, stride_1);                           \
  GENERATE_CONV_PROP_INPUT(LAYER, BLK, conv3, LAYER##_##BLK##_bn3_grad, stride_1);                            \
                                                                                                              \
  GENERATE_RELU_GRAD(LAYER, BLK, relu2, LAYER##_##BLK##_conv3_propinput, "y");                                \
  GENERATE_BN_GRAD(LAYER, BLK, bn2, LAYER##_##BLK##_relu2_grad);                                              \
  GENERATE_CONV_PROP_FILTER(LAYER, BLK, conv2, LAYER##_##BLK##_bn2_grad, stride_1);                           \
  GENERATE_CONV_PROP_INPUT(LAYER, BLK, conv2, LAYER##_##BLK##_bn2_grad, stride_1);                            \
                                                                                                              \
  GENERATE_RELU_GRAD(LAYER, BLK, relu1, LAYER##_##BLK##_conv2_propinput, "y");                                \
  GENERATE_BN_GRAD(LAYER, BLK, bn1, LAYER##_##BLK##_relu1_grad);                                              \
  GENERATE_CONV_PROP_FILTER(LAYER, BLK, conv1, LAYER##_##BLK##_bn1_grad, LAYER##_##BLK##_stride);             \
  GENERATE_CONV_PROP_INPUT(LAYER, BLK, conv1, LAYER##_##BLK##_bn1_grad, LAYER##_##BLK##_stride);              \
                                                                                                              \
  GENERATE_ADD_GRAD(LAYER, BLK, add5, LAYER##_##BLK##_conv1_propinput, LAYER##_##BLK##_conv1_propinput_label, \
                    LAYER##_##BLK##_conv4_propinput, LAYER##_##BLK##_conv4_propinput_label);                  \
                                                                                                              \
  auto &LAYER##_##BLK##_grad_output = LAYER##_##BLK##_add5_grad;                                              \
  auto &LAYER##_##BLK##_grad_output_label = "y"

// (Operator& input)
#define MAKE_NORMAL_BLOCK_GRAD(LAYER, BLK, input_dy, dy_label)                                                \
  GENERATE_RELU_GRAD(LAYER, BLK, relu5, input_dy, dy_label);                                                  \
                                                                                                              \
  GENERATE_BN_GRAD(LAYER, BLK, bn3, LAYER##_##BLK##_relu5_grad);                                              \
  GENERATE_CONV_PROP_FILTER(LAYER, BLK, conv3, LAYER##_##BLK##_bn3_grad, stride_1);                           \
  GENERATE_CONV_PROP_INPUT(LAYER, BLK, conv3, LAYER##_##BLK##_bn3_grad, stride_1);                            \
                                                                                                              \
  GENERATE_RELU_GRAD(LAYER, BLK, relu2, LAYER##_##BLK##_conv3_propinput, "y");                                \
  GENERATE_BN_GRAD(LAYER, BLK, bn2, LAYER##_##BLK##_relu2_grad);                                              \
  GENERATE_CONV_PROP_FILTER(LAYER, BLK, conv2, LAYER##_##BLK##_bn2_grad, stride_1);                           \
  GENERATE_CONV_PROP_INPUT(LAYER, BLK, conv2, LAYER##_##BLK##_bn2_grad, stride_1);                            \
                                                                                                              \
  GENERATE_RELU_GRAD(LAYER, BLK, relu1, LAYER##_##BLK##_conv2_propinput, "y");                                \
  GENERATE_BN_GRAD(LAYER, BLK, bn1, LAYER##_##BLK##_relu1_grad);                                              \
  GENERATE_CONV_PROP_FILTER(LAYER, BLK, conv1, LAYER##_##BLK##_bn1_grad, LAYER##_##BLK##_stride);             \
  GENERATE_CONV_PROP_INPUT(LAYER, BLK, conv1, LAYER##_##BLK##_bn1_grad, LAYER##_##BLK##_stride);              \
                                                                                                              \
  GENERATE_ADD_GRAD(LAYER, BLK, add5, LAYER##_##BLK##_conv1_propinput, LAYER##_##BLK##_conv1_propinput_label, \
                    input_dy, dy_label);                                                                      \
                                                                                                              \
  auto &LAYER##_##BLK##_grad_output = LAYER##_##BLK##_add5_grad;                                              \
  auto &LAYER##_##BLK##_grad_output_label = "y"

// (Operator& input_dy)
#define MAKE_RESIDUAL_LAYER_GRAD(LAYER, input_dy, dy_label)  \
  MAKE_RESIDUAL_BLOCK_GRAD(LAYER, blk1, input_dy, dy_label); \
                                                             \
  auto &LAYER##_grad_output = LAYER##_blk1_grad_output;      \
  auto &LAYER##_grad_output_label = LAYER##_blk1_grad_output_label;

// (Operator& input_dy)
#define MAKE_NORMAL_LAYER_GRAD(LAYER, input_dy, dy_label)  \
  MAKE_NORMAL_BLOCK_GRAD(LAYER, blk1, input_dy, dy_label); \
                                                           \
  auto &LAYER##_grad_output = LAYER##_blk1_grad_output;    \
  auto &LAYER##_grad_output_label = LAYER##_blk1_grad_output_label;

#define MAKE_RESNET50_GRAD(input_dy, dy_label)                                      \
  MAKE_NORMAL_LAYER_GRAD(layer16, input_dy, dy_label)                               \
  MAKE_NORMAL_LAYER_GRAD(layer15, layer16_grad_output, layer16_grad_output_label)   \
  MAKE_RESIDUAL_LAYER_GRAD(layer14, layer15_grad_output, layer15_grad_output_label) \
  MAKE_NORMAL_LAYER_GRAD(layer13, layer14_grad_output, layer14_grad_output_label)   \
  MAKE_NORMAL_LAYER_GRAD(layer12, layer13_grad_output, layer13_grad_output_label)   \
  MAKE_NORMAL_LAYER_GRAD(layer11, layer12_grad_output, layer12_grad_output_label)   \
  MAKE_NORMAL_LAYER_GRAD(layer10, layer11_grad_output, layer11_grad_output_label)   \
  MAKE_NORMAL_LAYER_GRAD(layer9, layer10_grad_output, layer10_grad_output_label)    \
  MAKE_RESIDUAL_LAYER_GRAD(layer8, layer9_grad_output, layer9_grad_output_label)    \
  MAKE_NORMAL_LAYER_GRAD(layer7, layer8_grad_output, layer8_grad_output_label)      \
  MAKE_NORMAL_LAYER_GRAD(layer6, layer7_grad_output, layer7_grad_output_label)      \
  MAKE_NORMAL_LAYER_GRAD(layer5, layer6_grad_output, layer6_grad_output_label)      \
  MAKE_RESIDUAL_LAYER_GRAD(layer4, layer5_grad_output, layer5_grad_output_label)    \
  MAKE_NORMAL_LAYER_GRAD(layer3, layer4_grad_output, layer4_grad_output_label)      \
  MAKE_NORMAL_LAYER_GRAD(layer2, layer3_grad_output, layer3_grad_output_label)      \
  MAKE_RESIDUAL_LAYER_GRAD(layer1, layer2_grad_output, layer2_grad_output_label)    \
                                                                                    \
  auto &resnet50_grad_output = layer1_grad_output;                                  \
  auto &resnet50_grad_output_label = layer1_grad_output_label;

bool resnet50(Graph &graph) {
  auto data = op::Data().set_attr_index(0);
  auto data1 = op::Data().set_attr_index(1);
  TensorDesc shape_desc(ge::Shape({32, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  data.update_output_desc_out(shape_desc);

  TensorDesc desc(ge::Shape({64, 3, 7, 7}), FORMAT_NCHW, DT_FLOAT);

  auto var = op::Variable("conv2d_var");
  var.update_output_desc_y(desc);
  var.update_input_desc_x(desc);

  auto varw1 = op::Variable("conv2d_varw1");
  varw1.update_output_desc_y(desc);

  auto conv2d = op::Conv2D("Translate")
                    .set_input_x(data)
                    .set_input_filter(var)
                    .set_attr_strides({1, 1, 2, 2})
                    .set_attr_pads({2, 3, 2, 3});
  TensorDesc desc_y;
  desc_y.SetFormat(FORMAT_NCHW); // shape: 32 64 112 112
  conv2d.update_output_desc_y(desc_y);

  TensorDesc desc1(ge::Shape({1, 64, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  auto var1 = op::Variable("bn_var1");
  var1.update_output_desc_y(desc1);

  auto var2 = op::Variable("bn_var2");
  var2.update_output_desc_y(desc1);

  auto var3 = op::Variable("bn_var3");
  var3.update_output_desc_y(desc1);

  auto var4 = op::Variable("bn_var4");
  var4.update_output_desc_y(desc1);

  TensorDesc desc2(ge::Shape({2048, 1001}), FORMAT_NCHW, DT_FLOAT);

  auto var5 = op::Variable("var5");
  var5.update_output_desc_y(desc2);

  auto var6 = op::Variable("var6");
  var6.update_output_desc_y(desc2);

  TensorDesc desclabel(ge::Shape({1, 1001, 1, 1}), FORMAT_NCHW, DT_FLOAT);

  auto label1 = op::Variable("label1");
  label1.update_output_desc_y(desclabel);

  TensorDesc descmatlabel(ge::Shape({1, 1001, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  auto matvar = op::Variable("matvar");
  matvar.update_output_desc_y(descmatlabel);

  auto matvar1 = op::Variable("matvar1");
  matvar1.update_output_desc_y(descmatlabel);

  auto bn = op::FusedBatchNorm()
                .set_input_x(conv2d, "y")
                .set_input_scale(var1)
                .set_input_b(var2)
                .set_input_mean(var3)
                .set_input_variance(var4)
                .set_attr_mode(1)
                .set_attr_epsilon(1e-5)
                .set_attr_is_training(true)
                .set_attr_is_training_fusion(true)
                .set_attr_moving_average_fraction(994352128);

  auto relu = op::Relu().set_input_x(bn, "y");

  auto maxpool = op::MaxPoolWithArgmax()
                     .set_input_x(relu, "y")
                     .set_attr_ksize({1, 3, 3, 1})
                     .set_attr_padding("SAME")
                     .set_attr_strides({1, 2, 2, 1});

  MAKE_RESNET50(maxpool);
  std::vector<Operator> inputs{data};  //,var,var1,layer1_blk1_bn1_b,var3,var4};
  std::vector<Operator> outputs{};

  graph.SetInputs(inputs).SetOutputs(outputs);
  return true;
}

#define GENERATE_CONSTANT_USE_DESC(OPNUM, desc, val)                                 \
  uint32_t OPNUM##_size = desc.GetShape().GetShapeSize();                            \
  Tensor OPNUM##_tensor;                                                             \
  OPNUM##_tensor.SetTensorDesc(desc);                                                \
  if (desc.GetDataType() == DT_FLOAT) {                                              \
    float *OPNUM##_data = new float[OPNUM##_size];                                   \
    for (int i = 0; i < (int)OPNUM##_size; i++) {                                    \
      *(OPNUM##_data + i) = val;                                                     \
    }                                                                                \
    OPNUM##_tensor.SetData((uint8_t *)OPNUM##_data, OPNUM##_size * sizeof(float));   \
    delete[] OPNUM##_data;                                                           \
  }                                                                                  \
  if (desc.GetDataType() == DT_INT64) {                                              \
    int64_t *OPNUM##_data = new int64_t[OPNUM##_size];                               \
    for (int i = 0; i < (int)OPNUM##_size; i++) {                                    \
      *(OPNUM##_data + i) = val;                                                     \
    }                                                                                \
    OPNUM##_tensor.SetData((uint8_t *)OPNUM##_data, OPNUM##_size * sizeof(int64_t)); \
    delete[] OPNUM##_data;                                                           \
  }                                                                                  \
  auto OPNUM##_constant = op::Constant().set_attr_value(OPNUM##_tensor);             \
  OPNUM##_constant.update_output_desc_y(desc);

#define GENERATE_VAR_LAYER(OPNUM, desc, input)                                                        \
  auto OPNUM##_weight = op::Variable(string(#OPNUM));                                                 \
  OPNUM##_weight.update_output_desc_y(desc);                                                          \
  auto OPNUM##_assign = op::Assign().set_input_ref(OPNUM##_weight).set_input_value(OPNUM##_constant); \
                                                                                                      \
  input.push_back(OPNUM##_weight);

#define GENERATE_VAR_LAYER_1(OPNUM, desc, var_format, input, name)                                    \
  auto OPNUM##_weight = op::Variable(string(name));                                                   \
  OPNUM##_weight.update_output_desc_y(desc);                                                          \
  auto OPNUM##_assign = op::Assign().set_input_ref(OPNUM##_weight).set_input_value(OPNUM##_constant); \
                                                                                                      \
  input.push_back(OPNUM##_weight);

int BuildInitVarGraph(Graph &graph) {
  std::vector<Operator> inputs{};
  std::vector<Operator> outputs{};

  TensorDesc desc(ge::Shape({64, 3, 7, 7}), FORMAT_NCHW, DT_FLOAT);
  GENERATE_CONSTANT_USE_DESC(conv2d_var, desc, 0.01);
  GENERATE_VAR_LAYER(conv2d_var, desc, inputs);

  GENERATE_CONSTANT_USE_DESC(conv2d_varw1, desc, 0.01);
  GENERATE_VAR_LAYER(conv2d_varw1, desc, inputs);

  TensorDesc desc1(ge::Shape({1, 64, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  GENERATE_CONSTANT_USE_DESC(bn_var1, desc1, 0.01);
  GENERATE_VAR_LAYER(bn_var1, desc1, inputs);
  GENERATE_CONSTANT_USE_DESC(bn_var2, desc1, 0.01);
  GENERATE_VAR_LAYER(bn_var2, desc1, inputs);
  GENERATE_CONSTANT_USE_DESC(bn_var3, desc1, 0.01);
  GENERATE_VAR_LAYER(bn_var3, desc1, inputs);
  GENERATE_CONSTANT_USE_DESC(bn_var4, desc1, 0.01);
  GENERATE_VAR_LAYER(bn_var4, desc1, inputs);

  TensorDesc desc2(ge::Shape({2048, 1001}), FORMAT_NCHW, DT_FLOAT);
  GENERATE_CONSTANT_USE_DESC(var5, desc2, 0.01);
  GENERATE_VAR_LAYER(var5, desc2, inputs);
  GENERATE_CONSTANT_USE_DESC(var6, desc2, 0.01);
  GENERATE_VAR_LAYER(var6, desc2, inputs);

  TensorDesc desclabel(ge::Shape({1, 1001, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  GENERATE_CONSTANT_USE_DESC(label1, desclabel, 0.1);
  GENERATE_VAR_LAYER(label1, desclabel, inputs);

  TensorDesc descmatlabel(ge::Shape({1, 1001, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  GENERATE_CONSTANT_USE_DESC(matvar, descmatlabel, 0.01);
  GENERATE_VAR_LAYER(matvar, descmatlabel, inputs);
  GENERATE_CONSTANT_USE_DESC(matvar1, descmatlabel, 0.01);
  GENERATE_VAR_LAYER(matvar1, descmatlabel, inputs);

  MAKE_RESNET50_VAR(inputs);

  TensorDesc ctrl(ge::Shape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT64);

  GENERATE_CONSTANT_USE_DESC(iterations_per_loop, ctrl, 100);
  GENERATE_VAR_LAYER_1(iterations_per_loop, ctrl, "4D", inputs, "npu_runconfig/iterations_per_loop");
  GENERATE_CONSTANT_USE_DESC(loop_cond, ctrl, 0);
  GENERATE_VAR_LAYER_1(loop_cond, ctrl, "4D", inputs, "npu_runconfig/loop_cond");
  GENERATE_CONSTANT_USE_DESC(one, ctrl, 1);
  GENERATE_VAR_LAYER_1(one, ctrl, "4D", inputs, "npu_runconfig/one");
  GENERATE_CONSTANT_USE_DESC(zero, ctrl, 0);
  GENERATE_VAR_LAYER_1(zero, ctrl, "4D", inputs, "npu_runconfig/zero");

  graph.SetInputs(inputs).SetOutputs(outputs);
  return 0;
}
int TestBuildGraphTest(Func fun, Graph &graph, vector<ge::Tensor> &inputs, vector<ge::Tensor> &outputs) {
  bool graph_ret = fun(graph);
  ge::Tensor shapeTensor;
  TensorDesc shape_desc(ge::Shape({32, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  uint32_t sizeshape = shape_desc.GetShape().GetShapeSize();
  printf("[test] desc size filter shape:%u\n", sizeshape);
  shapeTensor.SetTensorDesc(shape_desc);
  vector<float> dataValuec;
  for (int i = 0; i < sizeshape; i++) {
    // dataValuec.push_back((float)(i%255));
    dataValuec.push_back(1);
  }

  shapeTensor.SetData((uint8_t *)dataValuec.data(), 4 * sizeshape);
  inputs.push_back(shapeTensor);

  ge::Tensor shapeTensor1;
  TensorDesc shape_desc1(ge::Shape({1, 32, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  uint32_t sizeshape1 = shape_desc1.GetShape().GetShapeSize();
  printf("[test] desc size filter shape:%u\n", sizeshape1);
  shapeTensor1.SetTensorDesc(shape_desc1);
  vector<int32_t> dataValuec1;
  for (int i = 0; i < sizeshape1; i++) {
    dataValuec1.push_back(1);
  }

  shapeTensor1.SetData((uint8_t *)dataValuec1.data(), 4 * sizeshape1);
  // inputs.push_back(shapeTensor1);

  return 0;
}
int runTrainGraph(Func fun, int loopCount) {
  printf("GE BBIT begin...\n");
  std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

  std::map<std::string, std::string> ge_options = {
      {"device_id", "0"}, {"rank_table_file", ""}, {"graphType", "1"}, {"ge.graphRunMode", "2"}};

  std::map<std::string, std::string> session_options = {{"a", "b"}, {TRAIN_FLAG, "1"}};

  ge::Status ret;

  // init ge
  ret = GEInitialize_api_new("train", "fe,plugin");
  printf("ge::GEInitialize ret:%d\n", ret);

  // init session
  ge::Session session(session_options);

  int graphId_initvar = 1;
  ge::Graph graph_initvar("initVarGraph");
  bool graph_ret = BuildInitVarGraph(graph_initvar);

  // session addgraph
  int graphId = 0;

  // build graph
  ge::Graph graph("bigGraph");
  std::vector<ge::Tensor> inputs;
  ge::Tensor outputTensor;
  std::vector<ge::Tensor> outputs;
  graph_ret = TestBuildGraphTest(fun, graph, inputs, outputs);
  printf("TestReluGrad ret:%d\n", graph_ret);

  ret = session.AddGraph(graphId_initvar, graph_initvar);
  printf("session.AddVarGraph ret:%d\n", ret);
  if (ret) return ret;

  ret = session.AddGraph(graphId, graph);
  printf("session.AddGraph ret:%d\n", ret);
  if (ret) return ret;

  std::vector<ge::Tensor> inputs1;
  std::vector<ge::Tensor> outputs1;
  ret = session.RunGraph(graphId_initvar, inputs1, outputs1);

  if (ret != SUCCESS) {
    return ret;
  }
  // add loop for test of stabilty:
  for (int i = 0; i < loopCount; i++) {
    // session rungraph
    printf("loopCount:%d\n", loopCount);
    ret = session.RunGraph(graphId, inputs, outputs);
    printf("session.RunGraph ret:%d\n", ret);
    if (ret) return ret;

    // define 99999 as loop forever
    if (loopCount == 99999) i = 0;
  }
  std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
  auto millisecondsduration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  auto ms = millisecondsduration.count();
  std::stringstream ss;
  ss << ms << "ms";
  std::string run_time = ss.str();
  printf("run time is : %s \n", run_time.c_str());

  return 0;
}

int main(int argc, char *argv[]) {
  // add loop for test of stabilty:
  int loopCount = 1;
  if (argc >= 2) loopCount = atoi(argv[1]);

  Status ret = SUCCESS;
  ret = runTrainGraph(resnet50, loopCount);
  if (ret == SUCCESS) {
    std::cout << "[train resnet50 success]" << std::endl;
  } else {
    std::cout << "!!! train resnet50 fail !!!" << std::endl;
  }
  return ret;
}
