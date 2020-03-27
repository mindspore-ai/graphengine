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

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#include "common.h"
#include "model.h"

#define MAX_HEAD_SIZE 50

using namespace std;
using namespace ge;

void update_op_format(Operator ops, Format format) {
  printf("set format begin.........\n");
  ge::TensorDesc tensor_desc_x = ops.GetInputDesc("x");
  ge::TensorDesc tensor_desc_y = ops.GetOutputDesc("y");
  Format f_x0 = tensor_desc_x.GetFormat();
  Format f_y0 = tensor_desc_x.GetFormat();
  printf("before set  x format:%d \n", f_x0);
  printf("before set  y format:%d \n", f_y0);
  printf("format to be set is :%d \n", format);
  tensor_desc_x.SetFormat(format);
  tensor_desc_y.SetFormat(format);
  ops.UpdateInputDesc("x", tensor_desc_x);
  ops.UpdateOutputDesc("y", tensor_desc_y);
  Format f_x = tensor_desc_x.GetFormat();
  Format f_y = tensor_desc_y.GetFormat();
  printf("after set  x format:%d \n", f_x);
  printf("after set  y format:%d \n", f_y);
}

/// getDimInfo: get dim info from data file
/// param:
/// fp: the testing datafile object
///
/// return :
/// dim_info: array to store the info of the dim in datafile, like [4,3,3,6,3,162(3*3*6*3)],4 is dim size,3,3,6,3 is the
/// dim shape data_size: the size of the testing data including the data file
void getDimInfo(FILE *fp, std::vector<uint64_t> &dim_info) {
  // get dim info from hisi testing data file
  uint32_t *dim_buffer = (uint32_t *)malloc(MAX_HEAD_SIZE * sizeof(uint32_t));
  fread(dim_buffer, sizeof(uint32_t), MAX_HEAD_SIZE, fp);
  dim_info.push_back(*dim_buffer);  // get dim size

  // get data shape to compute the datasize
  uint64_t data_size = 1;
  uint32_t i = 1;
  for (; i <= dim_info[0]; i++) {
    dim_info.push_back(*(dim_buffer + i));
    data_size *= *(dim_buffer + i);
  }
  dim_info.push_back(data_size);

  free(dim_buffer);
}

/// readTestDataFile: read test date from hisi .t datafile
/// param:
///  infile: the path of hisi .t datafile
/// return:
///  dim_info: array to store the info of the dim in datafile, like [4,3,3,6,3],4 is dim size,3,3,6,3 is the dim shape
void *readTestDataFile(std::string infile, std::vector<uint64_t> &dim_info) {
  FILE *fp;
  fp = fopen(infile.c_str(), "r");

  if (fp == NULL) {
    printf("ERROR: cant't open file %s\n", infile.c_str());
    return NULL;
  } else {
    getDimInfo(fp, dim_info);
    uint64_t data_size = dim_info[dim_info.size() - 1];

    fclose(fp);

    fp = fopen(infile.c_str(), "r");
    if (fp == NULL) {
      printf("ERROR: cant't open file %s\n", infile.c_str());
      return NULL;
    }
    uint32_t *memory = (uint32_t *)malloc((dim_info[0] + 1 + data_size) * sizeof(uint32_t));
    fread(memory, sizeof(uint32_t), (dim_info[0] + 1 + data_size), fp);
    fclose(fp);
    return memory + (dim_info[0] + 1);
  }
}

void *readUint8TestDataFile(std::string infile, int size) {
  FILE *fp;
  fp = fopen(infile.c_str(), "r");

  if (fp == NULL) {
    printf("ERROR: cant't open file %s\n", infile.c_str());
    return NULL;
  }
  uint8_t *memory = (uint8_t *)malloc((size) * sizeof(uint8_t));
  fread(memory, sizeof(uint8_t), (size), fp);
  fclose(fp);
  return memory;
}

/// allclose
/// param:
///  a:compared file a
///  b:compared file b
///  count: the count size which will compare
///  rtol:
///  atol:
/// return:
///  true or false
bool allclose(float *a, float *b, uint64_t count, float rtol = 1e-05, float atol = 1e-08) {
  uint32_t i = 0;

  for (; i < count; ++i) {
    if (fabs(a[i] - b[i]) > (atol + rtol * fabs(b[i]))) {
      printf("compara failed: i= %d, a[i]=%f, b[i]=%f,atol=%f,rtol=%f\n", i, a[i], b[i], atol, rtol);
      return false;
    }
  }

  return true;
}

/// compFp32WithTData: compare the data with the data in hisi .t file
/// param:
///  actual_output_data: the result of ge
///  expected_data_file: the path of hisi .t result file
///  rtol:
///  atol:
/// return:
///  true of false
bool compFp32WithTData(float *actual_output_data, std::string expected_data_file, float rtol = 1e-05, float atol = 1e-08) {
  std::vector<uint64_t> dim_info;
  float *expected_output_data = (float *)readTestDataFile(expected_data_file, dim_info);

  uint32_t i = 1;
  uint64_t data_size = 1;
  for (; i <= dim_info[0]; i++) {
    data_size *= dim_info[i];
  }
  return allclose(actual_output_data, expected_output_data, data_size, rtol, atol);
}

int SwitchDatatype(DataType dt) {
  int size = 1;
  if (dt == ge::DT_FLOAT) size = 4;
  if (dt == ge::DT_INT32) size = 4;
  if (dt == ge::DT_FLOAT16) size = 2;
  if (dt == ge::DT_INT64) size = 8;
  return size;
}

ge::Tensor genTensor(std::vector<int64_t> tensor_shape, Format format, DataType dt) {
  int size = 1;
  for (int i = 0; i < tensor_shape.size(); i++) {
    size = size * tensor_shape[i];
  }

  int data_type_size = SwitchDatatype(dt);

  size = abs(size * data_type_size);
  vector<uint8_t> data_value;

  if (size == 0) {
    TensorDesc input_tensor_desc = TensorDesc(ge::Shape(tensor_shape), format, dt);
    input_tensor_desc.SetRealDimCnt(tensor_shape.size());
    Tensor gen_tensor = Tensor(input_tensor_desc, data_value);
    return gen_tensor;
  }
  for (int i = 0; i < size; i++) {
    data_value.push_back(1);
  }
  TensorDesc input_tensor_desc = TensorDesc(ge::Shape(tensor_shape), format, dt);
  input_tensor_desc.SetRealDimCnt(tensor_shape.size());
  Tensor gen_tensor = Tensor(input_tensor_desc, data_value);
  return gen_tensor;
}

ge::Tensor genTensor_withVaule(std::vector<int64_t> tensor_shape, float value) {
  int size = 1;
  for (int i = 0; i < tensor_shape.size(); i++) {
    size = size * tensor_shape[i];
  }

  float *data_value = new float[size];
  for (int i = 0; i < size; i++) {
    *(data_value + i) = value;
  }
  Tensor gen_ge_tensor;
  TensorDesc input_tensor_desc = TensorDesc(ge::Shape(tensor_shape), FORMAT_NCHW);
  gen_ge_tensor.SetTensorDesc(input_tensor_desc);
  gen_ge_tensor.SetData((uint8_t *)data_value, size * 4);

  return gen_ge_tensor;
}

Tensor genTesnor_Shape_as_data(std::vector<int64_t> tensor_shape) {
  Format format = FORMAT_NCHW;
  DataType dt = DT_INT32;
  int size = tensor_shape.size();
  int32_t *tensor_data = new int32_t[size];
  std::cout << "shape tensor size:" << size << endl;
  for (int i = 0; i < size; i++) {
    *(tensor_data + i) = tensor_shape[i];
  }

  Tensor gen_tensor;
  TensorDesc input_tensor_desc = TensorDesc(ge::Shape({size}), FORMAT_NCHW, DT_INT32);
  gen_tensor.SetData((uint8_t *)tensor_data, size * GetDatTypeSize(dt));
  gen_tensor.SetTensorDesc(input_tensor_desc);

  return gen_tensor;
}

/// train_flag is 0 when infer; train_flag is 1 when train; train_flag is 0 default
/// run_mode_path is not 0,1,2 when TBE; run_mode_path is 1 when FE; run_mode_path is 0 default
/// run_mode_path is 2 now when AICPU, ge.enabledlocalFmkop is 1
ge::Status GEInitialize_api(string train_flag, string run_mode_path) {
  ge::Status ret;
  if (run_mode_path == "0") {
    const std::map<string, string> config = {
        {"device_id", "0,2,4,6"},
        {"rank_table_file", "hccl from csa/paas"},
        {"ge.graphRunMode", train_flag},
        {"ge.aicpuFlag", "1"},
        {"ge.feFlag", "1"},
        {DDK_VERSION_FLAG, "1.60.T17.B830"},
        {"ge.soLoadPath",
         "/usr/local/HiAI/runtime/lib64/plugin/opskernel/libfe.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/"
         "libaicpu_plugin.so"}};
    ret = ge::GEInitialize(config);
  } else if (run_mode_path == "1") {
    const std::map<string, string> config = {
        {"device_id", "0,2,4,6"},
        {"rank_table_file", "hccl from csa/paas"},
        {"ge.graphRunMode", train_flag},
        {"ge.feFlag", "1"},
        {DDK_VERSION_FLAG, "1.60.T17.B830"},
        {TBE_PLUGIN_PATH_FLAG, "/usr/local/HiAI/runtime/lib64/tbe_plugin/bert"},
        {"ge.soLoadPath", "/usr/local/HiAI/runtime/lib64/plugin/opskernel/libfe.so"}};
    ret = ge::GEInitialize(config);
  } else if (run_mode_path == "2") {
    const std::map<string, string> config = {{"device_id", "0,2,4,6"},
                                             {"rank_table_file", "hccl from csa/paas"},
                                             {"ge.graphRunMode", train_flag},
                                             {LOCAL_FMKOP_FLAG, "1"}};
    ret = ge::GEInitialize(config);
  } else {
    const std::map<string, string> config = {
        {"device_id", "0,2,4,6"},
        {"rank_table_file", "hccl from csa/paas"},
        {"ge.graphRunMode", train_flag},
        {DDK_VERSION_FLAG, "1.60.T17.B830"},
        {TBE_PLUGIN_PATH_FLAG, "/usr/local/HiAI/runtime/lib64/tbe_plugin/" + run_mode_path}};
    ret = ge::GEInitialize(config);
  }
  std::cout << "GEInitialize_ret is " << ret << std::endl;

  return ret;
}

/// train_flag is infer default
/// run_mode: is multi group of [fe,aicpu,bert,deeplabv3,mobilenetv2,single_path_nas,ssd]
/// but bert,deeplabv3,mobilenetv2,single_path_nas,ssd can only set one value from array
/// eg:"fe,aicpu,bert" or "fe", default is “fe”
/// "fe,aicpu,bert" remain open fe aicpu and bert
ge::Status GEInitialize_api_new(string train_flag, string run_mode) {
  ge::Status ret;
  vector<string> modes;

  char *strs = new char[run_mode.length() + 1];
  strcpy(strs, run_mode.c_str());
  const char *delim = ",";
  char *p = strtok(strs, delim);
  while (p) {
    string s = p;        // transform substr to string
    modes.push_back(s);  // save to result array
    p = strtok(NULL, delim);
  }

  std::map<string, string> config = {
      {"device_id", "0,2,4,6"},
      {"rank_table_file", "hccl from csa/paas"},
      {DDK_VERSION_FLAG, "1.60.T17.B830"},
      {"ge.opsProtoLibPath", "/usr/local/HiAI/runtime/ops/op_proto/built-in/libopsproto.so"}};
  if (train_flag == "infer")
    config.insert(pair<string, string>("ge.graphRunMode", "0"));
  else if (train_flag == "train")
    config.insert(pair<string, string>("ge.graphRunMode", "1"));
  else
    std::cout << "GeInitialize give the error param" << std::endl;

  for (int i = 0; i < modes.size(); i++) {
    if (modes[i] == "fe") {
      config.insert(pair<string, string>("ge.feFlag", "1"));
      if (config.find("ge.soLoadPath") != config.end()) {
        config["ge.soLoadPath"] =
            "/usr/local/HiAI/runtime/lib64/plugin/opskernel/libfe.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/"
            "libaicpu_plugin.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libge_local_engine.so:/usr/local/HiAI/"
            "runtime/lib64/plugin/opskernel/librts_engine.so";
      } else {
        config.insert(pair<string, string>(
            "ge.soLoadPath",
            "/usr/local/HiAI/runtime/lib64/plugin/opskernel/libfe.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/"
            "libge_local_engine.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/librts_engine.so"));
      }
    } else if (modes[i] == "aicpu") {
      config.insert(pair<string, string>("ge.aicpuFlag", "1"));
      if (config.find("ge.soLoadPath") != config.end()) {
        config["ge.soLoadPath"] =
            "/usr/local/HiAI/runtime/lib64/plugin/opskernel/libfe.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/"
            "libaicpu_plugin.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libge_local_engine.so:/usr/local/HiAI/"
            "runtime/lib64/plugin/opskernel/librts_engine.so";
      } else {
        config.insert(pair<string, string>(
            "ge.soLoadPath",
            "/usr/local/HiAI/runtime/lib64/plugin/opskernel/libaicpu_plugin.so:/usr/local/HiAI/runtime/lib64/plugin/"
            "opskernel/libge_local_engine.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/librts_engine.so"));
      }
    } else if (modes[i] == "bert" || modes[i] == "deeplabv3" || modes[i] == "mobilenetv2" ||
               modes[i] == "single_path_nas" || modes[i] == "ssd") {
      config.insert(pair<string, string>(TBE_PLUGIN_PATH_FLAG, "/usr/local/HiAI/runtime/lib64/tbe_plugin/" + modes[i]));
    } else if (modes[i] == "plugin") {

    } else
      std::cout << "GeInitialize give the error param" << std::endl;
  }
  ret = ge::GEInitialize(config);

  std::cout << "GEInitialize_ret is " << ret << std::endl;

  return ret;
}

ge::Status GEFinalize_api() {
  ge::Status ret = ge::GEFinalize();
  std::cout << "GEFinalize ret is " << ret << std::endl;

  return ret;
}

/// set train_flag
/// if run_mode_path is "fe" remain FE process; "fe,plugin" is FE and TBE plugin process
/// "aicpu" is open aicpu plugin
int RunGraph_initData(Graph &graph, string op_name, map<string, std::vector<int64_t>> attr_test, string train_flag,
                      string run_mode_path) {
  std::map<string, string> options = {{RUN_FLAG, "1"}};
  uint32_t graph_id = 0;

  ge::Status ret = GEInitialize_api_new(train_flag, run_mode_path);
  EXPECT_EQ(ret, ge::SUCCESS);

  ge::Session *session = new Session(options);
  ASSERT_TRUE(session != NULL);

  std::vector<Tensor> input;
  if (attr_test.find("input1") != attr_test.end()) {
    Tensor input_tensor = genTensor(attr_test["input1"]);
    input.push_back(input_tensor);
  }
  if (attr_test.find("input2") != attr_test.end()) {
    Tensor input_tensor = genTensor(attr_test["input2"]);
    input.push_back(input_tensor);
  }
  if (attr_test.find("input3") != attr_test.end()) {
    Tensor input_tensor = genTensor(attr_test["input3"]);
    input.push_back(input_tensor);
  }
  std::vector<Tensor> output;

  ret = session->AddGraph(graph_id, graph);
  EXPECT_EQ(ret, ge::SUCCESS);
  if (train_flag == "1") {
    setenv("GE_TRAIN", "1", true);
    ret = session->RunGraph(graph_id, input, output);
    setenv("GE_TRAIN", "0", true);
  } else {
    ret = session->RunGraph(graph_id, input, output);
  }
  delete session;
  GEFinalize_api();

  if (ret != ge::SUCCESS) {
    std::cout << " run graph failed" << std::endl;
    return -1;
  } else {
    return 0;
  }
}

ge::Status session_add_and_run_graph(ge::Session *session, uint32_t graph_id, Graph &graph, std::vector<Tensor> inputs,
                                     std::vector<Tensor> &outputs) {
  ge::Status ret = session->AddGraph(graph_id, graph);
  EXPECT_EQ(ret, ge::SUCCESS);
  ret = session->RunGraph(graph_id, inputs, outputs);

  return ret;
}

ge::Session *create_session() {
  // Init session
  std::map<string, string> options = {{"a", "b"}, {TRAIN_FLAG, "1"}};
  ge::Session *session = new Session(options);
  ASSERT_TRUE(session != NULL);

  return session;
}

ge::Session *create_aipp_session() {
  // Init session
  std::map<string, string> options = {{"a", "b"}, {TRAIN_FLAG, "1"}, {"ge.insertOpFile", "/root/host/ge/aipp.cfg"}};
  ge::Session *session = new Session(options);
  ASSERT_TRUE(session != NULL);

  return session;
}

int buildCheckPointGraph(Graph &graph, map<string, TensorDesc> variables) {
  std::vector<Operator> inputs{};
  std::vector<Operator> outputs{};

  for (map<string, TensorDesc>::iterator it = variables.begin(); it != variables.end(); ++it) {
    auto var = op::Variable(string(it->first));
    var.update_output_desc_y(it->second);
    inputs.push_back(var);
    graph.AddOp(var);
  }

  auto save = op::Save().create_dynamic_input_tensors(inputs.size());
  for (int i = 0; i < inputs.size(); i++) {
    save.set_dynamic_input_tensors(i, inputs[i]);
  }

  graph.SetInputs(inputs).SetOutputs(outputs);
  return 0;
}

int buildInitGraph(Graph &graph, std::vector<TensorDesc> desc_var, std::vector<std::string> name_var,
                   std::vector<float> values_var) {
  std::vector<Operator> inputs{};
  std::vector<Operator> outputs{};

  for (int i = 0; i < desc_var.size(); i++) {
    desc_var[i].SetRealDimCnt(desc_var[i].GetShape().GetDimNum());
    auto tensor_data = genTensor_withVaule(desc_var[i].GetShape().GetDims(), values_var[i]);
    auto var_constant = op::Constant().set_attr_value(tensor_data);
    var_constant.update_output_desc_y(desc_var[i]);

    auto var_init = op::Variable(string(name_var[i]));
    var_init.update_output_desc_y(desc_var[i]);
    auto var_assign = op::Assign().set_input_ref(var_init).set_input_value(var_constant);
    inputs.push_back(var_init);
  }
  graph.SetInputs(inputs).SetOutputs(outputs);
  return 0;
}

int buildInitGraph_other_dataType(Graph &graph, std::vector<TensorDesc> desc_var, std::vector<std::string> name_var) {
  std::vector<Operator> inputs{};
  std::vector<Operator> outputs{};

  for (int i = 0; i < desc_var.size(); i++) {
    desc_var[i].SetRealDimCnt(desc_var[i].GetShape().GetDimNum());
    auto tensor_data = genTensor(desc_var[i].GetShape().GetDims(), desc_var[i].GetFormat(), desc_var[i].GetDataType());
    auto var_constant = op::Constant().set_attr_value(tensor_data);
    var_constant.update_output_desc_y(desc_var[i]);

    auto var_init = op::Variable(string(name_var[i]));
    var_init.update_output_desc_y(desc_var[i]);
    auto var_assign = op::Assign().set_input_ref(var_init).set_input_value(var_constant);
    inputs.push_back(var_init);

    graph.AddOp(var_constant);
    graph.AddOp(var_init);
    graph.AddOp(var_assign);
  }
  graph.SetInputs(inputs).SetOutputs(outputs);
  return 0;
}

bool build_multi_input_multi_output_graph(Graph &graph) {
  auto data1 = op::Data("Data1").set_attr_index(0);
  auto data2 = op::Data("Data2").set_attr_index(1);

  vector<uint64_t> dim_info;

  auto relu1 = op::Relu("Relu1").set_input_x(data1);
  auto relu2 = op::Relu("Relu2").set_input_x(data2);

  auto eltwise = op::Eltwise("Eltwise")
                     .create_dynamic_input___input(2)
                     .set_dynamic_input___input(0, relu1)
                     .set_dynamic_input___input(1, relu2)
                     .set_attr_mode(1)
                     .set_attr_coeff({1, 1});

  auto eltwise1 = op::Eltwise("Eltwise1")
                      .create_dynamic_input___input(2)
                      .set_dynamic_input___input(0, eltwise)
                      .set_dynamic_input___input(1, eltwise)
                      .set_attr_mode(1)
                      .set_attr_coeff({1, 1});

  auto eltwise2 = op::Eltwise("Eltwise2")
                      .create_dynamic_input___input(2)
                      .set_dynamic_input___input(0, eltwise)
                      .set_dynamic_input___input(1, eltwise)
                      .set_attr_mode(1)
                      .set_attr_coeff({1, 1});

  std::vector<Operator> inputs{data1, data2};
  std::vector<Operator> outputs{eltwise1, eltwise2};
  graph.SetInputs(inputs).SetOutputs(outputs);
  return true;
}

void build_big_graph(Graph &graph, map<string, std::vector<int64_t>> attr) {
  auto data = op::Data("Data").set_attr_index(0);
  auto weight = op::Const("weight1").set_attr_value(genTensor(attr["weight"]));
  vector<int64_t> weight_shape(attr["weight"].begin(), attr["weight"].end());
  TensorDesc weight_desc(ge::Shape(weight_shape), FORMAT_NCHW, DT_FLOAT);
  weight.update_output_desc_y(weight_desc);
  auto conv_1 = op::Conv2D("conv1").set_input_x(data).set_input_filter(weight);

  auto conv_2 = op::Conv2D("conv2").set_input_x(conv_1).set_input_filter(weight);
  auto conv_3 = op::Conv2D("conv3").set_input_x(conv_2).set_input_filter(weight);
  auto conv_4 = op::Conv2D("conv4").set_input_x(conv_3).set_input_filter(weight);
  auto conv_5 = op::Conv2D("conv5").set_input_x(conv_4).set_input_filter(weight);
  auto conv_6 = op::Conv2D("conv6").set_input_x(conv_5).set_input_filter(weight);
  auto conv_7 = op::Conv2D("conv7").set_input_x(conv_6).set_input_filter(weight);
  auto conv_8 = op::Conv2D("conv8").set_input_x(conv_7).set_input_filter(weight);
  auto conv_9 = op::Conv2D("conv9").set_input_x(conv_8).set_input_filter(weight);
  auto conv_10 = op::Conv2D("conv10").set_input_x(conv_9).set_input_filter(weight);
  auto conv_11 = op::Conv2D("conv11").set_input_x(conv_10).set_input_filter(weight);
  auto conv_12 = op::Conv2D("conv12").set_input_x(conv_11).set_input_filter(weight);
  auto conv_13 = op::Conv2D("conv13").set_input_x(conv_12).set_input_filter(weight);
  auto conv_14 = op::Conv2D("conv14").set_input_x(conv_13).set_input_filter(weight);
  auto conv_15 = op::Conv2D("conv15").set_input_x(conv_14).set_input_filter(weight);
  auto conv_16 = op::Conv2D("conv16").set_input_x(conv_15).set_input_filter(weight);
  auto conv_17 = op::Conv2D("conv17").set_input_x(conv_16).set_input_filter(weight);
  auto conv_18 = op::Conv2D("conv18").set_input_x(conv_17).set_input_filter(weight);
  auto conv_19 = op::Conv2D("conv19").set_input_x(conv_18).set_input_filter(weight);
  auto conv_20 = op::Conv2D("conv20").set_input_x(conv_19).set_input_filter(weight);
  auto conv_21 = op::Conv2D("conv21").set_input_x(conv_20).set_input_filter(weight);
  auto conv_22 = op::Conv2D("conv22").set_input_x(conv_21).set_input_filter(weight);
  auto conv_23 = op::Conv2D("conv23").set_input_x(conv_22).set_input_filter(weight);
  auto conv_24 = op::Conv2D("conv24").set_input_x(conv_23).set_input_filter(weight);
  auto conv_25 = op::Conv2D("conv25").set_input_x(conv_24).set_input_filter(weight);
  auto conv_26 = op::Conv2D("conv26").set_input_x(conv_25).set_input_filter(weight);
  auto conv_27 = op::Conv2D("conv27").set_input_x(conv_26).set_input_filter(weight);
  auto conv_28 = op::Conv2D("conv28").set_input_x(conv_27).set_input_filter(weight);
  auto conv_29 = op::Conv2D("conv29").set_input_x(conv_28).set_input_filter(weight);
  auto conv_30 = op::Conv2D("conv30").set_input_x(conv_29).set_input_filter(weight);
  auto conv_31 = op::Conv2D("conv31").set_input_x(conv_30).set_input_filter(weight);
  auto conv_32 = op::Conv2D("conv32").set_input_x(conv_31).set_input_filter(weight);
  auto conv_33 = op::Conv2D("conv33").set_input_x(conv_32).set_input_filter(weight);
  auto conv_34 = op::Conv2D("conv34").set_input_x(conv_33).set_input_filter(weight);
  auto conv_35 = op::Conv2D("conv35").set_input_x(conv_34).set_input_filter(weight);
  auto conv_36 = op::Conv2D("conv36").set_input_x(conv_35).set_input_filter(weight);
  auto conv_37 = op::Conv2D("conv37").set_input_x(conv_36).set_input_filter(weight);
  auto conv_38 = op::Conv2D("conv38").set_input_x(conv_37).set_input_filter(weight);
  auto conv_39 = op::Conv2D("conv39").set_input_x(conv_38).set_input_filter(weight);
  auto conv_40 = op::Conv2D("conv40").set_input_x(conv_39).set_input_filter(weight);
  auto conv_41 = op::Conv2D("conv41").set_input_x(conv_40).set_input_filter(weight);
  auto conv_42 = op::Conv2D("conv42").set_input_x(conv_41).set_input_filter(weight);
  auto conv_43 = op::Conv2D("conv43").set_input_x(conv_42).set_input_filter(weight);
  auto conv_44 = op::Conv2D("conv44").set_input_x(conv_43).set_input_filter(weight);
  auto conv_45 = op::Conv2D("conv45").set_input_x(conv_44).set_input_filter(weight);
  auto conv_46 = op::Conv2D("conv46").set_input_x(conv_45).set_input_filter(weight);
  auto conv_47 = op::Conv2D("conv47").set_input_x(conv_46).set_input_filter(weight);
  auto conv_48 = op::Conv2D("conv48").set_input_x(conv_47).set_input_filter(weight);
  auto conv_49 = op::Conv2D("conv49").set_input_x(conv_48).set_input_filter(weight);
  auto conv_50 = op::Conv2D("conv50").set_input_x(conv_49).set_input_filter(weight);
  auto conv_51 = op::Conv2D("conv51").set_input_x(conv_50).set_input_filter(weight);
  auto conv_52 = op::Conv2D("conv52").set_input_x(conv_51).set_input_filter(weight);
  auto conv_53 = op::Conv2D("conv53").set_input_x(conv_52).set_input_filter(weight);
  auto conv_54 = op::Conv2D("conv54").set_input_x(conv_53).set_input_filter(weight);
  auto conv_55 = op::Conv2D("conv55").set_input_x(conv_54).set_input_filter(weight);
  auto conv_56 = op::Conv2D("conv56").set_input_x(conv_55).set_input_filter(weight);
  auto conv_57 = op::Conv2D("conv57").set_input_x(conv_56).set_input_filter(weight);
  auto conv_58 = op::Conv2D("conv58").set_input_x(conv_57).set_input_filter(weight);
  auto conv_59 = op::Conv2D("conv59").set_input_x(conv_58).set_input_filter(weight);
  auto conv_60 = op::Conv2D("conv60").set_input_x(conv_59).set_input_filter(weight);
  auto conv_61 = op::Conv2D("conv61").set_input_x(conv_60).set_input_filter(weight);
  auto conv_62 = op::Conv2D("conv62").set_input_x(conv_61).set_input_filter(weight);
  auto conv_63 = op::Conv2D("conv63").set_input_x(conv_62).set_input_filter(weight);
  auto conv_64 = op::Conv2D("conv64").set_input_x(conv_63).set_input_filter(weight);
  auto conv_65 = op::Conv2D("conv65").set_input_x(conv_64).set_input_filter(weight);
  auto conv_66 = op::Conv2D("conv66").set_input_x(conv_65).set_input_filter(weight);
  auto conv_67 = op::Conv2D("conv67").set_input_x(conv_66).set_input_filter(weight);
  auto conv_68 = op::Conv2D("conv68").set_input_x(conv_67).set_input_filter(weight);
  auto conv_69 = op::Conv2D("conv69").set_input_x(conv_68).set_input_filter(weight);
  auto conv_70 = op::Conv2D("conv70").set_input_x(conv_69).set_input_filter(weight);
  auto conv_71 = op::Conv2D("conv71").set_input_x(conv_70).set_input_filter(weight);
  auto conv_72 = op::Conv2D("conv72").set_input_x(conv_71).set_input_filter(weight);
  auto conv_73 = op::Conv2D("conv73").set_input_x(conv_72).set_input_filter(weight);
  auto conv_74 = op::Conv2D("conv74").set_input_x(conv_73).set_input_filter(weight);
  auto conv_75 = op::Conv2D("conv75").set_input_x(conv_74).set_input_filter(weight);
  auto conv_76 = op::Conv2D("conv76").set_input_x(conv_75).set_input_filter(weight);
  auto conv_77 = op::Conv2D("conv77").set_input_x(conv_76).set_input_filter(weight);
  auto conv_78 = op::Conv2D("conv78").set_input_x(conv_77).set_input_filter(weight);
  auto conv_79 = op::Conv2D("conv79").set_input_x(conv_78).set_input_filter(weight);
  auto conv_80 = op::Conv2D("conv80").set_input_x(conv_79).set_input_filter(weight);
  auto conv_81 = op::Conv2D("conv81").set_input_x(conv_80).set_input_filter(weight);
  auto conv_82 = op::Conv2D("conv82").set_input_x(conv_81).set_input_filter(weight);
  auto conv_83 = op::Conv2D("conv83").set_input_x(conv_82).set_input_filter(weight);
  auto conv_84 = op::Conv2D("conv84").set_input_x(conv_83).set_input_filter(weight);
  auto conv_85 = op::Conv2D("conv85").set_input_x(conv_84).set_input_filter(weight);
  auto conv_86 = op::Conv2D("conv86").set_input_x(conv_85).set_input_filter(weight);
  auto conv_87 = op::Conv2D("conv87").set_input_x(conv_86).set_input_filter(weight);
  auto conv_88 = op::Conv2D("conv88").set_input_x(conv_87).set_input_filter(weight);
  auto conv_89 = op::Conv2D("conv89").set_input_x(conv_88).set_input_filter(weight);
  auto conv_90 = op::Conv2D("conv90").set_input_x(conv_89).set_input_filter(weight);
  auto conv_91 = op::Conv2D("conv91").set_input_x(conv_80).set_input_filter(weight);
  auto conv_92 = op::Conv2D("conv92").set_input_x(conv_91).set_input_filter(weight);
  auto conv_93 = op::Conv2D("conv93").set_input_x(conv_92).set_input_filter(weight);
  auto conv_94 = op::Conv2D("conv94").set_input_x(conv_93).set_input_filter(weight);
  auto conv_95 = op::Conv2D("conv95").set_input_x(conv_94).set_input_filter(weight);
  auto conv_96 = op::Conv2D("conv96").set_input_x(conv_95).set_input_filter(weight);
  auto conv_97 = op::Conv2D("conv97").set_input_x(conv_96).set_input_filter(weight);
  auto conv_98 = op::Conv2D("conv98").set_input_x(conv_97).set_input_filter(weight);
  auto conv_99 = op::Conv2D("conv99").set_input_x(conv_98).set_input_filter(weight);
  auto conv_100 = op::Conv2D("conv100").set_input_x(conv_99).set_input_filter(weight);
  auto conv_101 = op::Conv2D("conv101").set_input_x(conv_100).set_input_filter(weight);
  auto conv_102 = op::Conv2D("conv102").set_input_x(conv_101).set_input_filter(weight);
  auto conv_103 = op::Conv2D("conv103").set_input_x(conv_102).set_input_filter(weight);
  auto conv_104 = op::Conv2D("conv104").set_input_x(conv_103).set_input_filter(weight);
  auto conv_105 = op::Conv2D("conv105").set_input_x(conv_104).set_input_filter(weight);
  auto conv_106 = op::Conv2D("conv106").set_input_x(conv_105).set_input_filter(weight);
  auto conv_107 = op::Conv2D("conv107").set_input_x(conv_106).set_input_filter(weight);
  auto conv_108 = op::Conv2D("conv108").set_input_x(conv_107).set_input_filter(weight);
  auto conv_109 = op::Conv2D("conv109").set_input_x(conv_108).set_input_filter(weight);
  auto conv_110 = op::Conv2D("conv110").set_input_x(conv_109).set_input_filter(weight);
  auto conv_111 = op::Conv2D("conv111").set_input_x(conv_110).set_input_filter(weight);
  auto conv_112 = op::Conv2D("conv112").set_input_x(conv_111).set_input_filter(weight);
  auto conv_113 = op::Conv2D("conv113").set_input_x(conv_112).set_input_filter(weight);
  auto conv_114 = op::Conv2D("conv114").set_input_x(conv_113).set_input_filter(weight);
  auto conv_115 = op::Conv2D("conv115").set_input_x(conv_114).set_input_filter(weight);
  auto conv_116 = op::Conv2D("conv116").set_input_x(conv_115).set_input_filter(weight);
  auto conv_117 = op::Conv2D("conv117").set_input_x(conv_116).set_input_filter(weight);
  auto conv_118 = op::Conv2D("conv118").set_input_x(conv_117).set_input_filter(weight);
  auto conv_119 = op::Conv2D("conv119").set_input_x(conv_118).set_input_filter(weight);
  auto conv_120 = op::Conv2D("conv120").set_input_x(conv_119).set_input_filter(weight);
  auto conv_121 = op::Conv2D("conv121").set_input_x(conv_120).set_input_filter(weight);
  auto conv_122 = op::Conv2D("conv122").set_input_x(conv_121).set_input_filter(weight);
  auto conv_123 = op::Conv2D("conv123").set_input_x(conv_122).set_input_filter(weight);
  auto conv_124 = op::Conv2D("conv124").set_input_x(conv_123).set_input_filter(weight);
  auto conv_125 = op::Conv2D("conv125").set_input_x(conv_124).set_input_filter(weight);
  auto conv_126 = op::Conv2D("conv126").set_input_x(conv_125).set_input_filter(weight);
  auto conv_127 = op::Conv2D("conv127").set_input_x(conv_126).set_input_filter(weight);
  auto conv_128 = op::Conv2D("conv128").set_input_x(conv_127).set_input_filter(weight);
  auto conv_129 = op::Conv2D("conv129").set_input_x(conv_128).set_input_filter(weight);
  auto conv_130 = op::Conv2D("conv130").set_input_x(conv_129).set_input_filter(weight);

  std::vector<Operator> inputs{data};
  std::vector<Operator> outputs{conv_130};
  graph.SetInputs(inputs).SetOutputs(outputs);
}

int GetDatTypeSize(DataType dt) {
  int dailation = 1;
  if (dt == ge::DT_FLOAT)
    dailation = 4;
  else if (dt == ge::DT_FLOAT16)
    dailation = 2;
  else if (dt == ge::DT_INT16)
    dailation = 2;
  else if (dt == ge::DT_UINT16)
    dailation = 2;
  else if (dt == ge::DT_INT32)
    dailation = 4;
  else if (dt == ge::DT_UINT32)
    dailation = 4;
  else if (dt == ge::DT_INT64)
    dailation = 8;
  else if (dt == ge::DT_UINT64)
    dailation = 8;
  else if (dt == ge::DT_INT8)
    dailation = 1;

  return dailation;
}

int buildConvGraph_new(Graph &graph, std::vector<TensorDesc> desc_var, std::vector<std::string> name_var, int flag,
                       Format format) {
  auto data_x_shape = op::Data("xShape").set_attr_index(0);
  auto var = op::Variable(name_var[0]);
  auto var1 = op::Variable(name_var[1]);    //add one seat of ApplyMomentum()
  auto label1 = op::Variable(name_var[2]);  //add one seat of ApplyMomentum()
  auto conv2dgrad = op::Conv2DBackpropFilterD("output_1");
  auto test2 = op::ApplyMomentum();

  var.update_output_desc_y(desc_var[0]);
  var1.update_output_desc_y(desc_var[1]);
  label1.update_output_desc_y(desc_var[2]);

  graph.AddOp(var);
  graph.AddOp(var1);
  graph.AddOp(label1);

  auto conv2d = op::Conv2D().set_input_x(data_x_shape).set_input_filter(var).set_attr_strides({1, 1, 1, 1});
  update_op_format(conv2d, format);
  ge::TensorDesc tensor_desc_w = conv2d.GetInputDesc("filter");
  tensor_desc_w.SetFormat(format);
  conv2d.UpdateInputDesc("filter", tensor_desc_w);

  if (flag >= 1) {
    conv2dgrad.set_input_x(data_x_shape)
        .set_attr_filter_sizes(desc_var[0].GetShape().GetDims())
        .set_input_out_backprop(conv2d)
        .set_attr_strides({1, 1})
        .set_attr_pads({0, 0, 0, 0});
    update_op_format(conv2dgrad, format);
    graph.AddOp(conv2dgrad);
  }
  if (flag >= 2) {
    // set conv2dgrad var
    test2.set_input_accum(var1)
        .set_input_grad(conv2dgrad)
        .set_input_lr(label1)
        .set_input_momentum(label1)
        .set_input_var(var);
    graph.AddOp(test2);
  }

  std::vector<Operator> inputs{data_x_shape};  // set all val
  std::vector<Operator> outputs{conv2d};
  graph.SetInputs(inputs).SetOutputs(outputs);
  graph.AddOp(conv2d);

  return 0;
}

/// load bin data_fail
/// input_path: path of bin data_file
/// shapes: the shape of Tensor
/// ft: the format of Tensor
/// dt: the dataType of Tensor
Tensor load_variable_input_data(string input_path, std::vector<int64_t> shapes, Format ft, DataType dt) {
  vector<uint64_t> dim_info1;

  uint8_t *input_data = (uint8_t *)readTestDataFile(input_path, dim_info1);  // common.h
  TensorDesc input_tensor_desc = TensorDesc(ge::Shape(shapes), ft, dt);
  input_tensor_desc.SetRealDimCnt(shapes.size());
  Tensor input_tensor = Tensor(input_tensor_desc, input_data, GetDatTypeSize(dt) * dim_info1[dim_info1[0] + 1]);
  return input_tensor;
}
