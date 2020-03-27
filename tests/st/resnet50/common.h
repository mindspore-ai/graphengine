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

#ifndef GE_COMMON_H
#define GE_COMMON_H
#include "common/ge_inner_error_codes.h"
#include "utils/tensor_utils.h"

#define MY_USER_GE_LOGI(...) GE_LOG_INFO(1, __VA_ARGS__)
#define MY_USER_GE_LOGW(...) GE_LOG_WARN(1, __VA_ARGS__)
#define MY_USER_GE_LOGE(...) GE_LOG_ERROR(1, 3, __VA_ARGS__)

#ifndef USER_GE_LOGI
#define USER_GE_LOGI MY_USER_GE_LOGI
#endif  // USER_GE_LOGI

#ifndef USER_GE_LOGW
#define USER_GE_LOGW MY_USER_GE_LOGW
#endif  // USER_GE_LOGW

#ifndef USER_GE_LOGE
#define USER_GE_LOGE MY_USER_GE_LOGE
#endif  // USER_GE_LOGE

/// train_flag is 0 when infer, train_flag is 1 when train.this param is set for RunGranph_readData() and
/// RunGraph_initData()
#define TRAIN_FLAG_INFER "infer"
#define TRAIN_FLAG_TRAIN "train"

#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "ge_api.h"
#include "graph.h"
#include "ptest.h"
#include "ops/all_ops.h"
using namespace std;
using namespace ge;

// read bin file and compile result
void update_op_format(Operator ops, Format format = ge::FORMAT_NCHW);
void getDimInfo(FILE *fp, std::vector<uint64_t> &dim_info);
void *readTestDataFile(std::string infile, std::vector<uint64_t> &dim_info);
void *readUint8TestDataFile(std::string infile, int size);
bool allclose(float *a, float *b, uint64_t count, float rtol, float atol);
bool compFp32WithTData(float *actualOutputData, std::string expectedDataFile, float rtol, float atol);
Tensor load_variable_input_data(string inputpath, std::vector<int64_t> shapes, Format ft = ge::FORMAT_NCHW,
                                DataType dt = ge::DT_FLOAT);
// constructor Tensor
int GetDatTypeSize(DataType dt);
ge::Tensor genTensor(std::vector<int64_t> Tensorshape, Format format = ge::FORMAT_NCHW, DataType dt = ge::DT_FLOAT);
ge::Tensor genTensor_withVaule(std::vector<int64_t> Tensorshape, float value = 1);
Tensor genTesnor_Shape_as_data(std::vector<int64_t> Tensorshape);
// Init GE
ge::Status GEInitialize_api(string train_flag = "0", string run_mode_path = "0");
ge::Status GEInitialize_api_new(string train_flag = "infer", string run_mode = "fe");
ge::Status GEFinalize_api();
// constructor session and build graph
ge::Session *create_aipp_session();
ge::Session *create_session();
ge::Status session_add_and_run_graph(ge::Session *session, uint32_t graphId, Graph &graph, std::vector<Tensor> inputs,
                                     std::vector<Tensor> &outputs);

// common interface for infer
int RunGraph_initData(Graph &graph, string op_name, map<string, std::vector<int64_t>> attr_Test,
                      string train_flag = "infer", string run_mode_path = "fe");
void Inputs_load_Data(string op_name, std::vector<Tensor> &input, map<string, std::vector<int64_t>> attr_Test,
                      Format format = ge::FORMAT_NCHW, DataType dt = ge::DT_FLOAT);
bool comparaData(std::vector<Tensor> &output, string op_name, map<string, std::vector<int64_t>> attr_Test);
int RunGraph_readData(Graph &graph, string op_name, map<string, std::vector<int64_t>> attr_Test,
                      string train_flag = "infer", string run_mode_path = "fe", Format format = ge::FORMAT_NCHW,
                      DataType dt = ge::DT_FLOAT);

// common interface for train
int buildCheckPointGraph(Graph &graph, map<string, TensorDesc> variables);
int buildInitGraph(Graph &graph, std::vector<TensorDesc> desc_var, std::vector<std::string> name_var,
                   std::vector<float> values_var);
int buildInitGraph_other_dataType(Graph &graph, std::vector<TensorDesc> desc_var, std::vector<std::string> name_var);

bool build_multi_input_multi_output_graph(Graph &graph);
void build_big_graph(Graph &graph, map<string, std::vector<int64_t>> Attr);
int buildConvGraph_new(Graph &graph, std::vector<TensorDesc> desc_var, std::vector<std::string> name_var, int flag = 2);

#endif  // GE_COMMON_H
