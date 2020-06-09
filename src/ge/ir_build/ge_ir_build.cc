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

#include "external/ge/ge_ir_build.h"

#include <vector>
#include "common/auth/file_saver.h"
#include "external/register/register_types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/string_util.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "framework/omg/omg_inner_types.h"
#include "framework/omg/omg_inner_types.h"
#include "ge/ge_api_types.h"
#include "generator/ge_generator.h"
#include "graph/compute_graph.h"
#include "graph/ge_tensor.h"
#include "graph/utils/type_utils.h"
#include "init/gelib.h"
#include "ir_build/atc_ir_common.h"
#include "model/ge_model.h"

using domi::GetContext;
using std::string;
using namespace std;

namespace ge {

static std::map<std::string, domi::domiTensorFormat_t> input_format_str_to_geformat = {
  {"ND", domi::DOMI_TENSOR_ND},     {"NCHW", domi::DOMI_TENSOR_NCHW},       {"NHWC", domi::DOMI_TENSOR_NHWC},
  {"CHWN", domi::DOMI_TENSOR_CHWN}, {"NC1HWC0", domi::DOMI_TENSOR_NC1HWC0}, {"NHWC1C0", domi::DOMI_TENSOR_NHWC1C0},
};
const std::string IR_OPTION_TARGET = "target";
const std::string IR_OPTION_MODE = "mode";
const std::string IR_OP_CONF_DELIMITER = ":";

graphStatus aclgrphBuildInitialize(std::map<std::string, std::string> global_options) {
  GELOGD("Enter aclgrphInitialize start!");
  std::shared_ptr<ge::GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    GELOGI("aclgrphInitialize start!");
    auto ret = ge::GELib::Initialize(global_options);
    if (ret != ge::SUCCESS) {
      GELOGE(ret, "GE initialize failed!");
      return GRAPH_FAILED;
    }
  }
  GELOGW("gelib has been initialized!");
  return GRAPH_SUCCESS;
}

void aclgrphBuildFinalize() {
  if (ge::GELib::GetInstance() != nullptr && ge::GELib::GetInstance()->InitFlag()) {
    (void)ge::GELib::GetInstance()->Finalize();
    return;
  }
  GELOGW("[Notice] gelib has not been initialized!do nothing!");
}

class Impl {
 public:
  Impl() {
    GetContext().format = domi::DOMI_TENSOR_ND;
    GetContext().input_nodes_format_map.clear();
    GetContext().output_formats.clear();
    GetContext().user_input_dims.clear();
    GetContext().input_dims.clear();
    GetContext().op_conf_map.clear();
    GetContext().out_nodes_map.clear();
    GetContext().user_out_nodes.clear();
    GetContext().net_format = domi::DOMI_TENSOR_RESERVED;
    GetContext().type = domi::FMK_TYPE_RESERVED;
    GetContext().run_mode = ONLY_PRE_CHECK;
    GetContext().train_flag = false;
    GetContext().fp16_high_precision = HIGH_PRECISION_DEFAULT;
    GetContext().output_type.clear();
    GetContext().net_name.clear();
    GetContext().is_dynamic_input = false;
    GetContext().dynamic_batch_size.clear();
    GetContext().dynamic_image_size.clear();
  };
  ~Impl() { (void)generator_.Finalize(); };
  graphStatus CheckOptions(const std::map<std::string, std::string> &options);
  graphStatus CreateInputsForIRBuild(const ge::Graph &graph, vector<ge::GeTensor> &inputs);
  graphStatus Init(const std::map<std::string, std::string> &options);
  graphStatus BuildModel(const Graph &graph, const std::map<std::string, std::string> &options,
                         ModelBufferData &ge_models);
  graphStatus InitDomiOmgContext(const string &input_shape, const string &input_format, const string &net_format,
                                 bool is_dynamic_input);

 public:
  ge::GeGenerator generator_;
  std::map<std::string, std::string> options_;
  bool is_dynamic_input_ = false;
};

graphStatus Impl::CheckOptions(const std::map<std::string, std::string> &options) {
  for (auto &ele : options) {
    auto it = ge::ir_option::ir_builder_suppported_options.find(ele.first);
    if (it == ge::ir_option::ir_builder_suppported_options.end()) {
      GELOGE(GRAPH_PARAM_INVALID, "input options include unsupported option(%s).Please check!", ele.first.c_str());
      return GRAPH_PARAM_INVALID;
    }
    options_.insert(ele);
  }
  return GRAPH_SUCCESS;
}
graphStatus Impl::Init(const std::map<std::string, std::string> &options) {
  // 1. check options
  graphStatus ret = CheckOptions(options);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "user input options is not illegal!Please check!");
    return ret;
  }

  string input_shape = options_.find("input_shape") == options_.end() ? "" : options_["input_shape"];
  string input_format = options_.find("input_format") == options_.end() ? "" : options_["input_format"];
  string net_format = options_.find("net_format") == options_.end() ? "" : options_["net_format"];
  string dynamic_batch_size = options_.find(ge::ir_option::DYNAMIC_BATCH_SIZE) == options_.end()
                                ? ""
                                : options_[ge::ir_option::DYNAMIC_BATCH_SIZE];
  string dynamic_image_size = options_.find(ge::ir_option::DYNAMIC_IMAGE_SIZE) == options_.end()
                                ? ""
                                : options_[ge::ir_option::DYNAMIC_IMAGE_SIZE];

  auto status = CheckDynamicBatchSizeOrImageSizeParamValid(dynamic_batch_size, dynamic_image_size, input_shape,
                                                           input_format, is_dynamic_input_);
  if (status != ge::SUCCESS) {
    GELOGE(GRAPH_PARAM_INVALID, "check dynamic batch size or image size failed!");
    return GRAPH_PARAM_INVALID;
  }
  GELOGD("user input dynamic_batch_size:%s dynamic_image_size:%s", dynamic_batch_size.c_str(),
         dynamic_image_size.c_str());
  GetContext().dynamic_batch_size = dynamic_batch_size;
  GetContext().dynamic_image_size = dynamic_image_size;

  // for IR builder.Only support om mode, so here fixed;
  options_.insert(std::pair<string, string>(string(IR_OPTION_MODE), to_string(0)));
  options_.insert(std::pair<string, string>(string(IR_OPTION_TARGET), "mini"));
  options_.insert(std::pair<string, string>(string(ge::RUN_FLAG), to_string(0)));
  options_.insert(std::pair<string, string>(string(ge::TRAIN_FLAG), to_string(0)));
  options_.insert(std::pair<string, string>(string(ge::SAVE_ORIGINAL_MODEL), to_string(0)));

  // 3. init generator with options_
  ret = generator_.Initialize(options_);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "generator Initialize failed!");
    return ret;
  }
  // 4.parse and init Context with input shape format and net format info
  return this->InitDomiOmgContext(input_shape, input_format, net_format, is_dynamic_input_);
}
graphStatus Impl::CreateInputsForIRBuild(const ge::Graph &graph, vector<ge::GeTensor> &inputs) {
  auto compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);
  for (ge::NodePtr &input_node : compute_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(input_node);
    ge::OpDescPtr op = input_node->GetOpDesc();
    GE_CHECK_NOTNULL(op);
    if (op->GetType() == DATA) {
      GELOGI("Data op inputDesc size is: %zu", op->GetAllInputsDesc().size());
      ge::GeTensorDesc tensor = op->GetInputDesc(0);
      string data_op_name = op->GetName();
      GELOGI("Data op name is: %s", data_op_name.c_str());
      ge::GeShape data_shape;
      auto iter = GetContext().input_dims.find(data_op_name);
      if (iter != GetContext().input_dims.end()) {
        data_shape = ge::GeShape(iter->second);
        GELOGI("Data op get shape from Context.");
      } else {
        data_shape = tensor.GetShape();
        GELOGI("Data op get shape from InputDesc in geir graph.");
      }

      ge::DataType data_type = tensor.GetDataType();
      string data_type_str = ge::TypeUtils::DataTypeToSerialString(data_type);
      GELOGI("Data op get data type:%s from InputDesc in ge ir graph.", data_type_str.c_str());

      ge::GeTensor inputTensor;
      ge::GeTensorDesc desc(data_shape, ge::Format(GetContext().format), data_type);
      inputTensor.SetTensorDesc(desc);
      inputs.push_back(inputTensor);
    }
  }
  GELOGD("CreateInputsForIRBuild, inputs size is: %zu", inputs.size());
  return GRAPH_SUCCESS;
}
graphStatus Impl::BuildModel(const Graph &graph, const std::map<std::string, std::string> &options,
                             ModelBufferData &model) {
  // 1. init GeGenerator with user optios
  graphStatus ret = Init(options);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "Build IR model Init Failed!");
    return ret;
  }

  // 2. construct input
  std::vector<GeTensor> inputs;
  if (!GetContext().is_dynamic_input) {  // if dynamic input , no need to creat inputs
    ret = CreateInputsForIRBuild(graph, inputs);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(ret, "CreateInputsForIRBuild failed!");
      return ret;
    }
  }

  // 3. build IR model
  ret = generator_.GenerateOnlineModel(graph, inputs, model);

  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "GenerateOnlineModel failed!");
    return ret;
  }

  return GRAPH_SUCCESS;
}
graphStatus Impl::InitDomiOmgContext(const string &input_shape, const string &input_format, const string &net_format,
                                     bool is_dynamic_input) {
  // Clear omgcontext data first
  GetContext().input_dims.clear();
  GetContext().user_input_dims.clear();
  GetContext().is_dynamic_input = is_dynamic_input;
  // the default value is ND
  GetContext().format = domi::DOMI_TENSOR_ND;
  if (!input_format.empty()) {
    auto iter = input_format_str_to_geformat.find(input_format);
    if (iter != input_format_str_to_geformat.end()) {
      GetContext().format = iter->second;
    } else {
      GELOGE(GRAPH_PARAM_INVALID, "Input format %s not support , expect ND/NCHW/NHWC/CHWN/NC1HWC0/NHWC1C0.",
             input_format.c_str());
      return GRAPH_PARAM_INVALID;
    }
  }
  // Input is empty, do not process
  if (input_shape.empty()) {
    return GRAPH_SUCCESS;
  }

  if (!ParseInputShape(input_shape, GetContext().input_dims, GetContext().user_input_dims, is_dynamic_input)) {
    GELOGE(GRAPH_PARAM_INVALID, "Failed to parse input shape: %s", input_shape.c_str());
    return GRAPH_PARAM_INVALID;
  }
  return GRAPH_SUCCESS;
}

graphStatus aclgrphBuildModel(const ge::Graph &graph, const std::map<std::string, std::string> &build_options,
                              ModelBufferData &model) {
  GELOGD("Enter aclmdlBuildModel process!");
  Impl builder;
  return builder.BuildModel(graph, build_options, model);
}

graphStatus aclgrphSaveModel(const string &output_file, const ModelBufferData &model) {
  GELOGD("Enter aclmdlSaveModel process!");
  if (model.data.get() == nullptr || model.length == 0) {
    GELOGE(GRAPH_PARAM_INVALID, "input model is not illegal");
    return GRAPH_PARAM_INVALID;
  }
  return FileSaver::SaveToFile((output_file + ".om"), reinterpret_cast<void *>(model.data.get()),
                               static_cast<uint32_t>(model.length));
}
}  // namespace ge
