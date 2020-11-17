/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "graph/ge_global_options.h"
#include "init/gelib.h"
#include "ir_build/atc_ir_common.h"
#include "model/ge_model.h"
#include "graph/shape_refiner.h"
#include "graph/opsproto_manager.h"

using std::string;
using namespace std;

namespace ge {
namespace {
const std::string IR_OPTION_TARGET = "target";
const std::string IR_OPTION_MODE = "mode";
const std::string IR_OP_CONF_DELIMITER = ":";
const std::string IR_OPTION_LOG_LEVEL_DEFAULT = "default";
const std::string IR_OPTION_BUFFER_OPTIMIZE_DEFAULT = "l2_optimize";
const std::string IR_OPTION_DISABLE_REUSE_MEMORY_DEFAULT = "0";
const std::string IR_OPTION_ENABLE_COMPRESS_WEIGHT_DEFAULT = "false";
}  // namespace

static graphStatus CheckGlobalOptions(std::map<std::string, std::string> &global_options) {
  // check param disable_reuse_memory
  std::string disable_reuse_memory = global_options.find(ge::ir_option::EXEC_DISABLE_REUSED_MEMORY) ==
                                         global_options.end()
                                         ? IR_OPTION_DISABLE_REUSE_MEMORY_DEFAULT
                                         : global_options[ge::ir_option::EXEC_DISABLE_REUSED_MEMORY];
  GE_CHK_BOOL_EXEC(ge::CheckDisableReuseMemoryParamValid(disable_reuse_memory) == ge::SUCCESS,
      return ge::GRAPH_PARAM_INVALID, "check disable_reuse_memory failed!");
  global_options[ge::ir_option::EXEC_DISABLE_REUSED_MEMORY] = disable_reuse_memory;
  // check buffer_optimize
  std::string buffer_optimize = global_options.find(ge::ir_option::BUFFER_OPTIMIZE) == global_options.end()
                                    ? IR_OPTION_BUFFER_OPTIMIZE_DEFAULT
                                    : global_options[ge::ir_option::BUFFER_OPTIMIZE];
  GE_CHK_BOOL_EXEC(ge::CheckBufferOptimizeParamValid(buffer_optimize) == ge::SUCCESS,
      return ge::GRAPH_PARAM_INVALID, "check buffer optimize failed!");
  global_options[ge::ir_option::BUFFER_OPTIMIZE] = buffer_optimize;
  // check enable_single_stream
  std::string enable_single_stream = global_options.find(ge::ir_option::ENABLE_SINGLE_STREAM) == global_options.end()
                                         ? ""
                                         : global_options[ge::ir_option::ENABLE_SINGLE_STREAM];
  GE_CHK_BOOL_EXEC(ge::CheckEnableSingleStreamParamValid(enable_single_stream) == ge::SUCCESS,
      return ge::GRAPH_PARAM_INVALID, "check enable single stream failed!");
  // check compress_weight
  std::string enable_compress_weight = global_options.find(ge::ir_option::ENABLE_COMPRESS_WEIGHT) ==
                                           global_options.end()
                                           ? IR_OPTION_ENABLE_COMPRESS_WEIGHT_DEFAULT
                                           : global_options[ge::ir_option::ENABLE_COMPRESS_WEIGHT];
  std::string compress_weight_conf = global_options.find(ge::ir_option::COMPRESS_WEIGHT_CONF) == global_options.end()
                                         ? ""
                                         : global_options[ge::ir_option::COMPRESS_WEIGHT_CONF];
  GE_CHK_BOOL_EXEC(ge::CheckCompressWeightParamValid(enable_compress_weight, compress_weight_conf) == ge::SUCCESS,
      return ge::GRAPH_PARAM_INVALID, "check compress weight failed!");
  global_options[ge::ir_option::ENABLE_COMPRESS_WEIGHT] = (enable_compress_weight == "true") ?
                                                     ge::kEnableCompressWeightTrue :
                                                     ge::kEnableCompressWeightFalse;
  // check optypelist_for_implmode and op_select_implmode
  std::string optypelist_for_implmode = global_options.find(ge::ir_option::OPTYPELIST_FOR_IMPLMODE) ==
                                            global_options.end()
                                            ? ""
                                            : global_options[ge::ir_option::OPTYPELIST_FOR_IMPLMODE];
  std::string op_select_implmode = global_options.find(ge::ir_option::OP_SELECT_IMPL_MODE) ==
                                       global_options.end()
                                       ? ""
                                       : global_options[ge::ir_option::OP_SELECT_IMPL_MODE];
  GE_CHK_BOOL_EXEC(
      ge::CheckImplmodeParamValid(optypelist_for_implmode, op_select_implmode) == ge::SUCCESS,
      return ge::GRAPH_PARAM_INVALID, "check optypelist_for_implmode and op_select_implmode failed!");
  global_options[ge::ir_option::OP_SELECT_IMPL_MODE] = op_select_implmode;

  // set precision mode default value
  std::string precision_mode = global_options.find(ge::ir_option::PRECISION_MODE) ==
                               global_options.end()
                               ? "force_fp16"
                               : global_options[ge::ir_option::PRECISION_MODE];
  global_options[ge::ir_option::PRECISION_MODE] = precision_mode;

  return GRAPH_SUCCESS;
}

static void GetOpsProtoPath(string &opsproto_path) {
  GELOGI("Start to get ops proto path schedule.");
  const char *path_env = std::getenv("ASCEND_OPP_PATH");
  if (path_env != nullptr) {
    string path = path_env;
    string file_path = RealPath(path.c_str());
    if (file_path.empty()) {
      GELOGE(FAILED, "File path %s is invalid.", path.c_str());
      return;
    }
    opsproto_path = (path + "/op_proto/custom/" + ":") + (path + "/op_proto/built-in/");
    GELOGI("Get opsproto so path from env : %s", path.c_str());
    return;
  }
  string path_base = PluginManager::GetPath();
  GELOGI("path_base is %s", path_base.c_str());
  path_base = path_base.substr(0, path_base.rfind('/'));
  path_base = path_base.substr(0, path_base.rfind('/') + 1);
  opsproto_path = (path_base + "ops/op_proto/custom/" + ":") + (path_base + "ops/op_proto/built-in/");
}

graphStatus aclgrphBuildInitialize(std::map<std::string, std::string> global_options) {
  GELOGD("Enter aclgrphInitialize start!");
  // check global options
  if (CheckGlobalOptions(global_options) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_PARAM_INVALID, "Check global options falied!");
    return GRAPH_PARAM_INVALID;
  }
  // print global option map
  ge::PrintOptionMap(global_options, "global option");

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
    omg_context_ = domi::GetContext();
    omg_context_.format = domi::DOMI_TENSOR_ND;
    omg_context_.input_nodes_format_map.clear();
    omg_context_.output_formats.clear();
    omg_context_.user_input_dims.clear();
    omg_context_.input_dims.clear();
    omg_context_.op_conf_map.clear();
    omg_context_.out_nodes_map.clear();
    omg_context_.user_out_nodes.clear();
    omg_context_.net_format = domi::DOMI_TENSOR_RESERVED;
    omg_context_.type = domi::FRAMEWORK_RESERVED;
    omg_context_.run_mode = ONLY_PRE_CHECK;
    omg_context_.train_flag = false;
    omg_context_.output_type.clear();
    omg_context_.is_dynamic_input = false;
    omg_context_.dynamic_batch_size.clear();
    omg_context_.dynamic_image_size.clear();
    omg_context_.dynamic_dims.clear();
  };
  ~Impl() { (void)generator_.Finalize(); };
  graphStatus CheckOptions(const std::map<std::string, std::string> &options);
  graphStatus CreateInputsForIRBuild(const ge::Graph &graph, vector<ge::GeTensor> &inputs);
  graphStatus Init(const std::map<std::string, std::string> &options);
  graphStatus BuildModel(const Graph &graph, const std::map<std::string, std::string> &options,
                         ModelBufferData &ge_models);
  graphStatus InitDomiOmgContext(const string &input_shape, const string &input_format, const string &net_format,
                                 bool is_dynamic_input);
  void SetRtSocVersion();
  void UpdateThreadContext();
  void LoadOpsProto();s
 public:
  ge::GeGenerator generator_;
  std::map<std::string, std::string> options_;
  bool is_dynamic_input_ = false;
  OmgContext omg_context_;
};

graphStatus Impl::CheckOptions(const std::map<std::string, std::string> &options) {
  for (auto &ele : options) {
    auto it = ge::ir_option::ir_builder_suppported_options.find(ele.first);
    if (it == ge::ir_option::ir_builder_suppported_options.end()) {
      auto it_lx_fusion = ir_builder_supported_options_for_lx_fusion.find(ele.first);
      if (it_lx_fusion == ir_builder_supported_options_for_lx_fusion.end()) {
        GELOGE(GRAPH_PARAM_INVALID, "input options include unsupported option(%s).Please check!",
               ele.first.c_str());
        return GRAPH_PARAM_INVALID;
      }
    }
    options_.insert(ele);
  }
  // Check options build_mode and build_step.
  std::string build_mode;
  auto it = options_.find(BUILD_MODE);
  if (it != options_.end() && !(it->second.empty())) {
    if (build_mode_options.find(it->second) == build_mode_options.end()) {
      GELOGE(GRAPH_PARAM_INVALID, "Build mode:%s is unsupported. Please check!", it->second.c_str());
      return GRAPH_PARAM_INVALID;
    }
    build_mode = it->second;
  }
  it = options_.find(BUILD_STEP);
  if (it != options_.end() && !(it->second.empty())) {
    if (build_step_options.find(it->second) == build_step_options.end()) {
      GELOGE(GRAPH_PARAM_INVALID, "Build step:%s is unsupported. Please check!", it->second.c_str());
      return GRAPH_PARAM_INVALID;
    }
  } else {
    if (build_mode == BUILD_MODE_TUNING) {
      GELOGE(GRAPH_PARAM_INVALID, "Build mode tuning must specify build step. Please check!");
      return GRAPH_PARAM_INVALID;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus Impl::Init(const std::map<std::string, std::string> &options) {
  // 1. check options
  graphStatus ret = CheckOptions(options);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "User input options are illegal! Please check!");
    return ret;
  }

  std::string build_mode = (options_.find(BUILD_MODE) == options_.end() || options_[BUILD_MODE] == BUILD_MODE_NORMAL)
                           ? "" : options_[BUILD_MODE];
  options_[BUILD_MODE] = build_mode;
  // set log level
  std::string log = options_.find(ge::ir_option::LOG_LEVEL) == options_.end()
                        ? IR_OPTION_LOG_LEVEL_DEFAULT
                        : options_[ge::ir_option::LOG_LEVEL];
  GE_CHK_BOOL_RET_STATUS_NOLOG(ge::CheckLogParamValidAndSetLogLevel(log) == 0, GRAPH_PARAM_INVALID);
  options_[ge::ir_option::LOG_LEVEL] = log;

  string input_shape = options_.find("input_shape") == options_.end() ? "" : options_["input_shape"];
  string input_format = options_.find("input_format") == options_.end() ? "" : options_["input_format"];
  string net_format = options_.find("net_format") == options_.end() ? "" : options_["net_format"];
  string dynamic_batch_size = options_.find(ge::ir_option::DYNAMIC_BATCH_SIZE) == options_.end()
                                  ? ""
                                  : options_[ge::ir_option::DYNAMIC_BATCH_SIZE];
  string dynamic_image_size = options_.find(ge::ir_option::DYNAMIC_IMAGE_SIZE) == options_.end()
                                  ? ""
                                  : options_[ge::ir_option::DYNAMIC_IMAGE_SIZE];
  string dynamic_dims =
      options_.find(ge::ir_option::DYNAMIC_DIMS) == options_.end() ? "" : options_[ge::ir_option::DYNAMIC_DIMS];

  auto status = CheckDynamicInputParamValid(dynamic_batch_size, dynamic_image_size, dynamic_dims, input_shape,
                                            input_format, is_dynamic_input_);
  if (status != ge::SUCCESS) {
    GELOGE(GRAPH_PARAM_INVALID, "Check dynamic input size failed!");
    return GRAPH_PARAM_INVALID;
  }
  GELOGD("User input dynamic_batch_size:%s, dynamic_image_size:%s, dynamic_dims:%s.", dynamic_batch_size.c_str(),
         dynamic_image_size.c_str(), dynamic_dims.c_str());
  omg_context_.dynamic_batch_size = dynamic_batch_size;
  omg_context_.dynamic_image_size = dynamic_image_size;
  omg_context_.dynamic_dims = dynamic_dims;
  // check output_type
  std::string output_type = options_.find(ge::ir_option::OUTPUT_TYPE) == options_.end()
                                ? ""
                                : options_[ge::ir_option::OUTPUT_TYPE];
  GE_CHK_BOOL_EXEC(ge::CheckOutputTypeParamValid(output_type) == ge::SUCCESS,
      return ge::GRAPH_PARAM_INVALID, "check output type failed!");
  // check insert_op_conf
  std::string insert_op_conf = options_.find(ge::ir_option::INSERT_OP_FILE) == options_.end()
                                   ? ""
                                   : options_[ge::ir_option::INSERT_OP_FILE];
  GE_CHK_BOOL_EXEC(ge::CheckInsertOpConfParamValid(std::string(insert_op_conf)) == ge::SUCCESS,
      return ge::GRAPH_PARAM_INVALID, "check insert op conf failed!");

  GE_CHK_BOOL_EXEC(insert_op_conf.empty() || dynamic_dims.empty(),
                   return ge::GRAPH_PARAM_INVALID, "dynamic dims function does not support aipp");

  // for IR builder.Only support om mode, so here fixed;
  options_.insert(std::pair<string, string>(string(IR_OPTION_MODE), to_string(0)));
  options_.insert(std::pair<string, string>(string(IR_OPTION_TARGET), "mini"));
  options_.insert(std::pair<string, string>(string(ge::RUN_FLAG), to_string(0)));
  options_.insert(std::pair<string, string>(string(ge::TRAIN_FLAG), to_string(0)));
  options_.insert(std::pair<string, string>(string(ge::SAVE_ORIGINAL_MODEL), to_string(0)));
  // print ge option map
  ge::PrintOptionMap(options_, "ge option");

  SetRtSocVersion();
  UpdateThreadContext();
  // 3. init generator with options_
  ret = generator_.Initialize(options_, omg_context_);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "generator Initialize failed!");
    return ret;
  }
  // 4.parse and init Context with input shape format and net format info
  return this->InitDomiOmgContext(input_shape, input_format, net_format, is_dynamic_input_);
}

void Impl::SetRtSocVersion() {
  const auto &global_options = GetMutableGlobalOptions();
  auto it = global_options.find(ge::SOC_VERSION);
  if (it != global_options.end()) {
    const char *soc_version = it->second.c_str();
    rtError_t rt_ret = rtSetSocVersion(soc_version);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("Set soc version %s failed. ret:0x%X", soc_version, rt_ret);
    }
    GELOGI("Set soc version %s success.", soc_version);
  }
}

void Impl::UpdateThreadContext() {
  GetThreadLocalContext().SetGlobalOption(GetMutableGlobalOptions());
  GetThreadLocalContext().SetGraphOption(options_);
}

void Impl::LoadOpsProto() {
  string opsproto_path;
  GetOpsProtoPath(opsproto_path);
  GELOGI("Get opsproto path is %s", opsproto_path.c_str());
  OpsProtoManager *manager = OpsProtoManager::Instance();
  map<string, string> option_tmp;
  option_tmp.emplace(std::pair<string, string>(string("ge.opsProtoLibPath"), opsproto_path));
  (void)manager->Initialize(option_tmp);
}

graphStatus Impl::CreateInputsForIRBuild(const ge::Graph &graph, vector<ge::GeTensor> &inputs) {
  auto compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);
  int64_t index = 0;
  for (ge::NodePtr &input_node : compute_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(input_node);
    ge::OpDescPtr op = input_node->GetOpDesc();
    GE_CHECK_NOTNULL(op);
    if (op->GetType() == DATA) {
      (void)AttrUtils::SetInt(op, ATTR_NAME_INDEX, index++);
      GELOGI("Data op inputDesc size: %zu", op->GetAllInputsDesc().size());
      ge::GeTensorDesc tensor = op->GetInputDesc(0);
      string data_op_name = op->GetName();
      GELOGI("Data op name: %s", data_op_name.c_str());
      ge::GeShape data_shape;
      auto iter = omg_context_.input_dims.find(data_op_name);
      if (iter != omg_context_.input_dims.end()) {
        data_shape = ge::GeShape(iter->second);
        GELOGI("Data op get shape from Context.");
      } else {
        data_shape = tensor.GetShape();
        GELOGI("Data op get shape from InputDesc in ge ir graph.");
      }
      // If user point input format, do work for all data ops; else do according to tensor_desc
      auto data_format = omg_context_.format != domi::DOMI_TENSOR_ND ?
        ge::TypeUtils::DomiFormatToFormat(omg_context_.format) : tensor.GetFormat();
      ge::DataType data_type = tensor.GetDataType();
      string data_type_str = ge::TypeUtils::DataTypeToSerialString(data_type);
      GELOGI("Data op get data type:%s from InputDesc in ge ir graph.", data_type_str.c_str());

      ge::GeTensor inputTensor;
      ge::GeTensorDesc desc(data_shape, ge::Format(data_format), data_type);
      inputTensor.SetTensorDesc(desc);
      inputs.push_back(inputTensor);
    }
  }
  GELOGD("CreateInputsForIRBuild, inputs size: %zu", inputs.size());
  return GRAPH_SUCCESS;
}
graphStatus Impl::BuildModel(const Graph &graph, const std::map<std::string, std::string> &options,
                             ModelBufferData &model) {
  // 1. init GeGenerator with user optios
  graphStatus ret = Init(options);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "Build ir model Init failed!");
    return ret;
  }

  // 2. construct input
  std::vector<GeTensor> inputs;
  if (!omg_context_.is_dynamic_input) {  // if dynamic input , no need to creat inputs
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
  omg_context_.input_dims.clear();
  omg_context_.user_input_dims.clear();
  omg_context_.is_dynamic_input = is_dynamic_input;
  // the default value is ND
  omg_context_.format = domi::DOMI_TENSOR_ND;
  if (!input_format.empty()) {
    auto iter = ge::input_format_str_to_geformat.find(input_format);
    if (iter != ge::input_format_str_to_geformat.end()) {
      omg_context_.format = iter->second;
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

  if (!ParseInputShape(input_shape, omg_context_.input_dims, omg_context_.user_input_dims, is_dynamic_input)) {
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
    GELOGE(GRAPH_PARAM_INVALID, "input model is illegal");
    return GRAPH_PARAM_INVALID;
  }
  return FileSaver::SaveToFile((output_file + ".om"), reinterpret_cast<void*>(model.data.get()),
                               static_cast<uint32_t>(model.length));
}

graphStatus aclgrphGetIRVersion(int *major_version, int *minor_version, int *patch_version) {
  GELOGD("Enter aclgrphGetIRVersion process!");
  GE_CHECK_NOTNULL(major_version);
  GE_CHECK_NOTNULL(minor_version);
  GE_CHECK_NOTNULL(patch_version);
  *major_version = IR_MAJOR_VERSION;
  *minor_version = IR_MINOR_VERSION;
  *patch_version = IR_PATCH_VERSION;
  return GRAPH_SUCCESS;
}

graphStatus aclgrphInferShapeAndType(ge::Graph &graph) {
  Impl builder;
  builder.LoadOpsProto();

  auto compute_graph = GraphUtils::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);

  auto root_graph = compute_graph->GetParentGraph();
  if (root_graph != nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "Input param should not be subgraph");
    return GRAPH_PARAM_INVALID;
  }

  auto ret = compute_graph->InferOriginFormat();
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "Acl InferOriginFormat failed.");
    return ret;
  }

  for (auto &node: compute_graph->GetAllNodes()) {
    graphStatus ret = ShapeRefiner::InferShapeAndType(node);
    if (ret == GRAPH_PARAM_INVALID) {
      GELOGW("Can not find infershape func.");
      continue;
    } else if (ret != GRAPH_SUCCESS) {
      GELOGE(ret, "Acl infershape failed.");
      return ret;
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus aclgrphDumpGraph(const ge::Graph &graph, const char *file, const size_t len) {
  GE_CHECK_NOTNULL(file);

  if (len > PATH_MAX || len != strlen(file) || strlen(file) == 0) {
    GELOGE(GRAPH_PARAM_INVALID, "File path invalid.");
    return GRAPH_PARAM_INVALID;
  }

  auto compute_graph = GraphUtils::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);

  string full_path(file, len);
  for (size_t i = 0; i < len; i++) {
    if (full_path[i] == '\\') {
      full_path.replace(i, 1, "/");
    }
  }

  string suffix;
  string file_path;
  int pos = full_path.rfind("/");
  if (pos != -1) {
    suffix = full_path.substr(pos + 1, -1);
    file_path = full_path.substr(0, pos);
  } else {
    suffix = full_path;
    file_path = "./";
  }

  if (suffix.empty()) {
    suffix = compute_graph->GetName();
    if (suffix.empty()) {
      suffix = "graph";
    }
  }

  char path[PATH_MAX] = {0};
  if (realpath(file_path.c_str(), path) == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "Dump file path:%s  is invalid.", file);
    return GRAPH_PARAM_INVALID;
  }

  GraphUtils::DumpGEGrph(compute_graph, string(path), suffix);
  GraphUtils::DumpGrphToOnnx(*compute_graph, string(path), suffix);
  uint64_t i = 0;
  for (const auto &sub_graph_func : compute_graph->GetAllSubgraphs()) {
    auto sub_graph_func_name = suffix + std::string("_sub_graph_") + std::to_string(i++);
    GraphUtils::DumpGEGrph(sub_graph_func, string(path), sub_graph_func_name);
    GraphUtils::DumpGrphToOnnx(*sub_graph_func, string(path), sub_graph_func_name);
  }

  return GRAPH_SUCCESS;
}

}  // namespace ge
