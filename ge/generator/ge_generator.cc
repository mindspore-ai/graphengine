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

#include "generator/ge_generator.h"

#include <atomic>

#include "common/ge/ge_util.h"
#include "common/ge/plugin_manager.h"
#include "common/helper/model_helper.h"
#include "common/helper/om_file_helper.h"
#include "common/util.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "ge/ge_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/manager/graph_manager.h"
#include "graph/manager/util/rt_context_util.h"
#include "graph/opsproto_manager.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"
#include "init/gelib.h"
#include "model/ge_model.h"

using std::map;
using std::string;
using std::vector;

namespace {
const char *const kAttrOpType = "op_type";
const char *const kEngineNameDefault = "default";
const char *const kVectorEngine = "VectorEngine";
const char *const kAIcoreEngine = "AIcoreEngine";
const char *const kFileNameSuffix = "online";

std::map<ge::OpEngineType, std::string> engine_type_map{
    {ge::ENGINE_SYS, kEngineNameDefault}, {ge::ENGINE_AICORE, kAIcoreEngine}, {ge::ENGINE_VECTOR, kVectorEngine}};

bool ContainsDynamicInpus(const ge::OpDesc &op_desc) {
  for (auto &tensor_desc : op_desc.GetAllInputsDescPtr()) {
    if (tensor_desc->MutableShape().IsUnknownShape()) {
      GELOGI("Contains unknown shape input. set is_dynamic_input to true.");
      return true;
    }
  }
  return false;
}
}  // namespace

namespace ge {
static Status CheckEngineTypeSupport(const OpDescPtr &op_desc, OpEngineType engine_type) {
  GE_CHECK_NOTNULL_EXEC(op_desc, return PARAM_INVALID);
  if (engine_type == ENGINE_SYS) {
    GELOGI("CheckEngineType: use default engine.");
    return SUCCESS;
  }

  // get op engine name
  string op_engine_name;
  auto iter = engine_type_map.find(engine_type);
  if (iter != engine_type_map.end()) {
    op_engine_name = iter->second;
    GELOGI("CheckEngineType: engine type: %d", static_cast<int>(engine_type));
  } else {
    ErrorManager::GetInstance().ATCReportErrMessage("E14001", {"opname", "optype", "value", "reason"},
        {op_desc->GetName(), op_desc->GetType(), "engine type",
        "it only support kEngineNameDefault/kAIcoreEngine/kVectorEngine"});
    GELOGE(FAILED, "CheckEngineType: engine type: %d not support", static_cast<int>(engine_type));
    return FAILED;
  }

  if (op_desc->HasAttr(ATTR_NAME_UNREGST_OPPATH)) {
    op_desc->SetOpEngineName(op_engine_name);
    op_desc->SetOpKernelLibName(op_engine_name);
    return SUCCESS;
  }
  // set op engine name and opkernelLib. when engine support
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if ((instance_ptr == nullptr) || (!instance_ptr->InitFlag())) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "CheckEngineType failed.");
    return FAILED;
  }
  OpsKernelManager &ops_kernel_manager = instance_ptr->OpsKernelManagerObj();
  std::vector<OpInfo> op_infos = ops_kernel_manager.GetOpsKernelInfo(op_desc->GetType());
  if (op_infos.empty()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E14001", {"opname", "optype", "value", "reason"},
        {op_desc->GetName(), op_desc->GetType(), "optype", "it can not find"});
    GELOGE(FAILED, "CheckEngineType: Can not get op info by op type %s", op_desc->GetType().c_str());
    return FAILED;
  }
  string kernel_name;
  for (const auto &it : op_infos) {
    if (it.engine == op_engine_name) {
      kernel_name = it.opKernelLib;
      break;
    }
  }
  if (kernel_name.empty()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E14001", {"opname", "optype", "value", "reason"},
        {op_desc->GetName(), op_desc->GetType(), "engine name" + FmtToStr(op_engine_name), "it can not find"});
    GELOGE(FAILED, "CheckEngineType:Can not find ops kernel, engine name: %s.", op_engine_name.c_str());
    return FAILED;
  }
  auto &kernel_map = ops_kernel_manager.GetAllOpsKernelInfoStores();
  auto kernel_info_store = kernel_map.find(kernel_name);
  if (kernel_info_store != kernel_map.end()) {
    std::string unsupported_reason;
    if (kernel_info_store->second->CheckSupported(op_desc, unsupported_reason)) {
      op_desc->SetOpEngineName(op_engine_name);
      op_desc->SetOpKernelLibName(kernel_name);
      GELOGI("CheckEngineType:Set OpKernelLibName %s and engine name %s into op_desc %s", kernel_name.c_str(),
             op_engine_name.c_str(), op_desc->GetName().c_str());
      return SUCCESS;
    } else {
      ErrorManager::GetInstance().ATCReportErrMessage(
        "E13002", {"optype", "opskernel", "reason"}, {op_desc->GetType(), kernel_name, unsupported_reason});
      GELOGE(FAILED, "CheckEngineType: check support failed, Op type %s of ops kernel %s is unsupported, reason:%s",
             op_desc->GetType().c_str(), kernel_name.c_str(), unsupported_reason.c_str());
      return FAILED;
    }
  } else {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E13003", {"opname", "optype"}, {op_desc->GetName(), op_desc->GetType()});
    GELOGE(FAILED,
           "CheckEngineType:Can not find any supported ops kernel info store by kernel_name %s,"
           "op type is %s, op name is %s",
           kernel_name.c_str(), op_desc->GetType().c_str(), op_desc->GetName().c_str());
  }
  return FAILED;
}

static Status AddInputs(const ComputeGraphPtr &graph, const NodePtr &node, GeTensorDesc &tensor, int32_t index,
                        bool attr) {
  GE_CHECK_NOTNULL_EXEC(graph, return PARAM_INVALID);
  GE_CHECK_NOTNULL_EXEC(node, return PARAM_INVALID);

  auto format = tensor.GetFormat();
  auto data_type = tensor.GetDataType();
  if (format == FORMAT_RESERVED && data_type == DT_UNDEFINED) {
    return SUCCESS;
  }

  string op_type;
  bool is_const = false;
  (void)AttrUtils::GetBool(tensor, CONST_ATTR_NAME_INPUT, is_const);
  if (is_const) {
    GELOGD("Get input[%d] is const", index);
    op_type = CONSTANTOP;
  } else if (!AttrUtils::GetStr(tensor, kAttrOpType, op_type) || op_type.empty()) {
    op_type = DATA;
  }

  string op_name = node->GetName() + "_in_" + std::to_string(index);
  OpDescPtr data_op = MakeShared<ge::OpDesc>(op_name, op_type);
  if (data_op == nullptr) {
    return FAILED;
  }
  if (is_const) {
    ConstGeTensorPtr tensor_value;
    if (!AttrUtils::GetTensor(tensor, ge::ATTR_NAME_WEIGHTS, tensor_value)) {
      GELOGE(FAILED, "Get value failed, node name:%s.", tensor.GetName().c_str());
      return FAILED;
    }
    if (!AttrUtils::SetTensor(data_op, ge::ATTR_NAME_WEIGHTS, tensor_value)) {
      GELOGE(FAILED, "Set attr ATTR_NAME_WEIGHTS fail.");
      return FAILED;
    }
  }

  (void)AttrUtils::SetBool(data_op, "_is_single_op", true);

  GE_CHK_BOOL_EXEC(data_op->AddInputDesc(tensor) == GRAPH_SUCCESS, return FAILED, "Add input desc fail.");
  GE_CHK_BOOL_EXEC(data_op->AddOutputDesc(tensor) == GRAPH_SUCCESS, return FAILED, "Add output desc fail.");
  if (attr) {
    GE_CHK_BOOL_EXEC(AttrUtils::SetInt(data_op, ATTR_NAME_INDEX, index), return FAILED, "Set index fail.");
  }

  ge::NodePtr arg_node = graph->AddNode(data_op);
  GE_CHK_BOOL_EXEC(arg_node != nullptr, return FAILED, "Insert Data node fail.");

  GE_CHK_STATUS(GraphUtils::AddEdge(arg_node->GetOutDataAnchor(0), node->GetInDataAnchor(index)),
                "Add edge[%s->%s] fail.", data_op->GetName().c_str(), node->GetName().c_str());

  return SUCCESS;
}

static Status AddOutputs(const ComputeGraphPtr &graph, const NodePtr &node, const vector<GeTensor> &outputs) {
  OpDescPtr op_desc = MakeShared<ge::OpDesc>(graph->GetName() + "_" + NODE_NAME_NET_OUTPUT, NETOUTPUT);
  if (op_desc == nullptr) {
    return FAILED;
  }
  (void)AttrUtils::SetBool(op_desc, "_is_single_op", true);
  int32_t count = 0;
  for (const auto &out_desc : outputs) {
    GeTensorDesc tensor = out_desc.GetTensorDesc();
    TensorUtils::SetInputTensor(tensor, true);
    GE_CHK_BOOL_EXEC(op_desc->AddInputDesc(tensor) == GRAPH_SUCCESS, return FAILED, "Add input desc fail");

    TensorUtils::SetInputTensor(tensor, false);
    TensorUtils::SetOutputTensor(tensor, true);
    GE_CHK_BOOL_EXEC(op_desc->AddOutputDesc(tensor) == GRAPH_SUCCESS, return FAILED, "Add output desc fail");
    count++;
  }
  GE_CHECK_NOTNULL_EXEC(graph, return PARAM_INVALID);
  ge::NodePtr out_node = graph->AddNode(op_desc);
  GE_CHK_BOOL_EXEC(out_node != nullptr, return FAILED, "Insert Output node fail.");
  GE_CHECK_NOTNULL_EXEC(node, return PARAM_INVALID);
  for (int32_t i = 0; i < count; ++i) {
    GE_CHK_STATUS(GraphUtils::AddEdge(node->GetOutDataAnchor(i), out_node->GetInDataAnchor(i)),
                  "Add edge[%s->%s] fail.", node->GetName().c_str(), out_node->GetName().c_str());
  }

  return SUCCESS;
}

static void GetOpsProtoPath(string &opsproto_path) {
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

class GeGenerator::Impl {
 public:
  Impl(OmgContext &omg_context) : omg_context_(omg_context) {}
  ~Impl() = default;

  Status BuildModel(const Graph &graph, const vector<GeTensor> &inputs, GeRootModelPtr &ge_models);

  Status SaveModel(const string &file_name_prefix, GeModelPtr &models, ModelBufferData &model);

  Status SaveRootModel(const string &file_name_prefix, GeRootModelPtr &model, ModelBufferData &model_buff);

  Status SaveParams(GeModelPtr &ge_model, const string &type, const map<string, GeAttrValue> &attrs,
                    const vector<GeTensor> &inputs, const vector<GeTensor> &outputs);

  Status GenerateInfershapeGraph(const Graph &graph);

  OmgContext &omg_context_;
  GraphManager graph_manager_;
  SaveParam save_param_;
  bool is_offline_ = true;
  bool is_singleop_unregistered_ = false;
  std::string build_mode_;
  std::string build_step_;
  static std::mutex mutex_;

 private:
  static std::string Trim(const std::string &str);
  bool ParseVersion(const std::string &line, std::string &version);
  bool GetVersionFromPath(const std::string &file_path, std::string &version);
  bool SetAtcVersionInfo(AttrHolder &obj);
  bool SetOppVersionInfo(AttrHolder &obj);
};

Status GeGenerator::Initialize(const map<string, string> &options) {
  return Initialize(options, domi::GetContext());
}

Status GeGenerator::Initialize(const map<string, string> &options, OmgContext &omg_context) {
  impl_ = ge::MakeShared<Impl>(omg_context);
  if (impl_ == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Make shared failed");
    return MEMALLOC_FAILED;
  }
  string opsproto_path;
  GetOpsProtoPath(opsproto_path);
  GELOGI("Get opsproto path is %s", opsproto_path.c_str());
  OpsProtoManager *manager = OpsProtoManager::Instance();
  map<string, string> option_tmp;
  option_tmp.emplace(std::pair<string, string>(string("ge.opsProtoLibPath"), opsproto_path));
  (void)manager->Initialize(option_tmp);

  Status ret = impl_->graph_manager_.Initialize(options);
  if (ret != SUCCESS) {
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_INIT_FAILED, "Graph manager initialize failed.");
    return GE_GENERATOR_GRAPH_MANAGER_INIT_FAILED;
  }
  // get ek file
  auto iter = options.find(EK_FILE);
  if (iter != options.end()) {
    impl_->save_param_.ek_file = iter->second;
  }
  // get cert file
  iter = options.find(CERT_FILE);
  if (iter != options.end()) {
    impl_->save_param_.cert_file = iter->second;
  }
  // get hw key file
  iter = options.find(HW_KEY_FILE);
  if (iter != options.end()) {
    impl_->save_param_.hw_key_file = iter->second;
  }
  // get private file
  iter = options.find(PRIVATE_KEY_FILE);
  if (iter != options.end()) {
    impl_->save_param_.pri_key_file = iter->second;
  }

  // get build mode
  iter = options.find(BUILD_MODE);
  if (iter != options.end()) {
    impl_->build_mode_ = iter->second;
  }
  // get build step
  iter = options.find(BUILD_STEP);
  if (iter != options.end()) {
    impl_->build_step_ = iter->second;
  }
  return SUCCESS;
}

Status GeGenerator::Finalize() {
  GE_CHECK_NOTNULL_EXEC(impl_, return PARAM_INVALID);
  Status ret = impl_->graph_manager_.Finalize();
  if (ret != SUCCESS) {
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_FINALIZE_FAILED, "Graph manager finalize failed.");
    return GE_GENERATOR_GRAPH_MANAGER_FINALIZE_FAILED;
  }
  return SUCCESS;
}

Status GeGenerator::GenerateOfflineModel(const Graph &graph, const string &file_name_prefix,
                                         const vector<GeTensor> &inputs) {
  GELOGI("Start to generate offline model.");
  ModelBufferData model;
  return GenerateModel(graph, file_name_prefix, inputs, model, true);
}

Status GeGenerator::GenerateOnlineModel(const Graph &graph, const vector<GeTensor> &inputs, ModelBufferData &model) {
  return GenerateModel(graph, "online", inputs, model, false);
}

Status GeGenerator::GenerateInfershapeGraph(const Graph &graph) {
  GE_CHECK_NOTNULL_EXEC(impl_, return PARAM_INVALID);

  Status ret = impl_->GenerateInfershapeGraph(graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "Dump infershape json failed");
    if (impl_->graph_manager_.Finalize() != SUCCESS) {
      GELOGE(FAILED, "graph_manager finalize fail.");
    }
    return ret;
  }
  GELOGI("Generate infer shape graph success");
  return SUCCESS;
}

std::mutex GeGenerator::Impl::mutex_;

// Remove the space and tab before and after the string
std::string GeGenerator::Impl::Trim(const std::string &str) {
  if (str.empty()) {
    return str;
  }

  std::string::size_type start = str.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return str;
  }

  std::string::size_type end = str.find_last_not_of(" \t\r\n") + 1;
  return str.substr(start, end);
}

// Parsing the command line
bool GeGenerator::Impl::ParseVersion(const std::string &line, std::string &version) {
  std::string flag = "Version=";
  std::string temp = Trim(line);

  if (temp.empty()) {
    GELOGW("line is empty.");
    return false;
  }

  std::string::size_type pos = temp.find(flag);
  if (pos == std::string::npos) {
    GELOGW("Incorrect line [%s], it must include [%s].", line.c_str(), flag.c_str());
    return false;
  }

  if (temp.size() == flag.size()) {
    GELOGW("version information is empty. %s", line.c_str());
    return false;
  }

  version = temp.substr(pos + flag.size());

  return true;
}

bool GeGenerator::Impl::GetVersionFromPath(const std::string &file_path, std::string &version) {
  // Normalize the path
  string resolved_file_path = RealPath(file_path.c_str());
  if (resolved_file_path.empty()) {
    GELOGW("Invalid input file path [%s], make sure that the file path is correct.", file_path.c_str());
    return false;
  }
  std::ifstream fs(resolved_file_path, std::ifstream::in);
  if (!fs.is_open()) {
    GELOGW("Open %s failed.", file_path.c_str());
    return false;
  }

  std::string line;
  if (getline(fs, line)) {
    if (!ParseVersion(line, version)) {
      GELOGW("Parse version failed. content is [%s].", line.c_str());
      fs.close();
      return false;
    }
  } else {
    GELOGW("No version information found in the file path:%s", file_path.c_str());
    fs.close();
    return false;
  }

  fs.close();  // close the file
  return true;
}

// Set package version information in the model
bool GeGenerator::Impl::SetAtcVersionInfo(AttrHolder &obj) {
  std::string path_base = ge::GELib::GetPath();
  path_base = path_base.substr(0, path_base.rfind('/'));
  path_base = path_base.substr(0, path_base.rfind('/') + 1);

  std::string version_path = path_base + "version.info";
  std::string version;
  if (!GetVersionFromPath(version_path, version)) {
    GELOGW("Get atc version information failed!");
    return false;
  }
  // set version info
  if (!ge::AttrUtils::SetStr(obj, ATTR_MODEL_ATC_VERSION, version)) {
    GELOGW("Ge model set atc version failed!");
    return false;
  }
  return true;
}

// Set package version information in the model
bool GeGenerator::Impl::SetOppVersionInfo(AttrHolder &obj) {
  const char *path_env = std::getenv("ASCEND_OPP_PATH");
  if (path_env == nullptr) {
    GELOGW("Get environment variable ASCEND_OPP_PATH failed!");
    return false;
  }
  std::string version_path = path_env;
  version_path += "/version.info";
  std::string version;
  if (!GetVersionFromPath(version_path, version)) {
    GELOGW("Get opp version information failed!");
    return false;
  }
  // set version info
  if (!ge::AttrUtils::SetStr(obj, ATTR_MODEL_OPP_VERSION, version)) {
    GELOGW("Ge model set opp version failed!");
    return false;
  }
  return true;
}

Status GeGenerator::GenerateModel(const Graph &graph, const string &file_name_prefix, const vector<GeTensor> &inputs,
                                  ModelBufferData &model, bool is_offline) {
  rtContext_t ctx = nullptr;
  auto rt = rtCtxGetCurrent(&ctx);
  if (rt != RT_ERROR_NONE) {
    GELOGD("Current ctx is null.");
    ctx = nullptr;
  }

  GeRootModelPtr ge_root_model = nullptr;
  GE_CHECK_NOTNULL_EXEC(impl_, return PARAM_INVALID);
  impl_->is_offline_ = is_offline;
  Status ret = impl_->BuildModel(graph, inputs, ge_root_model);
  if (ret != SUCCESS) {
    GELOGE(ret, "Build model failed.");
    if (impl_->graph_manager_.Finalize() != SUCCESS) {
      GELOGE(FAILED, "graph_manager finalize fail.");
    }
    return ret;
  }

  /// BUILD_MODE_TUNING with BUILD_STEP_BEFORE_UB_MATCH no need save model;
  /// BUILD_MODE_TUNING with BUILD_STEP_AFTER_BUILDER no need save model;
  /// BUILD_MODE_TUNING with BUILD_STEP_AFTER_BUILDER_SUB no need save model.
  if ((impl_->build_mode_ == BUILD_MODE_TUNING) &&
      (impl_->build_step_ == BUILD_STEP_BEFORE_UB_MATCH || impl_->build_step_ == BUILD_STEP_AFTER_BUILDER ||
       impl_->build_step_ == BUILD_STEP_AFTER_BUILDER_SUB)) {
    GELOGI("Build mode:%s with step:%s no need SaveModel.",
           impl_->build_mode_.c_str(),
           impl_->build_step_.c_str());
    return SUCCESS;
  }

  GE_CHECK_NOTNULL(ge_root_model);
  GE_CHECK_NOTNULL(ge_root_model->GetRootGraph());
  ModelHelper model_helper;
  string model_name = "";
  Status name_ret = model_helper.GetModelNameFromMergedGraphName(ge_root_model->GetRootGraph()->GetName(),
                                                                 model_name);
  if (name_ret != SUCCESS) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10000", {"parameter"}, {"output"});
    GELOGE(FAILED, "Get model_name failed. Param --output is invalid.");
    return PARAM_INVALID;
  }
  map<string, GeModelPtr> name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
  GeModelPtr &ge_model = name_to_ge_model[ge_root_model->GetRootGraph()->GetName()];
  GE_RETURN_WITH_LOG_IF_FALSE(ge_model != nullptr, "ge_model cannot be null");
  ge_model->SetName(model_name);
  ret = impl_->SaveRootModel(file_name_prefix, ge_root_model, model);
  if (ret != SUCCESS) {
    GELOGE(ret, "Save model failed");
    if (impl_->graph_manager_.Finalize() != SUCCESS) {
      GELOGE(FAILED, "graph_manager finalize fail.");
    }
    return ret;
  }

  if (ctx != nullptr) {
    (void)rtCtxSetCurrent(ctx);
  }

  return SUCCESS;
}

namespace {
  bool IsNeedConnectInputOpForSingleOp(GeTensorDesc &tensor_desc) {
    bool is_need = true;
    // format and dtype is all reserved, stand for Optional input. When singleop scene
    if (tensor_desc.GetFormat() == FORMAT_RESERVED && tensor_desc.GetDataType() == DT_UNDEFINED) {
      is_need = false;
    }
    return is_need;
  }
}

Status GeGenerator::CheckForSingleOp(OpDescPtr &op_desc, const vector<GeTensor> &inputs,
                                     const vector<GeTensor> &outputs) {
  GE_CHECK_NOTNULL_EXEC(op_desc, return PARAM_INVALID);
  if (!inputs.empty() && (inputs.size() != op_desc->GetAllInputsSize())) {
    ErrorManager::GetInstance().ATCReportErrMessage("E14001", {"opname", "optype", "value", "reason"},
        {op_desc->GetName(), op_desc->GetType(), "inputs size" + FmtToStr(op_desc->GetAllInputsSize()),
        "tensor size is " + FmtToStr(inputs.size())});
    GELOGE(PARAM_INVALID, "Tensor size: %zu, Inputs size: %zu", inputs.size(), op_desc->GetAllInputsSize());
    return PARAM_INVALID;
  }
  if (!outputs.empty() && (outputs.size() != op_desc->GetOutputsSize())) {
    ErrorManager::GetInstance().ATCReportErrMessage("E14001", {"opname", "optype", "value", "reason"},
        {op_desc->GetName(), op_desc->GetType(), "outputs size" + FmtToStr(op_desc->GetOutputsSize()),
        "tensor size is " + FmtToStr(outputs.size())});
    GELOGE(PARAM_INVALID, "Tensor size: %zu, Outputs size: %zu", outputs.size(), op_desc->GetOutputsSize());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status GeGenerator::BuildSingleOp(OpDescPtr &op_desc, const vector<GeTensor> &inputs, const vector<GeTensor> &outputs,
                                  const string &model_file_name, OpEngineType engine_type, ModelBufferData &model_buff,
                                  bool is_offline) {
  if (!is_offline) {
    (void)AttrUtils::SetBool(op_desc, ATTR_DYNAMIC_SHAPE_SINGLE_AICPU, true);
  }

  if (CheckForSingleOp(op_desc, inputs, outputs) != SUCCESS) {
    GELOGE(PARAM_INVALID, "input param is invalid when build single op!");
    return PARAM_INVALID;
  }
  OmgContext &omg_context = (impl_ == nullptr) ? domi::GetContext() : impl_->omg_context_;
  omg_context.is_dynamic_input = ContainsDynamicInpus(*op_desc);

  if (op_desc->HasAttr(ATTR_NAME_UNREGST_OPPATH)) {
    impl_->is_singleop_unregistered_ = true;
  }

  // 0. Save original attributes.
  OpDescPtr op_desc_tmp = AttrUtils::CloneOpDesc(op_desc);
  GE_CHECK_NOTNULL(op_desc_tmp);

  // 1. check engine type when compile online
  if (model_file_name == kFileNameSuffix) {
    Status ret = CheckEngineTypeSupport(op_desc, engine_type);
    if (ret != SUCCESS) {
      GELOGE(ret, "check engine type failed.");
      return ret;
    }
  }

  // 2. Create ComputeGraph.
  string name = ge::CurrentTimeInStr() + "_" + model_file_name;
  Graph graph;
  if (BuildSingleOpGraph(op_desc, inputs, outputs, name, graph) != ge::SUCCESS) {
    GELOGE(GRAPH_FAILED, "make graph fail.");
    return GRAPH_FAILED;
  }
  GELOGI("ATC parser success in single op build.");

  GeRootModelPtr ge_root_model = nullptr;
  GE_CHECK_NOTNULL_EXEC(impl_, return PARAM_INVALID);
  impl_->is_offline_ = is_offline;
  GE_CHK_STATUS_RET_NOLOG(impl_->BuildModel(graph, inputs, ge_root_model));
  map<string, GeAttrValue> op_attrs = op_desc_tmp->GetAllAttrs();
  GE_CHECK_NOTNULL(ge_root_model);
  GE_CHECK_NOTNULL(ge_root_model->GetRootGraph());
  map<string, GeModelPtr> name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
  if (name_to_ge_model.empty()) {
    GELOGE(PARAM_INVALID, "GetSubgraphInstanceNameToModel is empty.");
    return PARAM_INVALID;
  }
  GeModelPtr &ge_model = name_to_ge_model.begin()->second;
  GELOGD("The opType in op_desc_tmp is [%s]", op_desc_tmp->GetType().c_str());
  GE_CHK_STATUS_RET_NOLOG(impl_->SaveParams(ge_model, op_desc_tmp->GetType(), op_attrs, inputs, outputs));
  GE_CHK_STATUS_RET_NOLOG(impl_->SaveModel(model_file_name, ge_model, model_buff));
  return SUCCESS;
}

/**
 * @ingroup ge
 * @brief Compiling a single operator into an offline model
 * @param [in] OpDescPtr &op_desc: Operator description info that needs to be compiled into an offline model file
 * @param [in] vector<GeTensor> &inputs: Operator input data description information.
 * @param [in] vector<GeTensor> &outputs: Operator output data description information.
 * @param [in] const string &model_file_name: Offline model filename.
 * @return SUCCESS handle successfully / others handle failed
 */
Status GeGenerator::BuildSingleOpModel(OpDescPtr &op_desc, const vector<GeTensor> &inputs,
                                       const vector<GeTensor> &outputs, const string &model_file_name) {
  GELOGI("Start to build single op offline model.");
  ModelBufferData model_buff;
  OpEngineType engine_type = ENGINE_SYS;
  return BuildSingleOp(op_desc, inputs, outputs, model_file_name, engine_type, model_buff, true);
}

/**
 * @ingroup ge
 * @brief Compiling a single operator into online buffer
 * @param [in] OpDescPtr &op_desc: Operator description info that needs to be compiled into an offline model file
 * @param [in] vector<GeTensor> &inputs: Operator input data description information.
 * @param [in] vector<GeTensor> &outputs: Operator output data description information.
 * @param [in] engine_type: specific engine.
 * @param [out] ModelBufferData &Model_buff: Model_buff: model buffer of the op.
 * @return SUCCESS handle successfully / others handle failed
 */
Status GeGenerator::BuildSingleOpModel(OpDescPtr &op_desc, const vector<GeTensor> &inputs,
                                       const vector<GeTensor> &outputs, OpEngineType engine_type,
                                       ModelBufferData &model_buff) {
  GELOGI("Start to build single op online");
  return BuildSingleOp(op_desc, inputs, outputs, kFileNameSuffix, engine_type, model_buff, false);
}

Status GeGenerator::BuildSingleOpGraph(OpDescPtr &op_desc, const vector<GeTensor> &inputs,
                                       const vector<GeTensor> &outputs, std::string graph_name, Graph &graph) {
  ge::ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>(graph_name);
  GE_CHECK_NOTNULL_EXEC(compute_graph, return INTERNAL_ERROR);

  // 1. Add Node to ComputeGraph.
  NodePtr op_node = compute_graph->AddNode(op_desc);
  GE_CHECK_NOTNULL_EXEC(op_node, return INTERNAL_ERROR);

  // 2. Create InputData node.
  int32_t arg_index = 0;
  if (inputs.empty()) {
    for (const auto &input_desc : op_desc->GetAllInputsDescPtr()) {
      GE_CHECK_NOTNULL_EXEC(input_desc, return INTERNAL_ERROR);
      if (!IsNeedConnectInputOpForSingleOp(*input_desc)) {
        continue;
      }
      GE_CHK_STATUS_RET_NOLOG(AddInputs(compute_graph, op_node, *input_desc, arg_index, false));
      arg_index++;
    }
  } else {
    for (const auto &in_desc : inputs) {
      GeTensorDesc input_desc = in_desc.GetTensorDesc();
      GE_CHK_STATUS_RET_NOLOG(AddInputs(compute_graph, op_node, input_desc, arg_index, true));
      arg_index++;
    }
  }

  // 3. Create Output node.
  if (!outputs.empty()) {
    GE_CHK_STATUS_RET_NOLOG(AddOutputs(compute_graph, op_node, outputs));
  }

  // dump ComputeGraph node.
  compute_graph->Dump();
  graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);

  return SUCCESS;
}

Status GeGenerator::Impl::SaveParams(GeModelPtr &ge_model, const string &type, const map<string, GeAttrValue> &attrs,
                                     const vector<GeTensor> &inputs, const vector<GeTensor> &outputs) {
  GE_CHECK_NOTNULL_EXEC(ge_model, return PARAM_INVALID);
  GE_CHK_BOOL_EXEC_NOLOG(graph_manager_.SaveParams(*ge_model, type, attrs, inputs, outputs) == SUCCESS,
                         (void)graph_manager_.Finalize();
                         return FAILED);

  return SUCCESS;
}

Status GeGenerator::Impl::SaveModel(const string &file_name_prefix, GeModelPtr &model, ModelBufferData &model_buff) {
  // set atc version
  if (!SetAtcVersionInfo(*(model.get()))) {
    GELOGW("SetPackageVersionInfo of atc failed!");
  }
  // set opp version
  if (!SetOppVersionInfo(*(model.get()))) {
    GELOGW("SetPackageVersionInfo of ops failed!");
  }
  ModelHelper model_helper;
  model_helper.SetSaveMode(is_offline_);
  Status ret = model_helper.SaveToOmModel(model, save_param_, file_name_prefix, model_buff);
  if (ret != SUCCESS) {
    GELOGE(ret, "Save to om model failed");
    return ret;
  }
  return SUCCESS;
}

Status GeGenerator::Impl::SaveRootModel(const string &file_name_prefix, GeRootModelPtr &ge_root_model,
                                        ModelBufferData &model_buff) {
  bool is_unknown_shape = false;
  auto ret = ge_root_model->CheckIsUnknownShape(is_unknown_shape);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Check root model is unkonwn shape failed");
    return FAILED;
  }
  GELOGD("begin save root model, cur model is unkonwn shape model ? : %d", is_unknown_shape);
  GE_CHK_BOOL_EXEC(!ge_root_model->GetSubgraphInstanceNameToModel().empty(), return FAILED,
                   "ge root model has no sub model")
  GeModelPtr model_root = nullptr;
  if (is_unknown_shape) {
    model_root = make_shared<GeModel>();
    model_root->SetGraph(GraphUtils::CreateGraphFromComputeGraph(ge_root_model->GetRootGraph()));
    ge_root_model->SetSubgraphInstanceNameToModel(ge_root_model->GetRootGraph()->GetName(), model_root);
    model_root->SetName(ge_root_model->GetRootGraph()->GetName());
  } else {
    model_root = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  }
  // set atc version
  if (!SetAtcVersionInfo(*(model_root.get()))) {
    GELOGW("SetPackageVersionInfo of atc failed!");
  }
  // set opp version
  if (!SetOppVersionInfo(*(model_root.get()))) {
    GELOGW("SetPackageVersionInfo of ops failed!");
  }
  ModelHelper model_helper;
  model_helper.SetSaveMode(is_offline_);
  ret = model_helper.SaveToOmRootModel(ge_root_model, save_param_, file_name_prefix, model_buff, is_unknown_shape);
  if (ret != SUCCESS) {
    GELOGE(ret, "Save to om model failed");
    return ret;
  }
  return SUCCESS;
}

Status GeGenerator::Impl::BuildModel(const Graph &graph, const vector<GeTensor> &inputs,
                                     GeRootModelPtr &ge_root_model) {
  static std::atomic<GraphId> atomic_graph_id(0);
  auto graph_id = atomic_graph_id.fetch_add(1);
  const std::map<std::string, std::string> options;
  Status ret = graph_manager_.AddGraph(graph_id, graph, options, omg_context_);
  if (ret != SUCCESS) {
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED, "GraphManager add graph fail, graph id: %u", graph_id);
    (void)graph_manager_.Finalize();
    return GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED;
  }

  graph_manager_.SetOptionsRunGraphFlag(false);

  static std::atomic<uint64_t> atomic_session_id(0);
  auto session_id = atomic_session_id.fetch_add(1);
  if (is_singleop_unregistered_) {
    ret = graph_manager_.BuildGraphForUnregisteredOp(graph_id, inputs, ge_root_model, session_id);
  } else {
    ret = graph_manager_.BuildGraph(graph_id, inputs, ge_root_model, session_id);
  }

  if (ret != SUCCESS) {
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_BUILD_GRAPH_FAILED, "GraphManager build graph fail, graph id: %u", graph_id);
    VarManagerPool::Instance().RemoveVarManager(session_id);
    return GE_GENERATOR_GRAPH_MANAGER_BUILD_GRAPH_FAILED;
  }

  VarManagerPool::Instance().RemoveVarManager(session_id);

  return SUCCESS;
}

Status GeGenerator::Impl::GenerateInfershapeGraph(const Graph &graph) {
  static std::atomic<GraphId> atomic_graph_id(0);
  auto graph_id = atomic_graph_id.fetch_add(1);
  const std::map<std::string, std::string> options;
  Status ret = graph_manager_.AddGraph(graph_id, graph, options, omg_context_);
  if (ret != SUCCESS) {
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED, "GraphManager add graph failed, graph id: %u", graph_id);
    (void)graph_manager_.Finalize();
    return GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED;
  }

  ret = graph_manager_.GenerateInfershapeGraph(graph_id);
  if (ret != SUCCESS) {
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_BUILD_GRAPH_FAILED, "GraphManager generate graph failed");
    return GE_GENERATOR_GRAPH_MANAGER_BUILD_GRAPH_FAILED;
  }

  return SUCCESS;
}
}  // namespace ge
