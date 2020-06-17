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

#include "generator/ge_generator.h"
#include "common/ge/ge_util.h"
#include "common/ge/plugin_manager.h"
#include "common/helper/model_helper.h"
#include "common/helper/om_file_helper.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "ge/ge_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/manager/graph_manager.h"
#include "graph/opsproto_manager.h"
#include "graph/utils/graph_utils.h"
#include "model/ge_model.h"
#include "init/gelib.h"

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
    GELOGE(FAILED, "CheckEngineType: engine type: %d not support", static_cast<int>(engine_type));
    return FAILED;
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
    GELOGE(FAILED, "CheckEngineType:Can not find ops kernel,engine name: %s.", op_engine_name.c_str());
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
      GELOGE(FAILED, "CheckEngineType: check support failed, Op type %s of ops kernel %s is unsupported, reason:%s",
             op_desc->GetType().c_str(), kernel_name.c_str(), unsupported_reason.c_str());
      return FAILED;
    }
  } else {
    GELOGE(FAILED,
           "CheckEngineType:Can not find any supported ops kernel info store by kernel_name %s,"
           "op type is %s, op name is %s",
           kernel_name.c_str(), op_desc->GetType().c_str(), op_desc->GetName().c_str());
  }
  return FAILED;
}

static Status AddInputs(const ComputeGraphPtr &graph, const NodePtr &node, const GeTensorDesc &tensor, int32_t index,
                        bool attr) {
  GE_CHECK_NOTNULL_EXEC(graph, return PARAM_INVALID);
  GE_CHECK_NOTNULL_EXEC(node, return PARAM_INVALID);
  string op_type;
  if (!AttrUtils::GetStr(tensor, kAttrOpType, op_type) || op_type.empty()) {
    op_type = DATA;
  }

  string op_name = node->GetName() + "_in_" + std::to_string(index);
  OpDescPtr data_op = MakeShared<ge::OpDesc>(op_name, op_type);
  if (data_op == nullptr) {
    return FAILED;
  }

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
  OpDescPtr op_desc = MakeShared<ge::OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  if (op_desc == nullptr) {
    return FAILED;
  }
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

class GeGenerator::Impl {
 public:
  Status BuildModel(const Graph &graph, const vector<GeTensor> &inputs, GraphId &graph_id, GeRootModelPtr &ge_models);

  Status SaveModel(const string &file_name_prefix, GeModelPtr &models, ModelBufferData &model);

  Status SaveParams(GeModelPtr &ge_model, const string &type, const map<string, GeAttrValue> &attrs,
                    const vector<GeTensor> &inputs, const vector<GeTensor> &outputs);

  Status GenerateInfershapeGraph(const Graph &graph, GraphId &graph_id);

  GraphManager graph_manager_;
  SaveParam save_param_;
  bool is_offline_ = true;
};

Status GeGenerator::Initialize(const map<string, string> &options) {
  impl_ = ge::MakeShared<Impl>();
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
  GraphId graph_id;
  GE_CHECK_NOTNULL_EXEC(impl_, return PARAM_INVALID);

  Status ret = impl_->GenerateInfershapeGraph(graph, graph_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "Dump infershape json failed");
    if (impl_->graph_manager_.Finalize() != SUCCESS) {
      GELOGE(FAILED, "graph_manager finalize fail.");
    }
    return ret;
  }
  GELOGI("GenerateInfershapeGraph success.");
  return SUCCESS;
}

Status GeGenerator::GenerateModel(const Graph &graph, const string &file_name_prefix, const vector<GeTensor> &inputs,
                                  ModelBufferData &model, bool is_offline) {
  GraphId graph_id;
  GeRootModelPtr ge_root_model = nullptr;
  GE_CHECK_NOTNULL_EXEC(impl_, return PARAM_INVALID);
  // using output as model_name (ignore ".om")
  int start_position = file_name_prefix.find_last_of('/') + 1;
  int end_position = file_name_prefix.length() - 3;
  const string model_name = file_name_prefix.substr(start_position, end_position - start_position);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(model_name.empty(), return PARAM_INVALID, "om name is not valid!");
  impl_->is_offline_ = is_offline;
  Status ret = impl_->BuildModel(graph, inputs, graph_id, ge_root_model);
  if (ret != SUCCESS) {
    GELOGE(ret, "Build model failed");
    if (impl_->graph_manager_.Finalize() != SUCCESS) {
      GELOGE(FAILED, "graph_manager finalize fail.");
    }
    return ret;
  }
  GE_CHECK_NOTNULL(ge_root_model);
  GE_CHECK_NOTNULL(ge_root_model->GetRootGraph());
  map<string, GeModelPtr> name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
  GeModelPtr &ge_model = name_to_ge_model[ge_root_model->GetRootGraph()->GetName()];

  GE_RETURN_WITH_LOG_IF_FALSE(ge_model != nullptr, "ge_model can not be null");
  ge_model->SetName(model_name);
  ret = impl_->SaveModel(file_name_prefix, ge_model, model);
  if (ret != SUCCESS) {
    GELOGE(ret, "Save model failed");
    if (impl_->graph_manager_.Finalize() != SUCCESS) {
      GELOGE(FAILED, "graph_manager finalize fail.");
    }
    return ret;
  }
  GELOGI("GenerateOfflineModel success.");
  return SUCCESS;
}

Status GeGenerator::BuildSingleOp(OpDescPtr &op_desc, const vector<GeTensor> &inputs, const vector<GeTensor> &outputs,
                                  const string &model_file_name, OpEngineType engine_type, ModelBufferData &model_buff,
                                  bool is_offline) {
  GE_CHECK_NOTNULL_EXEC(op_desc, return PARAM_INVALID);
  if (!inputs.empty() && (inputs.size() != op_desc->GetInputsSize())) {
    GELOGE(PARAM_INVALID, "Tensor size: %zu, Inputs size:%zu", inputs.size(), op_desc->GetInputsSize());
    return PARAM_INVALID;
  }
  if (!outputs.empty() && (outputs.size() != op_desc->GetOutputsSize())) {
    GELOGE(PARAM_INVALID, "Tensor size: %zu, Outputs size:%zu", outputs.size(), op_desc->GetOutputsSize());
    return PARAM_INVALID;
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
  ge::ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>(name);
  if (compute_graph == nullptr) {
    return INTERNAL_ERROR;
  }
  GE_CHECK_NOTNULL_EXEC(compute_graph, return INTERNAL_ERROR);

  // 3. Add Node to ComputeGraph.
  NodePtr op_node = compute_graph->AddNode(op_desc);
  GE_CHECK_NOTNULL_EXEC(op_node, return INTERNAL_ERROR);

  // 4. Create InputData node.
  int32_t arg_index = 0;
  if (inputs.empty()) {
    for (const auto &input_desc : op_desc->GetAllInputsDescPtr()) {
      GE_CHECK_NOTNULL_EXEC(input_desc, return INTERNAL_ERROR);
      GE_CHK_STATUS_RET_NOLOG(AddInputs(compute_graph, op_node, *input_desc, arg_index, false));
      arg_index++;
    }
  } else {
    for (const auto &in_desc : inputs) {
      const GeTensorDesc input_desc = in_desc.GetTensorDesc();
      GE_CHK_STATUS_RET_NOLOG(AddInputs(compute_graph, op_node, input_desc, arg_index, true));
      arg_index++;
    }
  }

  // 5. Create Output node.
  if (!outputs.empty()) {
    GE_CHK_STATUS_RET_NOLOG(AddOutputs(compute_graph, op_node, outputs));
  }

  // dump ComputeGraph.
  compute_graph->Dump();
  Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  GELOGI("ATC parser success in single op schedule.");

  GraphId graph_id;
  GeRootModelPtr ge_root_model = nullptr;
  GE_CHECK_NOTNULL_EXEC(impl_, return PARAM_INVALID);
  impl_->is_offline_ = is_offline;
  GE_CHK_STATUS_RET_NOLOG(impl_->BuildModel(graph, inputs, graph_id, ge_root_model));
  map<string, GeAttrValue> op_attrs = op_desc_tmp->GetAllAttrs();
  GE_CHECK_NOTNULL(ge_root_model);
  GE_CHECK_NOTNULL(ge_root_model->GetRootGraph());
  map<string, GeModelPtr> name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
  GeModelPtr &ge_model = name_to_ge_model[ge_root_model->GetRootGraph()->GetName()];
  GELOGD("The opType in op_desc_tmp is: %s", op_desc_tmp->GetType().c_str());
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
  GELOGI("Start to Build Single Op Offline Model.");
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
  GELOGI("Start to Build Single Op Online");
  return BuildSingleOp(op_desc, inputs, outputs, kFileNameSuffix, engine_type, model_buff, false);
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
  ModelHelper model_helper;
  model_helper.SetSaveMode(is_offline_);
  Status ret = model_helper.SaveToOmModel(model, save_param_, file_name_prefix, model_buff);
  if (ret != SUCCESS) {
    GELOGE(ret, "Save to Om model failed");
    return ret;
  }
  return SUCCESS;
}

Status GeGenerator::Impl::BuildModel(const Graph &graph, const vector<GeTensor> &inputs, GraphId &graph_id,
                                     GeRootModelPtr &ge_root_model) {
  static GraphId id = 0;
  const std::map<std::string, std::string> options;
  Status ret = graph_manager_.AddGraph(id, graph, options);
  if (ret != SUCCESS) {
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED, "GraphManager add graph failed, id: %u", id);
    (void)graph_manager_.Finalize();
    return GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED;
  }

  GELOGI("models inputs.size()=%zu", inputs.size());
  graph_manager_.SetOptionsRunGraphFlag(false);
  ret = graph_manager_.BuildGraph(id, inputs, ge_root_model);
  if (ret != SUCCESS) {
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_BUILD_GRAPH_FAILED, "GraphManager build graph failed, id: %u", id);
    return GE_GENERATOR_GRAPH_MANAGER_BUILD_GRAPH_FAILED;
  }

  graph_id = id;
  id += 1;

  return SUCCESS;
}

Status GeGenerator::Impl::GenerateInfershapeGraph(const Graph &graph, GraphId &graph_id) {
  static GraphId id = 0;
  const std::map<std::string, std::string> options;
  Status ret = graph_manager_.AddGraph(id, graph, options);
  if (ret != SUCCESS) {
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED, "graphManager add graph failed, id: %u", id);
    (void)graph_manager_.Finalize();
    return GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED;
  }

  ret = graph_manager_.GenerateInfershapeGraph(id);
  if (ret != SUCCESS) {
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_BUILD_GRAPH_FAILED, "GraphManager BuildGraph failed, id: %u", id);
    return GE_GENERATOR_GRAPH_MANAGER_BUILD_GRAPH_FAILED;
  }

  graph_id = id;
  id += 1;

  return SUCCESS;
}

}  // namespace ge
