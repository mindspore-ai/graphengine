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

using std::map;
using std::string;
using std::vector;

namespace {
const char *const kAttrOpType = "op_type";
}

namespace ge {
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
  GELOGI("Start to get ops proto path schedule");
  const char *path_env = std::getenv("ASCEND_OPP_PATH");
  if (path_env != nullptr) {
    string path = path_env;
    string file_path = RealPath(path.c_str());
    if (file_path.empty()) {
      GELOGE(FAILED, "File path %s is invalid.", path.c_str());
      return;
    }
    opsproto_path = (path + "/op_proto/built-in/" + ":") + (path + "/op_proto/custom/");
    GELOGI("Get opsproto so path from env : %s", path.c_str());
    return;
  }
  string path_base = PluginManager::GetPath();
  GELOGI("path_base is %s", path_base.c_str());
  path_base = path_base.substr(0, path_base.rfind('/'));
  path_base = path_base.substr(0, path_base.rfind('/') + 1);
  opsproto_path = (path_base + "ops/op_proto/built-in/" + ":") + (path_base + "ops/op_proto/custom/");
}

class GeGenerator::Impl {
 public:
  Status BuildModel(const Graph &graph, const vector<GeTensor> &inputs, GraphId &graph_id,
                    vector<GeModelPtr> &ge_models);

  Status SaveModel(const string &file_name_prefix, vector<GeModelPtr> models);

  Status SaveParams(GeModelPtr &ge_model, const string &type, const map<string, GeAttrValue> &attrs,
                    const vector<GeTensor> &inputs, const vector<GeTensor> &outputs);

  GraphManager graph_manager_;
  SaveParam save_param_;
};

Status GeGenerator::Initialize(const map<string, string> &options) {
  impl_ = ge::MakeShared<Impl>();
  if (impl_ == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Make shared failed");
    return MEMALLOC_FAILED;
  }
  string opsproto_path;
  GetOpsProtoPath(opsproto_path);
  GELOGI("opsproto_path is %s", opsproto_path.c_str());
  OpsProtoManager *manager = OpsProtoManager::Instance();
  map<string, string> option_tmp;
  option_tmp.emplace(std::pair<string, string>(string("ge.opsProtoLibPath"), opsproto_path));
  (void)manager->Initialize(option_tmp);

  Status ret = impl_->graph_manager_.Initialize(options);
  if (ret != SUCCESS) {
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_INIT_FAILED, "Graph manager initialize failed");
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
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_FINALIZE_FAILED, "Graph manager finalize failed");
    return GE_GENERATOR_GRAPH_MANAGER_FINALIZE_FAILED;
  }
  return SUCCESS;
}

Status GeGenerator::GenerateOfflineModel(const Graph &graph, const string &file_name_prefix,
                                         const vector<GeTensor> &inputs) {
  GELOGI("Start to GenerateOfflineModel.");
  GraphId graph_id;
  vector<GeModelPtr> ge_models;
  GE_CHECK_NOTNULL_EXEC(impl_, return PARAM_INVALID);

  string model_name;
  auto compute_graph = GraphUtils::GetComputeGraph(graph);
  if (compute_graph == nullptr) {
    GELOGW("Get compute graph fail.");
  } else {
    model_name = compute_graph->GetName();
  }

  Status ret = impl_->BuildModel(graph, inputs, graph_id, ge_models);
  if (ret != SUCCESS) {
    GELOGE(ret, "Build model failed");
    if (impl_->graph_manager_.Finalize() != SUCCESS) {
      GELOGE(FAILED, "graph_manager finalize fail.");
    }
    return ret;
  }

  if (!model_name.empty() && !ge_models.empty()) {
    ge_models[0]->SetName(model_name);
  }

  ret = impl_->SaveModel(file_name_prefix, ge_models);
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
  GE_CHECK_NOTNULL_EXEC(op_desc, return PARAM_INVALID);
  if (!inputs.empty() && (inputs.size() != op_desc->GetInputsSize())) {
    GELOGE(PARAM_INVALID, "Tensor size: %zu, Inputs size:%zu", inputs.size(), op_desc->GetInputsSize());
    return PARAM_INVALID;
  }
  if (!outputs.empty() && (outputs.size() != op_desc->GetOutputsSize())) {
    GELOGE(PARAM_INVALID, "Tensor size: %zu, Outputs size:%zu", outputs.size(), op_desc->GetOutputsSize());
    return PARAM_INVALID;
  }

  // 1. Create ComputeGraph.
  string name = ge::CurrentTimeInStr() + "_" + model_file_name;
  ge::ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>(name);
  if (compute_graph == nullptr) {
    return INTERNAL_ERROR;
  }
  GE_CHECK_NOTNULL_EXEC(compute_graph, return INTERNAL_ERROR);

  // 2. Add Node to ComputeGraph.
  NodePtr op_node = compute_graph->AddNode(op_desc);
  GE_CHECK_NOTNULL_EXEC(op_node, return INTERNAL_ERROR);

  // 3. Create InputData node.
  int64_t in_size = static_cast<int64_t>(op_desc->GetInputsSize());
  GE_CHK_BOOL_EXEC(AttrUtils::SetInt(op_desc, ATTR_NAME_N, in_size), return FAILED, "Op[%s] Set N fail",
                   op_desc->GetName().c_str());
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

  // 4. Create Output node.
  if (!outputs.empty()) {
    GE_CHK_STATUS_RET_NOLOG(AddOutputs(compute_graph, op_node, outputs));
  }

  // dump ComputeGraph.
  compute_graph->Dump();
  Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  GELOGI("ATC parser success.");

  GraphId graph_id;
  vector<GeModelPtr> ge_models;
  GE_CHECK_NOTNULL_EXEC(impl_, return PARAM_INVALID);
  GE_CHK_STATUS_RET_NOLOG(impl_->BuildModel(graph, inputs, graph_id, ge_models));

  if (!ge_models.empty()) {
    map<string, GeAttrValue> op_attrs = op_desc->GetAllAttrs();
    GE_CHK_STATUS_RET_NOLOG(impl_->SaveParams(ge_models[0], op_desc->GetType(), op_attrs, inputs, outputs));
  }

  GE_CHK_STATUS_RET_NOLOG(impl_->SaveModel(model_file_name, ge_models));
  return SUCCESS;
}

Status GeGenerator::Impl::SaveParams(GeModelPtr &ge_model, const string &type, const map<string, GeAttrValue> &attrs,
                                     const vector<GeTensor> &inputs, const vector<GeTensor> &outputs) {
  GE_CHECK_NOTNULL_EXEC(ge_model, return PARAM_INVALID);
  GE_CHK_BOOL_EXEC_NOLOG(graph_manager_.SaveParams(*ge_model, type, attrs, inputs, outputs) == SUCCESS,
                         graph_manager_.Finalize();
                         return FAILED);

  return SUCCESS;
}

Status GeGenerator::Impl::SaveModel(const string &file_name_prefix, vector<GeModelPtr> models) {
  // to be change to ModelHelper interface
  if (models.empty()) {
    GELOGE(FAILED, "models are empty.");
    return FAILED;
  }

  ModelHelper model_helper;
  Status ret = model_helper.SaveToOmModel(models[0], save_param_, file_name_prefix);
  if (ret != SUCCESS) {
    GELOGE(ret, "Save to Om model failed");
    return ret;
  }
  return SUCCESS;
}

Status GeGenerator::Impl::BuildModel(const Graph &graph, const vector<GeTensor> &inputs, GraphId &graph_id,
                                     vector<GeModelPtr> &ge_models) {
  static GraphId id = 0;
  const std::map<std::string, std::string> options;
  Status ret = graph_manager_.AddGraph(id, graph, options);
  if (ret != SUCCESS) {
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED, "graphManager AddGraph failed, id: %u", id);
    graph_manager_.Finalize();
    return GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED;
  }

  GELOGI("models' inputs.size()=%zu", inputs.size());
  ret = graph_manager_.BuildGraph(id, inputs, ge_models);
  if (ret != SUCCESS) {
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_BUILD_GRAPH_FAILED, "graphManager BuildGraph failed, id: %u", id);
    return GE_GENERATOR_GRAPH_MANAGER_BUILD_GRAPH_FAILED;
  }

  graph_id = id;
  id += 1;

  return SUCCESS;
}
}  // namespace ge
