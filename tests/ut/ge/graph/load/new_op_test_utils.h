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

#ifndef OME_REBUILD_OME_OP_TEST_UTILS_H
#define OME_REBUILD_OME_OP_TEST_UTILS_H

#include <gtest/gtest.h>
#include <memory>
#include <utility>

#include "common/fmk_types.h"
#include "common/helper/model_helper.h"
#include "common/op/attr_value_util.h"
#include "common/properties_manager.h"
#include "common/types.h"
#include "executor/ge_executor.h"
#include "graph/buffer.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_attr_value.h"
#include "graph/model_serialize.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "proto/ge_ir.pb.h"

#define protected public
#define private public
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/new_model_manager/davinci_model.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#undef protected
#undef private

using namespace domi;
using namespace ge;

class GlobalModelData {
 public:
  GlobalModelData() {}

  ~GlobalModelData() {
    if (data_.model_data != nullptr) {
      delete[](uint8_t *) data_.model_data;
      data_.model_data = nullptr;
    }
  }

  ge::ModelData data_;
};

static GlobalModelData g_modelData;

class OmeTestOpUtils {
 public:
  static void InitModel(std::shared_ptr<ge::DavinciModel> davinciModel) { InitModel(*davinciModel); }
  static ge::NodePtr GenNodeFromOpDesc(ge::OpDescPtr opDesc) {
    if (!opDesc) {
      return nullptr;
    }

    // return std::make_shared<ge::Node>(opDesc, nullptr);
    auto g = std::make_shared<ge::ComputeGraph>("g");
    return g->AddNode(std::move(opDesc));
  }

  static void AddInputOutputToTaskModel(std::shared_ptr<ge::Model> model,
                                        std::shared_ptr<domi::ModelTaskDef> modelTaskDef) {
    uint32_t stream_num111 = modelTaskDef->stream_num();
    uint32_t weights_num = modelTaskDef->weight_size();
    uint32_t mem_num = modelTaskDef->memory_size();

    int64_t memorySize = 0;
    int64_t weightSize = 0;
    (void)ge::AttrUtils::GetInt(model.get(), ATTR_MODEL_MEMORY_SIZE, memorySize);
    (void)ge::AttrUtils::GetInt(model.get(), ATTR_MODEL_WEIGHT_SIZE, weightSize);
    // Save memory_size/weight_size/stream_num/event_num to proto
    modelTaskDef->set_memory_size(memorySize);
    modelTaskDef->set_weight_size(weightSize);
    int64_t stream_num = 0;
    (void)ge::AttrUtils::GetInt(model.get(), ATTR_MODEL_STREAM_NUM, stream_num);
    modelTaskDef->set_stream_num(stream_num);

    ge::ComputeGraphPtr graph = ge::GraphUtils::GetComputeGraph(model->GetGraph());
    vector<ConstOpDescPtr> opDescPtrs;
    for (auto nodePtr : graph->GetAllNodes()) {
      if (nodePtr->GetType() == DATA_TYPE || nodePtr->GetType() == ANN_DATA_TYPE) {
        opDescPtrs.push_back(nodePtr->GetOpDesc());
        continue;
      }

      for (auto tensorDesc : nodePtr->GetOpDesc()->GetAllOutputsDescPtr()) {
        bool isOutput = false;
        ge::TensorUtils::GetOutputTensor(*tensorDesc, isOutput);
        if (isOutput) {
          // output Op and add to array
          opDescPtrs.push_back(nodePtr->GetOpDesc());
          break;
        }
      }
    }

    // save multi OpDescPtr to attr
    ge::ModelSerialize modelS;
    for (auto opDescPtr : opDescPtrs) {
      ge::Buffer buffer = modelS.SerializeOpDesc(opDescPtr);
      modelTaskDef->add_op(string(reinterpret_cast<const char *>(buffer.GetData()), buffer.GetSize()));
    }

    int64_t runMode = -1;
    for (auto nodePtr : graph->GetAllNodes()) {
      // TE CUSTOM op need to init
      if (ge::AttrUtils::GetInt(nodePtr->GetOpDesc(), ATTR_NAME_IMPLY_TYPE, runMode) &&
          runMode != (uint32_t)domi::ImplyType::BUILDIN && runMode != (uint32_t)domi::ImplyType::INVALID) {
        (*(modelTaskDef->mutable_attr()))["contain_custom"] = "1";
        break;
      }
    }
  }

  static void LoadStandardModelDataLocal(ge::ModelData &data) {
    static const std::string STANDARD_MODEL_DATA_PATH =
        "llt/framework/domi/ut/ome/test/data/standard_partition_model.txt";
    ge::proto::ModelDef modelDef;
    ReadProtoFromText(STANDARD_MODEL_DATA_PATH.c_str(), &modelDef);

    data.model_len = modelDef.ByteSizeLong();
    data.model_data = new uint8_t[data.model_len];
    modelDef.SerializePartialToArray(data.model_data, data.model_len);
  }
  static void InitModel(ge::DavinciModel &davinciModel) {
    ge::ModelData data;
    LoadStandardModelDataLocal(data);
    std::shared_ptr<ge::Model> model_ = std::make_shared<ge::Model>();
    ge::Model::Load((uint8_t *)data.model_data, data.model_len, *model_);

    GeModelPtr ge_model;
    ModelHelper::TransModelToGeModel(model_, ge_model);
    davinciModel.Assign(ge_model);

    if (data.model_data != nullptr) {
      delete[](uint8_t *) data.model_data;
    }
  }

  static void InitEmptyModel(ge::DavinciModel &davinciModel) {
    auto model = std::make_shared<ge::Model>();
    ge::AttrUtils::SetInt(model, ATTR_MODEL_MEMORY_SIZE, 81000000);
    ge::AttrUtils::SetInt(model, ATTR_MODEL_WEIGHT_SIZE, 4100000);
    ge::AttrUtils::SetInt(model, ATTR_MODEL_STREAM_NUM, 1);
    ge::AttrUtils::SetInt(model, ATTR_MODEL_EVENT_NUM, 1);
    ge::AttrUtils::SetInt(model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0x123);
    ge::AttrUtils::SetInt(model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0x456);
    ge::AttrUtils::SetInt(model, ATTR_MODEL_BATCH_NUM, 1);

    //        ge::AttrUtils::SetStr(model, ATTR_MODEL_TARGET_TYPE, "MINI"); // domi::MINI

    auto computeGraph = std::make_shared<ge::ComputeGraph>("graph");
    ge::GeAttrValue::BYTES buffer(4100000, 0);
    ge::AttrUtils::SetBytes(computeGraph, "weights_data", buffer);
    auto graph = ge::GraphUtils::CreateGraphFromComputeGraph(computeGraph);
    model->SetGraph(graph);

    GeModelPtr ge_model;
    ModelHelper::TransModelToGeModel(model, ge_model);

    davinciModel.Assign(ge_model);
  }

  static void InitModelWithoutMem(ge::DavinciModel &davinciModel) { InitModel(davinciModel); }

  static Status ModelLoadStub(const uint8_t *data, size_t len, ge::Model &model) {
    auto computeGraph = std::make_shared<ge::ComputeGraph>("graph");
    auto graph = ge::GraphUtils::CreateGraphFromComputeGraph(computeGraph);
    model.SetGraph(graph);
    return domi::SUCCESS;
  }
  static void InitDefaultTensorDesc(ge::GeTensorDesc &tensorDesc) {}
  static void AddInputDesc(ge::OpDescPtr opDesc, vector<int64_t> shape, ge::Format format, ge::DataType dataType,
                           int64_t dataSize = 0) {
    ge::GeTensorDesc tensorDesc(ge::GeShape(shape), format, dataType);
    InitDefaultTensorDesc(tensorDesc);
    ge::TensorUtils::SetSize(tensorDesc, dataSize);
    opDesc->AddInputDesc(tensorDesc);
  }
  static void AddOutputDesc(ge::OpDescPtr opDesc, vector<int64_t> shape, ge::Format format, ge::DataType dataType,
                            int64_t dataSize = 0) {
    ge::GeTensorDesc tensorDesc(ge::GeShape(shape), format, dataType);
    InitDefaultTensorDesc(tensorDesc);
    ge::TensorUtils::SetSize(tensorDesc, dataSize);
    opDesc->AddOutputDesc(tensorDesc);
  }
  static void AddWeight(ge::NodePtr nodePtr, uint8_t *data, size_t dataLen, vector<int64_t> shape = {},
                        ge::Format format = ge::FORMAT_NCHW, ge::DataType dataType = ge::DT_FLOAT) {
    ge::GeTensorDesc tensorDesc(ge::GeShape(shape), format, dataType);

    vector<ge::GeTensorPtr> weigths = ge::OpDescUtils::MutableWeights(nodePtr);
    weigths.push_back(std::make_shared<ge::GeTensor>(tensorDesc, data, dataLen));
    ge::OpDescUtils::SetWeights(nodePtr, weigths);
  }
  static ge::OpDescPtr CreateOpDesc() {
    auto opDesc = std::make_shared<ge::OpDesc>();
    return opDesc;
  }
};

class OmeTestOpDescBuilder {
 public:
  OmeTestOpDescBuilder(ge::OpDescPtr orgOpDesc = nullptr) : orgOpDesc_(orgOpDesc) {
    if (orgOpDesc_) {
      streamId_ = orgOpDesc_->GetStreamId();
    }
  }

  OmeTestOpDescBuilder &SetStreamId(int64_t streamId) {
    streamId_ = streamId;
    return *this;
  }
  OmeTestOpDescBuilder &SetWorkspace(vector<int64_t> workspace) {
    workspace_ = workspace;
    return *this;
  }
  OmeTestOpDescBuilder &SetWorkspaceBytes(vector<int64_t> workspaceBytes) {
    workspaceBytes_ = workspaceBytes;
    return *this;
  }
  OmeTestOpDescBuilder &SetType(const string &type) {
    type_ = type;
    return *this;
  }
  OmeTestOpDescBuilder &SetName(const string &name) {
    name_ = name;
    return *this;
  }
  OmeTestOpDescBuilder &SetInputs(vector<int64_t> inputs) {
    inputsDataOffeset_ = inputs;
    return *this;
  }
  OmeTestOpDescBuilder &AddInput(int64_t input) {
    inputsDataOffeset_.push_back(input);
    return *this;
  }
  OmeTestOpDescBuilder &SetOutputs(vector<int64_t> outputs) {
    outputsDataOffeset_ = outputs;
    return *this;
  }
  OmeTestOpDescBuilder &AddOutput(int64_t output) {
    outputsDataOffeset_.push_back(output);
    return *this;
  }

  OmeTestOpDescBuilder &SetEventId(int64_t eventId) {
    eventId_ = eventId;
    return *this;
  }

  OmeTestOpDescBuilder &Setscopeid(int64_t scopeid) {
    scopeid_ = scopeid;
    return *this;
  }

  ge::GeTensorDesc &AddInputDesc(vector<int64_t> shape, ge::Format format, ge::DataType dataType,
                                 int64_t dataSize = 0) {
    ge::GeTensorDesc tensorDesc(ge::GeShape(shape), format, dataType);
    OmeTestOpUtils::InitDefaultTensorDesc(tensorDesc);
    ge::TensorUtils::SetSize(tensorDesc, dataSize);
    inputTensorDescs.push_back(tensorDesc);
    return inputTensorDescs.back();
  }
  ge::GeTensorDesc &AddInputDesc(vector<int64_t> shape, ge::Format format, ge::DataType dataType, int64_t realdimcnt,
                                 int64_t dataSize) {
    ge::GeTensorDesc tensorDesc(ge::GeShape(shape), format, dataType);
    OmeTestOpUtils::InitDefaultTensorDesc(tensorDesc);
    ge::TensorUtils::SetSize(tensorDesc, dataSize);
    ge::TensorUtils::SetRealDimCnt(tensorDesc, realdimcnt);
    inputTensorDescs.push_back(tensorDesc);
    return inputTensorDescs.back();
  }

  ge::GeTensorDesc &AddOutputDesc(vector<int64_t> shape, ge::Format format, ge::DataType dataType,
                                  int64_t dataSize = 0) {
    ge::GeTensorDesc tensorDesc(ge::GeShape(shape), format, dataType);
    OmeTestOpUtils::InitDefaultTensorDesc(tensorDesc);
    ge::TensorUtils::SetSize(tensorDesc, dataSize);
    outputTensorDescs.push_back(tensorDesc);
    return outputTensorDescs.back();
  }

  ge::GeTensorDesc &AddOutputDesc(vector<int64_t> shape, ge::Format format, ge::DataType dataType, int64_t realdimcnt,
                                  int64_t dataSize) {
    ge::GeTensorDesc tensorDesc(ge::GeShape(shape), format, dataType);
    OmeTestOpUtils::InitDefaultTensorDesc(tensorDesc);
    ge::TensorUtils::SetSize(tensorDesc, dataSize);
    ge::TensorUtils::SetRealDimCnt(tensorDesc, realdimcnt);
    outputTensorDescs.push_back(tensorDesc);
    return outputTensorDescs.back();
  }

  ge::GeTensorPtr AddWeight(uint8_t *data, size_t dataLen, vector<int64_t> shape = {},
                            ge::Format format = ge::FORMAT_NCHW, ge::DataType dataType = ge::DT_FLOAT) {
    ge::GeTensorDesc tensorDesc(ge::GeShape(shape), format, dataType);

    weights_.emplace_back(std::make_shared<ge::GeTensor>(tensorDesc, data, dataLen));
    return weights_.back();
  }
  ge::NodePtr Finish() {
    ge::OpDescPtr opDesc;
    if (orgOpDesc_) {
      opDesc = orgOpDesc_;
    } else {
      opDesc = OmeTestOpUtils::CreateOpDesc();  // std::make_shared<ge::OpDesc>(name_, type_);
    }
    if (!type_.empty()) {
      opDesc->SetType(type_);
    }
    if (!name_.empty()) {
      opDesc->SetName(name_);
    }

    opDesc->SetStreamId(streamId_);
    ge::AttrUtils::SetInt(opDesc, "id", 1);

    if (eventId_ != -1) {
      ge::AttrUtils::SetInt(opDesc, SEND_ATTR_EVENT_ID, eventId_);
    }

    if (scopeid_ != -1) {
      ge::AttrUtils::SetInt(opDesc, "fusion_scope", scopeid_);
    }
    // ge::AttrUtils::SetInt(opDesc, ATTR_NAME_STREAM_ID, streamId_);
    // if(!inputsDataOffeset_.empty())
    {
      vector<int64_t> inputs;
      inputs = opDesc->GetInputOffset();
      inputs.insert(inputs.end(), inputsDataOffeset_.begin(), inputsDataOffeset_.end());

      opDesc->SetInputOffset(inputs);
    }
    // if(!outputsDataOffeset_.empty())
    {
      vector<int64_t> outputs;
      outputs = opDesc->GetOutputOffset();
      outputs.insert(outputs.end(), outputsDataOffeset_.begin(), outputsDataOffeset_.end());

      opDesc->SetOutputOffset(outputs);
    }
    // if(!workspace_.empty())
    {
      vector<int64_t> workspace = opDesc->GetWorkspace();
      workspace.insert(workspace.end(), workspace_.begin(), workspace_.end());

      opDesc->SetWorkspace(workspace);
    }
    // if(!workspaceBytes_.empty())
    {
      vector<int64_t> workspaceBytes;
      workspaceBytes = opDesc->GetWorkspaceBytes();
      workspaceBytes.insert(workspaceBytes.end(), workspaceBytes_.begin(), workspaceBytes_.end());

      opDesc->SetWorkspaceBytes(workspaceBytes);
    }
    for (auto &tensorDesc : inputTensorDescs) {
      opDesc->AddInputDesc(tensorDesc);
    }
    for (auto &tensorDesc : outputTensorDescs) {
      opDesc->AddOutputDesc(tensorDesc);
    }

    static std::shared_ptr<ge::ComputeGraph> graph;
    // clear graph
    graph = std::make_shared<ge::ComputeGraph>("g");

    ge::NodePtr nodeOp = graph->AddNode(opDesc);
    // for(int i=0; i < inputTensorDescs.size(); i++)
    for (int i = 0; i < opDesc->GetInputsSize(); i++) {
      ge::OpDescPtr srcOpDesc = std::make_shared<ge::OpDesc>();

      ge::GeTensorDesc srcOutDesc;
      srcOpDesc->AddOutputDesc(srcOutDesc);

      ge::NodePtr srcNode = graph->AddNode(srcOpDesc);
      if (nullptr == srcNode) {
        DOMI_LOGE("Finish: nullptr == srcNode");
      }
      Status res = ge::GraphUtils::AddEdge(srcNode->GetOutDataAnchor(0), nodeOp->GetInDataAnchor(i));
      if (domi::SUCCESS != res) {
        DOMI_LOGE("Finish: GraphUtils::AddEdge failed");
      }
      // ge::NodePtr srcNode = node->GetOwnerComputeGraph()->AddNodeFront(srcOpDesc);
      // node->AddLinkFrom(srcNode);
    }

    {
      vector<ge::GeTensorPtr> weights;
      weights = ge::OpDescUtils::MutableWeights(nodeOp);
      weights.insert(weights.end(), weights_.begin(), weights_.end());

      ge::OpDescUtils::SetWeights(nodeOp, weights);
    }

    *this = OmeTestOpDescBuilder(opDesc);  // clear up

    return nodeOp;
  }

 private:
  ge::OpDescPtr orgOpDesc_;
  int64_t streamId_ = 0;
  string type_;
  string name_;
  vector<int64_t> inputsDataOffeset_;   // input
  vector<int64_t> outputsDataOffeset_;  // output
  vector<ge::GeTensorDesc> inputTensorDescs;
  vector<ge::GeTensorDesc> outputTensorDescs;
  vector<int64_t> workspace_;
  vector<int64_t> workspaceBytes_;
  vector<ge::GeTensorPtr> weights_;
  int64_t eventId_ = -1;
  int64_t scopeid_ = -1;

  // std::shared_ptr<ge::ComputeGraph> graph_;
};

#endif  // OME_REBUILD_OME_OP_TEST_UTILS_H
