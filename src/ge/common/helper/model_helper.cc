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

#include "framework/common/helper/model_helper.h"

#include "common/ge/ge_util.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/debug/log.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/omg/version.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/new_model_manager/davinci_model_parser.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"

using domi::ModelTaskDef;
using std::string;

namespace {
const int64_t kOriginalOmPartitionNum = 1;
}

namespace ge {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ModelHelper::~ModelHelper() { (void)ReleaseLocalModelData(); }

Status ModelHelper::SaveModelPartition(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, ModelPartitionType type,
                                       const uint8_t *data, size_t size) {
  if (size < 1 || size > UINT32_MAX) {
    GELOGE(PARAM_INVALID, "Add model partition failed, partition size %zu invalid", size);
    return PARAM_INVALID;
  }
  if (data == nullptr) {
    GELOGE(PARAM_INVALID, "Add model partition failed, data is null");
    return PARAM_INVALID;
  }
  ModelPartition partition_model;
  partition_model.data = const_cast<uint8_t *>(data);
  partition_model.size = static_cast<uint32_t>(size);
  partition_model.type = type;
  if (om_file_save_helper->AddPartition(partition_model) != SUCCESS) {
    GELOGE(PARAM_INVALID, "Add model partition failed, partition size %zu", size);
    return PARAM_INVALID;
  }
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status ModelHelper::SaveToOmModel(const GeModelPtr &ge_model,
                                                                                   const SaveParam &save_param,
                                                                                   const std::string &output_file,
                                                                                   ModelBufferData &model) {
  if (output_file.empty()) {
    GELOGE(FAILED, "GraphBuilder SaveModel received invalid file name prefix");
    return FAILED;
  }

  GE_IF_BOOL_EXEC(ge_model == nullptr, GELOGE(FAILED, "Ge_model is nullptr"); return FAILED);
  std::shared_ptr<OmFileSaveHelper> om_file_save_helper = ge::MakeShared<OmFileSaveHelper>();
  GE_CHECK_NOTNULL(om_file_save_helper);
  ModelPtr model_tmp = ge::MakeShared<ge::Model>(ge_model->GetName(), ge_model->GetPlatformVersion());
  if (model_tmp == nullptr) {
    GELOGE(FAILED, "Create Model %s Ptr failed", ge_model->GetName().c_str());
    return FAILED;
  }
  model_tmp->SetGraph(ge_model->GetGraph());
  model_tmp->SetVersion(ge_model->GetVersion());
  model_tmp->SetAttr(ge_model->MutableAttrMap());

  ge::Buffer model_buffer;
  (void)model_tmp->Save(model_buffer);
  GELOGI("MODEL_DEF size is %zu", model_buffer.GetSize());
  if (model_buffer.GetSize() > 0) {
    if (SaveModelPartition(om_file_save_helper, ModelPartitionType::MODEL_DEF, model_buffer.GetData(),
                           model_buffer.GetSize()) != SUCCESS) {
      GELOGE(PARAM_INVALID, "Add model graph partition failed");
      return PARAM_INVALID;
    }
  }
  auto ge_model_weight = ge_model->GetWeight();
  GELOGI("WEIGHTS_DATA size is %zu , %p", ge_model_weight.GetSize(), ge_model_weight.GetData());
  if (SaveModelPartition(om_file_save_helper, ModelPartitionType::WEIGHTS_DATA, ge_model_weight.GetData(),
                         ge_model_weight.GetSize()) != SUCCESS) {
    GELOGW("Add weight partition failed");  // weight is not necessary
  }

  TBEKernelStore tbe_kernel_store = ge_model->GetTBEKernelStore();
  GELOGI("TBE_KERNELS size is %zu", tbe_kernel_store.DataSize());
  if (tbe_kernel_store.DataSize() > 0) {
    if (SaveModelPartition(om_file_save_helper, ModelPartitionType::TBE_KERNELS, tbe_kernel_store.Data(),
                           tbe_kernel_store.DataSize()) != SUCCESS) {
      GELOGE(PARAM_INVALID, "Add tbe kernel partition failed");
      return PARAM_INVALID;
    }
  }

  // no need to check value, DATA->NetOutput
  (void)tbe_kernel_store.Load(tbe_kernel_store.Data(), tbe_kernel_store.DataSize());

  std::shared_ptr<ModelTaskDef> model_task_def = ge_model->GetModelTaskDefPtr();
  if (model_task_def == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Create model task def ptr failed");
    return FAILED;
  }
  size_t partition_task_size = model_task_def->ByteSizeLong();
  GE_IF_BOOL_EXEC(partition_task_size == 0 || partition_task_size > INT_MAX,
                  GELOGE(FAILED, "Model_def's byte size (%zu) is invalid!", partition_task_size);
                  return FAILED);

  ge::Buffer task_buffer(partition_task_size);
  if (task_buffer.GetSize() == 0) {
    GELOGE(MEMALLOC_FAILED, "Alloc model task def buffer failed");
    return MEMALLOC_FAILED;
  }
  (void)model_task_def->SerializePartialToArray(task_buffer.GetData(), static_cast<int>(partition_task_size));

  GELOGI("TASK_INFO op_size:%d, stream_num:%u", model_task_def->op().size(), model_task_def->stream_num());
  GELOGI("TASK_INFO size is %zu", partition_task_size);

  if (SaveModelPartition(om_file_save_helper, ModelPartitionType::TASK_INFO, task_buffer.GetData(),
                         partition_task_size) != SUCCESS) {
    GELOGE(PARAM_INVALID, "Add model task def partition failed");
    return PARAM_INVALID;
  }
  // Save target/version to model_header
  ModelFileHeader &model_header = om_file_save_helper->GetModelFileHeader();
  model_header.platform_type = ge_model->GetPlatformType();
  model_header.om_ir_version = ge_model->GetVersion();
  std::string platform_version = ge_model->GetPlatformVersion();
  GELOGI("Platform version save: %s", platform_version.c_str());

  errno_t err;
  err = memcpy_s(model_header.platform_version, PLATFORM_VERSION_LEN, platform_version.c_str(),
                 platform_version.size() + 1);
  if (err != EOK) {
    GELOGE(MEMALLOC_FAILED, "ModelHelper SaveModel failed while allocating memory for platform_version.");
    return MEMALLOC_FAILED;
  }
  string version = reinterpret_cast<char *>(model_header.platform_version);
  GELOGI("Platform version save: %s", version.c_str());

  size_t name_size = ge_model->GetName().size();
  name_size = name_size > (MODEL_NAME_LENGTH - 1) ? (MODEL_NAME_LENGTH - 1) : name_size;
  err = memcpy_s(model_header.name, MODEL_NAME_LENGTH, ge_model->GetName().c_str(), name_size);
  if (err != EOK) {
    GELOGE(MEMALLOC_FAILED, "ModelHelper SaveModel failed while allocating memory for name");
    return MEMALLOC_FAILED;
  }
  string model_name = reinterpret_cast<char *>(model_header.name);
  GELOGI("Model name save:%s", model_name.c_str());

  Status ret = om_file_save_helper->SaveModel(save_param, output_file.c_str(), model, is_offline_);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "OmFileSaveHelper SaveModel return fail.");
    return FAILED;
  }
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status
ModelHelper::SaveOriginalGraphToOmModel(const ge::Graph &graph, const std::string &output_file) {
  if (output_file.empty()) {
    GELOGE(FAILED, "SaveModel received invalid file name prefix");
    return FAILED;
  }
  // Get computegraph from graph
  auto compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  if (compute_graph == nullptr) {
    GELOGE(FAILED, "SaveModel fail for compute_graph null");
    return FAILED;
  }
  GE_DUMP(compute_graph, "OriginalGraph");
  // Model
  ModelPtr model_ptr = ge::MakeShared<ge::Model>();
  GE_CHECK_NOTNULL_EXEC(model_ptr, return MEMALLOC_FAILED);
  std::string original_model_name = compute_graph->GetName() + "_original";
  model_ptr->SetName(original_model_name);
  model_ptr->SetGraph(graph);
  model_ptr->SetVersion(static_cast<uint32_t>(OM_PROTO_VERSION));
  string framework_version;
  Status frame_rt = PlatformVersionManager::GetPlatformVersion(framework_version);
  if (frame_rt == SUCCESS) {
    uint32_t counter = 0;
    string model_framework_version = framework_version + "." + std::to_string(counter);
    model_ptr->SetPlatformVersion(model_framework_version);
  }
  // Model def
  ge::Buffer model_buffer;
  ge::graphStatus status = model_ptr->Save(model_buffer);
  if (status != ge::GRAPH_SUCCESS) {
    GELOGE(FAILED, "SaveModel fail for save buffer fail");
    return FAILED;
  }
  std::shared_ptr<OmFileSaveHelper> om_file_save_helper = ge::MakeShared<OmFileSaveHelper>();
  GE_CHECK_NOTNULL_EXEC(om_file_save_helper, return MEMALLOC_FAILED);
  ModelPartition partition_model;
  partition_model.data = model_buffer.GetData();
  partition_model.size = static_cast<uint32_t>(model_buffer.GetSize());
  partition_model.type = ModelPartitionType::MODEL_DEF;
  GELOGI("Original Model type[%u],size[%u]", partition_model.type, partition_model.size);
  if (partition_model.data != nullptr && partition_model.size > 0) {
    (void)om_file_save_helper->AddPartition(partition_model);
    // Condition of AddPartition is established, no need to check value
  }
  // Save target/version to model_header
  ModelFileHeader &model_header = om_file_save_helper->GetModelFileHeader();
  model_header.om_ir_version = model_ptr->GetVersion();
  model_header.headsize = MODEL_FILE_HEAD_LEN;
  std::string platform_version = model_ptr->GetPlatformVersion();
  errno_t err = memcpy_s(model_header.platform_version, PLATFORM_VERSION_LEN, platform_version.c_str(),
                         platform_version.size() + 1);
  if (err != EOK) {
    GELOGE(FAILED, "ModelHelper SaveModel failed for platform_version");
    return FAILED;
  }
  size_t name_size = model_ptr->GetName().size();
  name_size = name_size > (MODEL_NAME_LENGTH - 1) ? (MODEL_NAME_LENGTH - 1) : name_size;
  err = memcpy_s(model_header.name, MODEL_NAME_LENGTH, model_ptr->GetName().c_str(), name_size);
  if (err != EOK) {
    GELOGE(FAILED, "ModelHelper SaveModel memory copy failed");
    return FAILED;
  }
  ModelBufferData model;
  Status ret = om_file_save_helper->SaveModelToFile(output_file.c_str(), model, is_offline_);
  return (ret == SUCCESS ? SUCCESS : FAILED);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status ModelHelper::LoadModel(const ge::ModelData &model_data) {
  if (model_data.model_data == nullptr || model_data.model_len == 0) {
    GELOGE(FAILED, "Model_data is nullptr, or model_data_size is 0");
    return FAILED;
  }

  if (is_assign_model_) {
    GELOGE(FAILED, "Model helper has already loaded!");
    return FAILED;
  }

  if (ReleaseLocalModelData() != SUCCESS) {
    GELOGE(FAILED, "ReleaseLocalModelData failed.");
    return FAILED;
  }

  if (ge::DavinciModelParser::ParseModelContent(model_data, model_addr_tmp_, model_len_tmp_) != SUCCESS) {
    GELOGE(FAILED, "Parse model content failed!");
    return FAILED;
  }

  file_header_ = reinterpret_cast<ModelFileHeader *>(model_data.model_data);

  OmFileLoadHelper om_load_helper;
  if (om_load_helper.Init(model_addr_tmp_, model_len_tmp_) != SUCCESS) {
    GELOGE(FAILED, "Om_load_helper init failed");
    model_addr_tmp_ = nullptr;
    return FAILED;
  }
  auto partition_table = reinterpret_cast<ModelPartitionTable *>(model_addr_tmp_);
  if (partition_table->num == kOriginalOmPartitionNum) {
    model_addr_tmp_ = nullptr;
    GELOGE(FAILED, "om model is error,please use executable om model");
    return FAILED;
  }
  // Encrypt model need to del temp model/no encrypt model don't need to del model
  model_addr_tmp_ = nullptr;

  if (GenerateGeModel(om_load_helper) != SUCCESS) {
    GELOGE(FAILED, "GenerateGeModel failed");
    return FAILED;
  }

  is_assign_model_ = true;
  return SUCCESS;
}

Status ModelHelper::GenerateGeModel(OmFileLoadHelper &om_load_helper) {
  model_ = ge::MakeShared<ge::GeModel>();
  GE_CHECK_NOTNULL(model_);
  Status ret = LoadModelData(om_load_helper);
  if (ret != SUCCESS) {
    return ret;
  }
  ret = LoadWeights(om_load_helper);
  if (ret != SUCCESS) {
    return ret;
  }
  ret = LoadTask(om_load_helper);
  if (ret != SUCCESS) {
    return ret;
  }
  ret = LoadTBEKernelStore(om_load_helper);
  if (ret != SUCCESS) {
    return ret;
  }
  return SUCCESS;
}

Status ModelHelper::LoadModelData(OmFileLoadHelper &om_load_helper) {
  ModelPartition partition_model_def;
  // no need to check value, DATA->NetOutput
  om_load_helper.GetModelPartition(ModelPartitionType::MODEL_DEF, partition_model_def);
  GELOGI("Model_def partition addr:%p,size:%u", partition_model_def.data, partition_model_def.size);

  ge::Model model;
  if (ge::Model::Load(partition_model_def.data, partition_model_def.size, model) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Load model failed.");
    return INTERNAL_ERROR;
  }

  SetModelToGeModel(model);

  return SUCCESS;
}

void ModelHelper::SetModelToGeModel(ge::Model &model) {
  model_->SetGraph(model.GetGraph());
  model_->SetName(model.GetName());
  model_->SetVersion(model.GetVersion());
  model_->SetPlatformVersion(model.GetPlatformVersion());
  model_->SetAttr(model.MutableAttrMap());
}

Status ModelHelper::LoadWeights(OmFileLoadHelper &om_load_helper) {
  ModelPartition partition;
  if (om_load_helper.GetModelPartition(ModelPartitionType::WEIGHTS_DATA, partition) != SUCCESS) {
    GELOGE(FAILED, "Get weight model partition failed.");
    return FAILED;
  }
  ge::Buffer weight = ge::Buffer::CopyFrom(partition.data, partition.size);
  model_->SetWeight(weight);

  GELOGI("GetWeight size:%u", partition.size);
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status ModelHelper::LoadTask(OmFileLoadHelper &om_load_helper) {
  ModelPartition task_partition;
  if (om_load_helper.GetModelPartition(ModelPartitionType::TASK_INFO, task_partition) != SUCCESS) {
    GELOGE(FAILED, "Get task model partition failed.");
    return FAILED;
  }
  std::shared_ptr<ModelTaskDef> task = ge::MakeShared<ModelTaskDef>();
  GE_CHECK_NOTNULL(task);
  if (task_partition.size != 0) {
    if (!ReadProtoFromArray(task_partition.data, task_partition.size, task.get())) {
      GELOGE(INTERNAL_ERROR, "ReadProtoFromArray failed.");
      return INTERNAL_ERROR;
    }
    GELOGI("TASK_INFO op_size:%zu, stream_num:%u", task->op().size(), task->stream_num());
  }
  model_->SetModelTaskDef(task);
  return SUCCESS;
}

Status ModelHelper::LoadTBEKernelStore(OmFileLoadHelper &om_load_helper) {
  // Load tbe kernels
  ModelPartition partition_kernel_def;
  TBEKernelStore kernel_store;
  if (om_load_helper.GetModelPartition(ModelPartitionType::TBE_KERNELS, partition_kernel_def) == SUCCESS) {
    GELOGI("Kernels partition size:%u", partition_kernel_def.size);
    if (kernel_store.Load(partition_kernel_def.data, partition_kernel_def.size)) {
      GELOGI("Load tbe kernels success");
    } else {
      GELOGW("Load tbe kernels failed");
    }
  }
  model_->SetTBEKernelStore(kernel_store);
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY GeModelPtr ModelHelper::GetGeModel() {
  if (model_ != nullptr) {
    return model_;
  }

  GELOGI("Model has not been loaded!");
  std::shared_ptr<ge::GeModel> out_model = ge::MakeShared<ge::GeModel>();
  if (out_model == nullptr) {
    return nullptr;
  }
  return out_model;
}

Status ModelHelper::ReleaseLocalModelData() noexcept {
  Status result = SUCCESS;
  if (model_addr_tmp_ != nullptr) {
    errno_t ret = memset_s(static_cast<void *>(model_addr_tmp_), model_len_tmp_, 0, model_len_tmp_);
    if (ret != EOK) {
      GELOGE(FAILED, "Failed to memset memory, error-code %d", ret);
      result = FAILED;
    }
    delete[] model_addr_tmp_;
    model_addr_tmp_ = nullptr;
    model_len_tmp_ = 0;
  }
  return result;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status ModelHelper::GetBaseNameFromFileName(const string &file_name,
                                                                                             string &base_name) {
  GELOGD("Get base_name from file, file_name:%s", file_name.c_str());
  GE_CHK_BOOL_EXEC_WARN(!file_name.empty(), return FAILED, "File path may not valid, check params --output");
  size_t start_position = 0;
  // using output as base_name (ignore ".om")
  size_t filename_suffixes = 3;
  if (file_name.find_last_of('/') != string::npos) {
    start_position = file_name.find_last_of('/') + 1;
  }
  size_t end_position = file_name.length() - filename_suffixes;
  base_name = file_name.substr(start_position, end_position - start_position);
  GE_CHK_BOOL_EXEC_WARN(!base_name.empty(), return FAILED, "Get base_name failed, check params --output");
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status
ModelHelper::GetModelNameFromMergedGraphName(const string &graph_name, string &model_name) {
  GELOGD("Get model_name from graph_name, graph_name:%s", graph_name.c_str());
  // this can only be used after merged graph(graph name will be append with "_x", x is index);
  GE_CHK_BOOL_EXEC_WARN(!graph_name.empty(), return FAILED, "File path may not valid, check params --output");
  size_t start_position = 0;
  size_t end_position = graph_name.length();
  // using graph as model_name (ignore "_x", x is the index of graph)
  if (graph_name.find_last_of('_') != string::npos) {
    end_position = graph_name.find_last_of('_');
  }
  model_name = graph_name.substr(start_position, end_position);
  GE_CHK_BOOL_EXEC_WARN(!model_name.empty(), return FAILED, "Get model_name failed, check params --output");
  return SUCCESS;
}
}  // namespace ge
