/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#include "common/checker.h"
#include "common/helper/model_parser_base.h"
#include "common/plugin/plugin_manager.h"
#include "framework/omg/version.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "framework/omg/omg_inner_types.h"
#include "mmpa/mmpa_api.h"
#include "external/graph/types.h"
#include "common/proto_util.h"
#include "graph/ge_local_context.h"
#include "graph/utils/file_utils.h"
#include "common/helper/file_saver.h"

namespace {
const uint32_t kOriginalOmPartitionNum = 1U;
const int32_t kSocVersionLen = 50;
const int32_t kModuleTypeAicore = 4;
const int32_t kModuleTypeVectorCore = 7;
const int32_t kInfoTypeCoreNum = 3;
const string kOpsProto = "libopsproto_rt2.0.so";
const string kOpMaster = "libopmaster_rt2.0.so";
const string kInner = "built-in";
const string kOpsProtoPath = "/op_proto/lib/";
const string kOpMasterPath = "/op_impl/ai_core/tbe/op_tiling/lib/";
}

namespace ge {
ModelHelper::~ModelHelper() {}

Status ModelHelper::SaveModelPartition(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                       const ModelPartitionType type, const uint8_t* const data,
                                       const size_t size, const size_t model_index) const {
  if ((size < 1U) || ((size > UINT32_MAX) && (type != WEIGHTS_DATA))) {
    GELOGE(PARAM_INVALID, "[Add][ModelPartition]Failed, partition size %zu invalid", size);
    if (size > UINT32_MAX) {
      std::string item = "item";
      if (type == MODEL_DEF) {
        item = "model info";
      } else if (type == TASK_INFO) {
        item = "task info";
      } else if (type == TBE_KERNELS) {
        item = "tbe kernels";
      } else if (type == CUST_AICPU_KERNELS) {
        item = "aicpu kernels";
      } else if (type == SO_BINS) {
        item = "so bins";
      } else {
        // do nothing.
      }
      ErrorManager::GetInstance().ATCReportErrMessage("E19023", {"size", "item", "maxsize"},
                                                      {std::to_string(size), item, std::to_string(UINT32_MAX)});
    }
    REPORT_INNER_ERROR("E19999", "Add model partition failed, partition size %zu "
                       "invalid", size);
    return PARAM_INVALID;
  }
  if (data == nullptr) {
    GELOGE(PARAM_INVALID, "[Add][ModelPartition]Failed, data is null");
    REPORT_INNER_ERROR("E19999", "Add model partition failed, data is null");
    return PARAM_INVALID;
  }
  ModelPartition partition_model;
  partition_model.data = data;
  partition_model.size = static_cast<uint64_t>(size);
  partition_model.type = type;
  if (om_file_save_helper->AddPartition(partition_model, model_index) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Add][ModelPartition]Failed, partition size %zu", size);
    REPORT_CALL_ERROR("E19999", "Add model partition failed, partition size %zu", size);
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status ModelHelper::SaveSizeToModelDef(const GeModelPtr &ge_model, const size_t model_index) const {
  std::vector<int64_t> om_info;
  GELOGD("SaveSizeToModelDef weight_data_size is %zu, ge_model_weight data is %p", ge_model->GetWeightSize(),
         ge_model->GetWeightData());
  om_info.push_back(static_cast<int64_t>(ge_model->GetWeightSize()));

  const auto &tbe_kernel_store = ge_model->GetTBEKernelStore();
  GELOGD("SaveSizeToModelDef tbe_kernels_size is %zu", tbe_kernel_store.DataSize());
  om_info.push_back(static_cast<int64_t>(tbe_kernel_store.DataSize()));

  const auto &cust_aicpu_kernel_store = ge_model->GetCustAICPUKernelStore();
  GELOGD("SaveSizeToModelDef cust aicpu kernels size is %zu", cust_aicpu_kernel_store.DataSize());
  om_info.push_back(static_cast<int64_t>(cust_aicpu_kernel_store.DataSize()));

  const std::shared_ptr<domi::ModelTaskDef> model_task_def = ge_model->GetModelTaskDefPtr();
  if (model_task_def == nullptr) {
    GELOGD("SaveSizeToModelDef task_info_size is 0.");
    om_info.push_back(0);
  } else {
    const size_t partition_task_size = model_task_def->ByteSizeLong();
    GELOGD("SaveSizeToModelDef task_info_size is %zu", partition_task_size);
    om_info.push_back(static_cast<int64_t>(partition_task_size));
  }

  if (model_index == 0U) {
    GELOGD("SaveSizeToModelDef so store size is %zu", GetOpStoreDataSize());
    om_info.push_back(static_cast<int64_t>(GetOpStoreDataSize()));
  } else {
    om_info.push_back(0U); // only first model save so.
  }

  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListInt(*(ge_model.get()), "om_info_list", om_info),
                   GELOGE(FAILED, "SetListInt of om_info_list failed.");
                   return FAILED);

  return SUCCESS;
}

Status ModelHelper::SaveModelDef(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                                 ge::Buffer &model_buffer, const size_t model_index) const {
  const ModelPtr model_tmp = ge::MakeShared<ge::Model>(ge_model->GetName(), ge_model->GetPlatformVersion());
  if (model_tmp == nullptr) {
    GELOGE(FAILED, "[Creat][Model]Failed, Model %s Ptr", ge_model->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "Create Model %s Ptr failed", ge_model->GetName().c_str());
    return FAILED;
  }
  model_tmp->SetGraph(ge_model->GetGraph());
  model_tmp->SetVersion(ge_model->GetVersion());
  const Status ret = SaveSizeToModelDef(ge_model, model_index);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Save][SizeToModelDef]Failed, model %s, error_code %u",
           ge_model->GetName().c_str(), ret);
    REPORT_CALL_ERROR("E19999", "Save SizeToModelDef failed, model %s, error_code %u",
                      ge_model->GetName().c_str(), ret);
    return ret;
  }

  model_tmp->SetAttr(ge_model->MutableAttrMap());
  (void)model_tmp->Save(model_buffer);
  GELOGD("MODEL_DEF size is %zu", model_buffer.GetSize());
  if (model_buffer.GetSize() > 0U) {
    if (SaveModelPartition(om_file_save_helper, ModelPartitionType::MODEL_DEF, model_buffer.GetData(),
                           model_buffer.GetSize(), model_index) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Add][ModelPartition]Failed, model %s, model_def size %zu, model_index %zu",
             ge_model->GetName().c_str(), model_buffer.GetSize(), model_index);
      REPORT_CALL_ERROR("E19999", "Add model graph partititon failed, model %s, model_def %zu, "
                        "model_index %zu", ge_model->GetName().c_str(), model_buffer.GetSize(), model_index);
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

Status ModelHelper::SaveModelWeights(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                                     const size_t model_index) const {
  GELOGD("WEIGHTS_DATA size is %zu, %p", ge_model->GetWeightSize(), ge_model->GetWeightData());
  // weight is not necessary
  if (ge_model->GetWeightSize() > 0U) {
    GE_CHK_STATUS_RET(SaveModelPartition(om_file_save_helper,
                                         ModelPartitionType::WEIGHTS_DATA,
                                         ge_model->GetWeightData(),
                                         ge_model->GetWeightSize(),
                                         model_index),
                      "Add weight partition failed");
  }
  return SUCCESS;
}

Status ModelHelper::SaveModelTbeKernel(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                       const GeModelPtr &ge_model, const size_t model_index) const {
  const auto &tbe_kernel_store = ge_model->GetTBEKernelStore();
  GELOGD("TBE_KERNELS size is %zu", tbe_kernel_store.DataSize());
  if (tbe_kernel_store.DataSize() > 0U) {
    GE_CHK_STATUS_RET(
        SaveModelPartition(om_file_save_helper, ModelPartitionType::TBE_KERNELS,
                           tbe_kernel_store.Data(), tbe_kernel_store.DataSize(),
                           model_index),
        "Add tbe kernel partition failed");
  }
  return SUCCESS;
}

Status ModelHelper::SaveModelCustAICPU(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                       const GeModelPtr &ge_model, const size_t model_index) const {
  const auto &cust_aicpu_kernel_store = ge_model->GetCustAICPUKernelStore();
  GELOGD("cust aicpu kernels size is %zu", cust_aicpu_kernel_store.DataSize());
  if (cust_aicpu_kernel_store.DataSize() > 0U) {
    GE_CHK_STATUS_RET(SaveModelPartition(om_file_save_helper,
                                         ModelPartitionType::CUST_AICPU_KERNELS,
                                         cust_aicpu_kernel_store.Data(),
                                         cust_aicpu_kernel_store.DataSize(), model_index),
                      "Add cust aicpu kernel partition failed");
  }
  return SUCCESS;
}

Status ModelHelper::SaveModelTaskDef(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                                     ge::Buffer &task_buffer, const size_t model_index) const {
  const std::shared_ptr<domi::ModelTaskDef> model_task_def = ge_model->GetModelTaskDefPtr();
  if (model_task_def == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Creat][ModelTaskDef]Failed, it is nullptr, "
           "model %s", ge_model->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "Creat model task def failed, it is nullptr, model %s",
                      ge_model->GetName().c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  const size_t partition_task_size = model_task_def->ByteSizeLong();
  GE_IF_BOOL_EXEC((partition_task_size == 0U) || (static_cast<int32_t>(partition_task_size) > INT_MAX),
                  GELOGE(FAILED, "[Check][ModelDefSize]Invalid, size %zu, model %s",
                         partition_task_size, ge_model->GetName().c_str());
                  REPORT_CALL_ERROR("E19999", "Model def size %zu check invalid, model %s",
                                    partition_task_size, ge_model->GetName().c_str());
                      return FAILED);

  task_buffer = ge::Buffer(partition_task_size);
  if (task_buffer.GetSize() == 0U) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Allocate][ModelTaskDefBuffer]Failed, "
           "model def size %zu, model %s", partition_task_size, ge_model->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "Allocate model task def buffer failed, model def size %zu "
                      "model %s", partition_task_size, ge_model->GetName().c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  (void)model_task_def->SerializePartialToArray(task_buffer.GetData(), static_cast<int32_t>(partition_task_size));

  GELOGD("TASK_INFO op_size:%d, stream_num:%u", model_task_def->op().size(), model_task_def->stream_num());
  GELOGD("TASK_INFO size is %zu", partition_task_size);

  if (SaveModelPartition(om_file_save_helper, ModelPartitionType::TASK_INFO, task_buffer.GetData(),
                         partition_task_size, model_index) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Add][ModelTaskDefPartition]Failed, model def size %zu, "
           "model_index %zu, model %s",
           partition_task_size, model_index, ge_model->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "Add model task def partition failed, model def size %zu "
                      "model_index %zu, model %s",
                      partition_task_size, model_index, ge_model->GetName().c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status ModelHelper::SaveModelHeader(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                                    const size_t model_num, bool need_check_os_cpu) const {
  // Save target/version to model_header
  ModelFileHeader &model_header = om_file_save_helper->GetModelFileHeader();
  model_header.platform_type = ge_model->GetPlatformType();
  model_header.om_ir_version = ge_model->GetVersion();
  model_header.model_num = static_cast<uint32_t>(model_num);
  const std::string platform_version = ge_model->GetPlatformVersion();

  errno_t err = memcpy_s(model_header.platform_version, static_cast<size_t>(PLATFORM_VERSION_LEN),
                         platform_version.c_str(), platform_version.size() + 1U);
  if (err != EOK) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION,
           "[Save][Model]Failed while allocating memory for platform_version %s, model %s, "
           "errno %d",
           platform_version.c_str(), ge_model->GetName().c_str(), err);
    REPORT_CALL_ERROR("E19999",
                      "ModelHelper save model %s failed while "
                      "allocating memory for platform_version %s, errno %d",
                      ge_model->GetName().c_str(), platform_version.c_str(), err);
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  const std::string version = reinterpret_cast<char_t *>(model_header.platform_version);
  GELOGD("Platform version save: %s", version.c_str());
  if (need_check_os_cpu) {
    model_header.need_check_os_cpu_info = static_cast<uint8_t>(OsCpuInfoCheckTyep::NEED_CHECK);
    GELOGD("need_check_os_cpu_info save:%u", model_header.need_check_os_cpu_info);
  }

  size_t name_size = ge_model->GetName().size();
  name_size = (name_size > (MODEL_NAME_LENGTH - 1U)) ? (MODEL_NAME_LENGTH - 1U) : name_size;
  err = memcpy_s(model_header.name, MODEL_NAME_LENGTH, ge_model->GetName().c_str(), name_size);
  if (err != EOK) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION,
           "[Save][Model]Failed while allocating memory for model %s, errno %d",
           ge_model->GetName().c_str(), err);
    REPORT_CALL_ERROR("E19999", "ModelHelper save model failed while allocating memory "
                      "for model %s,errno %d", ge_model->GetName().c_str(), err);
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  const std::string model_name = reinterpret_cast<char_t *>(model_header.name);
  GELOGD("Model name save:%s", model_name.c_str());
  return SUCCESS;
}

Status ModelHelper::GetSoBinData(string &cpu_info, string &os_info) {
  std::string opp_path;
  GE_ASSERT_SUCCESS(PluginManager::GetOppPath(opp_path), "[GetOppPath] failed, os:%s,cpu:%s",
      cpu_info.c_str(), os_info.c_str());

  std::vector<std::string> vendors; // to do: add customize defined opp path
  // insert inner so path
  vendors.push_back(kInner);
  for (size_t i = 0U; i < vendors.size(); i++) {
    std::string op_proto_path = opp_path;
    std::string op_master_path = opp_path;
    op_proto_path += vendors[i] + kOpsProtoPath + os_info + "/" + cpu_info + "/" + kOpsProto;
    uint32_t bin_len = 0U;
    auto op_proto_bin = GetBinFromFile(op_proto_path, bin_len);
#ifdef ONLY_COMPILE_OPEN_SRC
    if (op_proto_bin != nullptr) {
      const OpSoBinPtr proto_bin_ptr = ge::MakeShared<OpSoBin>(kOpsProto, vendors[i], std::move(op_proto_bin), bin_len);
      GE_ASSERT_NOTNULL(proto_bin_ptr);
      op_so_store_.AddKernel(proto_bin_ptr);
    }
#else
    GE_ASSERT_NOTNULL(op_proto_bin, " open proto so fail, path=%s", op_proto_path.c_str());
    const OpSoBinPtr proto_bin_ptr = ge::MakeShared<OpSoBin>(kOpsProto, vendors[i], std::move(op_proto_bin), bin_len);
    GE_ASSERT_NOTNULL(proto_bin_ptr);
    op_so_store_.AddKernel(proto_bin_ptr);
#endif

    op_master_path += vendors[i] + kOpMasterPath + os_info + "/" + cpu_info + "/" + kOpMaster;
    auto op_master_bin = GetBinFromFile(op_master_path, bin_len);
#ifdef ONLY_COMPILE_OPEN_SRC
    if (op_master_bin != nullptr) {
      const OpSoBinPtr master_bin_ptr
          = ge::MakeShared<OpSoBin>(kOpMaster, vendors[i], std::move(op_master_bin), bin_len);
      GE_ASSERT_NOTNULL(master_bin_ptr);
      op_so_store_.AddKernel(master_bin_ptr);
    }
#else
    GE_ASSERT_NOTNULL(op_master_bin, " open master so fail, path=%s", op_master_path.c_str());
    const OpSoBinPtr master_bin_ptr = ge::MakeShared<OpSoBin>(kOpMaster, vendors[i], std::move(op_master_bin), bin_len);
    GE_ASSERT_NOTNULL(master_bin_ptr);
    op_so_store_.AddKernel(master_bin_ptr);
#endif
  }

  GE_ASSERT_TRUE(op_so_store_.Build());
  return SUCCESS;
}

const uint8_t *ModelHelper::GetOpSoStoreData() const { return op_so_store_.Data(); }

size_t ModelHelper::GetOpStoreDataSize() const { return op_so_store_.DataSize(); }

Status ModelHelper::SaveSoStoreModelPartitionInfo(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                                  const GeRootModelPtr &ge_root_model,
                                                  string &output_file_name) {
  GELOGD(" so in om flag:%d", ge_root_model->GetSoInOmFlag());
  if (ge_root_model->GetSoInOmFlag()) {
    std::string host_env_os;
    std::string host_env_cpu;
    (void)GetThreadLocalContext().GetOption("ge.host_env_os", host_env_os);
    (void)GetThreadLocalContext().GetOption("ge.host_env_cpu", host_env_cpu);

    GEEVENT("get so data.cpu:%s,os:%s", host_env_cpu.c_str(), host_env_os.c_str());
    if (host_env_os.empty() || host_env_cpu.empty()) {
      GELOGW("[SaveSoStoreModelPartitionInfo] get cpu or os info empty!!");
      return SUCCESS;
    }

    GE_ASSERT_SUCCESS(GetSoBinData(host_env_cpu, host_env_os));
#ifdef ONLY_COMPILE_OPEN_SRC
    if (op_so_store_.GetKernelNum() == 0U) {
      GELOGW("[SaveSoStoreModelPartitionInfo] get cpu and os info empty!!");
      return SUCCESS;
    }
#endif

    const uint8_t * const op_so_data = GetOpSoStoreData();
    const auto data_size = GetOpStoreDataSize();
    GE_ASSERT_SUCCESS(SaveModelPartition(om_file_save_helper, ModelPartitionType::SO_BINS,
        op_so_data, data_size, 0), "Add so store failed");
    const auto position = output_file_name.find(".om");
    const string cpu_os_str = "_" + host_env_os + "_" + host_env_cpu;
    if (position < output_file_name.length()) {
      (void)output_file_name.insert(position, cpu_os_str);
    } else {
      (void)output_file_name.append(cpu_os_str);
    }
    is_so_store_ = true;
  }
  return SUCCESS;
}

Status ModelHelper::SaveAllModelPartiton(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                         const GeModelPtr &ge_model, ge::Buffer &model_buffer, ge::Buffer &task_buffer,
                                         const size_t model_index) const {
  GE_CHK_STATUS_RET(EnsureKernelBuilt(ge_model), "ensure kernel built failed, model=%s.", ge_model->GetName().c_str());
  if (SaveModelDef(om_file_save_helper, ge_model, model_buffer, model_index) != SUCCESS) {
    GELOGE(FAILED, "[Save][ModelDef]Failed, model %s, model index %zu",
           ge_model->GetName().c_str(), model_index);
    REPORT_CALL_ERROR("E19999", "ModelHelper save model def failed, model %s, model index %zu",
                      ge_model->GetName().c_str(), model_index);
    return FAILED;
  }

  if (SaveModelWeights(om_file_save_helper, ge_model, model_index) != SUCCESS) {
    GELOGE(FAILED, "[Save][ModelWeights]Failed, model %s, model index %zu",
           ge_model->GetName().c_str(), model_index);
    REPORT_CALL_ERROR("E19999", "ModelHelper save mode weights failed, model %s, model index %zu",
                      ge_model->GetName().c_str(), model_index);
    return FAILED;
  }

  if (SaveModelTbeKernel(om_file_save_helper, ge_model, model_index) != SUCCESS) {
     GELOGE(FAILED, "[Save][ModelTbeKernel]Failed, model %s, model index %zu",
            ge_model->GetName().c_str(), model_index);
     REPORT_CALL_ERROR("E19999", "ModelHelper save model tbe kernel failed, model %s, "
                       "model index %zu", ge_model->GetName().c_str(), model_index);
    return FAILED;
  }

  if (SaveModelCustAICPU(om_file_save_helper, ge_model, model_index) != SUCCESS) {
    GELOGE(FAILED, "[Save][ModelCustAICPU]Failed, model %s, model index %zu",
           ge_model->GetName().c_str(), model_index);
    REPORT_CALL_ERROR("E19999", "ModelHelper save model cust aicpu failed, model %s "
                      "model index %zu", ge_model->GetName().c_str(), model_index);
    return FAILED;
  }

  if (SaveModelTaskDef(om_file_save_helper, ge_model, task_buffer, model_index) != SUCCESS) {
    GELOGE(FAILED, "[Save][TaskDef]Failed, model %s, model index %zu",
           ge_model->GetName().c_str(), model_index);
    REPORT_CALL_ERROR("E19999", "ModelHelper save task def failed, model %s, model index %zu",
                      ge_model->GetName().c_str(), model_index);
    return FAILED;
  }
  return SUCCESS;
}

Status ModelHelper::SaveToOmModel(const GeModelPtr &ge_model, const SaveParam &save_param,
                                  const std::string &output_file, ModelBufferData &model,
                                  const GeRootModelPtr &ge_root_model) {
  if (output_file.empty()) {
    GELOGE(FAILED, "[Save][Model]GraphBuilder SaveModel received invalid file name prefix, "
           "model %s", ge_model->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "GraphBuilder SaveModel received invalid file name prefix, "
                      "model %s", ge_model->GetName().c_str());
    return FAILED;
  }

  GE_IF_BOOL_EXEC(ge_model == nullptr, GELOGE(FAILED, "Ge_model is nullptr"); return FAILED);
  std::shared_ptr<OmFileSaveHelper> om_file_save_helper = ge::MakeShared<OmFileSaveHelper>();
  GE_CHECK_NOTNULL(om_file_save_helper);
  ge::Buffer model_buffer;
  ge::Buffer task_buffer;
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetStr(*(ge_model.get()), ATTR_MODEL_ATC_CMDLINE,
                   domi::GetContext().atc_cmdline),
                   GELOGE(FAILED, "SetStr for atc_cmdline failed.");
                   return FAILED);
  std::string cur_version;
  auto ret = GetOppVersion(cur_version);
  if ((ret != SUCCESS) || (!ge::AttrUtils::SetStr(*(ge_model.get()), ATTR_MODEL_OPP_VERSION, cur_version))) {
    GELOGW("Ge model set opp version failed!");
  }

  string output_file_name = output_file;
  if ((ge_root_model != nullptr) && (is_offline_)) {
    GE_ASSERT_SUCCESS(SaveSoStoreModelPartitionInfo(om_file_save_helper, ge_root_model,
        output_file_name), "[SaveSoStoreModelPartition]Failed");
  }

  ret = SaveAllModelPartiton(om_file_save_helper, ge_model, model_buffer, task_buffer);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Save][AllModelPartition]Failed, model %s, error_code %u",
           ge_model->GetName().c_str(), ret);
    REPORT_CALL_ERROR("E19999", "OmFileSaveHelper save all model partition failed, model %s "
                       "error_code %u", ge_model->GetName().c_str(), ret);
    return ret;
  }

  // static case 3th param is model_num :1U
  ret = SaveModelHeader(om_file_save_helper, ge_model, 1U, is_so_store_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Save][ModelHeader]Failed, model %s, error_code %u",
           ge_model->GetName().c_str(), ret);
    REPORT_CALL_ERROR("E19999", "OmFileSaveHelper save model header failed, model %s "
                       "error_code %u", ge_model->GetName().c_str(), ret);
    return ret;
  }

  ret = om_file_save_helper->SaveModel(save_param, output_file_name.c_str(), model, is_offline_);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Save][Model]Failed, model %s, output file %s",
           ge_model->GetName().c_str(), output_file.c_str());
    REPORT_CALL_ERROR("E19999", "OmFileSaveHelper save model failed, model %s, "
                       "output file %s", ge_model->GetName().c_str(), output_file.c_str());
    return ret;
  }
  return SUCCESS;
}

Status ModelHelper::SaveToOmRootModel(const GeRootModelPtr &ge_root_model, const SaveParam &save_param,
                                      const std::string &output_file, ModelBufferData &model,
                                      const bool is_unknown_shape) {
  GE_CHECK_NOTNULL(ge_root_model);
  GE_IF_BOOL_EXEC(ge_root_model == nullptr,
                  GELOGE(FAILED, "[Check][GERootModel]Ge_root_model is nullptr");
                  REPORT_INNER_ERROR("E19999", "Ge_root_model check failed, it is nullptr");
                  return FAILED);

  const auto &name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
  GE_IF_BOOL_EXEC(name_to_ge_model.empty(),
                  GELOGE(FAILED, "[Get][SubModel]Ge_root_model has no sub model");
                  REPORT_INNER_ERROR("E19999", "Ge_root_model has no sub model");
                  return FAILED);
  GE_IF_BOOL_EXEC(output_file.empty(),
                  GELOGE(FAILED, "[Save][Model]GraphBuilder SaveModel received invalid "
                         "file name prefix");
                  REPORT_INNER_ERROR("E19999", "GraphBuilder SaveModel received invalid "
                                     "file name prefix");
                  return FAILED);
  if (!is_unknown_shape) {
    auto &model_root = name_to_ge_model.begin()->second;
    return SaveToOmModel(model_root, save_param, output_file, model, ge_root_model);
  }

  std::shared_ptr<OmFileSaveHelper> om_file_save_helper = ge::MakeShared<OmFileSaveHelper>();
  GE_CHECK_NOTNULL(om_file_save_helper);

  GeModelPtr first_ge_model;
  const auto &first_model_it = name_to_ge_model.find(ge_root_model->GetRootGraph()->GetName());
  if (first_model_it == name_to_ge_model.end()) {
    first_ge_model = MakeShared<GeModel>();
    GE_CHECK_NOTNULL(first_ge_model);
    first_ge_model->SetGraph(ge_root_model->GetRootGraph());
  } else {
    first_ge_model = first_model_it->second;
  }
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetStr(*(first_ge_model.get()), ATTR_MODEL_ATC_CMDLINE,
                   domi::GetContext().atc_cmdline),
                   GELOGE(FAILED, "SetStr for atc_cmdline failed.");
                   return FAILED);
  std::string cur_version;
  const auto get_ret = GetOppVersion(cur_version);
  if ((get_ret != SUCCESS) || (!ge::AttrUtils::SetStr(*(first_ge_model.get()), ATTR_MODEL_OPP_VERSION, cur_version))) {
    GELOGW("Ge model set opp version failed!");
  }
  // ge root model must be the first to be loaded
  std::vector<std::string> model_names{ge_root_model->GetRootGraph()->GetName()};
  for (auto &item : name_to_ge_model) {
    if (item.first != model_names.front()) {
      model_names.emplace_back(item.first);
    }
  }

  std::vector<ge::Buffer> model_buffers(model_names.size());
  std::vector<ge::Buffer> task_buffers(model_names.size());

  string output_file_name = output_file;
  // only save in model index 0
  if (is_offline_) {
    GE_ASSERT_SUCCESS(SaveSoStoreModelPartitionInfo(om_file_save_helper, ge_root_model,
        output_file_name), "[SaveSoStoreModelPartitionInfo]Failed");
  }

  size_t cur_index = 0U;
  bool is_partitioned = false;
  const auto &root_graph = ge_root_model->GetRootGraph();
  // shape partitioned scene no need to load root_graph ge_model.
  (void) AttrUtils::GetBool(root_graph,  ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_partitioned);
  if ((model_names.size() > 1U) && is_partitioned) {
    GELOGD("only save first model MODEL_DEF");
    if (SaveModelDef(om_file_save_helper, first_ge_model, model_buffers[cur_index], cur_index) != SUCCESS) {
      GELOGE(FAILED, "[Save][ModelDef]Failed, cur_index %zu", cur_index);
      REPORT_INNER_ERROR("E19999", "Save model def failed, cur_index %zu", cur_index);
      return FAILED;
    }
    ++cur_index;
  }

  for (; cur_index < model_names.size(); ++cur_index) {
    const auto model_name = model_names[cur_index];
    GELOGD("cur model %s index is %zu", model_name.c_str(), cur_index);
    const GeModelPtr &ge_model = name_to_ge_model.at(model_name);
    const auto ret = SaveAllModelPartiton(om_file_save_helper, ge_model, model_buffers[cur_index],
                                          task_buffers[cur_index], cur_index);
    if (ret != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "[Save][AllModelPartition]Failed, model name %s, cur_index %zu",
             model_name.c_str(), cur_index);
      REPORT_CALL_ERROR("E19999", "Save all model %s partition failed, cur_index %zu",
                        model_name.c_str(), cur_index);
      return INTERNAL_ERROR;
    }
  }

  auto ret =
    SaveModelHeader(om_file_save_helper, first_ge_model, model_names.size(), is_so_store_);
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Save][ModelHeader]Failed, model name %s",
           first_ge_model->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "Save model %s header failed", first_ge_model->GetName().c_str());
    return INTERNAL_ERROR;
  }

  ret = om_file_save_helper->SaveRootModel(save_param, output_file_name.c_str(), model, is_offline_);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Save][Model]OmFileSaveHelper save model eturn fail, output_file %s",
           output_file_name.c_str());
    REPORT_CALL_ERROR("E19999", "OmFileSaveHelper save model return fail, output_file %s",
                      output_file_name.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status ModelHelper::SaveOriginalGraphToOmModel(const ge::Graph &graph, const std::string &output_file) const {
  if (output_file.empty()) {
    GELOGE(FAILED, "[Save][Model]Received invalid file name prefix, output_file %s", output_file.c_str());
    REPORT_INNER_ERROR("E19999", "Save model received invalid file name prefix, output_file %s", output_file.c_str());
    return FAILED;
  }
  // Get computegraph from graph
  const auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  if (compute_graph == nullptr) {
    GELOGE(FAILED, "[Save][Model]Failed for compute_graph null");
    REPORT_INNER_ERROR("E19999", "Save model failed for compute_graph null");
    return FAILED;
  }
  GE_DUMP(compute_graph, "OriginalGraph");
  // Model
  const ModelPtr model_ptr = ge::MakeShared<ge::Model>();
  GE_CHECK_NOTNULL_EXEC(model_ptr, return MEMALLOC_FAILED);
  const std::string original_model_name = compute_graph->GetName() + "_original";
  model_ptr->SetName(original_model_name);
  model_ptr->SetGraph(compute_graph);
  model_ptr->SetVersion(static_cast<uint32_t>(OM_PROTO_VERSION));
  std::string framework_version;
  const Status frame_rt = PlatformVersionManager::GetPlatformVersion(framework_version);
  if (frame_rt == SUCCESS) {
    const std::string model_framework_version = framework_version + "." + std::to_string(0);
    model_ptr->SetPlatformVersion(model_framework_version);
  }
  // Model def
  ge::Buffer model_buffer;
  const ge::graphStatus status = model_ptr->Save(model_buffer);
  if (status != ge::GRAPH_SUCCESS) {
    GELOGE(FAILED, "[Save][Model]Failed for save buffer fail, model %s",
           model_ptr->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "Save model %s failed for save buffer fail",
                      model_ptr->GetName().c_str());
    return FAILED;
  }
  const std::shared_ptr<OmFileSaveHelper> om_file_save_helper = ge::MakeShared<OmFileSaveHelper>();
  GE_CHECK_NOTNULL_EXEC(om_file_save_helper, return MEMALLOC_FAILED);
  ModelPartition partition_model;
  partition_model.data = model_buffer.GetData();
  partition_model.size = static_cast<uint32_t>(model_buffer.GetSize());
  partition_model.type = ModelPartitionType::MODEL_DEF;
  GELOGI("Original Model type[%u],size[%" PRIu64 "]", partition_model.type, partition_model.size);
  if ((partition_model.data != nullptr) && (partition_model.size > 0U)) {
    (void)om_file_save_helper->AddPartition(partition_model);
    // Condition of AddPartition is established, no need to check value
  }
  // Save target/version to model_header
  ModelFileHeader &model_header = om_file_save_helper->GetModelFileHeader();
  model_header.om_ir_version = model_ptr->GetVersion();
  model_header.headsize = MODEL_FILE_HEAD_LEN;
  const std::string platform_version = model_ptr->GetPlatformVersion();
  errno_t
      err = memcpy_s(model_header.platform_version, static_cast<size_t>(PLATFORM_VERSION_LEN), platform_version.c_str(),
                     platform_version.size() + 1U);
  if (err != EOK) {
    GELOGE(FAILED, "[Save][Model]Failed for platform_version %s, model %s, errno %d",
           platform_version.c_str(), model_ptr->GetName().c_str(), err);
    REPORT_CALL_ERROR("E19999", "Save model %s failed for platform_version %s, errno %d",
                      model_ptr->GetName().c_str(), platform_version.c_str(), err);
    return FAILED;
  }
  size_t name_size = model_ptr->GetName().size();
  name_size = (name_size > (MODEL_NAME_LENGTH - 1U)) ? (MODEL_NAME_LENGTH - 1U) : name_size;
  err = memcpy_s(model_header.name, MODEL_NAME_LENGTH, model_ptr->GetName().c_str(), name_size);
  if (err != EOK) {
    GELOGE(FAILED, "[Save][Model]Failed for memory copy %s failed, errno %d",
           model_ptr->GetName().c_str(), err);
    REPORT_CALL_ERROR("E19999", "Save model failed for memory copy %s failed, errno %d",
                      model_ptr->GetName().c_str(), err);
    return FAILED;
  }
  ModelBufferData model;
  const Status ret = om_file_save_helper->SaveModelToFile(output_file.c_str(), model, is_offline_);
  return ((ret == SUCCESS) ? SUCCESS : FAILED);
}

Status ModelHelper::LoadModel(const ge::ModelData &model_data) {
  if ((model_data.model_data == nullptr) || (model_data.model_len == 0U)) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID,
           "[Load][Model]Model_data is nullptr or model_data_size is 0");
    REPORT_INNER_ERROR("E19999", "Load model failed, "
                       "Model_data is nullptr or model_data_size is 0");
    return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
  }

  if (is_assign_model_) {
    GELOGE(ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED, "[Load][Model]Model helper has already loaded!");
    return ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED;
  }

  uint8_t *model_data_addr = nullptr;
  uint64_t model_data_size = 0UL;
  Status status = ModelParserBase::ParseModelContent(model_data, model_data_addr, model_data_size);
  if (status != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Parse][ModelContent]Failed!");
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  file_header_ = reinterpret_cast<ModelFileHeader *>(model_data.model_data);
  OmFileLoadHelper om_load_helper;
  status = om_load_helper.Init(model_data_addr, model_data_size, file_header_);
  if (status != SUCCESS) {
    GELOGE(status, "[Init][OmLoadHelper]Failed");
    model_data_addr = nullptr;
    return status;
  }
  const auto partition_table = reinterpret_cast<ModelPartitionTable *>(model_data_addr);
  if (partition_table->num == kOriginalOmPartitionNum) {
    model_data_addr = nullptr;
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][OmModel]Error, please use executable om model");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  // Encrypt model need to del temp model/no encrypt model don't need to del model
  model_data_addr = nullptr;

  status = GenerateGeModel(om_load_helper, model_, 0U, false);
  if (status != SUCCESS) {
    GELOGE(status, "[Generate][GEModel]Failed");
    return status;
  }
  GELOGD("in ModelHelper::LoadModel, is_assign_model_ is setted to true!");
  is_assign_model_ = true;
  return SUCCESS;
}

Status ModelHelper::GetModelFileHead(const ge::ModelData &model_data, const ModelFileHeader *&file_header) {
  if (model_data.model_data == nullptr) {
    REPORT_INPUT_ERROR("E10003", std::vector<std::string>({"parameter", "value", "reason"}),
                       std::vector<std::string>({"om", model_data.om_name, "model data can not be nullptr"}));
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Param] Invalid model. Model data can not be nullptr.");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  if (model_data.model_len < sizeof(ModelFileHeader)) {
    REPORT_INPUT_ERROR("E10003", std::vector<std::string>({"parameter", "value", "reason"}),
                       std::vector<std::string>({"om", model_data.om_name, "invalid om file"}));
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID,
           "[Check][Param] Invalid model. Model data size %" PRIu64 " must be greater than or equal to %zu.",
           model_data.model_len, sizeof(ModelFileHeader));
    return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
  }
  file_header = reinterpret_cast<const ModelFileHeader *>(model_data.model_data);
  return SUCCESS;
}

Status ModelHelper::GetOppVersion(std::string &version) {
  std::string opp_path;
  (void)PluginManager::GetOppPath(opp_path);
  const std::string version_path = opp_path + "/version.info";
  if (!PluginManager::GetVersionFromPath(version_path, version)) {
    GELOGW("Get opp version information from %s failed!", version_path.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status ModelHelper::CheckOsCpuInfoAndOppVersion() {
  if (file_header_->need_check_os_cpu_info == static_cast<uint8_t>(OsCpuInfoCheckTyep::NEED_CHECK)) {
    std::string host_env_os;
    std::string host_env_cpu;
    (void) ge::AttrUtils::GetStr(*(model_.get()), ATTR_MODEL_HOST_ENV_OS, host_env_os);
    (void) ge::AttrUtils::GetStr(*(model_.get()), ATTR_MODEL_HOST_ENV_CPU, host_env_cpu);
    std::string cur_host_env_os;
    std::string cur_host_env_cpu;
    ge::PluginManager::GetCurEnvPackageOsAndCpuType(cur_host_env_os, cur_host_env_cpu);
    if ((host_env_os.compare(cur_host_env_os) != 0) || (host_env_cpu.compare(cur_host_env_cpu) != 0)) {
      REPORT_INNER_ERROR("E19999", "The os/cpu type of the model does not match the current system,"
                         "Model is[%s][%s], system is[%s][%s], please use the matching platform",
                         host_env_os.c_str(), host_env_cpu.c_str(),
                         cur_host_env_os.c_str(), cur_host_env_cpu.c_str());
      GELOGE(FAILED, "The os/cpu type of the model does not match the current system,"
             "Model is[%s][%s], system is[%s][%s]",
             host_env_os.c_str(), host_env_cpu.c_str(),
             cur_host_env_os.c_str(), cur_host_env_cpu.c_str());
      return FAILED;
    }
    GELOGD("Check os[%s], cpu[%s] success.", host_env_os.c_str(), host_env_cpu.c_str());
  }

  if ((file_header_->need_check_os_cpu_info == static_cast<uint8_t>(OsCpuInfoCheckTyep::NO_CHECK))
      && is_unknown_shape_model_) {
    std::string version;
    (void) ge::AttrUtils::GetStr(*(model_.get()), ATTR_MODEL_OPP_VERSION, version);
    std::string cur_version;
    GE_CHK_STATUS_RET(GetOppVersion(cur_version), "[Get][opp version] failed.");
    if (version.compare(cur_version) != 0) {
      REPORT_INNER_ERROR("E19999", "The opp version of the model does not match the current opp run package,"
                         "Model is[%s], opp run package is[%s], try to convert the om again!",
                         version.c_str(), cur_version.c_str());
      GELOGE(FAILED, "The opp version of the model does not match the current opp run package,"
             "Model is[%s], opp run package is[%s]", version.c_str(), cur_version.c_str());
      return FAILED;
    }
    GELOGD("Check opp version[%s] success.", version.c_str());
  }
  return SUCCESS;
}

Status ModelHelper::LoadRootModel(const ge::ModelData &model_data) {
  if (is_assign_model_) {
    GELOGE(ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED, "[Load][RootModel]Model helper has already loaded!");
    return ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED;
  }
  GE_CHK_STATUS_RET(GetModelFileHead(model_data, file_header_), "[Get][ModelFileHead] failed.");
  // model verison 1.0 file header does not have model_num member
  is_unknown_shape_model_ = ModelParserBase::IsDynamicModel(*file_header_);
  GELOGD("Cur om model is ge root model or no %d, model version %u",
         static_cast<int32_t>(is_unknown_shape_model_), file_header_->version);
  uint8_t *model_data_addr = nullptr;
  uint64_t model_data_size = 0UL;
  Status status = ModelParserBase::ParseModelContent(model_data, model_data_addr, model_data_size);
  if (status != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Parse][RootModelContent]Failed!");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  OmFileLoadHelper om_load_helper;
  if (is_unknown_shape_model_) {
    status = om_load_helper.Init(model_data_addr, model_data_size, file_header_->model_num, file_header_);
  } else {
    status = om_load_helper.Init(model_data_addr, model_data_size, file_header_);
  }
  if (status != SUCCESS) {
    GELOGE(status, "[Init][OmLoadHelper]Failed");
    model_data_addr = nullptr;
    return status;
  }
  // Encrypt model need to del temp model/no encrypt model don't need to del model
  model_data_addr = nullptr;

  status = GenerateGeRootModel(om_load_helper);
  if (status != SUCCESS) {
    GELOGE(status, "[Generate][GERootModel]Failed");
    return status;
  }
  GELOGD("In ModelHelper::LoadRootModel, is_assign_model_ is setted to true!");
  is_assign_model_ = true;

  if (is_repack_so_) {
    GELOGD("ModelHelper::repack so!");
    return SUCCESS;
  }

  if (CheckOsCpuInfoAndOppVersion() != SUCCESS) {
    GELOGE(FAILED, "[Check][OsCpuOppVersion]Failed");
    return FAILED;
  }
  return SUCCESS;
}

Status ModelHelper::GenerateGeModel(const OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model,
                                    const size_t mode_index, const bool is_dyn_root) const {
  cur_model = MakeShared<GeModel>();
  GE_CHECK_NOTNULL(cur_model);
  Status ret = LoadModelData(om_load_helper, cur_model, mode_index);
  if (ret != SUCCESS) {
    return ACL_ERROR_GE_EXEC_LOAD_MODEL_PARTITION_FAILED;
  }

  // shape partitioned scene no need to load root_graph ge_model.
  if (is_dyn_root && IsPartitionedGraph(cur_model)) {
    return SUCCESS;
  }

  ret = LoadWeights(om_load_helper, cur_model, mode_index);
  if (ret != SUCCESS) {
    return ACL_ERROR_GE_EXEC_LOAD_WEIGHT_PARTITION_FAILED;
  }
  ret = LoadTask(om_load_helper, cur_model, mode_index);
  if (ret != SUCCESS) {
    return ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED;
  }
  ret = LoadTBEKernelStore(om_load_helper, cur_model, mode_index);
  if (ret != SUCCESS) {
    return ACL_ERROR_GE_EXEC_LOAD_KERNEL_PARTITION_FAILED;
  }
  ret = LoadCustAICPUKernelStore(om_load_helper, cur_model, mode_index);
  if (ret != SUCCESS) {
    return ACL_ERROR_GE_EXEC_LOAD_KERNEL_PARTITION_FAILED;
  }
  return SUCCESS;
}

Status ModelHelper::GenerateGeRootModel(const OmFileLoadHelper &om_load_helper) {
  GELOGD("Begin to generate ge root model");
  root_model_ = MakeShared<GeRootModel>();
  GE_CHECK_NOTNULL(root_model_);
  if (!is_unknown_shape_model_) {
    GE_CHK_STATUS_RET(GenerateGeModel(om_load_helper, model_, 0U, false), "[Generate][GERootModel]Failed");
    GE_CHECK_NOTNULL(model_);
    root_model_->SetRootGraph(model_->GetGraph());
    root_model_->SetModelName(model_->GetName());
    GE_CHK_STATUS_RET(LoadOpSoBin(om_load_helper, root_model_), "[Generate][LoadOpSoBin]Failed");
    root_model_->SetSubgraphInstanceNameToModel(model_->GetName(), model_);
    return SUCCESS;
  }

  for (size_t mode_index = 0U;  mode_index < file_header_->model_num; ++mode_index) {
    GeModelPtr cur_model;
    GE_CHK_STATUS_RET_NOLOG(GenerateGeModel(om_load_helper, cur_model, mode_index, mode_index == 0U));
    if (mode_index == 0U) {
      root_model_->SetRootGraph(cur_model->GetGraph());
      root_model_->SetModelId(cur_model->GetModelId());
      root_model_->SetModelName(cur_model->GetName());
      GE_CHK_STATUS_RET(LoadOpSoBin(om_load_helper, root_model_), "[Generate][LoadOpSoBin]Failed");
      model_ = cur_model;

      if (IsPartitionedGraph(cur_model)) {
        continue;
      }
    }

    root_model_->SetSubgraphInstanceNameToModel(cur_model->GetName(), cur_model);
  }

  return SUCCESS;
}

bool ModelHelper::IsPartitionedGraph(const GeModelPtr &cur_model) const {
  // shape partitioned scene no need to load root_graph ge_model.
  bool is_partitioned = false;
  const auto &root_graph = cur_model->GetGraph();
  (void)AttrUtils::GetBool(root_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_partitioned);
  return is_partitioned;
}

void ModelHelper::SetModelToGeModel(const GeModelPtr &ge_model, Model &model) {
  ge_model->SetGraph(model.GetGraph());
  ge_model->SetName(model.GetName());
  ge_model->SetVersion(model.GetVersion());
  ge_model->SetPlatformVersion(model.GetPlatformVersion());
  ge_model->SetAttr(model.MutableAttrMap());
}

Status ModelHelper::LoadModelData(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                                  const size_t mode_index) const {
  ModelPartition partition_model_def;
  // no need to check value, DATA->NetOutput
  (void) om_load_helper.GetModelPartition(ModelPartitionType::MODEL_DEF, partition_model_def, mode_index);
  GELOGD("Model_def partition addr:%p,size:%" PRIu64 "", partition_model_def.data, partition_model_def.size);

  ge::Model model;
  if (ge::Model::Load(partition_model_def.data, static_cast<size_t>(partition_model_def.size), model) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Load][Model]Failed, model_def partition addr:%p, size:%" PRIu64 "",
           partition_model_def.data, partition_model_def.size);
    REPORT_CALL_ERROR("E19999", "Load model failed, model_def partition addr:%p, size:%" PRIu64 "",
                      partition_model_def.data, partition_model_def.size);
    return INTERNAL_ERROR;
  }

  SetModelToGeModel(cur_model, model);
  return SUCCESS;
}

Status ModelHelper::LoadWeights(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                                const size_t mode_index) const {
  ModelPartition partition;
  if (om_load_helper.GetModelPartition(ModelPartitionType::WEIGHTS_DATA, partition, mode_index) != SUCCESS) {
    GELOGE(FAILED, "[Get][ModelPartition]Failed, GetWeight size:%" PRIu64 "", partition.size);
    REPORT_CALL_ERROR("E19999", "[Get][ModelPartition]Failed, GetWeight size:%" PRIu64 "",
                      partition.size);
    return FAILED;
  }
  if (is_shared_weight_) {
    GELOGD("current weight is shared, size:%" PRIu64 "", partition.size);
    DataBuffer weight_buf(const_cast<uint8_t *>(partition.data), static_cast<size_t>(partition.size));
    cur_model->SetWeightDataBuf(weight_buf);
  } else {
    const ge::Buffer weight = ge::Buffer::CopyFrom(partition.data, static_cast<size_t>(partition.size));
    cur_model->SetWeight(weight);
  }
  GELOGD("GetWeight size:%" PRIu64 "", partition.size);
  return SUCCESS;
}

Status ModelHelper::LoadTask(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                             const size_t mode_index) const {
  ModelPartition task_partition;
  if (om_load_helper.GetModelPartition(ModelPartitionType::TASK_INFO, task_partition, mode_index) != SUCCESS) {
    GELOGE(FAILED, "Get task model partition failed.");
    GELOGE(FAILED, "[Get][ModelTaskPartition]Failed, task_partition size %" PRIu64 ", mode_index %zu",
           task_partition.size, mode_index);
    REPORT_CALL_ERROR("E19999", "Get model task partition failed, "
                       "task_partition size %" PRIu64 ", mode_index %zu", task_partition.size, mode_index);
    return FAILED;
  }
  const std::shared_ptr<domi::ModelTaskDef> task = ge::MakeShared<domi::ModelTaskDef>();
  GE_CHECK_NOTNULL(task);
  if (task_partition.size != 0U) {
    if (!ReadProtoFromArray(task_partition.data, static_cast<int32_t>(task_partition.size), task.get())) {
      GELOGE(INTERNAL_ERROR, "[Read][ProtoFromArray]Failed, task_partition size %" PRIu64 "",
             task_partition.size);
      REPORT_CALL_ERROR("E19999", "Read proto from array failed, task_partition size %" PRIu64 "",
                        task_partition.size);
      return INTERNAL_ERROR;
    }
    GELOGD("TASK_INFO op_size:%d, stream_num:%u", task->op().size(), task->stream_num());
  }
  cur_model->SetModelTaskDef(task);
  return SUCCESS;
}

Status ModelHelper::LoadTBEKernelStore(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                                       const size_t mode_index) const {
  // Load tbe kernels
  ModelPartition partition_kernel_def;
  if (om_load_helper.GetModelPartition(ModelPartitionType::TBE_KERNELS, partition_kernel_def, mode_index) == SUCCESS) {
    GELOGD("Kernels partition size:%" PRIu64 "", partition_kernel_def.size);
    if (cur_model->LoadTBEKernelStore(partition_kernel_def.data, static_cast<size_t>(partition_kernel_def.size))) {
      GELOGD("Load tbe kernels success");
    } else {
      GELOGW("Load tbe kernels failed");
    }
  }
  return SUCCESS;
}

Status ModelHelper::LoadCustAICPUKernelStore(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                                             const size_t mode_index) const {
  // Load cust aicpu kernels
  ModelPartition partition_kernel_def;
  if (om_load_helper.GetModelPartition(ModelPartitionType::CUST_AICPU_KERNELS, partition_kernel_def, mode_index)
      == SUCCESS) {
    GELOGD("Kernels partition size:%" PRIu64 "", partition_kernel_def.size);
    if (cur_model->LoadAICPUKernelStore(partition_kernel_def.data, partition_kernel_def.size)) {
      GELOGD("Load cust aicpu kernels success");
    } else {
      GELOGW("Load cust aicpu kernels failed");
    }
  }
  return SUCCESS;
}

Status ModelHelper::LoadOpSoBin(const OmFileLoadHelper &om_load_helper,
                                const GeRootModelPtr &ge_root_model) const {
  ModelPartition partition_kernel_def;
  if (om_load_helper.GetModelPartition(ModelPartitionType::SO_BINS, partition_kernel_def, 0U)
      == SUCCESS) {
    GELOGD("Kernels partition size:%" PRIu64 "", partition_kernel_def.size);
    if (ge_root_model->LoadSoBinData(partition_kernel_def.data, partition_kernel_def.size)) {
      SaveOpSoInfo(ge_root_model);
      GELOGD("Load so bin store success");
    } else {
      GELOGW("Load so bin store failed");
    }
  }
  return SUCCESS;
}

void ModelHelper::SaveOpSoInfo(const GeRootModelPtr &ge_root_model) const {
  SoInOmInfo so_info;
  (void) ge::AttrUtils::GetStr(*(model_.get()), "host_env_os", so_info.os_info);
  (void) ge::AttrUtils::GetStr(*(model_.get()), "host_env_cpu", so_info.cpu_info);
  (void) ge::AttrUtils::GetStr(*(model_.get()), ATTR_MODEL_OPP_VERSION, so_info.opp_version);
  ge_root_model->SetSoInOmInfo(so_info);
}

GeModelPtr ModelHelper::GetGeModel() {
  if (model_ != nullptr) {
    return model_;
  }

  GELOGD("Model has not been loaded!");
  const std::shared_ptr<ge::GeModel> out_model = ge::MakeShared<ge::GeModel>();
  if (out_model == nullptr) {
    return nullptr;
  }
  return out_model;
}

GeRootModelPtr ModelHelper::GetGeRootModel() {
  if (root_model_ != nullptr) {
    return root_model_;
  }

  GELOGD("Model has not been loaded!");
  const std::shared_ptr<ge::GeRootModel> out_model = ge::MakeShared<ge::GeRootModel>();
  if (out_model == nullptr) {
    return nullptr;
  }

  if (model_ != nullptr) {
    const auto root_graph = model_->GetGraph();
    if (root_graph != nullptr) {
      out_model->SetRootGraph(root_graph);
      out_model->SetModelName(model_->GetName());
      out_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), model_);
    }
  }
  return out_model;
}

Status ModelHelper::GetBaseNameFromFileName(const std::string &file_name, std::string &base_name) const {
  GELOGD("Get base_name from file, file_name:%s", file_name.c_str());
  if (file_name.empty()) {
    GELOGW("File path may not valid, check params --output");
    return FAILED;
  }
  size_t start_position = 0U;
  // using output as base_name (ignore ".om")
  const size_t filename_suffixes = 3U;
  if (file_name.find_last_of('/') != std::string::npos) {
    start_position = file_name.find_last_of('/') + 1U;
  }
  const size_t end_position = file_name.length() - filename_suffixes;
  base_name = file_name.substr(start_position, end_position - start_position);
  if (base_name.empty()) {
    GELOGW("Get base_name failed, check params --output");
    return FAILED;
  }
  return SUCCESS;
}

Status ModelHelper::GetModelNameFromMergedGraphName(const ComputeGraphPtr &compute_graph,
                                                    std::string &model_name) const {
  const auto graph_name = compute_graph->GetName();
  GELOGD("Get model_name from graph_name, graph_name:%s", graph_name.c_str());
  // this can only be used after merged graph(graph name will be append with "_x", x is index);
  if (graph_name.empty()) {
    GELOGW("File path may not valid, check params --output");
    return FAILED;
  }
  const size_t start_position = 0U;
  size_t end_position = graph_name.length();
  // using graph as model_name (ignore "_x", x is the index of graph)
  if (graph_name.find_last_of('_') != std::string::npos) {
    end_position = graph_name.find_last_of('_');
  }
  model_name = graph_name.substr(start_position, end_position);
  std::string pne_id;
  (void)AttrUtils::GetStr(compute_graph, ATTR_NAME_PROCESS_NODE_ENGINE_ID, pne_id);
  if ((!pne_id.empty()) && (model_name.length() > pne_id.length())) {
    pne_id = "_" + pne_id;
    const std::string end_name = model_name.substr(model_name.length() - pne_id.length(), model_name.length());
    if (end_name == pne_id) {
      model_name = model_name.substr(start_position, model_name.length() - pne_id.length());
    }
  }
  if (model_name.empty()) {
    GELOGW("Get model_name failed, check params --output");
    return FAILED;
  }
  GELOGD("Get model_name from graph_name, after format graph_name:%s", model_name.c_str());
  return SUCCESS;
}

Status ModelHelper::HandleDeviceInfo(fe::PlatFormInfos &platform_infos) const {
  int32_t device_id = -1;
  GE_CHK_RT_RET(rtGetDevice(&device_id));

  char_t soc_version[kSocVersionLen] = {0};
  GE_CHK_RT_RET(rtGetSocVersion(soc_version, kSocVersionLen));

  fe::PlatformInfo platform_info;
  int32_t virtual_type = 0;
  GE_CHK_STATUS_RET(GetPlatformInfo(device_id, soc_version, platform_info, virtual_type), "Get platform info failed.");

  GE_CHK_STATUS_RET(SetPlatformInfos(soc_version, platform_info, platform_infos), "Set platform infos failed.");

  GELOGD("Success to handle device info, device id: %d, soc_version: %s, virtual_type: %d.", device_id, soc_version,
         virtual_type);
  return SUCCESS;
}

Status ModelHelper::GetPlatformInfo(int32_t device_id, const std::string &soc_version,
                                    fe::PlatformInfo &platform_info, int32_t &virtual_type) const {
  if (device_id < 0) {
    GELOGE(FAILED, "Device id: %d is invalid, skip update hardware info.", device_id);
    return FAILED;
  }
#ifdef __GNUC__
  if (fe::PlatformInfoManager::GeInstance().InitializePlatformInfo() != 0U) {
    GELOGE(FAILED, "Initialize platform info failed.");
    return FAILED;
  }

  fe::OptionalInfo optional_info;
  if (fe::PlatformInfoManager::GeInstance().GetPlatformInfo(soc_version, platform_info, optional_info) != 0U) {
    GELOGE(FAILED, "Unable to get platform info.");
    return FAILED;
  }
#endif
  int64_t aic_core_cnt;
  GE_CHK_RT_RET(rtGetDeviceInfo(static_cast<uint32_t>(device_id), kModuleTypeAicore, kInfoTypeCoreNum, &aic_core_cnt));
  if (static_cast<uint32_t>(aic_core_cnt) < platform_info.soc_info.ai_core_cnt) {
    virtual_type = 1;
  }
  GELOGI("Change ai_core_cnt from platform %u to rts %ld.", platform_info.soc_info.ai_core_cnt, aic_core_cnt);
  platform_info.soc_info.ai_core_cnt = static_cast<uint32_t>(aic_core_cnt);

  int64_t vector_core_cnt = kModuleTypeVectorCore;
  // some chips have no vectore core
  (void)rtGetDeviceInfo(static_cast<uint32_t>(device_id), kModuleTypeVectorCore, kInfoTypeCoreNum, &vector_core_cnt);
  GELOGI("Change vector_core_cnt from platform %u to rts %ld.", platform_info.soc_info.vector_core_cnt,
         vector_core_cnt);
  platform_info.soc_info.vector_core_cnt = static_cast<uint32_t>(vector_core_cnt);

  rtAiCoreMemorySize_t aicore_memory_size;
  GE_CHK_RT_RET(rtAiCoreMemorySizes(&aicore_memory_size));
  GELOGI("Change l2_size from platform %lu to rts %u.", platform_info.soc_info.l2_size, aicore_memory_size.l2Size);
  platform_info.soc_info.l2_size = aicore_memory_size.l2Size;

  size_t free_mem = 0U;
  size_t total_mem_size = 0U;
  GE_CHK_RT_RET(rtMemGetInfoEx(RT_MEMORYINFO_HBM, &free_mem, &total_mem_size));
  GELOGI("Change memory_size from platform %lu to rts %zu.", platform_info.soc_info.memory_size, total_mem_size);
  platform_info.soc_info.memory_size = total_mem_size;

  GELOGI("Get platform info of device id: %d, aicore num: %ld, vector core num: %ld, l2size: %u, memory_size: %zu,"
         "virtual_type: %d.", device_id, aic_core_cnt, vector_core_cnt, aicore_memory_size.l2Size, total_mem_size,
         virtual_type);

  return SUCCESS;
}

Status ModelHelper::SetPlatformInfos(const std::string &soc_version, const fe::PlatformInfo &platform_info,
                                     fe::PlatFormInfos &platform_infos) const {
#ifdef __GNUC__
  fe::OptionalInfos optional_infos;
  if (fe::PlatformInfoManager::GeInstance().GetPlatformInfos(soc_version, platform_infos, optional_infos) != 0) {
    GELOGE(FAILED, "Unable to get platform info of soc_version: %s.", soc_version.c_str());
    return FAILED;
  }

  std::map<std::string, std::string> res;
  if (!platform_infos.GetPlatformResWithLock("SoCInfo", res)) {
    GELOGE(FAILED, "Unable to get platform info.");
    return FAILED;
  }

  res["ai_core_cnt"] = std::to_string(platform_info.soc_info.ai_core_cnt);
  res["vector_core_cnt"] = std::to_string(platform_info.soc_info.vector_core_cnt);
  res["l2_size"] = std::to_string(platform_info.soc_info.l2_size);
  res["memory_size"] = std::to_string(platform_info.soc_info.memory_size);

  platform_infos.SetPlatformResWithLock("SoCInfo", res);
  if (fe::PlatformInfoManager::GeInstance().UpdatePlatformInfos(soc_version, platform_infos) != 0U) {
    GELOGE(FAILED, "Update platform infos failed.");
    return FAILED;
  }
#endif
  return SUCCESS;
}

void ModelHelper::SetRepackSoFlag(const bool val) {
  is_repack_so_ = val;
}

Status ModelHelper::LoadModelDataAndPackSo(const ModelBufferData &model, const std::string &output_file) {
  ModelData model_data;
  model_data.model_data = model.data.get();
  model_data.model_len = model.length;

  GE_ASSERT_SUCCESS(LoadRootModel(model_data), "Load Root Model fail");
  GELOGD("LoadRootModel success.");
  ModelBufferData model_buff;
  SaveParam save_param;
  is_offline_ = true;
  if (!root_model_->CheckAndSetNeedSoInOM()) {
    return FileSaver::SaveToFile(output_file, static_cast<void *>(model.data.get()), model.length);
  }

  const auto root_graph = root_model_->GetRootGraph();
  GE_ASSERT_NOTNULL(root_graph);
  const auto root_ge_model = GetGeModel();
  root_model_->SetSubgraphInstanceNameToModel(root_graph->GetName(), root_ge_model);

  GE_ASSERT_SUCCESS(SaveToOmRootModel(root_model_, save_param, output_file, model_buff, is_unknown_shape_model_),
      "SaveToOmRootModel fail");
  GELOGD("SaveToOmRootModel success.");
  return SUCCESS;
}

Status ModelHelper::EnsureKernelBuilt(const GeModelPtr &model) {
  if (model->GetTBEKernelStore().DataSize() == 0U) {
    TBEKernelStore tbe_kernel_store = model->GetTBEKernelStore();
    GE_CHK_BOOL_RET_STATUS(tbe_kernel_store.Build(), FAILED, "tbe_kernel_store build failed, model=%s.",
                           model->GetName().c_str());
    model->SetTBEKernelStore(tbe_kernel_store);
  }

  if (model->GetCustAICPUKernelStore().DataSize() == 0U) {
    CustAICPUKernelStore cust_aicpu_kernel_store = model->GetCustAICPUKernelStore();
    GE_CHK_BOOL_RET_STATUS(cust_aicpu_kernel_store.Build(), FAILED, "cust_aicpu_kernel_store build failed, model=%s.",
                           model->GetName().c_str());
    model->SetCustAICPUKernelStore(cust_aicpu_kernel_store);
  }
  return SUCCESS;
}
}  // namespace ge
