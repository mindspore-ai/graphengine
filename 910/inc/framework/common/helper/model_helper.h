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

#ifndef INC_FRAMEWORK_COMMON_HELPER_MODEL_HELPER_H_
#define INC_FRAMEWORK_COMMON_HELPER_MODEL_HELPER_H_

#include <memory>
#include <string>

#include "framework/common/helper/om_file_helper.h"
#include "framework/common/helper/model_save_helper.h"
#include "framework/common/types.h"
#include "graph/model.h"
#include "platform/platform_info.h"
#include "common/op_so_store/op_so_store.h"

namespace ge {
class GeModel;
class GeRootModel;
class GE_FUNC_VISIBILITY ModelHelper : public ModelSaveHelper {
 public:
  ModelHelper() noexcept = default;
  virtual ~ModelHelper() override = default;
  ModelHelper(const ModelHelper &) = default;
  ModelHelper &operator=(const ModelHelper &) & = default;

  Status SaveToOmModel(const GeModelPtr &ge_model, const std::string &output_file,
                       ge::ModelBufferData &model, const GeRootModelPtr &ge_root_model = nullptr) override;
  Status GenerateGeModel(const OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model,
                         GeModelPtr &first_ge_model, const size_t mode_index,
                         const bool is_dyn_root) const;
  Status SaveToOmRootModel(const GeRootModelPtr &ge_root_model, const std::string &output_file,
                           ModelBufferData &model, const bool is_unknown_shape) override;
  Status SaveOriginalGraphToOmModel(const ge::Graph &graph, const std::string &output_file) const;
  Status LoadModel(const ge::ModelData &model_data);
  Status LoadRootModel(const ge::ModelData &model_data);
  static Status GetModelFileHead(const ge::ModelData &model_data, const ModelFileHeader *&file_header);
  static Status SetModelToGeModel(const GeModelPtr &ge_model, const GeModelPtr &first_ge_model, Model &model);
  static std::string GetOutputFileName() { return output_file_name_; }
  Status LoadPartInfoFromModel(const ge::ModelData &model_data, ModelPartition &partition);

  GeModelPtr GetGeModel();
  GeRootModelPtr GetGeRootModel();
  virtual void SetSaveMode(const bool val) override {
    is_offline_ = val;
  }

  void SetSharedWeightFlag(const bool val) {
    is_shared_weight_ = val;
  }

  bool GetModelType() const {
    return is_unknown_shape_model_;
  }

  Status GetBaseNameFromFileName(const std::string &file_name, std::string &base_name) const;
  Status GetModelNameFromMergedGraphName(const ComputeGraphPtr &compute_graph, std::string &model_name) const;

  // for soft sync op
  Status GetHardwareInfo(std::map<std::string, std::string> &options) const;
  Status HandleDeviceInfo(fe::PlatFormInfos &platform_infos) const;
  Status GetPlatformInfo(int32_t device_id, const std::string &soc_version, fe::PlatformInfo &platform_info,
                         int32_t &virtual_type) const;
  Status SetPlatformInfos(const std::string &soc_version, const fe::PlatformInfo &platform_info,
                          fe::PlatFormInfos &platform_infos) const;

  Status CheckOsCpuInfoAndOppVersion();

  Status GetSoBinData(const string &cpu_info, const string &os_info);

  const uint8_t *GetOpSoStoreData() const;

  size_t GetOpStoreDataSize() const;

  void SetRepackSoFlag(const bool val);

  Status LoadModelDataAndPackSo(const ModelData &model_data, const std::string &output_file);

  static constexpr const char_t *kFilePreffix = ".exeom";
  static constexpr const char_t *kDebugPreffix = ".dbg";

 protected:
  Status SaveModelCustAICPU(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                            const size_t model_index = 0U) const;
  static Status GetOppVersion(std::string &version);
  static Status EnsureKernelBuilt(const GeModelPtr &model);
  Status SaveSoStoreModelPartitionInfo(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                       const GeRootModelPtr &ge_root_model, string &output_file_name,
                                       const GeModelPtr &first_ge_model);
  Status SaveModelHeader(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                         const size_t model_num = 1U, const bool need_check_os_cpu = false,
                         const bool is_unknow_shape = false) const;
  Status SaveModelPartition(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const ModelPartitionType type,
                            const uint8_t* const data, const size_t size, const size_t model_index) const;
  Status SaveModelWeights(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                          const size_t model_index = 0U) const;
  Status SaveModelIntroduction(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                               const GeModelPtr &ge_model, const size_t model_index = 0U) const;
  Status SaveModelTbeKernel(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                            const size_t model_index = 0U) const;
  GeModelPtr model_;
  bool is_so_store_ = false;

 private:
  bool is_assign_model_ = false;
  bool is_offline_ = true;
  bool is_unknown_shape_model_ = false;
  bool is_shared_weight_ = false;
  const ModelFileHeader *file_header_ = nullptr;
  GeRootModelPtr root_model_;
  OpSoStore op_so_store_;
  bool is_repack_so_ = false;
  bool is_need_compress_ = true;
  static std::string output_file_name_;
  std::unordered_set<std::string> custom_compiler_versions_{};

  bool IsPartitionedGraph(const GeModelPtr &cur_model) const;

  Status GenerateGeRootModel(const OmFileLoadHelper &om_load_helper);

  Status LoadModelData(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                       const GeModelPtr &first_ge_model, const size_t mode_index) const;
  virtual Status LoadWeights(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                     const size_t mode_index) const;
  Status LoadTask(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model, const size_t mode_index) const;
  Status LoadTBEKernelStore(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                            const size_t mode_index) const;
  Status LoadCustAICPUKernelStore(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                                  const size_t mode_index) const;


  Status SaveModelDef(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                      Buffer &model_buffer, const size_t model_index = 0U) const;
  Status SaveSizeToModelDef(const GeModelPtr &ge_model, const size_t model_index) const;

  Status SaveModelTaskDef(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                          Buffer &task_buffer, const size_t model_index = 0U) const;
  Status SaveAllModelPartiton(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                              Buffer &model_buffer, Buffer &task_buffer, const size_t model_index = 0U) const;

  Status LoadOpSoBin(const OmFileLoadHelper &om_load_helper, const GeRootModelPtr &ge_root_model) const;

  void SaveOpSoInfo(const GeRootModelPtr &ge_root_model) const;
  Status SetModelCompilerVersion(const GeModelPtr &first_ge_model);
  Status LoadAndStoreOppSo(const string &path);
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_HELPER_MODEL_HELPER_H_
