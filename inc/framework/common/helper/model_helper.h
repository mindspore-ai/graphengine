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

#ifndef INC_FRAMEWORK_COMMON_HELPER_MODEL_HELPER_H_
#define INC_FRAMEWORK_COMMON_HELPER_MODEL_HELPER_H_

#include <memory>
#include <string>

#include "common/fmk_types.h"
#include "common/helper/om_file_helper.h"
#include "common/types.h"
#include "graph/model.h"
#include "model/ge_model.h"
#include "model/ge_root_model.h"

namespace ge {
class ModelHelper {
 public:
  ModelHelper() = default;
  ~ModelHelper();

  Status SaveToOmModel(const GeModelPtr &ge_model, const SaveParam &save_param, const std::string &output_file,
                       ge::ModelBufferData &model);
  Status SaveToOmRootModel(const GeRootModelPtr &ge_root_model, const SaveParam &save_param, const string &output_file,
                           ModelBufferData &model, bool is_unknown_shape);
  Status SaveOriginalGraphToOmModel(const ge::Graph &graph, const std::string &output_file);
  Status LoadModel(const ge::ModelData &model_data);
  Status LoadRootModel(const ge::ModelData &model_data);
  Status GetModelBufferData(ge::ModelBufferData &model);

  const ModelFileHeader *GetFileHeader() const { return file_header_; }

  GeModelPtr GetGeModel();
  GeRootModelPtr GetGeRootModel();
  void SetSaveMode(bool val) { is_offline_ = val; }
  bool GetSaveMode(void) const { return is_offline_; }
  bool GetModelType() const { return is_unknown_shape_model_; };

  Status GetBaseNameFromFileName(const std::string &file_name, std::string &base_name);
  Status GetModelNameFromMergedGraphName(const std::string &graph_name, std::string &model_name);

 private:
  bool is_assign_model_ = false;
  bool is_offline_ = true;
  bool is_unknown_shape_model_ = false;
  ModelFileHeader *file_header_ = nullptr;
  // Encrypted model need delete temp model and unencrypted model need not delete model
  uint8_t *model_addr_tmp_ = nullptr;
  uint32_t model_len_tmp_ = 0;
  GeModelPtr model_;
  GeRootModelPtr root_model_;

  ModelHelper(const ModelHelper &);
  ModelHelper &operator=(const ModelHelper &);
  Status GenerateGeModel(OmFileLoadHelper &om_load_helper);
  Status GenerateGeRootModel(OmFileLoadHelper &om_load_helper);
  Status LoadModelData(OmFileLoadHelper &om_load_helper);
  void SetModelToGeModel(GeModelPtr &ge_model, Model &model);
  Status LoadModelData(OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model, size_t mode_index);
  Status LoadWeights(OmFileLoadHelper &om_load_helper);
  Status LoadWeights(OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model, size_t mode_index);
  Status LoadTask(OmFileLoadHelper &om_load_helper);
  Status LoadTask(OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model, size_t mode_index);
  Status LoadTBEKernelStore(OmFileLoadHelper &om_load_helper);
  Status LoadTBEKernelStore(OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model, size_t mode_index);
  Status LoadCustAICPUKernelStore(OmFileLoadHelper &om_load_helper);
  Status LoadCustAICPUKernelStore(OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model, size_t mode_index);
  Status ReleaseLocalModelData() noexcept;
  Status SaveModelPartition(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, ModelPartitionType type,
                            const uint8_t *data, size_t size, size_t model_index);
  Status SaveModelDef(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                      Buffer &model_buffer, size_t model_index = 0);
  Status SaveSizeToModelDef(const GeModelPtr &ge_model);
  Status SaveModelWeights(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                          size_t model_index = 0);
  Status SaveModelTbeKernel(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                            size_t model_index = 0);
  Status SaveModelCustAICPU(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                            size_t model_index = 0);
  Status SaveModelTaskDef(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                          Buffer &task_buffer, size_t model_index = 0);
  Status SaveModelHeader(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                         size_t model_num = 1);
  Status SaveAllModelPartiton(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                              Buffer &model_buffer, Buffer &task_buffer, size_t model_index = 0);
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_HELPER_MODEL_HELPER_H_
