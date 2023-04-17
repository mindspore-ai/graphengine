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
#include "common/model/ge_model.h"
#include "common/model/ge_root_model.h"
#include "framework/common/types.h"
#include "graph/model.h"

namespace ge {
class GE_FUNC_VISIBILITY ModelHelper {
 public:
  ModelHelper() = default;
  ~ModelHelper();

  Status SaveToOmModel(const GeModelPtr &ge_model, const SaveParam &save_param, const std::string &output_file,
                       ge::ModelBufferData &model) const;
  Status GenerateGeModel(const OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model, const size_t mode_index,
                         const bool is_dyn_root) const;
  Status SaveToOmRootModel(const GeRootModelPtr &ge_root_model, const SaveParam &save_param,
                           const std::string &output_file, ModelBufferData &model, const bool is_unknown_shape) const;
  Status SaveOriginalGraphToOmModel(const ge::Graph &graph, const std::string &output_file) const;
  Status LoadModel(const ge::ModelData &model_data);
  Status LoadRootModel(const ge::ModelData &model_data);
  static Status GetModelFileHead(const ge::ModelData &model_data, ModelFileHeader *&file_header);
  static void SetModelToGeModel(const GeModelPtr &ge_model, Model &model);

  GeModelPtr GetGeModel();
  GeRootModelPtr GetGeRootModel();
  void SetSaveMode(const bool val) {
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

 private:
  bool is_assign_model_ = false;
  bool is_offline_ = true;
  bool is_unknown_shape_model_ = false;
  bool is_shared_weight_ = false;
  ModelFileHeader *file_header_ = nullptr;
  GeModelPtr model_;
  GeRootModelPtr root_model_;

  ModelHelper(const ModelHelper &) = default;
  ModelHelper &operator=(const ModelHelper &) = default;

  bool IsPartitionedGraph(const GeModelPtr &cur_model) const;

  Status GenerateGeRootModel(const OmFileLoadHelper &om_load_helper);

  Status LoadModelData(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                       const size_t mode_index) const;
  Status LoadWeights(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                     const size_t mode_index) const;
  Status LoadTask(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model, const size_t mode_index) const;
  Status LoadTBEKernelStore(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                            const size_t mode_index) const;
  Status LoadCustAICPUKernelStore(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                                  const size_t mode_index) const;

  Status SaveModelPartition(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const ModelPartitionType type,
                            const uint8_t *const data, const size_t size, const size_t model_index) const;
  Status SaveModelDef(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                      Buffer &model_buffer, const size_t model_index = 0U) const;
  Status SaveSizeToModelDef(const GeModelPtr &ge_model) const;
  Status SaveModelWeights(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                          const size_t model_index = 0U) const;
  Status SaveModelTbeKernel(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                            const size_t model_index = 0U) const;
  Status SaveModelCustAICPU(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                            const size_t model_index = 0U) const;
  Status SaveModelTaskDef(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                          Buffer &task_buffer, const size_t model_index = 0U) const;
  Status SaveModelHeader(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                         const size_t model_num = 1U) const;
  Status SaveAllModelPartiton(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                              Buffer &model_buffer, Buffer &task_buffer, const size_t model_index = 0U) const;
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_HELPER_MODEL_HELPER_H_
