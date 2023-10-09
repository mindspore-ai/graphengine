/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef INC_FRAMEWORK_PNE_FLOW_MODEL_HELPER_H_
#define INC_FRAMEWORK_PNE_FLOW_MODEL_HELPER_H_
#include "pne/flow_model.h"
#include "common/ge_common/ge_types.h"

namespace ge {
class GE_FUNC_VISIBILITY FlowModelHelper {
 public:
  static Status LoadToFlowModel(const std::string &model_path, FlowModelPtr &flow_model);
  static Status UpdateSessionGraphId(const FlowModelPtr &flow_model, const std::string &session_graph_id);
  static Status SaveToOmModel(const FlowModelPtr &flow_model, const std::string &output_file);
  static Status LoadFlowModelFromBuffData(const ModelBufferData &model_buffer_data,
                                          ge::FlowModelPtr &flow_model);
  static Status LoadFlowModelFromOmFile(const char_t *const model_path, ge::FlowModelPtr &flow_model);
 private:
  static Status LoadGeRootModelToFlowModel(const ModelData &model, FlowModelPtr &flow_model);
  static Status UpdateSessionGraphId(const ComputeGraphPtr &graph, const std::string &session_graph_id);
  static Status TransModelDataToFlowModel(const ge::ModelData &model_data, ge::FlowModelPtr &flow_model);
};

}  // namespace ge

#endif  // INC_FRAMEWORK_PNE_FLOW_MODEL_HELPER_H_