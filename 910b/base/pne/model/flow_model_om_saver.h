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

#ifndef BASE_PNE_MODEL_FLOW_MODEL_OM_SAVER_H_
#define BASE_PNE_MODEL_FLOW_MODEL_OM_SAVER_H_

#include "framework/pne/flow_model.h"
#include "google/protobuf/message.h"
#include "common/model/model_relation.h"
#include "framework/common/helper/om_file_helper.h"
#include "graph/buffer.h"

namespace ge {
class FlowModelOmSaver {
 public:
  explicit FlowModelOmSaver(const FlowModelPtr &flow_model) : flow_model_(flow_model) {}
  ~FlowModelOmSaver() = default;
  Status SaveToOm(const std::string &output_file);
  Status SaveToModelData(ModelBufferData &model_buff);

 private:
  Status AddModelDefPartition();
  Status AddFlowModelPartition();
  Status AddFlowSubModelPartitions();
  Status AddFlowModelCompileResource(flow_model::proto::FlowModelDef &flow_model_def) const;
  Status UpdateModelHeader();
  Status AddPartition(const google::protobuf::Message &partition_msg, ModelPartitionType partition_type);
  Status AddPartition(Buffer &buffer, ModelPartitionType partition_type);
  Status SaveFlowModelToFile(const std::string &output_file);
  Status SaveFlowModelToDataBuffer(ModelBufferData &model_buff);

  /**
   * @brief fix non standard graph load failed.
   * flow model is seperate by partitionCall, graph output node and subgraph is incorrect.
   * now just remove output nodes and subgraphs.
   * @param graph graph.
   */
  static void FixNonStandardGraph(const ComputeGraphPtr &graph);

  const FlowModelPtr flow_model_;
  OmFileSaveHelper om_file_save_helper_;
  // used for cache partition buffer before save to file.
  std::vector<Buffer> buffers_;
};
}  // namespace ge
#endif  // BASE_PNE_MODEL_FLOW_MODEL_OM_SAVER_H_