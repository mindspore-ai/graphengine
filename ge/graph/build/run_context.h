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

#ifndef GE_GRAPH_BUILD_RUN_CONTEXT_H_
#define GE_GRAPH_BUILD_RUN_CONTEXT_H_

#include <vector>
#include "common/ge_inner_error_codes.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "framework/common/types.h"
#include "graph/compute_graph.h"
#include "graph/model.h"
#include "runtime/rt.h"

namespace ge {
class RunContextUtil {
 public:
  RunContextUtil() = default;

  virtual ~RunContextUtil();

  // Init mem info.
  ge::Status InitMemInfo(uint8_t *data_mem_base, uint64_t data_mem_size,
                         std::map<int64_t, uint8_t *> mem_type_to_data_mem_base,
                         std::map<int64_t, uint64_t> mem_type_to_data_mem_size,
                         uint8_t *weight_mem_base, uint64_t weight_mem_size);

  ge::Status CreateRunContext(Model &model_def, const ComputeGraphPtr &graph, Buffer &buffer,
                              const uint64_t session_id);

  RunContext &GetRunContext();

  RunContext run_context_;

 private:
  // Create Rt model/stream/event/label for task generate
  ge::Status CreateRtModelResources(uint32_t stream_num, uint32_t event_num, uint32_t label_num);

  // Destroy Rt model/stream/event/label
  void DestroyRtModelResources() noexcept;

  // Model
  rtModel_t rt_model_ = nullptr;
  std::vector<rtStream_t> stream_list_;
  std::vector<rtEvent_t> event_list_;
  std::vector<rtEvent_t> label_list_;

  // Mem info
  uint8_t *data_mem_base_ = nullptr;
  uint64_t data_mem_size_ = 0;
  uint8_t *weight_mem_base_ = nullptr;
  uint64_t weight_mem_size_ = 0;
  std::map<int64_t, uint8_t *> mem_type_to_data_mem_base_;
  std::map<int64_t, uint64_t> mem_type_to_data_mem_size_;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_RUN_CONTEXT_H_
