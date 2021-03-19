/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "graph/build/memory/hybrid_mem_assigner.h"
#include <utility>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "graph/build/memory/binary_block_mem_assigner.h"
#include "graph/build/memory/max_block_mem_assigner.h"

namespace ge {
HybridMemAssigner::HybridMemAssigner(ge::ComputeGraphPtr compute_graph)
    : mem_offset_(0), p2p_mem_offset_(0), compute_graph_(std::move(compute_graph)), priority_assigner_(nullptr) {}

Status HybridMemAssigner::AssignMemory(std::unique_ptr<BlockMemAssigner> &block_assigner, size_t &mem_size) {
  vector<int64_t> ranges;
  GE_CHECK_NOTNULL(block_assigner);
  if (block_assigner->GetMemoryRanges(ranges) != SUCCESS) {
    GELOGE(FAILED, "GetMemoryRanges Fail!");
    return FAILED;
  }
  GE_IF_BOOL_EXEC(ranges.empty(), return SUCCESS);

  block_assigner->AssignMemoryWithReuse(ranges);

  mem_size = block_assigner->GetMemOffset();
  return SUCCESS;
}

Status HybridMemAssigner::Assign() {
  if (GraphUtils::GetRefMapping(compute_graph_, symbol_to_anchors_, anchor_to_symbol_) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Get ref-mapping for graph %d failed.", compute_graph_->GetName().c_str());
    return FAILED;
  }

  std::unique_ptr<BlockMemAssigner> binary_assigner(new (std::nothrow) BinaryBlockMemAssigner(
      compute_graph_, anchor_to_symbol_, symbol_to_anchors_));
  GE_CHECK_NOTNULL(binary_assigner);

  std::unique_ptr<BlockMemAssigner> max_assigner(new (std::nothrow) MaxBlockMemAssigner(
      compute_graph_, anchor_to_symbol_, symbol_to_anchors_));
  GE_CHECK_NOTNULL(max_assigner);

  size_t bin_mem_size = 0;
  size_t max_mem_size = 0;

  GE_CHK_STATUS_RET(AssignMemory(binary_assigner, bin_mem_size), "BinaryBlock Method AssignMemory Fail!");
  GE_CHK_STATUS_RET(AssignMemory(max_assigner, max_mem_size), "MaxBlock Method AssignMemory Fail!");

  std::unique_ptr<BlockMemAssigner> priority_assigner;

  GELOGD("Binary-block memory size:%zu, max-block memory size:%zu", bin_mem_size, max_mem_size);
  if (bin_mem_size <= max_mem_size) {
    GELOGD("Use binary-block memory assigner method");
    priority_assigner = std::move(binary_assigner);
  } else {
    GELOGI("Use max-block memory assigner method");
    priority_assigner = std::move(max_assigner);
  }

  priority_assigner->SetOpMemOffset(false);
  mem_offset_ = priority_assigner->GetMemOffset();
  p2p_mem_offset_ = priority_assigner->GetP2PMemOffset();
  priority_assigner_ = std::move(priority_assigner);

  return SUCCESS;
}
}  // namespace ge
