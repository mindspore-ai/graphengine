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

#include "graph/build/memory/buffer_pool_mem_assigner.h"
#include "common/omg_util.h"
#include "graph/utils/tensor_utils.h"
#include "framework/common/util.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "common/math/math_util.h"
#include "common/util/error_manager/error_manager.h"

namespace ge {
namespace {
const size_t kBufferPoolNodeMemInfoLength = 2;
const uint32_t kBufferPoolNodeOutputSizeIndex = 0;
const uint32_t kBufferPoolNodeOutputOffsetIndex = 1;
} // namespace

Status BufferPoolMemAssigner::Assign() {
  if (compute_graph_ == nullptr) {
    GELOGE(PARAM_INVALID, "[Check][Graph]Graph is nullptr");
    REPORT_INNER_ERROR("E19999", "Input graph is nullptr");
    return PARAM_INVALID;
  }
  Status ret = InitAssigner(compute_graph_);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Init][Assigner]Graph:%s.", compute_graph_->GetName().c_str());
    return FAILED;
  }
  ret = AssignOutput();
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Assign][Output]Graph:%s.", compute_graph_->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status BufferPoolMemAssigner::GetOutputMemoryType(const NodePtr &node, size_t idx, int64_t &memory_type) {
  GE_CHECK_NOTNULL(node->GetOpDesc());
  memory_type = RT_MEMORY_HBM;
  std::vector<int64_t> type_list;
  bool has_mem_type = ge::AttrUtils::GetListInt(node->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, type_list);
  if (has_mem_type && (type_list.size() != node->GetOpDesc()->GetOutputsSize() || idx >= type_list.size())) {
    GELOGE(PARAM_INVALID, "[Check][OutputParam]Output param invalid, output size:%zu, mem type size:%zu, index:%zu.",
           node->GetOpDesc()->GetOutputsSize(), type_list.size(), idx);
    REPORT_INNER_ERROR("E19999", "Output param invalid, output size:%zu, mem type size:%zu, index:%zu.",
                       node->GetOpDesc()->GetOutputsSize(), type_list.size(), idx);
    return PARAM_INVALID;
  }
  memory_type = has_mem_type ? type_list[idx] : RT_MEMORY_HBM;
  return SUCCESS;
}

Status BufferPoolMemAssigner::InitAssigner(const ComputeGraphPtr &graph) {
  for (const NodePtr &node : graph->GetAllNodes()) {
    int64_t buffer_pool_id = 0;
    int64_t buffer_pool_size = 0;
    bool get_attr = AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_BUFFER_POOL_ID, buffer_pool_id);
    get_attr = get_attr && (AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_BUFFER_POOL_SIZE, buffer_pool_size));
    if (get_attr) {
      std::string batch_label;
      (void) AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, batch_label);
      buffer_pool_nodes_[batch_label][buffer_pool_id].emplace_back(node);
      auto iter = buffer_pool_size_[batch_label].find(buffer_pool_id);
      if (iter == buffer_pool_size_[batch_label].end()) {
        buffer_pool_size_[batch_label][buffer_pool_id] = buffer_pool_size;
      }
      Status ret = InitMemOffsetBase(node);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Init][MemOffsetBase]Batch label:%s.", batch_label.c_str());
        REPORT_INNER_ERROR("E19999", "Failed to init offset base, batch label:%s.", batch_label.c_str());
        return ret;
      }
    }
  }

  int64_t max_size = 0;
  for (const auto &iter : buffer_pool_size_) {
    std::string batch_label = iter.first;
    int64_t batch_offset = mem_offset_base_;
    for (const auto &buffer_pool : iter.second) {
      int64_t buffer_pool_id = buffer_pool.first;
      int64_t buffer_pool_size = buffer_pool.second;
      buffer_pool_offset_base_[batch_label][buffer_pool_id] = batch_offset;
      FMK_INT64_ADDCHECK(buffer_pool_size, kBufferPoolMemAlignSize);
      AlignMemSize(buffer_pool_size, kBufferPoolMemAlignSize);
      FMK_INT64_ADDCHECK(batch_offset, (buffer_pool_size + kBufferPoolMemAlignSize));
      batch_offset += (buffer_pool_size + kBufferPoolMemAlignSize);
    }
    int64_t batch_mem_size = batch_offset - mem_offset_base_;
    GELOGI("[Init][Assigner]Get batch mem size, batch label:%s, mem size:%ld.", batch_label.c_str(), batch_mem_size);
    if (max_size < batch_mem_size) {
      max_size = batch_mem_size;
    }
  }
  FMK_INT64_ADDCHECK(mem_offset_base_, max_size);
  mem_offset_ = static_cast<size_t>(mem_offset_base_ + max_size);
  GELOGI("[Init][Assigner]Init buffer pool mem assigner successfully, "
         "mem type:%ld, mem offset base:%ld, mem offset:%zu.", mem_type_, mem_offset_base_, mem_offset_);
  return SUCCESS;
}

Status BufferPoolMemAssigner::InitMemOffsetBase(const NodePtr &node) {
  int64_t mem_type;
  Status ret = GetOutputMemoryType(node, static_cast<size_t>(kBufferPoolNodeOutIndex), mem_type);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][MemType]Node:%s, index:%u.", node->GetName().c_str(), kBufferPoolNodeOutIndex);
    REPORT_INNER_ERROR("E19999", "Failed to get output memory type, node:%s, index:%u.",
                       node->GetName().c_str(), kBufferPoolNodeOutIndex);
    return ret;
  }
  if (mem_type_ != mem_type && init_offset_base_) {
    GELOGE(PARAM_INVALID, "[Check][MemType]The memory type of all buffer pool nodes must be the same, node:%s, "
           "required:%ld, actually: %ld", node->GetName().c_str(), mem_type_, mem_type);
    REPORT_INNER_ERROR("E19999", "The memory type of all buffer pool nodes must be the same, node:%s, "
                                 "required:%ld, actually: %ld", node->GetName().c_str(), mem_type_, mem_type);
    return PARAM_INVALID;
  }
  if (!init_offset_base_) {
    auto iter = mem_type_to_offset_.find(mem_type);
    if (iter == mem_type_to_offset_.end()) {
      GELOGE(PARAM_INVALID, "[Check][MemType]Memory type is not supported, node:%s, mem type:%ld.",
             node->GetName().c_str(), mem_type);
      REPORT_INNER_ERROR("E19999", "Memory type is not supported, node:%s, mem type:%ld.",
                         node->GetName().c_str(), mem_type);
      return PARAM_INVALID;
    }
    mem_offset_base_ = static_cast<int64_t>(iter->second);
    FMK_INT64_ADDCHECK(mem_offset_base_, (kBufferPoolMemAlignSize + kBufferPoolMemAlignSize));
    AlignMemSize(mem_offset_base_, kBufferPoolMemAlignSize);
    // The HCOM nodes may access the previous 512 bytes.
    mem_offset_base_ += kBufferPoolMemAlignSize;
    mem_type_ = mem_type;
    init_offset_base_ = true;
    GELOGI("[Init][MemOffsetBase]Init offset base:%ld, memory type:%ld", mem_offset_base_, mem_type);
  }
  return SUCCESS;
}

Status BufferPoolMemAssigner::AssignOutput() {
  for (auto &batch_pool_nodes_map : buffer_pool_nodes_) {
    std::string batch_label = batch_pool_nodes_map.first;
    for (auto &pool_nodes_map : batch_pool_nodes_map.second) {
      int64_t buffer_pool_id = pool_nodes_map.first;
      auto iter_buffer_id_size = buffer_pool_size_[batch_label].find(buffer_pool_id);
      if (iter_buffer_id_size == buffer_pool_size_[batch_label].end()) {
        GELOGE(INTERNAL_ERROR, "[Get][BufferPoolSize]Pool id:%ld.", buffer_pool_id);
        REPORT_INNER_ERROR("E19999", "Failed to get buffer pool size, pool id:%ld.", buffer_pool_id);
        return INTERNAL_ERROR;
      }
      auto iter_buffer_id_offset = buffer_pool_offset_base_[batch_label].find(buffer_pool_id);
      if (iter_buffer_id_offset == buffer_pool_offset_base_[batch_label].end()) {
        GELOGE(INTERNAL_ERROR, "[Get][BufferPoolBaseOffset]Pool id:%ld.", buffer_pool_id);
        REPORT_INNER_ERROR("E19999", "Failed to get buffer pool base offset, pool id:%ld.", buffer_pool_id);
        return INTERNAL_ERROR;
      }
      int64_t buffer_pool_size = iter_buffer_id_size->second;
      int64_t output_offset_base = iter_buffer_id_offset->second;
      Status ret = AssignOutputInOneBufferPool(batch_label, output_offset_base, pool_nodes_map.second);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Assign][OneBufferPool]Batch label:%s, pool id:%ld, pool size:%ld, offset base:%ld.",
               batch_label.c_str(), buffer_pool_id, buffer_pool_size, output_offset_base);
        REPORT_INNER_ERROR("E19999", "Failed to assign output memory, batch label:%s, "
                           "pool id:%ld, pool size:%ld, offset base:%ld.",
                           batch_label.c_str(), buffer_pool_id, buffer_pool_size, output_offset_base);
        return ret;
      }
      GELOGI("[Assign][Output]Assign output successfully, batch label:%s, pool id:%ld, pool size:%ld, offset base:%ld.",
             batch_label.c_str(), buffer_pool_id, buffer_pool_size, output_offset_base);
    }
  }
  return SUCCESS;
}

Status BufferPoolMemAssigner::AssignOutputInOneBufferPool(const std::string &batch_label,
                                                          int64_t output_offset_base,
                                                          const std::vector<NodePtr> &buffer_pool_nodes) {
  for (const NodePtr &node : buffer_pool_nodes) {
    int64_t output_size = 0;
    Status ret = GetMemorySize(node, output_size);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Get][MemSize]Node:%s.", node->GetName().c_str());
      REPORT_INNER_ERROR("E19999", "Failed to get output size, node:%s.", node->GetName().c_str());
      return ret;
    }
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    vector<int64_t> memory_size_and_offset;
    bool get_attr = AttrUtils::GetListInt(op_desc, ATTR_NAME_BUFFER_POOL_NODE_SIZE_AND_OFFSET, memory_size_and_offset);
    if (!get_attr || memory_size_and_offset.size() != kBufferPoolNodeMemInfoLength) {
      GELOGE(PARAM_INVALID, "[Get][Attr]Node:%s, mem info size:%zu, required size:%zu.",
             node->GetName().c_str(), memory_size_and_offset.size(), kBufferPoolNodeMemInfoLength);
      REPORT_INNER_ERROR("E19999", "Failed to get pool node memory info, node:%s, info size:%zu, required size:%zu.",
                         node->GetName().c_str(), memory_size_and_offset.size(), kBufferPoolNodeMemInfoLength);
      return PARAM_INVALID;
    }
    if (output_size != memory_size_and_offset[kBufferPoolNodeOutputSizeIndex]) {
      GELOGE(PARAM_INVALID, "[Check][MemSize]Something wrong with memory size, pre size:%ld, curr size:%ld, node:%s.",
             memory_size_and_offset[kBufferPoolNodeOutputSizeIndex], output_size, node->GetName().c_str());
      REPORT_INNER_ERROR("E19999", "Something wrong with memory size, pre size:%ld, curr size:%ld, node:%s.",
                         memory_size_and_offset[kBufferPoolNodeOutputSizeIndex], output_size, node->GetName().c_str());
      return PARAM_INVALID;
    }

    int64_t logical_offset = memory_size_and_offset[kBufferPoolNodeOutputOffsetIndex];
    vector<int64_t> output_list = {(output_offset_base + logical_offset)};
    op_desc->SetOutputOffset(output_list);
    // log for IMAS tools
    GELOGI("[IMAS]Set %s name[%s] optype[%s] %s[%u] offset to [%ld] streamid[%ld] memtype[%ld] "
           "size[%zu] realsize[%zu] noalignsize[%zu] life time begin[%d] life time end[%d] "
           "child[%d:%d:%d:%d:%d] isref[%d] batch[%s]",
           compute_graph_->GetName().c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str(),
           "output", kBufferPoolNodeOutIndex, output_list[kBufferPoolNodeOutIndex], op_desc->GetStreamId(), mem_type_,
           static_cast<size_t>(output_size), static_cast<size_t>(output_size), static_cast<size_t>(output_size),
           0, 0, 0, 0, 0, 0, 0, 0, batch_label.c_str());
  }
  return SUCCESS;
}

}  // namespace ge
