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

#ifndef INC_COMMON_UTILS_AI_CORE_COMMON_TYPES_H_
#define INC_COMMON_UTILS_AI_CORE_COMMON_TYPES_H_

#include "graph/anchor.h"
#include "graph/types.h"
#include "runtime/kernel.h"
#include <map>
#include <string>
#include <vector>

namespace fe {
struct FusionOpSrc {
  uint32_t src_op_id;
  ge::AnchorPtr src_anchor;
  int32_t fusion_src_index;
  int32_t fusion_dst_index;
};

struct FusionOpDst {
  uint32_t dst_op_id;
  ge::AnchorPtr dst_anchor;
};

struct FusionDataFlow {
  std::pair<ge::AnchorPtr, ge::AnchorPtr> edge;
  std::pair<std::string, ge::AnchorPtr> node_dataindex_pair;
};

typedef struct tag_l2_fusion_data {
  uint32_t l2Index;
  uint64_t l2Addr;
  uint64_t l2PageNum;
} L2FusionData_t;
typedef std::map<uint64_t, L2FusionData_t> L2FusionDataMap_t;

typedef struct tag_fe_sm_desc {
  rtL2Ctrl_t l2ctrl;
  std::string node_name[8];
  uint8_t output_index[8];
} fe_sm_desc_t;

typedef struct TagTaskL2FusionInfo {
  std::string node_name;
  fe_sm_desc_t l2_info;
  L2FusionDataMap_t input;
  L2FusionDataMap_t output;
  uint32_t is_used;
} TaskL2FusionInfo_t;

using L2FusionInfoPtr = std::shared_ptr<TaskL2FusionInfo_t>;

typedef struct ToOpStruct {
  int64_t op_l1_space = 0;
  std::vector<int64_t> op_l1_fusion_type;
  int64_t op_l1_workspace_flag = 0; // for workspace flag
  int64_t op_l1_workspace_size = 0;
  std::vector<std::vector<int64_t>> valid_input_shape;
  std::vector<std::vector<int64_t>> valid_output_shape;
  std::vector<std::vector<int64_t>>
      slice_input_offset; // conv & pooling & ReadSelect
  std::vector<std::vector<int64_t>> slice_output_offset; // WriteSelect
  std::vector<uint32_t> total_shape;
  uint32_t split_index = 0;
  ToOpStruct() {
    // set invalid value for essential variable
    op_l1_space = -1;
    op_l1_workspace_size = -1;
  }
} ToOpStruct_t;

enum SlicePattern {
  ELEMENT_WISE = 0,
  ELEMENT_WISE_BROADCAST,
  BROADCAST,
  SLIDING_WINDOW,
  SLIDING_WINDOW_DECONV,
  CUBE_MATMUL,
  SLICE_PATTERN_REDUCE,
  SLICE_PATTERN_RESIZE,
  SLICE_PATTERN_SCATTER,
  SLICE_PATTERN_SEGMENT,
  PATTERN_RESERVED
};

enum OpImplType {
  EN_IMPL_CUSTOM_CONSTANT_CCE = 0,   // custom constant op
  EN_IMPL_CUSTOM_TIK,                // custom tik op
  EN_IMPL_CUSTOM_TBE,                // custom tbe op
  EN_IMPL_HW_CONSTANT_CCE,           // Huawei built-in constant op
  EN_IMPL_HW_GENERAL_CCE,            // Huawei built-in cce op
  EN_IMPL_HW_TIK,                    // Huawei built-in tik op
  EN_IMPL_HW_TBE,                    // Huawei built-in tbe op
  EN_IMPL_RL,                        // RL op
  EN_IMPL_PLUGIN_TBE,                // Huawei built-in tbe plugin op
  EN_IMPL_VECTOR_CORE_HW_TBE,        // Huawei built-in tbe op
  EN_IMPL_VECTOR_CORE_CUSTOM_TBE,    // custom tbe op
  EN_IMPL_NON_PERSISTENT_CUSTOM_TBE, // custom tbe op
  EN_RESERVED                        // reserved value
};

// Dont change the order, only add new mode in the end
enum L2Mode { EN_L2_CLOSE = 0, EN_L2_BUFFER_OPTIMIZE, EN_L2_CACHE_NORMAL, EN_L2_CACHE_RC };
enum BufferFusionMode { EN_OPTIMIZE_DISABLE = 0, EN_L2_BUFFER, EN_L2_FUSION};

static const std::map<ge::DataType, uint32_t> DATATYPE_SIZE_MAP{
    {ge::DT_FLOAT, sizeof(float)},
    {ge::DT_FLOAT16, sizeof(int16_t)},
    {ge::DT_INT8, sizeof(int8_t)},
    {ge::DT_INT32, sizeof(int32_t)},
    {ge::DT_UINT8, sizeof(uint8_t)},
    {ge::DT_UINT32, sizeof(uint32_t)},
    {ge::DT_INT16, sizeof(int16_t)},
    {ge::DT_UINT16, sizeof(uint16_t)},
    {ge::DT_INT64, sizeof(int64_t)},
    {ge::DT_UINT64, sizeof(uint64_t)},
    {ge::DT_DOUBLE, sizeof(double)},
    {ge::DT_BOOL, sizeof(bool)},
    {ge::DT_DUAL, sizeof(float) + sizeof(int8_t)},
    {ge::DT_DUAL_SUB_UINT8, sizeof(int8_t)},
    {ge::DT_DUAL_SUB_INT8, sizeof(int8_t)}
};

enum OpReduceType {
  REDUCE_MEAN = 0,
  REDUCE_ADD,
  REDUCE_MAX,
  REDUCE_MIN,
};

}
#endif
