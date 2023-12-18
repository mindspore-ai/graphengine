/*
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

#ifndef INC_FRAMEWORK_COMMON_TLV_NANO_DBG_DESC_H_
#define INC_FRAMEWORK_COMMON_TLV_NANO_DBG_DESC_H_

#include "framework/common/tlv/tlv.h"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma pack(1)  // single-byte alignment

/************************** definition of dbg data desc *************************************/
#define DBG_DATA_HEAD_MAGIC  0x5A5A5A5AU
struct DbgDataHead {
  uint32_t version_id;
  uint32_t magic;
  uint8_t tlv[0]; // TlvHead
};

/********************************************************************************************/
/************************** definition of dbg data Level 1 tlv ******************************/
#define DBG_L1_TLV_TYPE_MODEL_NAME        0U  // uint8_t[]
#define DBG_L1_TLV_TYPE_OP_DESC           1U

struct DbgOpDescParamTlv1 {
  uint32_t task_id;
  uint32_t stream_id;
  uint32_t logic_stream_id;
  int32_t task_type;
  uint32_t block_dim;
  uint8_t datadump_is_multiop;
  uint32_t l2_tlv_list_len;
  uint8_t l2_tlv[0]; // TlvHead
};

struct DbgOpDescTlv1 {
  uint32_t num;
  uint8_t param[0]; // DbgOpDescParamTlv1
};

/********************************************************************************************/
/************************** definition of dbg data desc Level 2 tlv *************************/
#define DBG_L2_TLV_TYPE_OP_NAME                  0U   // uint8_t[]
#define DBG_L2_TLV_TYPE_OP_TYPE                  1U   // uint8_t[]
#define DBG_L2_TLV_TYPE_ORI_OP_NAME              2U   // uint8_t[]
#define DBG_L2_TLV_TYPE_L1_SUB_GRAPH_NO          3U   // uint8_t[]
#define DBG_L2_TLV_TYPE_INPUT_DESC               4U
#define DBG_L2_TLV_TYPE_OUTPUT_DESC              5U
#define DBG_L2_TLV_TYPE_WORKSPACE_DESC           6U
#define DBG_L2_TLV_TYPE_OP_BUF                   7U
#define DBG_L2_TLV_TYPE_MEM_INFO                 8U
#define DBG_L2_TLV_TYPE_MAX                      9U

/************************** definition of dbg input tlv's value *****************************/
#define DBG_INPUT_DESC_L3_TLV_TYPE_SHAPE_DIMS          0U
#define DBG_INPUT_DESC_L3_TLV_TYPE_ORI_SHAPE_DIMS      1U
#define DBG_INPUT_DESC_L3_TLV_TYPE_MAX                 2U

struct DbgInputDescParamTlv2 {
  int32_t data_type;
  int32_t format;
  int32_t addr_type;
  uint64_t addr;
  uint64_t offset;
  uint64_t size;
  uint32_t l3_tlv_list_len;
  uint8_t  l3_tlv[0]; // TlvHead
};

struct DbgInputDescTlv2 {
  uint32_t num;
  uint8_t param[0]; // DbgInputDescParamTlv2
};

/********************************************************************************************/
/************************** definition of dbg output tlv's value ****************************/
#define DBG_OUTPUT_DESC_L3_TLV_TYPE_SHAPE_DIMS          0U
#define DBG_OUTPUT_DESC_L3_TLV_TYPE_ORI_SHAPE_DIMS      1U
#define DBG_OUTPUT_DESC_L3_TLV_TYPE_ORI_NAME            2U
#define DBG_OUTPUT_DESC_L3_TLV_TYPE_MAX                 3U

struct DbgOutputDescParamTlv2 {
  int32_t data_type;
  int32_t format;
  int32_t addr_type;
  int32_t original_index;
  int32_t original_data_type;
  int32_t original_format;
  uint64_t addr;
  uint64_t offset;
  uint64_t size;
  uint32_t l3_tlv_list_len;
  uint8_t  l3_tlv[0]; // TlvHead
};

struct DbgOutputDescTlv2 {
  uint32_t num;
  uint8_t param[0]; // DbgOutputDescParamTlv2
};

/************************** definition of dbg workspace tlv's value *************************/
struct DbgWorkspaceDescParamTlv2 {
  int32_t type;
  uint64_t data_addr;
  uint64_t size;
  uint32_t l3_tlv_list_len;
  uint8_t  l3_tlv[0]; // TlvHead
};

struct DbgWorkspaceDescTlv2 {
  uint32_t num;
  uint8_t param[0]; // DbgWorkspaceDescParamTlv2
};

/********************************************************************************************/
/************************** definition of dbg op buf tlv's value ****************************/
// L2 tlv op buffer list
struct DbgOpBufParamTlv2 {
  uint8_t type;
  uint64_t addr;
  uint64_t size;
  uint32_t l3_tlv_list_len;
  uint8_t  l3_tlv[0]; // TlvHead
};

struct DbgOpBufTlv2 {
  uint32_t num;
  uint8_t param[0]; // DbgOpBufParamTlv2
};

/************************** definition of dbg mem info tlv's value **************************/
// L2 tlv mem info list
struct DbgOpMemInfoTlv2 {
  uint64_t input_mem_size;
  uint64_t output_mem_size;
  uint64_t weight_mem_size;
  uint64_t workspace_mem_size;
  uint64_t total_mem_size;
  uint32_t l3_tlv_list_len;
  uint8_t l3_tlv[0]; // TlvHead
};

struct DbgOpMemInfosTlv2 {
  uint32_t num;
  uint8_t param[0]; // DbgOpMemInfoTlv2
};

/********************************************************************************************/
#pragma pack() // Cancels single-byte alignment

#if defined(__cplusplus)
}
#endif

#endif  // INC_FRAMEWORK_COMMON_TLV_NANO_DBG_DESC_H_