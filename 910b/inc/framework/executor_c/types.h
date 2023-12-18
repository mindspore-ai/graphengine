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

#ifndef INC_FRAMEWORK_EXECUTOR_C_TYPES_H_
#define INC_FRAMEWORK_EXECUTOR_C_TYPES_H_
#include <stdint.h>
#include <stddef.h>
#include "vector.h"
#if defined(__cplusplus)
extern "C" {
#endif
#define MODEL_FILE_CHECKSUM_LENGTH 64
#define MODEL_NAME_LENGTH          32
#define USER_DEFINE_INFO_LENGTH    32
#define PLATFORM_VERSION_LEN       20
#define MODEL_FILE_RESERVED_LENGTH 75
#define MODEL_FILE_MAGIC_NUM       0x444F4D49

enum ModelEncryptType {
  UNENCRYPTED,
  ENCRYPTED
};

enum ModelCheckType {
  CHECK,
  UNCHECK
};

typedef enum {
  MODEL_DEF = 0,
  WEIGHTS_DATA = 1,
  TASK_INFO = 2,
  TBE_KERNELS = 3,
  CUST_AICPU_KERNELS = 4,
  SO_BINS = 5,
  FLOW_MODEL = 6,
  FLOW_SUBMODEL = 7,
  MODEL_INOUT_INFO = 8,
  STATIC_TASK_DESC = 9,
  DYNAMIC_TASK_DESC = 10,
  TASK_PARAM = 11,
  PRE_MODEL_DESC = 20,
  PRE_MODEL_SQE = 21,
  PRE_KERNEL_ARGS = 22
} ModelPartitionType;


typedef struct {
  uint32_t magic;                               // magic number of DOMI
  uint32_t headsize ;                           // length of the model header. The value is fixed at 256
  uint32_t version;                             // version 1.0
  uint8_t checksum[MODEL_FILE_CHECKSUM_LENGTH]; // signature
  uint32_t length;  // Ciphertext length. In the non-encryption model, the length is the plaintext length.
  // whether encrypted 0:not encrypt, 1:encrypt
  uint8_t is_encrypt;
  uint8_t is_checksum;                          // whether to check the checksum
  uint8_t modeltype;                            // 0:IR model 1:standard model 2:OM Tiny model 3:flow model
  uint8_t genmode;                              // 0：offline generate 1：online generate
  uint8_t name[MODEL_NAME_LENGTH];              // Model name, which contains 32 characters
  uint32_t ops;                                 // Computing power (Kops)
  uint8_t userdefineinfo[USER_DEFINE_INFO_LENGTH];  // User-defined information. The value contains 32 characters
  uint32_t om_ir_version;
  uint32_t model_num;
  uint8_t platform_version[PLATFORM_VERSION_LEN];
  uint8_t platform_type;
  uint8_t reserved[MODEL_FILE_RESERVED_LENGTH];  // Reserved field 64
} ModelFileHeader;

typedef struct {
  ModelPartitionType type;
  uint64_t mem_offset;
  uint64_t mem_size;
} ModelPartitionMemInfo;

typedef struct {
  uint32_t num;
  ModelPartitionMemInfo partition[0];
} ModelPartitionTable;

enum WeightType {PREFETCH_EVERTIME = 0, PREFETCH_ALL};
enum TagType {INPUT_DESC_TAG = 0, OUTPUT_DESC_TAG, NAME_TAG = 10, DIMES_TAG = 11};

typedef struct {
  uint8_t tag;
  uint32_t len;
  uint8_t name[0];
} IOParamName;

typedef struct {
  uint8_t tag;
  uint32_t len;
  uint8_t dim[0];
} IOParamDims;

typedef enum {
  FORMAT_NCHW = 0,   // NCHW
  FORMAT_NHWC,       // NHWC
  FORMAT_ND,         // Nd Tensor
  FORMAT_NC1HWC0,    // NC1HWC0
  FORMAT_FRACTAL_Z,  // FRACTAL_Z
  FORMAT_NC1C0HWPAD = 5,
  FORMAT_NHWC1C0,
  FORMAT_FSR_NCHW,
  FORMAT_FRACTAL_DECONV,
  FORMAT_C1HWNC0,
  FORMAT_FRACTAL_DECONV_TRANSPOSE = 10,
  FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS,
  FORMAT_NC1HWC0_C04,    // NC1HWC0, C0 is 4
  FORMAT_FRACTAL_Z_C04,  // FRACZ, C0 is 4
  FORMAT_CHWN,
  FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS = 15,
  FORMAT_HWCN,
  FORMAT_NC1KHKWHWC0,  // KH,KW kernel h& kernel w maxpooling max output format
  FORMAT_BN_WEIGHT,
  FORMAT_FILTER_HWCK,  // filter input tensor format
  FORMAT_HASHTABLE_LOOKUP_LOOKUPS = 20,
  FORMAT_HASHTABLE_LOOKUP_KEYS,
  FORMAT_HASHTABLE_LOOKUP_VALUE,
  FORMAT_HASHTABLE_LOOKUP_OUTPUT,
  FORMAT_HASHTABLE_LOOKUP_HITS,
  FORMAT_C1HWNCoC0 = 25,
  FORMAT_MD,
  FORMAT_NDHWC,
  FORMAT_FRACTAL_ZZ,
  FORMAT_FRACTAL_NZ,
  FORMAT_NCDHW = 30,
  FORMAT_DHWCN,  // 3D filter input tensor format
  FORMAT_NDC1HWC0,
  FORMAT_FRACTAL_Z_3D,
  FORMAT_CN,
  FORMAT_NC = 35,
  FORMAT_DHWNC,
  FORMAT_FRACTAL_Z_3D_TRANSPOSE, // 3D filter(transpose) input tensor format
  FORMAT_FRACTAL_ZN_LSTM,
  FORMAT_FRACTAL_Z_G,
  FORMAT_RESERVED = 40,
  FORMAT_ALL,
  FORMAT_NULL,
  FORMAT_ND_RNN_BIAS,
  FORMAT_FRACTAL_ZN_RNN,
  FORMAT_NYUV = 45,
  FORMAT_NYUV_A,
  FORMAT_NCL,
  // Add new formats definition here
  FORMAT_END,
  // FORMAT_MAX defines the max value of Format.
  // Any Format should not exceed the value of FORMAT_MAX.
  // ** Attention ** : FORMAT_MAX stands for the SPEC of enum Format and almost SHOULD NOT be used in code.
  //                   If you want to judge the range of Format, you can use FORMAT_END.
  FORMAT_MAX = 0xff
} Format;

typedef enum {
  DT_FLOAT = 0,            // float type
  DT_FLOAT16 = 1,          // fp16 type
  DT_INT8 = 2,             // int8 type
  DT_INT16 = 6,            // int16 type
  DT_UINT16 = 7,           // uint16 type
  DT_UINT8 = 4,            // uint8 type
  DT_INT32 = 3,            //
  DT_INT64 = 9,            // int64 type
  DT_UINT32 = 8,           // unsigned int32
  DT_UINT64 = 10,          // unsigned int64
  DT_BOOL = 12,            // bool type
  DT_DOUBLE = 11,          // double type
  DT_STRING = 13,          // string type
  DT_DUAL_SUB_INT8 = 14,   // dual output int8 type
  DT_DUAL_SUB_UINT8 = 15,  // dual output uint8 type
  DT_COMPLEX64 = 16,       // complex64 type
  DT_COMPLEX128 = 17,      // complex128 type
  DT_QINT8 = 18,           // qint8 type
  DT_QINT16 = 19,          // qint16 type
  DT_QINT32 = 20,          // qint32 type
  DT_QUINT8 = 21,          // quint8 type
  DT_QUINT16 = 22,         // quint16 type
  DT_RESOURCE = 23,        // resource type
  DT_STRING_REF = 24,      // string ref type
  DT_DUAL = 25,            // dual output type
  DT_VARIANT = 26,         // dt_variant type
  DT_BF16 = 27,            // bf16 type
  DT_UNDEFINED = 28,       // Used to indicate a DataType field has not been set.
  DT_INT4 = 29,            // int4 type
  DT_UINT1 = 30,           // uint1 type
  DT_INT2 = 31,            // int2 type
  DT_UINT2 = 32,           // uint2 type
  DT_MAX                   // Mark the boundaries of data types
} DataType;
#pragma pack(push)
#pragma pack(1)
typedef struct {
  uint32_t task_num;
  uint64_t workspace_size;
  uint64_t weight_size;
  enum WeightType weight_type;
  uint8_t profile_enable;
  uint8_t model_interrupt;
} ModelDesc;
#pragma pack(pop)

typedef struct {
  uint32_t type : 2;
  uint32_t pre_p : 2;
  uint32_t post_p : 2;
  uint32_t cond_s : 2;
  uint32_t res : 2;
  uint32_t prefetch_num : 2;
  uint32_t block_dim : 2;
  uint32_t code_size : 2;
  uint32_t soft_user : 2;
  uint32_t task_pc_offset : 2;
  uint32_t task_param_offset : 2;
} TaskDesc;

typedef struct {
  char *name;
  size_t size;
  Format format;
  DataType dataType;
  Vector dims;
} ModelInOutTensorDesc;

typedef struct {
  Vector input_desc;
  Vector output_desc;
} ModelInOutInfo;

enum ModelDescType {
  MODEL_INPUT_DESC,
  MODEL_OUTPUT_DESC
};

typedef struct {
  int32_t type;
  uint32_t length;
  uint8_t *value;
} ModelDescTlvConfig;

typedef struct {
  uint64_t size;
  Format format;
  DataType dt;
  uint32_t name_len;
  uint32_t dims_len;
  uint32_t dimsV2_len;
  uint32_t shape_range_len;
} ModelTensorDescBaseInfo;

typedef struct {
  ModelPartitionType type;
  uint8_t *data;
  uint32_t size;
} ModelPartition;

#if defined(__cplusplus)
}
#endif

#endif  // INC_FRAMEWORK_EXECUTOR_C_TYPES_H_