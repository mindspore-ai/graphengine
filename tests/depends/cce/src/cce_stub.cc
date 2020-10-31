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

#include <vector>
#include <cce/cce.h>
#include <cce/dnn.h>
#include <cce/compiler_stub.h>
#include <cce/taskdown_api.h>

#include "cce/optimizer/fusion_engine.h"
#include "common/op/attr_value_util.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"

using namespace cce;
using namespace std;
using namespace ge;
using namespace fusion;

uint64_t global_mem_base = 0;

namespace cce {
#define DIM_MAX_SIZE 8
static const uint32_t C0 = 16;
struct tagCcPad {};
struct tagCcConvolution {};

struct tagCcLRN {};

struct tagCcFasterRcnnProposal {};
struct tagCcRoiAlign {};
struct tagCcBatchNorm {};
struct tagCcDetectpostprocess {};

struct tagCcSsdDetectionOutput {};

struct tagCcRefinedetDetectionOutput {};

struct tagCcMsrGenerateRpnProposals {};

struct tagCcFilter {
  vector<uint32_t> dims;
};

struct tagCcTensor {
  ccTensorFormat_t format;
  ccDataType_t data_type;
  uint32_t dim_cnt;
  int32_t real_dim_cnt;
  uint32_t data_size;
  int32_t dim_buf[DIM_MAX_SIZE];
  int32_t stride_buf[DIM_MAX_SIZE];
};

typedef struct tagCcPooling {
  ccPoolingMode_t mode;
  ccPaddingMode_t pad_mode;
  ccNanPropagation_t max_pooling_nan_opt;
  uint32_t dim_cnt;
  int32_t window_dim[6];
  int32_t padding[6];
  int32_t stride[6];
} ccPooling_t;

struct tagCcActivation {};

struct tagCcFasterRcnnDetectionOutput {};
struct tagCcSpatialTransformer {};

struct tagCcPower {};
struct tagCcResizeBilinear {};
struct tagCcSsdNormalize {};
struct tagCcSsdPostProcessor {};
struct tagCcSsdPriorBox {};
struct tagCcPsRoiPooling {};

struct tagMsrFastRcnnPredictions {};
struct tagCcPRelu {};
struct tagCcStridedSlice {};

struct tagCcStridedSliceAttrs {};

struct tagCcRnn {};

struct tagCcArgmaxmin {};

typedef struct tagCcLog {
  ccDataType_t data_type;
  uint32_t param_cnt;
} ccLog_t;
typedef struct tagCcLog *ccLogDescriptor_t;

struct tagCcPadV2 {};

ccStatus_t ccGetPadV2OutputDim(const ccTensorDescriptor_t x_desc, const ccPadV2Descriptor_t pad_desc, int32_t *dim_cnt,
                               int32_t dim[], int32_t dim_len) {
  *dim_cnt = 4;
  dim[0] = 1;
  dim[1] = 2;
  dim[2] = 2;
  dim[3] = 3;
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccPadV2Forward(ccHandle_t handle, const ccPadV2Descriptor_t pad_desc, const void *alpha,
                          const ccTensorDescriptor_t x_desc, const void *x, const void *beta,
                          const ccTensorDescriptor_t output_desc, void *output) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccCreatePadV2Descriptor(ccPadV2Descriptor_t *pad_desc) { return CC_STATUS_SUCCESS; }

ccStatus_t ccDestroyPadV2Descriptor(ccPadV2Descriptor_t *pad_desc) { return CC_STATUS_SUCCESS; }

ccStatus_t ccSetKernelOpMap(ccHandle_t handle) { return CC_STATUS_SUCCESS; }

ccStatus_t ccDataDumpForward(ccHandle_t handle, const void *buffer, const uint64_t buf_len, const uint32_t task_index) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetPadV2Descriptor(ccPadV2Descriptor_t pad_desc, const int32_t pad_shape_cnt,
                                const int32_t pad_shape_low[], const int32_t pad_shape_high[],
                                const ccPadMode_t pad_mode, const void *pad_value, const ccDataType_t pad_value_type) {
  return CC_STATUS_SUCCESS;
}

struct tagCcYoloDetectionOutput {
  ccYoloVersion_t yolo_version;
  uint32_t net_h;
  uint32_t net_w;
  uint32_t post_top_k;
  uint32_t classes;
  float nms_threshold;
  float iou_thre_decay;
  float coor_scale_factor;
  bool relative;
  float obj_threshold;
  float cls_threshold;
  uint32_t bias_num;
  float *bias;
};

struct tagCcYoloRegion {};

struct tagCcEltwise {};

struct tagCcHashTableLookup {};

struct tagCcEmbeddingAttnDecoder {};
struct tagNonMaxSuppression {};

struct tagCcArcSinCos {};
struct tagCcPow {};
struct tagCcConcatFive2Four_t {};
struct tagCcConcatFour2Five_t {};

ccStatus_t ccCreatePowDescriptor(ccPowDescriptor_t *pow_desc) {
  *pow_desc = new tagCcPow();
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetPowDescriptor(ccPowDescriptor_t pow_desc, ccDataType_t data_type, uint32_t param_cnt) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccDestroyPowDescriptor(ccPowDescriptor_t *pow_desc) {
  if (nullptr == pow_desc) {
    return CC_STATUS_BAD_PARAM;
  }

  delete *pow_desc;
  *pow_desc = 0;
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccPowForward(ccHandle_t handle, const ccPowDescriptor_t pow_desc, const void *pow_param, const void *alpha,
                        const ccTensorDescriptor_t x_desc, const void *x, const ccTensorDescriptor_t y_desc,
                        const void *y, const void *beta, const ccTensorDescriptor_t z_desc, void *z) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccLogicalOrForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t x_desc, const void *x,
                              const ccTensorDescriptor_t y_desc, const void *y, const void *beta,
                              const ccTensorDescriptor_t output_desc, void *output) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccCompareForward(ccHandle_t handle, ccCompareType_t compare_type, const void *alpha,
                            const ccTensorDescriptor_t x_desc, const void *x, const ccTensorDescriptor_t y_desc,
                            const void *y, const void *beta, const ccTensorDescriptor_t output_desc, void *output) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccGetCompareOutputDim(const ccTensorDescriptor_t x_desc, const ccTensorDescriptor_t y_desc, int32_t *dim_cnt,
                                 int32_t *dim, int32_t dim_len) {
  *dim_cnt = 4;
  dim[0] = 1;
  dim[1] = 1;
  dim[2] = 1;
  dim[3] = 1;
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccArcTanForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t x_desc, const void *x,
                           const void *beta, const ccTensorDescriptor_t y_desc, void *y) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccAtanhForward(ccHandle_t handle, const void *alpha, const ccTensorDescriptor_t x_desc, const void *x,
                          const void *beta, const ccTensorDescriptor_t y_desc, void *y) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccIsDepthwiseHighPerformance(int32_t input_n, int32_t input_c, int32_t input_h, int32_t input_w,
                                        int32_t filter_n, int32_t filter_c, int32_t filter_h, int32_t filter_w,
                                        int32_t dilation_h, int32_t dilation_w, int32_t pad_h_head, int32_t pad_h_tail,
                                        int32_t pad_w_head, int32_t pad_w_tail, int32_t stride_h, int32_t stride_w,
                                        int32_t group_num, bool &is_high_performance, bool is_quant,
                                        ccDataType_t input_data_type, ccDataType_t output_data_type) {
  is_high_performance = true;
  return CC_STATUS_SUCCESS;
}

struct tagCcSpaceToBatch {};

struct tagCcBatchToSpace {};

struct tagCcResizeNearestNeighbor {};

ccStatus_t ccGetStream(ccHandle_t handle, rtStream_t *stream_id) { return CC_STATUS_SUCCESS; }

ccStatus_t ccGetRtVersion(uint32_t *count) { return CC_STATUS_SUCCESS; }

ccStatus_t ccDestroyTensorDescriptor(ccTensorDescriptor_t *tensor_desc) {
  if (nullptr == tensor_desc) {
    return CC_STATUS_BAD_PARAM;
  }
  delete *tensor_desc;
  *tensor_desc = 0;
  return CC_STATUS_SUCCESS;
}
ccStatus_t ccDestroyFilterDescriptor(ccFilterDescriptor_t *filter_desc) {
  delete *filter_desc;
  *filter_desc = 0;
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccGetFilterSizeInBytes(const ccFilterDescriptor_t filter_desc, uint32_t *size) {
  *size = filter_desc->dims[0] * filter_desc->dims[1] * filter_desc->dims[2] * filter_desc->dims[3] * sizeof(float);
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccTransFilter(const ccFilterDescriptor_t w_desc, const void *w, ccFilterDescriptor_t y_desc, void *y,
                         uint32_t y_size_in_bytes) {
  y = const_cast<void *>(w);

  return CC_STATUS_SUCCESS;
}

ccStatus_t ccCreateTensorDescriptor(ccTensorDescriptor_t *tensor_desc) {
  *tensor_desc = new tagCcTensor();
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetTensor4dDescriptor(ccTensorDescriptor_t tensor_desc, ccTensorFormat_t format, ccDataType_t data_type,
                                   int32_t n, int32_t c, int32_t h, int32_t w) {
  if (CC_TENSOR_NHWC == format) {
    tensor_desc->dim_buf[0] = n;
    tensor_desc->dim_buf[1] = h;
    tensor_desc->dim_buf[2] = w;
    tensor_desc->dim_buf[3] = c;
  } else {
    tensor_desc->dim_buf[0] = n;
    tensor_desc->dim_buf[1] = c;
    tensor_desc->dim_buf[2] = h;
    tensor_desc->dim_buf[3] = w;
  }
  tensor_desc->dim_cnt = 4;
  tensor_desc->data_type = data_type;
  tensor_desc->format = format;
  tensor_desc->data_size = n * c * h * w * sizeof(data_type);
  return CC_STATUS_SUCCESS;
}
ccStatus_t ccGetTensorSizeInBytes(const ccTensorDescriptor_t tensor_desc, uint32_t *size) {
  if ((NULL == tensor_desc) || (NULL == size)) {
    return CC_STATUS_BAD_PARAM;
  }
  *size = tensor_desc->data_size;
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccGetTensorMemorySizeInBytes(const ccTensorDescriptor_t tensor_desc, uint32_t *size) {
  *size = tensor_desc->data_size;
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccCreateFilterDescriptor(ccFilterDescriptor_t *filter_desc) {
  *filter_desc = new tagCcFilter();
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetFilter4dDescriptor(ccFilterDescriptor_t filter_desc, ccTensorFormat_t format, ccDataType_t data_type,
                                   int32_t k, int32_t c, int32_t h, int32_t w) {
  filter_desc->dims.push_back(k);
  filter_desc->dims.push_back(c);
  filter_desc->dims.push_back(h);
  filter_desc->dims.push_back(w);

  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetFilterFractalDescriptor(ccFilterDescriptor_t filter_desc, ccTensorFormat_t format,
                                        ccDataType_t data_type, int32_t k, int32_t c, int32_t h, int32_t w) {
  filter_desc->dims.push_back(k);
  filter_desc->dims.push_back(c);
  filter_desc->dims.push_back(h);
  filter_desc->dims.push_back(w);

  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetStream(ccHandle_t handle, rtStream_t stream_id) { return CC_STATUS_SUCCESS; }
ccStatus_t ccCreatePoolingMaskDescriptor(ccTensorDescriptor_t *pooling_mask_desc) {
  *pooling_mask_desc = new tagCcTensor();
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetPoolingMaskTensorDescriptor(ccTensorDescriptor_t tensor_desc, ccTensorFormat_t format,
                                            ccDataType_t data_type, int32_t n, int32_t c, int32_t h, int32_t w,
                                            int32_t window_h, int32_t window_w) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetFilter6dDescriptor(ccTensorDescriptor_t filter_desc, ccTensorFormat_t format, ccDataType_t data_type,
                                   int32_t c1, int32_t h, int32_t w, int32_t n, int32_t co, int32_t c0) {
  return CC_STATUS_SUCCESS;
}

/// @ingroup dnn
/// @brief get the format and dimcnt of GeTensor
/// @param [in] tensor_desc   descriptor of tensor
/// @param [in|out] format   point to format
/// @return ccStatus_t
ccStatus_t ccGetTensorFormat(const ccTensorDescriptor_t tensor_desc, ccTensorFormat_t *format) {
  *format = tensor_desc->format;
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccTransTensor(const ccTensorDescriptor_t x_desc, const void *x, const ccTensorDescriptor_t y_desc, void *y,
                         uint32_t y_size_in_bytes) {
  return CC_STATUS_SUCCESS;
}
void cceSysInit() {}

bool compilerStubFree() { return true; }

bool compilerStubInit() { return true; }

ccStatus_t ccSetInt8Filter4dDescriptor(ccFilterDescriptor_t filter_desc, ccTensorFormat_t format,
                                       ccDataType_t data_type, int32_t k, int32_t c, int32_t h, int32_t w,
                                       ccDataType_t output_data_type) {
  filter_desc->dims.push_back(k);
  filter_desc->dims.push_back(c);
  filter_desc->dims.push_back(h);
  filter_desc->dims.push_back(w);

  return CC_STATUS_SUCCESS;
}
ccStatus_t ccSetTensorNdDescriptor(ccTensorDescriptor_t tensor_desc, ccDataType_t data_type, int32_t dim_cnt,
                                   int32_t dimA[]) {
  tensor_desc->data_type = data_type;
  tensor_desc->data_size = sizeof(data_type);
  for (int32_t i = 0; i < dim_cnt; i++) {
    tensor_desc->data_size = tensor_desc->data_size * dimA[i];
  }
  tensor_desc->format = CC_TENSOR_ND;
  return CC_STATUS_SUCCESS;
}

ccStatus_t CceProfilingConfig(const char *target, const char *job_ctx, uint32_t flag) { return CC_STATUS_SUCCESS; }
ccStatus_t ccSetTensorRealDimCnt(ccTensorDescriptor_t tensor_desc, int32_t real_dim_cnt) {
  if (tensor_desc != NULL && tensor_desc != nullptr) {
    tensor_desc->real_dim_cnt = real_dim_cnt;
  }
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccGetTensorRealDimCnt(ccTensorDescriptor_t tensor_desc, int32_t *real_dim_cnt) {
  *real_dim_cnt = tensor_desc->real_dim_cnt;
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetQuantizeFactors(ccQuantizeDescriptor_t quantize_info, ccScaleValueMode_t scale_val_mode,
                                const uint16_t *scale, const uint16_t *offset, const uint8_t *offset_pad) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetReQuantizeFactors(ccQuantizeDescriptor_t quantize_info, ccScaleValueMode_t scale_val_mode,
                                  const uint16_t *scale_rq, const uint16_t *next_layer_offset,
                                  const int32_t *offset_w) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetDeQuantizeFactors(ccQuantizeDescriptor_t quantize_info, ccScaleValueMode_t scale_val_mode,
                                  const uint16_t *scale_dq, const int32_t *offset_w) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetQuantizeAlgoAndScaleType(ccQuantizeDescriptor_t quantize_info, ccQuantizeAlgo_t quant_algo,
                                         ccScaleType_t scale_type, bool relu_flag) {
  return CC_STATUS_SUCCESS;
}
ccStatus_t ccPrintTimeStat() { return CC_STATUS_SUCCESS; }
ccStatus_t ccSetModelId(ccHandle_t handle, uint32_t model_id) { return CC_STATUS_SUCCESS; }

ccStatus_t ccGetKernelContext(rtStream_t stream_id, ccOpContext &op_context) {
  if (stream_id == nullptr) {
    op_context.kernelType = ccKernelType::TE;
  } else {
    op_context.kernelType = ccKernelType::CCE_AI_CORE;
    op_context.opId = 1;
    op_context.kernelFuncId = 1;
    op_context.isFlowtable = true;
    op_context.opCount = 1;
    op_context.opIndex2[0] = 0;
  }

  return CC_STATUS_SUCCESS;
}

ccStatus_t ccUpdateKernelArgs(ccOpContext &op_context, uint64_t data_base_addr, uint64_t weight_base_addr,
                              uint64_t variable_base_addr, void *args_addr, uint64_t args_size, void *l2ctrl_addr) {
  return CC_STATUS_SUCCESS;
}
ccStatus_t ccGetKernelArgsAddrs(ccOpContext &op_context, void *args_addr, uint64_t args_size, void *l2ctrl_addr,
                                std::vector<ccOpAddrsInfo> &op_addrs_info) {
  // cce
  ccOpAddrsInfo tmp_op_addrs_info;
  uint64_t tmp_input = (uint64_t)global_mem_base;
  tmp_op_addrs_info.addrPos = &tmp_input;
  tmp_op_addrs_info.addrData = tmp_input;
  op_addrs_info.push_back(tmp_op_addrs_info);

  uint64_t tmp_output = (uint64_t)(global_mem_base + 5476352);
  tmp_op_addrs_info.addrPos = &tmp_output;
  tmp_op_addrs_info.addrData = tmp_output;
  op_addrs_info.push_back(tmp_op_addrs_info);
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccSetKernelArgs(std::vector<ccOpAddrsInfo> &date_info) { return CC_STATUS_SUCCESS; }
}  // namespace cce
// ccFusion no namespace
ccStatus_t ccFusionStart(ccHandle_t handle, uint32_t graph_id, uint32_t init_flag, CceFusionMemCfg_t mem_cfg) {
  return CC_STATUS_SUCCESS;
}

//???ccFusion ????namespace cce??
ccStatus_t ccFusionStart(ccHandle_t handle, uint32_t graph_id, uint32_t init_flag, uint32_t addr_change_flag) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t ccFusionEnd(ccHandle_t handle, uint32_t graph_id) { return CC_STATUS_SUCCESS; }

ccStatus_t ccFusionTaskEnd(ccHandle_t handle, uint32_t graph_id) { return CC_STATUS_SUCCESS; }

ccStatus_t ccKernelLaunchRepeat(ccHandle_t handle) { return CC_STATUS_SUCCESS; }

ccStatus_t ccKernelDelete(ccHandle_t handle) { return CC_STATUS_SUCCESS; }

ccStatus_t cce::ccSetTensorFormat(cce::tagCcTensor *, cce::tagCcTensorFormat) { return CC_STATUS_SUCCESS; }

namespace fusion {
uint32_t BufferFusion(std::shared_ptr<ge::ComputeGraph>, std::shared_ptr<ge::ComputeGraph>, bool) { return 0; }

uint32_t BufferFusionTrain(std::shared_ptr<ge::ComputeGraph>, std::shared_ptr<ge::ComputeGraph>) { return 0; }

uint32_t GraphFusionTrain(ge::ComputeGraphPtr orig_graph, ge::ComputeGraphPtr fusion_graph) { return 0; }
}  // namespace fusion
namespace fusion {
using namespace ge;

uint32_t Fusion(ComputeGraphPtr model_graph, ComputeGraphPtr fusion_graph, kScopeNodeMap_t &te_fusion_map) {
  OpDescPtr op_def_a = std::make_shared<OpDesc>();
  op_def_a->SetName("reduction_nd");
  op_def_a->SetType("reduction_nd");

  GeTensorDescPtr v_input_desc = std::make_shared<GeTensorDesc>();
  op_def_a->AddInputDesc(*v_input_desc);

  vector<int64_t> v_input;
  v_input.push_back(0);
  op_def_a->SetInputOffset(v_input);

  GeTensorDesc input_desc = op_def_a->GetInputDesc(0);
  input_desc.SetFormat(FORMAT_NCHW);
  input_desc.SetDataType(DT_FLOAT);
  input_desc.SetShape(GeShape({1, 3, 5, 5}));
  ge::TensorUtils::SetSize(input_desc, 192);
  ge::TensorUtils::SetRealDimCnt(input_desc, 4);

  GeTensorDescPtr output_desc = std::make_shared<GeTensorDesc>();
  op_def_a->AddOutputDesc(*output_desc);

  output_desc->SetFormat(FORMAT_NCHW);
  output_desc->SetDataType(DT_FLOAT);
  output_desc->SetShape(GeShape({1, 3, 5}));
  ge::TensorUtils::SetSize(*output_desc, 96);
  ge::TensorUtils::SetRealDimCnt(*output_desc, 3);

  OpDescPtr op_def_b = std::make_shared<OpDesc>();
  op_def_b->SetName("transdata_1");
  op_def_b->SetType("TransData");

  int stream_num = 1;
  int flag = 0;

  // make_graph_nd(graph);
  NodePtr node_a = fusion_graph->AddNode(op_def_a);
  NodePtr node_b = fusion_graph->AddNode(op_def_b);

  GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
  int32_t a = 1;
  int32_t b = 2;

  AttrUtils::SetInt(op_def_a, "fusion_scope", a);
  AttrUtils::SetInt(op_def_b, "fusion_scope", b);

  vector<NodePtr> node_list1;
  node_list1.push_back(node_a);
  vector<NodePtr> node_list2;
  node_list2.push_back(node_b);
  te_fusion_map[1] = node_list1;
  te_fusion_map[2] = node_list2;

  return FUSION_STATUS_SUCCESS;
}

uint32_t FusionTaskBuild(cce::ccHandle_t cc_handle, ge::ComputeGraphPtr fusion_graph, ge::Buffer &buffer,
                         ModelRes &model_res, std::vector<TaskDef> &task_def_list_) {
  TaskDef task_def_temp;
  task_def_list_.push_back(task_def_temp);

  return FUSION_STATUS_SUCCESS;
}
uint32_t GraphFusion(ge::ComputeGraphPtr orig_graph, ge::ComputeGraphPtr fusion_graph) {
  *fusion_graph = *orig_graph;
  return FUSION_STATUS_SUCCESS;
}

void FusionTaskBuildComplete(std::vector<ccHandle_t> cc_handle_list) { return; }

}  // namespace fusion

ccStatus_t cce::ccSetTensorDescriptorQuantizeParam(ccTensorDescriptor_t tensor_desc,
                                                   const ccVecQuantizePara_t *vec_quantize_para) {
  return CC_STATUS_SUCCESS;
}

ccStatus_t cce::ccSetAllOffsetQuantizeFactors(ccQuantizeDescriptor_t quantize_info, const uint8_t *offset_w,
                                              const uint8_t *offset_d, const uint16_t *scale_req,
                                              const uint16_t *offset_d_next) {
  return CC_STATUS_SUCCESS;
}
